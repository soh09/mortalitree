"""Stage B: RGB tree-dataset pretraining. Encoder frozen; only neck + head trained."""
import copy
import time
from pathlib import Path
from typing import Optional

import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..data.deepforest_dataset import DeepForestDataset, deepforest_collate_fn
from ..model.detector import TreeDetector
from ..train.losses import batch_matching_loss
from ..train.schedulers import build_stage_b_optimizer_and_scheduler


def _wandb_log(metrics: dict) -> None:
    """Log one history row to W&B if (and only if) a run is active. No explicit
    step — wandb auto-increments, one row per call, so call it once per epoch.
    Keeps src decoupled from the Modal/wandb setup."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass


def _wandb_images(key: str, images: list) -> dict:
    """Return {key: [wandb.Image, ...]} to merge into an epoch's log dict, so the
    images land in the same wandb.log call (same step) as that epoch's scalars.
    Empty dict if there's no active run or no images."""
    if not images:
        return {}
    try:
        import wandb
        if wandb.run is not None:
            return {key: [wandb.Image(im) for im in images]}
    except ImportError:
        pass
    return {}


def _split_train_val(dataset: Dataset, val_fraction: float, seed: int = 42):
    """Split into train/val. If the dataset exposes `group_keys` (e.g. a quartered
    NeonPatchDataset, where one parent tile yields several samples), split at the
    *group* level so all of a parent's quarters land in the same split — otherwise
    quarters of one tile leak across train/val and inflate val performance. Falls
    back to a plain per-sample random split when no group keys are available."""
    keys = getattr(dataset, "group_keys", None)
    if keys is None:
        n_val = max(1, int(len(dataset) * val_fraction))
        return random_split(dataset, [len(dataset) - n_val, n_val])

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    parents = list(groups)
    random.Random(seed).shuffle(parents)
    n_val_p = max(1, int(len(parents) * val_fraction))
    val_parents = set(parents[:n_val_p])
    train_idx, val_idx = [], []
    for k, idxs in groups.items():
        (val_idx if k in val_parents else train_idx).extend(idxs)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def run_stage_b(
    model: TreeDetector,
    annotations_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    total_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    val_fraction: float = 0.1,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
    device: Optional[str] = None,
    num_workers: int = 4,
    dataset: Optional[Dataset] = None,
    collate_fn=None,
    viz_every: int = 5,
    n_viz_tiles: int = 4,
    viz_conf_thresh: float = 0.5,
    on_checkpoint=None,
    parent_eval: bool = True,
    parent_eval_every: int = 1,
    parent_eval_train_parents: int = 1000,
    parent_eval_conf: float = 0.5,
    parent_eval_nms_iou: float = 0.6,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Default to the DeepForest RGB JSON; callers may pass a pre-built dataset
    # (e.g. NeonPatchDataset for 4-band NEON patches) and matching collate_fn.
    if dataset is None:
        if annotations_path is None:
            raise ValueError("Provide either `dataset` or `annotations_path`.")
        dataset = DeepForestDataset(annotations_path, augment=True)
    if collate_fn is None:
        collate_fn = deepforest_collate_fn

    train_ds, val_ds = _split_train_val(dataset, val_fraction, seed=42)

    # Augment train only: random_split gives two Subsets over the SAME dataset,
    # so repoint the val subset at a shallow copy with augmentation off. The copy
    # shares the parsed items/data (no re-read) — only `augment` differs. This
    # keeps the val loss a clean, deterministic signal.
    eval_dataset = dataset
    if getattr(dataset, "augment", None) is not None:
        val_clean = copy.copy(dataset)
        val_clean.augment = None
        val_ds.dataset = val_clean
        eval_dataset = val_clean   # augment-off, shares .samples/.parent_gt

    # Parent-level (stitched 256) eval each epoch: needs a dataset that carries
    # parent_gt (NeonPatchDataset). Precompute a fixed train subset (first N
    # parents) so the train/val parent metrics use a stable, cheap sample.
    do_parent_eval = parent_eval and hasattr(dataset, "parent_gt")
    if do_parent_eval:
        from collections import OrderedDict
        tr_by_parent: "OrderedDict[str, list]" = OrderedDict()
        for i in train_ds.indices:
            tr_by_parent.setdefault(dataset.samples[i]["parent"], []).append(i)
        train_eval_idx = [
            i for p in list(tr_by_parent)[:parent_eval_train_parents]
            for i in tr_by_parent[p]
        ]
        val_eval_idx = list(val_ds.indices)

    # Data-loading is the bottleneck (many small tifs on a network volume), so
    # keep workers alive across epochs and prefetch deeper to hide read latency.
    loader_extra = dict(persistent_workers=True, prefetch_factor=4) if num_workers > 0 else {}
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, **loader_extra,
    )

    optimizer, scheduler = build_stage_b_optimizer_and_scheduler(
        model, lr, weight_decay, total_epochs, warmup_epochs
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(total_epochs):
        model.train()
        # Keep encoder frozen
        model.encoder.encoder.eval()
        for p in model.encoder.encoder.parameters():
            p.requires_grad = False

        train_loss = _run_epoch(model, train_loader, optimizer, device, lam_cls, lam_l1, lam_giou, train=True,
                                desc=f"epoch {epoch+1}/{total_epochs} train")
        val_loss   = _run_epoch(model, val_loader, None, device, lam_cls, lam_l1, lam_giou, train=False,
                                desc=f"epoch {epoch+1}/{total_epochs} val")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(f"[Stage B] Epoch {epoch+1}/{total_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                ckpt_dir / "stage_b_best.pt",
            )
            # Persist immediately so a later crash/timeout can't lose the best model.
            if on_checkpoint is not None:
                on_checkpoint()
        else:
            patience_counter += 1

        log_dict = {
            "stage_b/epoch": epoch + 1,
            "stage_b/train_loss": train_loss,
            "stage_b/val_loss": val_loss,
            "stage_b/best_val_loss": best_val_loss,
            "stage_b/lr": lr,
            "stage_b/patience_counter": patience_counter,
        }

        # Parent-level (stitched 256) metrics — directly comparable to the
        # un-quartered baseline. Count MAE + F1 + the stitched matching loss, on
        # val and a fixed train subset (per-quarter train/val loss above isn't
        # comparable across tile sizes; these parent metrics are).
        if do_parent_eval and (epoch + 1) % parent_eval_every == 0:
            from ..eval.stitch import stitch_eval
            ev = dict(device=device, batch_size=batch_size, num_workers=num_workers,
                      conf_thresh=parent_eval_conf, nms_iou=parent_eval_nms_iou,
                      lam_cls=lam_cls, lam_l1=lam_l1, lam_giou=lam_giou)
            vp = stitch_eval(model, eval_dataset, indices=val_eval_idx, **ev)
            tp = stitch_eval(model, eval_dataset, indices=train_eval_idx, **ev)
            log_dict.update({
                "stage_b/parent_val_count_mae": vp["count"]["mae"],
                "stage_b/parent_val_f1": vp["f1_at_0.5"],
                "stage_b/parent_val_stitch_loss": vp["stitch_loss"],
                "stage_b/parent_train_count_mae": tp["count"]["mae"],
                "stage_b/parent_train_f1": tp["f1_at_0.5"],
                "stage_b/parent_train_stitch_loss": tp["stitch_loss"],
            })
            print(f"[Stage B] parent  val: mae={vp['count']['mae']:.2f} "
                  f"f1={vp['f1_at_0.5']:.3f} loss={vp['stitch_loss']:.4f}  |  "
                  f"train: mae={tp['count']['mae']:.2f} f1={tp['f1_at_0.5']:.3f} "
                  f"loss={tp['stitch_loss']:.4f}")

        # Periodic visual check: a few val tiles with predicted vs GT boxes,
        # merged into this epoch's single log call so the media actually shows.
        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                from ..eval.visualize import make_prediction_panels
                panels = make_prediction_panels(
                    model, val_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                log_dict.update(_wandb_images("stage_b/predictions", panels))
                print(f"[Stage B] Logged {len(panels)} prediction visuals at epoch {epoch+1}")
            except Exception as exc:
                print(f"[Stage B] WARNING: prediction viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

        if patience_counter >= patience:
            print(f"[Stage B] Early stopping at epoch {epoch+1}")
            break

    # Save final checkpoint
    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / "stage_b_final.pt")
    print(f"[Stage B] Done. Best val loss: {best_val_loss:.4f}")


def _run_epoch(model, loader, optimizer, device, lam_cls, lam_l1, lam_giou, train: bool,
               desc: str = "", log_every: int = 20) -> float:
    total_loss = 0.0
    n_batches = 0
    n_total = len(loader)
    t0 = time.time()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for i, batch in enumerate(loader):
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("boxes", "exhaustive")
            }
            if train:
                optimizer.zero_grad()
            cls_logits, pred_boxes = model(batch_gpu)
            loss = batch_matching_loss(
                cls_logits, pred_boxes,
                batch["boxes"], batch["exhaustive"],
                lam_cls, lam_l1, lam_giou,
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if log_every and (i % log_every == 0 or i == n_total - 1):
                rate = (i + 1) / max(1e-9, time.time() - t0)
                print(f"[Stage B] {desc}  batch {i+1}/{n_total}  "
                      f"loss={loss.item():.4f}  avg={total_loss/n_batches:.4f}  "
                      f"{rate:.2f} it/s", flush=True)
    return total_loss / max(1, n_batches)
