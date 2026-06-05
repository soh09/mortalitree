"""Stage B: RGB tree-dataset pretraining. Encoder frozen; only neck + head trained."""
import copy
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..data.deepforest_dataset import DeepForestDataset, deepforest_collate_fn
from ..eval.metrics import compute_detection_metrics
from ..model.detector import TreeDetector
from ..train.losses import (
    batch_matching_loss,
    batch_encoder_proposal_loss,
    ccm_loss,
)
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


def _trainable_state_dict(model) -> dict:
    """Model weights minus the frozen Clay encoder (keys under ``encoder.``).

    Stage B keeps the encoder frozen the whole time, so its ~311M params (~1.2 GB)
    never change and don't belong in a checkpoint — they're reloaded from the Clay
    checkpoint when the model is rebuilt. This keeps each saved file to just the
    from-scratch neck/CCM/CGFE/head (~tens of MB).
    """
    return {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}


def _save_ckpt(path, model, optimizer, scheduler, epoch, best_val_loss,
               patience_counter, extra=None):
    """Save full training state so a run can be resumed exactly (not just weights).

    Trainable weights are stored under the "model" key (frozen encoder excluded);
    warm-start loaders that read ``ckpt["model"]`` with ``strict=False`` (e.g.
    Stage C's `stage_b_checkpoint`) keep working, since the missing encoder keys
    are already populated from the Clay checkpoint.
    """
    state = {
        "epoch": epoch,
        "model": _trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


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
    lam_ccm: float = 1.0,
    lam_enc: float = 1.0,
    cgfe_start_epoch: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: int = 4,
    dataset: Optional[Dataset] = None,
    collate_fn=None,
    viz_every: int = 5,
    n_viz_tiles: int = 5,
    viz_conf_thresh: float = 0.5,
    ckpt_every: int = 0,
    resume_from: Optional[str] = None,
    on_checkpoint=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # DQ-DETR two-phase schedule (see stage_c): CGFE off for the first half so
    # the CCM counting head stabilises, then on for the rest.
    if cgfe_start_epoch is None:
        cgfe_start_epoch = total_epochs // 2

    # Default to the DeepForest RGB JSON; callers may pass a pre-built dataset
    # (e.g. NeonPatchDataset for 4-band NEON patches) and matching collate_fn.
    if dataset is None:
        if annotations_path is None:
            raise ValueError("Provide either `dataset` or `annotations_path`.")
        dataset = DeepForestDataset(annotations_path, augment=True)
    if collate_fn is None:
        collate_fn = deepforest_collate_fn

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Augment train only: random_split gives two Subsets over the SAME dataset,
    # so repoint the val subset at a shallow copy with augmentation off. The copy
    # shares the parsed items/data (no re-read) — only `augment` differs. This
    # keeps the val loss a clean, deterministic signal.
    if getattr(dataset, "augment", None) is not None:
        val_clean = copy.copy(dataset)
        val_clean.augment = None
        val_ds.dataset = val_clean

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    optimizer, scheduler = build_stage_b_optimizer_and_scheduler(
        model, lr, weight_decay, total_epochs, warmup_epochs
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    # Resume: restore weights + optimizer + scheduler + bookkeeping and continue
    # from the next epoch. The CGFE phase and LR are derived from the epoch each
    # loop, so they fall out correctly once start_epoch is restored. (The encoder
    # is frozen throughout Stage B, so the optimizer's param set never changes —
    # this resume is exact.)
    start_epoch = 0
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        # strict=False: the checkpoint omits the frozen encoder (already loaded
        # from the Clay checkpoint at model construction); only the trainable
        # neck/CCM/CGFE/head weights are present.
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            print(f"[Stage B] WARNING: unexpected keys in resume ckpt: {unexpected[:5]}...")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"[Stage B] Resuming from {resume_from}: starting at epoch "
              f"{start_epoch+1}/{total_epochs} (best_val_loss={best_val_loss:.4f}, "
              f"patience={patience_counter})")
    elif resume_from is not None:
        print(f"[Stage B] resume_from='{resume_from}' not found — starting fresh.")

    if start_epoch >= total_epochs:
        print(f"[Stage B] resumed epoch {start_epoch} >= total_epochs={total_epochs}; "
              f"nothing to train. Raise total_epochs to continue.")
        return

    for epoch in range(start_epoch, total_epochs):
        model.train()
        # Keep encoder frozen
        model.encoder.encoder.eval()
        for p in model.encoder.encoder.parameters():
            p.requires_grad = False

        cgfe_on = epoch >= cgfe_start_epoch
        if model.enable_cgfe != cgfe_on:
            print(f"[Stage B] Epoch {epoch+1}: {'enabling' if cgfe_on else 'disabling'} CGFE")
        model.set_cgfe_enabled(cgfe_on)

        train_comp = _run_epoch(model, train_loader, optimizer, device,
                                lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc, train=True,
                                desc=f"epoch {epoch+1}/{total_epochs} train")
        val_metrics, val_comp = _val_epoch(model, val_loader, device,
                                           lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
        train_loss, val_loss = train_comp["total"], val_comp["total"]
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        phase = 2 if cgfe_on else 1
        print(
            f"[Stage B] Epoch {epoch+1}/{total_epochs} (phase {phase})  "
            f"train={train_loss:.4f} (det={train_comp['det']:.3f} enc={train_comp['enc']:.3f} "
            f"ccm={train_comp['ccm']:.3f})  val={val_loss:.4f}  "
            f"f1={val_metrics['f1']:.4f}  prec={val_metrics['precision']:.4f}  "
            f"rec={val_metrics['recall']:.4f}  mAP50={val_metrics['mAP50']:.4f}  lr={lr:.2e}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            _save_ckpt(ckpt_dir / "stage_b_best.pt", model, optimizer, scheduler,
                       epoch, best_val_loss, patience_counter, extra={"val_loss": val_loss})
            # Persist immediately so a later crash/timeout can't lose the best model.
            if on_checkpoint is not None:
                on_checkpoint()
        else:
            patience_counter += 1

        # Periodic full-state checkpoint (rolling) for resuming an interrupted run.
        # Written after patience_counter is updated so the resumed state is exact.
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            _save_ckpt(ckpt_dir / "stage_b_last.pt", model, optimizer, scheduler,
                       epoch, best_val_loss, patience_counter, extra={"val_loss": val_loss})
            if on_checkpoint is not None:
                on_checkpoint()
            print(f"[Stage B] Wrote resume checkpoint stage_b_last.pt at epoch {epoch+1}")

        # Per-epoch scalar curves. Each is logged twice: once under the global
        # `stage_b/...` key (continuous across the whole run) and once under a
        # phase-namespaced key (`stage_b/phase1/...` or `phase2/...`) so each
        # DQ-DETR training phase — CGFE-off counting-stabilization vs CGFE-on —
        # has its own loss/eval curve that only spans its epochs.
        curves = {
            "train_loss": train_loss,
            "train_loss_det": train_comp["det"],
            "train_loss_enc": train_comp["enc"],
            "train_loss_ccm": train_comp["ccm"],
            "val_loss": val_loss,
            "val_loss_det": val_comp["det"],
            "val_loss_enc": val_comp["enc"],
            "val_loss_ccm": val_comp["ccm"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_mAP50": val_metrics["mAP50"],
            "val_mAP50_95": val_metrics["mAP50_95"],
            "val_count_mae": val_metrics["mae"],
            "val_count_rmse": val_metrics["rmse"],
            "val_count_r2": val_metrics["r2"],
        }
        log_dict = {
            "stage_b/epoch": epoch + 1,
            "stage_b/lr": lr,
            "stage_b/cgfe_enabled": int(cgfe_on),
            "stage_b/phase": phase,
            "stage_b/best_val_loss": best_val_loss,
            "stage_b/patience_counter": patience_counter,
        }
        phase_prefix = f"stage_b/phase{phase}"
        for name, value in curves.items():
            log_dict[f"stage_b/{name}"] = value
            log_dict[f"{phase_prefix}/{name}"] = value

        # Periodic visual check: prediction panels (pred green + GT blue) and
        # CCM density-map panels ([RGB | density heatmap | overlay]) on eval tiles.
        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                from ..eval.visualize import make_prediction_panels, make_density_panels
                train_panels = make_prediction_panels(
                    model, train_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                val_panels = make_prediction_panels(
                    model, val_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                density_panels = make_density_panels(
                    model, val_loader, device=device, n_tiles=n_viz_tiles,
                )
                log_dict.update(_wandb_images("stage_b/train_predictions", train_panels))
                log_dict.update(_wandb_images("stage_b/val_predictions", val_panels))
                log_dict.update(_wandb_images("stage_b/val_density", density_panels))
                log_dict.update(_wandb_images(f"{phase_prefix}/val_density", density_panels))
                print(
                    f"[Stage B] Logged {len(train_panels)} train + {len(val_panels)} val "
                    f"+ {len(density_panels)} density visuals at epoch {epoch+1}"
                )
            except Exception as exc:
                print(f"[Stage B] WARNING: viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

        if patience_counter >= patience:
            print(f"[Stage B] Early stopping at epoch {epoch+1}")
            break

    # Save final checkpoint
    _save_ckpt(ckpt_dir / "stage_b_final.pt", model, optimizer, scheduler,
               epoch, best_val_loss, patience_counter)
    print(f"[Stage B] Done. Best val loss: {best_val_loss:.4f}")


def _loss_components(model, out, batch, lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc):
    """DQ-DETR's three loss terms, kept separate for per-component curves.

    Returns a dict of tensors: ``total`` (the optimized loss, with lam weighting)
    plus the three raw components ``det`` / ``enc`` / ``ccm`` for logging.
    """
    det = batch_matching_loss(
        out["cls_logits"], out["pred_boxes"],
        batch["boxes"], batch["exhaustive"], lam_cls, lam_l1, lam_giou,
    )
    enc = batch_encoder_proposal_loss(
        out["enc_cls_logits"], out["enc_boxes"],
        batch["boxes"], batch["exhaustive"],
        lam_cls=lam_cls, lam_l1=lam_l1, lam_giou=lam_giou,
    )
    ccm = ccm_loss(out["count_logits"], batch["boxes"], model.ccm_params)
    total = det + lam_enc * enc + lam_ccm * ccm
    return {"total": total, "det": det, "enc": enc, "ccm": ccm}


def _accumulate(sums: dict, comps: dict) -> None:
    for k, v in comps.items():
        sums[k] = sums.get(k, 0.0) + v.item()


def _val_epoch(model, loader, device, lam_cls, lam_l1, lam_giou,
               lam_ccm, lam_enc) -> tuple[dict, dict]:
    model.eval()
    sums: dict = {}
    n = 0
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("boxes", "exhaustive")
            }
            out = model(batch_gpu)
            comps = _loss_components(model, out, batch, lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
            _accumulate(sums, comps)
            n += 1
            cls_logits, pred_boxes = out["cls_logits"], out["pred_boxes"]
            scores = cls_logits.sigmoid()
            for i in range(cls_logits.shape[0]):
                all_pred_boxes.append(pred_boxes[i].cpu())
                all_pred_scores.append(scores[i].cpu())
                all_gt_boxes.append(batch["boxes"][i])
    avg = {k: v / max(1, n) for k, v in sums.items()}
    metrics = compute_detection_metrics(all_pred_boxes, all_pred_scores, all_gt_boxes)
    return metrics, avg


def _run_epoch(model, loader, optimizer, device, lam_cls, lam_l1, lam_giou,
               lam_ccm, lam_enc, train: bool, desc: str = "", log_every: int = 20) -> dict:
    sums: dict = {}
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
            out = model(batch_gpu)
            comps = _loss_components(model, out, batch, lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
            loss = comps["total"]
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            _accumulate(sums, comps)
            n_batches += 1
            if log_every and (i % log_every == 0 or i == n_total - 1):
                rate = (i + 1) / max(1e-9, time.time() - t0)
                print(f"[Stage B] {desc}  batch {i+1}/{n_total}  "
                      f"loss={loss.item():.4f}  avg={sums['total']/n_batches:.4f}  "
                      f"(det={comps['det'].item():.3f} enc={comps['enc'].item():.3f} "
                      f"ccm={comps['ccm'].item():.3f})  {rate:.2f} it/s", flush=True)
    return {k: v / max(1, n_batches) for k, v in sums.items()}
