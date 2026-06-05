"""Stage B: RGB tree-dataset pretraining. Encoder frozen; only neck + head trained."""
import copy
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

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


def run_stage_b_finetune(
    model: TreeDetector,
    checkpoint_path: str,
    dataset: Dataset,
    collate_fn=None,
    checkpoint_dir: str = "checkpoints",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    val_fraction: float = 0.1,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
    device: Optional[str] = None,
    num_workers: int = 4,
    viz_every: int = 5,
    n_viz_tiles: int = 4,
    viz_conf_thresh: float = 0.5,
    on_checkpoint=None,
    out_prefix: str = "stage_b_ft",
):
    """Continue Stage B from a saved checkpoint at a FLAT learning rate.

    Same setup as run_stage_b (encoder frozen, neck+head trained, Hungarian
    matching loss) but: (1) weights are initialised from `checkpoint_path`, and
    (2) the LR is held constant at `lr` for all `epochs` — no warmup, no cosine
    decay, no early stopping. Checkpoints are written as
    {out_prefix}_best.pt / {out_prefix}_final.pt so the original run isn't
    clobbered."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Resume from the Stage B checkpoint. Accept both the {"model": ...} dicts
    # this script saves and a bare state_dict.
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    prev_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"[Stage B FT] Loaded weights from {checkpoint_path}"
          + (f" (epoch {prev_epoch})" if prev_epoch is not None else ""))

    if collate_fn is None:
        collate_fn = deepforest_collate_fn

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Turn augmentation off for the val subset (see run_stage_b for rationale).
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

    # Flat LR: AdamW over neck+head only, no scheduler.
    import torch.optim as optim
    optimizer = optim.AdamW(
        model.neck_head_parameters(), lr=lr, weight_decay=weight_decay
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epoch = 0
    for epoch in range(epochs):
        model.train()
        # Keep encoder frozen.
        model.encoder.encoder.eval()
        for p in model.encoder.encoder.parameters():
            p.requires_grad = False

        train_loss = _run_epoch(model, train_loader, optimizer, device, lam_cls, lam_l1, lam_giou, train=True,
                                desc=f"epoch {epoch+1}/{epochs} train")
        val_loss   = _run_epoch(model, val_loader, None, device, lam_cls, lam_l1, lam_giou, train=False,
                                desc=f"epoch {epoch+1}/{epochs} val")

        print(f"[Stage B FT] Epoch {epoch+1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                ckpt_dir / f"{out_prefix}_best.pt",
            )
            if on_checkpoint is not None:
                on_checkpoint()

        log_dict = {
            "stage_b_ft/epoch": epoch + 1,
            "stage_b_ft/train_loss": train_loss,
            "stage_b_ft/val_loss": val_loss,
            "stage_b_ft/best_val_loss": best_val_loss,
            "stage_b_ft/lr": lr,
        }

        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                from ..eval.visualize import make_prediction_panels
                panels = make_prediction_panels(
                    model, val_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                log_dict.update(_wandb_images("stage_b_ft/predictions", panels))
                print(f"[Stage B FT] Logged {len(panels)} prediction visuals at epoch {epoch+1}")
            except Exception as exc:
                print(f"[Stage B FT] WARNING: prediction viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / f"{out_prefix}_final.pt")
    if on_checkpoint is not None:
        on_checkpoint()
    print(f"[Stage B FT] Done. Best val loss: {best_val_loss:.4f}")


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
