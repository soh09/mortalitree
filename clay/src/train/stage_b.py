"""Stage B: RGB tree-dataset pretraining. Encoder frozen; only neck + head trained."""
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from ..data.deepforest_dataset import DeepForestDataset, deepforest_collate_fn
from ..model.detector import TreeDetector
from ..train.losses import batch_matching_loss
from ..train.schedulers import build_stage_b_optimizer_and_scheduler


def run_stage_b(
    model: TreeDetector,
    annotations_path: str,
    checkpoint_dir: str,
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
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dataset = DeepForestDataset(annotations_path, augment=True)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=deepforest_collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=deepforest_collate_fn, num_workers=num_workers,
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

        train_loss = _run_epoch(model, train_loader, optimizer, device, lam_cls, lam_l1, lam_giou, train=True)
        val_loss   = _run_epoch(model, val_loader, None, device, lam_cls, lam_l1, lam_giou, train=False)
        scheduler.step()

        print(f"[Stage B] Epoch {epoch+1}/{total_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                ckpt_dir / "stage_b_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Stage B] Early stopping at epoch {epoch+1}")
                break

    # Save final checkpoint
    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / "stage_b_final.pt")
    print(f"[Stage B] Done. Best val loss: {best_val_loss:.4f}")


def _run_epoch(model, loader, optimizer, device, lam_cls, lam_l1, lam_giou, train: bool) -> float:
    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
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
    return total_loss / max(1, n_batches)
