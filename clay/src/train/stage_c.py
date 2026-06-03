"""Stage C: NAIP finetuning with late encoder unfreezing."""
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from ..data.naip_dataset import NAIPTileDataset, naip_collate_fn
from ..model.detector import TreeDetector
from ..train.losses import batch_matching_loss
from ..train.schedulers import build_stage_b_optimizer_and_scheduler, build_stage_c_optimizer_and_scheduler
from ..eval.metrics import compute_f1_at_iou


def run_stage_c(
    model: TreeDetector,
    train_annotations_path: str,
    val_annotations_path: str,
    checkpoint_dir: str,
    norm_stats_path: Optional[str] = None,
    total_epochs: int = 80,
    frozen_epochs: int = 30,
    batch_size: int = 8,
    neck_head_lr: float = 5e-4,
    encoder_lr: float = 1e-5,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    unfreeze_n_blocks: int = 2,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
    device: Optional[str] = None,
    num_workers: int = 4,
    stage_b_checkpoint: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if stage_b_checkpoint is not None:
        ckpt = torch.load(stage_b_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[Stage C] Loaded Stage B checkpoint: {stage_b_checkpoint}")

    train_ds = NAIPTileDataset(train_annotations_path, norm_stats_path, augment=True)
    val_ds   = NAIPTileDataset(val_annotations_path,   norm_stats_path, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=naip_collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=naip_collate_fn, num_workers=num_workers,
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience = 15
    patience_counter = 0

    # Phase 1: frozen encoder
    optimizer, scheduler = build_stage_b_optimizer_and_scheduler(
        model, neck_head_lr, weight_decay, frozen_epochs, warmup_epochs
    )

    for epoch in range(total_epochs):
        # Unfreeze after frozen_epochs
        if epoch == frozen_epochs:
            print(f"[Stage C] Epoch {epoch+1}: unfreezing last {unfreeze_n_blocks} encoder blocks")
            model.unfreeze_last_n_encoder_blocks(unfreeze_n_blocks)
            optimizer, scheduler = build_stage_c_optimizer_and_scheduler(
                model, neck_head_lr, encoder_lr, weight_decay,
                total_epochs - frozen_epochs, warmup_epochs,
            )

        model.train()
        if epoch < frozen_epochs:
            # Keep encoder frozen
            model.encoder.encoder.eval()
            for p in model.encoder.encoder.parameters():
                p.requires_grad = False

        train_loss = _run_epoch(model, train_loader, optimizer, device, lam_cls, lam_l1, lam_giou, train=True)
        val_f1, val_loss = _val_epoch(model, val_loader, device, lam_cls, lam_l1, lam_giou)
        scheduler.step()

        print(
            f"[Stage C] Epoch {epoch+1}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_f1": val_f1},
                ckpt_dir / "stage_c_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Stage C] Early stopping at epoch {epoch+1}")
                break

    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / "stage_c_final.pt")
    print(f"[Stage C] Done. Best val F1: {best_val_f1:.4f}")


def _run_epoch(model, loader, optimizer, device, lam_cls, lam_l1, lam_giou, train: bool) -> float:
    total_loss = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("boxes", "exhaustive", "tile_paths")
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


def _val_epoch(model, loader, device, lam_cls, lam_l1, lam_giou) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ("boxes", "exhaustive", "tile_paths")
            }
            cls_logits, pred_boxes = model(batch_gpu)
            loss = batch_matching_loss(
                cls_logits, pred_boxes,
                batch["boxes"], batch["exhaustive"],
                lam_cls, lam_l1, lam_giou,
            )
            total_loss += loss.item()
            scores = cls_logits.sigmoid()
            for i in range(cls_logits.shape[0]):
                all_pred_boxes.append(pred_boxes[i].cpu())
                all_pred_scores.append(scores[i].cpu())
                all_gt_boxes.append(batch["boxes"][i])
    avg_loss = total_loss / max(1, len(loader))
    f1 = compute_f1_at_iou(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh=0.5)
    return f1, avg_loss
