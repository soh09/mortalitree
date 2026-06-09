"""Stage C: NAIP finetuning with late encoder unfreezing."""
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from ..data.naip_dataset import NAIPTileDataset, naip_collate_fn
from ..model.detector import TreeDetector
from ..train.losses import (
    batch_matching_loss,
    batch_encoder_proposal_loss,
    ccm_loss,
)
from ..train.schedulers import build_stage_b_optimizer_and_scheduler, build_stage_c_optimizer_and_scheduler
from ..eval.metrics import compute_detection_metrics


def _wandb_log(metrics: dict) -> None:
    """Log one history row to W&B if a run is active. No explicit step — wandb
    auto-increments, one row per call, so call it once per epoch."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass


def _wandb_images(key: str, images: list) -> dict:
    """Return {key: [wandb.Image, ...]} to merge into an epoch's log dict (empty
    if no active run or no images)."""
    if not images:
        return {}
    try:
        import wandb
        if wandb.run is not None:
            return {key: [wandb.Image(im) for im in images]}
    except ImportError:
        pass
    return {}


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
    lam_ccm: float = 1.0,
    lam_enc: float = 1.0,
    cgfe_start_epoch: Optional[int] = None,
    device: Optional[str] = None,
    num_workers: int = 4,
    stage_b_checkpoint: Optional[str] = None,
    viz_every: int = 5,
    n_viz_tiles: int = 5,
    viz_conf_thresh: float = 0.5,
    on_checkpoint=None,
):
    # Two-phase DQ-DETR schedule: Stage 1 keeps CGFE off so the counting head
    # (CCM) stabilises; Stage 2 enables it once the density map is reliable.
    # Default the switch to the first half of the training budget.
    if cgfe_start_epoch is None:
        cgfe_start_epoch = total_epochs // 2
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
        # DQ-DETR two-phase: enable CGFE once counting has had time to stabilise.
        cgfe_on = epoch >= cgfe_start_epoch
        if model.enable_cgfe != cgfe_on:
            print(f"[Stage C] Epoch {epoch+1}: {'enabling' if cgfe_on else 'disabling'} CGFE")
        model.set_cgfe_enabled(cgfe_on)

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

        train_loss = _run_epoch(model, train_loader, optimizer, device,
                                lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc, train=True)
        val_metrics, val_loss = _val_epoch(model, val_loader, device,
                                           lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
        val_f1 = val_metrics["f1"]
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"[Stage C] Epoch {epoch+1}/{total_epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  f1={val_f1:.4f}  prec={val_metrics['precision']:.4f}  "
            f"rec={val_metrics['recall']:.4f}  mAP50={val_metrics['mAP50']:.4f}  lr={lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_f1": val_f1},
                ckpt_dir / "stage_c_best.pt",
            )
            # Persist immediately so a later crash/timeout can't lose the best model.
            if on_checkpoint is not None:
                on_checkpoint()
        else:
            patience_counter += 1

        log_dict = {
            "stage_c/epoch": epoch + 1,
            "stage_c/train_loss": train_loss,
            "stage_c/val_loss": val_loss,
            "stage_c/val_f1": val_f1,
            "stage_c/val_precision": val_metrics["precision"],
            "stage_c/val_recall": val_metrics["recall"],
            "stage_c/val_mAP50": val_metrics["mAP50"],
            "stage_c/val_mAP50_95": val_metrics["mAP50_95"],
            "stage_c/val_count_mae": val_metrics["mae"],
            "stage_c/val_count_rmse": val_metrics["rmse"],
            "stage_c/val_count_r2": val_metrics["r2"],
            "stage_c/best_val_f1": best_val_f1,
            "stage_c/lr": lr,
            "stage_c/encoder_unfrozen": int(epoch >= frozen_epochs),
            "stage_c/cgfe_enabled": int(cgfe_on),
            "stage_c/patience_counter": patience_counter,
        }

        # Periodic visual check: 5 train tiles + 5 val tiles with predicted (green)
        # and GT (blue) boxes, merged into this epoch's single log call.
        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                from ..eval.visualize import make_prediction_panels
                train_panels = make_prediction_panels(
                    model, train_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                val_panels = make_prediction_panels(
                    model, val_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                log_dict.update(_wandb_images("stage_c/train_predictions", train_panels))
                log_dict.update(_wandb_images("stage_c/val_predictions", val_panels))
                print(
                    f"[Stage C] Logged {len(train_panels)} train + {len(val_panels)} val "
                    f"visuals at epoch {epoch+1}"
                )
            except Exception as exc:
                print(f"[Stage C] WARNING: prediction viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

        if patience_counter >= patience:
            print(f"[Stage C] Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / "stage_c_final.pt")
    print(f"[Stage C] Done. Best val F1: {best_val_f1:.4f}")


def _total_loss(model, out, batch, lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc):
    """Detection + two-stage proposal + CCM losses (DQ-DETR's three terms)."""
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
    return det + lam_enc * enc + lam_ccm * ccm


def _run_epoch(model, loader, optimizer, device, lam_cls, lam_l1, lam_giou,
               lam_ccm, lam_enc, train: bool) -> float:
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
            out = model(batch_gpu)
            loss = _total_loss(model, out, batch, lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def _val_epoch(model, loader, device, lam_cls, lam_l1, lam_giou,
               lam_ccm, lam_enc) -> tuple[dict, float]:
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
            out = model(batch_gpu)
            loss = _total_loss(model, out, batch, lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
            total_loss += loss.item()
            cls_logits, pred_boxes = out["cls_logits"], out["pred_boxes"]
            scores = cls_logits.sigmoid()
            for i in range(cls_logits.shape[0]):
                all_pred_boxes.append(pred_boxes[i].cpu())
                all_pred_scores.append(scores[i].cpu())
                all_gt_boxes.append(batch["boxes"][i])
    avg_loss = total_loss / max(1, len(loader))
    metrics = compute_detection_metrics(all_pred_boxes, all_pred_scores, all_gt_boxes)
    return metrics, avg_loss
