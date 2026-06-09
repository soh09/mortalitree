"""Stage-C finetuning for the patch-quartering model.

Each 256 tile is split into a 2x2 grid of 128px quarters. Training runs the
fixed-query detector on the quarters (per-quarter GT, per-quarter Hungarian
matching loss). Evaluation, visualization and the reported metrics are all
parent-level: quarter predictions are stitched back to the 256 frame (mapped by
crop offset + NMS) and scored against the full parent GT — directly comparable to
the un-quartered detectors' Stage-C numbers.

Logs train + val parent-level f1/precision/recall/mAP/count each epoch, renders
per-epoch stitched prediction-vs-GT panels on fixed train + val parents, and
writes model-named checkpoints (quarter_stage_c_epoch{NNN}.pt) every ckpt_every.
"""
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..data.naip_quarter_dataset import (
    QuarteredNAIPDataset,
    quarter_collate_fn,
    stitch_metrics,
    make_quarter_panels,
)
from ..train.losses import batch_matching_loss
from ..train.schedulers import (
    build_stage_b_optimizer_and_scheduler,
    build_stage_c_optimizer_and_scheduler,
)


def _wandb_log(metrics: dict) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass


def _wandb_images(key: str, images: list) -> dict:
    if not images:
        return {}
    try:
        import wandb
        if wandb.run is not None:
            return {key: [wandb.Image(im) for im in images]}
    except ImportError:
        pass
    return {}


def _to_device(batch, device):
    skip = ("boxes", "exhaustive", "parent", "crop")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items() if k not in skip}


def _quarter_loss(model, batch, device, lam_cls, lam_l1, lam_giou):
    out = model(_to_device(batch, device))
    return batch_matching_loss(out["cls_logits"], out["pred_boxes"],
                               batch["boxes"], batch["exhaustive"],
                               lam_cls, lam_l1, lam_giou)


def _train_epoch(model, loader, optimizer, device, lam_cls, lam_l1, lam_giou) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        loss = _quarter_loss(model, batch, device, lam_cls, lam_l1, lam_giou)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def _val_loss(model, loader, device, lam_cls, lam_l1, lam_giou) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += _quarter_loss(model, batch, device, lam_cls, lam_l1, lam_giou).item()
        n += 1
    return total / max(1, n)


def _metric_log(prefix: str, metrics: dict, loss: float) -> dict:
    return {
        f"{prefix}/loss": loss,
        f"{prefix}/f1": metrics["f1"],
        f"{prefix}/precision": metrics["precision"],
        f"{prefix}/recall": metrics["recall"],
        f"{prefix}/mAP50": metrics["mAP50"],
        f"{prefix}/mAP50_95": metrics["mAP50_95"],
        f"{prefix}/count_mae": metrics["mae"],
        f"{prefix}/count_rmse": metrics["rmse"],
        f"{prefix}/count_r2": metrics["r2"],
    }


def run_stage_c_quarter(
    model,
    ckpt_prefix: str,
    train_annotations_path: str,
    val_annotations_path: str,
    checkpoint_dir: str,
    norm_stats_path: Optional[str] = None,
    total_epochs: int = 80,
    frozen_epochs: int = 30,
    batch_size: int = 32,
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
    viz_every: int = 1,
    n_viz_tiles: int = 5,
    viz_conf_thresh: float = 0.5,
    eval_conf_thresh: float = 0.5,
    eval_iou_thresh: float = 0.5,
    nms_iou: float = 0.6,
    ckpt_every: int = 5,
    patience: int = 20,
    on_checkpoint=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if stage_b_checkpoint is not None:
        ckpt = torch.load(stage_b_checkpoint, map_location=device, weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Stage C:{ckpt_prefix}] loaded Stage-B weights from {stage_b_checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    train_ds = QuarteredNAIPDataset(train_annotations_path, norm_stats_path, augment=True)
    train_eval_ds = QuarteredNAIPDataset(train_annotations_path, norm_stats_path, augment=False)
    val_ds = QuarteredNAIPDataset(val_annotations_path, norm_stats_path, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=quarter_collate_fn, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=quarter_collate_fn, num_workers=num_workers)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    patience_counter = 0
    optimizer, scheduler = build_stage_b_optimizer_and_scheduler(
        model, neck_head_lr, weight_decay, frozen_epochs, warmup_epochs)

    last_epoch = 0
    for epoch in range(total_epochs):
        last_epoch = epoch
        if epoch == frozen_epochs:
            print(f"[Stage C:{ckpt_prefix}] epoch {epoch+1}: unfreezing last "
                  f"{unfreeze_n_blocks} encoder blocks")
            model.unfreeze_last_n_encoder_blocks(unfreeze_n_blocks)
            optimizer, scheduler = build_stage_c_optimizer_and_scheduler(
                model, neck_head_lr, encoder_lr, weight_decay,
                total_epochs - frozen_epochs, warmup_epochs)

        if epoch < frozen_epochs:
            model.encoder.encoder.eval()
            for p in model.encoder.encoder.parameters():
                p.requires_grad = False

        train_loss = _train_epoch(model, train_loader, optimizer, device,
                                  lam_cls, lam_l1, lam_giou)
        val_loss = _val_loss(model, val_loader, device, lam_cls, lam_l1, lam_giou)

        # Stitched parent-level metrics on train (un-augmented) and val.
        ev = dict(conf_thresh=eval_conf_thresh, iou_thresh=eval_iou_thresh,
                  nms_iou=nms_iou, batch_size=batch_size, num_workers=num_workers)
        train_metrics = stitch_metrics(model, train_eval_ds, device, **ev)
        val_metrics = stitch_metrics(model, val_ds, device, **ev)
        val_f1 = val_metrics["f1"]
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"[Stage C:{ckpt_prefix}] epoch {epoch+1}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_f1={train_metrics['f1']:.4f}  val_f1={val_f1:.4f}  "
            f"val_prec={val_metrics['precision']:.4f}  val_rec={val_metrics['recall']:.4f}  "
            f"val_mAP50={val_metrics['mAP50']:.4f}  lr={lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_f1": val_f1},
                       ckpt_dir / f"{ckpt_prefix}_stage_c_best.pt")
            if on_checkpoint is not None:
                on_checkpoint()
        else:
            patience_counter += 1

        log_dict = {
            "stage_c/epoch": epoch + 1,
            "stage_c/lr": lr,
            "stage_c/encoder_unfrozen": int(epoch >= frozen_epochs),
            "stage_c/best_val_f1": max(best_val_f1, 0.0),
            "stage_c/patience_counter": patience_counter,
        }
        log_dict.update(_metric_log("stage_c/train", train_metrics, train_loss))
        log_dict.update(_metric_log("stage_c/val", val_metrics, val_loss))

        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                tr = make_quarter_panels(model, train_eval_ds, device,
                                         n_parents=n_viz_tiles, conf_thresh=viz_conf_thresh,
                                         nms_iou=nms_iou)
                va = make_quarter_panels(model, val_ds, device,
                                         n_parents=n_viz_tiles, conf_thresh=viz_conf_thresh,
                                         nms_iou=nms_iou)
                log_dict.update(_wandb_images("stage_c/train_predictions", tr))
                log_dict.update(_wandb_images("stage_c/val_predictions", va))
            except Exception as exc:
                print(f"[Stage C:{ckpt_prefix}] WARNING viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_f1": val_f1},
                       ckpt_dir / f"{ckpt_prefix}_stage_c_epoch{epoch+1:03d}.pt")
            if on_checkpoint is not None:
                on_checkpoint()

        if patience_counter >= patience:
            print(f"[Stage C:{ckpt_prefix}] early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": last_epoch, "model": model.state_dict()},
               ckpt_dir / f"{ckpt_prefix}_stage_c_final.pt")
    if on_checkpoint is not None:
        on_checkpoint()
    print(f"[Stage C:{ckpt_prefix}] done. best val F1 = {best_val_f1:.4f}")
    return best_val_f1
