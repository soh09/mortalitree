"""Unified Stage-C NAIP finetuning for all three detectors.

Handles both the fixed-query models (q150 / q500 -> FixedQueryTreeDetector, plain
Hungarian detection loss) and the DQ-DETR model (TreeDetector, detection +
two-stage-proposal + CCM counting loss). Both detectors return a dict with
``cls_logits`` / ``pred_boxes`` so the eval, viz and checkpointing paths are
shared; the only ``is_dqdetr`` branches are the extra loss terms, the CGFE phase
switch, and the CCM density-map visualization.

Per epoch it logs train AND val detection metrics (f1 / precision / recall /
mAP / count error) and losses to W&B, renders prediction-vs-GT panels on both
train and val tiles (plus CCM density overlays for DQ-DETR), and every
``ckpt_every`` epochs writes a model-named checkpoint
(``{prefix}_stage_c_epoch{NNN}.pt``) alongside rolling best/final.
"""
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..data.naip_dataset import NAIPTileDataset, naip_collate_fn
from ..train.losses import (
    batch_matching_loss,
    batch_encoder_proposal_loss,
    ccm_loss,
)
from ..train.schedulers import (
    build_stage_b_optimizer_and_scheduler,
    build_stage_c_optimizer_and_scheduler,
)
from ..eval.metrics import compute_detection_metrics


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


def _total_loss(model, out, batch, is_dqdetr,
                lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc):
    """Detection loss (+ two-stage proposal + CCM for DQ-DETR)."""
    det = batch_matching_loss(
        out["cls_logits"], out["pred_boxes"],
        batch["boxes"], batch["exhaustive"], lam_cls, lam_l1, lam_giou,
    )
    if not is_dqdetr:
        return det
    enc = batch_encoder_proposal_loss(
        out["enc_cls_logits"], out["enc_boxes"],
        batch["boxes"], batch["exhaustive"],
        lam_cls=lam_cls, lam_l1=lam_l1, lam_giou=lam_giou,
    )
    ccm = ccm_loss(out["count_logits"], batch["boxes"], model.ccm_params)
    return det + lam_enc * enc + lam_ccm * ccm


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in ("boxes", "exhaustive", "tile_paths")}


def _train_epoch(model, loader, optimizer, device, is_dqdetr,
                 lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(_to_device(batch, device))
        loss = _total_loss(model, out, batch, is_dqdetr,
                           lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def _evaluate(model, loader, device, is_dqdetr,
              lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc,
              conf_thresh, iou_thresh) -> tuple[dict, float]:
    """Detection metrics + mean loss over a loader (model put in eval mode)."""
    model.eval()
    total, n = 0.0, 0
    pred_boxes, pred_scores, gt_boxes = [], [], []
    for batch in loader:
        out = model(_to_device(batch, device))
        total += _total_loss(model, out, batch, is_dqdetr,
                             lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc).item()
        n += 1
        scores = out["cls_logits"].sigmoid()
        for i in range(out["cls_logits"].shape[0]):
            pred_boxes.append(out["pred_boxes"][i].cpu())
            pred_scores.append(scores[i].cpu())
            gt_boxes.append(batch["boxes"][i])
    metrics = compute_detection_metrics(
        pred_boxes, pred_scores, gt_boxes, iou_thresh, conf_thresh)
    return metrics, total / max(1, n)


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


def run_stage_c_multi(
    model,
    is_dqdetr: bool,
    ckpt_prefix: str,
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
    viz_every: int = 1,
    n_viz_tiles: int = 5,
    viz_conf_thresh: float = 0.5,
    eval_conf_thresh: float = 0.5,
    eval_iou_thresh: float = 0.5,
    ckpt_every: int = 5,
    patience: int = 20,
    on_checkpoint=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # DQ-DETR two-phase: CGFE off for the first stretch (CCM stabilises), then on.
    if is_dqdetr and cgfe_start_epoch is None:
        cgfe_start_epoch = total_epochs // 2
    model = model.to(device)

    if stage_b_checkpoint is not None:
        ckpt = torch.load(stage_b_checkpoint, map_location=device, weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[Stage C:{ckpt_prefix}] loaded Stage-B weights from {stage_b_checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Train loader (augmented). A second, un-augmented train loader is used for
    # honest train-set metrics/visuals so they aren't measured on jittered tiles.
    train_ds = NAIPTileDataset(train_annotations_path, norm_stats_path, augment=True)
    train_eval_ds = NAIPTileDataset(train_annotations_path, norm_stats_path, augment=False)
    val_ds = NAIPTileDataset(val_annotations_path, norm_stats_path, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=naip_collate_fn, num_workers=num_workers,
                              pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False,
                                   collate_fn=naip_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=naip_collate_fn, num_workers=num_workers)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0   # so epoch 1 always writes a _best.pt (even at F1=0 early on)
    patience_counter = 0

    optimizer, scheduler = build_stage_b_optimizer_and_scheduler(
        model, neck_head_lr, weight_decay, frozen_epochs, warmup_epochs)

    last_epoch = 0
    for epoch in range(total_epochs):
        last_epoch = epoch
        if is_dqdetr:
            cgfe_on = epoch >= cgfe_start_epoch
            if model.enable_cgfe != cgfe_on:
                print(f"[Stage C:{ckpt_prefix}] epoch {epoch+1}: "
                      f"{'enabling' if cgfe_on else 'disabling'} CGFE")
            model.set_cgfe_enabled(cgfe_on)
        else:
            cgfe_on = False

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

        train_loss = _train_epoch(model, train_loader, optimizer, device, is_dqdetr,
                                  lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc)

        # Train + val metrics (both in eval mode; train metrics on un-augmented tiles).
        train_metrics, train_eval_loss = _evaluate(
            model, train_eval_loader, device, is_dqdetr,
            lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc,
            eval_conf_thresh, eval_iou_thresh)
        val_metrics, val_loss = _evaluate(
            model, val_loader, device, is_dqdetr,
            lam_cls, lam_l1, lam_giou, lam_ccm, lam_enc,
            eval_conf_thresh, eval_iou_thresh)
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

        # Best-so-far update (before logging so the row reflects this epoch).
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
            "stage_c/cgfe_enabled": int(cgfe_on),
            "stage_c/best_val_f1": max(best_val_f1, 0.0),
            "stage_c/patience_counter": patience_counter,
        }
        log_dict.update(_metric_log("stage_c/train", train_metrics, train_loss))
        log_dict.update(_metric_log("stage_c/val", val_metrics, val_loss))

        # Per-epoch visuals: predictions-vs-GT on fixed train + val tiles, and the
        # CCM density overlay for DQ-DETR.
        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                from ..eval.visualize import make_prediction_panels, make_density_panels
                tr = make_prediction_panels(model, train_eval_loader, device=device,
                                            conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles)
                va = make_prediction_panels(model, val_loader, device=device,
                                            conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles)
                log_dict.update(_wandb_images("stage_c/train_predictions", tr))
                log_dict.update(_wandb_images("stage_c/val_predictions", va))
                if is_dqdetr:
                    tr_d = make_density_panels(model, train_eval_loader,
                                               device=device, n_tiles=n_viz_tiles)
                    va_d = make_density_panels(model, val_loader,
                                               device=device, n_tiles=n_viz_tiles)
                    log_dict.update(_wandb_images("stage_c/train_density", tr_d))
                    log_dict.update(_wandb_images("stage_c/val_density", va_d))
            except Exception as exc:
                print(f"[Stage C:{ckpt_prefix}] WARNING viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

        # Rolling model-named checkpoint every ckpt_every epochs.
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_f1": val_f1},
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
