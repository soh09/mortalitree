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
    device: Optional[str] = None,
    num_workers: int = 4,
    stage_b_checkpoint: Optional[str] = None,
    viz_every: int = 5,
    n_viz_tiles: int = 4,
    viz_conf_thresh: float = 0.5,
    on_checkpoint=None,
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
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"[Stage C] Epoch {epoch+1}/{total_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  lr={lr:.2e}"
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
            "stage_c/best_val_f1": best_val_f1,
            "stage_c/lr": lr,
            "stage_c/encoder_unfrozen": int(epoch >= frozen_epochs),
            "stage_c/patience_counter": patience_counter,
        }

        # Periodic visual check, merged into this epoch's single log call.
        if viz_every and ((epoch + 1) % viz_every == 0 or epoch == 0):
            try:
                from ..eval.visualize import make_prediction_panels
                panels = make_prediction_panels(
                    model, val_loader, device=device,
                    conf_thresh=viz_conf_thresh, n_tiles=n_viz_tiles,
                )
                log_dict.update(_wandb_images("stage_c/predictions", panels))
                print(f"[Stage C] Logged {len(panels)} prediction visuals at epoch {epoch+1}")
            except Exception as exc:
                print(f"[Stage C] WARNING: prediction viz failed at epoch {epoch+1}: {exc}")

        _wandb_log(log_dict)

        if patience_counter >= patience:
            print(f"[Stage C] Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / "stage_c_final.pt")
    print(f"[Stage C] Done. Best val F1: {best_val_f1:.4f}")


def _dump_stitch_preds(result: dict, epoch, split: str, conf_thresh: float, out_dir) -> None:
    """Write the stitched per-tile predictions from a `stitch_eval(return_preds=True)`
    result to {out_dir}/{split}_epoch{epoch}.json. Boxes are normalized cxcywh in the
    256 parent frame; all NMS'd predictions are saved (unfiltered) so a threshold can
    be applied offline."""
    import json
    from pathlib import Path

    tiles = []
    for parent, pb, ps, gt in zip(result["parents"], result["pred_boxes"],
                                  result["pred_scores"], result["gt_boxes"]):
        tiles.append({
            "tile": Path(parent).name,
            "pred_boxes": pb.tolist(),
            "pred_scores": ps.tolist(),
            "gt_boxes": gt.tolist(),
        })
    tag = epoch if isinstance(epoch, str) else f"{epoch:03d}"
    path = Path(out_dir) / f"{split}_epoch{tag}.json"
    with open(path, "w") as f:
        json.dump({"epoch": epoch, "split": split, "conf_thresh": conf_thresh,
                   "frame": "parent-256 normalized cxcywh", "tiles": tiles}, f)


def _save_stitch_panels(result: dict, epoch, split: str, conf_thresh: float,
                        out_dir, n_tiles: int = 4) -> None:
    """Render up to `n_tiles` parent tiles with GT (blue) and predicted boxes above
    `conf_thresh` (green, scored) drawn on the 256 RGB, saved as one PNG grid at
    {out_dir}/{split}_epoch{epoch}.png. Uses matplotlib (no cv2 dependency)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import rasterio
    from pathlib import Path

    n = min(n_tiles, len(result["parents"]))
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n):
        parent = result["parents"][i]
        pb = result["pred_boxes"][i].numpy()
        ps = result["pred_scores"][i].numpy()
        gt = result["gt_boxes"][i].numpy()
        try:
            with rasterio.open(parent) as src:
                rgb = src.read(indexes=[1, 2, 3]).astype(np.float32)
        except Exception:
            continue
        rgb = (rgb - rgb.min()) / max(1e-6, float(rgb.max() - rgb.min()))
        img = np.clip(rgb, 0, 1).transpose(1, 2, 0)
        H, W = img.shape[:2]
        ax = axes.ravel()[i]
        ax.imshow(img)

        def _draw(boxes, color, scores=None):
            for j, (cx, cy, w, h) in enumerate(boxes):
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                ax.add_patch(Rectangle((x1, y1), w * W, h * H, fill=False,
                                       edgecolor=color, linewidth=0.6))

        _draw(gt, "blue")
        keep = ps >= conf_thresh
        _draw(pb[keep], "lime")
        ax.set_title(f"{Path(parent).name}\npred={int(keep.sum())} gt={len(gt)}", fontsize=7)

    tag = epoch if isinstance(epoch, str) else f"{epoch:03d}"
    fig.suptitle(f"{split} epoch {epoch} (GT=blue, pred≥{conf_thresh}=green)", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{split}_epoch{tag}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _best_threshold(result: dict, thresholds) -> tuple:
    """Sweep confidence thresholds over a `stitch_eval(return_preds=True)` result and
    return (best_threshold, best_f1) at IoU 0.5. Used to pick an operating point on
    val instead of hard-coding 0.5 — this detector is high-recall, so its best-F1
    threshold is well below 0.5 (spec: 'sweep on val for best F1')."""
    from ..eval.metrics import compute_f1_at_iou
    best_th, best_f1 = thresholds[0], -1.0
    for th in thresholds:
        f1 = compute_f1_at_iou(result["pred_boxes"], result["pred_scores"],
                               result["gt_boxes"], 0.5, th)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1


def run_stage_c_quartered(
    model: TreeDetector,
    train_annotations_path: str,
    val_annotations_path: str,
    checkpoint_dir: str,
    test_annotations_path: Optional[str] = None,
    norm_stats_path: Optional[str] = None,
    tiles_root: Optional[str] = None,
    tile_size: int = 256,
    n_split: int = 2,
    total_epochs: int = 80,
    frozen_epochs: int = 30,
    batch_size: int = 16,
    neck_head_lr: float = 5e-4,
    encoder_lr: float = 1e-5,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    unfreeze_n_blocks: int = 2,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
    conf_thresh: float = 0.5,
    nms_iou: float = 0.6,
    patience: int = 15,
    device: Optional[str] = None,
    num_workers: int = 4,
    stage_b_checkpoint: Optional[str] = None,
    out_prefix: str = "stage_c_quarter",
    log_dir: Optional[str] = None,
    save_preds_every: int = 1,
    panels_every: int = 1,
    n_panel_tiles: int = 4,
    panel_conf_thresh: float = 0.25,
    on_checkpoint=None,
):
    """Stage-C fine-tune of the *quarter* (128-px) detector on NAIP JSON tiles.

    Each 256 tile is split into ``n_split`` x ``n_split`` quarters (128 px); the
    model trains per-quarter with the Hungarian matching loss. Validation and the
    final test run stitch each tile's quarter predictions back into the 256 parent
    frame (NMS de-dups crowns caught by two adjacent quarters) and score against
    the full 256-frame GT — so the reported F1 is directly comparable to the
    un-quartered baseline and matches quartered inference.

    Recipe (matches run_stage_c): the Clay encoder is frozen for ``frozen_epochs``
    while neck+head train, then the last ``unfreeze_n_blocks`` encoder blocks are
    unfrozen at a small LR. The best checkpoint is chosen by stitched val F1; at the
    end it is reloaded and evaluated once on the held-out test set.
    """
    from ..data.naip_dataset import QuarteredNAIPDataset
    from ..data.deepforest_dataset import deepforest_collate_fn
    from ..eval.stitch import stitch_eval

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)

    if stage_b_checkpoint is not None:
        ckpt = torch.load(stage_b_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        crit = [k for k in missing if k.startswith(("neck.", "head."))
                or "encoder.encoder" in k]
        if crit:
            raise RuntimeError(f"Stage B checkpoint missing critical weights: {crit[:5]} ...")
        print(f"[Stage C-q] Loaded Stage B checkpoint: {stage_b_checkpoint}")

    dskw = dict(norm_stats_path=norm_stats_path, tile_size=tile_size, n_split=n_split,
                tiles_root=tiles_root)
    train_ds = QuarteredNAIPDataset(train_annotations_path, augment=True,  **dskw)
    val_ds   = QuarteredNAIPDataset(val_annotations_path,   augment=False, **dskw)
    test_ds  = (QuarteredNAIPDataset(test_annotations_path, augment=False, **dskw)
                if test_annotations_path else None)

    print(f"[Stage C-q] train: {len(train_ds)} quarters / {len(train_ds.parent_gt)} tiles | "
          f"val: {len(val_ds)} quarters / {len(val_ds.parent_gt)} tiles"
          + (f" | test: {len(test_ds)} quarters / {len(test_ds.parent_gt)} tiles"
             if test_ds else ""))

    # Warn if any quarter outgrew the query budget (Hungarian matching drops GT).
    q = model.head.num_queries
    max_boxes = max((len(s["boxes"]) for s in train_ds.samples), default=0)
    print(f"[Stage C-q] densest train quarter = {max_boxes} boxes vs num_queries={q}"
          + ("  *** GT WILL BE DROPPED — raise num_queries ***" if max_boxes > q else "  (ok)"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=deepforest_collate_fn, num_workers=num_workers, pin_memory=True,
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{out_prefix}_best.pt"

    # Per-epoch artifacts: metrics.csv (train+val each epoch), preds/ (stitched
    # per-tile predictions JSON), panels/ (GT-vs-pred PNGs).
    import csv as _csv
    log_path = Path(log_dir) if log_dir else ckpt_dir
    log_path.mkdir(parents=True, exist_ok=True)
    preds_dir = log_path / "preds"; preds_dir.mkdir(exist_ok=True)
    panels_dir = log_path / "panels"; panels_dir.mkdir(exist_ok=True)
    csv_fields = ["epoch", "train_loss", "val_stitch_loss", "val_f1",
                  "val_f1_bestth", "val_best_conf", "val_count_mae", "val_map50",
                  "best_val_f1", "lr", "encoder_unfrozen", "patience_counter"]
    # Confidence thresholds swept on val each epoch to pick the F1 operating point.
    sweep_ths = [round(0.05 * i, 2) for i in range(1, 13)]   # 0.05 .. 0.60
    csv_file = open(log_path / "metrics.csv", "w", newline="")
    csv_writer = _csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader(); csv_file.flush()

    stitch_kw = dict(device=device, batch_size=batch_size, num_workers=num_workers,
                     conf_thresh=conf_thresh, nms_iou=nms_iou,
                     lam_cls=lam_cls, lam_l1=lam_l1, lam_giou=lam_giou)

    best_val_f1 = 0.0          # tracks the best swept-threshold val F1
    best_conf = conf_thresh    # val-chosen operating threshold of the best checkpoint
    patience_counter = 0

    # Phase 1: frozen encoder (neck+head only).
    optimizer, scheduler = build_stage_b_optimizer_and_scheduler(
        model, neck_head_lr, weight_decay, frozen_epochs, warmup_epochs
    )

    for epoch in range(total_epochs):
        if epoch == frozen_epochs:
            print(f"[Stage C-q] Epoch {epoch+1}: unfreezing last {unfreeze_n_blocks} encoder blocks")
            model.unfreeze_last_n_encoder_blocks(unfreeze_n_blocks)
            optimizer, scheduler = build_stage_c_optimizer_and_scheduler(
                model, neck_head_lr, encoder_lr, weight_decay,
                total_epochs - frozen_epochs, warmup_epochs,
            )

        model.train()
        if epoch < frozen_epochs:
            model.encoder.encoder.eval()
            for p in model.encoder.encoder.parameters():
                p.requires_grad = False

        train_loss = _run_epoch(model, train_loader, optimizer, device,
                                lam_cls, lam_l1, lam_giou, train=True)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Stitched parent-256 validation each epoch. Request the per-tile preds
        # when this epoch will dump JSON and/or render panels.
        # Stitched parent-256 validation each epoch. Always keep the per-tile preds
        # so we can sweep the confidence threshold (this model's best F1 sits well
        # below 0.5); dump/panels reuse the same preds.
        dump = save_preds_every and (epoch + 1) % save_preds_every == 0
        panel = panels_every and (epoch + 1) % panels_every == 0
        vp = stitch_eval(model, val_ds, return_preds=True, **stitch_kw)
        val_f1 = vp["f1_at_0.5"]                          # fixed 0.5 (reference)
        val_best_conf, val_f1_bestth = _best_threshold(vp, sweep_ths)
        val_loss = vp["stitch_loss"]
        print(
            f"[Stage C-q] Epoch {epoch+1}/{total_epochs}  train_loss={train_loss:.4f}  "
            f"val_stitch_loss={val_loss:.4f}  val_f1@0.5={val_f1:.4f}  "
            f"val_f1@{val_best_conf:.2f}={val_f1_bestth:.4f}  "
            f"val_count_mae={vp['count']['mae']:.2f}  lr={lr:.2e}"
        )

        # Select the best checkpoint by the swept-threshold val F1 (the honest
        # operating point), and remember the threshold that achieved it.
        if val_f1_bestth > best_val_f1:
            best_val_f1 = val_f1_bestth
            best_conf = val_best_conf
            patience_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_f1": val_f1_bestth, "conf_thresh": val_best_conf}, best_path)
            if on_checkpoint is not None:
                on_checkpoint()
        else:
            patience_counter += 1

        _wandb_log({
            "stage_c_q/epoch": epoch + 1,
            "stage_c_q/train_loss": train_loss,
            "stage_c_q/val_stitch_loss": val_loss,
            "stage_c_q/val_f1": val_f1,
            "stage_c_q/val_f1_bestth": val_f1_bestth,
            "stage_c_q/val_best_conf": val_best_conf,
            "stage_c_q/val_count_mae": vp["count"]["mae"],
            "stage_c_q/val_map50": vp["map"]["mAP50"],
            "stage_c_q/best_val_f1": best_val_f1,
            "stage_c_q/lr": lr,
            "stage_c_q/encoder_unfrozen": int(epoch >= frozen_epochs),
            "stage_c_q/patience_counter": patience_counter,
        })

        csv_writer.writerow({
            "epoch": epoch + 1, "train_loss": round(train_loss, 6),
            "val_stitch_loss": round(val_loss, 6), "val_f1": round(val_f1, 6),
            "val_f1_bestth": round(val_f1_bestth, 6), "val_best_conf": val_best_conf,
            "val_count_mae": round(vp["count"]["mae"], 4),
            "val_map50": round(vp["map"]["mAP50"], 6),
            "best_val_f1": round(best_val_f1, 6), "lr": lr,
            "encoder_unfrozen": int(epoch >= frozen_epochs),
            "patience_counter": patience_counter,
        })
        csv_file.flush()
        if dump:
            _dump_stitch_preds(vp, epoch + 1, "val", conf_thresh, preds_dir)
        if panel:
            _save_stitch_panels(vp, epoch + 1, "val", panel_conf_thresh, panels_dir, n_panel_tiles)

        if patience_counter >= patience:
            print(f"[Stage C-q] Early stopping at epoch {epoch+1}")
            break

    csv_file.close()
    torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_dir / f"{out_prefix}_final.pt")
    if on_checkpoint is not None:
        on_checkpoint()
    print(f"[Stage C-q] Done. Best stitched val F1: {best_val_f1:.4f}")

    # Final test: reload the best checkpoint, lock the operating threshold on VAL,
    # then score the held-out test set at that threshold (no peeking at test).
    if test_ds is not None:
        if best_path.exists():
            ckpt_best = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt_best["model"])
            best_conf = ckpt_best.get("conf_thresh", best_conf)
            print(f"[Stage C-q] Reloaded best checkpoint ({best_path.name}) for test.")
        # Re-sweep on val with the exact checkpoint being tested.
        vb = stitch_eval(model, val_ds, return_preds=True, **stitch_kw)
        best_conf, val_best_f1 = _best_threshold(vb, sweep_ths)
        tp = stitch_eval(model, test_ds, return_preds=True, **stitch_kw)
        test_f1_05 = tp["f1_at_0.5"]
        test_f1_best = compute_f1_at_iou(tp["pred_boxes"], tp["pred_scores"],
                                         tp["gt_boxes"], 0.5, best_conf)
        print("[Stage C-q] TEST (stitched, parent-256):")
        print(f"    F1@0.5                 = {test_f1_05:.4f}")
        print(f"    F1@val-best({best_conf:.2f})       = {test_f1_best:.4f}   "
              f"(val F1 at that thresh = {val_best_f1:.4f})")
        print(f"    mAP50       = {tp['map']['mAP50']:.4f}")
        print(f"    mAP50-95    = {tp['map']['mAP50_95']:.4f}")
        print(f"    count MAE   = {tp['count']['mae']:.2f}  RMSE={tp['count']['rmse']:.2f}  R2={tp['count']['r2']:.3f}")
        print(f"    stitch_loss = {tp['stitch_loss']:.4f}  over {tp['n_parents']} tiles")
        _dump_stitch_preds(tp, "test", "test", best_conf, preds_dir)
        _save_stitch_panels(tp, "test", "test", panel_conf_thresh, panels_dir, n_panel_tiles)
        import json
        test_summary = {
            "f1_at_0.5": test_f1_05,
            "f1_at_val_best": test_f1_best,
            "val_best_conf": best_conf,
            "val_f1_at_best_conf": val_best_f1,
            "map50": tp["map"]["mAP50"], "map50_95": tp["map"]["mAP50_95"],
            "count": tp["count"], "stitch_loss": tp["stitch_loss"],
            "n_tiles": tp["n_parents"], "best_val_f1": best_val_f1,
        }
        with open(log_path / "test_metrics.json", "w") as f:
            json.dump(test_summary, f, indent=2, default=float)
        _wandb_log({
            "stage_c_q/test_f1": test_f1_05,
            "stage_c_q/test_f1_bestth": test_f1_best,
            "stage_c_q/test_best_conf": best_conf,
            "stage_c_q/test_map50": tp["map"]["mAP50"],
            "stage_c_q/test_map50_95": tp["map"]["mAP50_95"],
            "stage_c_q/test_count_mae": tp["count"]["mae"],
        })
        print(f"[Stage C-q] Artifacts in {log_path}: metrics.csv, preds/, panels/, test_metrics.json")
        return {"best_val_f1": best_val_f1, "test": tp,
                "test_f1_bestth": test_f1_best, "best_conf": best_conf}
    print(f"[Stage C-q] Artifacts in {log_path}: metrics.csv, preds/, panels/")
    return {"best_val_f1": best_val_f1, "test": None}


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
