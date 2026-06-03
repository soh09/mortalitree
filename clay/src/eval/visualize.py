"""Qualitative diagnostic plots for tree detection results."""
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def tile_to_rgb(pixels: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) normalized tile to an (H, W, 3) uint8 RGB for display.
    Clay's NAIP channel order is R,G,B,NIR; channels 0,1,2 are already RGB.
    """
    img = pixels.cpu().numpy()
    # Undo approximate normalization and clip to [0, 1]
    rgb = img[[0, 1, 2], ...]   # R=ch0, G=ch1, B=ch2
    rgb = (rgb - rgb.min()) / max(1e-6, rgb.max() - rgb.min())
    return (np.clip(rgb, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    color: tuple = (0, 255, 0),
    gt_boxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw predicted and GT boxes on an RGB image (H, W, 3)."""
    try:
        import cv2
    except ImportError:
        return image

    H, W = image.shape[:2]
    img = image.copy()

    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # blue for GT

    for idx, box in enumerate(boxes):
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if scores is not None:
            label = f"{scores[idx]:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return img


def visualize_predictions(
    model,
    dataset,
    output_dir: str,
    n_tiles: int = 10,
    conf_thresh: float = 0.5,
    device: str = "cpu",
):
    """Save diagnostic prediction images for n_tiles from the dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from ..data.naip_dataset import naip_collate_fn

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=naip_collate_fn, num_workers=0)
    model.eval()

    for i, batch in enumerate(loader):
        if i >= n_tiles:
            break
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in ("boxes", "exhaustive", "tile_paths")
        }
        with torch.no_grad():
            cls_logits, pred_boxes = model(batch_gpu)

        scores = cls_logits[0].sigmoid().cpu()
        keep = scores >= conf_thresh
        pb = pred_boxes[0].cpu()[keep].numpy()
        ps = scores[keep].numpy()
        gt_b = batch["boxes"][0].numpy() if batch["boxes"][0].numel() > 0 else np.zeros((0, 4))

        rgb = tile_to_rgb(batch["pixels"][0])
        img = draw_boxes(rgb, pb, ps, color=(0, 255, 0), gt_boxes=gt_b)

        n_pred, n_gt = len(pb), len(gt_b)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.set_title(f"Tile {i}: pred={n_pred}  gt={n_gt}", fontsize=9)
        ax.axis("off")
        fig.savefig(out_dir / f"tile_{i:04d}.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

    print(f"[visualize] Saved {min(n_tiles, i+1)} diagnostic tiles to {out_dir}")


def plot_precision_recall(prec: np.ndarray, rec: np.ndarray, ap: float, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.step(rec, prec, where="post", linewidth=1.5)
    ax.fill_between(rec, prec, alpha=0.2, step="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall  (AP={ap:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
