"""
Copy-Paste Data Augmentation for DeepForest tree detection.

Usage:
    python splat_augment.py \
        --background castle_fire_CIR_highlight_new.tif \
        --templates tree1.png tree2.png tree3.png tree4.png tree5.png \
        --output_dir output \
        --n_images 50 \
        --trees_per_image 30

Outputs:
    output/img_0000.png ... img_00NN.png
    output/annotations.csv  (DeepForest format)
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def random_transform(template: Image.Image, scale_range=(0.85, 1.15)) -> Image.Image:
    """Return a randomly rotated, flipped, and scaled copy of *template*."""
    img = template.copy()

    # Random flips
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Random rotation — expand=True keeps the whole crown visible after rotation
    angle = random.uniform(0, 360)
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Random scale
    scale = random.uniform(*scale_range)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def tight_bbox(alpha: np.ndarray, ox: int, oy: int):
    """
    Compute the tight bounding box of non-transparent pixels in *alpha*,
    offset by (ox, oy) into the canvas coordinate system.

    Returns (xmin, ymin, xmax, ymax) or None if the alpha is all zero.
    """
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (ox + int(cmin), oy + int(rmin), ox + int(cmax), oy + int(rmax))


# ---------------------------------------------------------------------------
# Overlap guard (IoU-based)
# ---------------------------------------------------------------------------

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def find_placement(
    tree_w: int,
    tree_h: int,
    canvas_w: int,
    canvas_h: int,
    placed_boxes: list,
    max_iou: float = 0.15,
    max_tries: int = 30,
):
    """
    Try up to *max_tries* random positions; return the first one whose IoU
    with every already-placed box is below *max_iou*.  Falls back to the
    last attempted position if no clear spot is found.
    """
    for _ in range(max_tries):
        x = random.randint(0, max(0, canvas_w - tree_w))
        y = random.randint(0, max(0, canvas_h - tree_h))
        candidate = (x, y, x + tree_w, y + tree_h)
        if all(iou(candidate, b) < max_iou for b in placed_boxes):
            return x, y
    # fallback — accept overlap rather than skip the tree entirely
    x = random.randint(0, max(0, canvas_w - tree_w))
    y = random.randint(0, max(0, canvas_h - tree_h))
    return x, y


# ---------------------------------------------------------------------------
# Core splatting routine
# ---------------------------------------------------------------------------

def splat_image(
    background: Image.Image,
    templates: list,
    n_trees: int,
    scale_range=(0.85, 1.15),
    max_iou: float = 0.15,
):
    """
    Paste *n_trees* randomly transformed templates onto a copy of *background*.

    Returns:
        canvas  – PIL Image (RGB)
        boxes   – list of (xmin, ymin, xmax, ymax) in canvas coords
    """
    canvas = background.convert("RGBA").copy()
    cw, ch = canvas.size
    boxes = []

    for _ in range(n_trees):
        tmpl = random.choice(templates)
        tree = random_transform(tmpl, scale_range)

        tw, th = tree.size
        if tw > cw or th > ch:
            # Template larger than canvas after scaling — skip
            continue

        x, y = find_placement(tw, th, cw, ch, boxes, max_iou)

        # Alpha-blend tree onto canvas
        canvas.paste(tree, (x, y), mask=tree.split()[3])

        alpha_arr = np.array(tree.split()[3])
        bbox = tight_bbox(alpha_arr, x, y)
        if bbox:
            boxes.append(bbox)

    return canvas.convert("RGB"), boxes


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def load_templates(paths: list) -> list:
    templates = []
    for p in paths:
        img = Image.open(p).convert("RGBA")
        templates.append(img)
    if not templates:
        raise ValueError("No template images loaded.")
    return templates


def load_background(path: str) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img


def generate_batch(
    background_path: str,
    template_paths: list,
    output_dir: str,
    n_images: int = 50,
    trees_per_image: int = 30,
    scale_range=(0.85, 1.15),
    max_iou: float = 0.15,
):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    background = load_background(background_path)
    templates = load_templates(template_paths)

    print(f"Background size : {background.size}")
    print(f"Templates loaded: {len(templates)}")
    print(f"Generating {n_images} images × {trees_per_image} trees each")
    print(f"Output dir      : {output_dir}")
    print()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

        for i in range(n_images):
            img_name = f"img_{i:04d}.png"
            img_path = os.path.join(output_dir, img_name)

            canvas, boxes = splat_image(
                background,
                templates,
                n_trees=trees_per_image,
                scale_range=scale_range,
                max_iou=max_iou,
            )
            canvas.save(img_path)

            for xmin, ymin, xmax, ymax in boxes:
                writer.writerow([img_name, xmin, ymin, xmax, ymax, "Tree"])

            placed = len(boxes)
            print(f"  [{i+1:>4}/{n_images}] {img_name}  trees placed: {placed}")

    print(f"\nDone. Annotations saved to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Copy-Paste tree augmentation for DeepForest")
    parser.add_argument("--background", required=True, help="Path to background image (GeoTIFF or PNG)")
    parser.add_argument("--templates", nargs="+", required=True, help="Paths to tree crown PNGs (with alpha)")
    parser.add_argument("--output_dir", default="output", help="Directory for synthetic images + CSV")
    parser.add_argument("--n_images", type=int, default=50, help="Number of synthetic images to generate")
    parser.add_argument("--trees_per_image", type=int, default=30, help="Trees to splat per image")
    parser.add_argument("--scale_min", type=float, default=0.85, help="Minimum scale factor (default 0.85)")
    parser.add_argument("--scale_max", type=float, default=1.15, help="Maximum scale factor (default 1.15)")
    parser.add_argument("--max_iou", type=float, default=0.15, help="Max allowed overlap IoU between trees")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    generate_batch(
        background_path=args.background,
        template_paths=args.templates,
        output_dir=args.output_dir,
        n_images=args.n_images,
        trees_per_image=args.trees_per_image,
        scale_range=(args.scale_min, args.scale_max),
        max_iou=args.max_iou,
    )


if __name__ == "__main__":
    main()
