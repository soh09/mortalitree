"""Quick sanity check: open a NEON RGB tile, report its actual GSD,
and dump 6x6 patches at 153.6 m to look at."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

HERE = Path(__file__).parent
TIF = HERE / "2019_YELL_2_527000_4974000_image.tif"
OUT_DIR = HERE / "yell_patches_check"
OUT_DIR.mkdir(exist_ok=True)

TARGET_GSD = 0.6
PATCH_PX = 256
PATCH_M = PATCH_PX * TARGET_GSD   # 153.6 m
SRC_TILE_M = 1000

with rasterio.open(TIF) as src:
    print(f"shape:       {src.height} x {src.width}")
    print(f"bands:       {src.count} ({src.dtypes})")
    print(f"bounds:      {src.bounds}")
    print(f"crs:         {src.crs}")
    print(f"transform:   {src.transform}")
    px_w = abs(src.transform.a)
    px_h = abs(src.transform.e)
    print(f"native GSD:  {px_w:.3f} m/px (W), {px_h:.3f} m/px (H)")
    print(
        f"tile size:   {src.width * px_w:.1f} m W x {src.height * px_h:.1f} m H"
    )

    w, e = src.bounds.left,  src.bounds.right
    s, n = src.bounds.bottom, src.bounds.top
    n_per_side = int(SRC_TILE_M // PATCH_M)   # 6

    fig, axes = plt.subplots(n_per_side, n_per_side, figsize=(18, 18))
    for r in range(n_per_side):
        for c in range(n_per_side):
            patch_left = w + c * PATCH_M
            patch_top  = n - r * PATCH_M
            patch_right  = patch_left + PATCH_M
            patch_bottom = patch_top  - PATCH_M

            cl, rt = (~src.transform) * (patch_left, patch_top)
            cr, rb = (~src.transform) * (patch_right, patch_bottom)
            col0, col1 = int(min(cl, cr)), int(max(cl, cr))
            row0, row1 = int(min(rt, rb)), int(max(rt, rb))
            win = Window(col0, row0, col1 - col0, row1 - row0)
            rgb = src.read(
                [1, 2, 3], window=win,
                out_shape=(3, PATCH_PX, PATCH_PX),
                resampling=Resampling.average,
            )
            rgb = np.transpose(rgb, (1, 2, 0))
            # Per-patch p2/p98 stretch so dark tiles aren't crushed
            nz = rgb[rgb > 0]
            lo, hi = (np.percentile(nz, (2, 98)) if nz.size else (0, 1))
            disp = np.clip((rgb.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

            ax = axes[r, c]
            ax.imshow(disp)
            ax.set_title(
                f"r{r}c{c}  native:{col1 - col0}px",
                fontsize=8,
            )
            ax.axis("off")

    fig.suptitle(
        f"{TIF.name} - 6x6 patches @ 0.6 m GSD (256 px = 153.6 m)\n"
        f"native GSD = {px_w:.2f} m/px → downsample {px_w / TARGET_GSD:.1f}x",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "grid.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")
    plt.show()
