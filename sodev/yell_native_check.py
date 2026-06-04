"""One center 153.6 m x 153.6 m patch rendered two ways:
  - native (0.1 m GSD, 1536 x 1536 px) — what the source actually has
  - downsampled (0.6 m GSD, 256 x 256 px) — what Stage B sees
If the native one shows fine detail and the downsampled one is a blob, the
6x downsample is doing its job. If the native one is *also* blurry, the
source data has a problem."""
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

PATCH_M = 153.6
NATIVE_GSD = 0.1
TARGET_GSD = 0.6
NATIVE_PX = int(PATCH_M / NATIVE_GSD)   # 1536
TARGET_PX = int(PATCH_M / TARGET_GSD)   # 256

with rasterio.open(TIF) as src:
    cx_utm = (src.bounds.left + src.bounds.right) / 2
    cy_utm = (src.bounds.top  + src.bounds.bottom) / 2
    patch_left = cx_utm - PATCH_M / 2
    patch_top  = cy_utm + PATCH_M / 2
    patch_right  = patch_left + PATCH_M
    patch_bottom = patch_top  - PATCH_M

    cl, rt = (~src.transform) * (patch_left, patch_top)
    cr, rb = (~src.transform) * (patch_right, patch_bottom)
    col0, col1 = int(min(cl, cr)), int(max(cl, cr))
    row0, row1 = int(min(rt, rb)), int(max(rt, rb))
    win = Window(col0, row0, col1 - col0, row1 - row0)

    native = src.read([1, 2, 3], window=win)   # ~1536 x 1536
    down   = src.read([1, 2, 3], window=win,
                      out_shape=(3, TARGET_PX, TARGET_PX),
                      resampling=Resampling.average)
print(f"native window: {native.shape[1]} x {native.shape[2]} px")
print(f"downsampled:   {down.shape[1]} x {down.shape[2]} px")

def to_disp(a):
    a = np.transpose(a, (1, 2, 0))
    nz = a[a > 0]
    lo, hi = (np.percentile(nz, (2, 98)) if nz.size else (0, 1))
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(to_disp(native))
axes[0].set_title(
    f"native @ 0.1 m\n{native.shape[1]} x {native.shape[2]} px = 153.6 m",
    fontsize=11,
)
axes[1].imshow(to_disp(down))
axes[1].set_title(
    f"Stage-B input @ 0.6 m\n{down.shape[1]} x {down.shape[2]} px = 153.6 m",
    fontsize=11,
)
for ax in axes:
    ax.axis("off")
fig.suptitle(
    f"{TIF.name} — same 153.6 m patch, two scales",
    fontsize=12,
)
plt.tight_layout()
out_png = OUT_DIR / "native_vs_down.png"
plt.savefig(out_png, dpi=160, bbox_inches="tight")
print(f"wrote {out_png}")
