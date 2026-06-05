"""Stage B NEON 4-band patch dataset.

Reads the patches + labels produced by `dataset_dev/modal_pipeline.py`:

    patches_root/{SITE}/{imgname}.tif        # 4-band R,G,B,NIR uint8, 256x256
    labels_csv                               # one row per box

labels.csv columns (see modal_pipeline.py):
    site, imgname, xmin, ymin, xmax, ymax, score, ndvi, lat, lon,
    acquisition_yyyymm, patch_left_utm, patch_top_utm, source_geo_index

Boxes in the CSV are pixel coords in [0, source_size]; here they are converted
to the model's internal format — normalized (cx, cy, w, h) in [0, 1] (spec §3.1) —
so the returned dict is shape-compatible with `deepforest_collate_fn`. All 4 NAIP
bands + 4 wavelengths are passed to Clay's dynamic embedding, matching Stage C.

Quartering (`quarter=True`): each 256-px source patch is split into a grid of
smaller tiles (default 2x2 → 128 px). A box is assigned to the single tile that
contains its *center*, then clipped to the tile — so a crown straddling an
internal seam is owned by exactly one tile (no duplication), and the model learns
to fire only when a crown's center is in-tile. This keeps trees-per-tile (and
thus the required query budget Q) low while recovering the dense-forest patches
that a per-256 `max_trees` filter would otherwise drop. Each returned sample
carries `parent`/`quarter` so predictions can be stitched back to the 256 frame
for parent-level metrics (head-to-head with the un-quartered baseline).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import NAIPAugmentation
from .clay_meta import encode_latlon, encode_time

# NAIP 4-band wavelengths (μm), R, G, B, NIR — matches GeoTIFF band order 1..4
# and Clay's naip.band_order (spec §3.1).
NAIP_WAVELENGTHS = torch.tensor([0.665, 0.560, 0.493, 0.842])
# Clay's NAIP per-band DN-scale stats, R, G, B, NIR (spec §3.1 / metadata.yaml).
NAIP_MEAN = np.array([110.16, 115.41, 98.15, 139.04], dtype=np.float32)
NAIP_STD = np.array([47.23, 39.82, 35.43, 49.86], dtype=np.float32)
NAIP_GSD = 1.0  # match Clay's NAIP pretraining GSD


def _boxes_to_norm_cxcywh(xyxy: np.ndarray, off_x: float, off_y: float,
                          out_size: int) -> np.ndarray:
    """Shift pixel-xyxy boxes by a crop offset, clip to the [0, out_size] tile,
    drop sub-pixel boxes, and return normalized (cx, cy, w, h) in [0, 1]."""
    if len(xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1 = np.clip(xyxy[:, 0] - off_x, 0, out_size)
    y1 = np.clip(xyxy[:, 1] - off_y, 0, out_size)
    x2 = np.clip(xyxy[:, 2] - off_x, 0, out_size)
    y2 = np.clip(xyxy[:, 3] - off_y, 0, out_size)
    keep = (x2 - x1 >= 1.0) & (y2 - y1 >= 1.0)
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    cx = (x1 + x2) / 2.0 / out_size
    cy = (y1 + y2) / 2.0 / out_size
    w = (x2 - x1) / out_size
    h = (y2 - y1) / out_size
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


class NeonPatchDataset(Dataset):
    """One sample per (quarter of a) patch; all of a tile's boxes grouped.

    Two ways to get 128-px quarters:
      * pre-quartered (recommended) — point `labels_csv` at the pipeline's
        `labels_quarter.csv` (128 tifs on disk, one row group per quarter) and
        `parent_labels_csv` at `labels_parent.csv` (full 256 GT). Set by passing
        `parent_labels_csv`.
      * on-the-fly — point `labels_csv` at a 256 `labels.csv` and pass
        `quarter=True`; each 256 tif is split at read time. Kept for the
        sparse-data control; the pre-quartered path is what the pipeline produces.

    Each sample carries `read_crop` (window into the on-disk tif) and a
    `stitch` offset (where the tile sits in its 256 parent) so predictions can be
    stitched back to the parent frame (see eval/stitch.py). `source_size` is the
    parent frame size used for that stitching.
    """

    def __init__(
        self,
        labels_csv: str,
        patches_root: str,
        tile_size: int = 256,          # tile size ON DISK (256 baseline / 128 quarter)
        augment: bool = True,
        quarter: bool = False,         # on-the-fly: split each 256 tif into a grid
        n_split: int = 2,              # grid per side for on-the-fly quartering
        max_trees: int = 0,            # drop tiles with > this many boxes (0 = keep all)
        parent_labels_csv: Optional[str] = None,  # set -> pre-quartered mode
        parent_size: int = 256,        # parent frame size (pre-quartered stitching)
        packed_dir: Optional[str] = None,  # set -> read pixels from per-site memmaps
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ):
        import pandas as pd

        self.patches_root = Path(patches_root)
        self.augment = NAIPAugmentation() if augment else None
        self.mean = norm_mean if norm_mean is not None else NAIP_MEAN
        self.std = norm_std if norm_std is not None else NAIP_STD
        self.samples: list[dict] = []
        # parent_gt: FULL 256-frame GT per parent (normalized cxcywh), used to
        # score stitched quarter predictions against the baseline's ground truth.
        self.parent_gt: dict[str, np.ndarray] = {}
        # Packed mode: read pixels from {packed_dir}/{SITE}.npy memmaps (via
        # packed_index.csv) instead of opening one GeoTIFF per tile.
        self.packed = packed_dir is not None
        self.packed_dir = Path(packed_dir) if packed_dir else None
        self._mmaps: dict = {}
        self._packed_loc: dict = {}
        if self.packed:
            pidx = pd.read_csv(self.packed_dir / "packed_index.csv")
            self._packed_loc = {
                str(r.imgname): (str(r.site), int(r.row))
                for r in pidx.itertuples(index=False)
            }

        if parent_labels_csv is not None:
            self._init_prequartered(pd, labels_csv, parent_labels_csv,
                                    tile_size, parent_size, max_trees)
        else:
            self._init_onthefly(pd, labels_csv, tile_size, quarter, n_split, max_trees)

    def _init_onthefly(self, pd, labels_csv, tile_size, quarter, n_split, max_trees):
        self.source_size = tile_size
        n = n_split if quarter else 1
        self.out_size = tile_size // n
        if quarter and tile_size % n != 0:
            raise ValueError(f"tile_size {tile_size} not divisible by n_split {n}")

        df = pd.read_csv(labels_csv)
        xyxy_all = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        cx_all = (xyxy_all[:, 0] + xyxy_all[:, 2]) / 2.0
        cy_all = (xyxy_all[:, 1] + xyxy_all[:, 3]) / 2.0

        for imgname, g in df.groupby("imgname", sort=False):
            r0 = g.iloc[0]
            idx = g.index.to_numpy()
            xyxy = xyxy_all[idx]
            cx, cy = cx_all[idx], cy_all[idx]
            self.parent_gt[imgname] = _boxes_to_norm_cxcywh(xyxy, 0, 0, tile_size)
            meta = self._meta(r0, imgname)
            for gr in range(n):
                for gc in range(n):
                    off_x, off_y = gc * self.out_size, gr * self.out_size
                    if quarter:
                        in_cell = (
                            (cx >= off_x) & (cx < off_x + self.out_size) &
                            (cy >= off_y) & (cy < off_y + self.out_size)
                        )
                        cell_xyxy = xyxy[in_cell]
                    else:
                        cell_xyxy = xyxy
                    boxes = _boxes_to_norm_cxcywh(cell_xyxy, off_x, off_y, self.out_size)
                    if len(boxes) == 0 or (max_trees and len(boxes) > max_trees):
                        continue
                    self.samples.append({
                        **meta, "boxes": boxes,
                        "read_crop": (off_x, off_y, self.out_size),  # window into the 256 tif
                        "stitch": (off_x, off_y, self.out_size),     # offset in the parent
                        "parent": imgname,
                    })

    def _init_prequartered(self, pd, labels_csv, parent_labels_csv,
                           tile_size, parent_size, max_trees):
        self.source_size = parent_size       # stitch predictions back to this frame
        self.out_size = tile_size            # quarter tif size on disk (e.g. 128)

        df = pd.read_csv(labels_csv)
        xyxy_all = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        for imgname, g in df.groupby("imgname", sort=False):
            r0 = g.iloc[0]
            xyxy = xyxy_all[g.index.to_numpy()]
            boxes = _boxes_to_norm_cxcywh(xyxy, 0, 0, tile_size)   # already 128-frame
            if len(boxes) == 0 or (max_trees and len(boxes) > max_trees):
                continue
            qr, qc = int(r0["q_row"]), int(r0["q_col"])
            self.samples.append({
                **self._meta(r0, imgname), "imgname": str(imgname),
                "boxes": boxes,
                "read_crop": (0, 0, tile_size),                    # the whole 128 tif
                "stitch": (qc * tile_size, qr * tile_size, tile_size),
                "parent": str(r0["parent"]),
            })

        pdf = pd.read_csv(parent_labels_csv)
        pxyxy = pdf[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        for parent, g in pdf.groupby("parent", sort=False):
            self.parent_gt[str(parent)] = _boxes_to_norm_cxcywh(
                pxyxy[g.index.to_numpy()], 0, 0, parent_size)

    def _meta(self, r0, imgname) -> dict:
        has_ex = "exhaustive" in r0.index
        return dict(
            tile_path=str(self.patches_root / str(r0["site"]) / f"{imgname}.tif"),
            lat=float(r0["lat"]), lon=float(r0["lon"]),
            week=self._week_of_yyyymm(str(r0["acquisition_yyyymm"])),
            exhaustive=bool(r0["exhaustive"]) if has_ex else True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        if self.packed:
            pixels = self._load_packed(s["imgname"])             # (4, out, out) raw DN
        else:
            pixels = self._load_tile(s["tile_path"], s["read_crop"])
        boxes = torch.from_numpy(s["boxes"])                     # (N, 4) cxcywh [0,1]
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Augment on raw DN (mirrors DeepForestDataset ordering), then normalize.
        if self.augment is not None:
            pixels, boxes = self.augment(pixels, boxes)

        mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)
        pixels = (pixels - mean) / std

        return {
            "pixels":      pixels,
            "wavelengths": NAIP_WAVELENGTHS,
            "gsd":         torch.tensor([NAIP_GSD], dtype=torch.float32),
            "time":        encode_time(s["week"], 12.0),
            "latlon":      encode_latlon(s["lat"], s["lon"]),
            "boxes":       boxes,
            "exhaustive":  s["exhaustive"],
            "tile_path":   s["tile_path"],
            "parent":      s["parent"],
            "crop":        s["stitch"],     # parent-frame offset for stitching
        }

    @property
    def group_keys(self) -> list:
        """Parent imgname per sample — used for a leakage-free, group-aware
        train/val split (all quarters of a parent stay in one split)."""
        return [s["parent"] for s in self.samples]

    def _load_tile(self, path: str, crop) -> torch.Tensor:
        import rasterio
        from rasterio.windows import Window
        off_x, off_y, size = crop
        with rasterio.open(path) as src:
            data = src.read(
                indexes=[1, 2, 3, 4],
                window=Window(off_x, off_y, size, size),
            ).astype(np.float32)                                 # (4, size, size)
        return torch.from_numpy(data)

    def _load_packed(self, imgname: str) -> torch.Tensor:
        """Read one (4, out, out) tile from the per-site memmap — no GeoTIFF open.
        Memmaps are opened lazily and cached per worker process."""
        site, row = self._packed_loc[imgname]
        mm = self._mmaps.get(site)
        if mm is None:
            mm = np.load(self.packed_dir / f"{site}.npy", mmap_mode="r")
            self._mmaps[site] = mm
        return torch.from_numpy(np.array(mm[row], dtype=np.float32))   # copy out of the mmap

    @staticmethod
    def _week_of_yyyymm(yyyymm: str) -> int:
        """ISO week of the 15th of the acquisition month (robust mid-month proxy)."""
        from datetime import datetime
        for fmt in ("%Y-%m", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(yyyymm[: len(fmt) + 2], fmt)
                return datetime(dt.year, dt.month, 15).isocalendar()[1]
            except ValueError:
                continue
        return 26  # mid-year fallback
