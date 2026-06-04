"""Stage B NEON 4-band patch dataset.

Reads the patches + labels produced by `dataset_dev/modal_pipeline.py`:

    patches_root/{SITE}/{imgname}.tif        # 4-band R,G,B,NIR uint8, 256x256
    labels_csv                               # one row per box

labels.csv columns (see modal_pipeline.py):
    site, imgname, xmin, ymin, xmax, ymax, score, ndvi, lat, lon,
    acquisition_yyyymm, patch_left_utm, patch_top_utm, source_geo_index

Boxes in the CSV are pixel coords in [0, tile_size]; here they are converted to
the model's internal format — normalized (cx, cy, w, h) in [0, 1] (spec §3.1) —
so the returned dict is shape-compatible with `deepforest_collate_fn`. Unlike
the RGB DeepForest wrapper this passes all 4 NAIP bands + 4 wavelengths to Clay's
dynamic embedding, matching the Stage C input distribution.
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


class NeonPatchDataset(Dataset):
    """One sample per patch (GeoTIFF); all of a patch's boxes grouped together."""

    def __init__(
        self,
        labels_csv: str,
        patches_root: str,
        tile_size: int = 256,
        augment: bool = True,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ):
        import pandas as pd

        self.patches_root = Path(patches_root)
        self.tile_size = tile_size
        self.augment = NAIPAugmentation() if augment else None
        self.mean = norm_mean if norm_mean is not None else NAIP_MEAN
        self.std = norm_std if norm_std is not None else NAIP_STD

        df = pd.read_csv(labels_csv)
        has_exhaustive = "exhaustive" in df.columns

        # Normalized cxcywh in [0, 1], computed vectorized over the whole frame.
        ts = float(tile_size)
        cx = (df["xmin"] + df["xmax"]).to_numpy(dtype=np.float32) / 2.0 / ts
        cy = (df["ymin"] + df["ymax"]).to_numpy(dtype=np.float32) / 2.0 / ts
        w = (df["xmax"] - df["xmin"]).to_numpy(dtype=np.float32) / ts
        h = (df["ymax"] - df["ymin"]).to_numpy(dtype=np.float32) / ts
        df = df.assign(_cx=cx, _cy=cy, _w=w, _h=h)

        # Group rows into one entry per patch.
        self.items: list[dict] = []
        for imgname, g in df.groupby("imgname", sort=False):
            r0 = g.iloc[0]
            boxes = np.stack(
                [g["_cx"].to_numpy(), g["_cy"].to_numpy(),
                 g["_w"].to_numpy(), g["_h"].to_numpy()],
                axis=1,
            ).astype(np.float32)
            self.items.append({
                "tile_path": str(self.patches_root / str(r0["site"]) / f"{imgname}.tif"),
                "boxes": boxes,
                "lat": float(r0["lat"]),
                "lon": float(r0["lon"]),
                "week": self._week_of_yyyymm(str(r0["acquisition_yyyymm"])),
                "exhaustive": bool(r0["exhaustive"]) if has_exhaustive else True,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        pixels = self._load_tile(item["tile_path"])              # (4, H, W) raw DN
        boxes = torch.from_numpy(item["boxes"])                  # (N, 4) cxcywh [0,1]
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
            "time":        encode_time(item["week"], 12.0),
            "latlon":      encode_latlon(item["lat"], item["lon"]),
            "boxes":       boxes,
            "exhaustive":  item["exhaustive"],
            "tile_path":   item["tile_path"],
        }

    def _load_tile(self, path: str) -> torch.Tensor:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(indexes=[1, 2, 3, 4]).astype(np.float32)   # (4, H, W)
        return torch.from_numpy(data)

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
