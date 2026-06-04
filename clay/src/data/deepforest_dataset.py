"""Stage B RGB tree dataset wrapper. 3 RGB bands + 3 wavelengths passed to Clay."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import NAIPAugmentation

# RGB wavelengths only (μm), in R, G, B order (matches GeoTIFF bands 1,2,3).
RGB_WAVELENGTHS = torch.tensor([0.665, 0.560, 0.493])
# Only 3 bands + 3 wavelengths are passed to Clay's dynamic embedding, which
# handles a variable band count natively (no zero NIR channel needed).
NAIP_GSD = 1.0  # match Clay's NAIP pretraining GSD (configs/metadata.yaml)


class DeepForestDataset(Dataset):
    """
    Wrapper for DeepForest NEON crown dataset (RGB only).

    Annotations JSON format (same structure as NAIP dataset):
    {
        "tile_path": "path/to/rgb_tile.tif",
        "boxes": [[cx, cy, w, h], ...],
        "exhaustive": true,
        "lat": float, "lon": float,
        "acquisition_date": "YYYY-MM-DD"
    }

    Only the 3 RGB bands + 3 wavelengths are passed to Clay's dynamic embedding,
    which handles a variable band count natively (no zero NIR channel needed).
    """

    def __init__(
        self,
        annotations_path: str,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        augment: bool = True,
        tile_size: int = 256,
    ):
        with open(annotations_path) as f:
            self.items = json.load(f)
        self.tile_size = tile_size
        self.augment = NAIPAugmentation() if augment else None
        # Default to Clay's NAIP R,G,B DN-scale stats so Stage B inputs land in the
        # same normalized domain as Stage C (RGB tiles are read as 0-255 DN).
        self.mean = norm_mean if norm_mean is not None else np.array([110.16, 115.41, 98.15], dtype=np.float32)
        self.std  = norm_std  if norm_std  is not None else np.array([47.23, 39.82, 35.43], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        pixels = self._load_rgb_tile(item["tile_path"])   # (3, H, W)
        boxes = torch.tensor(item.get("boxes", []), dtype=torch.float32)
        if boxes.ndim == 1 and boxes.numel() > 0:
            boxes = boxes.unsqueeze(0)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if self.augment is not None:
            pixels, boxes = self.augment(pixels, boxes)

        mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor(self.std,  dtype=torch.float32).view(3, 1, 1)
        pixels = (pixels - mean) / std

        lat = float(item.get("lat", 37.0))
        lon = float(item.get("lon", -119.0))
        date_str = item.get("acquisition_date", "2020-06-01")
        hour = float(item.get("hour_of_day", 12.0))
        week = self._week_of_year(date_str)

        return {
            "pixels":      pixels,
            "wavelengths": RGB_WAVELENGTHS,
            "gsd":         torch.tensor([NAIP_GSD], dtype=torch.float32),
            "time":        torch.tensor([week / 52.0, hour / 24.0], dtype=torch.float32),
            "latlon":      torch.tensor([lat, lon], dtype=torch.float32),
            "boxes":       boxes,
            "exhaustive":  bool(item.get("exhaustive", True)),
            "tile_path":   item["tile_path"],
        }

    def _load_rgb_tile(self, path: str) -> torch.Tensor:
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read(indexes=[1, 2, 3]).astype(np.float32)
        except ImportError:
            data = np.zeros((3, self.tile_size, self.tile_size), dtype=np.float32)
        return torch.from_numpy(data)

    @staticmethod
    def _week_of_year(date_str: str) -> int:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.isocalendar()[1]


def deepforest_collate_fn(batch: list) -> dict:
    pixels     = torch.stack([b["pixels"] for b in batch])
    wavelengths = batch[0]["wavelengths"]
    gsd        = torch.cat([b["gsd"] for b in batch])
    time       = torch.stack([b["time"] for b in batch])
    latlon     = torch.stack([b["latlon"] for b in batch])
    boxes      = [b["boxes"] for b in batch]
    exhaustive = [b["exhaustive"] for b in batch]
    return {
        "pixels":      pixels,
        "wavelengths": wavelengths,
        "gsd":         gsd,
        "time":        time,
        "latlon":      latlon,
        "boxes":       boxes,
        "exhaustive":  exhaustive,
    }
