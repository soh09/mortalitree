"""NAIP tile dataset: loads 4-band GeoTIFFs, extracts metadata, applies augmentation."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import NAIPAugmentation
from .clay_meta import encode_latlon, encode_time

# NAIP wavelengths (μm) in Clay's band order: R, G, B, NIR
# (from configs/metadata.yaml -> naip.band_order = [red, green, blue, nir]).
NAIP_WAVELENGTHS = torch.tensor([0.665, 0.560, 0.493, 0.842])
NAIP_GSD = 1.0  # meters/pixel — Clay's NAIP pretraining GSD (configs/metadata.yaml)


def load_naip_normalization_stats(yaml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-band NAIP mean/std (R, G, B, NIR order).

    Supports two layouts:
      1. Clay's own configs/metadata.yaml — top-level ``naip:`` with
         ``band_order`` and ``bands.mean``/``bands.std`` as band-keyed dicts.
      2. The project's flat ``configs/naip_normalization.yaml`` — ``metadata.naip.mean``
         / ``.std`` as plain lists already in R, G, B, NIR order.
    """
    import yaml
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    # Layout 1: real Clay metadata.yaml
    if "naip" in meta and "bands" in meta["naip"]:
        naip = meta["naip"]
        order = naip["band_order"]                 # e.g. [red, green, blue, nir]
        mean = np.array([naip["bands"]["mean"][b] for b in order], dtype=np.float32)
        std  = np.array([naip["bands"]["std"][b]  for b in order], dtype=np.float32)
        return mean, std

    # Layout 2: project flat config
    naip = meta["metadata"]["naip"]
    mean = np.array(naip["mean"], dtype=np.float32)
    std  = np.array(naip["std"],  dtype=np.float32)
    return mean, std


class NAIPTileDataset(Dataset):
    """
    Each item in the annotations JSON:
    {
        "tile_path": "path/to/tile.tif",
        "boxes": [[cx, cy, w, h], ...],   # normalized [0,1]
        "exhaustive": true,
        "lat": float, "lon": float,
        "acquisition_date": "YYYY-MM-DD",
        "hour_of_day": float (optional, default 12.0)
    }
    """

    def __init__(
        self,
        annotations_path: str,
        norm_stats_path: Optional[str] = None,
        augment: bool = True,
        tile_size: int = 256,
    ):
        with open(annotations_path) as f:
            self.items = json.load(f)

        self.tile_size = tile_size
        self.augment = NAIPAugmentation() if augment else None

        if norm_stats_path is not None:
            self.mean, self.std = load_naip_normalization_stats(norm_stats_path)
        else:
            # Fallback: Clay's NAIP DN-scale stats (0-255 range), R, G, B, NIR order.
            self.mean = np.array([110.16, 115.41, 98.15, 139.04], dtype=np.float32)
            self.std  = np.array([47.23, 39.82, 35.43, 49.86], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        pixels = self._load_tile(item["tile_path"])   # (4, H, W) float32
        boxes = torch.tensor(item.get("boxes", []), dtype=torch.float32)
        if boxes.ndim == 1 and boxes.numel() > 0:
            boxes = boxes.unsqueeze(0)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if self.augment is not None:
            pixels, boxes = self.augment(pixels, boxes)

        # Normalize
        mean = torch.tensor(self.mean, dtype=torch.float32).view(4, 1, 1)
        std  = torch.tensor(self.std,  dtype=torch.float32).view(4, 1, 1)
        pixels = (pixels - mean) / std

        # Metadata
        lat = float(item["lat"])
        lon = float(item["lon"])
        date_str = item.get("acquisition_date", "2020-06-01")
        hour = float(item.get("hour_of_day", 12.0))
        week = self._week_of_year(date_str)

        return {
            "pixels":     pixels,
            "wavelengths": NAIP_WAVELENGTHS,
            "gsd":        torch.tensor([NAIP_GSD], dtype=torch.float32),
            "time":       encode_time(week, hour),
            "latlon":     encode_latlon(lat, lon),
            "boxes":      boxes,
            "exhaustive": bool(item.get("exhaustive", True)),
            "tile_path":  item["tile_path"],
        }

    def _load_tile(self, path: str) -> torch.Tensor:
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)  # (C, H, W)
        except ImportError:
            # Fallback for environments without rasterio (testing)
            data = np.zeros((4, self.tile_size, self.tile_size), dtype=np.float32)
        return torch.from_numpy(data)

    @staticmethod
    def _week_of_year(date_str: str) -> int:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.isocalendar()[1]


def naip_collate_fn(batch: list) -> dict:
    """Collate a list of NAIPTileDataset items into a batch dict."""
    pixels     = torch.stack([b["pixels"] for b in batch])
    wavelengths = batch[0]["wavelengths"]
    gsd        = torch.cat([b["gsd"] for b in batch])
    time       = torch.stack([b["time"] for b in batch])
    latlon     = torch.stack([b["latlon"] for b in batch])
    boxes      = [b["boxes"] for b in batch]
    exhaustive = [b["exhaustive"] for b in batch]
    tile_paths = [b["tile_path"] for b in batch]
    return {
        "pixels":     pixels,
        "wavelengths": wavelengths,
        "gsd":        gsd,
        "time":       time,
        "latlon":     latlon,
        "boxes":      boxes,
        "exhaustive": exhaustive,
        "tile_paths": tile_paths,
    }
