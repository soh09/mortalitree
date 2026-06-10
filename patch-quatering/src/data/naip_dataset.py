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


class QuarteredNAIPDataset(Dataset):
    """Stage-C NAIP JSON dataset, quartered for the 128-px detector.

    Reads the same annotation JSON as NAIPTileDataset — 256-frame tiles whose
    boxes are normalized (cx, cy, w, h) in [0, 1] — but splits every tile into an
    ``n_split`` x ``n_split`` grid of sub-tiles (default 2x2 -> 128 px). A box is
    assigned to the single quarter that holds its *center* and then clipped to
    that quarter, so a crown straddling an internal seam is owned by exactly one
    quarter (no duplication) and the model learns to fire only when a crown's
    center is in-tile. This mirrors how the quarter model is run at inference
    (eval_checkpoints.quarter_stitch) and keeps trees-per-tile (so the required
    query budget Q) low.

    Each sample is shape-compatible with ``deepforest_collate_fn`` and also carries
    ``parent`` (the source tile path) and ``crop`` (the quarter's offset in the 256
    parent frame) so predictions can be stitched back to the parent and scored
    against the full 256-frame GT in ``self.parent_gt`` (see eval/stitch.py). Empty
    quarters are kept by default: with ``exhaustive=True`` they are useful "no tree
    here" negatives in training, and at eval time running all four quarters matches
    inference exactly.
    """

    def __init__(
        self,
        annotations_path: str,
        norm_stats_path: Optional[str] = None,
        augment: bool = True,
        tile_size: int = 256,
        n_split: int = 2,
        drop_empty: bool = False,
        tiles_root: Optional[str] = None,
    ):
        """``tiles_root``: if set, each item's ``tile_path`` is rebased onto this
        directory (keeping only the filename). The annotation JSONs store absolute
        Modal-volume paths (``/data/stage_c/tiles/...``); point ``tiles_root`` at the
        local tiles directory to read them without rewriting the JSON."""
        from .neon_dataset import _boxes_to_norm_cxcywh

        if tile_size % n_split != 0:
            raise ValueError(f"tile_size {tile_size} not divisible by n_split {n_split}")

        with open(annotations_path) as f:
            items = json.load(f)

        self.augment = NAIPAugmentation() if augment else None
        self.source_size = tile_size                 # parent frame for stitching
        self.out_size = tile_size // n_split         # quarter size (e.g. 128)

        if norm_stats_path is not None:
            self.mean, self.std = load_naip_normalization_stats(norm_stats_path)
        else:
            self.mean = np.array([110.16, 115.41, 98.15, 139.04], dtype=np.float32)
            self.std  = np.array([47.23, 39.82, 35.43, 49.86], dtype=np.float32)

        self.samples: list[dict] = []
        # parent_gt: FULL 256-frame GT per parent (normalized cxcywh), used to
        # score stitched quarter predictions (eval/stitch.stitch_eval).
        self.parent_gt: dict[str, np.ndarray] = {}

        S = float(tile_size)
        out = self.out_size
        for it in items:
            tp = str(it["tile_path"])
            if tiles_root is not None:
                tp = str(Path(tiles_root) / Path(tp).name)   # rebase onto local tiles dir
            parent = tp                                # local path; unique per tile -> parent key
            raw = np.asarray(it.get("boxes", []), dtype=np.float32).reshape(-1, 4)
            self.parent_gt[parent] = raw.copy()       # already 256-norm cxcywh

            # 256-norm cxcywh -> pixel xyxy in the 256 frame (+ pixel centers).
            if len(raw):
                cx, cy, w, h = raw[:, 0] * S, raw[:, 1] * S, raw[:, 2] * S, raw[:, 3] * S
                xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            else:
                cx = cy = np.zeros(0, dtype=np.float32)
                xyxy = np.zeros((0, 4), dtype=np.float32)

            meta = self._meta(it, parent)
            for gr in range(n_split):
                for gc in range(n_split):
                    off_x, off_y = gc * out, gr * out
                    in_cell = (
                        (cx >= off_x) & (cx < off_x + out) &
                        (cy >= off_y) & (cy < off_y + out)
                    )
                    cell_xyxy = xyxy[in_cell]
                    # Shift to the quarter, clip, drop sub-pixel, renormalize to [0,1].
                    boxes = _boxes_to_norm_cxcywh(cell_xyxy, off_x, off_y, out)
                    if drop_empty and len(boxes) == 0:
                        continue
                    self.samples.append({
                        **meta,
                        "boxes": boxes,
                        "read_crop": (off_x, off_y, out),   # window into the 256 tif
                        "stitch": (off_x, off_y, out),      # offset in the parent frame
                        "parent": parent,
                    })

    def _meta(self, it: dict, parent: str) -> dict:
        date_str = it.get("acquisition_date", "2020-06-01")
        return dict(
            tile_path=parent,
            lat=float(it["lat"]),
            lon=float(it["lon"]),
            week=self._week_of_year(date_str),
            hour=float(it.get("hour_of_day", 12.0)),
            exhaustive=bool(it.get("exhaustive", True)),
        )

    @property
    def group_keys(self) -> list:
        """Parent tile path per sample — for a leakage-free, group-aware split
        (all quarters of a parent stay in one split) when a single dataset is
        internally split. Unused here (train/val/test are separate files)."""
        return [s["parent"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        pixels = self._load_quarter(s["tile_path"], s["read_crop"])   # (4, out, out) raw DN
        boxes = torch.from_numpy(s["boxes"])
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if self.augment is not None:
            pixels, boxes = self.augment(pixels, boxes)

        mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
        std  = torch.tensor(self.std,  dtype=torch.float32).view(-1, 1, 1)
        pixels = (pixels - mean) / std

        return {
            "pixels":      pixels,
            "wavelengths": NAIP_WAVELENGTHS,
            "gsd":         torch.tensor([NAIP_GSD], dtype=torch.float32),
            "time":        encode_time(s["week"], s["hour"]),
            "latlon":      encode_latlon(s["lat"], s["lon"]),
            "boxes":       boxes,
            "exhaustive":  s["exhaustive"],
            "tile_path":   s["tile_path"],
            "parent":      s["parent"],
            "crop":        s["stitch"],     # parent-frame offset for stitching
        }

    def _load_quarter(self, path: str, crop) -> torch.Tensor:
        """Read one (4, size, size) quarter window from the 256 GeoTIFF."""
        import rasterio
        from rasterio.windows import Window
        off_x, off_y, size = crop
        with rasterio.open(path) as src:
            data = src.read(
                indexes=[1, 2, 3, 4],
                window=Window(off_x, off_y, size, size),
            ).astype(np.float32)
        return torch.from_numpy(data)

    @staticmethod
    def _week_of_year(date_str: str) -> int:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.isocalendar()[1]
