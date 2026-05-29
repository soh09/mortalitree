"""
PyTorch Dataset for Stage A MAE/SimMIM pretraining on 4-channel NAIP patches.

Reads .npy patches written by make_pretrain_patches.py, normalizes per-band
using stats.json, applies geometric-only augmentation (rotation + flips —
spectral identity must be preserved, so no per-band color jitter), and
generates a per-sample mask for SimMIM.

If $DATA_ROOT/splits.csv is present (written by make_splits.py), the
Dataset filters its rows by `split` ∈ {"train", "val"}; without splits.csv
the whole manifest is used. The split MUST be precomputed at the scene
level — see make_splits.py for the rationale.

The mask is a (n_mp, n_mp) bool grid where True = masked. n_mp is the
number of mask patches along one axis (e.g. 7 for img=224, mask_patch=32).
For val, pass deterministic_mask_seed=<int> so each sample gets the same
mask every epoch; that makes val loss directly comparable epoch-to-epoch
without the noise of fresh random masks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent))


class MaskGenerator:
    """SimMIM-style block mask.

    Stochastic by default (fresh permutation per call). If
    deterministic_seed is set, hashing it with sample_idx yields the same
    mask every time for the same sample — used for val so the loss is
    epoch-to-epoch comparable without random-mask noise.
    """

    def __init__(self, input_size: int = 224, mask_patch_size: int = 32,
                 mask_ratio: float = 0.5,
                 deterministic_seed: int | None = None):
        assert input_size % mask_patch_size == 0
        self.n_mp = input_size // mask_patch_size
        self.n_total = self.n_mp * self.n_mp
        self.n_masked = int(round(self.n_total * mask_ratio))
        self.deterministic_seed = deterministic_seed

    def __call__(self, sample_idx: int | None = None) -> torch.Tensor:
        if self.deterministic_seed is not None and sample_idx is not None:
            rng = np.random.default_rng(
                self.deterministic_seed * 1_000_003 + sample_idx
            )
            idx = rng.permutation(self.n_total)[: self.n_masked]
        else:
            idx = np.random.permutation(self.n_total)[: self.n_masked]
        mask = np.zeros(self.n_total, dtype=bool)
        mask[idx] = True
        return torch.from_numpy(mask.reshape(self.n_mp, self.n_mp))


def _load_stats(stats_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with open(stats_path) as fh:
        s = json.load(fh)
    mean = torch.tensor(s["mean"], dtype=torch.float32).view(4, 1, 1)
    std = torch.tensor(s["std"], dtype=torch.float32).view(4, 1, 1)
    return mean, std


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Random D4 transform (rotation 0/90/180/270 + flips). All bands together."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    return x


class NaipPretrainDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str | None = None,
        manifest_name: str = "manifest.csv",
        stats_name: str = "stats.json",
        splits_name: str = "splits.csv",
        split: str | None = None,
        input_size: int = 224,
        mask_patch_size: int = 32,
        mask_ratio: float = 0.5,
        augment: bool = True,
        deterministic_mask_seed: int | None = None,
    ):
        root = Path(data_root) if data_root is not None else DATA_ROOT
        self.root = root
        df = pd.read_csv(root / manifest_name)
        if len(df) == 0:
            raise SystemExit(f"Empty manifest at {root / manifest_name}")

        if split is not None:
            splits_path = root / splits_name
            if not splits_path.exists():
                raise SystemExit(
                    f"split={split!r} requested but no {splits_path}. "
                    f"Run make_splits.py first."
                )
            splits = pd.read_csv(splits_path)
            df = df.merge(splits[["scene_id", "split"]], on="scene_id")
            df = df[df["split"] == split]
            if len(df) == 0:
                raise SystemExit(
                    f"No patches for split={split!r} in {splits_path}. "
                    f"Available: {sorted(splits['split'].unique())}"
                )

        self.paths = df["rel_path"].tolist()
        self.mean, self.std = _load_stats(root / stats_name)
        self.augment = augment
        self.input_size = input_size
        self.deterministic_mask_seed = deterministic_mask_seed
        self.mask_gen = MaskGenerator(
            input_size, mask_patch_size, mask_ratio,
            deterministic_seed=deterministic_mask_seed,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = np.load(self.root / self.paths[idx])  # (4, H, W) uint8
        x = torch.from_numpy(arr).float() / 255.0   # (4, H, W) in [0, 1]
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            # patches should already be input_size, but guard against rebuilds
            x = x[:, : self.input_size, : self.input_size]
        if self.augment:
            x = _augment(x)
        x = (x - self.mean) / self.std
        sample_idx = idx if self.deterministic_mask_seed is not None else None
        mask = self.mask_gen(sample_idx=sample_idx)
        return x, mask
