"""
Train an RGB -> NIR translation model on NAIP patches.

Architecture: U-Net with ResNet-34 encoder (ImageNet-pretrained).
Loss: L1 + NDVI_WEIGHT * NDVI-L1, where NDVI uses the input red channel
and the predicted (or target) NIR.

Reads manifest.csv + splits.csv; train+val only (test is held out — run a
separate eval script once model selection is done).

Outputs:
  checkpoints/best.pt        best checkpoint by val NIR MAE
  checkpoints/train_log.csv  per-epoch metrics

Requires: torch, segmentation-models-pytorch, pandas, tqdm, pillow, numpy
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "manifest.csv"
SPLITS_PATH = ROOT / "splits.csv"
CKPT_DIR = ROOT / "checkpoints"
BEST_CKPT = CKPT_DIR / "best.pt"
LOG_CSV = CKPT_DIR / "train_log.csv"

# v1: train on native 0.6m only. Add 1.0/1.5/2.0 later as scale augmentation.
RESOLUTIONS_M = [0.6]
BATCH_SIZE = 32
NUM_WORKERS = min(8, os.cpu_count() or 4)
N_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
NDVI_WEIGHT = 0.5
COLOR_JITTER_RANGE = 0.10  # +/-10% brightness/contrast on RGB only


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NaipPatchDataset(Dataset):
    def __init__(self, split: str, resolutions_m: list[float], augment: bool):
        df = pd.read_csv(MANIFEST_PATH)
        splits = pd.read_csv(SPLITS_PATH)
        df = df.merge(splits, on="scene_id")
        df = df[df["split"] == split]
        df = df[df["resolution_m"].isin(resolutions_m)]
        if len(df) == 0:
            raise SystemExit(
                f"No patches for split={split} at resolutions {resolutions_m}. "
                f"Did make_splits.py run, and does the split actually exist?"
            )
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        rgb = np.array(Image.open(ROOT / row["rgb_path"]).convert("RGB"))  # (H,W,3) u8
        nir = np.array(Image.open(ROOT / row["nir_path"]).convert("L"))    # (H,W) u8
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        nir_t = torch.from_numpy(nir).unsqueeze(0).float() / 255.0
        if self.augment:
            rgb_t, nir_t = augment_pair(rgb_t, nir_t)
        return rgb_t, nir_t


def augment_pair(rgb: torch.Tensor, nir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Spatial transforms applied identically to RGB and NIR (preserves alignment).
    if torch.rand(1).item() < 0.5:
        rgb = torch.flip(rgb, dims=[-1])
        nir = torch.flip(nir, dims=[-1])
    if torch.rand(1).item() < 0.5:
        rgb = torch.flip(rgb, dims=[-2])
        nir = torch.flip(nir, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        rgb = torch.rot90(rgb, k, dims=(-2, -1))
        nir = torch.rot90(nir, k, dims=(-2, -1))
    # Mild RGB-only brightness/contrast jitter — simulates radiometric noise
    # without teaching the model that NIR scales with RGB exposure.
    if torch.rand(1).item() < 0.5:
        b = 1.0 + (torch.rand(1).item() - 0.5) * 2 * COLOR_JITTER_RANGE
        c = 1.0 + (torch.rand(1).item() - 0.5) * 2 * COLOR_JITTER_RANGE
        mean = rgb.mean(dim=(-2, -1), keepdim=True)
        rgb = ((rgb - mean) * c + mean) * b
        rgb = rgb.clamp(0, 1)
    return rgb, nir


def ndvi(nir: torch.Tensor, red: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (nir - red) / (nir + red + eps)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, rgb: torch.Tensor
                 ) -> tuple[torch.Tensor, float, float]:
    l1 = (pred - target).abs().mean()
    red = rgb[:, 0:1]
    ndvi_l1 = (ndvi(pred, red) - ndvi(target, red)).abs().mean()
    total = l1 + NDVI_WEIGHT * ndvi_l1
    return total, l1.item(), ndvi_l1.item()


class UnetWithImageNetNorm(nn.Module):
    """Wraps the SMP U-Net so the dataset can keep RGB in [0,1] (needed for
    NDVI loss) while the encoder still sees ImageNet-normalized inputs."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.base((rgb - self.mean) / self.std)


def build_model() -> nn.Module:
    base = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    return UnetWithImageNetNorm(base)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    nir_mae_sum = 0.0
    ndvi_mae_sum = 0.0
    n = 0
    for rgb, nir_t in loader:
        rgb = rgb.to(device, non_blocking=True)
        nir_t = nir_t.to(device, non_blocking=True)
        pred = model(rgb)
        nir_mae_sum += (pred - nir_t).abs().mean(dim=(1, 2, 3)).sum().item()
        red = rgb[:, 0:1]
        ndvi_mae_sum += (ndvi(pred, red) - ndvi(nir_t, red)).abs().mean(dim=(1, 2, 3)).sum().item()
        n += rgb.shape[0]
    return {"nir_mae": nir_mae_sum / n, "ndvi_mae": ndvi_mae_sum / n}


def main():
    CKPT_DIR.mkdir(exist_ok=True)
    device = pick_device()
    print(f"device: {device}")

    train_ds = NaipPatchDataset("train", RESOLUTIONS_M, augment=True)
    val_ds = NaipPatchDataset("val", RESOLUTIONS_M, augment=False)
    print(f"train: {len(train_ds)} patches, val: {len(val_ds)} patches")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    log_exists = LOG_CSV.exists()
    log_fh = open(LOG_CSV, "a", newline="")
    log_writer = csv.writer(log_fh)
    if not log_exists:
        log_writer.writerow([
            "epoch", "train_loss", "train_l1", "train_ndvi_l1",
            "val_nir_mae", "val_ndvi_mae", "lr", "epoch_secs",
        ])

    best_val_mae = float("inf")
    try:
        for epoch in range(1, N_EPOCHS + 1):
            t0 = time.time()
            model.train()
            loss_sum = l1_sum = ndvi_sum = 0.0
            steps = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{N_EPOCHS}", unit="batch")
            for rgb, nir_t in pbar:
                rgb = rgb.to(device, non_blocking=True)
                nir_t = nir_t.to(device, non_blocking=True)
                pred = model(rgb)
                loss, l1_val, ndvi_val = compute_loss(pred, nir_t, rgb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                l1_sum += l1_val
                ndvi_sum += ndvi_val
                steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 l1=f"{l1_val:.4f}",
                                 ndvi=f"{ndvi_val:.4f}",
                                 refresh=False)
            scheduler.step()

            val_metrics = evaluate(model, val_loader, device)
            epoch_secs = time.time() - t0
            print(f"epoch {epoch}: "
                  f"train_loss={loss_sum/steps:.4f} "
                  f"val_nir_mae={val_metrics['nir_mae']:.4f} "
                  f"val_ndvi_mae={val_metrics['ndvi_mae']:.4f} "
                  f"({epoch_secs:.1f}s)")

            log_writer.writerow([
                epoch,
                round(loss_sum / steps, 6),
                round(l1_sum / steps, 6),
                round(ndvi_sum / steps, 6),
                round(val_metrics["nir_mae"], 6),
                round(val_metrics["ndvi_mae"], 6),
                optimizer.param_groups[0]["lr"],
                round(epoch_secs, 1),
            ])
            log_fh.flush()

            if val_metrics["nir_mae"] < best_val_mae:
                best_val_mae = val_metrics["nir_mae"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": {
                        "encoder": "resnet34",
                        "resolutions_m": RESOLUTIONS_M,
                        "ndvi_weight": NDVI_WEIGHT,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                    },
                }, BEST_CKPT)
                print(f"  -> new best, saved {BEST_CKPT}")
    finally:
        log_fh.close()


if __name__ == "__main__":
    main()
