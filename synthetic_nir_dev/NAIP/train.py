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
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "manifest.csv"
SPLITS_PATH = ROOT / "splits.csv"
CKPT_DIR = ROOT / "checkpoints"

# Scale augmentation: train across all GSDs the patch builder generated. The
# lower-res patches cover more ground per 256px tile, adding scale + content
# diversity for free. Validate on native 0.6m so the metric stays comparable to
# earlier runs and matches the deployment resolution.
TRAIN_RESOLUTIONS_M = [0.6, 1.0, 1.5, 2.0]
VAL_RESOLUTIONS_M = [0.6]
BATCH_SIZE = 48
# Per-epoch random subsample. make_patches.py slices patches at STRIDE=128 on
# 256px windows (50% overlap => each pixel is covered ~4x), so the ~357k train
# patches carry ~4x redundancy. Training is GPU-compute-bound, so drawing a
# fresh random subset of this size each epoch cuts epoch time ~linearly while
# still covering the full dataset across epochs. Set to None for the full set.
SAMPLES_PER_EPOCH = 90_000
# Windows spawns a fresh process per worker (each re-imports torch, ~0.5GB) for
# BOTH the train and val loaders, so keep this modest on a 16GB box. 4 still
# feeds the GPU since training is compute-bound under AMP.
NUM_WORKERS = 4
N_EPOCHS = 30
WARMUP_EPOCHS = 2     # linear LR warmup before the cosine decay; steadies the
                      # randomly-initialized decoder over the pretrained encoder.
LR = 1e-4
WEIGHT_DECAY = 1e-4
NDVI_WEIGHT = 0.5
NDVI_EPS = 0.1        # denominator floor for NDVI on [0,1] data. Was 1e-6, which let
                      # dark/shadow pixels (tiny NIR+red) dominate the gradient.
NDVI_MASK_MIN = 0.1   # skip the NDVI term where (target NIR + red) < this: too little
                      # signal for NDVI to mean anything (mostly quantization noise).
COLOR_JITTER_RANGE = 0.10  # +/-10% brightness/contrast on RGB only

WANDB_PROJECT = "naip-rgb2nir"
WANDB_ENTITY = 'sohirota-stanford-university'  # set to your team/user, or leave None to use default
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # "online" | "offline" | "disabled"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NaipPatchDataset(Dataset):
    def __init__(self, split: str, resolutions_m: list[float], augment: bool):
        # Read only the columns we need: on Windows every DataLoader worker is a
        # fresh process that gets its own pickled copy of this dataset, so holding
        # the full manifest DataFrame would replicate ~hundreds of MB per worker.
        df = pd.read_csv(MANIFEST_PATH,
                         usecols=["scene_id", "resolution_m", "rgb_path", "nir_path"])
        splits = pd.read_csv(SPLITS_PATH)
        df = df.merge(splits, on="scene_id")
        df = df[df["split"] == split]
        df = df[df["resolution_m"].isin(resolutions_m)]
        if len(df) == 0:
            raise SystemExit(
                f"No patches for split={split} at resolutions {resolutions_m}. "
                f"Did make_splits.py run, and does the split actually exist?"
            )
        # Keep just the two path columns as plain lists and drop the DataFrame, so
        # what gets pickled to each worker process stays small.
        self.rgb_paths = df["rgb_path"].tolist()
        self.nir_paths = df["nir_path"].tolist()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = np.array(Image.open(ROOT / self.rgb_paths[idx]).convert("RGB"))  # (H,W,3) u8
        nir = np.array(Image.open(ROOT / self.nir_paths[idx]).convert("L"))    # (H,W) u8
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


def ndvi(nir: torch.Tensor, red: torch.Tensor, eps: float = NDVI_EPS) -> torch.Tensor:
    return (nir - red) / (nir + red + eps)


def ndvi_absdiff(pred: torch.Tensor, target: torch.Tensor, red: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel |NDVI(pred) - NDVI(target)| and a validity mask. The mask drops
    low-signal pixels (target NIR + red < NDVI_MASK_MIN) where NDVI is dominated
    by quantization noise; it keys off the *target*, so it doesn't move with the
    prediction. The raw-NIR L1 term still supervises these pixels."""
    ad = (ndvi(pred, red) - ndvi(target, red)).abs()
    mask = (target + red) >= NDVI_MASK_MIN
    return ad, mask


def compute_loss(pred: torch.Tensor, target: torch.Tensor, rgb: torch.Tensor
                 ) -> tuple[torch.Tensor, float, float]:
    l1 = (pred - target).abs().mean()
    red = rgb[:, 0:1]
    ad, mask = ndvi_absdiff(pred, target, red)
    ndvi_l1 = ad[mask].mean() if mask.any() else ad.mean() * 0.0
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
    use_amp = device.type == "cuda"
    nir_mae_sum = 0.0
    ndvi_mae_sum = 0.0
    n = 0
    for rgb, nir_t in tqdm(loader, desc="  val", unit="batch", leave=False):
        rgb = rgb.to(device, non_blocking=True)
        nir_t = nir_t.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(rgb)
        pred = pred.float()  # metrics in fp32 so val numbers stay precise/comparable
        nir_mae_sum += (pred - nir_t).abs().mean(dim=(1, 2, 3)).sum().item()
        red = rgb[:, 0:1]
        ad, mask = ndvi_absdiff(pred, nir_t, red)
        num = (ad * mask).sum(dim=(1, 2, 3))
        den = mask.sum(dim=(1, 2, 3)).clamp(min=1)
        ndvi_mae_sum += (num / den).sum().item()
        n += rgb.shape[0]
    return {"nir_mae": nir_mae_sum / n, "ndvi_mae": ndvi_mae_sum / n}


def main():
    CKPT_DIR.mkdir(exist_ok=True)
    device = pick_device()
    print(f"device: {device}")

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        config={
            "encoder": "resnet34",
            "train_resolutions_m": TRAIN_RESOLUTIONS_M,
            "val_resolutions_m": VAL_RESOLUTIONS_M,
            "batch_size": BATCH_SIZE,
            "samples_per_epoch": SAMPLES_PER_EPOCH,
            "n_epochs": N_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "ndvi_weight": NDVI_WEIGHT,
            "ndvi_eps": NDVI_EPS,
            "ndvi_mask_min": NDVI_MASK_MIN,
            "color_jitter_range": COLOR_JITTER_RANGE,
            "device": device.type,
            "amp_fp16": device.type == "cuda",
        },
    )

    run_id = wandb.run.id if wandb.run is not None else "norun"
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{run_id}"
    run_dir = CKPT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    best_ckpt = run_dir / "best.pt"
    log_csv = run_dir / "train_log.csv"
    print(f"run dir: {run_dir}")
    wandb.config.update({"run_dir": str(run_dir)})

    train_ds = NaipPatchDataset("train", TRAIN_RESOLUTIONS_M, augment=True)
    val_ds = NaipPatchDataset("val", VAL_RESOLUTIONS_M, augment=False)
    print(f"train: {len(train_ds)} patches, val: {len(val_ds)} patches")
    wandb.config.update({"train_patches": len(train_ds), "val_patches": len(val_ds)})

    pin = device.type == "cuda"
    # Subsample a fresh random subset each epoch (no replacement) so we skip the
    # ~4x patch-overlap redundancy without losing dataset coverage over a run.
    if SAMPLES_PER_EPOCH is not None and SAMPLES_PER_EPOCH < len(train_ds):
        train_sampler = torch.utils.data.RandomSampler(
            train_ds, replacement=False, num_samples=SAMPLES_PER_EPOCH)
        print(f"sampling {SAMPLES_PER_EPOCH}/{len(train_ds)} train patches per epoch")
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = build_model().to(device)
    # AMP (fp16 on the Turing tensor cores) ~= 1.6x here, identical math. NOTE:
    # channels_last benchmarked ~4x SLOWER on this smp U-Net (its upsample/concat
    # path doesn't propagate the layout, so every conv pays a transpose), so it is
    # deliberately not used. cudnn autotuning is neutral but harmless on fixed sizes.
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True  # fixed 256x256 inputs
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Per-step LR: linear warmup for WARMUP_EPOCHS, then cosine decay to ~0 over
    # the rest of the run. Stepped once per batch (see the training loop).
    steps_per_epoch = len(train_loader)
    total_steps = N_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    wandb.watch(model, log="gradients", log_freq=200)

    log_fh = open(log_csv, "w", newline="")
    log_writer = csv.writer(log_fh)
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
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(rgb)
                    loss, l1_val, ndvi_val = compute_loss(pred, nir_t, rgb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += loss.item()
                l1_sum += l1_val
                ndvi_sum += ndvi_val
                steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 l1=f"{l1_val:.4f}",
                                 ndvi=f"{ndvi_val:.4f}",
                                 refresh=False)

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

            wandb.log({
                "epoch": epoch,
                "train/loss": loss_sum / steps,
                "train/l1": l1_sum / steps,
                "train/ndvi_l1": ndvi_sum / steps,
                "val/nir_mae": val_metrics["nir_mae"],
                "val/ndvi_mae": val_metrics["ndvi_mae"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_secs": epoch_secs,
            })

            if val_metrics["nir_mae"] < best_val_mae:
                best_val_mae = val_metrics["nir_mae"]
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": {
                        "encoder": "resnet34",
                        "train_resolutions_m": TRAIN_RESOLUTIONS_M,
                        "ndvi_weight": NDVI_WEIGHT,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                    },
                }, best_ckpt)
                print(f"  -> new best, saved {best_ckpt}")
                wandb.summary["best_val_nir_mae"] = best_val_mae
                wandb.summary["best_epoch"] = epoch
    finally:
        log_fh.close()
        wandb.finish()


if __name__ == "__main__":
    main()
