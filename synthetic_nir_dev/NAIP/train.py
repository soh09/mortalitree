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
import random
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
# Model input size. make_patches.py stores patches larger than this (STORE_SIZE)
# and WITHOUT overlap; the dataset random-crops CROP_SIZE for train and
# center-crops for val. That gives fresh translational augmentation each epoch
# while keeping zero patch overlap on disk.
CROP_SIZE = 256
BATCH_SIZE = 48
# Per-epoch random subsample, or None for the full set. Patches are now sliced
# WITHOUT overlap (make_patches STRIDE == STORE_SIZE) and train-time random
# cropping supplies fresh translational diversity every epoch, so there's no
# overlap redundancy left to subsample away — use the full set. Set to an int
# to cap epoch size if you want shorter epochs.
SAMPLES_PER_EPOCH = None
# Windows spawns a fresh process per worker (each re-imports torch, ~0.5GB) for
# BOTH the train and val loaders, so keep this modest on a 16GB box. 4 still
# feeds the GPU since training is compute-bound under AMP.
NUM_WORKERS = 4
N_EPOCHS = 40
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

# Fixed seed so runs are comparable (e.g. before/after adding fire AOIs). Seeds
# python/numpy/torch and the per-epoch subsample sampler; DataLoader workers are
# reseeded deterministically via seed_worker. Not bit-exact across machines/CUDA
# versions, but stable on a fixed setup.
SEED = 42

# Burn-scar AOIs, used only for the fire-vs-healthy val summary. czu_big_basin
# is included — it's itself the CZU Lightning Complex scar (see aois.py). Edit
# this if the AOI set changes; names not present in val are simply ignored.
FIRE_AOIS = {
    "creek_huntington_lake", "castle_mountain_home",
    "north_complex_feather", "scu_henry_coe", "czu_big_basin",
}

WANDB_PROJECT = "naip-rgb2nir"
WANDB_ENTITY = 'sohirota-stanford-university'  # set to your team/user, or leave None to use default
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # "online" | "offline" | "disabled"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # Each DataLoader worker is a fresh process; PyTorch gives it a derived torch
    # seed, but numpy/random need reseeding so augmentation RNG is reproducible.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NaipPatchDataset(Dataset):
    def __init__(self, split: str, resolutions_m: list[float], augment: bool,
                 return_aoi: bool = False):
        # Read only the columns we need: on Windows every DataLoader worker is a
        # fresh process that gets its own pickled copy of this dataset, so holding
        # the full manifest DataFrame would replicate ~hundreds of MB per worker.
        df = pd.read_csv(MANIFEST_PATH,
                         usecols=["scene_id", "resolution_m", "rgb_path", "nir_path", "aoi"])
        splits = pd.read_csv(SPLITS_PATH)
        df = df.merge(splits, on="scene_id")
        df = df[df["split"] == split]
        df = df[df["resolution_m"].isin(resolutions_m)]
        if len(df) == 0:
            raise SystemExit(
                f"No patches for split={split} at resolutions {resolutions_m}. "
                f"Did make_splits.py run, and does the split actually exist?"
            )
        # Keep just the path columns as plain lists and drop the DataFrame, so
        # what gets pickled to each worker process stays small.
        self.rgb_paths = df["rgb_path"].tolist()
        self.nir_paths = df["nir_path"].tolist()
        self.augment = augment
        self.return_aoi = return_aoi
        # Stable AOI index table (small) so __getitem__ can hand back an integer
        # AOI id for per-AOI val metrics. aoi_names maps id -> name for readout.
        aois = df["aoi"].fillna("<blank>").tolist()
        self.aoi_names = sorted(set(aois))
        idx_of = {name: i for i, name in enumerate(self.aoi_names)}
        self.aoi_ids = [idx_of[a] for a in aois]

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, idx: int):
        rgb = np.array(Image.open(ROOT / self.rgb_paths[idx]).convert("RGB"))  # (H,W,3) u8
        nir = np.array(Image.open(ROOT / self.nir_paths[idx]).convert("L"))    # (H,W) u8
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        nir_t = torch.from_numpy(nir).unsqueeze(0).float() / 255.0
        # Random crop while training (translational aug), center crop for val so
        # the metric is deterministic. Then the flip/rot/jitter augments.
        rgb_t, nir_t = crop_pair(rgb_t, nir_t, CROP_SIZE, random_crop=self.augment)
        if self.augment:
            rgb_t, nir_t = augment_pair(rgb_t, nir_t)
        if self.return_aoi:
            return rgb_t, nir_t, self.aoi_ids[idx]
        return rgb_t, nir_t


def crop_pair(rgb: torch.Tensor, nir: torch.Tensor, size: int, random_crop: bool
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop RGB+NIR (both C,H,W) to size x size with identical coords so the pair
    stays aligned. Random offset when training (translational augmentation),
    center crop otherwise so val is deterministic. Patches are stored larger than
    `size` (make_patches STORE_SIZE); if a stored patch is exactly `size` the
    offset range collapses to 0 and this is a no-op. Uses torch RNG so it inherits
    the seeded per-worker seeding."""
    _, h, w = rgb.shape
    if h < size or w < size:
        raise ValueError(f"patch {h}x{w} smaller than crop size {size}")
    if random_crop:
        top = int(torch.randint(0, h - size + 1, (1,)).item())
        left = int(torch.randint(0, w - size + 1, (1,)).item())
    else:
        top = (h - size) // 2
        left = (w - size) // 2
    return rgb[:, top:top + size, left:left + size], nir[:, top:top + size, left:left + size]


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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             aoi_names: list[str]) -> dict:
    """Val NIR/NDVI MAE, overall and bucketed per AOI (and fire vs healthy).
    Expects the loader to yield (rgb, nir, aoi_id); aoi_names maps id -> name.
    The overall numbers are identical to a flat mean — buckets just re-sum."""
    model.eval()
    use_amp = device.type == "cuda"
    n_aoi = len(aoi_names)
    # Per-AOI running sums (CPU); index_add_ scatters each sample into its bucket.
    nir_sum = torch.zeros(n_aoi)
    ndvi_sum = torch.zeros(n_aoi)
    count = torch.zeros(n_aoi)
    for rgb, nir_t, aoi_id in tqdm(loader, desc="  val", unit="batch", leave=False):
        rgb = rgb.to(device, non_blocking=True)
        nir_t = nir_t.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(rgb)
        pred = pred.float()  # metrics in fp32 so val numbers stay precise/comparable
        per_nir = (pred - nir_t).abs().mean(dim=(1, 2, 3)).cpu()
        red = rgb[:, 0:1]
        ad, mask = ndvi_absdiff(pred, nir_t, red)
        num = (ad * mask).sum(dim=(1, 2, 3))
        den = mask.sum(dim=(1, 2, 3)).clamp(min=1)
        per_ndvi = (num / den).cpu()
        aoi_id = aoi_id.cpu().long()
        nir_sum.index_add_(0, aoi_id, per_nir)
        ndvi_sum.index_add_(0, aoi_id, per_ndvi)
        count.index_add_(0, aoi_id, torch.ones_like(per_nir))

    per_aoi = {}
    for i, name in enumerate(aoi_names):
        c = count[i].item()
        if c == 0:
            continue
        per_aoi[name] = {
            "nir_mae": nir_sum[i].item() / c,
            "ndvi_mae": ndvi_sum[i].item() / c,
            "n": int(c),
        }

    total = count.sum().item()
    metrics: dict = {
        "nir_mae": nir_sum.sum().item() / total,
        "ndvi_mae": ndvi_sum.sum().item() / total,
        "per_aoi": per_aoi,
    }
    # Fire vs healthy grouped means, weighted by patch count.
    for group, members in (("fire", FIRE_AOIS),
                           ("healthy", set(aoi_names) - FIRE_AOIS)):
        idx = [i for i, nm in enumerate(aoi_names) if nm in members]
        gc = count[idx].sum().item() if idx else 0.0
        if gc > 0:
            metrics[f"{group}_nir_mae"] = nir_sum[idx].sum().item() / gc
            metrics[f"{group}_ndvi_mae"] = ndvi_sum[idx].sum().item() / gc
    return metrics


def main():
    CKPT_DIR.mkdir(exist_ok=True)
    seed_everything(SEED)
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
            "seed": SEED,
            "fire_aois": sorted(FIRE_AOIS),
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
    val_ds = NaipPatchDataset("val", VAL_RESOLUTIONS_M, augment=False, return_aoi=True)
    print(f"train: {len(train_ds)} patches, val: {len(val_ds)} patches")
    wandb.config.update({"train_patches": len(train_ds), "val_patches": len(val_ds)})

    pin = device.type == "cuda"
    # Seeded generator drives both the per-epoch subsample and shuffle, so the
    # patch order is reproducible across runs with the same SEED.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(SEED)
    # Subsample a fresh random subset each epoch (no replacement) so we skip the
    # ~4x patch-overlap redundancy without losing dataset coverage over a run.
    if SAMPLES_PER_EPOCH is not None and SAMPLES_PER_EPOCH < len(train_ds):
        train_sampler = torch.utils.data.RandomSampler(
            train_ds, replacement=False, num_samples=SAMPLES_PER_EPOCH,
            generator=loader_gen)
        print(f"sampling {SAMPLES_PER_EPOCH}/{len(train_ds)} train patches per epoch")
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker, generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
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
    # Per-AOI val metrics go to their own long-format CSV so the main log stays
    # tidy. One row per (epoch, aoi).
    per_aoi_csv = run_dir / "val_per_aoi.csv"
    per_aoi_fh = open(per_aoi_csv, "w", newline="")
    per_aoi_writer = csv.writer(per_aoi_fh)
    per_aoi_writer.writerow(["epoch", "aoi", "n", "nir_mae", "ndvi_mae"])

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

            val_metrics = evaluate(model, val_loader, device, val_ds.aoi_names)
            epoch_secs = time.time() - t0
            print(f"epoch {epoch}: "
                  f"train_loss={loss_sum/steps:.4f} "
                  f"val_nir_mae={val_metrics['nir_mae']:.4f} "
                  f"val_ndvi_mae={val_metrics['ndvi_mae']:.4f} "
                  f"({epoch_secs:.1f}s)")
            if "fire_nir_mae" in val_metrics and "healthy_nir_mae" in val_metrics:
                print(f"  by group nir_mae: fire={val_metrics['fire_nir_mae']:.4f} "
                      f"healthy={val_metrics['healthy_nir_mae']:.4f}")

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

            for name, m in sorted(val_metrics["per_aoi"].items()):
                per_aoi_writer.writerow([
                    epoch, name, m["n"],
                    round(m["nir_mae"], 6), round(m["ndvi_mae"], 6),
                ])
            per_aoi_fh.flush()

            log_dict = {
                "epoch": epoch,
                "train/loss": loss_sum / steps,
                "train/l1": l1_sum / steps,
                "train/ndvi_l1": ndvi_sum / steps,
                "val/nir_mae": val_metrics["nir_mae"],
                "val/ndvi_mae": val_metrics["ndvi_mae"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_secs": epoch_secs,
            }
            for name, m in val_metrics["per_aoi"].items():
                log_dict[f"val_aoi/nir_mae/{name}"] = m["nir_mae"]
                log_dict[f"val_aoi/ndvi_mae/{name}"] = m["ndvi_mae"]
            for group in ("fire", "healthy"):
                if f"{group}_nir_mae" in val_metrics:
                    log_dict[f"val_group/nir_mae/{group}"] = val_metrics[f"{group}_nir_mae"]
                    log_dict[f"val_group/ndvi_mae/{group}"] = val_metrics[f"{group}_ndvi_mae"]
            wandb.log(log_dict)

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
                # Full per-AOI table only on improvement, worst-first, so the
                # console isn't flooded every epoch but you see where error lives.
                print("  per-AOI val nir_mae (worst first):")
                for name, m in sorted(val_metrics["per_aoi"].items(),
                                      key=lambda kv: kv[1]["nir_mae"], reverse=True):
                    print(f"    {name:<26} {m['nir_mae']:.4f}  "
                          f"(ndvi {m['ndvi_mae']:.4f}, n={m['n']})")
    finally:
        log_fh.close()
        per_aoi_fh.close()
        wandb.finish()


if __name__ == "__main__":
    main()
