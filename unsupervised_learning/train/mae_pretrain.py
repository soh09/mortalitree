"""
Stage A — Domain-adaptive SimMIM pretraining for the 4-channel Swin encoder.

Standalone training script. The Modal entrypoint imports `main()` and calls
it inside a GPU function, but the script can also be run locally:

    DATA_ROOT=/path/to/data python -m mortalitree.unsupervised_learning.train.mae_pretrain

Hyperparameters: AdamW lr 1.5e-4 cosine, 10 epoch warmup, wd 0.05,
batch 64+, 100-200 epochs, per-patch normalized MSE on all 4 channels.
Mask ratio 0.5 — lower than the MAE/SimMIM default of 0.75 because NAIP
canopy is dense and 75% masking starved the model of local context.

Validation: runs once per epoch on a scene-level held-out split
(splits.csv from make_splits.py). The val mask is deterministic per
sample, so val/loss is directly comparable epoch-to-epoch — no
random-mask noise. encoder_best.pt is selected on val loss.

Checkpointing: saves to {OUT_DIR}/ckpt_latest.pt periodically (full
training state, for resume) and to encoder_latest.pt / encoder_best.pt
(encoder weights only — what Stages B/C need). Auto-resumes from
ckpt_latest.pt if present.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.naip_dataset import NaipPretrainDataset
from ..model.simmim import SwinSimMIM, simmim_loss


@dataclass
class Config:
    # paths
    data_root: str = os.environ.get(
        "DATA_ROOT", str(Path(__file__).resolve().parents[1])
    )
    out_dir: str = os.environ.get(
        "OUT_DIR",
        str(Path(__file__).resolve().parents[1] / "checkpoints" / "stage_a"),
    )

    # model
    backbone: str = "swin_tiny_patch4_window7_224"
    img_size: int = 224
    in_chans: int = 4
    mask_patch_size: int = 32
    model_patch_size: int = 4
    pretrained_imagenet: bool = True

    # masking
    mask_ratio: float = 0.5

    # optim
    epochs: int = int(os.environ.get("EPOCHS", "100"))
    warmup_epochs: int = int(os.environ.get("WARMUP_EPOCHS", "10"))
    lr: float = 1.5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    batch_size: int = int(os.environ.get("BATCH_SIZE", "128"))
    num_workers: int = int(os.environ.get("NUM_WORKERS", "16"))
    # Higher prefetch hides volume latency: 16 workers * 8 = 128 patches in
    # flight (2 batches of 64). Tune up if GPU util is still <90% steady-state.
    prefetch_factor: int = int(os.environ.get("PREFETCH_FACTOR", "8"))
    grad_clip: float = 1.0

    # ON by default — direct-from-Volume reads worked for ~5 min then GPU
    # util collapsed to 0% (Modal Volume burst-throttle, likely). Cache uses
    # a parallel multiprocessing pool with progress prints, not the tar pipe
    # that hung previously. Flip off only for short debug runs.
    cache_to_local: bool = os.environ.get("CACHE_TO_LOCAL", "1") == "1"
    local_cache_dir: str = os.environ.get("LOCAL_CACHE_DIR", "/tmp/naip_cache")

    # mixed precision (bf16 on A100/H100, fp16 elsewhere)
    amp_dtype: str = os.environ.get("AMP_DTYPE", "bf16")

    # logging / checkpointing / eval
    log_every: int = 20            # log step loss every N optimizer steps
    sample_every_epochs: int = 5   # log image samples every N epochs
    ckpt_every_epochs: int = 5     # write ckpt_latest every N epochs
    eval_every_epochs: int = 1     # run val + log val/loss every N epochs
    val_mask_seed: int = 1234      # used for deterministic per-sample val masks
    seed: int = 0

    # wandb
    wandb_project: str = os.environ.get("WANDB_PROJECT", "mortalitree-pretrain")
    wandb_run_name: str | None = os.environ.get("WANDB_RUN_NAME")


def _cosine_lr(step: int, total_steps: int, warmup_steps: int,
               base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _build_optimizer(model: nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    # No weight decay on biases, LayerNorm params, or the mask token (standard
    # MAE/SimMIM practice — wd on 1-D params hurts).
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "mask_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.95),
    )


def _copy_one(rel: str, src_root: str, dst_root: str) -> None:
    """Top-level helper so multiprocessing.Pool can pickle it."""
    import shutil
    src = Path(src_root) / rel
    dst = Path(dst_root) / rel
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _cache_to_local(src_root: Path, dst_root: Path,
                    n_workers: int = 32) -> Path:
    """Mirror patches/, manifest.csv, stats.json from Modal Volume to local
    SSD using a multiprocessing Pool. Earlier attempts with a single tar pipe
    hung at zero CPU — many concurrent readers are more robust to Modal
    Volume's bursty per-file latency, and we get periodic progress prints
    so a hang is immediately visible.

    Idempotent: presence of dst_root/manifest.csv after the copy is the
    "complete" signal (we copy it last)."""
    import shutil
    from functools import partial
    from multiprocessing import Pool

    import pandas as pd

    dst_root.mkdir(parents=True, exist_ok=True)
    if (dst_root / "manifest.csv").exists() and (dst_root / "patches").exists():
        print(f"local cache already present at {dst_root}")
        # Refresh splits.csv from the source — its content changes more often
        # than the patch pool, so don't trust a cached copy.
        if (src_root / "splits.csv").exists():
            shutil.copy(src_root / "splits.csv", dst_root / "splits.csv")
        return dst_root

    t0 = time.time()
    print(f"caching {src_root} -> {dst_root}  (parallel, {n_workers} workers)")

    df = pd.read_csv(src_root / "manifest.csv")
    rel_paths = df["rel_path"].tolist()
    n_total = len(rel_paths)
    print(f"  {n_total} patches to copy")

    shutil.copy(src_root / "stats.json", dst_root / "stats.json")
    if (src_root / "splits.csv").exists():
        shutil.copy(src_root / "splits.csv", dst_root / "splits.csv")
    else:
        print("  WARN: no splits.csv in source — val will be unavailable")

    with Pool(processes=n_workers) as pool:
        copied = 0
        for _ in pool.imap_unordered(
            partial(_copy_one, src_root=str(src_root), dst_root=str(dst_root)),
            rel_paths, chunksize=200,
        ):
            copied += 1
            if copied % 5000 == 0 or copied == n_total:
                elapsed = time.time() - t0
                rate = copied / max(elapsed, 1e-6)
                eta = (n_total - copied) / max(rate, 1e-6)
                print(f"  cached {copied}/{n_total}  "
                      f"({elapsed:.0f}s elapsed, {rate:.0f} files/s, "
                      f"eta {eta:.0f}s)")

    # Manifest goes last — its presence is the cache-complete signal so we
    # can safely abort/retry mid-copy without leaving a half-written cache.
    shutil.copy(src_root / "manifest.csv", dst_root / "manifest.csv")
    print(f"cache built in {time.time() - t0:.0f}s")
    return dst_root


def _amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _denorm_rgb(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """x: (B, 4, H, W) normalized; returns (B, 3, H, W) in [0, 1] for display."""
    img = x * std + mean
    return img[:, :3].clamp(0, 1)


@torch.no_grad()
def _evaluate(
    model: SwinSimMIM, loader: DataLoader, device: torch.device,
    amp_dtype: torch.dtype, mask_patch_size: int,
) -> float:
    """Mean per-patch normalized MSE on the val split, with deterministic
    per-sample masks. Returns a single scalar comparable epoch-to-epoch."""
    model.eval()
    total = 0.0
    n_batches = 0
    for x, mask in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=(device.type == "cuda")):
            pred = model(x, mask)
            loss = simmim_loss(
                pred, x, mask, mask_patch_size=mask_patch_size, norm_pix=True,
            )
        total += float(loss.detach())
        n_batches += 1
    model.train()
    return total / max(n_batches, 1)


def _build_sample_panel(
    model: SwinSimMIM, sample: tuple[torch.Tensor, torch.Tensor],
    mean: torch.Tensor, std: torch.Tensor, device: torch.device,
) -> "list":
    """Build a few wandb.Image objects (orig, masked-input, recon) for logging."""
    import wandb
    x, mask = sample
    x = x.to(device)
    mask = mask.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x, mask)
    model.train()
    orig_rgb = _denorm_rgb(x, mean.to(device), std.to(device))
    recon_rgb = _denorm_rgb(pred, mean.to(device), std.to(device))
    # Build a masked-input visualization at the *image* scale
    m_img = mask.repeat_interleave(model.mask_patch_size, 1) \
                .repeat_interleave(model.mask_patch_size, 2) \
                .unsqueeze(1).float()  # (B, 1, H, W)
    masked_rgb = orig_rgb * (1.0 - m_img)
    panels = []
    n = min(4, x.shape[0])
    for i in range(n):
        panels.append(wandb.Image(
            torch.cat([orig_rgb[i], masked_rgb[i], recon_rgb[i]], dim=-1).cpu(),
            caption=f"sample {i}: orig | masked | recon",
        ))
    return panels


def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARN: no CUDA — pretraining on CPU is not realistic")

    # --- data ---
    data_root = Path(cfg.data_root)
    if cfg.cache_to_local:
        data_root = _cache_to_local(data_root, Path(cfg.local_cache_dir))

    # Use split-aware datasets when splits.csv exists; fall back to whole-pool
    # otherwise so debug runs without a split still work.
    splits_present = (data_root / "splits.csv").exists()
    train_split_arg = "train" if splits_present else None

    ds = NaipPretrainDataset(
        data_root=data_root,
        split=train_split_arg,
        input_size=cfg.img_size,
        mask_patch_size=cfg.mask_patch_size,
        mask_ratio=cfg.mask_ratio,
        augment=True,
    )
    print(f"train dataset: {len(ds)} patches at {data_root}"
          f"  (split={train_split_arg or 'all'})")
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=True,
    )

    val_loader: DataLoader | None = None
    val_ds: NaipPretrainDataset | None = None
    if splits_present:
        val_ds = NaipPretrainDataset(
            data_root=data_root,
            split="val",
            input_size=cfg.img_size,
            mask_patch_size=cfg.mask_patch_size,
            mask_ratio=cfg.mask_ratio,
            augment=False,
            deterministic_mask_seed=cfg.val_mask_seed,
        )
        # Halve val workers — val is short, big prefetch buffers waste RAM.
        val_workers = max(2, cfg.num_workers // 2)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=val_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=val_workers > 0,
            prefetch_factor=2 if val_workers > 0 else None,
            drop_last=False,
        )
        print(f"val   dataset: {len(val_ds)} patches  "
              f"(deterministic mask seed={cfg.val_mask_seed})")
    else:
        print("WARN: no splits.csv — skipping per-epoch val eval")

    # one fixed batch we re-use to log reconstructions across epochs.
    # Prefer val (deterministic masks → fair epoch-to-epoch comparisons).
    sample_source_ds = val_ds if val_ds is not None else ds
    sample_batch = next(iter(
        DataLoader(sample_source_ds, batch_size=4, shuffle=True, num_workers=0)
    ))

    # --- model ---
    model = SwinSimMIM(
        backbone_name=cfg.backbone,
        in_chans=cfg.in_chans,
        img_size=cfg.img_size,
        mask_patch_size=cfg.mask_patch_size,
        model_patch_size=cfg.model_patch_size,
        pretrained=cfg.pretrained_imagenet,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg.backbone}  params={n_params/1e6:.1f}M")

    optimizer = _build_optimizer(model, cfg.lr, cfg.weight_decay)
    amp_dtype = _amp_dtype(cfg.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    steps_per_epoch = len(loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    # --- wandb ---
    import wandb
    wandb.init(
        project=cfg.wandb_project, name=cfg.wandb_run_name,
        config=asdict(cfg), dir=cfg.out_dir,
    )
    wandb.watch(model, log=None)

    # --- resume ---
    latest_ckpt = Path(cfg.out_dir) / "ckpt_latest.pt"
    start_epoch = 0
    best_loss = float("inf")
    if latest_ckpt.exists():
        print(f"resuming from {latest_ckpt}")
        state = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best_loss = state.get("best_loss", float("inf"))

    # --- train ---
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.epochs}", unit="batch")
        for x, mask in pbar:
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            lr = _cosine_lr(global_step, total_steps, warmup_steps,
                            cfg.lr, cfg.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred = model(x, mask)
                loss = simmim_loss(
                    pred, x, mask, mask_patch_size=cfg.mask_patch_size,
                    norm_pix=True,
                )

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            loss_val = float(loss.detach())
            running += loss_val
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", refresh=False)

            if global_step % cfg.log_every == 0:
                wandb.log({
                    "train/loss_step": loss_val,
                    "train/lr": lr,
                    "train/epoch": epoch + global_step / max(1, total_steps),
                }, step=global_step)
            global_step += 1

        epoch_loss = running / max(1, n_batches)
        epoch_secs = time.time() - t0
        log_payload = {
            "train/loss_epoch": epoch_loss,
            "train/epoch_secs": epoch_secs,
            "epoch": epoch + 1,
        }

        # --- val ---
        val_loss: float | None = None
        if val_loader is not None and (
            (epoch + 1) % cfg.eval_every_epochs == 0 or epoch == cfg.epochs - 1
        ):
            t_val = time.time()
            val_loss = _evaluate(
                model, val_loader, device, amp_dtype, cfg.mask_patch_size,
            )
            log_payload["val/loss"] = val_loss
            log_payload["val/eval_secs"] = time.time() - t_val
            print(f"epoch {epoch+1}: train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  ({epoch_secs:.0f}s + "
                  f"{log_payload['val/eval_secs']:.0f}s eval)")
        else:
            print(f"epoch {epoch+1}: train_loss={epoch_loss:.4f}  "
                  f"({epoch_secs:.0f}s)")

        wandb.log(log_payload, step=global_step)

        if (epoch + 1) % cfg.sample_every_epochs == 0 or epoch == cfg.epochs - 1:
            panels = _build_sample_panel(
                model, sample_batch, ds.mean, ds.std, device
            )
            wandb.log({"samples/recon": panels}, step=global_step)

        # Selection metric: val_loss when available, train_loss otherwise.
        sel_loss = val_loss if val_loss is not None else epoch_loss

        # --- checkpoints ---
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or epoch == cfg.epochs - 1:
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_loss": val_loss,
                "best_loss": best_loss,
                "config": asdict(cfg),
            }
            torch.save(payload, latest_ckpt)
            print(f"  wrote {latest_ckpt}")
            torch.save(
                {"encoder": model.encoder.state_dict(),
                 "epoch": epoch + 1, "config": asdict(cfg)},
                Path(cfg.out_dir) / "encoder_latest.pt",
            )

        if sel_loss < best_loss:
            best_loss = sel_loss
            torch.save(
                {"encoder": model.encoder.state_dict(),
                 "epoch": epoch + 1, "loss": epoch_loss, "val_loss": val_loss,
                 "config": asdict(cfg)},
                Path(cfg.out_dir) / "encoder_best.pt",
            )

    print("done.")
    wandb.finish()


if __name__ == "__main__":
    main()
