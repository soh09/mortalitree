"""
Modal entrypoint for Stage A pretraining.

Four functions, all backed by a shared Modal Volume mounted at /data:

  download    → fetch NAIP COGs from Microsoft Planetary Computer
  tile        → slice COGs into 4-channel 224x224 .npy patches + manifest.csv
  stats       → compute per-band mean/std → stats.json
  train       → SimMIM pretraining on A100-40GB, logging to wandb

Typical full-pipeline run from the laptop:

  modal run mortalitree/unsupervised_learning/modal_app.py::download
  modal run mortalitree/unsupervised_learning/modal_app.py::tile
  modal run mortalitree/unsupervised_learning/modal_app.py::stats
  modal run --detach mortalitree/unsupervised_learning/modal_app.py::train

wandb credentials are read from a Modal Secret named `wandb` containing
`WANDB_API_KEY`. Create once with:

  modal secret create wandb WANDB_API_KEY=<your-key>
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "mortalitree-stage-a"
VOLUME_NAME = "mortalitree-naip-pretrain"
DATA_MOUNT = "/data"

# Files we sync into the image so the in-container code matches the repo.
LOCAL_PKG_DIR = Path(__file__).parent
REMOTE_PKG_DIR = "/app/mortalitree/unsupervised_learning"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgdal-dev", "gdal-bin")
    .pip_install_from_requirements(str(LOCAL_PKG_DIR / "requirements.txt"))
    # Sync the package into the image. `mortalitree/` is an implicit namespace
    # package (no __init__.py), which Python 3 imports correctly as long as
    # PYTHONPATH points at /app.
    .add_local_dir(str(LOCAL_PKG_DIR), REMOTE_PKG_DIR, copy=True)
    .env({
        "DATA_ROOT": DATA_MOUNT,
        "OUT_DIR": f"{DATA_MOUNT}/checkpoints/stage_a",
        "PYTHONPATH": "/app",
    })
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={DATA_MOUNT: volume},
    timeout=6 * 60 * 60,  # NAIP COGs are large; lots of them takes hours
    cpu=4.0,
    memory=8192,
)
def download() -> None:
    """Download every NAIP 2022 COG inside any of the CA forest AOIs."""
    from mortalitree.unsupervised_learning.data import download_naip
    download_naip.main()
    volume.commit()


@app.function(
    volumes={DATA_MOUNT: volume},
    timeout=6 * 60 * 60,
    cpu=8.0,
    memory=16384,
)
def tile() -> None:
    """Slice downloaded COGs into 4-channel 224x224 .npy patches."""
    from mortalitree.unsupervised_learning.data import make_pretrain_patches
    make_pretrain_patches.main()
    volume.commit()


@app.function(
    volumes={DATA_MOUNT: volume},
    timeout=60 * 60,
    cpu=4.0,
    memory=8192,
)
def splits() -> None:
    """Scene-level train/val split using raw .tif centroids. Run AFTER tile,
    BEFORE train. Writes splits.csv to the Volume. Idempotent — re-run any
    time you add scenes or change VAL_FRACTION / RESERVED_* env vars."""
    from mortalitree.unsupervised_learning.data import make_splits
    make_splits.main()
    volume.commit()


@app.function(
    volumes={DATA_MOUNT: volume},
    timeout=30 * 60,
    cpu=4.0,
    memory=8192,
)
def stats() -> None:
    """Compute per-band mean/std from a random sample of patches."""
    from mortalitree.unsupervised_learning.data import compute_naip_stats
    compute_naip_stats.main()
    volume.commit()


@app.function(
    volumes={DATA_MOUNT: volume},
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    cpu=16.0,                 # one CPU per dataloader worker, with margin
    memory=48 * 1024,
    # Default ephemeral disk on Modal is already ≥512 GiB, plenty for the
    # ~38 GB /tmp patch cache the trainer builds at startup.
    secrets=[modal.Secret.from_name("wandb")],
)
def train() -> None:
    """SimMIM pretraining on A100-40GB. Logs to wandb, checkpoints to volume."""
    from mortalitree.unsupervised_learning.train.mae_pretrain import Config, main
    cfg = Config()
    main(cfg)
    volume.commit()


@app.function(
    # Modal caps function timeouts at 86400s (24h). Orchestrator is idle
    # almost the entire time — just blocked on .remote() — so the cap
    # applies to wall-clock spent waiting, not compute. The train call
    # itself has its own 24h timeout running in a separate container.
    timeout=24 * 60 * 60,
    cpu=1.0,
    memory=512,
)
def splits_then_train() -> None:
    """Server-side chain: build splits.csv, then start pretraining. Each
    .remote() call runs in its own container with its own resources (the
    A100 is only allocated for train). Launch with:

        modal run --detach mortalitree/unsupervised_learning/modal_app.py::splits_then_train

    The orchestrator itself runs on Modal, so closing your laptop is fine.
    Modal caps function timeouts at 24h; the orchestrator stays alive for
    splits (~1h) + train (~22h cache+epochs), which fits — but if you want
    longer training runs, use spawn() instead of remote() so the orchestrator
    fires-and-forgets and exits in seconds."""
    print("=== splits ===")
    splits.remote()
    print("=== train (spawning, orchestrator will exit) ===")
    # spawn() returns immediately; the train function call lives independently
    # under the same app and survives the orchestrator's exit.
    train.spawn()
    print("=== orchestrator done — train continues server-side ===")


