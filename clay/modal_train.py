"""Modal training script for Clay tree detector.

Volumes (create once with `modal volume create`):
  clay-data         — upload annotation JSONs and GeoTIFF tiles here
  clay-checkpoints  — receives model checkpoints; also store clay-v1.5.ckpt here

Upload data:
  modal volume put clay-data  data/naip_train.json        /naip_train.json
  modal volume put clay-data  data/naip_val.json          /naip_val.json
  modal volume put clay-data  data/deepforest.json        /deepforest.json
  modal volume put clay-data  tiles/                      /tiles/

Clay checkpoint:
  Auto-fetched from HuggingFace to the clay-checkpoints volume on first run
  (or explicitly: modal run modal_train.py::fetch_clay_ckpt). No local download.

Run:
  modal run modal_train.py            # Stage B then Stage C
  modal run modal_train.py::stage_b   # Stage B only
  modal run modal_train.py::stage_c   # Stage C only (needs stage_b_best.pt in checkpoints volume)

Model and training hyperparameters are read from configs/stage_b.yaml and
configs/stage_c.yaml (mounted into the image), so the YAML files are the single
source of truth — keep them in sync with src defaults.
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    # git is required by `pip install git+https://...` below (slim has no git).
    .apt_install("git")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        index_url=TORCH_INDEX,
    )
    .pip_install(
        "scipy",
        "rasterio",
        "pyyaml",
        "matplotlib",
        "tqdm",
        "wandb",
        "requests",                 # _ensure_clay_ckpt() HF download
        "opencv-python-headless",   # draw_boxes() in eval/visualize.py
    )
    .pip_install("git+https://github.com/Clay-foundation/model.git")
    # Reassert the CUDA torch build in case the Clay install pulled a CPU wheel.
    .pip_install("torch==2.3.0", "torchvision==0.18.0", index_url=TORCH_INDEX)
    # Stream prints live: without this CPython block-buffers stdout on Modal
    # (stdout is a pipe, not a TTY), so per-epoch/per-batch prints don't appear
    # until the buffer fills or the process exits.
    .env({"PYTHONUNBUFFERED": "1"})
    # Ship local source + configs with the image (modern replacement for Mounts).
    .add_local_dir(HERE / "src", remote_path="/root/src")
    .add_local_dir(HERE / "configs", remote_path="/root/configs")
)

# ---------------------------------------------------------------------------
# App + volumes
# ---------------------------------------------------------------------------
app = modal.App("clay-tree-detector", image=image)

data_vol = modal.Volume.from_name("clay-data",        create_if_missing=True)
ckpt_vol = modal.Volume.from_name("clay-checkpoints", create_if_missing=True)
# NEON Stage-B data produced by dataset_dev/modal_pipeline.py lives on `mot`.
neon_vol = modal.Volume.from_name("mot",              create_if_missing=True)

DATA_DIR  = "/data"
CKPT_DIR  = "/checkpoints"
NEON_DIR  = "/neon"
CLAY_CKPT = "/checkpoints/clay-v1.5.ckpt"
# Public HF download URL (resolve/, not blob/ which is the web view).
HF_CLAY_URL = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
SEED      = 42

WANDB_PROJECT = "clay-tree-detector"
# W&B API key is read from a Modal secret. Create it once with:
#   modal secret create wandb WANDB_API_KEY=<your-key>
# If the secret is absent, training still runs — wandb just stays disabled.
try:
    _wandb_secret = [modal.Secret.from_name("wandb")]
except Exception:
    _wandb_secret = []

COMMON = dict(
    gpu="A10G",
    timeout=60 * 60 * 6,       # 6 hours max
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol, NEON_DIR: neon_vol},
    secrets=_wandb_secret,
)


def _wandb_init(run_name: str, cfg: dict, extra: dict | None = None):
    """Start a W&B run if WANDB_API_KEY is present; otherwise return None."""
    import os
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set — logging disabled.")
        return None
    import wandb
    config = {**cfg.get("training", {}), **cfg.get("loss", {}), **cfg.get("data", {})}
    if extra:
        config.update(extra)
    return wandb.init(project=WANDB_PROJECT, name=run_name, config=config)


def _ensure_clay_ckpt():
    """Download the Clay v1.5 checkpoint from HuggingFace to the volume if it's
    not already there. Runs container-side, so the (~1.5 GB) file never touches
    the local machine."""
    import os

    if os.path.exists(CLAY_CKPT):
        return
    import requests

    print(f"[clay] checkpoint missing — fetching from HuggingFace -> {CLAY_CKPT}")
    os.makedirs(os.path.dirname(CLAY_CKPT), exist_ok=True)
    tmp = CLAY_CKPT + ".part"
    with requests.get(HF_CLAY_URL, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    os.replace(tmp, CLAY_CKPT)
    ckpt_vol.commit()
    print(f"[clay] checkpoint ready ({os.path.getsize(CLAY_CKPT) / 1e9:.2f} GB)")


@app.function(image=image, volumes={CKPT_DIR: ckpt_vol}, timeout=60 * 60)
def fetch_clay_ckpt():
    """Standalone: pull the Clay checkpoint to the volume (run once if you like)."""
    _ensure_clay_ckpt()


def _setup(config_path: str):
    """Container-side setup: put src on the path, seed, load config + model."""
    import sys

    import torch
    import yaml

    sys.path.insert(0, "/root")
    torch.manual_seed(SEED)

    from src.model.clay_loader import load_clay_encoder
    from src.model.detector import TreeDetector

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    encoder = load_clay_encoder(CLAY_CKPT)
    model = TreeDetector(
        encoder,
        neck_in_channels=m["neck_in_channels"],
        neck_out_channels=m["neck_out_channels"],
        num_queries=m["num_queries"],
        hidden=m["hidden"],
        n_heads=m["n_heads"],
        n_decoder_layers=m["n_decoder_layers"],
        dropout=m["dropout"],
    )
    return model, cfg


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
@app.function(**COMMON)
def stage_b(
    data_source: str = "neon",
    stage_b_annotations: str = "/data/deepforest.json",
    labels_csv: str = "/neon/patches/labels.csv",
    patches_root: str = "/neon/patches",
    config: str = "/root/configs/stage_b.yaml",
):
    _ensure_clay_ckpt()
    model, cfg = _setup(config)
    from src.train.stage_b import run_stage_b

    t, l, d = cfg["training"], cfg["loss"], cfg.get("data", {})

    # Default: 4-band NEON patches (mot volume) produced by modal_pipeline.py.
    # Pass data_source="deepforest" to fall back to the RGB JSON on clay-data.
    dataset = collate_fn = None
    if data_source == "neon":
        from src.data.neon_dataset import NeonPatchDataset
        from src.data.deepforest_dataset import deepforest_collate_fn
        dataset = NeonPatchDataset(
            labels_csv=labels_csv,
            patches_root=patches_root,
            tile_size=d.get("tile_size", 256),
            augment=d.get("augment", True),
        )
        collate_fn = deepforest_collate_fn
        print(f"[Stage B] NEON dataset: {len(dataset)} patches from {labels_csv}")

    run = _wandb_init(
        "stage_b", cfg,
        extra={"stage": "B", "data_source": data_source,
               "n_patches": len(dataset) if dataset is not None else None},
    )
    try:
        run_stage_b(
            model=model,
            annotations_path=stage_b_annotations,
            dataset=dataset,
            collate_fn=collate_fn,
            checkpoint_dir=CKPT_DIR,
            total_epochs=t["total_epochs"],
            batch_size=t["batch_size"],
            lr=t["lr"],
            weight_decay=t["weight_decay"],
            warmup_epochs=t["warmup_epochs"],
            val_fraction=t["val_fraction"],
            lam_cls=l["lam_cls"],
            lam_l1=l["lam_l1"],
            lam_giou=l["lam_giou"],
            device="cuda",
            num_workers=t["num_workers"],
        )
    finally:
        if run is not None:
            run.finish()
    ckpt_vol.commit()
    print("Stage B complete. Checkpoints written to clay-checkpoints volume.")


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------
@app.function(**COMMON)
def stage_c(
    train_annotations: str = "/data/naip_train.json",
    val_annotations:   str = "/data/naip_val.json",
    stage_b_checkpoint: str = "/checkpoints/stage_b_best.pt",
    config: str = "/root/configs/stage_c.yaml",
):
    _ensure_clay_ckpt()
    model, cfg = _setup(config)
    from src.train.stage_c import run_stage_c

    t, l = cfg["training"], cfg["loss"]
    # configs store a repo-relative norm-stats path; map it to the mounted location.
    norm_stats = cfg.get("data", {}).get(
        "norm_stats_path", "configs/naip_normalization.yaml"
    )
    if not norm_stats.startswith("/"):
        norm_stats = "/root/" + norm_stats

    run = _wandb_init("stage_c", cfg, extra={"stage": "C"})
    try:
        run_stage_c(
            model=model,
            train_annotations_path=train_annotations,
            val_annotations_path=val_annotations,
            checkpoint_dir=CKPT_DIR,
            norm_stats_path=norm_stats,
            total_epochs=t["total_epochs"],
            frozen_epochs=t["frozen_epochs"],
            batch_size=t["batch_size"],
            neck_head_lr=t["neck_head_lr"],
            encoder_lr=t["encoder_lr"],
            weight_decay=t["weight_decay"],
            warmup_epochs=t["warmup_epochs"],
            unfreeze_n_blocks=t["unfreeze_n_blocks"],
            lam_cls=l["lam_cls"],
            lam_l1=l["lam_l1"],
            lam_giou=l["lam_giou"],
            device="cuda",
            num_workers=t["num_workers"],
            stage_b_checkpoint=stage_b_checkpoint,
        )
    finally:
        if run is not None:
            run.finish()
    ckpt_vol.commit()
    print("Stage C complete. Checkpoints written to clay-checkpoints volume.")


# ---------------------------------------------------------------------------
# Local entrypoint: run both stages in sequence
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    stage: str = "both",
    data_source: str = "neon",
    stage_b_annotations: str = "/data/deepforest.json",
    train_annotations:   str = "/data/naip_train.json",
    val_annotations:     str = "/data/naip_val.json",
):
    if stage in ("b", "both"):
        print(f"Launching Stage B (data_source={data_source})...")
        stage_b.remote(
            data_source=data_source,
            stage_b_annotations=stage_b_annotations,
        )

    if stage in ("c", "both"):
        print("Launching Stage C...")
        stage_c.remote(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
        )
