"""Modal training script for Clay tree detector.

Volumes (create once with `modal volume create`):
  clay-data         — upload annotation JSONs and GeoTIFF tiles here
  clay-checkpoints  — receives model checkpoints; also store clay-v1.5.ckpt here

Upload data:
  modal volume put clay-data  data/naip_train.json        /naip_train.json
  modal volume put clay-data  data/naip_val.json          /naip_val.json
  modal volume put clay-data  data/deepforest.json        /deepforest.json
  modal volume put clay-data  tiles/                      /tiles/

Upload Clay checkpoint:
  modal volume put clay-checkpoints  clay-v1.5.ckpt  /clay-v1.5.ckpt

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
    )
    .pip_install("git+https://github.com/Clay-foundation/model.git")
    # Reassert the CUDA torch build in case the Clay install pulled a CPU wheel.
    .pip_install("torch==2.3.0", "torchvision==0.18.0", index_url=TORCH_INDEX)
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

DATA_DIR  = "/data"
CKPT_DIR  = "/checkpoints"
CLAY_CKPT = "/checkpoints/clay-v1.5.ckpt"
SEED      = 42

COMMON = dict(
    gpu="A10G",
    timeout=60 * 60 * 6,       # 6 hours max
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
)


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
    stage_b_annotations: str = "/data/deepforest.json",
    config: str = "/root/configs/stage_b.yaml",
):
    model, cfg = _setup(config)
    from src.train.stage_b import run_stage_b

    t, l = cfg["training"], cfg["loss"]
    run_stage_b(
        model=model,
        annotations_path=stage_b_annotations,
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
    model, cfg = _setup(config)
    from src.train.stage_c import run_stage_c

    t, l = cfg["training"], cfg["loss"]
    # configs store a repo-relative norm-stats path; map it to the mounted location.
    norm_stats = cfg.get("data", {}).get(
        "norm_stats_path", "configs/naip_normalization.yaml"
    )
    if not norm_stats.startswith("/"):
        norm_stats = "/root/" + norm_stats

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
    ckpt_vol.commit()
    print("Stage C complete. Checkpoints written to clay-checkpoints volume.")


# ---------------------------------------------------------------------------
# Local entrypoint: run both stages in sequence
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    stage: str = "both",
    stage_b_annotations: str = "/data/deepforest.json",
    train_annotations:   str = "/data/naip_train.json",
    val_annotations:     str = "/data/naip_val.json",
):
    if stage in ("b", "both"):
        print("Launching Stage B...")
        stage_b.remote(stage_b_annotations=stage_b_annotations)

    if stage in ("c", "both"):
        print("Launching Stage C...")
        stage_c.remote(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
        )
