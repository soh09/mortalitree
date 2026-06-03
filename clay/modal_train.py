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
"""
import sys
import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "scipy",
        "rasterio",
        "pyyaml",
        "matplotlib",
        "tqdm",
    )
    .pip_install(
        "git+https://github.com/Clay-foundation/model.git",
    )
)

# ---------------------------------------------------------------------------
# App + volumes
# ---------------------------------------------------------------------------
app = modal.App("clay-tree-detector", image=image)

data_vol  = modal.Volume.from_name("clay-data",        create_if_missing=True)
ckpt_vol  = modal.Volume.from_name("clay-checkpoints", create_if_missing=True)

DATA_DIR  = "/data"
CKPT_DIR  = "/checkpoints"
CLAY_CKPT = "/checkpoints/clay-v1.5.ckpt"

# Mount local source and configs into the container
src_mount = modal.Mount.from_local_dir(
    "/Users/evanrobert/Documents/mortalitree/mortalitree/clay/src",
    remote_path="/root/src",
)
cfg_mount = modal.Mount.from_local_dir(
    "/Users/evanrobert/Documents/mortalitree/mortalitree/clay/configs",
    remote_path="/root/configs",
)

# ---------------------------------------------------------------------------
# Shared container setup
# ---------------------------------------------------------------------------
COMMON = dict(
    gpu="A10G",
    timeout=60 * 60 * 6,       # 6 hours max
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
    mounts=[src_mount, cfg_mount],
)


def _add_src_to_path():
    import sys, os
    sys.path.insert(0, "/root")


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
@app.function(**COMMON)
def stage_b(
    stage_b_annotations: str = "/data/deepforest.json",
    total_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    val_fraction: float = 0.1,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
    num_workers: int = 4,
):
    _add_src_to_path()
    import torch
    from src.model.clay_loader import load_clay_encoder
    from src.model.detector import TreeDetector
    from src.train.stage_b import run_stage_b

    encoder = load_clay_encoder(CLAY_CKPT)
    model = TreeDetector(encoder)

    run_stage_b(
        model=model,
        annotations_path=stage_b_annotations,
        checkpoint_dir=CKPT_DIR,
        total_epochs=total_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        val_fraction=val_fraction,
        lam_cls=lam_cls,
        lam_l1=lam_l1,
        lam_giou=lam_giou,
        device="cuda",
        num_workers=num_workers,
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
    norm_stats_path:   str = "/root/configs/naip_normalization.yaml",
    stage_b_checkpoint: str = "/checkpoints/stage_b_best.pt",
    total_epochs: int = 80,
    frozen_epochs: int = 30,
    unfreeze_n_blocks: int = 2,
    batch_size: int = 8,
    neck_head_lr: float = 5e-4,
    encoder_lr: float = 1e-5,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    lam_cls: float = 1.0,
    lam_l1: float = 5.0,
    lam_giou: float = 2.0,
    num_workers: int = 4,
):
    _add_src_to_path()
    import torch
    from src.model.clay_loader import load_clay_encoder
    from src.model.detector import TreeDetector
    from src.train.stage_c import run_stage_c

    encoder = load_clay_encoder(CLAY_CKPT)
    model = TreeDetector(encoder)

    run_stage_c(
        model=model,
        train_annotations_path=train_annotations,
        val_annotations_path=val_annotations,
        checkpoint_dir=CKPT_DIR,
        norm_stats_path=norm_stats_path,
        total_epochs=total_epochs,
        frozen_epochs=frozen_epochs,
        batch_size=batch_size,
        neck_head_lr=neck_head_lr,
        encoder_lr=encoder_lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        unfreeze_n_blocks=unfreeze_n_blocks,
        lam_cls=lam_cls,
        lam_l1=lam_l1,
        lam_giou=lam_giou,
        device="cuda",
        num_workers=num_workers,
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
