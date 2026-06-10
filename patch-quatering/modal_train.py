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
  modal run modal_train.py::stage_c_quarter   # quartered fine-tune of stage_b_best.pt on
                                              # /data/stage_c/{train,val,test}.json (128-px quarters)

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
    cpu=8.0,                    # feed the 16 dataloader workers (I/O-bound on the volume)
    timeout=60 * 60 * 24,
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


def _localize_packed(packed_dir):
    """Copy packed {SITE}.npy + packed_index.csv from the network volume to local
    container disk. Training shuffles, so reads are random-access; from the volume
    that means a network page-fault per access (GPU starves at ~50% even after
    packing). Copying once up front (sequential, fast) makes every epoch's reads
    hit local SSD / page cache instead — the difference between data-bound and
    GPU-bound. Returns the local dir (or the input unchanged if packed_dir is None)."""
    import os
    import shutil
    if not packed_dir:
        return packed_dir
    local = "/tmp/packed"
    os.makedirs(local, exist_ok=True)
    files = [f for f in sorted(os.listdir(packed_dir)) if f.endswith((".npy", ".csv"))]
    for f in files:
        dst = os.path.join(local, f)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(packed_dir, f), dst)
    print(f"[setup] localized {len(files)} packed files {packed_dir} -> {local}", flush=True)
    return local


def _setup(config_path: str, num_queries: int = 0):
    """Container-side setup: put src on the path, seed, load config + model.

    num_queries > 0 overrides model.num_queries from the YAML (so you can match
    it to a dataset's --max-trees from the CLI without editing the config)."""
    import sys

    import torch
    import yaml

    sys.path.insert(0, "/root")
    torch.manual_seed(SEED)

    from src.model.clay_loader import load_clay_encoder
    from src.model.detector import TreeDetector

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if num_queries:
        print(f"[setup] num_queries override: {cfg['model']['num_queries']} -> {num_queries}")
        cfg["model"]["num_queries"] = num_queries

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
    out_tag: str = "patches",
    num_queries: int = 0,
    labels_csv: str = "",
    patches_root: str = "",
    stage_b_annotations: str = "/data/deepforest.json",
    config: str = "/root/configs/stage_b.yaml",
):
    """out_tag: which pipeline output dir on the `mot` volume to train on
    (mounted at /neon/{out_tag}); mirrors modal_pipeline.py --out-tag.
    num_queries: override model.num_queries (set it to that run's --max-trees)."""
    _ensure_clay_ckpt()
    model, cfg = _setup(config, num_queries)
    from src.train.stage_b import run_stage_b

    t, l, d = cfg["training"], cfg["loss"], cfg.get("data", {})

    # Derive the data paths from out_tag unless explicitly overridden. In
    # prequartered mode the pipeline writes labels_quarter.csv (128 tiles) +
    # labels_parent.csv (full 256 GT for stitched eval).
    prequartered = d.get("prequartered", False)
    if prequartered:
        labels_csv = labels_csv or f"/neon/{out_tag}/labels_quarter.csv"
        parent_labels_csv = f"/neon/{out_tag}/labels_parent.csv"
    else:
        labels_csv = labels_csv or f"/neon/{out_tag}/labels.csv"
        parent_labels_csv = None
    patches_root = patches_root or f"/neon/{out_tag}"

    # Default: 4-band NEON patches (mot volume) produced by modal_pipeline.py.
    # Pass data_source="deepforest" to fall back to the RGB JSON on clay-data.
    dataset = collate_fn = None
    if data_source == "neon":
        from src.data.neon_dataset import NeonPatchDataset
        from src.data.deepforest_dataset import deepforest_collate_fn
        packed_dir = f"/neon/{out_tag}/packed" if (prequartered and d.get("packed", False)) else None
        packed_dir = _localize_packed(packed_dir)   # copy to local disk -> fast random reads
        dataset = NeonPatchDataset(
            labels_csv=labels_csv,
            patches_root=patches_root,
            tile_size=d.get("tile_size", 256),
            augment=d.get("augment", True),
            quarter=d.get("quarter", False),
            n_split=d.get("n_split", 2),
            max_trees=d.get("max_trees", 0),
            parent_labels_csv=parent_labels_csv,
            parent_size=d.get("parent_size", 256),
            packed_dir=packed_dir,
        )
        collate_fn = deepforest_collate_fn
        print(f"[Stage B] NEON dataset: {len(dataset)} tiles from {labels_csv}"
              + (f" (packed: {packed_dir})" if packed_dir else ""))

        # Q must be >= the densest tile or Hungarian matching silently drops GT
        # (spec gotcha #12). Warn loudly if the data outgrew num_queries.
        q = cfg["model"]["num_queries"]
        max_boxes = max((len(s["boxes"]) for s in dataset.samples), default=0)
        if max_boxes > q:
            print(f"[Stage B] *** WARNING: densest tile has {max_boxes} boxes > "
                  f"num_queries={q}; GT will be dropped. Re-run with "
                  f"--num-queries {max_boxes} (and match Stage C). ***")
        else:
            print(f"[Stage B] densest tile = {max_boxes} boxes <= num_queries={q} (ok)")

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
            on_checkpoint=ckpt_vol.commit,   # persist each new best immediately
        )
    finally:
        if run is not None:
            run.finish()
    ckpt_vol.commit()
    print("Stage B complete. Checkpoints written to clay-checkpoints volume.")


# ---------------------------------------------------------------------------
# Stage B evaluation — stitch quarter predictions back to the parent (256) frame
# ---------------------------------------------------------------------------
@app.function(**COMMON)
def eval_stage_b(
    checkpoint: str = "/checkpoints/stage_b_best.pt",
    out_tag: str = "patches_quarter",
    num_queries: int = 0,
    labels_csv: str = "",
    patches_root: str = "",
    config: str = "/root/configs/stage_b.yaml",
    split: str = "val",            # "val" (held-out parents) or "all"
    conf_thresh: float = 0.5,
    nms_iou: float = 0.6,
):
    """Parent-level (256-frame) metrics for a trained Stage-B model. Runs the
    model on quarter tiles, stitches predictions back into the parent frame,
    and scores against the full 256 GT — directly comparable to an un-quartered
    baseline. Reads quartering settings (prequartered/tile_size) from the config,
    and evaluates on the same held-out val parents the training used.

    For a rigorous held-out number, point --labels-csv at a separate test CSV
    and pass --split all. num_queries must match the trained checkpoint."""
    import json

    import torch

    model, cfg = _setup(config, num_queries)
    state = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict(state["model"])
    model = model.to("cuda")

    from src.data.neon_dataset import NeonPatchDataset
    from src.eval.stitch import stitch_eval
    from src.train.stage_b import _split_train_val

    d = cfg.get("data", {})
    prequartered = d.get("prequartered", False)
    if prequartered:
        labels_csv = labels_csv or f"/neon/{out_tag}/labels_quarter.csv"
        parent_labels_csv = f"/neon/{out_tag}/labels_parent.csv"
    else:
        labels_csv = labels_csv or f"/neon/{out_tag}/labels.csv"
        parent_labels_csv = None
    patches_root = patches_root or f"/neon/{out_tag}"

    # max_trees=0: score every available tile (quarters were already capped at
    # write time, so dense quarters' GT counts as misses against the full 256 GT).
    packed_dir = f"/neon/{out_tag}/packed" if (prequartered and d.get("packed", False)) else None
    packed_dir = _localize_packed(packed_dir)
    dataset = NeonPatchDataset(
        labels_csv=labels_csv, patches_root=patches_root,
        tile_size=d.get("tile_size", 256), augment=False,
        quarter=d.get("quarter", False), n_split=d.get("n_split", 2),
        max_trees=0, parent_labels_csv=parent_labels_csv,
        parent_size=d.get("parent_size", 256), packed_dir=packed_dir,
    )
    if split == "val":
        _, val = _split_train_val(dataset, cfg["training"]["val_fraction"], seed=42)
        val_parents = {dataset.samples[i]["parent"] for i in val.indices}
        dataset.samples = [s for s in dataset.samples if s["parent"] in val_parents]

    n_parents = len({s["parent"] for s in dataset.samples})
    print(f"[eval/{split}] {len(dataset)} tiles over {n_parents} parents")
    res = stitch_eval(
        model, dataset, device="cuda",
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        conf_thresh=conf_thresh, nms_iou=nms_iou,
    )
    print("[eval] parent-level metrics:\n" + json.dumps(res, indent=2, default=float))
    return res


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------
@app.function(**COMMON)
def stage_c(
    train_annotations: str = "/data/naip_train.json",
    val_annotations:   str = "/data/naip_val.json",
    stage_b_checkpoint: str = "/checkpoints/stage_b_best.pt",
    num_queries: int = 0,
    config: str = "/root/configs/stage_c.yaml",
):
    """num_queries must MATCH the Stage B run (the query embedding is part of the
    checkpoint being loaded), so pass the same value you used for Stage B."""
    _ensure_clay_ckpt()
    model, cfg = _setup(config, num_queries)
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
            on_checkpoint=ckpt_vol.commit,   # persist each new best immediately
        )
    finally:
        if run is not None:
            run.finish()
    ckpt_vol.commit()
    print("Stage C complete. Checkpoints written to clay-checkpoints volume.")


# ---------------------------------------------------------------------------
# Stage C (quartered): fine-tune the 128-px quarter detector on NAIP JSON tiles
# ---------------------------------------------------------------------------
@app.function(**COMMON)
def stage_c_quarter(
    train_annotations: str = "/data/stage_c/train.json",
    val_annotations:   str = "/data/stage_c/val.json",
    test_annotations:  str = "/data/stage_c/test.json",
    stage_b_checkpoint: str = "/checkpoints/stage_b_best.pt",
    num_queries: int = 0,
    config: str = "/root/configs/stage_c_quarter.yaml",
):
    """Fine-tune the quarter (128-px) detector from `stage_b_checkpoint` on the
    stage_c_data JSON splits. Every 256 tile is split into 2x2 128-px quarters for
    training; validation (each epoch) and the final test are scored on predictions
    stitched back to the 256 frame, so the F1 is comparable to the un-quartered
    baseline. num_queries defaults to the checkpoint's own query budget (the quarter
    Stage B used 150) — the query embedding is part of the saved weights, so it must
    match."""
    import torch

    _ensure_clay_ckpt()

    # Match the model's query budget to the checkpoint unless explicitly overridden.
    if not num_queries:
        sd = torch.load(stage_b_checkpoint, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        num_queries = int(sd["head.queries.weight"].shape[0])
        print(f"[Stage C-q] num_queries from checkpoint: {num_queries}")

    model, cfg = _setup(config, num_queries)
    from src.train.stage_c import run_stage_c_quartered

    t, l = cfg["training"], cfg["loss"]
    d, e = cfg.get("data", {}), cfg.get("eval", {})
    norm_stats = d.get("norm_stats_path", "configs/naip_normalization.yaml")
    if not norm_stats.startswith("/"):
        norm_stats = "/root/" + norm_stats

    run = _wandb_init(
        "stage_c_quarter", cfg,
        extra={"stage": "C-quarter", "resume_from": stage_b_checkpoint,
               "num_queries": num_queries},
    )
    try:
        res = run_stage_c_quartered(
            model=model,
            train_annotations_path=train_annotations,
            val_annotations_path=val_annotations,
            test_annotations_path=test_annotations,
            checkpoint_dir=CKPT_DIR,
            norm_stats_path=norm_stats,
            tile_size=d.get("tile_size", 256),
            n_split=d.get("n_split", 2),
            total_epochs=t["total_epochs"],
            frozen_epochs=t["frozen_epochs"],
            batch_size=t["batch_size"],
            neck_head_lr=t["neck_head_lr"],
            encoder_lr=t["encoder_lr"],
            weight_decay=t["weight_decay"],
            warmup_epochs=t["warmup_epochs"],
            unfreeze_n_blocks=t["unfreeze_n_blocks"],
            patience=t.get("patience", 15),
            lam_cls=l["lam_cls"],
            lam_l1=l["lam_l1"],
            lam_giou=l["lam_giou"],
            conf_thresh=e.get("conf_thresh", 0.5),
            nms_iou=e.get("nms_iou", 0.6),
            device="cuda",
            num_workers=t["num_workers"],
            stage_b_checkpoint=stage_b_checkpoint,
            on_checkpoint=ckpt_vol.commit,   # persist each new best immediately
        )
    finally:
        if run is not None:
            run.finish()
    ckpt_vol.commit()
    print("Stage C (quartered) complete. Wrote stage_c_quarter_best.pt / "
          "stage_c_quarter_final.pt to clay-checkpoints volume.")
    return res


# ---------------------------------------------------------------------------
# Local entrypoint: run both stages in sequence
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    stage: str = "both",
    data_source: str = "neon",
    out_tag: str = "patches",
    num_queries: int = 0,
    stage_b_annotations: str = "/data/deepforest.json",
    train_annotations:   str = "/data/naip_train.json",
    val_annotations:     str = "/data/naip_val.json",
):
    """out_tag selects the Stage B data dir (/neon/{out_tag}); num_queries
    overrides model.num_queries for BOTH stages (keep them equal)."""
    if stage in ("b", "both"):
        print(f"Launching Stage B (data_source={data_source}, out_tag={out_tag})...")
        stage_b.remote(
            data_source=data_source,
            out_tag=out_tag,
            num_queries=num_queries,
            stage_b_annotations=stage_b_annotations,
        )

    if stage in ("c", "both"):
        print("Launching Stage C...")
        stage_c.remote(
            train_annotations=train_annotations,
            val_annotations=val_annotations,
            num_queries=num_queries,
        )
