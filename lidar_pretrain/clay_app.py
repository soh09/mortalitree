"""
Clay DQ-DETR with an intermediate LiDAR stage, on Modal (see README.md, "Clay arms").

  Clay-A  Stage B checkpoint -> Stage C (hand labels)                 the current Clay model
  Clay-B  Stage B checkpoint -> LiDAR stage -> Stage C                the new arm

Stage B is the existing DQ-DETR checkpoint (`dq-detr_final.pt` on volume
`clay-checkpoints`; frozen Clay v1.5 encoder, neck + head trained on 4-band NEON
patches). The LiDAR stage is `stage_b_finetune`'s recipe (warm start, flat lr 5e-4,
CGFE on, encoder frozen, 10 epochs) on the 3DEP LiDAR crowns drawn on pre-fire 4-band
NAIP (build_pretrain.py --clay-dir; the same 400-chips-per-fire set as the DeepForest
arms). Stage C is `stage_c(model="dqdetr")` unchanged: 80 epochs, encoder unfrozen
(last 2 blocks) after 30, early stopping on val F1, best checkpoint kept.
Training code is the untracked `dq-detr-impl/` (found next to the repo, or via
$DQDETR_DIR), used as-is.

Scoring matches the DeepForest arms (lidar_pretrain/train.py): loose-threshold
predictions on val/test, score threshold + NMS chosen on val (best F1 at IoU 0.4;
"NMS 1.0" = none, as DETR is normally run), test scored once; results in
lidar-pretrain-work:/work/runs_clay/seed{S}/{ClayA,ClayB}/ (metrics.json, raw
predictions), wandb project mortalitree-lidar-pretrain, groups clay-seed{S}.

    modal deploy lidar_pretrain/clay_app.py
    python lidar_pretrain/clay_app.py --stage build,train      # headless; spawns `pipeline`
    python lidar_pretrain/clay_app.py --stage train --smoke     # 1-epoch check (runs_clay_smoke/)
    python lidar_pretrain/clay_app.py --stage build_full,train --runs runs_clay_full \
           --arms ClayA,ClayB_full10,ClayB_full1                # every eligible chip (8 build shards)

Every Stage C checkpoint (each 5 epochs, best, final) is kept, slimmed to the weights that changed,
under RUNS/seed{S}/{arm}/checkpoints/, with predictions on all 108 tiles in
raw_predictions_all_ckpts.csv.gz -- revalidate.py re-selects epoch + thresholds for a new split.
"""
import json, os, random, shutil, subprocess, sys, threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _find_dqdetr():
    if os.environ.get("DQDETR_DIR"):
        return Path(os.environ["DQDETR_DIR"])
    if (REPO / "dq-detr-impl").exists():
        return REPO / "dq-detr-impl"
    try:   # a git worktree: look in the main checkout
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
        main = Path(out.split("\n")[0].split(" ", 1)[1])
        if (main / "dq-detr-impl").exists():
            return main / "dq-detr-impl"
    except Exception:
        pass
    return None


DQ = _find_dqdetr()
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
WORK, CKPT, DATA = "/work", "/ckpt", "/data"
CLAY_CKPT = f"{CKPT}/clay-v1.5.ckpt"
STAGE_B = f"{CKPT}/dq-detr_final.pt"
WANDB_PROJECT = "mortalitree-lidar-pretrain"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", index_url=TORCH_INDEX)
    .pip_install("scipy", "rasterio", "pyyaml", "matplotlib", "tqdm", "wandb", "requests", "pandas", "pillow",
                 "opencv-python-headless", "pyproj", "shapely", "scikit-image", "affine")
    .pip_install("git+https://github.com/Clay-foundation/model.git")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", index_url=TORCH_INDEX)   # keep the CUDA build
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, "/repo/lidar_pretrain", ignore=["__pycache__", "lightning_logs"])
    .add_local_file(REPO / "baselines" / "chm_crowns.py", "/repo/baselines/chm_crowns.py")
    .add_local_dir(REPO / "single_tiles_flat" / "stage_c_data", "/repo/single_tiles_flat/stage_c_data",
                   ignore=["viewer", "readable", "predictions_all.json", "predictions_lidar_best.json"])
)
if DQ is not None:
    image = (image.add_local_dir(DQ / "src", "/root/src", ignore=["__pycache__"])
                  .add_local_dir(DQ / "configs", "/root/configs"))

app = modal.App("lidar-clay", image=image)
work_vol = modal.Volume.from_name("lidar-pretrain-work")
ckpt_vol = modal.Volume.from_name("clay-checkpoints")
data_vol = modal.Volume.from_name("clay-data")
GPU = dict(gpu="A10G", cpu=8, memory=32768, timeout=20 * 3600,
           volumes={WORK: work_vol, CKPT: ckpt_vol, DATA: data_vol},
           secrets=[modal.Secret.from_name("wandb")],
           retries=modal.Retries(max_retries=1, initial_delay=30.0))


def committer(stop, every=300):
    while not stop.wait(every):
        try:
            work_vol.commit()
        except Exception as e:
            print("commit failed:", e)


# ---------------- data ----------------
@app.function(cpu=8, memory=16384, timeout=6 * 3600, volumes={WORK: work_vol})
def build(max_chips: int = 400, workers: int = 24, out: str = "pretrain_clay"):
    """Same chip selection as the 400-per-fire DeepForest set, plus raw 4-band windows."""
    import tarfile
    work_vol.reload()
    d = Path(WORK, out)
    if d.exists():
        shutil.rmtree(d)
    subprocess.run([sys.executable, "build_pretrain.py", WORK, "--max-chips", str(max_chips), "--workers", str(workers),
                    "--out", out, "--images-dir", "/tmp/png", "--clay-dir", "/tmp/clay"],
                   check=True, cwd="/repo/lidar_pretrain")
    with tarfile.open("/tmp/clay_tiles.tar", "w") as tf:
        tf.add("/tmp/clay", arcname="tiles")
    shutil.copy("/tmp/clay_tiles.tar", d / "clay_tiles.tar")
    work_vol.commit()
    import pandas as pd
    lab = pd.read_csv(d / "labels_clay.csv")
    per = lab.groupby("imgname").size()
    return {"windows": int(per.size), "boxes": int(len(lab)), "max_boxes": int(per.max()), "mean_boxes": float(per.mean()),
            "by_fire": lab.groupby("site").imgname.nunique().to_dict()}


@app.function(cpu=8, memory=16384, timeout=8 * 3600, volumes={WORK: work_vol})
def build_shard(shard: int, num_shards: int, out: str = "pretrain_clay_full", max_chips: int = 0, workers: int = 16):
    """One shard of every eligible chip (as the DeepForest full set), 4-band windows -> one tar per shard."""
    import tarfile
    work_vol.reload()
    d = Path(WORK, out, f"shard{shard}")
    if d.exists():
        shutil.rmtree(d)
    subprocess.run([sys.executable, "build_pretrain.py", WORK, "--max-chips", str(max_chips), "--workers", str(workers),
                    "--out", out, "--shard", str(shard), "--num-shards", str(num_shards), "--images-dir", "/tmp/png",
                    "--clay-dir", "/tmp/clay"], check=True, cwd="/repo/lidar_pretrain")
    with tarfile.open("/tmp/clay_tiles.tar", "w") as tf:
        tf.add("/tmp/clay", arcname="tiles")
    shutil.copy("/tmp/clay_tiles.tar", Path(WORK, out, f"clay_tiles_shard{shard}.tar"))
    work_vol.commit()
    return {"shard": shard}


@app.function(cpu=2, memory=16384, volumes={WORK: work_vol})
def merge_shards(out: str, num_shards: int):
    import pandas as pd
    work_vol.reload()
    root = Path(WORK, out)
    lab = pd.concat([pd.read_csv(root / f"shard{k}" / "labels_clay.csv") for k in range(num_shards)])
    lab.to_csv(root / "labels_clay.csv", index=False); work_vol.commit()
    per = lab.groupby("imgname").size()
    return {"windows": int(per.size), "boxes": int(len(lab)), "max_boxes": int(per.max()),
            "mean_boxes": round(float(per.mean()), 1), "by_fire": lab.groupby("site").imgname.nunique().to_dict()}


# ---------------- model ----------------
def _cfg(name):
    import yaml
    with open(f"/root/configs/{name}.yaml") as f:
        return yaml.safe_load(f)


def _model(cfg):
    sys.path.insert(0, "/root")
    from src.model.clay_loader import load_clay_encoder
    from src.model.detector import TreeDetector
    m = cfg["model"]
    return TreeDetector(load_clay_encoder(CLAY_CKPT), neck_in_channels=m["neck_in_channels"],
                        neck_out_channels=m["neck_out_channels"], hidden=m["hidden"], n_heads=m["n_heads"],
                        n_decoder_layers=m["n_decoder_layers"], dropout=m["dropout"],
                        dynamic_query_list=m.get("dynamic_query_list", (200, 400, 600)),
                        ccm_cls_num=m.get("ccm_cls_num", 3), ccm_params=m.get("ccm_params", (100, 300)),
                        anchor_size=m.get("anchor_size", 0.05))


def _seed(s):
    import numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _wandb(name, group, config):
    import wandb
    return wandb.init(project=WANDB_PROJECT, name=name, group=group, job_type=name.rsplit("-", 1)[-1],
                      config=config, reinit=True)


def _need(path, what):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found at {path}")   # never silently train from scratch
    return path


def _stage_json(split, root):
    """Stage C split JSONs pointing at the tiles shipped in the image (not a volume)."""
    items = json.loads(Path(f"/repo/single_tiles_flat/stage_c_data/{split}.json").read_text())
    for it in items:
        it["tile_path"] = f"/repo/single_tiles_flat/stage_c_data/tiles/{Path(it['tile_path']).name}"
    p = Path(root) / f"{split}.json"; p.write_text(json.dumps(items)); return str(p)


# arm -> (LiDAR set on the volume, LiDAR-stage epochs); ClayA has no LiDAR stage
ARM_CFG = {"ClayA": None, "ClayB": ("pretrain_clay", 10),
           "ClayB_full10": ("pretrain_clay_full", 10), "ClayB_full1": ("pretrain_clay_full", 1)}


def lidar_stage(seed, out_dir, epochs, lr, smoke, data="pretrain_clay"):
    """stage_b_finetune's recipe on the LiDAR windows; resumes from lidar_last.pt; returns lidar_final.pt."""
    import tarfile
    sys.path.insert(0, "/root")
    from src.data.neon_dataset import NeonPatchDataset
    from src.data.deepforest_dataset import deepforest_collate_fn
    from src.train.stage_b import run_stage_b
    src = Path(WORK, data)
    local = Path("/tmp", data)
    if not local.exists():
        tars = sorted(src.glob("clay_tiles*.tar"))
        if not tars:
            _need(src / "clay_tiles.tar", "LiDAR 4-band tiles")
        for t in tars:
            with tarfile.open(t) as tf:
                tf.extractall(local)
    labels = str(_need(src / "labels_clay.csv", "LiDAR labels"))
    if smoke:
        import pandas as pd
        d = pd.read_csv(labels); keep = d.imgname.drop_duplicates().iloc[:64]
        labels = "/tmp/labels_smoke.csv"; d[d.imgname.isin(keep)].to_csv(labels, index=False)
    ds = NeonPatchDataset(labels_csv=labels, patches_root=str(local / "tiles"), tile_size=256, augment=True)
    print(f"[LiDAR stage] {len(ds)} windows from {data}, {epochs} epochs", flush=True)
    cfg = _cfg("stage_b"); t, l = cfg["training"], cfg["loss"]
    model = _model(cfg)
    last = Path(out_dir) / "lidar_last.pt"
    run_stage_b(model=model, dataset=ds, collate_fn=deepforest_collate_fn, checkpoint_dir=str(out_dir),
                total_epochs=epochs, batch_size=t["batch_size"], lr=lr, weight_decay=t["weight_decay"],
                warmup_epochs=0, val_fraction=t["val_fraction"], lam_cls=l["lam_cls"], lam_l1=l["lam_l1"],
                lam_giou=l["lam_giou"], lam_ccm=l.get("lam_ccm", 1.0), lam_enc=l.get("lam_enc", 1.0),
                cgfe_start_epoch=0, viz_every=0, ckpt_every=1,
                init_weights_from=None if last.exists() else _need(STAGE_B, "Stage B checkpoint"),
                resume_from=str(last) if last.exists() else None,
                flat_lr=True, ckpt_prefix="lidar", device="cuda", num_workers=t["num_workers"],
                on_checkpoint=work_vol.commit)
    return _need(Path(out_dir) / "lidar_final.pt", "LiDAR-stage checkpoint")


def stage_c(init_ckpt, local_dir, smoke, ckpt_every=5):
    """Stage C into a local folder (checkpoints are 1.2 GB before slimming).

    Early stopping is OFF: dq-detr-impl stops after 20 epochs without a new best val F1 at a fixed
    conf 0.5 / IoU 0.5, which for DQ-DETR is ~0-0.13 noise, so it fired at epochs 21-34 -- mostly
    before the encoder unfreezes at epoch 30 -- while val loss was still falling. All 80 epochs run;
    the checkpoint is chosen afterwards on val with the harness metric (see choose_and_score)."""
    print("[Stage C] early stopping off: all epochs, checkpoint chosen on val afterwards", flush=True)
    sys.path.insert(0, "/root")
    from src.train.stage_c_multi import run_stage_c_multi
    cfg = _cfg("stage_c"); t, l, ev = cfg["training"], cfg["loss"], cfg.get("eval", {})
    model = _model(cfg)
    base = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("encoder.")}
    js = {s: _stage_json(s, local_dir) for s in ("train", "val")}
    epochs, frozen = (2, 1) if smoke else (t["total_epochs"], t["frozen_epochs"])
    run_stage_c_multi(model=model, is_dqdetr=True, ckpt_prefix="dqdetr", train_annotations_path=js["train"],
                      val_annotations_path=js["val"], checkpoint_dir=str(local_dir),
                      norm_stats_path="/root/configs/naip_normalization.yaml", total_epochs=epochs, frozen_epochs=frozen,
                      batch_size=t["batch_size"], neck_head_lr=t["neck_head_lr"], encoder_lr=t["encoder_lr"],
                      weight_decay=t["weight_decay"], warmup_epochs=t["warmup_epochs"],
                      unfreeze_n_blocks=t["unfreeze_n_blocks"], lam_cls=l["lam_cls"], lam_l1=l["lam_l1"],
                      lam_giou=l["lam_giou"], lam_ccm=l.get("lam_ccm", 1.0), lam_enc=l.get("lam_enc", 1.0),
                      cgfe_start_epoch=0, device="cuda", num_workers=t["num_workers"], stage_b_checkpoint=init_ckpt,
                      viz_every=10, n_viz_tiles=5, eval_conf_thresh=ev.get("conf_thresh", 0.5),
                      eval_iou_thresh=ev.get("iou_thresh", 0.5), ckpt_every=ckpt_every, patience=10 ** 6,
                      on_checkpoint=None)
    _need(Path(local_dir) / "dqdetr_stage_c_best.pt", "Stage C best checkpoint")
    return model, base


def slim(state, base):
    """Drop encoder weights identical to the Clay v1.5 weights (frozen blocks); keep everything that changed.
    Load later with model.load_state_dict(slim, strict=False) on a freshly built model."""
    import torch
    return {k: v for k, v in state.items() if k not in base or not torch.equal(v.cpu(), base[k].cpu())}


def predict_all(model, tag):
    """Raw predictions (score >= 0.01) on all 108 hand-labelled tiles -> DataFrame with a split column."""
    import numpy as np, pandas as pd
    from torch.utils.data import DataLoader
    sys.path.insert(0, "/root")
    from src.data.naip_dataset import NAIPTileDataset, naip_collate_fn
    from src.eval.inference import predict_tile
    model.to("cuda").eval(); rows = []
    for split in ("train", "val", "test"):
        ds = NAIPTileDataset(_stage_json(split, "/tmp"), "/root/configs/naip_normalization.yaml", augment=False)
        for batch in DataLoader(ds, batch_size=8, shuffle=False, collate_fn=naip_collate_fn):
            bl, sl = predict_tile(model, batch, conf_thresh=0.01, device="cuda")
            for path, b, sc in zip(batch["tile_paths"], bl, sl):
                if not len(b):
                    continue
                xyxy = np.clip(np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2] * 256, 0, 256)
                rows += [{"ckpt": tag, "split": split, "image_path": f"{Path(path).stem}.png", "xmin": x0, "ymin": y0,
                          "xmax": x1, "ymax": y1, "score": float(s_)} for (x0, y0, x1, y1), s_ in zip(xyxy, sc)]
    return pd.DataFrame(rows)


def choose_and_score(allp, out_dir, arm, info):
    """Pick the kept checkpoint whose val F1 (IoU 0.4, val-selected threshold/NMS) is highest, then score it."""
    sys.path.insert(0, "/repo/lidar_pretrain")
    import numpy as np
    import train as T
    lab = Path("/tmp/labelled"); T.prep_labelled(lab)
    val_gt = T.gt_by_img(lab / "val.csv"); empty = (np.zeros((0, 4)), np.zeros(0)); per = {}
    for tag, d in allp[allp.split == "val"].groupby("ckpt"):
        raw = {i: (g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float), g.score.to_numpy(float))
               for i, g in d.groupby("image_path")}
        raw = {i: raw.get(i, empty) for i in val_gt}
        per[tag] = max(T.score(raw, val_gt, st, nt, [0.4])["iou0.4"]["f1"] for st in T.SCORES for nt in T.NMSS + [1.0])
    tag = max(per, key=per.get)
    print(f"  checkpoint chosen on val: {tag} (val F1@0.4 {per[tag]:.3f}); all: "
          + ", ".join(f"{k} {v:.3f}" for k, v in sorted(per.items())), flush=True)
    return score_raw(allp[allp.ckpt == tag], out_dir, arm, {**info, "chosen_checkpoint": tag, "val_f1_by_checkpoint": per})


def score_raw(df_all, out_dir, arm, info):
    """The DeepForest arms' scoring (train.py) on one checkpoint's val/test predictions."""
    import pandas as pd
    sys.path.insert(0, "/repo/lidar_pretrain")
    import train as T
    lab = Path("/tmp/labelled"); T.prep_labelled(lab)
    raw = {}
    for split in ("val", "test"):
        d = df_all[df_all.split == split]
        d.drop(columns=["ckpt", "split"]).to_csv(Path(out_dir) / f"raw_predictions_{split}.csv", index=False)
        imgs = pd.read_csv(lab / f"{split}.csv").image_path.unique()
        raw[split] = {i: (g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float), g.score.to_numpy(float))
                      for i, g in d.groupby("image_path")}
        for i in imgs:
            raw[split].setdefault(i, (__import__("numpy").zeros((0, 4)), __import__("numpy").zeros(0)))
    nmss = T.NMSS + [1.0]                      # 1.0 = no NMS (DETR's usual setting)
    val_gt = T.gt_by_img(lab / "val.csv")
    grid = [(st, nt, T.score(raw["val"], val_gt, st, nt, [0.4])["iou0.4"]["f1"]) for st in T.SCORES for nt in nmss]
    st, nt, vf1 = max(grid, key=lambda g: g[2])
    test_all, test_min5 = T.gt_by_img(lab / "test.csv"), T.gt_by_img(lab / "test.csv", 5)
    metrics = {"selected_on_val": {"score": st, "nms": nt, "val_f1_iou0.4": vf1},
               "test_all_tiles": T.score(raw["test"], test_all, st, nt),
               "test_min5": T.score(raw["test"], test_min5, st, nt),
               "test_fixed_ref_min5": T.score(raw["test"], test_min5, 0.5, 1.0)}   # Clay's own conf 0.5, no NMS
    rec = {"arm": arm, **info, "metrics": metrics}
    (Path(out_dir) / "metrics.json").write_text(json.dumps(rec, indent=1))
    return rec


def keep_checkpoints(model, base, local_dir, out_dir):
    """Every Stage C checkpoint (epochNNN, best, final): slimmed copy on the volume + predictions on all
    108 tiles, so best-epoch and threshold selection can be redone for any new val/test split (revalidate.py)."""
    import pandas as pd, torch
    ck = Path(out_dir) / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    preds, meta = [], {}
    for f in sorted(Path(local_dir).glob("dqdetr_stage_c_*.pt")):
        tag = f.stem.replace("dqdetr_stage_c_", "")
        sd = torch.load(f, map_location="cpu", weights_only=False)
        state = sd.get("model", sd); meta[tag] = {"epoch": sd.get("epoch"), "val_f1_conf0.5": sd.get("val_f1")}
        model.load_state_dict(state, strict=False)
        preds.append(predict_all(model, tag))
        torch.save({"epoch": sd.get("epoch"), "val_f1": sd.get("val_f1"), "model": slim(state, base),
                    "note": "slimmed: frozen Clay encoder weights omitted; load with strict=False on a fresh model"},
                   ck / f.name)
        print(f"  kept {f.name} (epoch {sd.get('epoch')}) + predictions", flush=True)
    allp = pd.concat(preds); allp.to_csv(Path(out_dir) / "raw_predictions_all_ckpts.csv.gz", index=False)
    (Path(out_dir) / "checkpoints.json").write_text(json.dumps(meta, indent=1))
    work_vol.commit()
    return allp


@app.function(**GPU)
def train_arm(seed: int, arm: str, smoke: bool = False, lidar_lr: float = 5e-4, runs: str = "runs_clay",
              data: str = "", reuse_lidar: str = ""):
    import wandb
    assert arm in ARM_CFG, arm
    work_vol.reload()
    if smoke:
        runs = runs + "_smoke"
    out = Path(WORK, runs, f"seed{seed}", arm); out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event(); threading.Thread(target=committer, args=(stop,), daemon=True).start()
    group = f"{'smoke-' if smoke else ''}{runs.replace('runs_', '')}-seed{seed}"
    lidar = ARM_CFG[arm]
    if lidar and data:                                    # override (smoke tests)
        lidar = (data, lidar[1])
    cfg = {"seed": seed, "arm": arm, "stage_b": STAGE_B, "lidar": lidar, "lidar_lr": lidar_lr, "smoke": smoke}
    short = group.replace("-seed", "-s")
    try:
        _need(CLAY_CKPT, "Clay v1.5 encoder"); _need(STAGE_B, "Stage B checkpoint")
        init = STAGE_B
        if lidar:
            lid = Path(WORK, runs, f"seed{seed}", f"lidar_{arm}")
            done = lid / "lidar_final.pt"
            if reuse_lidar:                                  # e.g. "runs_clay/seed{seed}/lidar_stage"
                done = Path(WORK, reuse_lidar.format(seed=seed), "lidar_final.pt")
                _need(done, "LiDAR-stage checkpoint to reuse")
            if done.exists():
                init = str(done); print(f"reusing {done}")
            else:
                _seed(seed)
                _wandb(f"{short}-{arm}-lidar", group, cfg)
                init = str(lidar_stage(seed, lid, 1 if smoke else lidar[1], lidar_lr, smoke, lidar[0]))
                wandb.finish()
        _seed(seed)
        run = _wandb(f"{short}-{arm}-stageC", group, cfg)
        local = Path("/tmp/stagec"); shutil.rmtree(local, ignore_errors=True); local.mkdir()
        model, base = stage_c(init, local, smoke)
        allp = keep_checkpoints(model, base, local, out)
        rec = choose_and_score(allp, out, arm, {"init_checkpoint": init, "seed": seed})
        m = rec["metrics"]
        run.summary.update({f"test/{k}/{kk}": vv for k, v in m["test_all_tiles"].items() if isinstance(v, dict)
                            for kk, vv in v.items()})
        run.summary.update({"test/count_mae": m["test_all_tiles"]["count_mae"],
                            **{f"val-selected/{k}": v for k, v in m["selected_on_val"].items()}})
        wandb.finish()
        return {k: m["test_all_tiles"][k]["f1"] for k in ("iou0.25", "iou0.4", "iou0.5")} | \
               {"mae": m["test_all_tiles"]["count_mae"], "val": m["selected_on_val"]}
    finally:
        stop.set(); work_vol.commit()


@app.function(cpu=8, memory=16384, timeout=3 * 3600, volumes={WORK: work_vol})
def revalidate(runs: str = "runs_clay", val: str = "", test: str = ""):
    """revalidate.py on the volume. val / test: comma-separated tile names (default: current split).
        modal run lidar_pretrain/clay_app.py::revalidate --runs runs_clay_full --val a,b,c --test d,e,f"""
    work_vol.reload()
    args = [sys.executable, "revalidate.py", WORK, runs]
    for flag, names in (("--val", val), ("--test", test)):
        if names:
            f = Path(f"/tmp/{flag[2:]}.json"); f.write_text(json.dumps([n.strip() for n in names.split(",") if n.strip()]))
            args += [flag, str(f)]
    r = subprocess.run(args, capture_output=True, text=True, cwd="/repo/lidar_pretrain")
    work_vol.commit()
    print(r.stdout[-6000:], r.stderr[-3000:])
    return r.stdout


@app.function(cpu=4, memory=8192, volumes={WORK: work_vol})
def summary(runs: str = "runs_clay"):
    work_vol.reload()
    a = subprocess.run([sys.executable, "summarize.py", WORK, runs], capture_output=True, text=True, cwd="/repo/lidar_pretrain")
    b = subprocess.run([sys.executable, "rescore_by_period.py", WORK, runs], capture_output=True, text=True,
                       cwd="/repo/lidar_pretrain")
    text = a.stdout + "\n\n# pre-/post-fire test split\n" + "\n".join(l for l in b.stdout.splitlines() if "Warning" not in l)
    Path(WORK, f"summary_{runs}.md").write_text(text); work_vol.commit()
    return text + a.stderr[-1500:] + b.stderr[-1500:]


STATUS = "pipeline_status_runs_clay.json"


def _status(**kw):
    p = Path(WORK, STATUS)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw); p.write_text(json.dumps(cur, indent=1)); work_vol.commit()
    print("status:", json.dumps(kw), flush=True)


@app.function(cpu=1, timeout=24 * 3600, volumes={WORK: work_vol})
def pipeline(stage: str = "build,train", seeds: str = "0,1,2", smoke: bool = False, arms: str = "ClayA,ClayB",
             runs: str = "runs_clay", reuse_lidar: str = ""):
    import datetime
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    S = set(stage.split(","))
    global STATUS
    STATUS = f"pipeline_status_{runs}.json"
    _status(started=now(), stage=stage, seeds=seeds, smoke=smoke, arms=arms, runs=runs, state="running", build=None,
            train=None, summary=None, error=None)
    try:
        if "build" in S:
            _status(build=build.remote())
        if "build_full" in S:
            n = 8
            list(build_shard.map(range(n), kwargs={"num_shards": n}))
            _status(build=merge_shards.remote("pretrain_clay_full", n))
        if "train" in S:
            jobs = [(int(s), arm) for s in seeds.split(",") for arm in arms.split(",")]
            done = {}
            for (s, arm), res in zip(jobs, train_arm.starmap(jobs, kwargs={"smoke": smoke, "runs": runs, "reuse_lidar": reuse_lidar},
                                                             return_exceptions=True)):
                done[f"seed{s}/{arm}"] = {"error": f"{type(res).__name__}: {res}"[:300]} \
                    if isinstance(res, BaseException) else res
                _status(train=done)
        if S & {"train", "summary"} and not smoke:
            _status(summary=summary.remote(runs))
        _status(state="done", finished=now())
    except Exception as e:
        _status(state="failed", error=f"{type(e).__name__}: {e}"[:500], finished=now())
        raise


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="spawn the Clay pipeline on the deployed app and exit")
    ap.add_argument("--stage", default="build,train"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="ClayA,ClayB", help=f"comma list of {list(ARM_CFG)}")
    ap.add_argument("--runs", default="runs_clay")
    ap.add_argument("--reuse-lidar", default="", help='e.g. "runs_clay/seed{seed}/lidar_stage"')
    a = ap.parse_args()
    call = modal.Function.from_name("lidar-clay", "pipeline").spawn(a.stage, a.seeds, a.smoke, a.arms, a.runs,
                                                                    a.reuse_lidar)
    print(f"spawned clay pipeline (stage={a.stage}, arms={a.arms}, runs={a.runs}, seeds={a.seeds}, smoke={a.smoke}): "
          f"call {call.object_id}")
