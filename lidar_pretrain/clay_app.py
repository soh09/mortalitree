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
ARMS = ("ClayA", "ClayB")

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


def lidar_stage(seed, out_dir, epochs, lr, smoke, data="pretrain_clay"):
    """stage_b_finetune's recipe on the LiDAR windows; returns the final checkpoint path."""
    import tarfile
    sys.path.insert(0, "/root")
    from src.data.neon_dataset import NeonPatchDataset
    from src.data.deepforest_dataset import deepforest_collate_fn
    from src.train.stage_b import run_stage_b
    src = Path(WORK, data)
    local = Path("/tmp/lidar_clay")
    if not local.exists():
        with tarfile.open(_need(src / "clay_tiles.tar", "LiDAR 4-band tiles")) as tf:
            tf.extractall(local)
    labels = str(src / "labels_clay.csv")
    if smoke:
        import pandas as pd
        d = pd.read_csv(labels); keep = d.imgname.drop_duplicates().iloc[:64]
        labels = "/tmp/labels_smoke.csv"; d[d.imgname.isin(keep)].to_csv(labels, index=False)
    ds = NeonPatchDataset(labels_csv=labels, patches_root=str(local / "tiles"), tile_size=256, augment=True)
    cfg = _cfg("stage_b"); t, l = cfg["training"], cfg["loss"]
    model = _model(cfg)
    run_stage_b(model=model, dataset=ds, collate_fn=deepforest_collate_fn, checkpoint_dir=str(out_dir),
                total_epochs=epochs, batch_size=t["batch_size"], lr=lr, weight_decay=t["weight_decay"],
                warmup_epochs=0, val_fraction=t["val_fraction"], lam_cls=l["lam_cls"], lam_l1=l["lam_l1"],
                lam_giou=l["lam_giou"], lam_ccm=l.get("lam_ccm", 1.0), lam_enc=l.get("lam_enc", 1.0),
                cgfe_start_epoch=0, viz_every=0, ckpt_every=1, init_weights_from=_need(STAGE_B, "Stage B checkpoint"),
                flat_lr=True, ckpt_prefix="lidar", device="cuda", num_workers=t["num_workers"],
                on_checkpoint=work_vol.commit)
    return _need(Path(out_dir) / "lidar_final.pt", "LiDAR-stage checkpoint")


def stage_c(init_ckpt, out_dir, smoke):
    sys.path.insert(0, "/root")
    from src.train.stage_c_multi import run_stage_c_multi
    cfg = _cfg("stage_c"); t, l, ev = cfg["training"], cfg["loss"], cfg.get("eval", {})
    model = _model(cfg)
    js = {s: _stage_json(s, out_dir) for s in ("train", "val")}
    epochs, frozen = (2, 1) if smoke else (t["total_epochs"], t["frozen_epochs"])
    run_stage_c_multi(model=model, is_dqdetr=True, ckpt_prefix="dqdetr", train_annotations_path=js["train"],
                      val_annotations_path=js["val"], checkpoint_dir=str(out_dir),
                      norm_stats_path="/root/configs/naip_normalization.yaml", total_epochs=epochs, frozen_epochs=frozen,
                      batch_size=t["batch_size"], neck_head_lr=t["neck_head_lr"], encoder_lr=t["encoder_lr"],
                      weight_decay=t["weight_decay"], warmup_epochs=t["warmup_epochs"],
                      unfreeze_n_blocks=t["unfreeze_n_blocks"], lam_cls=l["lam_cls"], lam_l1=l["lam_l1"],
                      lam_giou=l["lam_giou"], lam_ccm=l.get("lam_ccm", 1.0), lam_enc=l.get("lam_enc", 1.0),
                      cgfe_start_epoch=0, device="cuda", num_workers=t["num_workers"], stage_b_checkpoint=init_ckpt,
                      viz_every=10, n_viz_tiles=5, eval_conf_thresh=ev.get("conf_thresh", 0.5),
                      eval_iou_thresh=ev.get("iou_thresh", 0.5), ckpt_every=0, on_checkpoint=work_vol.commit)
    return model, _need(Path(out_dir) / "dqdetr_stage_c_best.pt", "Stage C best checkpoint")


def predict_and_score(model, ckpt, out_dir, arm, info):
    """Raw predictions on val/test -> the DeepForest arms' scoring (train.py)."""
    import numpy as np, pandas as pd, torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, "/root"); sys.path.insert(0, "/repo/lidar_pretrain")
    from src.data.naip_dataset import NAIPTileDataset, naip_collate_fn
    from src.eval.inference import predict_tile
    import train as T
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False); model.to("cuda").eval()
    lab = Path("/tmp/labelled"); T.prep_labelled(lab)
    raw = {}
    for split in ("val", "test"):
        ds = NAIPTileDataset(_stage_json(split, out_dir), "/root/configs/naip_normalization.yaml", augment=False)
        rows, r = [], {}
        for batch in DataLoader(ds, batch_size=8, shuffle=False, collate_fn=naip_collate_fn):
            bl, sl = predict_tile(model, batch, conf_thresh=0.01, device="cuda")
            for path, b, s in zip(batch["tile_paths"], bl, sl):
                name = f"{Path(path).stem}.png"
                xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2] * 256 \
                    if len(b) else np.zeros((0, 4))
                xyxy = np.clip(xyxy, 0, 256); r[name] = (xyxy, np.asarray(s, float))
                rows += [{"image_path": name, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "score": float(sc)}
                         for (x0, y0, x1, y1), sc in zip(xyxy, s)]
        raw[split] = r
        pd.DataFrame(rows).to_csv(Path(out_dir) / f"raw_predictions_{split}.csv", index=False)
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


@app.function(**GPU)
def train_arm(seed: int, arm: str, smoke: bool = False, lidar_epochs: int = 10, lidar_lr: float = 5e-4,
              runs: str = "runs_clay", data: str = "pretrain_clay"):
    import wandb
    assert arm in ARMS
    work_vol.reload()
    if smoke:
        runs = runs + "_smoke"
    out = Path(WORK, runs, f"seed{seed}", arm); out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event(); threading.Thread(target=committer, args=(stop,), daemon=True).start()
    group = f"{'smoke-' if smoke else ''}clay-seed{seed}"
    cfg = {"seed": seed, "arm": arm, "stage_b": STAGE_B, "lidar_epochs": lidar_epochs, "lidar_lr": lidar_lr,
           "smoke": smoke, "data": data}
    try:
        _need(CLAY_CKPT, "Clay v1.5 encoder"); _need(STAGE_B, "Stage B checkpoint")
        init = STAGE_B
        if arm == "ClayB":
            lid = Path(WORK, runs, f"seed{seed}", "lidar_stage")
            done = lid / "lidar_final.pt"
            if done.exists():
                init = str(done); print(f"reusing {done}")
            else:
                _seed(seed)
                _wandb(f"{group.replace('-seed', '-s')}-lidar", group, cfg)
                init = str(lidar_stage(seed, lid, 1 if smoke else lidar_epochs, lidar_lr, smoke, data))
                wandb.finish()
        _seed(seed)
        run = _wandb(f"{group.replace('-seed', '-s')}-{arm}-stageC", group, cfg)
        model, best = stage_c(init, out, smoke)
        rec = predict_and_score(model, best, out, arm, {"init_checkpoint": init, "seed": seed})
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


@app.function(cpu=4, memory=8192, volumes={WORK: work_vol})
def summary(runs: str = "runs_clay"):
    work_vol.reload()
    a = subprocess.run([sys.executable, "summarize.py", WORK, runs], capture_output=True, text=True, cwd="/repo/lidar_pretrain")
    b = subprocess.run([sys.executable, "rescore_by_period.py", WORK, runs], capture_output=True, text=True,
                       cwd="/repo/lidar_pretrain")
    text = a.stdout + "\n\n# pre-/post-fire test split\n" + "\n".join(l for l in b.stdout.splitlines() if "Warning" not in l)
    Path(WORK, f"summary_{runs}.md").write_text(text); work_vol.commit()
    return text + a.stderr[-1500:] + b.stderr[-1500:]


def _status(**kw):
    p = Path(WORK, "pipeline_status_clay.json")
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw); p.write_text(json.dumps(cur, indent=1)); work_vol.commit()
    print("status:", json.dumps(kw), flush=True)


@app.function(cpu=1, timeout=24 * 3600, volumes={WORK: work_vol})
def pipeline(stage: str = "build,train", seeds: str = "0,1,2", smoke: bool = False):
    import datetime
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    S = set(stage.split(","))
    _status(started=now(), stage=stage, seeds=seeds, smoke=smoke, state="running", build=None, train=None,
            summary=None, error=None)
    try:
        if "build" in S:
            _status(build=build.remote())
        if "train" in S:
            jobs = [(int(s), arm) for s in seeds.split(",") for arm in ARMS]
            done = {}
            for (s, arm), res in zip(jobs, train_arm.starmap(jobs, kwargs={"smoke": smoke}, return_exceptions=True)):
                done[f"seed{s}/{arm}"] = {"error": f"{type(res).__name__}: {res}"[:300]} \
                    if isinstance(res, BaseException) else res
                _status(train=done)
        if S & {"train", "summary"} and not smoke:
            _status(summary=summary.remote())
        _status(state="done", finished=now())
    except Exception as e:
        _status(state="failed", error=f"{type(e).__name__}: {e}"[:500], finished=now())
        raise


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="spawn the Clay pipeline on the deployed app and exit")
    ap.add_argument("--stage", default="build,train"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    call = modal.Function.from_name("lidar-clay", "pipeline").spawn(a.stage, a.seeds, a.smoke)
    print(f"spawned clay pipeline (stage={a.stage}, seeds={a.seeds}, smoke={a.smoke}): call {call.object_id}")
