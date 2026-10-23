#!/usr/bin/env python3
"""
Step 3: does pretraining on 3DEP LiDAR crowns help DeepForest on NAIP?

Arms (all start from DeepForest's prebuilt NEON model, deepforest==2.1.0):
  A  prebuilt -> fine-tune on the 75 hand-labelled train tiles           (current model)
  B  prebuilt -> pretrain on LiDAR crowns (build_pretrain.py) -> fine-tune as A
  C  prebuilt -> pretrain on LiDAR crowns, no fine-tuning                (labels alone)
B and C share one pretraining run per seed. Fine-tuning matches dfft/modal_train.py
(15 epochs, lr 0.001, batch 4, horizontal flips).

Evaluation: one inference pass per split with loose thresholds (score 0.01, NMS 0.99),
then score threshold and NMS IoU are chosen on **val** (best F1 at IoU 0.4) and test is
scored once with them. Matching is one-to-one greedy by score, as dfft/modal_sweep.py.
Reported on test: P/R/F1 at IoU 0.25/0.4/0.5, count MAE, for all 15 tiles and for
tiles with >= 5 crowns (the report's convention), plus a fixed reference setting
(score 0.1, DeepForest's NMS 0.05).

    python lidar_pretrain/train.py WORK --seed 0 [--arms A B C] [--pretrain-epochs 3] [--wandb-project NAME]
    python lidar_pretrain/summarize.py WORK

Writes WORK/runs/seed{S}/{A,B,C}/metrics.json (+ predictions), and the pretrained
checkpoint WORK/runs/seed{S}/pretrained.ckpt (reused by B and C, and on re-runs).

With --wandb-project, each phase is a wandb run in group "seed{S}": "s{S}-A" (fine-tune),
"s{S}-pretrain", "s{S}-B" (fine-tune after pretraining), "s{S}-C" (eval only). Curves:
DeepForest's train/val loss and val box precision/recall (IoU 0.4) every validation;
final test metrics go to each run's summary under test/... and val-selected/...
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

import common as C

IOUS = [0.25, 0.4, 0.5]
SCORES = [round(x, 2) for x in np.arange(0.05, 0.851, 0.05)]
NMSS = [0.05, 0.15, 0.3, 0.5]


# ---------------- data ----------------
def prep_labelled(lab):
    """Hand-labelled tiles -> PNG + DeepForest CSVs in `lab`, exactly as dfft/modal_train._prep."""
    img = lab / "images"; img.mkdir(parents=True, exist_ok=True)
    for tif in sorted((C.TILES / "tiles").glob("*.tif")):
        out = img / f"{tif.stem}.png"
        if not out.exists():
            with rasterio.open(tif) as src:
                Image.fromarray(C.stretch_to_rgb8(src.read([1, 2, 3]))).save(out)
    for split in ("train", "val", "test"):
        rows = []
        for it in C.split_items(split):
            name = f"{Path(it['tile_path']).stem}.png"
            for cx, cy, w, h in it["boxes"]:
                rows.append({"image_path": name, "xmin": max(0, int((cx - w / 2) * 256)), "ymin": max(0, int((cy - h / 2) * 256)),
                             "xmax": min(256, int((cx + w / 2) * 256)), "ymax": min(256, int((cy + h / 2) * 256)), "label": "Tree"})
        pd.DataFrame(rows).to_csv(lab / f"{split}.csv", index=False)
    return img


# ---------------- model ----------------
def load(ckpt=None):
    from deepforest import main as df_main
    if ckpt:
        return df_main.deepforest.load_from_checkpoint(str(ckpt))
    m = df_main.deepforest(); m.load_model()
    return m


def make_logger(out_dir, wandb_project, name, group, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    if wandb_project:
        from pytorch_lightning.loggers import WandbLogger
        return WandbLogger(project=wandb_project, name=name, group=group, job_type=name.split("-")[-1],
                           config=config, save_dir=str(out_dir))
    from pytorch_lightning.loggers import CSVLogger
    return CSVLogger(str(out_dir), name="log")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}/"))
        elif isinstance(v, (int, float)) and v is not None:
            out[f"{prefix}{k}"] = v
    return out


def fit(m, csv, root, val_csv, val_root, epochs, lr, batch, workers, out_dir, accelerator, logger,
        val_check_interval=1.0, grad_clip=0.0, ckpt_dir=None):
    """grad_clip > 0 clips the gradient norm (DeepForest doesn't by default). With ckpt_dir, a
    checkpoint is saved every epoch and training resumes from ckpt_dir/last.ckpt if present."""
    m.config["train"]["csv_file"] = str(csv); m.config["train"]["root_dir"] = str(root)
    m.config["validation"]["csv_file"] = str(val_csv); m.config["validation"]["root_dir"] = str(val_root)
    m.config["train"]["epochs"] = epochs; m.config["train"]["lr"] = lr
    m.config["batch_size"] = batch; m.config["workers"] = workers
    m.config["accelerator"] = accelerator
    m.config["validation"]["val_accuracy_interval"] = 1      # box P/R on val at every validation
    extra, callbacks, resume = {}, None, None
    if grad_clip:
        extra["gradient_clip_val"] = grad_clip
    if ckpt_dir:
        from pytorch_lightning.callbacks import ModelCheckpoint
        ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [ModelCheckpoint(dirpath=str(ckpt_dir), filename="{epoch}", save_last=True,
                                     save_top_k=-1, every_n_epochs=1)]
        resume = str(ckpt_dir / "last.ckpt") if (ckpt_dir / "last.ckpt").exists() else None
        if resume:
            print(f"resuming from {resume}", flush=True)
    else:
        extra["enable_checkpointing"] = False
    m.create_trainer(logger=logger, callbacks=callbacks, default_root_dir=str(out_dir),
                     log_every_n_steps=5, val_check_interval=val_check_interval, **extra)
    t = time.time(); m.trainer.fit(m, ckpt_path=resume)
    return round(time.time() - t, 1)


# ---------------- evaluation ----------------
def collect(m, csv, root):
    """Loose-threshold predictions -> {image: (boxes N x 4, scores N)} for every image in csv."""
    net = m.model
    net.score_thresh = 0.01; net.nms_thresh = 0.99
    net.detections_per_img = 3000; net.topk_candidates = 4000
    m.config["score_thresh"] = 0.01; m.config["nms_thresh"] = 0.99
    gt = pd.read_csv(csv)
    preds = m.predict_file(csv_file=str(csv), root_dir=str(root))
    if preds is None or len(preds) == 0 or "image_path" not in preds:     # e.g. a diverged model
        print(f"WARNING: no predictions on {Path(csv).name}", flush=True)
        return {img: (np.zeros((0, 4)), np.zeros(0)) for img in gt["image_path"].unique()}
    preds["image_path"] = preds["image_path"].apply(lambda p: Path(p).name)
    out = {}
    for img in gt["image_path"].unique():
        d = preds[preds["image_path"] == img]
        out[img] = (d[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float), d["score"].to_numpy(float))
    return out


def gt_by_img(csv, min_gt=0):
    gt = pd.read_csv(csv)
    return {img: d[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
            for img, d in gt.groupby("image_path") if len(d) >= min_gt}


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def greedy_tp(preds, gt, thr):
    M = iou_matrix(preds, gt); taken = np.zeros(len(gt), bool); tp = 0
    for i in range(len(preds)):
        if not M.shape[1]:
            break
        for j in np.argsort(-M[i]):
            if M[i, j] < thr:
                break
            if not taken[j]:
                taken[j] = True; tp += 1; break
    return tp


def score(raw, gts, score_t, nms_t, ious=IOUS):
    import torch
    from torchvision.ops import nms
    tot = {i: [0, 0, 0] for i in ious}; err = []
    for img, gt in gts.items():
        b, s = raw.get(img, (np.zeros((0, 4)), np.zeros(0)))
        keep = s >= score_t; b, s = b[keep], s[keep]
        if len(b):
            k = nms(torch.tensor(b, dtype=torch.float32), torch.tensor(s, dtype=torch.float32), nms_t).numpy()
            b, s = b[k], s[k]; b = b[np.argsort(-s)]
        for i in ious:
            tp = greedy_tp(b, gt, i); tot[i][0] += tp; tot[i][1] += len(b) - tp; tot[i][2] += len(gt) - tp
        err.append(abs(len(b) - len(gt)))
    res = {}
    for i, (tp, fp, fn) in tot.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res[f"iou{i}"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}
    res["count_mae"] = round(float(np.mean(err)), 2) if err else None
    res["n_tiles"] = len(gts)
    return res


def evaluate(m, lab, out_dir):
    img = lab / "images"
    raw = {s: collect(m, lab / f"{s}.csv", img) for s in ("val", "test")}
    val_gt = gt_by_img(lab / "val.csv")
    grid = [(st, nt, score(raw["val"], val_gt, st, nt, [0.4])["iou0.4"]["f1"]) for st in SCORES for nt in NMSS]
    st, nt, vf1 = max(grid, key=lambda g: g[2])
    test_all, test_min5 = gt_by_img(lab / "test.csv"), gt_by_img(lab / "test.csv", 5)
    out = {"selected_on_val": {"score": st, "nms": nt, "val_f1_iou0.4": vf1},
           "test_all_tiles": score(raw["test"], test_all, st, nt),
           "test_min5": score(raw["test"], test_min5, st, nt),
           "test_fixed_ref_min5": score(raw["test"], test_min5, 0.1, 0.05)}
    for s, r in raw.items():   # keep raw predictions for later re-scoring
        rows = [{"image_path": k, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "score": sc}
                for k, (bb, ss) in r.items() for (x0, y0, x1, y1), sc in zip(bb, ss)]
        pd.DataFrame(rows).to_csv(out_dir / f"raw_predictions_{s}.csv", index=False)
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--pretrain-csv", help="default WORK/pretrain/pretrain.csv")
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--pretrain-lr", type=float, default=0.001)
    ap.add_argument("--pretrain-batch", type=int, default=8)
    ap.add_argument("--pretrain-grad-clip", type=float, default=0.0,
                    help="clip gradient norm during pretraining (0 = off, DeepForest's default)")
    ap.add_argument("--pretrain-ckpt-every-epoch", action="store_true",
                    help="checkpoint each pretraining epoch to RUN/pretrain_ckpts and resume from it")
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--finetune-lr", type=float, default=0.001)
    ap.add_argument("--finetune-batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--max-pretrain-images", type=int, default=0, help="subsample (smoke tests)")
    ap.add_argument("--wandb-project", help="log curves and final metrics to this wandb project")
    ap.add_argument("--runs-dir", default="runs", help="results folder under WORK")
    ap.add_argument("--wandb-group-prefix", default="", help="e.g. 'smoke-' to keep test runs apart")
    ap.add_argument("--labelled-dir", help="where to write the hand-labelled PNGs/CSVs (default WORK/labelled)")
    ap.add_argument("--pretrain-val-every", type=float, default=0.25,
                    help="validate this fraction of a pretraining epoch (for smoother curves)")
    a = ap.parse_args()
    import pytorch_lightning as pl

    work = Path(a.work); lab = Path(a.labelled_dir) if a.labelled_dir else work / "labelled"; prep_labelled(lab)
    run = work / a.runs_dir / f"seed{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    pre_csv = Path(a.pretrain_csv) if a.pretrain_csv else work / "pretrain" / "pretrain.csv"
    pre_root = pre_csv.parent / "images"
    if a.max_pretrain_images:
        d = pd.read_csv(pre_csv); keep = sorted(d.image_path.unique())[:a.max_pretrain_images]
        pre_csv = run / "pretrain_subset.csv"; d[d.image_path.isin(keep)].to_csv(pre_csv, index=False)
    ft = dict(csv=lab / "train.csv", root=lab / "images", val_csv=lab / "val.csv", val_root=lab / "images",
              epochs=a.finetune_epochs, lr=a.finetune_lr, batch=a.finetune_batch, workers=a.workers, accelerator=a.accelerator)
    pre_ckpt = run / "pretrained.ckpt"
    group = f"{a.wandb_group_prefix}seed{a.seed}"
    logger = lambda name, d: make_logger(d, a.wandb_project, f"{a.wandb_group_prefix}s{a.seed}-{name}", group, vars(a))

    def finish(arm, m, info, lg=None):
        d = run / arm; d.mkdir(exist_ok=True)
        info["metrics"] = evaluate(m, lab, d)
        info["config"] = vars(a)
        (d / "metrics.json").write_text(json.dumps(info, indent=1))
        if a.wandb_project:
            import wandb
            lg = lg or logger(arm, d)
            run_ = lg.experiment
            mt = info["metrics"]
            run_.summary.update({**flatten(mt["test_all_tiles"], "test/"), **flatten(mt["test_min5"], "test_min5/"),
                                 **flatten(mt["test_fixed_ref_min5"], "test_ref_min5/"),
                                 **flatten(mt["selected_on_val"], "val-selected/"),
                                 **{k: v for k, v in info.items() if isinstance(v, (int, float))}})
            rows = [[i, mt["test_all_tiles"][f"iou{i}"]["precision"], mt["test_all_tiles"][f"iou{i}"]["recall"],
                     mt["test_all_tiles"][f"iou{i}"]["f1"]] for i in IOUS]
            run_.log({"test/by_iou": wandb.Table(columns=["iou", "precision", "recall", "f1"], data=rows)})
            wandb.finish()
        t = info["metrics"]["test_all_tiles"]
        print(f"[seed {a.seed}] {arm}: test F1@0.4 {t['iou0.4']['f1']:.3f}  F1@0.5 {t['iou0.5']['f1']:.3f}  "
              f"MAE {t['count_mae']}  (val-selected score {info['metrics']['selected_on_val']['score']}, "
              f"nms {info['metrics']['selected_on_val']['nms']})", flush=True)

    if "A" in a.arms:
        pl.seed_everything(a.seed, workers=True)
        lg = logger("A", run / "A")
        m = load(); secs = fit(m, out_dir=run / "A", logger=lg, **ft)
        finish("A", m, {"arm": "A", "finetune_seconds": secs}, lg)
    if {"B", "C"} & set(a.arms):
        n_img = pd.read_csv(pre_csv).image_path.nunique()
        if pre_ckpt.exists():
            pre_secs = json.loads((run / "pretrained.json").read_text())["seconds"]
            print(f"reusing {pre_ckpt}")
        else:
            pl.seed_everything(a.seed, workers=True)
            m = load()
            lg = logger("pretrain", run / "pretrain")
            pre_secs = fit(m, pre_csv, pre_root, lab / "val.csv", lab / "images", a.pretrain_epochs, a.pretrain_lr,
                           a.pretrain_batch, a.workers, run / "pretrain", a.accelerator, lg, a.pretrain_val_every,
                           a.pretrain_grad_clip, run / "pretrain_ckpts" if a.pretrain_ckpt_every_epoch else None)
            m.trainer.save_checkpoint(str(pre_ckpt))
            (run / "pretrained.json").write_text(json.dumps({"seconds": pre_secs, "images": n_img, "csv": str(pre_csv)}))
            if a.wandb_project:
                import wandb
                lg.experiment.summary.update({"pretrain_images": n_img, "pretrain_seconds": pre_secs}); wandb.finish()
        info = {"pretrain_images": n_img, "pretrain_seconds": pre_secs}
        if "C" in a.arms:
            finish("C", load(pre_ckpt), {"arm": "C", **info})
        if "B" in a.arms:
            pl.seed_everything(a.seed, workers=True)
            lg = logger("B", run / "B")
            m = load(pre_ckpt); secs = fit(m, out_dir=run / "B", logger=lg, **ft)
            finish("B", m, {"arm": "B", "finetune_seconds": secs, **info}, lg)


if __name__ == "__main__":
    main()
