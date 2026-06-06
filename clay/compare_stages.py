"""Compare Stage-B vs Stage-C checkpoints for each model on the held-out test set.

For every model (q150, q500, dqdetr, quarter) this loads its Stage-B checkpoint
(checkpoints/stage_b/) and its Stage-C checkpoint (checkpoints/stage_c/), runs
both on single_tiles_flat/stage_c_data/test.json, and reports:

  * a metrics table before (B) vs after (C) finetuning — precision / recall / F1 /
    mAP / count error — plus the delta, and
  * side-by-side qualitative panels per sample tile: RGB+GT | Stage-B preds |
    Stage-C preds (predictions green, ground truth blue), and
  * a confidence-threshold sweep: per-model P/R/F1-vs-conf plots (Stage B vs C)
    plus a combined F1-vs-conf overview across models.

All the model builders + inference paths are reused from eval_checkpoints.py:
  - q150 / q500 -> fixed-query TreeDetector            (run_inference)
  - dqdetr      -> DQ-DETR (encoder from clay.ckpt)    (run_inference_dqdetr)
  - quarter     -> fixed-query, run on 128 quarters    (run_inference_quarter)

The test tiles are already the 256x256 native-res crops written by
single_tiles_flat/build_stage_c_split.py (boxes normalized to the 256 frame), so
no cropping happens here — they are read as-is.

Example:
  python compare_stages.py
  python compare_stages.py --models q150,quarter --n-samples 6 --device mps
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import eval_checkpoints as E  # noqa: E402  (reuse builders + inference + viz)

REPO = HERE.parent

# model -> (kind, stage-B filename, stage-C filename). q is read off the
# checkpoint for fixed/quarter, so only the kind + paths are needed here.
MODELS = {
    "q150":    {"kind": "fixed",   "b": "stage_b/q150_final.pt",    "c": "stage_c/q150_stage_c_best.pt"},
    "q500":    {"kind": "fixed",   "b": "stage_b/q500_final.pt",    "c": "stage_c/q500_stage_c_best.pt"},
    "dqdetr":  {"kind": "dqdetr",  "b": "stage_b/dq-detr_final.pt", "c": "stage_c/dqdetr_stage_c_best.pt"},
    "quarter": {"kind": "quarter", "b": "stage_b/quarter_final.pt", "c": "stage_c/quarter_stage_c_best.pt"},
}

TABLE_KEYS = ["precision", "recall", "f1", "best_f1", "mAP50", "mAP50_95",
              "count_mae", "count_rmse"]


# --------------------------------------------------------------------------- #
# Test-set dataset (already-256 crops from test.json)
# --------------------------------------------------------------------------- #
class TestTileDataset:
    """One sample per test.json entry. Tiles are 256x256 4-band crops; boxes are
    already normalized to the 256 frame. Yields the same item dict shape that
    eval_checkpoints' run_inference* / collate expect."""

    def __init__(self, test_json, tiles_root, modal_root="/data/stage_c"):
        with open(test_json) as f:
            raw = json.load(f)
        self.tiles_root = Path(tiles_root)
        self.modal_root = modal_root
        self.items = []
        for it in raw:
            tp = it["tile_path"].replace(modal_root, str(self.tiles_root))
            if not Path(tp).exists():
                print(f"[warn] missing tile {tp} — skipping")
                continue
            boxes = np.asarray(it.get("boxes", []), dtype=np.float32).reshape(-1, 4)
            self.items.append({
                "tile_path": tp,
                "imgname": Path(tp).stem,
                "boxes": boxes,
                "lat": float(it.get("lat", 37.0)),
                "lon": float(it.get("lon", -119.0)),
                "date": it.get("acquisition_date", "2020-06-01"),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import rasterio
        from datetime import datetime
        it = self.items[idx]
        with rasterio.open(it["tile_path"]) as src:
            raw = src.read(indexes=[1, 2, 3, 4]).astype(np.float32)   # (4, 256, 256)
        mean = E.NAIP_MEAN.reshape(4, 1, 1)
        std = E.NAIP_STD.reshape(4, 1, 1)
        pixels = torch.from_numpy((raw - mean) / std)
        week = datetime.strptime(it["date"], "%Y-%m-%d").isocalendar()[1]
        return {
            "pixels": pixels,
            "wavelengths": E.NAIP_WAVELENGTHS,
            "gsd": torch.tensor([E.NAIP_GSD], dtype=torch.float32),
            "time": E.encode_time(week, 12.0),
            "latlon": E.encode_latlon(it["lat"], it["lon"]),
            "boxes": torch.from_numpy(it["boxes"]),
            "imgname": it["imgname"],
            "tile_path": it["tile_path"],
            "rgb": E.display_rgb(raw),
            "rgb_raw": raw[:3].astype(np.uint8).transpose(1, 2, 0),
        }


# --------------------------------------------------------------------------- #
# Per-model build + inference dispatch (works for both Stage B and Stage C)
# --------------------------------------------------------------------------- #
def eval_checkpoint(kind, ckpt_path, ds, args, device):
    """Build the model for `kind`, load `ckpt_path`, run inference on ds, return
    (metrics, res). Handles the architecture/encoder/inference differences."""
    if kind == "fixed":
        q = E.parse_q(ckpt_path)        # q150_*/q500_* -> 150/500
        model = E.build_model(q, ckpt_path, device)
        res = E.run_inference(model, ds, device, batch_size=args.batch_size)
    elif kind == "quarter":
        model, _ = E.build_quarter_model(ckpt_path, device)
        res = E.run_inference_quarter(model, ds, device, args.quarter_nms_iou)
    elif kind == "dqdetr":
        model = E.build_dqdetr_model(ckpt_path, args.clay_ckpt, args.dqdetr_config,
                                     device, enable_cgfe=not args.dqdetr_cgfe_off)
        res = E.run_inference_dqdetr(model, ds, device)
    else:
        raise ValueError(kind)
    metrics = E.evaluate(res, args.iou_thresh, args.conf_thresh)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return metrics, res


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(name, mB, mC):
    print(f"\n=== {name}:  Stage B -> Stage C  (test set, conf>={mB['conf_thresh']}) ===")
    print(f"  n_tiles={mB['n_tiles']}  n_gt_boxes={mB['n_gt_boxes']}")
    print(f"  {'metric':14s} {'stageB':>10s} {'stageC':>10s} {'delta':>10s}")
    for k in TABLE_KEYS:
        b, c = mB[k], mC[k]
        print(f"  {k:14s} {b:10.4f} {c:10.4f} {c - b:+10.4f}")


def side_by_side(name, res_b, res_c, out_dir, conf, n_samples, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_samples, len(res_b["gt_boxes"]))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(res_b["gt_boxes"]), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.7 * n))
    if n == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        rgb = res_b["rgbs"][i]
        gt = res_b["gt_boxes"][i].numpy()
        pb_b, _ = E.filter_by_conf(res_b["pred_boxes"][i], res_b["pred_scores"][i], conf)
        pb_c, _ = E.filter_by_conf(res_c["pred_boxes"][i], res_c["pred_scores"][i], conf)
        nm = res_b["imgnames"][i]
        E.draw(axes[row, 0], rgb, gt=gt, title=f"{nm}\nGT={len(gt)}")
        E.draw(axes[row, 1], rgb, gt=gt, preds=pb_b, title=f"Stage B  pred={len(pb_b)}")
        E.draw(axes[row, 2], rgb, gt=gt, preds=pb_c, title=f"Stage C  pred={len(pb_c)}")
    fig.suptitle(f"{name}: GT (blue) — Stage B vs Stage C predictions (green)  "
                 f"[conf>={conf}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path = out_dir / f"compare_stages_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved side-by-side panels -> {path}")


def _sweep(res, iou_thresh, thresholds):
    """Precision/recall/F1 at each confidence threshold for one stage's preds."""
    P, R, F = [], [], []
    for t in thresholds:
        pr = E.precision_recall_f1(res["pred_boxes"], res["pred_scores"],
                                   res["gt_boxes"], iou_thresh, float(t))
        P.append(pr["precision"]); R.append(pr["recall"]); F.append(pr["f1"])
    return {"precision": P, "recall": R, "f1": F}


def sweep_and_plot(name, res_b, res_c, out_dir, iou_thresh, thresholds):
    """Sweep the confidence threshold and plot P/R/F1 vs conf, Stage B vs C."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sb = _sweep(res_b, iou_thresh, thresholds)
    sc = _sweep(res_c, iou_thresh, thresholds)
    x = list(map(float, thresholds))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ("precision", "recall", "f1")):
        ax.plot(x, sb[key], "--o", ms=3, color="#1f77b4", label="Stage B")
        ax.plot(x, sc[key], "-o", ms=3, color="#d62728", label="Stage C")
        ax.set_xlabel("confidence threshold")
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs conf")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{name}: Stage B vs Stage C confidence sweep (IoU={iou_thresh})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"compare_stages_{name}_sweep.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved confidence sweep -> {path}")
    return {"thresholds": x, "stage_b": sb, "stage_c": sc}


def f1_overview(sweeps, out_dir):
    """One figure overlaying every model's F1-vs-conf curve, Stage B (dashed) vs C (solid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not sweeps:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, (name, sw) in enumerate(sweeps.items()):
        c = cmap(i % 10)
        ax.plot(sw["thresholds"], sw["stage_b"]["f1"], "--", color=c, alpha=0.6,
                label=f"{name} (B)")
        ax.plot(sw["thresholds"], sw["stage_c"]["f1"], "-", color=c,
                label=f"{name} (C)")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("F1")
    ax.set_ylim(bottom=0)
    ax.set_title("F1 vs confidence — Stage B (dashed) vs Stage C (solid)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "compare_stages_f1_overview.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote F1 overview -> {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints-dir", default=str(HERE / "checkpoints"))
    ap.add_argument("--test-json",
                    default=str(REPO / "single_tiles_flat/stage_c_data/test.json"))
    ap.add_argument("--tiles-root",
                    default=str(REPO / "single_tiles_flat/stage_c_data"),
                    help="local dir the test.json tile_path entries resolve against")
    ap.add_argument("--modal-root", default="/data/stage_c",
                    help="prefix in test.json tile_path to replace with --tiles-root")
    ap.add_argument("--models", default="q150,q500,dqdetr,quarter",
                    help="comma-separated subset of models to compare")
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out-dir", default=str(HERE / "eval_out/stage_compare"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-points", type=int, default=19,
                    help="number of confidence thresholds in [0.05, 0.95] to sweep/plot")
    ap.add_argument("--quarter-nms-iou", type=float, default=0.6)
    ap.add_argument("--clay-ckpt", default=str(HERE / "checkpoints/clay.ckpt"),
                    help="full Clay v1.5 checkpoint for the DQ-DETR Stage-B encoder")
    ap.add_argument("--dqdetr-config",
                    default=str(REPO / "dq-detr-impl/configs/stage_b.yaml"))
    ap.add_argument("--dqdetr-cgfe-off", action="store_true")
    args = ap.parse_args()

    device = E.pick_device(args.device)
    print(f"device: {device}")

    ds = TestTileDataset(args.test_json, args.tiles_root, args.modal_root)
    print(f"test set: {len(ds)} tiles from {args.test_json}")
    if len(ds) == 0:
        raise SystemExit("No test tiles found (check --tiles-root).")

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir)
    thresholds = np.linspace(0.05, 0.95, args.sweep_points)
    summary, sweeps = {}, {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        cfg = MODELS.get(name)
        if cfg is None:
            print(f"[skip] unknown model '{name}'")
            continue
        b_path, c_path = ckpt_dir / cfg["b"], ckpt_dir / cfg["c"]
        if not b_path.exists() or not c_path.exists():
            miss = [str(p) for p in (b_path, c_path) if not p.exists()]
            print(f"[skip] {name}: missing checkpoint(s) {miss}")
            continue
        if cfg["kind"] == "dqdetr" and not Path(args.clay_ckpt).exists():
            print(f"[skip] {name}: clay.ckpt not found at {args.clay_ckpt}")
            continue
        try:
            print(f"\n########## {name} ##########")
            print(f"[stage B] {b_path.name}")
            mB, resB = eval_checkpoint(cfg["kind"], str(b_path), ds, args, device)
            print(f"[stage C] {c_path.name}")
            mC, resC = eval_checkpoint(cfg["kind"], str(c_path), ds, args, device)
            print_table(name, mB, mC)
            side_by_side(name, resB, resC, out_dir, args.conf_thresh,
                         args.n_samples, args.seed)
            sweeps[name] = sweep_and_plot(name, resB, resC, out_dir,
                                          args.iou_thresh, thresholds)
            summary[name] = {"stage_b": mB, "stage_c": mC, "sweep": sweeps[name]}
        except Exception as e:
            print(f"[error] {name}: {type(e).__name__}: {e}")

    f1_overview(sweeps, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_stage_compare.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote metrics -> {out_dir / 'metrics_stage_compare.json'}")


if __name__ == "__main__":
    main()
