"""
Qualitative test-set figures for the fine-tuned DeepForest checkpoint.

Fig 1: score-threshold sweep  (rows = test tiles by density, cols = score thresh)
Fig 2: localization diagnostic (same preds, recolored by IoU 0.5 vs 0.25 match)

Inference config held at the val-selected min_size=1000, nms=0.3.

    modal run dfft/modal_viz.py
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("deepforest==2.1.0", "rasterio", "geopandas", "shapely", "pandas", "numpy",
                 "pillow", "matplotlib")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO / "single_tiles_flat" / "stage_c_data", remote_path="/data/stage_c_data")
)
app = modal.App("deepforest-viz", image=image)
ckpt_vol = modal.Volume.from_name("dfft-checkpoints", create_if_missing=True)

DATA, WORK, CKPT = "/data/stage_c_data", "/work/deepforest_data", "/checkpoints"

MIN_SIZE, NMS = 1000, 0.30
SCORES = [0.10, 0.20, 0.30, 0.40]
UPSCALE = 3

# dataviz categorical slots 4 / 1 / 5 — validated (CVD ΔE 13.0 protan, normal 27.5)
C_GT, C_TP, C_FP = "#eda100", "#2a78d6", "#e87ba4"
INK, INK_DIM, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _prep():
    import numpy as np, pandas as pd, rasterio
    from PIL import Image
    out = Path(WORK) / "images"; out.mkdir(parents=True, exist_ok=True)
    for tif in sorted((Path(DATA) / "tiles").glob("*.tif")):
        p = out / (tif.stem + ".png")
        if p.exists():
            continue
        with rasterio.open(tif) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        for i in range(3):
            lo, hi = np.percentile(rgb[i], 2), np.percentile(rgb[i], 98)
            if hi > lo:
                rgb[i] = (rgb[i] - lo) / (hi - lo)
            rgb[i] = np.clip(rgb[i], 0, 1)
        Image.fromarray((rgb * 255).astype(np.uint8).transpose(1, 2, 0)).save(p)
    for split in ["train", "val", "test"]:
        items = json.loads((Path(DATA) / f"{split}.json").read_text())
        rows = []
        for it in items:
            png = str(out / f"{Path(it['tile_path']).stem}.png")
            for cx, cy, w, h in it["boxes"]:
                rows.append({"image_path": png,
                             "xmin": max(0, int((cx-w/2)*256)), "ymin": max(0, int((cy-h/2)*256)),
                             "xmax": min(256, int((cx+w/2)*256)), "ymax": min(256, int((cy+h/2)*256)),
                             "label": "Tree"})
        pd.DataFrame(rows).to_csv(Path(WORK) / f"{split}.csv", index=False)


def _iou(a, b):
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); bb = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / np.maximum(aa[:, None]+bb[None, :]-inter, 1e-9)


def _match(preds, gt, thr):
    """preds sorted by score desc -> boolean mask of matched preds, n matched gt."""
    import numpy as np
    M = _iou(preds, gt); taken = np.zeros(len(gt), bool); hit = np.zeros(len(preds), bool)
    for i in range(len(preds)):
        if M.shape[1] == 0:
            break
        for j in np.argsort(-M[i]):
            if M[i, j] < thr:
                break
            if not taken[j]:
                taken[j] = True; hit[i] = True; break
    return hit, int(taken.sum())


def _panel(ax, img, caption=None):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if caption:
        ax.set_xlabel(caption, fontsize=8, color=INK_DIM, labelpad=5)


def _draw(img_path, gt, preds, hit, upscale=UPSCALE):
    from PIL import Image, ImageDraw
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img.width*upscale, img.height*upscale), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for b in gt:
        d.rectangle([b[0]*upscale, b[1]*upscale, b[2]*upscale, b[3]*upscale], outline=C_GT, width=2)
    for b, h in zip(preds, hit):
        d.rectangle([b[0]*upscale, b[1]*upscale, b[2]*upscale, b[3]*upscale],
                    outline=(C_TP if h else C_FP), width=2)
    return img


@app.function(cpu=8.0, memory=16384, timeout=60*60*3, volumes={CKPT: ckpt_vol})
def viz():
    import numpy as np, pandas as pd, torch, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from torchvision.ops import nms
    from deepforest import main as df_main

    _prep()
    model = df_main.deepforest.load_from_checkpoint(str(Path(CKPT) / "deepforest_finetuned.pt"))
    model.eval()
    net = model.model
    net.transform.min_size = (MIN_SIZE,); net.transform.max_size = MIN_SIZE*2
    net.score_thresh = 0.01; net.nms_thresh = 0.99
    net.detections_per_img = 3000; net.topk_candidates = 4000
    model.config["score_thresh"] = 0.01; model.config["nms_thresh"] = 0.99

    csv = str(Path(WORK) / "test.csv")
    gt_df = pd.read_csv(csv); gt_df["k"] = gt_df["image_path"].apply(lambda p: Path(p).name)
    raw = model.predict_file(csv_file=csv, root_dir="/")
    raw["k"] = raw["image_path"].apply(lambda p: Path(p).name)

    dens = gt_df.groupby("k").size().sort_values()
    dens = dens[dens >= 10]  # skip near-empty tiles; they reflect annotation gaps, not model behaviour
    picks = [dens.index[0], dens.index[len(dens)//3], dens.index[2*len(dens)//3], dens.index[-1]]
    img_dir = Path(WORK) / "images"
    print("tiles:", [(k, int(dens[k])) for k in picks])

    def boxes_for(k, score):
        d = raw[raw["k"] == k]
        if not len(d):
            return np.zeros((0, 4))
        b = d[["xmin", "ymin", "xmax", "ymax"]].to_numpy(float); s = d["score"].to_numpy(float)
        m = s >= score; b, s = b[m], s[m]
        if not len(b):
            return np.zeros((0, 4))
        keep = nms(torch.tensor(b, dtype=torch.float32), torch.tensor(s, dtype=torch.float32), NMS).numpy()
        b, s = b[keep], s[keep]
        return b[np.argsort(-s)]

    outdir = Path(CKPT) / "viz"; outdir.mkdir(parents=True, exist_ok=True)

    # ---- Fig 1: score threshold sweep ----
    ncol = len(SCORES) + 1
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.1*ncol, 3.5*len(picks)),
                             facecolor=SURFACE, layout="constrained")
    summary = []
    for r, k in enumerate(picks):
        g = gt_df[gt_df["k"] == k][["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
        ax = axes[r, 0]
        _panel(ax, _draw(str(img_dir/k), g, np.zeros((0, 4)), []))
        ax.set_ylabel(f"{k.replace('.png','')}\n{len(g)} crowns", fontsize=8, color=INK_DIM)
        if r == 0:
            ax.set_title("Ground truth", fontsize=10, color=INK)
        for c, sc in enumerate(SCORES, start=1):
            b = boxes_for(k, sc)
            hit, ngt = _match(b, g, 0.5)
            _panel(axes[r, c], _draw(str(img_dir/k), g, b, hit),
                   caption=f"{len(b)} pred · {int(hit.sum())} TP · {len(b)-int(hit.sum())} FP · {len(g)-ngt} FN")
            if r == 0:
                axes[r, c].set_title(f"score ≥ {sc:.1f}", fontsize=10, color=INK)
            summary.append({"tile": k, "score": sc, "n_gt": len(g), "n_pred": len(b),
                            "tp": int(hit.sum()), "fp": len(b)-int(hit.sum()), "fn": len(g)-ngt})
    handles = [Line2D([], [], color=C_GT, lw=2.5, label="Ground truth"),
               Line2D([], [], color=C_TP, lw=2.5, label="Prediction — matched (IoU ≥ 0.5)"),
               Line2D([], [], color=C_FP, lw=2.5, label="Prediction — unmatched")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK)
    fig.suptitle("Fine-tuned DeepForest on test tiles — confidence threshold sweep\n"
                 f"min_size={MIN_SIZE}, NMS={NMS}; matching at IoU 0.5",
                 fontsize=12, color=INK)
    fig.savefig(outdir / "score_sweep.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ---- Fig 2: localization diagnostic ----
    diag = picks[-2:]
    fig, axes = plt.subplots(len(diag), 2, figsize=(7.4, 3.9*len(diag)),
                             facecolor=SURFACE, layout="constrained")
    for r, k in enumerate(diag):
        g = gt_df[gt_df["k"] == k][["xmin", "ymin", "xmax", "ymax"]].to_numpy(float)
        b = boxes_for(k, 0.30)
        for c, thr in enumerate([0.5, 0.25]):
            hit, ngt = _match(b, g, thr)
            _panel(axes[r, c], _draw(str(img_dir/k), g, b, hit),
                   caption=f"{int(hit.sum())}/{len(b)} predictions matched · recall {ngt/max(len(g),1):.2f}")
            if c == 0:
                axes[r, c].set_ylabel(f"{k.replace('.png','')}\n{len(g)} crowns", fontsize=8, color=INK_DIM)
            if r == 0:
                axes[r, c].set_title(f"matched at IoU ≥ {thr}", fontsize=10, color=INK)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("Same predictions, looser matching — boxes land on trees but sit imprecisely\n"
                 "score ≥ 0.30; only the match criterion differs between columns",
                 fontsize=11.5, color=INK)
    fig.savefig(outdir / "localization.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    ckpt_vol.commit()
    print("saved viz/score_sweep.png, viz/localization.png")
    return summary


@app.local_entrypoint()
def main():
    s = viz.remote()
    import collections
    by = collections.defaultdict(list)
    for r in s:
        by[r["score"]].append(r)
    for sc, rows in sorted(by.items()):
        tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
        print(f"score {sc}: TP={tp} FP={fp} FN={fn}")
