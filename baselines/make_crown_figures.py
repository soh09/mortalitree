#!/usr/bin/env python3
"""
Static overlay figures of LiDAR crowns.

  crowns_on_chm.png     10 chips, sparse -> dense, crowns on the canopy height map
  crowns_hillshade.png  4 fixed 100 m crops x 4 panels:
                        LiDAR hillshade | + crowns | NAIP | NAIP + crowns

Chips for crowns_on_chm are picked from CROWNS_JSON (output of chm_crowns.py) at 10
evenly spaced ranks of crown count. The hillshade crops are hard-coded keys from the
forest sample of tar bin 10-411.

    uv run python baselines/make_crown_figures.py CHM_TAR NAIP_TAR CROWNS_JSON OUT_DIR
"""
import json, sys, tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.io import MemoryFile
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).parent))
from chm_crowns import crowns_from_chm, read_chm

INK, DIM, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
PINK = [0.91, 0.48, 0.64, 1.0]
ND = np.array([0.42, 0.62, 0.86])            # no LiDAR returns (water / gaps)
CMAP = LinearSegmentedColormap.from_list("h", ["#f3f1ea", "#9fc79f", "#3f7f4f", "#123d24"])

HILLSHADE_CROPS = [  # (key, crop row0, crop col0 in CHM metres, label)
    ("10_567579_4115114_2020-04-02", 60, 60, "distinct crowns"),
    ("10_571765_4116313_2020-04-14", 80, 40, "dense tall stand"),
    ("10_573451_4112341_2020-04-14", 0, 120, "lake + edge"),
    ("10_568365_4111626_2020-04-02", 120, 80, "short dense canopy"),
]
CROP = 100


def hillshade(z, az=315, alt=40):
    gy, gx = np.gradient(z)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    a, al = np.radians(az), np.radians(alt)
    return np.clip(np.sin(al) * np.sin(slope) + np.cos(al) * np.cos(slope) * np.cos(a - asp), 0, 1)


def boundaries(seg, crowns):
    keep = np.isin(seg, [g.label for g in crowns])
    return find_boundaries(np.where(keep, seg, 0), mode="inner")


def treetops(sm, crowns):
    return np.array([g.coords[np.argmax(sm[g.coords[:, 0], g.coords[:, 1]])] for g in crowns])


def tidy(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def crowns_on_chm(ctf, cn, recs, out):
    R = sorted(recs, key=lambda r: r["n_crowns"])
    picks = [R[int(i)] for i in np.linspace(0, len(R) - 1, 10)]
    fig, axes = plt.subplots(5, 2, figsize=(11, 27), facecolor=SURF, layout="constrained")
    for ax, r in zip(axes.flat, picks):
        chm, nod, _, _ = read_chm(ctf, cn[r["key"]])
        sm, seg, cr = crowns_from_chm(chm)
        im = ax.imshow(sm, cmap=CMAP, vmin=0, vmax=40, interpolation="nearest")
        ndl = np.zeros(sm.shape + (4,)); ndl[nod] = [*ND, 1]; ax.imshow(ndl, interpolation="nearest")
        ov = np.zeros(sm.shape + (4,)); ov[boundaries(seg, cr)] = PINK; ax.imshow(ov, interpolation="nearest")
        t = treetops(sm, cr)
        ax.scatter(t[:, 1], t[:, 0], s=4, c=INK, linewidths=0)
        tidy(ax)
        ax.set_title(f"{len(cr)} crowns · {r['forest_frac']*100:.0f}% of chip >5 m tall", fontsize=10, color=INK, loc="left")
        ax.set_xlabel(f"{r['key']}   (256 m × 256 m, 1 m/px)", fontsize=8, color=DIM)
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.5, pad=0.01)
    cb.set_label("canopy height (m)", color=INK); cb.ax.tick_params(colors=DIM)
    fig.suptitle("Crowns segmented from 3DEP LiDAR CHM — treetop local-maxima + watershed\n"
                 "pink = crown boundaries · black dots = treetops · blue = no LiDAR returns · sparse to dense",
                 fontsize=12, color=INK)
    fig.savefig(out / "crowns_on_chm.png", dpi=95, facecolor=SURF); plt.close(fig)


def crowns_hillshade(ctf, ntf, cn, nn, out):
    fig, axes = plt.subplots(len(HILLSHADE_CROPS), 4, figsize=(18, 4.85 * len(HILLSHADE_CROPS)),
                             facecolor=SURF, layout="constrained")
    for row, (k, r0, c0, lab) in zip(axes, HILLSHADE_CROPS):
        chm, nod, ct, _ = read_chm(ctf, cn[k])
        with MemoryFile(ntf.extractfile(nn[k]).read()).open() as n:
            naip = n.read()[:4]; nt = n.transform; ndate = Path(nn[k]).stem.split("_")[-1]
        sm, seg, cr = crowns_from_chm(chm)
        win = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
        X0, Y0 = ct * (c0, r0); X1, Y1 = ct * (c0 + CROP, r0 + CROP); ext = [X0, X1, Y1, Y0]
        shaded = CMAP(np.clip(sm[win] / 50, 0, 1))[..., :3] * (0.35 + 0.65 * hillshade(sm[win] * 1.5)[..., None])
        shaded[nod[win]] = ND
        ov = np.zeros((CROP, CROP, 4)); ov[boundaries(seg, cr)[win]] = PINK
        t = treetops(sm, cr); tx, ty = ct * (t[:, 1] + .5, t[:, 0] + .5)
        inw = (tx >= X0) & (tx <= X1) & (ty <= Y0) & (ty >= Y1)
        rgb = np.empty(naip.shape[1:] + (3,), np.float32)
        for i in range(3):
            lo, hi = np.percentile(naip[i], 2), np.percentile(naip[i], 98)
            rgb[..., i] = np.clip((naip[i] - lo) / max(hi - lo, 1), 0, 1)
        nX0, nY0 = nt * (0, 0); nX1, nY1 = nt * (naip.shape[2], naip.shape[1]); next_ = [nX0, nX1, nY1, nY0]
        panels = [("LiDAR hillshade", shaded, ext, False), ("LiDAR hillshade + crowns", shaded, ext, True),
                  (f"NAIP RGB ({ndate})", rgb, next_, False), ("NAIP RGB + LiDAR crowns, as paired", rgb, next_, True)]
        for ax, (title, img, e, labelled) in zip(row, panels):
            ax.imshow(img, extent=e, interpolation="nearest")
            if labelled:
                ax.imshow(ov, extent=ext, interpolation="nearest")
                ax.scatter(tx[inw], ty[inw], s=7, c=INK, edgecolors="white", linewidths=0.4, zorder=3)
            ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); tidy(ax)
            ax.set_title(title, fontsize=10, color=INK, loc="left")
        row[0].set_ylabel(f"{lab}\n{k}", fontsize=8.5, color=DIM)
    fig.legend(handles=[Line2D([], [], color=PINK, lw=2.5, label="crown boundary"),
                        Line2D([], [], marker="o", ls="", mfc=INK, mec="white", label="treetop"),
                        Patch(color=ND, label="no LiDAR returns (water / gaps)")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("Visual check of LiDAR crowns — 100 m × 100 m crops, same window in every panel\n"
                 "LiDAR = 3DEP canopy height as shaded relief (sun from NW) · NAIP = paired photo, alignment unverified",
                 fontsize=12, color=INK)
    fig.savefig(out / "crowns_hillshade.png", dpi=95, facecolor=SURF); plt.close(fig)


def main(chm_tar, naip_tar, crowns_json, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ctf, ntf = tarfile.open(chm_tar), tarfile.open(naip_tar)
    cn = {Path(m.name).stem: m.name for m in ctf.getmembers() if m.name.endswith(".tif")}
    nn = {Path(m.name).stem.rsplit("_", 1)[0]: m.name for m in ntf.getmembers() if m.name.endswith(".tif")}
    crowns_on_chm(ctf, cn, json.load(open(crowns_json)), out)
    crowns_hillshade(ctf, ntf, cn, nn, out)
    print(f"wrote {out/'crowns_on_chm.png'} and {out/'crowns_hillshade.png'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
