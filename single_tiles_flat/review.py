#!/usr/bin/env python3
"""
Tinder-style annotation quality reviewer for Z17 NAIP tiles.

Controls
--------
  P        accept (mark as good)
  Q        skip / reject
  Z        undo last decision
  ESC      quit (progress auto-saved)

Progress is saved to review_state.json after every keypress.
Re-running the script resumes where you left off.

Run with --finalize to copy accepted files into good/:
    python3 review.py --finalize
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPoly
from PIL import Image

# ── Directory layout ─────────────────────────────────────────────────────────

BASE = Path(__file__).parent

SPLITS = {
    "prefire":  {"ann": BASE / "prefire",  "img": BASE / "prefire_img"},
    "postfire": {"ann": BASE / "postfire", "img": BASE / "postfire_img"},
}

GOOD = {
    "prefire":  {"ann": BASE / "good" / "prefire",  "img": BASE / "good" / "prefire_img"},
    "postfire": {"ann": BASE / "postfire",           "img": BASE / "good" / "postfire_img"},
}
# override postfire ann destination
GOOD["postfire"]["ann"] = BASE / "good" / "postfire"

STATE_FILE = BASE / "review_state.json"

# ── Geo helpers ───────────────────────────────────────────────────────────────

def tile_bbox(z: int, x: int, y: int) -> tuple:
    """Return (west, south, east, north) WGS-84 for a Slippy Map tile."""
    n = 2 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


def geo_to_px(lon: float, lat: float, bbox: tuple, size: int = 256) -> tuple:
    """Map (lon, lat) → pixel (x, y) with y=0 at north (image convention)."""
    west, south, east, north = bbox
    px = (lon - west) / (east - west) * size
    py = (north - lat) / (north - south) * size
    return px, py


def load_polygons(ann_path: Path, bbox: tuple) -> list:
    """Return list of (N,2) pixel-coord arrays, one per polygon ring."""
    gj = json.loads(ann_path.read_text())
    out = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            rings = [r[0] for r in geom["coordinates"]]
        else:
            continue
        for ring in rings:
            pts = np.array([geo_to_px(c[0], c[1], bbox) for c in ring])
            out.append(pts)
    return out


def fire_name(ann_path: Path) -> str:
    try:
        gj = json.loads(ann_path.read_text())
        feats = gj.get("features", [])
        if feats:
            return feats[0]["properties"].get("FIRENAME", "")
    except Exception:
        pass
    return ""

# ── Reviewer ─────────────────────────────────────────────────────────────────

class Reviewer:
    def __init__(self):
        self.state = self._load_state()
        self.items = self._build_queue()
        self.n = len(self.items)
        self.idx = 0
        self.history: list[tuple] = []   # (idx, key, decision)

        if self.n == 0:
            total_reviewed = len(self.state["reviewed"])
            total_accepted = len(self.state["accepted"])
            print(f"Nothing left to review. "
                  f"{total_reviewed} reviewed, {total_accepted} accepted.")
            print("Run  python3 review.py --finalize  to copy accepted files.")
            return

        self._setup_figure()
        self.show()
        plt.show()

    # ── State I/O ─────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
        return {"reviewed": {}, "accepted": []}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _build_queue(self) -> list:
        reviewed = set(self.state["reviewed"].keys())
        items = []
        for split, cfg in SPLITS.items():
            for ann in sorted(cfg["ann"].glob("*.geojson")):
                key = f"{split}/{ann.stem}"
                if key not in reviewed:
                    items.append((split, ann.stem))
        already = len(reviewed)
        if already:
            print(f"Resuming: {already} already reviewed, "
                  f"{len(items)} remaining.")
        return items

    # ── Figure setup ──────────────────────────────────────────────────────────

    def _setup_figure(self):
        plt.rcParams["keymap.quit"] = []   # prevent 'q' from closing the window
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.patch.set_facecolor("#111")
        self.ax.set_facecolor("#111")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event",     self.on_close)
        plt.tight_layout(pad=0)

    # ── Display ───────────────────────────────────────────────────────────────

    def show(self):
        if self.idx >= self.n:
            n_acc = len(self.state["accepted"])
            self.ax.clear()
            self.ax.set_facecolor("#111")
            self.ax.text(0.5, 0.5,
                         f"All done!\n{self.n} reviewed\n{n_acc} accepted\n\n"
                         "Run  python3 review.py --finalize  to copy files.",
                         ha="center", va="center", color="white", fontsize=14,
                         transform=self.ax.transAxes)
            self.ax.axis("off")
            self.fig.canvas.draw()
            return

        split, stem = self.items[self.idx]
        parts = stem.split("_")
        z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
        bbox = tile_bbox(z, x, y)

        ann_path = SPLITS[split]["ann"] / f"{stem}.geojson"
        img_path = SPLITS[split]["img"] / f"{stem}.png"

        self.ax.clear()
        self.ax.set_facecolor("#111")

        # Image
        if img_path.exists():
            img = np.array(Image.open(img_path))
            self.ax.imshow(img, interpolation="bilinear")
        else:
            self.ax.text(0.5, 0.5, "image not found", ha="center", va="center",
                         color="gray", transform=self.ax.transAxes)

        # Annotation polygons
        poly_count = 0
        if ann_path.exists():
            polys = load_polygons(ann_path, bbox)
            poly_count = len(polys)
            for pts in polys:
                patch = MplPoly(pts, closed=True, fill=False,
                                edgecolor="#00ff88", linewidth=1.2, alpha=0.85)
                self.ax.add_patch(patch)

        self.ax.axis("off")

        fname = fire_name(ann_path) if ann_path.exists() else ""
        n_acc = len(self.state["accepted"])
        self.fig.suptitle(
            f"{split.upper()}   {stem}   {fname}\n"
            f"{poly_count} annotations     "
            f"[{self.idx + 1} / {self.n}]   accepted so far: {n_acc}\n"
            "P = good    Q = skip    Z = undo    ESC = quit",
            color="white", fontsize=9, y=0.99,
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Key handling ──────────────────────────────────────────────────────────

    def _record(self, decision: str):
        split, stem = self.items[self.idx]
        key = f"{split}/{stem}"
        self.state["reviewed"][key] = decision
        if decision == "accepted":
            self.state["accepted"].append(key)
        self.history.append((self.idx, key, decision))
        self._save_state()

    def on_key(self, event):
        if event.key == "p":
            self._record("accepted")
            self.idx += 1
            self.show()
        elif event.key == "q":
            self._record("rejected")
            self.idx += 1
            self.show()
        elif event.key == "z":
            self._undo()
        elif event.key == "escape":
            self._save_state()
            plt.close()

    def _undo(self):
        if not self.history:
            return
        prev_idx, key, decision = self.history.pop()
        del self.state["reviewed"][key]
        if decision == "accepted" and key in self.state["accepted"]:
            self.state["accepted"].remove(key)
        self._save_state()
        self.idx = prev_idx
        self.show()

    def on_close(self, _event):
        self._save_state()

# ── Finalize ─────────────────────────────────────────────────────────────────

def finalize():
    if not STATE_FILE.exists():
        print("No review_state.json found — nothing to finalize.")
        return

    state = json.loads(STATE_FILE.read_text())
    accepted = state.get("accepted", [])
    if not accepted:
        print("No accepted tiles yet.")
        return

    for split_cfg in GOOD.values():
        split_cfg["ann"].mkdir(parents=True, exist_ok=True)
        split_cfg["img"].mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for key in accepted:
        split, stem = key.split("/", 1)
        if split not in SPLITS:
            continue

        src_ann = SPLITS[split]["ann"] / f"{stem}.geojson"
        src_tif = SPLITS[split]["img"] / f"{stem}.tif"
        src_png = SPLITS[split]["img"] / f"{stem}.png"

        dst_ann = GOOD[split]["ann"] / f"{stem}.geojson"
        dst_tif = GOOD[split]["img"] / f"{stem}.tif"
        dst_png = GOOD[split]["img"] / f"{stem}.png"

        for src, dst in [(src_ann, dst_ann), (src_tif, dst_tif), (src_png, dst_png)]:
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"  missing: {src}")
                skipped += 1

    print(f"Finalized: {len(accepted)} tiles → good/")
    print(f"  {copied} files copied, {skipped} source files missing")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NAIP tile annotation reviewer")
    parser.add_argument("--finalize", action="store_true",
                        help="Copy accepted tiles to good/ and exit")
    args = parser.parse_args()

    if args.finalize:
        finalize()
    else:
        Reviewer()


if __name__ == "__main__":
    main()
