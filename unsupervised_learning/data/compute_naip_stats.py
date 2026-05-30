"""
Compute per-band mean and std across the NAIP patch pool. Result written to
stats.json next to manifest.csv. Detection / pretraining code reads this at
load time to normalize each band correctly.

Important: NIR has a very different distribution from RGB on NAIP — using
ImageNet stats on the NIR channel silently corrupts spectral information.
Always normalize with these corpus stats.

Uses a random sample of patches (default 2000) for speed; with O(50k) patches
in the pool the difference between a sample and the full pool is negligible.

Output JSON:
  {"mean": [r, g, b, nir], "std": [r, g, b, nir], "n_patches_sampled": N}
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent))
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
STATS_PATH = DATA_ROOT / "stats.json"

N_SAMPLE = int(os.environ.get("STATS_N_SAMPLE", "2000"))
SEED = 0


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"No manifest at {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    if len(df) == 0:
        raise SystemExit("Manifest is empty.")

    n = min(N_SAMPLE, len(df))
    rng = random.Random(SEED)
    rows = rng.sample(list(df["rel_path"]), n)
    print(f"Sampling {n} of {len(df)} patches")

    # Welford's online algorithm in float64 — single pass, no big array in RAM.
    # Tracks mean and M2 (sum of squared deviations) per channel separately.
    count = 0
    mean = np.zeros(4, dtype=np.float64)
    m2 = np.zeros(4, dtype=np.float64)

    for rel in tqdm(rows, desc="scan"):
        arr = np.load(DATA_ROOT / rel)  # (4, H, W) uint8
        # Cast to float32 in [0, 1] to match what the model sees at training time.
        x = arr.astype(np.float32) / 255.0  # (4, H, W)
        for c in range(4):
            ch = x[c].ravel()
            k = ch.size
            new_count = count + k
            delta = ch.mean() - mean[c]
            mean[c] += delta * k / new_count
            # M2 update — variance accumulated over all prior data plus this batch
            ch_var = ch.var() * k
            m2[c] += ch_var + delta * delta * count * k / new_count
        count += x[0].size  # same H*W for every channel

    var = m2 / max(count - 1, 1)
    std = np.sqrt(var)

    out = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_patches_sampled": n,
        "n_pixels_per_channel": int(count),
        "note": "values are for inputs scaled to [0, 1] (uint8 / 255)",
    }
    STATS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {STATS_PATH}")
    print(f"  mean: {mean}")
    print(f"  std:  {std}")


if __name__ == "__main__":
    main()
