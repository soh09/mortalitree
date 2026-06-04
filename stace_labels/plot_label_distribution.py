"""Plot the distribution of number of labels (bounding boxes) per patch."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LABELS_CSV = "eval_set/labels.csv"
THRESHOLDS = [50, 100, 150, 200]

df = pd.read_csv(LABELS_CSV)
counts = df.groupby("imgname").size().sort_values()

print(f"Total patches:  {len(counts)}")
print(f"Total labels:   {len(df)}")
print(f"Labels/patch:   min={counts.min()}, median={counts.median():.0f}, "
      f"mean={counts.mean():.1f}, max={counts.max()}")
print()
for t in THRESHOLDS:
    n = (counts < t).sum()
    print(f"  Patches with < {t:3d} labels: {n:3d} / {len(counts)}  "
          f"({100 * n / len(counts):.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Label distribution per patch", fontsize=14, fontweight="bold")

# --- histogram ---
ax = axes[0]
ax.hist(counts, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
for t in THRESHOLDS:
    ax.axvline(t, color="tomato", linestyle="--", linewidth=1,
               label=f"< {t}: {(counts < t).sum()} patches")
ax.set_xlabel("Labels per patch")
ax.set_ylabel("Number of patches")
ax.set_title("Histogram")
ax.legend(fontsize=8, title="Threshold lines")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# --- sorted bar (rank plot) ---
ax2 = axes[1]
ax2.bar(range(len(counts)), counts.values, color="steelblue", width=1.0)
for t in THRESHOLDS:
    ax2.axhline(t, color="tomato", linestyle="--", linewidth=1,
                label=f"< {t}: {(counts < t).sum()} patches")
ax2.set_xlabel("Patch rank (sorted by label count)")
ax2.set_ylabel("Labels per patch")
ax2.set_title("Sorted patch label counts")
ax2.legend(fontsize=8, title="Threshold lines")

plt.tight_layout()
out = "label_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
