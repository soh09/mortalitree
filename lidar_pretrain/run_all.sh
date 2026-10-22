#!/usr/bin/env bash
# Train and score arms A, B, C for several seeds, then print the summary table.
#   bash lidar_pretrain/run_all.sh WORK [SEEDS...]      (default seeds: 0 1 2)
set -euo pipefail
WORK=${1:?usage: run_all.sh WORK [SEEDS...]}; shift || true
SEEDS=${@:-0 1 2}
cd "$(dirname "$0")"
for s in $SEEDS; do
  python train.py "$WORK" --seed "$s" 2>&1 | tee -a "$WORK/train_seed$s.log"
done
python summarize.py "$WORK" | tee "$WORK/summary.md"
