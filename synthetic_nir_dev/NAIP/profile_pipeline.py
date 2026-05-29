"""
Profile the train.py input pipeline to find the bottleneck.

Isolates each stage so we can see where the ~2500s/epoch goes:
  1. raw byte read from disk        (disk I/O only)
  2. PIL open+decode+to-numpy       (decode only)
  3. full Dataset.__getitem__       (decode + tensor + augment, single proc)
  4. real DataLoader @ NUM_WORKERS  (the actual feed rate training sees)
  5. model forward+backward         (compute; CPU here -> caveated)

Run:  python profile_pipeline.py
"""

import os
import time

import numpy as np
import torch
from PIL import Image

import train  # reuse the exact dataset / model / constants

N_FILES = 200      # for raw-read / decode / getitem micro-benchmarks
N_BATCHES = 40     # for the DataLoader throughput test
N_COMPUTE = 5      # forward+backward steps for the compute probe


def bench(label, fn, n):
    # one warmup, then timed
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    print(f"{label:42s} {1e3*dt/n:8.2f} ms/it   {n/dt:8.1f} it/s")
    return dt / n


def main():
    print(f"cpu_count={os.cpu_count()}  torch={torch.__version__}  "
          f"cuda={torch.cuda.is_available()}  NUM_WORKERS={train.NUM_WORKERS}")

    ds = train.NaipPatchDataset("train", train.RESOLUTIONS_M, augment=True)
    print(f"train dataset: {len(ds)} patches\n")

    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(ds), size=N_FILES).tolist()
    rgb_paths = [train.ROOT / ds.df.iloc[i]["rgb_path"] for i in idxs]
    nir_paths = [train.ROOT / ds.df.iloc[i]["nir_path"] for i in idxs]

    # 1. raw byte read (disk only) ------------------------------------------
    it = iter(idxs)
    def read_bytes():
        i = next(it, None)
        if i is None:
            return
        for p in (train.ROOT / ds.df.iloc[i]["rgb_path"],
                  train.ROOT / ds.df.iloc[i]["nir_path"]):
            with open(p, "rb") as fh:
                fh.read()
    total_bytes = sum(os.path.getsize(p) for p in rgb_paths + nir_paths)
    print(f"avg file size: rgb+nir pair ~ {total_bytes/N_FILES/1024:.1f} KiB\n")
    it = iter(idxs)
    bench("1. raw byte read (rgb+nir pair)", read_bytes, N_FILES)

    # 2. PIL decode to numpy ------------------------------------------------
    di = iter(idxs)
    def decode():
        i = next(di, None)
        if i is None:
            return
        row = ds.df.iloc[i]
        np.array(Image.open(train.ROOT / row["rgb_path"]).convert("RGB"))
        np.array(Image.open(train.ROOT / row["nir_path"]).convert("L"))
    di = iter(idxs)
    bench("2. PIL open+decode->numpy (pair)", decode, N_FILES)

    # 3. full __getitem__ (decode + tensor + augment) -----------------------
    gi = iter(idxs)
    def getitem():
        i = next(gi, None)
        if i is None:
            return
        ds[i]
    gi = iter(idxs)
    per_item = bench("3. Dataset.__getitem__ (1 proc)", getitem, N_FILES)
    print(f"   -> single-thread ceiling: {1/per_item:7.1f} img/s\n")

    # 4. real DataLoader throughput -----------------------------------------
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=train.BATCH_SIZE, shuffle=True,
        num_workers=train.NUM_WORKERS, pin_memory=False,
        persistent_workers=train.NUM_WORKERS > 0,
    )
    it = iter(loader)
    next(it)  # warmup: pay worker spawn + first fill
    t0 = time.perf_counter()
    seen = 0
    for _ in range(N_BATCHES):
        rgb, nir = next(it)
        seen += rgb.shape[0]
    dt = time.perf_counter() - t0
    img_s = seen / dt
    print(f"4. DataLoader @ {train.NUM_WORKERS} workers"
          f"            {1e3*dt/N_BATCHES:8.2f} ms/batch  {img_s:8.1f} img/s")
    del it, loader
    print(f"   -> predicted pure-data epoch: {len(ds)/img_s/60:6.1f} min "
          f"(actual run1 epoch ~41.7 min)\n")

    # 5. compute probe (CPU here -> caveat) ---------------------------------
    dev = train.pick_device()
    model = train.build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rgb = torch.rand(train.BATCH_SIZE, 3, 256, 256, device=dev)
    nir = torch.rand(train.BATCH_SIZE, 1, 256, 256, device=dev)
    def step():
        pred = model(rgb)
        loss, _, _ = train.compute_loss(pred, nir, rgb)
        opt.zero_grad(); loss.backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
    per_step = bench(f"5. fwd+bwd bs={train.BATCH_SIZE} on {dev.type}", step, N_COMPUTE)
    print(f"   -> compute-only epoch: {len(ds)/train.BATCH_SIZE*per_step/60:6.1f} min "
          f"({'GPU would be far faster' if dev.type!='cuda' else 'on this GPU'})")


if __name__ == "__main__":
    main()
