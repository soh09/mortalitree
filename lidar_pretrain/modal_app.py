"""
Run the 3DEP-pretraining experiment on Modal (see README.md for what it tests).

Data never touches the laptop: each fire's CHM tars are downloaded inside a Modal
container straight from the dataset server, eligible chips are extracted to the
volume `lidar-pretrain-work`, pre-fire NAIP is read from Planetary Computer, and
training runs one GPU container per seed, logging to wandb (secret `wandb`).

Headless (nothing depends on the laptop staying connected):

    modal deploy lidar_pretrain/modal_app.py                    # once, and after any code change
    python lidar_pretrain/modal_app.py --stage train            # spawns `pipeline` on Modal and exits
    python lidar_pretrain/modal_app.py --stage all              # select -> build -> pack -> train -> summary
    python lidar_pretrain/modal_app.py --stage train --smoke    # 1-epoch GPU check, own wandb group
    python lidar_pretrain/modal_app.py --profile full           # every eligible chip (8 build shards), arms B and C,
                                                                # results in runs_full/, wandb groups full-seed*
    python lidar_pretrain/modal_app.py --profile full2 --stage train   # same set, + grad clipping and
                                                                # per-epoch checkpoints (resumable); runs_full2/

Follow progress in the Modal dashboard (app `lidar-pretrain`), in wandb, or in
/work/pipeline_status.json (`modal volume get lidar-pretrain-work pipeline_status.json -`).
Interactive (tied to this terminal): `modal run lidar_pretrain/modal_app.py --stage ...`.

Volume layout (/work): chm_chips/, chips.json, coverage.json, pretrain/{images,pretrain.csv,...},
runs/seed{S}/... (metrics, raw predictions, pretrained.ckpt). Download results with
`modal volume get lidar-pretrain-work runs ./runs`.
"""
import json, shutil, subprocess, sys, threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORK = "/work"
WANDB_PROJECT = "mortalitree-lidar-pretrain"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("deepforest==2.1.0", "rasterio", "pandas", "pillow", "pyproj", "shapely",
                 "scikit-image", "scipy", "wandb")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, "/repo/lidar_pretrain", ignore=["__pycache__", "lightning_logs"])
    .add_local_file(REPO / "baselines" / "chm_crowns.py", "/repo/baselines/chm_crowns.py")
    .add_local_dir(REPO / "single_tiles_flat" / "stage_c_data", "/repo/single_tiles_flat/stage_c_data",
                   ignore=["viewer", "readable", "predictions_all.json"])
)
app = modal.App("lidar-pretrain", image=image)
vol = modal.Volume.from_name("lidar-pretrain-work", create_if_missing=True)
FIRES = ["czu", "scu", "north", "creek", "castle"]


def run(cmd, commit_every=300):
    """Run a script, committing the volume every few minutes so progress survives preemption."""
    print("$", " ".join(cmd), flush=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(commit_every):
            try:
                vol.commit()
            except Exception as e:
                print("commit failed:", e)
    t = threading.Thread(target=committer, daemon=True); t.start()
    try:
        subprocess.run(cmd, check=True, cwd="/repo/lidar_pretrain")
    finally:
        stop.set(); vol.commit()


@app.function(cpu=2, memory=4096, timeout=6 * 3600, volumes={WORK: vol})
def select_fire(fire: str):
    run([sys.executable, "select_chips.py", WORK, "--fires", fire, "--suffix", f"_{fire}",
         "--tar-dir", "/tmp/chm", "--delete-tars"])
    return json.loads(Path(WORK, f"coverage_{fire}.json").read_text())


@app.function(cpu=1, volumes={WORK: vol})
def merge_chips():
    vol.reload()
    chips, cov = [], {}
    for f in FIRES:
        p = Path(WORK, f"chips_{f}.json")
        if p.exists():
            chips += json.loads(p.read_text()); cov.update(json.loads(Path(WORK, f"coverage_{f}.json").read_text()))
    Path(WORK, "chips.json").write_text(json.dumps(chips))
    Path(WORK, "coverage.json").write_text(json.dumps(cov, indent=1))
    vol.commit()
    return cov


@app.function(cpu=8, memory=16384, timeout=8 * 3600, volumes={WORK: vol})
def build(max_chips: int = 400, workers: int = 24):
    vol.reload()
    run([sys.executable, "build_pretrain.py", WORK, "--max-chips", str(max_chips), "--workers", str(workers)])
    return json.loads(Path(WORK, "pretrain", "chips_used.json").read_text())


def tar_dir(src, dst):
    """Directory of PNGs -> one tar (one large file on the volume instead of thousands)."""
    import tarfile
    with tarfile.open(dst, "w") as tf:
        for p in sorted(Path(src).iterdir()):
            tf.add(p, arcname=p.name)


@app.function(cpu=8, memory=16384, timeout=8 * 3600, volumes={WORK: vol})
def build_shard(shard: int, num_shards: int, out: str, max_chips: int = 0, workers: int = 16):
    """One shard of the chip list; PNGs go to local disk and reach the volume as one tar."""
    vol.reload()
    d = Path(WORK, out, f"shard{shard}")
    if d.exists():                                     # images are local, so never resume a shard
        shutil.rmtree(d)
    run([sys.executable, "build_pretrain.py", WORK, "--max-chips", str(max_chips), "--workers", str(workers),
         "--out", out, "--shard", str(shard), "--num-shards", str(num_shards), "--images-dir", "/tmp/imgs"])
    tar_dir("/tmp/imgs", "/tmp/images.tar"); shutil.copy("/tmp/images.tar", Path(WORK, out, f"images_shard{shard}.tar"))
    vol.commit()
    used = json.loads((d / "chips_used.json").read_text())
    return {"shard": shard, "chips": len(used), "images": sum(u["windows"] for u in used)}


@app.function(cpu=2, memory=16384, volumes={WORK: vol})
def merge_shards(out: str, num_shards: int):
    import pandas as pd
    vol.reload()
    root = Path(WORK, out)
    csvs = [pd.read_csv(root / f"shard{k}" / "pretrain.csv") for k in range(num_shards)]
    used = sum((json.loads((root / f"shard{k}" / "chips_used.json").read_text()) for k in range(num_shards)), [])
    pd.concat(csvs).to_csv(root / "pretrain.csv", index=False)
    (root / "chips_used.json").write_text(json.dumps(used, indent=1))
    vol.commit()
    by_fire = {}
    for u in used:
        f = by_fire.setdefault(u["fire"], {"chips": 0, "images": 0}); f["chips"] += 1; f["images"] += u["windows"]
    return {"chips": len(used), "images": sum(u["windows"] for u in used), "boxes": int(sum(len(c) for c in csvs)),
            "by_fire": by_fire}


@app.function(cpu=8, memory=8192, timeout=3 * 3600, volumes={WORK: vol})
def pack():
    """Pretraining PNGs -> one tar on the volume (thousands of small volume reads are slow;
    one large file is fast). Reads in parallel."""
    import io, tarfile
    from concurrent.futures import ThreadPoolExecutor
    vol.reload()
    src = Path(WORK, "pretrain", "images"); dst = Path(WORK, "pretrain", "images.tar")
    names = sorted(p.name for p in src.iterdir())
    tmp = Path("/tmp/images.tar")
    with tarfile.open(tmp, "w") as tf, ThreadPoolExecutor(64) as pool:
        for i, (n, b) in enumerate(zip(names, pool.map(lambda n: (src / n).read_bytes(), names))):
            ti = tarfile.TarInfo(n); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(names)}", flush=True)
    shutil.copy(tmp, dst); vol.commit()
    return f"{len(names)} images, {dst.stat().st_size/1e6:.0f} MB -> {dst}"


@app.function(gpu="L4", cpu=8, memory=32768, timeout=20 * 3600, volumes={WORK: vol},
              secrets=[modal.Secret.from_name("wandb")],
              retries=modal.Retries(max_retries=2, initial_delay=30.0))   # resumes from per-epoch checkpoints
def train(seed: int, smoke: bool = False, extra: str = "", pretrain_set: str = "pretrain", runs: str = "runs",
          arms: str = "A,B,C", group: str = ""):
    import tarfile
    vol.reload()
    # pretraining images onto local disk: thousands of small reads are faster there
    src = Path(WORK, pretrain_set); local = Path("/tmp", pretrain_set)
    if not local.exists():
        (local / "images").mkdir(parents=True)
        tars = sorted(src.glob("images*.tar"))
        for t in tars:
            with tarfile.open(t) as tf:
                tf.extractall(local / "images")
        if not tars:
            shutil.copytree(src / "images", local / "images", dirs_exist_ok=True)
        shutil.copy(src / "pretrain.csv", local / "pretrain.csv")
        print(f"pretraining images on local disk: {len(list((local / 'images').iterdir()))} from {len(tars)} tar(s)", flush=True)
    work = f"{WORK}/smoke" if smoke else WORK
    Path(work).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "train.py", work, "--seed", str(seed), "--workers", "6",
           "--pretrain-csv", str(local / "pretrain.csv"), "--labelled-dir", "/tmp/labelled",
           "--wandb-project", WANDB_PROJECT, "--runs-dir", runs, "--arms", *arms.split(","),
           "--wandb-group-prefix", ("smoke-" if smoke else "") + group]
    if smoke:   # separate folder and wandb group, so it can't mix with real results
        cmd += ["--finetune-epochs", "1", "--pretrain-epochs", "1", "--max-pretrain-images", "64"]
    run(cmd + extra.split())
    root = Path(work, runs, f"seed{seed}")
    return {arm: json.loads((root / arm / "metrics.json").read_text())["metrics"]["test_all_tiles"]
            for arm in ("A", "B", "C") if (root / arm / "metrics.json").exists()}


@app.function(cpu=4, memory=8192, volumes={WORK: vol})
def summary(runs: str = "runs", baseline_runs: str = ""):
    vol.reload()
    out = subprocess.run([sys.executable, "summarize.py", WORK, runs], capture_output=True, text=True,
                         cwd="/repo/lidar_pretrain")
    rs_dirs = [baseline_runs, runs] if baseline_runs else [runs]
    rs = subprocess.run([sys.executable, "rescore_by_period.py", WORK, *rs_dirs], capture_output=True, text=True,
                        cwd="/repo/lidar_pretrain")
    text = out.stdout + "\n\n# pre-/post-fire test split\n" + "\n".join(
        l for l in rs.stdout.splitlines() if "Warning" not in l)
    Path(WORK, f"summary_{runs}.md").write_text(text); vol.commit()
    return text + out.stderr[-2000:] + rs.stderr[-2000:]


def status_file(runs):
    return Path(WORK, "pipeline_status.json" if runs == "runs" else f"pipeline_status_{runs}.json")


def make_status(runs):
    def status(**kw):
        p = status_file(runs)
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(kw); p.write_text(json.dumps(cur, indent=1)); vol.commit()
        print("status:", json.dumps(kw), flush=True)
    return status


PROFILES = {   # named configurations; "full" = every eligible chip, compared against the 400-per-fire A arm
    "default": dict(pretrain_set="pretrain", runs="runs", arms="A,B,C", group="", baseline_runs=""),
    "full": dict(pretrain_set="pretrain_full", runs="runs_full", arms="B,C", group="full-", baseline_runs="runs"),
    # full v2: same data and settings plus gradient clipping and per-epoch checkpoints (v1 seed 2 diverged)
    "full2": dict(pretrain_set="pretrain_full", runs="runs_full2", arms="B,C", group="full2-", baseline_runs="runs",
                  extra="--pretrain-grad-clip 10 --pretrain-ckpt-every-epoch"),
}


@app.function(cpu=1, timeout=24 * 3600, volumes={WORK: vol})
def pipeline(stage: str = "all", seeds: str = "0,1,2", max_chips: int = 400, smoke: bool = False, extra: str = "",
             profile: str = "default"):
    """Runs the stages server-side, so a spawned call survives the laptop disconnecting."""
    import datetime
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    status = make_status(PROFILES[profile]["runs"])
    status(started=now(), stage=stage, seeds=seeds, smoke=smoke, extra=extra, profile=profile, state="running",
           train=None, summary=None, error=None)
    try:
        _stages(stage, seeds, max_chips, smoke, extra, profile, status)
        status(state="done", finished=now())
    except Exception as e:
        status(state="failed", error=f"{type(e).__name__}: {e}"[:500], finished=now())
        raise


def _stages(stage, seeds, max_chips, smoke, extra, profile="default", report=lambda **k: None):
    P = PROFILES[profile]
    S = set(stage.split(","))                       # e.g. "build,train"
    if S & {"all", "select"}:
        for cov in select_fire.map(FIRES):
            print(json.dumps(cov))
        cov = merge_chips.remote(); print(json.dumps(cov, indent=1))
        report(select={f: c["eligible_chips"] for f, c in cov.items()})
    if S & {"all", "build"} and profile == "default":
        used = build.remote(max_chips)
        print(f"pretraining set: {len(used)} chips, {sum(d['windows'] for d in used)} images")
        report(build={"chips": len(used), "images": sum(d["windows"] for d in used)})
        report(pack=pack.remote())
    if S & {"all", "build"} and profile not in ("default", "full2"):
        n = 8
        shards = list(build_shard.map(range(n), kwargs={"num_shards": n, "out": P["pretrain_set"], "max_chips": 0}))
        print(json.dumps(shards))
        m = merge_shards.remote(P["pretrain_set"], n); print(json.dumps(m))
        report(build=m)
    if "pack" in S:
        report(pack=pack.remote())
    if S & {"all", "train"}:
        report(train="running")
        done = {}
        kw = {"smoke": smoke, "extra": " ".join(x for x in (P.get("extra", ""), extra) if x),
              **{k: P[k] for k in ("pretrain_set", "runs", "arms", "group")}}
        # one seed failing must not cancel the others
        for seed, res in zip(seeds.split(","), train.map([int(s) for s in seeds.split(",")], kwargs=kw,
                                                          return_exceptions=True)):
            if isinstance(res, BaseException):
                done[f"seed{seed}"] = {"error": f"{type(res).__name__}: {res}"[:300]}
                print(f"seed {seed} failed: {res}", flush=True); report(train=done); continue
            done[f"seed{seed}"] = {k: {i: v[i]["f1"] for i in ("iou0.4", "iou0.5")} | {"mae": v["count_mae"]}
                                   for k, v in res.items()}
            print(f"seed {seed}: " + json.dumps(done[f"seed{seed}"]))
            report(train=done)
    if S & {"all", "summary", "train"} and not smoke:
        table = summary.remote(P["runs"], P["baseline_runs"]); print(table)
        report(summary=table)


@app.local_entrypoint()
def main(stage: str = "all", seeds: str = "0,1,2", max_chips: int = 400, smoke: bool = False, extra: str = "",
         profile: str = "default"):
    """Interactive: runs the stages from this terminal (stops if it disconnects)."""
    _stages(stage, seeds, max_chips, smoke, extra, profile)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="spawn the pipeline on the deployed app and exit")
    ap.add_argument("--stage", default="all"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-chips", type=int, default=400); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra", default=""); ap.add_argument("--profile", default="default", choices=list(PROFILES))
    a = ap.parse_args()
    call = modal.Function.from_name("lidar-pretrain", "pipeline").spawn(a.stage, a.seeds, a.max_chips, a.smoke, a.extra,
                                                                        a.profile)
    print(f"spawned pipeline (stage={a.stage}, profile={a.profile}, seeds={a.seeds}, smoke={a.smoke}): "
          f"call {call.object_id}")
