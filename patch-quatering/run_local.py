"""Local (no-Modal) launcher for the quartered Stage-C fine-tune.

Runs the same `run_stage_c_quartered` pipeline as `modal_train.py::stage_c_quarter`,
but against locally-available data and on this machine's GPU (CUDA), Apple GPU
(MPS), or CPU. Use this when the stage_c_data tiffs + JSONs live on disk rather
than on the Modal volumes.

Prerequisites (one-time):
  1. Clay model package (provides the encoder):
       pip install "git+https://github.com/Clay-foundation/model.git"
  2. Clay v1.5 checkpoint (~1.5 GB), e.g.:
       curl -L -o clay-v1.5.ckpt \\
         https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt

The annotation JSONs store absolute Modal paths (/data/stage_c/tiles/...); this
script ignores those directories and reads the tiffs from --tiles-root instead
(default: <stage-c-dir>/tiles), so the JSONs don't need rewriting.

Example:
  python run_local.py \\
    --stage-c-dir /Users/so/Desktop/trees/single_tiles_flat/stage_c_data \\
    --stage-b-checkpoint /Users/so/Desktop/trees/clay/stage_b_best.pt \\
    --clay-ckpt ./clay-v1.5.ckpt
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_STAGE_C = "/Users/so/Desktop/trees/single_tiles_flat/stage_c_data"
DEFAULT_STAGE_B = "/Users/so/Desktop/trees/clay/stage_b_best.pt"


def _pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-c-dir", default=DEFAULT_STAGE_C,
                    help="dir holding train/val/test.json + tiles/ (default: %(default)s)")
    ap.add_argument("--train-json", default=None, help="override train annotations path")
    ap.add_argument("--val-json", default=None, help="override val annotations path")
    ap.add_argument("--test-json", default=None, help="override test annotations path")
    ap.add_argument("--tiles-root", default=None,
                    help="dir with the tiffs (default: <stage-c-dir>/tiles)")
    ap.add_argument("--stage-b-checkpoint", default=DEFAULT_STAGE_B,
                    help="quarter Stage-B weights to fine-tune (default: %(default)s)")
    ap.add_argument("--clay-ckpt", default=str(HERE / "clay-v1.5.ckpt"),
                    help="Clay v1.5 encoder checkpoint (default: %(default)s)")
    ap.add_argument("--config", default=str(HERE / "configs" / "stage_c_quarter.yaml"))
    ap.add_argument("--checkpoint-dir", default=str(HERE / "checkpoints_local"))
    ap.add_argument("--num-queries", type=int, default=0,
                    help="0 = auto-detect from the Stage-B checkpoint (recommended)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override dataloader workers (default: from config)")
    ap.add_argument("--epochs", type=int, default=None, help="override total_epochs")
    ap.add_argument("--log-dir", default=None,
                    help="where metrics.csv + preds/ + panels/ go (default: checkpoint dir)")
    ap.add_argument("--save-preds-every", type=int, default=1,
                    help="dump stitched val preds JSON every N epochs (0=off)")
    ap.add_argument("--panels-every", type=int, default=1,
                    help="render val GT-vs-pred PNG panels every N epochs (0=off)")
    ap.add_argument("--n-panel-tiles", type=int, default=4,
                    help="how many tiles per panel PNG")
    ap.add_argument("--panel-conf", type=float, default=0.25,
                    help="confidence threshold for boxes DRAWN in panels (decoupled "
                         "from the eval conf_thresh used for F1/metrics); lower it to "
                         "see preds while the model is still undertrained")
    args = ap.parse_args()

    # Let unsupported MPS ops fall back to CPU instead of erroring out.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, str(HERE))

    import torch
    import yaml

    sc = Path(args.stage_c_dir)
    train_json = args.train_json or str(sc / "train.json")
    val_json   = args.val_json   or str(sc / "val.json")
    test_json  = args.test_json  or str(sc / "test.json")
    tiles_root = args.tiles_root or str(sc / "tiles")

    for p in (train_json, val_json, test_json, tiles_root,
              args.stage_b_checkpoint, args.config):
        if not Path(p).exists():
            sys.exit(f"[run_local] missing required path: {p}")
    if not Path(args.clay_ckpt).exists():
        sys.exit(
            f"[run_local] Clay encoder checkpoint not found: {args.clay_ckpt}\n"
            "Download it once, e.g.:\n"
            "  curl -L -o clay-v1.5.ckpt "
            "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
        )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    m, t, l = cfg["model"], cfg["training"], cfg["loss"]
    d, e = cfg.get("data", {}), cfg.get("eval", {})

    # num_queries must match the Stage-B checkpoint (the query embedding is part of
    # the saved weights). Auto-detect unless explicitly overridden.
    num_queries = args.num_queries
    if not num_queries:
        sd = torch.load(args.stage_b_checkpoint, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        num_queries = int(sd["head.queries.weight"].shape[0])
        print(f"[run_local] num_queries from checkpoint: {num_queries}")

    device = _pick_device(args.device)
    torch.manual_seed(42)
    print(f"[run_local] device={device}")

    from src.model.clay_loader import load_clay_encoder
    from src.model.detector import TreeDetector
    from src.train.stage_c import run_stage_c_quartered

    encoder = load_clay_encoder(args.clay_ckpt)
    model = TreeDetector(
        encoder,
        neck_in_channels=m["neck_in_channels"],
        neck_out_channels=m["neck_out_channels"],
        num_queries=num_queries,
        hidden=m["hidden"],
        n_heads=m["n_heads"],
        n_decoder_layers=m["n_decoder_layers"],
        dropout=m["dropout"],
    )

    # configs store a repo-relative norm-stats path; resolve it next to this script.
    norm_stats = d.get("norm_stats_path", "configs/naip_normalization.yaml")
    if not Path(norm_stats).is_absolute():
        norm_stats = str(HERE / norm_stats)

    res = run_stage_c_quartered(
        model=model,
        train_annotations_path=train_json,
        val_annotations_path=val_json,
        test_annotations_path=test_json,
        checkpoint_dir=args.checkpoint_dir,
        norm_stats_path=norm_stats,
        tiles_root=tiles_root,
        tile_size=d.get("tile_size", 256),
        n_split=d.get("n_split", 2),
        total_epochs=args.epochs or t["total_epochs"],
        frozen_epochs=t["frozen_epochs"],
        batch_size=t["batch_size"],
        neck_head_lr=t["neck_head_lr"],
        encoder_lr=t["encoder_lr"],
        weight_decay=t["weight_decay"],
        warmup_epochs=t["warmup_epochs"],
        unfreeze_n_blocks=t["unfreeze_n_blocks"],
        patience=t.get("patience", 15),
        lam_cls=l["lam_cls"],
        lam_l1=l["lam_l1"],
        lam_giou=l["lam_giou"],
        conf_thresh=e.get("conf_thresh", 0.5),
        nms_iou=e.get("nms_iou", 0.6),
        device=device,
        num_workers=args.num_workers if args.num_workers is not None else t["num_workers"],
        stage_b_checkpoint=args.stage_b_checkpoint,
        log_dir=args.log_dir,
        save_preds_every=args.save_preds_every,
        panels_every=args.panels_every,
        n_panel_tiles=args.n_panel_tiles,
        panel_conf_thresh=args.panel_conf,
    )
    msg = f"[run_local] done. best_val_f1={res['best_val_f1']:.4f}"
    if res.get("test") is not None:
        msg += (f"  test_f1@val-best({res['best_conf']:.2f})={res['test_f1_bestth']:.4f}"
                f"  (f1@0.5={res['test']['f1_at_0.5']:.4f})")
    print(msg + f"  | checkpoints in {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
