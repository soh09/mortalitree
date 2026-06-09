"""Entry point: train Stage B then Stage C."""
import argparse
import torch
import yaml

from src.model.clay_loader import load_clay_encoder
from src.model.detector import TreeDetector
from src.train.stage_b import run_stage_b
from src.train.stage_c import run_stage_c


def build_model(cfg: dict, clay_checkpoint: str) -> TreeDetector:
    encoder = load_clay_encoder(clay_checkpoint)
    m = cfg["model"]
    return TreeDetector(
        encoder=encoder,
        neck_in_channels=m["neck_in_channels"],
        neck_out_channels=m["neck_out_channels"],
        hidden=m["hidden"],
        n_heads=m["n_heads"],
        n_decoder_layers=m["n_decoder_layers"],
        dropout=m["dropout"],
        dynamic_query_list=m.get("dynamic_query_list", (200, 400, 600)),
        ccm_cls_num=m.get("ccm_cls_num", 3),
        ccm_params=m.get("ccm_params", (100, 300)),
        anchor_size=m.get("anchor_size", 0.05),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["b", "c", "both"], default="both")
    parser.add_argument("--clay_checkpoint", required=True)
    parser.add_argument("--stage_b_annotations", default=None)
    parser.add_argument("--train_annotations", default=None)
    parser.add_argument("--val_annotations",   default=None)
    parser.add_argument("--stage_b_checkpoint", default=None,
                        help="Stage B checkpoint to load for Stage C")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--norm_stats",     default="configs/naip_normalization.yaml")
    parser.add_argument("--device",         default=None)
    args = parser.parse_args()

    with open("configs/stage_b.yaml") as f:
        cfg_b = yaml.safe_load(f)
    with open("configs/stage_c.yaml") as f:
        cfg_c = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage in ("b", "both"):
        assert args.stage_b_annotations, "--stage_b_annotations required for Stage B"
        model = build_model(cfg_b, args.clay_checkpoint)
        run_stage_b(
            model=model,
            annotations_path=args.stage_b_annotations,
            checkpoint_dir=args.checkpoint_dir,
            total_epochs=cfg_b["training"]["total_epochs"],
            batch_size=cfg_b["training"]["batch_size"],
            lr=cfg_b["training"]["lr"],
            weight_decay=cfg_b["training"]["weight_decay"],
            warmup_epochs=cfg_b["training"]["warmup_epochs"],
            val_fraction=cfg_b["training"]["val_fraction"],
            lam_cls=cfg_b["loss"]["lam_cls"],
            lam_l1=cfg_b["loss"]["lam_l1"],
            lam_giou=cfg_b["loss"]["lam_giou"],
            lam_ccm=cfg_b["loss"].get("lam_ccm", 1.0),
            lam_enc=cfg_b["loss"].get("lam_enc", 1.0),
            cgfe_start_epoch=cfg_b["training"].get("cgfe_start_epoch"),
            device=device,
            num_workers=cfg_b["training"]["num_workers"],
        )

    if args.stage in ("c", "both"):
        assert args.train_annotations, "--train_annotations required for Stage C"
        assert args.val_annotations,   "--val_annotations required for Stage C"
        model = build_model(cfg_c, args.clay_checkpoint)
        stage_b_ckpt = args.stage_b_checkpoint
        if stage_b_ckpt is None and args.stage == "both":
            stage_b_ckpt = f"{args.checkpoint_dir}/stage_b_best.pt"
        run_stage_c(
            model=model,
            train_annotations_path=args.train_annotations,
            val_annotations_path=args.val_annotations,
            checkpoint_dir=args.checkpoint_dir,
            norm_stats_path=args.norm_stats,
            total_epochs=cfg_c["training"]["total_epochs"],
            frozen_epochs=cfg_c["training"]["frozen_epochs"],
            batch_size=cfg_c["training"]["batch_size"],
            neck_head_lr=cfg_c["training"]["neck_head_lr"],
            encoder_lr=cfg_c["training"]["encoder_lr"],
            weight_decay=cfg_c["training"]["weight_decay"],
            warmup_epochs=cfg_c["training"]["warmup_epochs"],
            unfreeze_n_blocks=cfg_c["training"]["unfreeze_n_blocks"],
            lam_cls=cfg_c["loss"]["lam_cls"],
            lam_l1=cfg_c["loss"]["lam_l1"],
            lam_giou=cfg_c["loss"]["lam_giou"],
            lam_ccm=cfg_c["loss"].get("lam_ccm", 1.0),
            lam_enc=cfg_c["loss"].get("lam_enc", 1.0),
            cgfe_start_epoch=cfg_c["training"].get("cgfe_start_epoch"),
            device=device,
            num_workers=cfg_c["training"]["num_workers"],
            stage_b_checkpoint=stage_b_ckpt,
        )


if __name__ == "__main__":
    main()
