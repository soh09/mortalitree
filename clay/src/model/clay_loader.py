"""Load Clay v1.5 encoder and wrap it for batch-dict forward pass.

Clay v1.5 (the released v1.5.0 checkpoint) is the *large* MAE: dim=1024,
depth=24, patch_size=8. For a 256x256 input that yields a 32x32 grid of
1024-dim patch tokens at stride 8 (not the 16x16 / 768-dim / stride-16 grid a
patch-16 base model would give). The wrapper below reads these dimensions off
the encoder at runtime instead of hardcoding them.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat


# Encoder dims per Clay model size (from claymodel.model.clay_mae_*).
_CLAY_SIZE_ARGS = {
    "tiny":  dict(dim=192,  depth=6,  heads=4,  dim_head=48, mlp_ratio=2),
    "small": dict(dim=384,  depth=6,  heads=6,  dim_head=64, mlp_ratio=2),
    "base":  dict(dim=768,  depth=12, heads=12, dim_head=64, mlp_ratio=4),
    "large": dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4),
}


def load_clay_encoder(checkpoint_path: str, strict: bool = False) -> nn.Module:
    """Load Clay v1.5's encoder from a Lightning checkpoint.

    Builds *only* the Encoder and copies the encoder weights out of the
    checkpoint's state_dict, rather than reconstructing the full ClayMAE module.
    This skips instantiating the DINOv2 teacher (a ~1 GB timm pretrained download
    that is never used at inference/finetuning time). Mask/shuffle are disabled so
    the encoder returns all patch tokens in their original spatial order.
    """
    try:
        from claymodel.model import Encoder
    except ImportError as e:
        raise ImportError(
            "claymodel not found. Install from https://github.com/Clay-foundation/model"
        ) from e

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {}) or {}
    model_size = hparams.get("model_size", "large")   # released v1.5.0 is large
    patch_size = hparams.get("patch_size", 8)
    if model_size not in _CLAY_SIZE_ARGS:
        raise ValueError(f"Unknown Clay model_size '{model_size}'")

    encoder = Encoder(
        mask_ratio=0.0,
        patch_size=patch_size,
        shuffle=False,
        **_CLAY_SIZE_ARGS[model_size],
    )

    # Pull the encoder weights out of the full-model state dict (keys are prefixed
    # "model.encoder."). Fall back to other layouts if that prefix isn't present.
    state_dict = ckpt.get("state_dict", ckpt)
    enc_state = {
        k[len("model.encoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("model.encoder.")
    }
    if not enc_state:
        enc_state = {
            k[len("encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
    if not enc_state:
        enc_state = state_dict  # assume it is already an encoder-only state dict

    missing, _ = encoder.load_state_dict(enc_state, strict=strict)
    if "cls_token" in missing:
        raise RuntimeError(
            "Encoder weights not found in checkpoint (cls_token missing) — the "
            "encoder would be randomly initialized. Check the checkpoint format."
        )

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


class ClayEncoderWrapper(nn.Module):
    """Wraps the Clay encoder with batch-dict input and spatial output.

    Mirrors Clay's own finetuning encoder (claymodel/finetune/segment/factory.py
    SegmentEncoder.forward): it runs patch-embed → positional/metadata encoding →
    transformer, skipping mask_out/shuffle so the patch tokens stay in spatial
    order. Returns a feature map of shape (B, D, H_p, W_p) where D = encoder.dim
    and H_p = W_p = input_size // patch_size.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    @property
    def dim(self) -> int:
        return self.encoder.dim

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch keys: pixels (B,C,H,W), wavelengths (C,), gsd (B,),
                    time (B,2), latlon (B,2)
        Returns spatial feature map: (B, encoder.dim, H_p, W_p)
        """
        enc = self.encoder
        cube = batch["pixels"]
        B, C, H, W = cube.shape

        # Patchify + per-patch wavelength embeddings, then add pos/metadata
        # encoding. Clay's dynamic embedding handles a variable number of bands,
        # so 3-band (RGB) or 4-band (RGB+NIR) input both work here.
        # Clay builds one positional encoding for the whole batch, so gsd must be
        # a scalar (posemb_sincos_2d_with_gsd multiplies a (dim//4,) tensor by it).
        # Our tiles share a fixed GSD, so collapse the (B,) gsd to a scalar.
        gsd = batch["gsd"]
        if torch.is_tensor(gsd) and gsd.ndim > 0:
            gsd = gsd[0]

        patches, _ = enc.to_patch_embed(cube, batch["wavelengths"])
        patches = enc.add_encodings(patches, batch["time"], batch["latlon"], gsd)

        # Prepend the cls token, run the transformer, then drop the cls token.
        cls_tokens = repeat(enc.cls_token, "1 1 D -> B 1 D", B=B)
        patches = torch.cat((cls_tokens, patches), dim=1)
        patches = enc.transformer(patches)
        patches = patches[:, 1:, :]                       # (B, H_p*W_p, D)

        H_p = H // enc.patch_size
        W_p = W // enc.patch_size
        expected = H_p * W_p
        assert patches.shape[1] == expected, (
            f"Expected {expected} patch tokens ({H_p}x{W_p}), got "
            f"{patches.shape[1]}. Check input size vs patch_size={enc.patch_size}."
        )
        return rearrange(patches, "B (H W) D -> B D H W", H=H_p, W=W_p)

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n transformer blocks of the Clay encoder.

        Clay's Encoder stores blocks at encoder.transformer.layers (a ModuleList
        of [attn, ff] block pairs inside the Transformer wrapper class).
        """
        transformer = self.encoder.transformer
        blocks = None
        for attr in ("layers", "blocks", "transformer_blocks"):
            if hasattr(transformer, attr):
                blocks = getattr(transformer, attr)
                break
        if blocks is None:
            raise AttributeError(
                "Cannot find transformer blocks at encoder.transformer.[layers|blocks]. "
                "Inspect encoder.transformer.named_children() to find the correct attribute."
            )
        total = len(blocks)
        for i, block in enumerate(blocks):
            if i >= total - n:
                for p in block.parameters():
                    p.requires_grad = True
