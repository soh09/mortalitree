"""Load Clay v1.5 encoder and wrap it for batch-dict forward pass.

Clay v1.5 (the released v1.5.0 checkpoint) is the *large* MAE: dim=1024,
depth=24, patch_size=8. For a 224x224 input that yields a 28x28 grid of
1024-dim patch tokens at stride 8 (not the 14x14 / 768-dim / stride-16 grid a
patch-16 base model would give). The wrapper below reads these dimensions off
the encoder at runtime instead of hardcoding them.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat


def load_clay_encoder(checkpoint_path: str, strict: bool = False) -> nn.Module:
    """Load Clay v1.5 from a Lightning checkpoint and return the encoder."""
    try:
        from claymodel.module import ClayMAEModule
    except ImportError as e:
        raise ImportError(
            "claymodel not found. Install from https://github.com/Clay-foundation/model"
        ) from e

    clay = ClayMAEModule.load_from_checkpoint(checkpoint_path, strict=strict)
    encoder = clay.model.encoder
    # Disable masking *and* shuffling so the encoder returns all patch tokens in
    # their original spatial (row-major) order. The wrapper below also bypasses
    # mask_out entirely, but we set these defensively in case anything calls the
    # base Encoder.forward.
    encoder.mask_ratio = 0.0
    encoder.shuffle = False
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
