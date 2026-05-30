"""
Inflate a 3-channel ImageNet-pretrained patch embed to 4 channels (R,G,B,NIR).

Copy RGB weights as-is; initialize NIR weights as the mean of RGB so the new
channel starts in a vegetation-ish basin instead of random or red-ish init.
This is the recipe from the project's implementation spec.
"""

import torch
import torch.nn as nn


def inflate_patch_embed_to_4ch(backbone):
    """Mutate `backbone` in place: replace its 3-ch patch_embed.proj with a
    4-ch Conv2d, copying RGB weights and seeding NIR as mean(RGB)."""
    old = backbone.patch_embed.proj  # Conv2d(3, C, k, s)
    if old.in_channels == 4:
        return backbone  # already inflated
    if old.in_channels != 3:
        raise ValueError(
            f"Expected 3-channel patch embed, got {old.in_channels}-channel."
        )
    new = nn.Conv2d(
        4, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.patch_embed.proj = new
    return backbone
