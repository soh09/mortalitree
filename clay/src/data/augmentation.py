"""Band-aware augmentations for 4-channel NAIP imagery."""
import random
import torch
import torch.nn.functional as F


def rotate90(pixels: torch.Tensor, boxes: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate image and boxes by k*90 degrees. pixels: (C, H, W), boxes: (N, 4) cxcywh [0,1]."""
    pixels = torch.rot90(pixels, k, dims=[-2, -1])
    if boxes.shape[0] == 0:
        return pixels, boxes
    cx, cy, w, h = boxes.unbind(-1)
    for _ in range(k % 4):
        cx, cy, w, h = 1.0 - cy, cx, h, w
    boxes = torch.stack([cx, cy, w, h], dim=-1)
    return pixels, boxes


def hflip(pixels: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = pixels.flip(-1)
    if boxes.shape[0] == 0:
        return pixels, boxes
    cx, cy, w, h = boxes.unbind(-1)
    boxes = torch.stack([1.0 - cx, cy, w, h], dim=-1)
    return pixels, boxes


def vflip(pixels: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = pixels.flip(-2)
    if boxes.shape[0] == 0:
        return pixels, boxes
    cx, cy, w, h = boxes.unbind(-1)
    boxes = torch.stack([cx, 1.0 - cy, w, h], dim=-1)
    return pixels, boxes


def brightness_contrast_jitter(
    pixels: torch.Tensor,
    brightness: float = 0.1,
    contrast: float = 0.1,
) -> torch.Tensor:
    """Apply uniform brightness/contrast jitter across all bands."""
    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    mean = pixels.mean()
    return (pixels * b - mean) * c + mean


def per_band_gain_jitter(pixels: torch.Tensor, max_gain: float = 0.05) -> torch.Tensor:
    """Apply independent small gain per band to simulate inter-year radiometric drift."""
    C = pixels.shape[0]
    gains = 1.0 + torch.empty(C, 1, 1).uniform_(-max_gain, max_gain)
    return pixels * gains.to(pixels.device)


class NAIPAugmentation:
    def __init__(
        self,
        rotate: bool = True,
        flip: bool = True,
        brightness: float = 0.1,
        contrast: float = 0.1,
        per_band_gain: float = 0.05,
    ):
        self.rotate = rotate
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.per_band_gain = per_band_gain

    def __call__(
        self, pixels: torch.Tensor, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotate:
            k = random.randint(0, 3)
            pixels, boxes = rotate90(pixels, boxes, k)
        if self.flip:
            if random.random() < 0.5:
                pixels, boxes = hflip(pixels, boxes)
            if random.random() < 0.5:
                pixels, boxes = vflip(pixels, boxes)
        if self.brightness > 0 or self.contrast > 0:
            pixels = brightness_contrast_jitter(pixels, self.brightness, self.contrast)
        if self.per_band_gain > 0:
            pixels = per_band_gain_jitter(pixels, self.per_band_gain)
        return pixels, boxes
