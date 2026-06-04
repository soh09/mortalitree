"""Shape sanity test: builds each component with random tensors and verifies outputs."""
import torch
from src.model.neck import StrippedViTDetNeck
from src.model.head import BoxDetectionHead, SinCos2DPositionalEncoding
from src.train.losses import matching_loss, batch_matching_loss

B, Q = 2, 124
# Clay v1.5 large: dim=1024, patch_size=8 -> 256/8 = 32x32 stride-8 tokens.
CLAY_DIM = 1024
GRID = 32        # 256 // 8
NECK_GRID = GRID * 2   # neck upsamples 2x to stride 4 -> 64x64
device = "cpu"


def test_positional_encoding():
    pe = SinCos2DPositionalEncoding(128)
    out = pe(NECK_GRID, NECK_GRID, device)   # neck output grid (stride 4)
    assert out.shape == (NECK_GRID * NECK_GRID, 128), f"PE shape wrong: {out.shape}"
    print("  SinCos2DPositionalEncoding: OK")


def test_neck():
    neck = StrippedViTDetNeck(in_channels=CLAY_DIM, out_channels=128)
    x = torch.randn(B, CLAY_DIM, GRID, GRID)          # (B, 1024, 32, 32) stride 8
    out = neck(x)
    assert out.shape == (B, 128, NECK_GRID, NECK_GRID), f"Neck output shape wrong: {out.shape}"  # stride 4
    print("  StrippedViTDetNeck: OK")


def test_head():
    head = BoxDetectionHead(feat_channels=128, num_queries=Q, hidden=128, n_heads=4, n_decoder_layers=3)
    feat = torch.randn(B, 128, NECK_GRID, NECK_GRID)
    cls_logits, boxes = head(feat)
    assert cls_logits.shape == (B, Q),    f"cls_logits shape wrong: {cls_logits.shape}"
    assert boxes.shape      == (B, Q, 4), f"boxes shape wrong: {boxes.shape}"
    assert boxes.min() >= 0 and boxes.max() <= 1, "boxes not in [0,1]"
    print("  BoxDetectionHead: OK")


def test_neck_and_head_together():
    neck = StrippedViTDetNeck(in_channels=CLAY_DIM)
    head = BoxDetectionHead()
    x = torch.randn(B, CLAY_DIM, GRID, GRID)
    p3 = neck(x)
    cls_logits, boxes = head(p3)
    assert cls_logits.shape == (B, Q)
    assert boxes.shape      == (B, Q, 4)
    print("  Neck + Head together: OK")


def test_matching_loss():
    cls_logits = torch.randn(Q)
    pred_boxes = torch.rand(Q, 4)
    # Make boxes valid: ensure w,h > 0 and don't overflow
    pred_boxes[:, 2:] = pred_boxes[:, 2:].clamp(0.05, 0.3)
    pred_boxes[:, :2] = pred_boxes[:, :2].clamp(0.1, 0.9)
    gt_boxes = torch.rand(5, 4)
    gt_boxes[:, 2:] = gt_boxes[:, 2:].clamp(0.05, 0.3)
    gt_boxes[:, :2] = gt_boxes[:, :2].clamp(0.1, 0.9)

    loss_exhaustive = matching_loss(cls_logits, pred_boxes, gt_boxes, tile_exhaustive=True)
    loss_sparse     = matching_loss(cls_logits, pred_boxes, gt_boxes, tile_exhaustive=False)
    assert loss_exhaustive.shape == (), "Loss should be scalar"
    assert loss_sparse.shape     == (), "Loss should be scalar"
    assert not torch.isnan(loss_exhaustive), "NaN in exhaustive loss"
    assert not torch.isnan(loss_sparse),     "NaN in sparse loss"
    print(f"  matching_loss exhaustive={loss_exhaustive.item():.4f}  sparse={loss_sparse.item():.4f}: OK")


def test_batch_matching_loss():
    cls_logits_batch = torch.randn(B, Q)
    pred_boxes_batch = torch.rand(B, Q, 4)
    pred_boxes_batch[..., 2:] = pred_boxes_batch[..., 2:].clamp(0.05, 0.3)
    pred_boxes_batch[..., :2] = pred_boxes_batch[..., :2].clamp(0.1, 0.9)
    gt_boxes_list = []
    for _ in range(B):
        gt = torch.rand(7, 4)
        gt[:, 2:] = gt[:, 2:].clamp(0.05, 0.3)
        gt[:, :2] = gt[:, :2].clamp(0.1, 0.9)
        gt_boxes_list.append(gt)
    exhaustive_list = [True, False]
    loss = batch_matching_loss(cls_logits_batch, pred_boxes_batch, gt_boxes_list, exhaustive_list)
    assert not torch.isnan(loss), "NaN in batch loss"
    print(f"  batch_matching_loss={loss.item():.4f}: OK")


def test_augmentation():
    from src.data.augmentation import NAIPAugmentation
    aug = NAIPAugmentation()
    pixels = torch.rand(4, 256, 256)
    boxes  = torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.3, 0.08, 0.08]])
    out_pixels, out_boxes = aug(pixels, boxes)
    assert out_pixels.shape == (4, 256, 256)
    assert out_boxes.shape  == (2, 4)
    assert out_boxes[:, :2].min() >= 0 and out_boxes[:, :2].max() <= 1
    print("  NAIPAugmentation: OK")


if __name__ == "__main__":
    print("Running shape tests...")
    test_positional_encoding()
    test_neck()
    test_head()
    test_neck_and_head_together()
    test_matching_loss()
    test_batch_matching_loss()
    test_augmentation()
    print("\nAll shape tests passed.")
