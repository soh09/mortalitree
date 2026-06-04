"""Learning rate schedulers: cosine with linear warmup, layer-wise LR."""
import math
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warm up from a nonzero fraction so epoch 0 actually trains
            # (epoch/warmup would give lr=0 on the first epoch).
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_stage_b_optimizer_and_scheduler(model, lr: float, weight_decay: float, total_epochs: int, warmup_epochs: int = 5):
    """AdamW + cosine-warmup for Stage B (neck + head only)."""
    import torch.optim as optim
    params = model.neck_head_parameters()
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = cosine_with_warmup(optimizer, warmup_epochs, total_epochs)
    return optimizer, scheduler


def build_stage_c_optimizer_and_scheduler(
    model,
    neck_head_lr: float,
    encoder_lr: float,
    weight_decay: float,
    total_epochs: int,
    warmup_epochs: int = 5,
):
    """AdamW with separate LR groups for neck/head vs unfrozen encoder blocks."""
    import torch.optim as optim
    param_groups = [
        {"params": model.neck_head_parameters(), "lr": neck_head_lr},
    ]
    encoder_params = model.encoder_parameters()
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = cosine_with_warmup(optimizer, warmup_epochs, total_epochs)
    return optimizer, scheduler
