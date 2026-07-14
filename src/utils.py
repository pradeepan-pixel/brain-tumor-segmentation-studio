"""
utils.py — Loss functions, metrics, and checkpoint helpers for 3D BraTS segmentation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LOSS FUNCTIONS
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation (single foreground class: tumor).
    logits: raw model output, shape [B, 1, D, H, W]
    target: binary mask, shape [B, 1, D, H, W] or [B, D, H, W]
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)

        if target.dim() == probs.dim() - 1:
            target = target.unsqueeze(1)
        target = target.float()

        probs = probs.contiguous().view(probs.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss. This is the standard choice for BraTS-style
    tumor segmentation — BCE stabilizes early training (dice alone has
    unstable/vanishing gradients on very small or empty masks), Dice directly
    optimizes for overlap and handles the massive class imbalance
    (tumor is often <5% of the volume).
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)
        target = target.float()

        dice_loss = self.dice(logits, target)
        bce_loss = self.bce(logits, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

@torch.no_grad()
def dice_score(logits, target, threshold=0.5, smooth=1e-5):
    """
    Hard Dice score for logging/validation (not for backprop).
    Returns a python float.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    if target.dim() == preds.dim() - 1:
        target = target.unsqueeze(1)
    target = target.float()

    preds = preds.contiguous().view(preds.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ---------------------------------------------------------------------------
# CHECKPOINT HELPERS
# ---------------------------------------------------------------------------

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0)
    best_dice = checkpoint.get("best_dice", 0.0)
    return start_epoch, best_dice


class AverageMeter:
    """Tracks running average of a metric (loss, dice, etc.) across a training epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
