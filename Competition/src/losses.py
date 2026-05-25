import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCELoss(nn.Module):
    def __init__(self, ignore_val: int = 255):
        super().__init__()
        self.ignore_val = ignore_val
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_val
        return self.bce(logits[mask], targets[mask])


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_val: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_val = ignore_val

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_val

        preds = torch.sigmoid(logits[mask])
        gt = targets[mask]

        intersection = (preds * gt).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + gt.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_val: int = 255,
    ):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.ignore_val = ignore_val

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = (targets != self.ignore_val).float()
        binary_gt  = (targets > 0.5).float() * valid_mask

        logits = logits.float()
        p      = torch.sigmoid(logits)
        ce     = F.binary_cross_entropy_with_logits(logits, binary_gt, reduction="none")
        p_t    = p * binary_gt + (1.0 - p) * (1.0 - binary_gt)
        alpha_t = self.alpha * binary_gt + (1.0 - self.alpha) * (1.0 - binary_gt)
        loss   = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        loss   = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp_min(1.0)


class BEVLoss(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.9,
        focal_gamma: float = 2.0,
        dice_weight: float = 2.0,
        ignore_val:  int   = 255,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ignore_val  = ignore_val
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_val=ignore_val)
        self.dice  = DiceLoss(smooth=1.0, ignore_val=ignore_val)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)
