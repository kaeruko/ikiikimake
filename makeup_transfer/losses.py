"""Transparency-aware objectives from arXiv:2509.02445v2, equations 3--5.

The paper writes spatial sums in equation 3 but does not define batch
reduction.  The default here is a mean over the batch, RGB channels and pixels
for resolution-independent training; ``reduction='sum'`` gives its literal
sum over all batch elements.  Color equations print an L2 norm while the prose
specifies mean squared differences.  ``loss_type='mse'`` follows the prose;
``loss_type='l2'`` selects the printed vector norm explicitly.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Reduction = Literal["mean", "sum"]
ColorLossType = Literal["mse", "l2"]


def _validate_rgba(image: Tensor, name: str) -> None:
    if image.ndim != 4 or image.shape[1] != 4:
        raise ValueError(f"{name} must have shape (batch, 4, height, width)")


def _validate_pair(prediction: Tensor, target: Tensor) -> None:
    _validate_rgba(prediction, "prediction")
    _validate_rgba(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return values.mean() if values.numel() else values.sum()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'mean' or 'sum'")


def reconstruction_loss(pred: Tensor, target: Tensor, reduction: Reduction = "mean") -> Tensor:
    """Equation 3: target-alpha-weighted RGB L1, with no alpha-channel error.

    The denominator for ``mean`` is the total RGB element count, rather than
    the alpha sum.  Thus nearly transparent targets contribute less loss.
    """
    _validate_pair(pred, target)
    weighted_error = target[:, 3:4] * (pred[:, :3] - target[:, :3]).abs()
    return _reduce(weighted_error, reduction)


def alpha_loss(pred: Tensor, target: Tensor, has_alpha: Tensor, reduction: Reduction = "mean") -> Tensor:
    """Alpha L1 on graphics-rendered rows only; exclude all k-means pseudo labels.

    ``has_alpha`` supplies one boolean per batch row.  An all-pseudo-label
    batch produces a differentiable zero so training can still call backward.
    The mean denominator includes only supervised rows and their pixels.
    """
    _validate_pair(pred, target)
    selected = torch.as_tensor(has_alpha, device=pred.device, dtype=torch.bool).reshape(-1)
    if selected.numel() != pred.shape[0]:
        raise ValueError("has_alpha must contain exactly one flag per batch row")
    return _reduce((pred[:, 3:4] - target[:, 3:4]).abs()[selected], reduction)


def _lip_mask(mask: Tensor, rgba: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] not in (1, 4):
        raise ValueError("lip mask must have shape (batch, height, width), (batch, 1, height, width), or repeated four channels")
    if mask.shape[0] != rgba.shape[0] or mask.shape[2:] != rgba.shape[2:]:
        raise ValueError("lip mask must match the RGBA batch and spatial dimensions")
    if mask.shape[1] == 4 and not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError("a four-channel lip mask must repeat the same spatial mask")
    return mask[:, :1].to(device=rgba.device, dtype=rgba.dtype)


def color_regression_target(target: Tensor, mask: Tensor) -> Tensor:
    """Equation 4 target: mean RGB inside lip segmentation, not alpha-weighted.

    Only RGB enters the three-dimensional target; the published equation
    suppresses this channel selection despite specifying C: RGBA -> RGB.
    Binary masks reproduce the stated pixel count; soft masks use their sum.
    Empty masks yield a zero target and are excluded by the color loss helpers.
    """
    _validate_rgba(target, "target")
    spatial_mask = _lip_mask(mask, target)
    pixel_count = spatial_mask.sum(dim=(2, 3))
    denominator = torch.where(pixel_count > 0, pixel_count, torch.ones_like(pixel_count))
    return (target[:, :3] * spatial_mask).sum(dim=(2, 3)) / denominator


def _color_distance(predicted: Tensor, target: Tensor, loss_type: ColorLossType) -> Tensor:
    if loss_type == "mse":
        return (predicted - target).square().mean(dim=1)
    if loss_type == "l2":
        return torch.linalg.vector_norm(predicted - target, ord=2, dim=1)
    raise ValueError("loss_type must be 'mse' or 'l2'")


def regressor_loss(
    regressor: nn.Module,
    target: Tensor,
    mask: Tensor,
    *,
    loss_type: ColorLossType = "mse",
    reduction: Reduction = "mean",
) -> Tensor:
    """Pretrain C on graphics masks, predicting their mean segmented RGB."""
    _validate_rgba(target, "target")
    spatial_mask = _lip_mask(mask, target)
    target = target.detach()
    prediction = regressor(target * spatial_mask)
    average_color = color_regression_target(target, spatial_mask)
    valid = spatial_mask.sum(dim=(1, 2, 3)) > 0
    return _reduce(_color_distance(prediction, average_color, loss_type)[valid], reduction)


def lip_color_loss(
    regressor: nn.Module,
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    loss_type: ColorLossType = "mse",
    reduction: Reduction = "mean",
) -> Tensor:
    """Equation 5, preserving the path from frozen C to generated RGBA.

    The caller freezes regressor parameters and sets eval mode after its
    pretraining phase.  Only the target branch is evaluated under no_grad.
    """
    _validate_pair(pred, target)
    spatial_mask = _lip_mask(mask, target)
    predicted_color = regressor(pred * spatial_mask)
    with torch.no_grad():
        target_color = regressor(target * spatial_mask)
    valid = spatial_mask.sum(dim=(1, 2, 3)) > 0
    return _reduce(_color_distance(predicted_color, target_color, loss_type)[valid], reduction)


def generator_adversarial_loss(logits: Tensor) -> Tensor:
    """Non-saturating cross-entropy GAN objective on raw discriminator logits."""
    return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Mean of real and fake cross-entropy terms (fake must be detached upstream)."""
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return (real_loss + fake_loss) * 0.5
