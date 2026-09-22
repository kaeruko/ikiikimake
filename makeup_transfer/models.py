"""Networks for arXiv:2509.02445v2, Appendix A, Table 2.

All public image tensors use NCHW layout and values in [0, 1].  Instantiate a
separate generator/discriminator pair for each of eye, lip, and cheek regions.
The optional ``strided`` discriminator/regressor is a memory-saving adaptation;
``paper`` preserves the stride-one convolutions printed in the appendix.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_image(image: Tensor, channels: int, name: str) -> None:
    if image.ndim != 4 or image.shape[1] != channels:
        raise ValueError(f"{name} must have shape (batch, {channels}, height, width)")
    if not image.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor in [0, 1]")


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class _UNetSkipBlock(nn.Module):
    """Eight-level pix2pix U-Net with concatenated encoder/decoder skips."""

    def __init__(
        self,
        outer_channels: int,
        inner_channels: int,
        *,
        input_channels: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        input_channels = outer_channels if input_channels is None else input_channels
        downconv = nn.Conv2d(input_channels, inner_channels, 4, 2, 1, bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace=False)
        uprelu = nn.ReLU(inplace=False)
        if outermost:
            layers = [
                downconv,
                submodule,
                uprelu,
                nn.ConvTranspose2d(inner_channels * 2, outer_channels, 4, 2, 1),
                nn.Tanh(),
            ]
        elif innermost:
            layers = [
                downrelu,
                downconv,
                uprelu,
                nn.ConvTranspose2d(inner_channels, outer_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(outer_channels),
            ]
        else:
            layers = [
                downrelu,
                downconv,
                nn.BatchNorm2d(inner_channels),
                submodule,
                uprelu,
                nn.ConvTranspose2d(inner_channels * 2, outer_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(outer_channels),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.model(inputs)
        return output if self.outermost else torch.cat((inputs, output), dim=1)


class MakeupGenerator(nn.Module):
    """Pix2pix U-Net-256 with four input and four output channels.

    ``forward`` accepts either concatenated RGB/average-alpha input or an RGB
    tensor and a separate one-channel average-alpha tensor.  Inputs are mapped
    from [0, 1] to [-1, 1] internally; the original tanh output is mapped back
    to [0, 1], including its alpha channel.  ``base_channels=64`` matches the
    cited pix2pix architecture; smaller values support smoke tests.
    """

    def __init__(self, base_channels: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.base_channels = base_channels
        width = base_channels
        block = _UNetSkipBlock(width * 8, width * 8, innermost=True)
        for _ in range(3):
            block = _UNetSkipBlock(width * 8, width * 8, submodule=block, dropout=dropout)
        block = _UNetSkipBlock(width * 4, width * 8, submodule=block)
        block = _UNetSkipBlock(width * 2, width * 4, submodule=block)
        block = _UNetSkipBlock(width, width * 2, submodule=block)
        self.model = _UNetSkipBlock(4, width, input_channels=4, submodule=block, outermost=True)
        self.apply(_initialize_weights)

    def forward(self, inputs: Tensor, average_alpha: Tensor | None = None) -> Tensor:
        if average_alpha is not None:
            _validate_image(inputs, 3, "reference RGB")
            _validate_image(average_alpha, 1, "average alpha")
            if inputs.shape[0] != average_alpha.shape[0] or inputs.shape[2:] != average_alpha.shape[2:]:
                raise ValueError("reference RGB and average alpha must have matching batch/spatial sizes")
            inputs = torch.cat((inputs, average_alpha), dim=1)
        _validate_image(inputs, 4, "generator input")
        if any(size < 256 or size % 256 for size in inputs.shape[2:]):
            raise ValueError("U-Net-256 requires spatial dimensions that are positive multiples of 256")
        return (self.model(inputs * 2.0 - 1.0) + 1.0) * 0.5


def _convolution_stack(input_channels: int, base_channels: int, architecture: str) -> nn.Sequential:
    if base_channels < 1:
        raise ValueError("base_channels must be positive")
    if architecture not in {"paper", "strided"}:
        raise ValueError("architecture must be 'paper' or 'strided'")
    stride = 1 if architecture == "paper" else 2
    layers: list[nn.Module] = []
    for index, multiplier in enumerate((1, 2, 4, 8)):
        output_channels = base_channels * multiplier
        layers.append(nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False))
        if index:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        input_channels = output_channels
    return nn.Sequential(*layers)


class ConditionalDiscriminator(nn.Module):
    """Conditioned RGB + RGBA discriminator returning one raw logit per image.

    Table 2 has an unresolved size mismatch: its four stride-one convolutions
    and final unpadded 3x3 convolution produce 254x254 values for a 256x256
    image, while the next layer expects 36.  We explicitly insert adaptive
    average pooling to 6x6 before its stated 36 -> 18 -> 1 classifier.
    ``architecture='strided'`` changes the first four strides to two to reduce
    memory; it is an optional adaptation, not the published architecture.
    """

    def __init__(self, base_channels: int = 64, architecture: str = "paper") -> None:
        super().__init__()
        self.base_channels = base_channels
        self.architecture = architecture
        self.features = _convolution_stack(7, base_channels, architecture)
        self.classifier = nn.Sequential(
            nn.Conv2d(base_channels * 8, 1, 3, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(36, 18, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(18, 1, bias=False),
        )
        self.apply(_initialize_weights)

    def forward(self, reference_rgb: Tensor, rgba: Tensor | None = None) -> Tensor:
        if rgba is not None:
            _validate_image(reference_rgb, 3, "reference RGB")
            _validate_image(rgba, 4, "makeup RGBA")
            if reference_rgb.shape[0] != rgba.shape[0] or reference_rgb.shape[2:] != rgba.shape[2:]:
                raise ValueError("reference RGB and makeup RGBA must have matching batch/spatial sizes")
            inputs = torch.cat((reference_rgb, rgba), dim=1)
        else:
            inputs = reference_rgb
        _validate_image(inputs, 7, "discriminator input")
        minimum = 3 if self.architecture == "paper" else 33
        if min(inputs.shape[2:]) < minimum:
            raise ValueError(f"{self.architecture} discriminator inputs must be at least {minimum}x{minimum}")
        return self.classifier(self.features(inputs))


class LipColorRegressor(nn.Module):
    """Appendix Table 2 RGBA-to-RGB regressor, with no output activation.

    Freeze its parameters and put it in eval mode after pretraining.  Do not
    wrap its prediction branch in ``no_grad`` during generator training:
    gradients through the frozen regressor must still reach predicted RGBA.
    """

    def __init__(self, base_channels: int = 64, architecture: str = "paper") -> None:
        super().__init__()
        self.base_channels = base_channels
        self.architecture = architecture
        self.features = _convolution_stack(4, base_channels, architecture)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(base_channels * 8, 3, bias=False))
        self.apply(_initialize_weights)

    def forward(self, rgba: Tensor) -> Tensor:
        _validate_image(rgba, 4, "color regressor input")
        return self.head(self.features(rgba))
