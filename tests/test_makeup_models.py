from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("Install requirements-makeup.txt to run makeup model tests") from error
    raise
from torch import nn

from makeup_transfer.losses import (
    alpha_loss,
    color_regression_target,
    discriminator_loss,
    generator_adversarial_loss,
    lip_color_loss,
    reconstruction_loss,
    regressor_loss,
)
from makeup_transfer.models import ConditionalDiscriminator, LipColorRegressor, MakeupGenerator


class TransparencyObjectiveTests(unittest.TestCase):
    def test_reconstruction_uses_target_alpha_and_excludes_alpha_error(self) -> None:
        target = torch.zeros(1, 4, 1, 2)
        target[:, 3, 0, 0] = 0.25
        pred = torch.ones_like(target, requires_grad=True)
        loss = reconstruction_loss(pred, target)
        self.assertAlmostEqual(loss.item(), 0.125)
        self.assertAlmostEqual(reconstruction_loss(pred, target, reduction="sum").item(), 0.75)
        loss.backward()
        self.assertEqual(pred.grad[:, 3:].abs().sum().item(), 0)
        self.assertEqual(pred.grad[:, :3, :, 1].abs().sum().item(), 0)
        self.assertGreater(pred.grad[:, :3, :, 0].abs().sum().item(), 0)

    def test_alpha_supervises_only_synthetic_rows(self) -> None:
        target = torch.zeros(2, 4, 2, 2)
        pred = torch.ones_like(target)
        pred[0, 3] = 0.25
        pred.requires_grad_(True)
        loss = alpha_loss(pred, target, torch.tensor([True, False]))
        self.assertEqual(loss.item(), 0.25)
        loss.backward()
        self.assertEqual(pred.grad[1].abs().sum().item(), 0)
        self.assertEqual(pred.grad[0, :3].abs().sum().item(), 0)
        self.assertGreater(pred.grad[0, 3].abs().sum().item(), 0)

    def test_all_pseudo_labels_have_differentiable_zero_alpha_loss(self) -> None:
        pred = torch.rand(2, 4, 2, 2, requires_grad=True)
        loss = alpha_loss(pred, torch.zeros_like(pred), torch.tensor([False, False]))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(pred.grad.abs().sum().item(), 0)

    def test_color_target_is_rgb_mean_in_segmentation_not_alpha_weighted(self) -> None:
        target = torch.zeros(1, 4, 1, 3)
        target[0, :, 0, 0] = torch.tensor([0.2, 0.4, 0.8, 1.0])
        target[0, :, 0, 1] = torch.tensor([0.8, 0.6, 0.2, 0.0])
        target[0, :, 0, 2] = 10
        mask = torch.tensor([[[[1, 1, 0]]]], dtype=torch.float32)
        torch.testing.assert_close(color_regression_target(target, mask), torch.tensor([[0.5, 0.5, 0.5]]))
        torch.testing.assert_close(color_regression_target(target, mask.expand(-1, 4, -1, -1)), torch.tensor([[0.5, 0.5, 0.5]]))

    def test_color_losses_ignore_empty_regions(self) -> None:
        regressor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(4, 3))
        pred = torch.rand(2, 4, 4, 4, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.zeros(2, 1, 4, 4)
        self.assertEqual(regressor_loss(regressor, target, mask).item(), 0)
        loss = lip_color_loss(regressor.requires_grad_(False), pred, target, mask)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(pred.grad.abs().sum().item(), 0)

    def test_frozen_color_regressor_still_propagates_generator_gradients(self) -> None:
        regressor = LipColorRegressor(base_channels=2).eval().requires_grad_(False)
        # Known positive weights guarantee a nonzero derivative for this fixture.
        for module in regressor.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.constant_(module.weight, 0.05)
        pred = torch.full((1, 4, 8, 8), 0.7, requires_grad=True)
        target = torch.full_like(pred, 0.2, requires_grad=True)
        mask = torch.ones(1, 1, 8, 8)
        loss = lip_color_loss(regressor, pred, target, mask)
        loss.backward()
        self.assertGreater(pred.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(parameter.grad is None for parameter in regressor.parameters()))

    def test_color_norm_choice_and_regressor_training(self) -> None:
        regressor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(4, 3, bias=False))
        nn.init.zeros_(regressor[-1].weight)
        target = torch.ones(1, 4, 2, 2)
        mask = torch.ones(1, 1, 2, 2)
        loss = regressor_loss(regressor, target, mask)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(regressor_loss(regressor, target, mask, loss_type="l2").item(), math.sqrt(3), places=6)
        loss.backward()
        self.assertGreater(regressor[-1].weight.grad.abs().sum().item(), 0)

    def test_adversarial_losses_use_stable_logits(self) -> None:
        zero = torch.zeros(2, 1)
        self.assertAlmostEqual(generator_adversarial_loss(zero).item(), math.log(2), places=6)
        self.assertAlmostEqual(discriminator_loss(zero, zero).item(), math.log(2), places=6)
        self.assertLess(discriminator_loss(torch.full_like(zero, 100), torch.full_like(zero, -100)).item(), 1e-6)
        self.assertTrue(torch.isfinite(generator_adversarial_loss(torch.full_like(zero, -100))))


class NetworkTests(unittest.TestCase):
    def test_generator_shapes_range_both_input_apis_and_backward(self) -> None:
        generator = MakeupGenerator(base_channels=2, dropout=0).train()
        reference = torch.rand(1, 3, 256, 256)
        prior = torch.rand(1, 1, 256, 256)
        rgba = generator(reference, prior)
        self.assertEqual(rgba.shape, (1, 4, 256, 256))
        self.assertTrue(bool(((rgba >= 0) & (rgba <= 1)).all()))
        rgba.mean().backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in generator.parameters()))
        generator.eval()
        with torch.no_grad():
            torch.testing.assert_close(generator(reference, prior), generator(torch.cat((reference, prior), dim=1)))
        with self.assertRaisesRegex(ValueError, "multiples of 256"):
            generator(torch.zeros(1, 4, 128, 128))

    def test_discriminator_appendix_and_compact_shapes_and_conditioning(self) -> None:
        reference = torch.rand(1, 3, 64, 64)
        rgba = torch.rand(1, 4, 64, 64, requires_grad=True)
        for architecture in ("paper", "strided"):
            with self.subTest(architecture=architecture):
                discriminator = ConditionalDiscriminator(base_channels=2, architecture=architecture).eval()
                logits = discriminator(reference, rgba)
                self.assertEqual(logits.shape, (1, 1))
                torch.testing.assert_close(logits, discriminator(torch.cat((reference, rgba), dim=1)))
                logits.sum().backward()
                self.assertIsNotNone(rgba.grad)
                convs = [module for module in discriminator.features if isinstance(module, nn.Conv2d)]
                self.assertEqual([conv.stride for conv in convs], [(1, 1) if architecture == "paper" else (2, 2)] * 4)
                self.assertTrue(all(module.bias is None for module in discriminator.modules() if isinstance(module, (nn.Conv2d, nn.Linear))))

    def test_regressor_has_three_unconstrained_outputs_and_no_biases(self) -> None:
        for architecture in ("paper", "strided"):
            regressor = LipColorRegressor(base_channels=2, architecture=architecture).eval()
            self.assertEqual(regressor(torch.rand(2, 4, 32, 32)).shape, (2, 3))
            self.assertTrue(all(module.bias is None for module in regressor.modules() if isinstance(module, (nn.Conv2d, nn.Linear))))


if __name__ == "__main__":
    unittest.main()
