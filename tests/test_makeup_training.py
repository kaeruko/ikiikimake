"""CPU integration coverage for real training, resumption, and held-out evaluation."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
try:
    import torch
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("Install requirements-makeup.txt to run makeup training tests") from error
    raise

from makeup_transfer.evaluate import evaluate
from makeup_transfer.models import ConditionalDiscriminator, MakeupGenerator
from makeup_transfer.train import TrainConfig, train

training = importlib.import_module("makeup_transfer.train")
REGIONS = ("eye", "lip", "cheek")


def make_prepared_fixture(root: Path) -> dict:
    """Write a tiny valid prepared cache, preserving distinct source identities."""
    root.mkdir(parents=True)
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    mask = (((xx - 128) / 75) ** 2 + ((yy - 128) / 40) ** 2 < 1).astype(np.float32)[..., None]
    alpha = mask * 0.65
    natural = np.stack((0.35 + xx / 1024, 0.3 + yy / 1280, np.full_like(xx, 0.25)), axis=2)
    manifest = {"records": [], "geometry": "geometry.npz", "average_alpha": {}}
    np.savez(root / "geometry.npz", fixture=np.array([1], dtype=np.int32))
    for region_index, region in enumerate(REGIONS):
        average_name = f"{region}_average_alpha.npy"
        np.save(root / average_name, alpha)
        manifest["average_alpha"][region] = average_name
        for split_index, split in enumerate(("train", "val", "test")):
            color = np.empty((256, 256, 3), dtype=np.float32)
            color[:] = [0.75 - region_index * 0.1, 0.15 + split_index * 0.05, 0.35]
            target = np.concatenate((color, alpha), axis=2)
            reference = natural * (1 - alpha) + color * alpha
            path = f"{region}_{split}.npz"
            np.savez_compressed(root / path, input=reference, condition=reference,
                                target=target, mask=mask, has_alpha=np.float32(1))
            manifest["records"].append({"region": region, "split": split, "path": path,
                                         "source": "synthetic", "source_id": f"fixture-{split}"})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def tiny_training_config() -> TrainConfig:
    return TrainConfig(epochs=1, batch_size=1, base_channels=2, architecture="strided",
                       color_epochs=1, color_batch_size=1, max_steps=1, color_max_steps=1,
                       augment=False, workers=0, device="cpu", seed=73)


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class MakeupTrainingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="makeup_training_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.prepared = self.root / "prepared"
        self.output = self.root / "checkpoints"
        self.manifest = make_prepared_fixture(self.prepared)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def test_three_region_training_frozen_color_resume_and_held_out_evaluation(self) -> None:
        initial_generators = []
        frozen_regressors = []
        discriminator_modes = []
        original_pretrain = training.pretrain_color

        def capture_generator(*args, **kwargs):
            model = MakeupGenerator(*args, **kwargs)
            initial_generators.append(clone_state(model))
            return model

        def capture_frozen_regressor(*args, **kwargs):
            model = original_pretrain(*args, **kwargs)
            frozen_regressors.append((model, clone_state(model)))
            return model

        def capture_discriminator(*args, **kwargs):
            model = ConditionalDiscriminator(*args, **kwargs)
            model.register_forward_pre_hook(lambda module, inputs: discriminator_modes.append(
                (module.training, any(parameter.requires_grad for parameter in module.parameters()))))
            return model

        config = tiny_training_config()
        with redirect_stdout(io.StringIO()), patch.object(training, "MakeupGenerator", side_effect=capture_generator), \
                patch.object(training, "pretrain_color", side_effect=capture_frozen_regressor), \
                patch.object(training, "ConditionalDiscriminator", side_effect=capture_discriminator):
            result = train(self.prepared, self.output, config=config)
        self.assertEqual(set(result), set(REGIONS))
        self.assertEqual(len(initial_generators), 3)
        self.assertTrue(discriminator_modes)
        self.assertTrue(all(mode for mode, _ in discriminator_modes))
        self.assertTrue(any(not gradients_enabled for _, gradients_enabled in discriminator_modes))
        digest = hashlib.sha256((self.prepared / "manifest.json").read_bytes()).hexdigest()
        first_states = {}
        for index, region in enumerate(REGIONS):
            state = torch.load(self.output / f"{region}.pt", weights_only=True)
            first_states[region] = state
            self.assertEqual(state["region"], region)
            self.assertEqual(state["steps"], 1)
            self.assertEqual(state["epoch"], 1)
            self.assertTrue(state["training_complete"])
            self.assertFalse(state["paper_training_schedule_complete"])
            self.assertFalse(state["partial_epoch"])
            self.assertEqual(state["manifest_sha256"], digest)
            self.assertEqual(state["average_alpha"].shape, (1, 256, 256))
            self.assertEqual(state["validation"]["samples"], 1)
            for optimizer_name in ("optimizer_g", "optimizer_d"):
                optimizer_state = state[optimizer_name]["state"]
                self.assertTrue(optimizer_state)
                self.assertTrue(all(int(entry["step"]) == 1 for entry in optimizer_state.values()))
            self.assertTrue(any(name.endswith("weight") and not torch.equal(value, state["generator"][name])
                                for name, value in initial_generators[index].items()))
        self.assertEqual((self.output / "geometry.npz").read_bytes(), (self.prepared / "geometry.npz").read_bytes())
        color_checkpoint = torch.load(self.output / "color_regressor.pt", weights_only=True)
        self.assertEqual(color_checkpoint["steps"], 1)
        self.assertTrue(color_checkpoint["optimizer"]["state"])
        model, frozen_state = frozen_regressors[0]
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, frozen_state[name], rtol=0, atol=0)
            torch.testing.assert_close(value, color_checkpoint["model"][name], rtol=0, atol=0)

        # Continue the GANs while retaining the already-pretrained, frozen C.
        with redirect_stdout(io.StringIO()), patch.object(training, "pretrain_color", side_effect=capture_frozen_regressor):
            resumed = train(self.prepared, self.output, config=replace(config, epochs=2, max_steps=2), resume=True)
        for region in REGIONS:
            state = torch.load(self.output / f"{region}.pt", weights_only=True)
            self.assertEqual(resumed[region]["steps"], 2)
            self.assertEqual(state["epoch"], 2)
            self.assertTrue(all(int(entry["step"]) == 2 for entry in state["optimizer_g"]["state"].values()))
            self.assertTrue(any(name.endswith("weight") and not torch.equal(value, state["generator"][name])
                                for name, value in first_states[region]["generator"].items()))
        resumed_color, resumed_color_initial = frozen_regressors[1]
        for name, value in resumed_color.state_dict().items():
            torch.testing.assert_close(value, resumed_color_initial[name], rtol=0, atol=0)
            torch.testing.assert_close(value, color_checkpoint["model"][name], rtol=0, atol=0)

        report_dir = self.root / "evaluation"
        report = evaluate(self.prepared, self.output, report_dir, device="cpu", batch_size=1)
        self.assertEqual(report["split"], "test")
        self.assertFalse(report["paper_benchmark"])
        self.assertEqual(set(report["regions"]), set(REGIONS))
        for region, metrics in report["regions"].items():
            self.assertEqual(metrics["samples"], 1)
            self.assertEqual(metrics["synthetic_alpha_samples"], 1)
            self.assertEqual(metrics["checkpoint_steps"], 2)
            self.assertTrue(np.isfinite(metrics["premultiplied_rgb_psnr_db"]))
            self.assertGreaterEqual(metrics["alpha_mae"], 0)
            self.assertLessEqual(metrics["alpha_mae"], 1)
            with Image.open(report_dir / f"{region}_comparison.png") as image:
                self.assertEqual(image.size, (768, 284))
        self.assertEqual(json.loads((report_dir / "metrics.json").read_text(encoding="utf-8")), report)
        with self.assertRaises(FileExistsError):
            train(self.prepared, self.output, config=config)
        with self.assertRaises(FileExistsError):
            evaluate(self.prepared, self.output, report_dir, device="cpu")

    def test_resuming_rejects_changed_dataset_manifest(self) -> None:
        with redirect_stdout(io.StringIO()):
            train(self.prepared, self.output, regions=("eye",), config=tiny_training_config())
        self.manifest["records"][0]["source_id"] = "different-training-source"
        (self.prepared / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        original_files = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, "dataset differs"):
            train(self.prepared, self.output, regions=("eye",),
                  config=replace(tiny_training_config(), epochs=2, max_steps=2), resume=True)
        self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, original_files)

    def test_resume_preflight_checks_all_requested_models_before_changing_files(self) -> None:
        config = tiny_training_config()
        with redirect_stdout(io.StringIO()):
            train(self.prepared, self.output, config=config)
        # Deliberately let C have pending work: a bad cheek checkpoint must be
        # caught before C, eye, or lip can advance their saved states.
        resumed = replace(config, epochs=2, max_steps=2, color_epochs=2, color_max_steps=2)
        for filename, field, replacement, message in (
            ("cheek.pt", "region", "eye", "region differs"),
            ("lip.pt", "architecture", "paper", "architecture differs"),
            ("color_regressor.pt", "manifest_sha256", "wrong-dataset", "dataset differs"),
            ("color_regressor.pt", "base_channels", 16, "architecture differs"),
        ):
            with self.subTest(checkpoint=filename, field=field):
                path = self.output / filename
                original_checkpoint = path.read_bytes()
                state = torch.load(path, weights_only=True)
                state[field] = replacement
                torch.save(state, path)
                before = {item.name: item.read_bytes() for item in self.output.iterdir()}
                with self.assertRaisesRegex(ValueError, message):
                    train(self.prepared, self.output, config=resumed, resume=True)
                self.assertEqual({item.name: item.read_bytes() for item in self.output.iterdir()}, before)
                path.write_bytes(original_checkpoint)

    def test_resume_uses_requested_generator_discriminator_and_color_learning_rates(self) -> None:
        config = tiny_training_config()
        with redirect_stdout(io.StringIO()):
            train(self.prepared, self.output, regions=("lip",), config=config)
        resumed = replace(config, epochs=2, max_steps=2, color_epochs=2, color_max_steps=2,
                          learning_rate=3e-4, discriminator_lr=4e-4, color_lr=7e-5)
        with redirect_stdout(io.StringIO()):
            train(self.prepared, self.output, regions=("lip",), config=resumed, resume=True)
        state = torch.load(self.output / "lip.pt", weights_only=True)
        color = torch.load(self.output / "color_regressor.pt", weights_only=True)
        for optimizer, expected in ((state["optimizer_g"], resumed.learning_rate),
                                    (state["optimizer_d"], resumed.discriminator_lr),
                                    (color["optimizer"], resumed.color_lr)):
            self.assertTrue(all(group["lr"] == expected for group in optimizer["param_groups"]))
        self.assertEqual(state["steps"], 2)
        self.assertEqual(color["steps"], 2)

    def test_validation_alpha_average_is_independent_of_pseudo_label_batching(self) -> None:
        with np.load(self.prepared / "eye_val.npz") as source:
            pseudo = {name: source[name].copy() for name in source.files}
        expected_alpha_error = float(np.abs(0.75 - pseudo["target"][..., 3]).mean())
        pseudo["has_alpha"] = np.float32(0)
        pseudo["target"][..., 3] = 0
        np.savez_compressed(self.prepared / "eye_pseudo_val.npz", **pseudo)
        self.manifest["records"].append({"region": "eye", "split": "val", "path": "eye_pseudo_val.npz",
                                         "source": "kmeans", "source_id": "fixture-pseudo-val"})
        (self.prepared / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

        class ConstantMask(torch.nn.Module):
            def forward(self, reference, average_alpha):
                return torch.full((reference.shape[0], 4, *reference.shape[2:]), 0.75, device=reference.device)

        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                result = training.validate_generator(ConstantMask(), self.prepared, "eye",
                    replace(tiny_training_config(), batch_size=batch_size), torch.device("cpu"))
                self.assertEqual(result["samples"], 2)
                self.assertEqual(result["synthetic_alpha_samples"], 1)
                self.assertAlmostEqual(result["alpha"], expected_alpha_error, places=6)


if __name__ == "__main__":
    unittest.main()
