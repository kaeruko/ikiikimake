"""Regional GAN training, frozen lip color supervision, and resumable checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MakeupDataset
from .losses import (alpha_loss, discriminator_loss, generator_adversarial_loss,
                     lip_color_loss, reconstruction_loss, regressor_loss)
from .models import ConditionalDiscriminator, LipColorRegressor, MakeupGenerator


@dataclass
class TrainConfig:
    epochs: int = 55
    batch_size: int = 8
    base_channels: int = 64
    architecture: str = "paper"
    learning_rate: float = 2e-4
    discriminator_lr: float = 2e-4
    color_epochs: int = 10
    color_batch_size: int = 32
    color_lr: float = 5e-5
    recon_weight: float = 100.0
    alpha_weight: float = 100.0
    adversarial_weight: float = 10.0
    color_weight: float = 50.0
    color_loss_type: str = "mse"
    workers: int = 0
    seed: int = 42
    device: str = "auto"
    max_steps: int | None = None
    color_max_steps: int | None = None
    amp: bool = False
    augment: bool = True

    def validate(self):
        for name in ("epochs", "batch_size", "base_channels", "color_epochs", "color_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.architecture not in ("paper", "strided"):
            raise ValueError("architecture must be paper or strided")
        if self.workers < 0:
            raise ValueError("workers must be nonnegative")
        for name in ("max_steps", "color_max_steps"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "discriminator_lr", "color_lr"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a compatible PyTorch build or use --device cpu")
    return result


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_save(payload, path: Path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _loader(dataset, batch_size, config, epoch=0, shuffle=True):
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=config.workers, generator=generator, drop_last=False)


def _log(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def _read_checkpoint(path: Path, config: TrainConfig, manifest_hash: str, *, region=None, device="cpu"):
    """Validate checkpoint identity before any output files are changed."""
    state = torch.load(path, map_location=device, weights_only=True)
    label = "color checkpoint" if region is None else f"{region} checkpoint"
    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Prepared dataset differs from the {label}")
    if region is not None and state.get("region") != region:
        raise ValueError(f"Checkpoint region differs from requested region {region}: {path}")
    if state.get("architecture") != config.architecture or state.get("base_channels") != config.base_channels:
        raise ValueError(f"Checkpoint architecture differs from requested configuration: {path}")
    required = {"epoch", "steps", "model", "optimizer"} if region is None else {
        "epoch", "steps", "generator", "discriminator", "optimizer_g", "optimizer_d"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Incomplete {label}: missing {', '.join(sorted(missing))}")
    return state


def pretrain_color(prepared: Path, output: Path, config: TrainConfig, device,
                   manifest_hash: str, resume: bool):
    path = output / "color_regressor.pt"
    model = LipColorRegressor(config.base_channels, architecture=config.architecture).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.color_lr, momentum=0.9)
    start, steps = 0, 0
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} already exists; use --resume or a new output directory")
        state = _read_checkpoint(path, config, manifest_hash, device=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config.color_lr
        start, steps = state["epoch"], state["steps"]
    dataset = MakeupDataset(prepared, "lip", split="train", augment=False)
    if not len(dataset):
        raise ValueError("No synthetic lip training samples are available")
    for epoch in range(start, config.color_epochs):
        if config.color_max_steps is not None and steps >= config.color_max_steps:
            break
        model.train()
        total, count = 0.0, 0
        completed = True
        loader = _loader(dataset, config.color_batch_size, config, epoch)
        for index, batch in enumerate(loader):
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = regressor_loss(model, batch["target"], batch["mask"], loss_type=config.color_loss_type)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite color regressor loss")
            loss.backward()
            optimizer.step()
            steps += 1
            count += len(batch["target"])
            total += loss.item() * len(batch["target"])
            if config.color_max_steps is not None and steps >= config.color_max_steps:
                completed = index + 1 == len(loader)
                break
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch + int(completed), "steps": steps,
                      "architecture": config.architecture, "base_channels": config.base_channels,
                      "manifest_sha256": manifest_hash, "config": asdict(config),
                      "partial_epoch": not completed}, path)
        _log(output / "history.jsonl", {"stage": "color", "epoch": epoch + 1,
             "steps": steps, "loss": total / max(count, 1), "partial_epoch": not completed})
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def validate_generator(model, prepared, region, config, device):
    dataset = MakeupDataset(prepared, region, split="val", augment=False)
    if not len(dataset):
        return {"samples": 0}
    model.eval()
    totals = {"reconstruction": 0.0, "alpha": 0.0}
    count, alpha_count = 0, 0
    for batch in _loader(dataset, config.batch_size, config, shuffle=False):
        batch = _batch_to_device(batch, device)
        prediction = model(batch["input"], batch["average_alpha"])
        number = len(prediction)
        totals["reconstruction"] += reconstruction_loss(prediction, batch["target"]).item() * number
        supervised = int(batch["has_alpha"].sum().item())
        totals["alpha"] += alpha_loss(prediction, batch["target"], batch["has_alpha"]).item() * supervised
        alpha_count += supervised
        count += number
    return {"samples": count, "synthetic_alpha_samples": alpha_count,
            "reconstruction": totals["reconstruction"] / count,
            "alpha": totals["alpha"] / alpha_count if alpha_count else None}


def train(prepared_dir, output_dir, *, regions=("eye", "lip", "cheek"),
          config: TrainConfig | None = None, resume=False):
    config = config or TrainConfig()
    config.validate()
    regions = tuple(regions)
    if not regions or any(region not in ("eye", "lip", "cheek") for region in regions):
        raise ValueError("Specify one or more of eye, lip, cheek")
    if len(set(regions)) != len(regions):
        raise ValueError("Requested regions must not contain duplicates")
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    manifest_path = prepared / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output must be empty: {output}; use --resume to continue")
    geometry_source = prepared / manifest.get("geometry", "geometry.npz")
    geometry_destination = output / "geometry.npz"
    geometry_bytes = geometry_source.read_bytes()
    if geometry_destination.exists():
        if geometry_destination.read_bytes() != geometry_bytes:
            raise ValueError("Checkpoint geometry differs from prepared geometry")
    # Check every requested model before pretraining C or updating run metadata.
    # This also catches a bad later-region checkpoint before earlier ones train.
    if resume:
        for region in regions:
            checkpoint = output / f"{region}.pt"
            if checkpoint.exists():
                _read_checkpoint(checkpoint, config, manifest_hash, region=region)
        color_checkpoint = output / "color_regressor.pt"
        if "lip" in regions and color_checkpoint.exists():
            _read_checkpoint(color_checkpoint, config, manifest_hash)
    datasets = {region: MakeupDataset(prepared, region, split="train", augment=config.augment)
                for region in regions}
    for region, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f"No {region} training samples in {prepared}")
    device = select_device(config.device)
    output.mkdir(parents=True, exist_ok=True)
    if not geometry_destination.exists():
        shutil.copyfile(geometry_source, geometry_destination)
    seed_everything(config.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    (output / "training_config.json").write_text(json.dumps({**asdict(config),
        "prepared_dir": str(prepared), "regions": list(regions),
        "manifest_sha256": manifest_hash, "torch_version": str(torch.__version__),
        "resolved_device": str(device), "paper": "https://arxiv.org/html/2509.02445v2",
        "resume_partial_epoch": "Weights and optimizer restored; an unfinished epoch is repeated."},
        indent=2), encoding="utf-8")
    color = pretrain_color(prepared, output, config, device, manifest_hash, resume) if "lip" in regions else None
    results = {}
    for region in regions:
        dataset = datasets[region]
        generator = MakeupGenerator(base_channels=config.base_channels).to(device)
        discriminator = ConditionalDiscriminator(config.base_channels, architecture=config.architecture).to(device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999))
        scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
        checkpoint = output / f"{region}.pt"
        start, steps = 0, 0
        if checkpoint.exists():
            if not resume:
                raise FileExistsError(checkpoint)
            state = _read_checkpoint(checkpoint, config, manifest_hash, region=region, device=device)
            generator.load_state_dict(state["generator"])
            discriminator.load_state_dict(state["discriminator"])
            optimizer_g.load_state_dict(state["optimizer_g"])
            optimizer_d.load_state_dict(state["optimizer_d"])
            for group in optimizer_g.param_groups:
                group["lr"] = config.learning_rate
            for group in optimizer_d.param_groups:
                group["lr"] = config.discriminator_lr
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            start, steps = state["epoch"], state["steps"]
        began = time.perf_counter()
        for epoch in range(start, config.epochs):
            if config.max_steps is not None and steps >= config.max_steps:
                break
            generator.train()
            discriminator.train()
            sums, count = {}, 0
            completed = True
            loader = _loader(dataset, config.batch_size, config, epoch)
            for index, batch in enumerate(loader):
                batch = _batch_to_device(batch, device)
                optimizer_d.zero_grad(set_to_none=True)
                optimizer_g.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                    prediction = generator(batch["input"], batch["average_alpha"])
                    real_logits = discriminator(batch["condition"], batch["target"])
                    fake_logits = discriminator(batch["condition"], prediction.detach())
                    d_loss = config.adversarial_weight * discriminator_loss(real_logits, fake_logits)
                if not torch.isfinite(d_loss):
                    raise FloatingPointError("Non-finite discriminator loss")
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
                discriminator.requires_grad_(False)
                # Keep D's BatchNorm in training mode for both adversarial steps;
                # only its parameters are frozen while gradients flow into G.
                with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                    recon = reconstruction_loss(prediction, batch["target"])
                    alpha = alpha_loss(prediction, batch["target"], batch["has_alpha"])
                    adversarial = generator_adversarial_loss(discriminator(batch["condition"], prediction))
                    color_loss = (lip_color_loss(color, prediction, batch["target"], batch["mask"],
                                  loss_type=config.color_loss_type) if region == "lip" else prediction.sum() * 0)
                    g_loss = (config.recon_weight * recon + config.alpha_weight * alpha
                              + config.adversarial_weight * adversarial + config.color_weight * color_loss)
                if not torch.isfinite(g_loss):
                    raise FloatingPointError("Non-finite generator loss")
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_g)
                scaler.update()
                discriminator.requires_grad_(True)
                discriminator.train()
                number = len(prediction)
                for name, value in (("generator", g_loss), ("discriminator", d_loss),
                                    ("reconstruction", recon), ("alpha", alpha), ("color", color_loss)):
                    sums[name] = sums.get(name, 0.0) + value.item() * number
                count += number
                steps += 1
                if config.max_steps is not None and steps >= config.max_steps:
                    completed = index + 1 == len(loader)
                    break
            validation = validate_generator(generator, prepared, region, config, device)
            payload = {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                       "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                       "scaler": scaler.state_dict(), "region": region, "base_channels": config.base_channels,
                       "architecture": config.architecture, "average_alpha": dataset.average_alpha.cpu(),
                       "geometry_path": str(geometry_destination), "epoch": epoch + int(completed),
                       "steps": steps, "manifest_sha256": manifest_hash, "config": asdict(config),
                       "partial_epoch": not completed,
                       "training_complete": completed and epoch + 1 >= config.epochs,
                       "limited_training": config.max_steps is not None or config.epochs < 55 or config.base_channels < 64,
                       "paper_training_schedule_complete": completed and epoch + 1 >= 55,
                       "validation": validation}
            _atomic_save(payload, checkpoint)
            _log(output / "history.jsonl", {"stage": region, "epoch": epoch + 1, "steps": steps,
                 "seconds": round(time.perf_counter() - began, 2), "partial_epoch": not completed,
                 **{key: value / max(count, 1) for key, value in sums.items()}, "validation": validation})
        results[region] = {"checkpoint": str(checkpoint), "steps": steps}
        del generator, discriminator, optimizer_g, optimizer_d
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results
