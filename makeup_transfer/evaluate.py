"""Held-out mask extraction diagnostics, deliberately distinct from paper benchmarks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from .data import MakeupDataset
from .models import MakeupGenerator
from .train import select_device


def _display_rgba(tensor):
    rgba = tensor.detach().cpu().permute(1, 2, 0).numpy()
    yy, xx = np.indices(rgba.shape[:2])
    checker = (0.72 + 0.18 * ((xx // 16 + yy // 16) % 2))[..., None]
    return np.clip((rgba[..., :3] * rgba[..., 3:] + checker * (1 - rgba[..., 3:])) * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def evaluate(prepared_dir, checkpoints_dir, output_dir, *, regions=("eye", "lip", "cheek"),
             split="test", device="auto", batch_size=1, max_samples=None):
    if split not in ("val", "test"):
        raise ValueError("Use held-out val or test split")
    if batch_size < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError("batch_size and max_samples must be positive")
    prepared, checkpoints, output = Path(prepared_dir), Path(checkpoints_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(device)
    digest = hashlib.sha256((prepared / "manifest.json").read_bytes()).hexdigest()
    report = {"protocol": "held-out regional canonical RGBA extraction", "split": split,
              "paper_benchmark": False,
              "note": "PSNR is for premultiplied RGB mask over black, not full-face transfer. FID/LPIPS/FID(I) are not computed.",
              "regions": {}}
    for region in regions:
        state = torch.load(checkpoints / f"{region}.pt", map_location="cpu", weights_only=True)
        if state.get("manifest_sha256") != digest:
            raise ValueError("Checkpoint and prepared manifest differ; use its original held-out split")
        model = MakeupGenerator(base_channels=state["base_channels"]).to(device).eval()
        model.load_state_dict(state["generator"])
        dataset = MakeupDataset(prepared, region, split=split, augment=False)
        if not len(dataset):
            report["regions"][region] = {"samples": 0, "status": "no_held_out_samples"}
            continue
        limited = Subset(dataset, range(min(len(dataset), max_samples))) if max_samples else dataset
        sums = {"alpha_weighted_rgb_mae": 0.0, "alpha_mae": 0.0, "premultiplied_rgb_mse": 0.0}
        count, alpha_count = 0, 0
        previews = []
        for batch in DataLoader(limited, batch_size=batch_size, shuffle=False):
            reference = batch["input"].to(device)
            average = batch["average_alpha"].to(device)
            target = batch["target"].to(device)
            pred = model(reference, average)
            recon = ((pred[:, :3] - target[:, :3]).abs() * target[:, 3:]).mean((1, 2, 3))
            alpha = (pred[:, 3:] - target[:, 3:]).abs().mean((1, 2, 3))
            valid = batch["has_alpha"].reshape(-1).bool().to(device)
            mse = ((pred[:, :3] * pred[:, 3:] - target[:, :3] * target[:, 3:]) ** 2).mean((1, 2, 3))
            sums["alpha_weighted_rgb_mae"] += recon.sum().item()
            sums["alpha_mae"] += alpha[valid].sum().item()
            sums["premultiplied_rgb_mse"] += mse[valid].sum().item()
            count += len(pred)
            alpha_count += valid.sum().item()
            for index in range(min(len(pred), 4 - len(previews))):
                rgb = np.clip(reference[index].cpu().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                previews.append(np.concatenate((rgb, _display_rgba(target[index]), _display_rgba(pred[index])), axis=1))
        mse = sums["premultiplied_rgb_mse"] / alpha_count if alpha_count else None
        report["regions"][region] = {"samples": count, "synthetic_alpha_samples": alpha_count,
            "alpha_weighted_rgb_mae": sums["alpha_weighted_rgb_mae"] / count,
            "alpha_mae": sums["alpha_mae"] / alpha_count if alpha_count else None,
            "premultiplied_rgb_psnr_db": float(-10 * np.log10(max(mse, 1e-12))) if mse is not None else None,
            "checkpoint_steps": state["steps"], "checkpoint_epoch": state["epoch"]}
        if previews:
            canvas = Image.new("RGB", (previews[0].shape[1], 28 + sum(p.shape[0] for p in previews)), "white")
            ImageDraw.Draw(canvas).text((8, 8), f"{region} | input / synthetic target / prediction (diagnostic)", fill="black")
            y = 28
            for preview in previews:
                canvas.paste(Image.fromarray(preview), (0, y))
                y += preview.shape[0]
            canvas.save(output / f"{region}_comparison.png")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
