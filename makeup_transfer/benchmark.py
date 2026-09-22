"""Held-out cross-face synthetic transfer evaluation following the paper's protocol.

The local data, procedural templates, landmark tracker and geometric masks differ
from the publication. These measurements are implementation diagnostics, not a
reproduction of its published scores.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .geometry import REGIONS, CanonicalGeometry, FaceDetector, crop_region, face_region_mask
from .inference import (
    MakeupStyle, _checkpoint_paths, _load_checkpoint, _read_rgb,
    _resolve_geometry, _write_image, extract_style, render_style,
)
from .synthesis import generate_style


def psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RGB PSNR in dB for uint8-range images; exact matches return infinity."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("PSNR requires equally shaped H x W x 3 RGB images.")
    errors = (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target.shape[:2] or not np.any(mask):
            raise ValueError("A PSNR mask must match the image and select at least one pixel.")
        errors = errors[mask]
    mse = float(errors.mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def _held_out_sources(manifest: Mapping, split: str) -> list[dict]:
    if split not in {"val", "test"}:
        raise ValueError("Benchmark split must be 'val' or 'test'; training faces are never evaluated.")
    sources = manifest.get("sources", [])
    training_ids = set(manifest.get("canonical_training_sources", []))
    training_ids.update(source["id"] for source in sources if source["split"] == "train")
    training_ids.update(record["source_id"] for record in manifest.get("records", []) if record["split"] == "train")
    selected = {}
    for source in sources:
        if source["split"] == split and source.get("kind") == "synthetic":
            if source["id"] in training_ids:
                raise ValueError(f"Training/held-out source overlap in manifest: {source['id']}")
            selected.setdefault(source["id"], source)
    if len(selected) < 2:
        raise ValueError(f"At least two distinct held-out synthetic sources are required in split={split!r}.")
    return list(selected.values())


def _json_values(value):
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def _review_grid(images: list[np.ndarray], labels: list[str], panel_size: int = 320) -> np.ndarray:
    panels = []
    for rgb, label in zip(images, labels):
        panel = np.full((panel_size + 34, panel_size, 3), 245, dtype=np.uint8)
        scale = min(panel_size / rgb.shape[1], panel_size / rgb.shape[0])
        resized = cv2.resize(rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))))
        y, x = 34 + (panel_size - resized.shape[0]) // 2, (panel_size - resized.shape[1]) // 2
        panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(panel, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (25, 25, 25), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _check_perceptual_dependencies() -> None:
    missing = [name for name in ("lpips", "pytorch_fid") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Optional perceptual metrics require: " + ", ".join(missing)
                           + ". Install lpips and pytorch-fid, or omit --perceptual.")


def _perceptual_metrics(output: Path, results: list[dict], device: str) -> dict:
    import torch
    import lpips
    from pytorch_fid.fid_score import calculate_fid_given_paths

    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    network = lpips.LPIPS(net="alex").to(selected_device).eval()

    def tensor(path):
        rgb = cv2.resize(_read_rgb(path), (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None].to(selected_device) / 127.5 - 1

    with torch.inference_mode():
        for result in results:
            result["lpips_alex_256"] = float(network(tensor(result["prediction"]), tensor(result["ground_truth"])).item())
    metrics = {"lpips_alex_256_mean": float(np.mean([item["lpips_alex_256"] for item in results])),
               "lpips_image_resize": [256, 256], "fid_feature_dimensions": 2048}
    if len(results) < 2:
        metrics["fid"] = metrics["fid_identity_synthetic"] = None
        metrics["fid_skip_reason"] = "FID covariance requires at least two generated pairs."
        return metrics
    arguments = dict(batch_size=1, device=selected_device, dims=2048, num_workers=0)
    metrics["fid"] = float(calculate_fid_given_paths([str(output / "ground_truth"), str(output / "predictions")], **arguments))
    metrics["fid_identity_synthetic"] = float(calculate_fid_given_paths([str(output / "targets"), str(output / "predictions")], **arguments))
    metrics["fid_identity_protocol"] = "Original targets vs. transfer using synthetic references; differs from the paper's real-reference FID(I)."
    metrics["fid_caution"] = "Small-sample FID is unstable; these local scores are not comparable to published dataset scores."
    return metrics


def benchmark(
    prepared_dir: str | Path,
    checkpoints_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    max_pairs: int = 8,
    seed: int = 42,
    device: str = "auto",
    landmark_model: str | Path = "models/face_landmarker.task",
    perceptual: bool = False,
) -> dict:
    """Apply the same sampled style to A and B, extract from A, transfer to B.

    Every original source appears in at most one pair. The split is a holdout of
    original image bytes, not necessarily person identity. Learned inference is
    required; missing weights are an error. Optional metrics may download their
    pretrained feature weights only when explicitly enabled by the caller.
    """
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer.")
    prepared, output = Path(prepared_dir), Path(output_dir)
    manifest_path = prepared / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sources = _held_out_sources(manifest, split)
    paths = _checkpoint_paths(checkpoints_dir, REGIONS)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    provenance = {}
    for region, path in paths.items():
        state = _load_checkpoint(path)
        training_manifest = state.get("manifest_sha256")
        if training_manifest is not None and training_manifest != manifest_hash:
            raise ValueError(f"Checkpoint {path} was trained using a different prepared manifest.")
        provenance[region] = {"manifest_verified": training_manifest == manifest_hash,
                              "training_complete": state.get("training_complete"),
                              "limited_training": state.get("limited_training"),
                              "paper_training_schedule_complete": state.get("paper_training_schedule_complete"),
                              "epoch": int(state.get("epoch", 0)), "steps": int(state.get("steps", 0))}
        del state
    geometry = _resolve_geometry(checkpoints_dir, None, REGIONS)
    prepared_geometry = CanonicalGeometry.load(prepared / manifest.get("geometry", "geometry.npz"))
    if geometry.canvas_size != prepared_geometry.canvas_size or geometry.boxes != prepared_geometry.boxes or not np.array_equal(geometry.landmarks, prepared_geometry.landmarks):
        raise ValueError("Checkpoint canonical geometry differs from prepared data geometry.")
    if perceptual:
        _check_perceptual_dependencies()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Benchmark output must be empty to avoid mixing metric samples: {output}")
    for directory in ("references", "targets", "ground_truth", "predictions", "reviews"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sources = [sources[index] for index in rng.permutation(len(sources))]
    results, skipped = [], []
    with FaceDetector(landmark_model) as detector:
        for index in range(0, len(sources) - 1, 2):
            if len(results) >= max_pairs:
                break
            source_a, source_b = sources[index:index + 2]
            images, landmarks = [], []
            failure = None
            for source in (source_a, source_b):
                path = Path(source["source"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != source["id"]:
                    raise ValueError(f"Source bytes changed since dataset preparation: {path}")
                rgb = _read_rgb(path)
                points = detector.detect(rgb)
                if points is None:
                    failure = f"No usable face in {path}"
                    break
                images.append(rgb)
                landmarks.append(points)
            if failure:
                skipped.append({"source_a": source_a["source"], "source_b": source_b["source"], "reason": failure})
                continue
            templates = generate_style(geometry, rng)
            known_style = MakeupStyle(geometry, {region: crop_region(templates[region], geometry, region) for region in REGIONS})
            reference, _ = render_style(images[0], known_style, landmarks[0])
            ground_truth, makeup_alpha = render_style(images[1], known_style, landmarks[1])
            try:
                learned_style = extract_style(reference, checkpoints_dir, geometry, detector, device=device)
            except ValueError as error:
                if "No usable single face" not in str(error):
                    raise
                skipped.append({"source_a": source_a["source"], "source_b": source_b["source"], "reason": str(error)})
                continue
            prediction, _ = render_style(images[1], learned_style, landmarks[1])
            face_mask = face_region_mask(landmarks[1], prediction.shape[:2]) > 0
            makeup_mask = makeup_alpha > 0.01
            pair_name = f"pair_{len(results):04d}.png"
            files = {"reference": output / "references" / pair_name,
                     "target": output / "targets" / pair_name,
                     "ground_truth": output / "ground_truth" / pair_name,
                     "prediction": output / "predictions" / pair_name,
                     "review": output / "reviews" / pair_name}
            for key, rgb in (("reference", reference), ("target", images[1]), ("ground_truth", ground_truth), ("prediction", prediction)):
                _write_image(files[key], rgb)
            _write_image(files["review"], _review_grid([reference, images[1], ground_truth, prediction],
                         ["A: makeup reference", "B: original target", "B: known ground truth", "B: learned transfer"]))
            outside = ~face_mask
            results.append({
                "source_a": source_a["source"], "source_b": source_b["source"],
                "source_a_id": source_a["id"], "source_b_id": source_b["id"],
                **{key: str(path.resolve()) for key, path in files.items()},
                "psnr_full_db": psnr(prediction, ground_truth),
                "psnr_face_db": psnr(prediction, ground_truth, face_mask),
                "psnr_makeup_db": psnr(prediction, ground_truth, makeup_mask) if np.any(makeup_mask) else None,
                "outside_face_max_abs_error": int(np.abs(prediction.astype(np.int16) - images[1].astype(np.int16))[outside].max(initial=0)),
                "style_extraction": dict(learned_style.metadata),
            })
    if not results:
        raise ValueError("No valid held-out face pairs could be evaluated.")
    summary = {key + "_mean": float(np.mean([item[key] for item in results if item[key] is not None]))
               for key in ("psnr_full_db", "psnr_face_db", "psnr_makeup_db")
               if any(item[key] is not None for item in results)}
    if perceptual:
        summary.update(_perceptual_metrics(output, results, device))
    report = _json_values({
        "protocol": "shared synthetic style on distinct held-out source A and target B",
        "paper": "https://arxiv.org/html/2509.02445v2",
        "prepared_dir": str(prepared.resolve()), "checkpoints_dir": str(Path(checkpoints_dir).resolve()),
        "output_dir": str(output.resolve()), "split": split, "seed": seed,
        "requested_pairs": max_pairs, "evaluated_pairs": len(results),
        "checkpoint_provenance": provenance, "summary": summary, "pairs": results, "skipped": skipped,
        "perceptual_metrics_enabled": perceptual,
        "notes": [
            "These are local synthetic diagnostic scores, not reproduced publication scores.",
            "Source images were not verified makeup-free or occlusion-free.",
            "Holdout uses source-image hashes; different photos of the same person may span splits.",
            "Each source appears at most once; all canonical geometry comes from training sources.",
            "Landmark geometry replaces a semantic face parser and cannot detect hair/hand occlusion.",
            "Full-image PSNR includes unchanged background; face and makeup-support PSNR are also reported.",
            "Optional fid_identity_synthetic uses synthetic references, not the paper's real-reference FID(I) protocol.",
            "The string Infinity denotes an exact pixel match.",
        ],
    })
    report_path = output / "benchmark.json"
    report["report_path"] = str(report_path.resolve())
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report
