"""Deterministic, source-disjoint paired-data preparation without PyTorch.

Natural faces in a local dataset are not automatically verified makeup-free.
Filter the source collection manually for a paper-matched reproduction.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .geometry import (REGIONS, DEFAULT_MODEL, FaceDetector, affine_align, build_canonical,
                       crop_region, read_rgb, region_landmarks, region_mask, tps_maps)
from .synthesis import alpha_blend, composite_templates, eye_kmeans_pseudo, generate_style

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__", "docs", "doc", "metadata", "assets", "thumbnails"}
IGNORED_IMAGES = {"ffhq-piecharts.png", "ffhq-teaser.png"}


def _source_name(path: Path) -> str:
    name = str(path).lower()
    return "ffhq" if "ffhq" in name else "fairface" if "fairface" in name else "other"


def inventory_images(root: str | Path, sources: list[str] | tuple[str, ...] | None = None) -> list[Path]:
    """Follow dataset junctions, deduplicate real paths, and ignore documentation."""
    root = Path(root).absolute()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    directories_seen, files_seen, paths = set(), set(), []
    for directory, dirs, names in os.walk(root, followlinks=True):
        real_directory = os.path.normcase(str(Path(directory).resolve()))
        if real_directory in directories_seen:
            dirs[:] = []
            continue
        directories_seen.add(real_directory)
        dirs[:] = sorted(d for d in dirs if d.lower() not in IGNORED_DIRECTORIES and not d.startswith("."))
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix.lower() not in IMAGE_SUFFIXES or name.lower() in IGNORED_IMAGES:
                continue
            if sources and _source_name(path) not in sources:
                continue
            real_path = os.path.normcase(str(path.resolve()))
            if real_path in files_seen:
                continue
            files_seen.add(real_path)
            paths.append(path)
    return sorted(paths, key=lambda path: str(path).lower())


def inventory_dataset(root: str | Path, sources=None) -> dict:
    paths = inventory_images(root, sources=sources)
    groups = {}
    for path in paths:
        group = _source_name(path)
        groups[group] = groups.get(group, 0) + 1
    return {"dataset_root": str(Path(root).absolute()), "total_images": len(paths),
            "groups": groups, "examples": [str(path) for path in paths[:8]],
            "notes": ["Images are not verified makeup-free or occlusion-free.",
                      "Documentation images are excluded; directory junctions are followed."]}


def source_split(source_ids, seed: int = 42, train_fraction: float = 0.8,
                 val_fraction: float = 0.1) -> dict[str, str]:
    """Split unique original-image IDs before deriving any synthetic variants."""
    ids = sorted(set(map(str, source_ids)))
    if not 0 < train_fraction <= 1 or not 0 <= val_fraction < 1 or train_fraction + val_fraction > 1:
        raise ValueError("Invalid train/validation fractions")
    order = np.random.default_rng(seed).permutation(len(ids))
    n = len(ids)
    if n >= 3 and train_fraction < 1 and val_fraction > 0:
        train_count = min(max(1, int(n * train_fraction)), n - 2)
        val_count = min(max(1, int(n * val_fraction)), n - train_count - 1)
    else:
        train_count = max(1, int(n * train_fraction)) if n else 0
        val_count = min(int(n * val_fraction), n - train_count)
    result = {}
    for position, index in enumerate(order):
        result[ids[index]] = "train" if position < train_count else "val" if position < train_count + val_count else "test"
    return result


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_artifact_hash(path: str | Path, expected_sha256: str | None) -> bool:
    """Verify a recorded artifact, tolerating legacy manifests without hashes."""
    if expected_sha256 is None:
        return False
    actual = _digest(Path(path))
    if actual != expected_sha256:
        raise ValueError(f"Prepared artifact SHA-256 mismatch: {path}; expected {expected_sha256}, got {actual}")
    return True


def verify_prepared_integrity(prepared_dir: str | Path, verify_samples: bool = True) -> dict:
    """Read-only verification of available geometry/prior/sample checksums.

    Missing checksum fields identify legacy, unverifiable artifacts; they do
    not make older prepared datasets unreadable.
    """
    root = Path(prepared_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    artifacts = [(manifest["geometry"], manifest.get("geometry_sha256"))]
    prior_hashes = manifest.get("average_alpha_sha256", {})
    artifacts.extend((path, prior_hashes.get(region)) for region, path in manifest["average_alpha"].items())
    if verify_samples:
        artifacts.extend((record["path"], record.get("sha256")) for record in manifest["records"])
    checked = sum(verify_artifact_hash(root / path, expected) for path, expected in artifacts)
    return {"checked": checked, "without_hash": len(artifacts) - checked,
            "dataset_fingerprint": manifest.get("dataset_fingerprint")}


def _provenance(model_path: str | Path) -> dict:
    package = Path(__file__).resolve().parent
    model = Path(model_path)
    return {
        "code_sha256": {name: _digest(package / name) for name in ("geometry.py", "synthesis.py", "prepare.py")},
        "landmark_model": str(model.absolute()),
        # A substituted detector in tests may not use a model file at all.
        "landmark_model_sha256": _digest(model) if model.is_file() else None,
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
    }


def _seed(seed: int, identity: str, variant: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}:{variant}".encode()).digest()[:8], "little")


def _save_rgb(path: Path, rgb: np.ndarray):
    if rgb.dtype != np.uint8:
        rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Unable to encode preview: {path}")
    encoded.tofile(path)


def _remap(image, maps):
    return cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _warp_rgba(rgba, maps):
    # Interpolate premultiplied RGB so transparent boundary pixels cannot bleed.
    premultiplied = rgba.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    warped = _remap(premultiplied, maps)
    warped[..., :3] = np.divide(warped[..., :3], warped[..., 3:4],
                                out=np.zeros_like(warped[..., :3]), where=warped[..., 3:4] > 1e-7)
    return warped


def prepare_dataset(dataset_root: str | Path, output_dir: str | Path,
                    max_images: int | None = None, variants: int = 3, seed: int = 42,
                    image_size: int = 256, model_path: str | Path | None = None,
                    real_makeup_dir: str | Path | None = None,
                    max_real_makeup: int | None = None, real_preview_count: int = 8,
                    sources=None, canvas_size: int = 512, overwrite: bool = False) -> dict:
    """Write float16 regional NPZ pairs and train-only canonical geometry/priors.

    ``max_images`` bounds original synthetic source images. Real-makeup inputs
    are opt-in and separately bounded by the same number when supplied.
    Failures are recorded in the manifest rather than silently used as faces.
    """
    if variants < 1 or image_size < 32 or canvas_size < 32:
        raise ValueError("variants must be positive and image/canvas sizes >= 32")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    if max_real_makeup is not None and max_real_makeup < 1:
        raise ValueError("max_real_makeup must be positive")
    if real_preview_count < 0:
        raise ValueError("real_preview_count must be nonnegative")
    output = Path(output_dir).absolute()
    if (output / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Prepared data already exists at {output}; use overwrite=True to replace the manifest")
    provenance = _provenance(model_path or DEFAULT_MODEL)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    if real_makeup_dir and real_preview_count:
        (output / "real_eye_previews").mkdir(exist_ok=True)
    paths = inventory_images(dataset_root, sources=sources)
    if not paths:
        raise ValueError("No source images found")
    rng = np.random.default_rng(seed)
    if max_images is not None and len(paths) > max_images:
        indices = sorted(rng.choice(len(paths), max_images, replace=False))
        paths = [paths[index] for index in indices]
    real_paths = inventory_images(real_makeup_dir) if real_makeup_dir else []
    if real_makeup_dir and not real_paths:
        raise ValueError(f"No real makeup images found in {Path(real_makeup_dir).absolute()}")
    real_limit = max_images if max_real_makeup is None else max_real_makeup
    if real_limit is not None and len(real_paths) > real_limit:
        indices = sorted(rng.choice(len(real_paths), real_limit, replace=False))
        real_paths = [real_paths[index] for index in indices]
    sources_by_id, skipped, duplicates = {}, [], []
    for path, kind in [(p, "synthetic") for p in paths] + [(p, "real_eye") for p in real_paths]:
        try:
            identity = _digest(path)
        except OSError as error:
            skipped.append({"source": str(path), "reason": str(error)})
            continue
        if identity in sources_by_id:
            duplicates.append({"source": str(path), "same_as": str(sources_by_id[identity]["path"])})
            continue
        sources_by_id[identity] = {"path": path, "kind": kind}
    # A single identity has one split even when its image occurs in both roots.
    splits = source_split(sources_by_id, seed=seed)
    detected = []
    print(f"Detecting faces in {len(sources_by_id)} unique source images...", flush=True)
    with FaceDetector(model_path or DEFAULT_MODEL) as detector:
        for index, (identity, item) in enumerate(sources_by_id.items()):
            try:
                image = read_rgb(item["path"])
                landmarks = detector.detect(image)
                if landmarks is None:
                    raise ValueError("Expected exactly one detectable face; no face or multiple faces found")
                detected.append({**item, "id": identity, "split": splits[identity],
                                 "landmarks": landmarks, "shape": image.shape[:2]})
            except (ValueError, OSError, RuntimeError) as error:
                skipped.append({"source": str(item["path"]), "reason": str(error)})
            if (index + 1) % 25 == 0 or index + 1 == len(sources_by_id):
                print(f"  Face pass: {index + 1}/{len(sources_by_id)}; valid={len(detected)}", flush=True)
    training_faces = [item["landmarks"] for item in detected if item["split"] == "train" and item["kind"] == "synthetic"]
    if not training_faces:
        raise ValueError("No valid synthetic training faces; increase --max-images or inspect source images")
    geometry = build_canonical(training_faces, canvas_size=canvas_size)
    geometry.save(output / "geometry.npz")
    canonical_controls = region_landmarks(geometry.landmarks)
    masks = {region: (crop_region(region_mask(geometry, region), geometry, region, image_size) > 0.5)
             .astype(np.float32)[..., None] for region in REGIONS}
    alpha_sums = {region: np.zeros((image_size, image_size, 1), np.float64) for region in REGIONS}
    alpha_counts = dict.fromkeys(REGIONS, 0)
    records, source_records = [], []
    preview_count = 0
    real_preview_written = 0
    for source_index, item in enumerate(detected):
        source_records.append({"source": str(item["path"]), "id": item["id"], "split": item["split"], "kind": item["kind"]})
        image = read_rgb(item["path"])
        aligned, points, _ = affine_align(image, item["landmarks"], geometry)
        aligned = aligned.astype(np.float32) / 255
        controls = region_landmarks(points)
        to_canonical = tps_maps(controls, canonical_controls, (canvas_size, canvas_size))
        to_face = tps_maps(canonical_controls, controls, (canvas_size, canvas_size))
        count = variants if item["kind"] == "synthetic" else 1
        for variant in range(count):
            variant_seed = _seed(seed, item["id"], variant)
            if item["kind"] == "synthetic":
                templates = generate_style(geometry, np.random.default_rng(variant_seed))
                full_rgba = composite_templates(templates)
                synthetic = alpha_blend(aligned, _warp_rgba(full_rgba, to_face))
                condition = _remap(synthetic, to_canonical)
                used_regions = REGIONS
                if variant == 0 and preview_count < 3:
                    scale = min(1, 512 / max(image.shape[:2]))
                    display = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
                    preview_maps = tps_maps(canonical_controls, region_landmarks(item["landmarks"] * scale), display.shape[:2])
                    rendered = alpha_blend(display, _warp_rgba(full_rgba, preview_maps))
                    montage = np.concatenate((display.astype(np.float32) / 255, rendered), axis=1)
                    _save_rgb(output / "previews" / f"{item['id'][:16]}_before_after.png", montage)
                    preview_count += 1
            else:
                synthetic = aligned
                condition = _remap(synthetic, to_canonical)
                templates = {"eye": eye_kmeans_pseudo(condition, region_mask(geometry, "eye"), seed=variant_seed)}
                used_regions = ("eye",)
                if real_preview_written < real_preview_count:
                    pseudo = templates["eye"]
                    alpha = np.clip(pseudo[..., 3:4], 0, 1)
                    alpha_rgb = np.repeat(alpha, 3, axis=2)
                    accent = np.array([1.0, 0.1, 0.8], np.float32)[None, None]
                    highlighted = np.clip(condition * (1 - 0.65 * alpha) + accent * (0.65 * alpha), 0, 1)
                    montage = np.concatenate((condition, alpha_rgb, highlighted), axis=1)
                    _save_rgb(output / "real_eye_previews" /
                              f"{item['id'][:16]}_canonical_alpha_overlay.png", montage)
                    real_preview_written += 1
            for region in used_regions:
                target = crop_region(templates[region], geometry, region, image_size)
                relative = f"samples/{item['id']}_{variant:02d}_{region}.npz"
                np.savez_compressed(output / relative,
                    input=crop_region(synthetic, geometry, region, image_size).astype(np.float16),
                    condition=crop_region(condition, geometry, region, image_size).astype(np.float16),
                    target=target.astype(np.float16), mask=masks[region].astype(np.float16),
                    has_alpha=np.array(item["kind"] == "synthetic", dtype=np.float32))
                records.append({"region": region, "split": item["split"], "path": relative,
                                "source": str(item["path"]), "source_id": item["id"],
                                "variant": variant, "kind": item["kind"],
                                "sha256": _digest(output / relative)})
                if item["split"] == "train" and item["kind"] == "synthetic":
                    alpha_sums[region] += target[..., 3:4]
                    alpha_counts[region] += 1
        if (source_index + 1) % 10 == 0 or source_index + 1 == len(detected):
            print(f"  Pair pass: {source_index + 1}/{len(detected)}; regional pairs={len(records)}", flush=True)
    average_alpha = {}
    for region in REGIONS:
        relative = f"average_alpha_{region}.npy"
        np.save(output / relative, (alpha_sums[region] / max(alpha_counts[region], 1)).astype(np.float32))
        average_alpha[region] = relative
    summary = {
        "unique_sources": len(sources_by_id),
        "detected_sources": len(detected),
        "detected_synthetic_sources": sum(item["kind"] == "synthetic" for item in detected),
        "detected_real_makeup_sources": sum(item["kind"] == "real_eye" for item in detected),
        "records": len(records),
        "records_by_kind": {
            kind: sum(record["kind"] == kind for record in records)
            for kind in ("synthetic", "real_eye")
        },
        "records_by_region": {
            region: sum(record["region"] == region for record in records)
            for region in REGIONS
        },
        "real_eye_previews": real_preview_written,
    }
    manifest = {
        "schema_version": 1, "paper": "https://arxiv.org/html/2509.02445v2",
        "dataset_root": str(Path(dataset_root).absolute()), "seed": seed,
        "variants": variants, "image_size": image_size, "canvas_size": canvas_size,
        "geometry": "geometry.npz", "average_alpha": average_alpha,
        "geometry_sha256": _digest(output / "geometry.npz"),
        "average_alpha_sha256": {region: _digest(output / path) for region, path in average_alpha.items()},
        "provenance": provenance,
        "average_alpha_training_counts": alpha_counts,
        "canonical_training_sources": [item["id"] for item in detected if item["split"] == "train" and item["kind"] == "synthetic"],
        "records": records, "sources": source_records, "skipped": skipped, "duplicates": duplicates,
        "summary": summary,
        "notes": ["Split unit is SHA-256 of original image bytes; all variants share its split.",
                  "Different photographs of the same person are not identity-deduplicated.",
                  "Canonical geometry and average alpha use synthetic TRAINING sources only.",
                  "Source faces were not verified makeup-free or occlusion-free.",
                  "Procedural templates and MediaPipe geometric masks substitute for unavailable paper assets.",
                  "Real makeup images supervise the eye model only, matching the paper's k-means pseudo-label path.",
                  "Real eye pseudo labels use their estimated alpha to weight RGB reconstruction but do not supervise alpha L1."],
    }
    # This includes artifact bytes and generation provenance, so a renderer
    # change cannot be mistaken for the same training data solely because the
    # source IDs, split, filenames, and random seed stayed unchanged.
    manifest["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # Write the manifest last so interrupted generation is never mistaken for completion.
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest
