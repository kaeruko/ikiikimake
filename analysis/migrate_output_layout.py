"""Migrate legacy output folders into the video-name based layout.

Run this once after pulling the output-layout cleanup commit. The script never
silently overwrites different files. It also rewrites local absolute paths
stored in text metadata so moved experiments remain usable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


ROOT = Path(__file__).resolve().parents[1]

MAPPINGS = {
    "cheek01_auto": "cheek01/auto",
    "makeup_video_search_before_60_180__last_180s": "makeup",
    "makeup_video_search_makeup1": "makeup/experiments/narrow",
    "makeup_video_search_makeup1_closeup": "makeup/experiments/closeup",
    "makeup_video_search_wide": "makeup/experiments/wide",
    "makeup_video_search_final_window": "makeup/experiments/late_window",
    "makeup_video_search_notebook": "makeup/experiments/notebook_partial",
    "makeup2_auto": "makeup2/auto",
    "makeup_video_search": "makeup2/search",
}

TEXT_SUFFIXES = {".csv", ".html", ".json", ".jsonl", ".txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if not source.is_dir():
        raise NotADirectoryError(source)
    destination.mkdir(parents=True, exist_ok=True)

    for item in sorted(source.iterdir(), key=lambda path: path.name):
        target = destination / item.name
        if item.is_dir():
            merge_directory(item, target)
            if item.exists() and not any(item.iterdir()):
                item.rmdir()
            continue

        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"Destination is not a file: {target}")
            if sha256_file(item) != sha256_file(target):
                raise FileExistsError(
                    f"Different file already exists; refusing overwrite: {target}"
                )
            item.unlink()
        else:
            shutil.move(str(item), str(target))

    if source.exists() and not any(source.iterdir()):
        source.rmdir()


def rewrite_paths(directory: Path, old_absolute: str, new_absolute: str) -> None:
    if not directory.is_dir():
        return
    replacements = (
        (old_absolute, new_absolute),
        (old_absolute.replace("\\", "\\\\"), new_absolute.replace("\\", "\\\\")),
    )
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        temp = path.with_name(path.name + ".rewrite_tmp")
        changed = False
        try:
            with path.open("r", encoding="utf-8", newline="") as source, temp.open(
                "w", encoding="utf-8", newline=""
            ) as destination:
                for chunk in source:
                    updated = chunk
                    for old, new in replacements:
                        updated = updated.replace(old, new)
                    if updated != chunk:
                        changed = True
                    destination.write(updated)
        except UnicodeDecodeError:
            if temp.exists():
                temp.unlink()
            continue
        except Exception:
            if temp.exists():
                temp.unlink()
            raise

        if changed:
            try:
                temp.replace(path)
            except PermissionError:
                # Windows can refuse replacing a read-only generated file.
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
                temp.replace(path)
        else:
            temp.unlink()


def refresh_selected_pair(directory: Path) -> None:
    selected_path = directory / "selected_pair.json"
    if not selected_path.is_file():
        return

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    for phase in ("before", "after"):
        item = selected.get(phase)
        if not isinstance(item, dict):
            raise ValueError(f"{selected_path}: missing {phase}")
        image = Path(item["image_path"])
        roi_dir = Path(item["roi_dir"])
        required = {
            "image_sha256": image,
            "roi_masks_sha256": roi_dir / "roi_masks.npz",
            "roi_points_sha256": roi_dir / "roi_points.json",
            "roi_overlay_sha256": roi_dir / "roi_overlay.png",
        }
        for key, path in required.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cannot refresh {selected_path}: missing {path}"
                )
            item[key] = sha256_file(path)

    selected_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    outputs = ROOT / "outputs"
    for old_name, new_name in MAPPINGS.items():
        old = outputs / old_name
        new = outputs / new_name
        old_absolute = str(old.resolve())
        new_absolute = str(new.resolve())

        if old.exists():
            print(f"move: {old.relative_to(ROOT)} -> {new.relative_to(ROOT)}")
            merge_directory(old, new)

        rewrite_paths(new, old_absolute, new_absolute)
        refresh_selected_pair(new)

    print("output migration complete")


if __name__ == "__main__":
    main()
