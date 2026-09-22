"""PyTorch reader for prepared regional samples (all RGB/alpha values in [0,1])."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import REGIONS
from .prepare import verify_artifact_hash


class MakeupDataset(Dataset):
    def __init__(self, prepared_dir: str | Path, region: str, split: str = "train",
                 augment: bool = False, verify_hashes: bool = True):
        self.root = Path(prepared_dir)
        if region not in REGIONS or split not in {"train", "val", "test", "all"}:
            raise ValueError("Invalid region or split")
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.records = [record for record in self.manifest["records"] if record["region"] == region
                        and (split == "all" or record["split"] == split)]
        self.region, self.split, self.augment = region, split, augment and split == "train"
        self.verify_hashes = verify_hashes
        self._verified_artifacts: dict[str, tuple[int, int]] = {}
        if self.manifest.get("geometry"):
            self._verify(self.manifest["geometry"], self.manifest.get("geometry_sha256"))
        self._verify(self.manifest["average_alpha"][region],
                     self.manifest.get("average_alpha_sha256", {}).get(region))
        prior = np.load(self.root / self.manifest["average_alpha"][region], allow_pickle=False).astype(np.float32)
        if prior.ndim == 2:
            prior = prior[..., None]
        self.average_alpha = torch.from_numpy(np.ascontiguousarray(prior.transpose(2, 0, 1)))

    def __len__(self):
        return len(self.records)

    def _verify(self, relative_path: str, expected: str | None):
        if not self.verify_hashes or expected is None:
            return
        path = self.root / relative_path
        info = path.stat()
        signature = (info.st_mtime_ns, info.st_size)
        if self._verified_artifacts.get(relative_path) != signature:
            verify_artifact_hash(path, expected)
            self._verified_artifacts[relative_path] = signature

    def __getitem__(self, index):
        record = self.records[index]
        # Hash once per unchanged file in this dataset/worker, not every epoch.
        self._verify(record["path"], record.get("sha256"))
        with np.load(self.root / record["path"], allow_pickle=False) as sample:
            arrays = {key: sample[key].astype(np.float32) for key in ("input", "condition", "target", "mask")}
            has_alpha = float(sample["has_alpha"])
        if self.augment:
            # Perturb only the affine input: target/condition/prior must stay
            # in canonical coordinates. Torch RNG respects DataLoader seeding.
            image = arrays["input"]
            h, w = image.shape[:2]
            scale = 0.95 + 0.10 * float(torch.rand(()))
            matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), 0, scale)
            matrix[:, 2] += (torch.rand(2).numpy() - 0.5) * np.array([w, h]) * 0.06
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            if float(torch.rand(())) < 0.25:
                image = cv2.GaussianBlur(image, (3, 3), 0.4 + float(torch.rand(())) * 0.5)
            arrays["input"] = image
        result = {key: torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))
                  for key, value in arrays.items()}
        result.update(has_alpha=torch.tensor(has_alpha, dtype=torch.float32),
                      average_alpha=self.average_alpha, path=record["path"], source=record["source"],
                      source_id=record.get("source_id", record["source"]), region=self.region)
        return result
