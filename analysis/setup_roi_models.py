"""Download versioned official MediaPipe models and verify their SHA-256."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

MODEL_SPECS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
}


def setup_models(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODEL_SPECS.items():
        destination = output / name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing model checksum mismatch: {destination}. Move it aside before retrying.")
            print(f"Verified: {destination}")
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded model checksum mismatch: {name}")
        # Exclusive creation avoids overwriting a concurrent setup or custom model.
        with destination.open("xb") as model_file:
            model_file.write(data)
        print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    setup_models(parser.parse_args().output)
