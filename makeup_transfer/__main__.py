"""Command line entry point; inventory/prepare do not require PyTorch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="arXiv 2509.02445v2: transparent makeup extraction and transfer")
    commands = root.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="Inspect local image sources without changing them")
    inventory.add_argument("--datasets", type=Path, default=Path("datasets"))
    prepare = commands.add_parser("prepare", help="Generate training/validation/test pairs")
    prepare.add_argument("--datasets", type=Path, default=Path("datasets"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-images", type=int)
    prepare.add_argument("--variants", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--landmark-model", type=Path, default=Path("models/face_landmarker.task"))
    prepare.add_argument("--real-makeup", type=Path, help="Optional real makeup images for paper-style eye pseudo labels")
    prepare.add_argument("--max-real-makeup", type=int,
                         help="Bound real-makeup images independently; defaults to --max-images when set")
    prepare.add_argument("--real-preview-count", type=int, default=8,
                         help="Save this many real-eye pseudo-label review montages")
    prepare.add_argument("--sources", nargs="+", choices=("ffhq", "fairface"))
    prepare.add_argument("--canvas-size", type=int, default=512)
    train = commands.add_parser("train", help="Pretrain lip regressor then train regional GANs")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--regions", nargs="+", choices=("eye", "lip", "cheek"), default=["eye", "lip", "cheek"])
    for name, default in (("epochs", 55), ("batch-size", 8), ("base-channels", 64),
                          ("color-epochs", 10), ("color-batch-size", 32), ("workers", 0), ("seed", 42)):
        train.add_argument("--" + name, type=int, default=default)
    train.add_argument("--max-steps", type=int, help="Total optimizer steps per region, for bounded smoke tests")
    train.add_argument("--color-max-steps", type=int)
    train.add_argument("--architecture", choices=("paper", "strided"), default="paper")
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--discriminator-lr", type=float, default=2e-4)
    train.add_argument("--color-loss-type", choices=("mse", "l2"), default="mse")
    train.add_argument("--device", default="auto")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--no-augment", action="store_true")
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate held-out regional RGBA extraction (not the paper benchmark)")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoints", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--regions", nargs="+", choices=("eye", "lip", "cheek"), default=["eye", "lip", "cheek"])
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--max-samples", type=int)
    benchmark = commands.add_parser("benchmark", help="Synthetic shared-style transfer between distinct held-out faces")
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--checkpoints", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--split", choices=("val", "test"), default="test")
    benchmark.add_argument("--max-pairs", type=int, default=8)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--landmark-model", type=Path, default=Path("models/face_landmarker.task"))
    benchmark.add_argument("--perceptual", action="store_true", help="Compute LPIPS/FID using optional packages and their pretrained weights")
    extract = commands.add_parser("extract", help="Extract canonical RGBA makeup patches from one reference image")
    extract.add_argument("--reference", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True, help="Output .npz file for extracted RGBA patches")
    extract.add_argument("--checkpoints", type=Path, required=True)
    extract.add_argument("--geometry", type=Path)
    extract.add_argument("--landmark-model", type=Path, default=Path("models/face_landmarker.task"))
    extract.add_argument("--device", default="auto")
    extract.add_argument("--regions", nargs="+", choices=("eye", "lip", "cheek"), default=["eye", "lip", "cheek"])
    for name in ("transfer", "video"):
        command = commands.add_parser(name, help="Apply one extracted style to an image" if name == "transfer" else "Reuse one style throughout a video")
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--target", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--checkpoints", type=Path, required=True)
        command.add_argument("--geometry", type=Path)
        command.add_argument("--landmark-model", type=Path, default=Path("models/face_landmarker.task"))
        command.add_argument("--device", default="auto")
        command.add_argument("--regions", nargs="+", choices=("eye", "lip", "cheek"), default=["eye", "lip", "cheek"])
        command.add_argument("--strength", type=float, default=1.0)
        if name == "transfer":
            command.add_argument("--semantic-mask", type=Path)
        else:
            command.add_argument("--max-frames", type=int)
            command.add_argument("--smoothing", type=float, default=0.6)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "inventory":
        from .prepare import inventory_dataset
        result = inventory_dataset(args.datasets)
    elif args.command == "prepare":
        from .prepare import prepare_dataset
        result = prepare_dataset(args.datasets, args.output, max_images=args.max_images,
                                 variants=args.variants, seed=args.seed, model_path=args.landmark_model,
                                 real_makeup_dir=args.real_makeup, max_real_makeup=args.max_real_makeup,
                                 real_preview_count=args.real_preview_count,
                                 sources=args.sources, canvas_size=args.canvas_size)
        result = {"manifest": str(args.output / "manifest.json"),
                  "records": len(result.get("records", [])), "summary": result.get("summary", {})}
    elif args.command == "train":
        from .train import TrainConfig, train
        fields = {key: value for key, value in vars(args).items() if key in TrainConfig.__dataclass_fields__}
        fields["augment"] = not args.no_augment
        result = train(args.data, args.output, regions=args.regions, config=TrainConfig(**fields), resume=args.resume)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.data, args.checkpoints, args.output, regions=args.regions,
                          split=args.split, device=args.device, batch_size=args.batch_size, max_samples=args.max_samples)
    elif args.command == "benchmark":
        from .benchmark import benchmark
        result = benchmark(args.data, args.checkpoints, args.output, split=args.split,
                           max_pairs=args.max_pairs, seed=args.seed, device=args.device,
                           landmark_model=args.landmark_model, perceptual=args.perceptual)
    elif args.command == "extract":
        from .inference import run_extract
        result = run_extract(args.reference, args.output, args.checkpoints,
                             geometry_path=args.geometry, landmark_model=args.landmark_model,
                             device=args.device, regions=args.regions)
    else:
        from .inference import run_image, run_video
        kwargs = {key: getattr(args, key) for key in ("reference", "target", "output", "checkpoints",
                  "landmark_model", "device", "regions", "strength")}
        kwargs["geometry_path"] = args.geometry
        if args.command == "transfer":
            result = run_image(**kwargs, semantic_mask=args.semantic_mask)
        else:
            result = run_video(**kwargs, smoothing=args.smoothing, max_frames=args.max_frames)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
