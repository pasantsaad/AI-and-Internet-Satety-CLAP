"""Train a YOLO object-detection model on the laptop."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model/checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Lower this if RAM/VRAM is limited")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, 0")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="smart-trash")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Create/update training/data.yaml first."
        )

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print("\nTraining finished.")
    print(f"Inspect the run directory: {Path(args.project) / args.name}")
    print("The best checkpoint is normally in weights/best.pt.")
    print("Evaluate on a held-out test set before relying on this model.")


if __name__ == "__main__":
    main()
