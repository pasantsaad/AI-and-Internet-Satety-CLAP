"""Basic checks for a YOLO detection dataset YAML and label files."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training/data.yaml", help="Dataset YAML path")
    args = parser.parse_args()

    yaml_path = Path(args.data).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text())
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    names = config.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}

    print(f"YAML: {yaml_path}")
    print(f"Dataset root: {root}")
    print("Class mapping:")
    for class_id, name in names.items():
        print(f"  {class_id}: {name}")

    for split in ("train", "val", "test"):
        entry = config.get(split)
        if not entry:
            print(f"\n[{split}] not configured")
            continue

        image_path = Path(entry)
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()

        print(f"\n[{split}] images: {image_path}")
        if not image_path.exists():
            print("  WARNING: path does not exist")
            continue

        image_files = [
            p for p in image_path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        print(f"  images found: {len(image_files)}")

        # Standard YOLO layout: labels directory mirrors images directory.
        label_dir = root / "labels" / split
        if not label_dir.exists():
            print(f"  WARNING: expected labels directory not found: {label_dir}")
            continue

        missing_labels = []
        invalid_lines = []
        label_files = 0
        for image in image_files:
            label_file = label_dir / f"{image.stem}.txt"
            if not label_file.exists():
                missing_labels.append(image.name)
                continue
            label_files += 1
            for line_no, line in enumerate(label_file.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    invalid_lines.append((label_file.name, line_no, "expected 5 values"))
                    continue
                try:
                    class_id = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_lines.append((label_file.name, line_no, "non-numeric value"))
                    continue
                if class_id not in {int(k) for k in names.keys()}:
                    invalid_lines.append((label_file.name, line_no, f"unknown class id {class_id}"))
                if any(x < 0 or x > 1 for x in coords):
                    invalid_lines.append((label_file.name, line_no, "coordinates outside [0, 1]"))

        print(f"  label files found for images: {label_files}/{len(image_files)}")
        if missing_labels:
            print(f"  WARNING: missing label files for {len(missing_labels)} images (first 10): {missing_labels[:10]}")
        if invalid_lines:
            print(f"  WARNING: invalid label rows: {len(invalid_lines)} (first 10):")
            for item in invalid_lines[:10]:
                print("   ", item)
        if not missing_labels and not invalid_lines:
            print("  Basic label checks passed. Still visually inspect annotations and class mapping.")


if __name__ == "__main__":
    main()
