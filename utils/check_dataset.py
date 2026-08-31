# ============================================================
# FILE: utils/check_dataset.py
#
# PROJECT: PoolGuard — Drowning Detection Model Training
#
# PURPOSE:
# Validate the dataset BEFORE launching a (paid) server
# training job: broken labels found here cost seconds, not
# GPU hours.
#
# RESPONSIBILITIES:
# - Verify the expected YOLO folder layout exists.
# - Verify every image has a label file and vice versa.
# - Verify each label line is valid YOLO format:
#   "class_id cx cy w h" with normalized 0–1 coordinates.
# - Report per-class instance counts (class balance) and
#   flag classes with too few examples.
#
# ARCHITECTURE:
#
# datasets/images/{train,val} + datasets/labels/{train,val}
#                      |
#             utils/check_dataset.py
#                      |
#        report: pairs / errors / class balance
#
# USAGE (from the repo root):
#   python utils/check_dataset.py
#   python utils/check_dataset.py --data configs/data.yaml
# ============================================================

import argparse
import sys
from collections import Counter
from pathlib import Path


# Image extensions Ultralytics accepts.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Warn when a class has fewer instances than this — heavily
# imbalanced classes train poorly.
MIN_INSTANCES_WARN = 50


def load_data_yaml(path: Path) -> dict:
    """
    Minimal reader for configs/data.yaml (path/train/val/names)
    without requiring PyYAML on the checking machine.
    """

    import re

    text = path.read_text(encoding="utf-8")
    data: dict = {"names": {}}

    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        # names entries look like "  0: drowning".
        m = re.match(r"\s+(\d+):\s*(\S+)", line)
        if m:
            data["names"][int(m.group(1))] = m.group(2)
            continue

        # top-level "key: value" entries.
        m = re.match(r"(\w+):\s*(.+)", line)
        if m and m.group(1) in ("path", "train", "val", "test"):
            data[m.group(1)] = m.group(2).strip()

    return data


def check_split(images_dir: Path, labels_dir: Path,
                names: dict) -> tuple[int, int, Counter]:
    """
    Check one split (train or val).

    Returns (pair_count, error_count, class_counter).
    """

    errors = 0
    class_counter: Counter = Counter()

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ) if images_dir.exists() else []

    labels = {
        p.stem: p for p in labels_dir.glob("*.txt")
    } if labels_dir.exists() else {}

    if not images:
        print(f"  [error] No images in {images_dir}")
        return 0, 1, class_counter

    # ---- Every image needs a label ------------------------
    pairs = 0
    for img in images:
        label = labels.pop(img.stem, None)

        if label is None:
            print(f"  [error] Missing label for image: {img.name}")
            errors += 1
            continue

        pairs += 1

        # ---- Validate each label line ----------------------
        for line_no, line in enumerate(
            label.read_text().splitlines(), start=1
        ):
            parts = line.split()
            if not parts:
                continue  # blank line is tolerated

            # Format: class cx cy w h  → exactly 5 numbers.
            if len(parts) != 5:
                print(f"  [error] {label.name}:{line_no} "
                      f"expected 5 values, got {len(parts)}")
                errors += 1
                continue

            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                print(f"  [error] {label.name}:{line_no} "
                      "non-numeric value")
                errors += 1
                continue

            # Class ID must exist in data.yaml.
            if cls not in names:
                print(f"  [error] {label.name}:{line_no} "
                      f"unknown class id {cls}")
                errors += 1
                continue

            # YOLO coordinates are normalized 0–1.
            if not all(0.0 <= v <= 1.0 for v in coords):
                print(f"  [error] {label.name}:{line_no} "
                      "coordinates not normalized (0-1)")
                errors += 1
                continue

            class_counter[names[cls]] += 1

    # ---- Labels without an image are orphans ---------------
    for stem in labels:
        print(f"  [warn ] Label without image: {stem}.txt")

    return pairs, errors, class_counter


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Validate the YOLO dataset before training"
    )
    parser.add_argument(
        "--data",
        default="configs/data.yaml",
        help="Dataset YAML (default: configs/data.yaml)"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    yaml_path = (root / args.data).resolve() \
        if not Path(args.data).is_absolute() else Path(args.data)

    if not yaml_path.exists():
        sys.exit(f"[error] Config not found: {yaml_path}")

    cfg = load_data_yaml(yaml_path)
    names = cfg.get("names", {})
    base = (yaml_path.parent.parent / cfg.get("path", "datasets")) \
        .resolve() if not Path(cfg.get("path", "")).is_absolute() \
        else Path(cfg["path"])

    print(f"[check] Dataset root : {base}")
    print(f"[check] Classes      : {names}\n")

    total_errors = 0
    total_counter: Counter = Counter()

    for split in ("train", "val"):
        rel = cfg.get(split)
        if rel is None:
            continue

        images_dir = base / rel
        # labels path mirrors images path (YOLO convention).
        labels_dir = Path(str(images_dir).replace("images", "labels"))

        print(f"[check] Split '{split}':")
        pairs, errors, counter = check_split(
            images_dir, labels_dir, names
        )
        print(f"  {pairs} valid image/label pairs, "
              f"{errors} errors\n")

        total_errors += errors
        total_counter.update(counter)

    # ---- Class balance report -------------------------------
    print("[check] Instances per class:")
    for cls_name in names.values():
        count = total_counter.get(cls_name, 0)
        flag = "  <-- LOW, add more data" \
            if count < MIN_INSTANCES_WARN else ""
        print(f"  {cls_name:<22} {count}{flag}")

    # Exit code lets server pipelines fail the job early.
    if total_errors:
        sys.exit(f"\n[check] FAILED with {total_errors} errors.")

    print("\n[check] Dataset looks good - ready to train.")


if __name__ == "__main__":
    main()
