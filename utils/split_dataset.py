# ============================================================
# FILE: utils/split_dataset.py
#
# PROJECT: PoolGuard — Drowning Detection Model Training
#
# PURPOSE:
# Split a flat folder of labeled images into the train/val
# layout that configs/data.yaml and Ultralytics expect.
#
# RESPONSIBILITIES:
# - Take a source folder containing images + YOLO .txt labels
#   mixed together (the usual export format from labeling
#   tools like Roboflow/CVAT/LabelImg).
# - Shuffle deterministically (fixed seed → reproducible).
# - Copy pairs into datasets/images/{train,val} and
#   datasets/labels/{train,val}.
#
# ARCHITECTURE:
#
# raw_export/                    datasets/
#   img001.jpg                     images/train/...
#   img001.txt        ──►          images/val/...
#   img002.jpg                     labels/train/...
#   ...                            labels/val/...
#
# USAGE (from the repo root):
#   python utils/split_dataset.py --source /path/to/raw_export
#   python utils/split_dataset.py --source raw/ --val 0.2
# ============================================================

import argparse
import random
import shutil
import sys
from pathlib import Path


# Image extensions accepted.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed seed so re-running produces the same split —
# essential for comparing two training runs fairly.
SEED = 42


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Split labeled images into train/val sets"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Folder containing images and matching .txt labels"
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.2,
        help="Validation fraction, 0-1 (default: 0.2 = 20%%)"
    )

    parser.add_argument(
        "--dest",
        default="datasets",
        help="Destination dataset root (default: datasets/)"
    )

    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"[error] Source folder not found: {source}")

    root = Path(__file__).resolve().parent.parent
    dest = (root / args.dest) if not Path(args.dest).is_absolute() \
        else Path(args.dest)

    # ---- Collect image/label pairs --------------------------
    pairs = []
    for img in sorted(source.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue

        label = img.with_suffix(".txt")
        if not label.exists():
            print(f"[warn ] Skipping (no label): {img.name}")
            continue

        pairs.append((img, label))

    if not pairs:
        sys.exit("[error] No image+label pairs found in source.")

    # ---- Deterministic shuffle and split ---------------------
    random.Random(SEED).shuffle(pairs)

    val_count = max(1, int(len(pairs) * args.val))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    # ---- Copy into the YOLO layout ---------------------------
    for split, split_pairs in (
        ("train", train_pairs),
        ("val", val_pairs),
    ):
        img_dir = dest / "images" / split
        lbl_dir = dest / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, label in split_pairs:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(label, lbl_dir / label.name)

    print(f"[split] Train: {len(train_pairs)} pairs")
    print(f"[split] Val  : {len(val_pairs)} pairs")
    print(f"[split] Written under: {dest}")
    print("[split] Next: python utils/check_dataset.py")


if __name__ == "__main__":
    main()
