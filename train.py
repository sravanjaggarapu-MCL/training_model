# ============================================================
# FILE: train.py
#
# PROJECT: PoolGuard — Drowning Detection Model Training
#
# PURPOSE:
# Train a YOLO model (YOLOv5 / YOLOv8 family via the
# Ultralytics API) on the pool drowning dataset, on a local
# machine or on the company training servers.
#
# RESPONSIBILITIES:
# - Parse training options from the command line.
# - Verify the dataset and config exist before training
#   (fail fast with a clear message, not a deep stack trace).
# - Run Ultralytics training with the chosen base model.
# - Copy the best checkpoint to weights/ with a clear name,
#   ready to drop into PoolGuard's detector.py.
#
# ARCHITECTURE:
#
# datasets/  ──►  configs/data.yaml  ──►  train.py
#                                            |
#                                     Ultralytics YOLO
#                                            |
#                              runs/ (full training logs)
#                                            |
#                              weights/<name>_best.pt
#                                            |
#                              PoolGuard detector.py (--model)
#
# USAGE:
#   python train.py                              # defaults
#   python train.py --model yolov5su.pt --epochs 150
#   python train.py --model yolov8s.pt --batch 32 --device 0
#   python train.py --resume                     # continue run
#
# NOTES:
# - "yolov5su.pt" is the Ultralytics-maintained YOLOv5-small;
#   "yolov8n.pt"/"yolov8s.pt" are newer and usually better
#   for the same speed. All train with this same script.
# - Base checkpoints download automatically on first use and
#   are cached in models/ (see MODELS_DIR below).
# ============================================================

import argparse
import shutil
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Repo root = the folder containing this file.
ROOT = Path(__file__).resolve().parent

# Dataset config consumed by Ultralytics.
DEFAULT_DATA_CONFIG = ROOT / "configs" / "data.yaml"

# Downloaded base checkpoints (yolov5su.pt, yolov8n.pt, ...).
MODELS_DIR = ROOT / "models"

# Final trained weights, named per run.
WEIGHTS_DIR = ROOT / "weights"

# Ultralytics run logs/checkpoints (gitignored).
RUNS_DIR = ROOT / "runs"


# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

def preflight(data_config: Path) -> None:
    """
    Verify everything needed for training exists.

    Failing here with a clear message saves a wasted server
    job that dies 10 minutes in with a cryptic error.
    """

    # The dataset YAML must exist.
    if not data_config.exists():
        sys.exit(
            f"[error] Dataset config not found: {data_config}\n"
            "        Copy configs/data.yaml and set the paths "
            "for your dataset."
        )

    # The datasets folder must contain actual data — the
    # placeholder files that live in git don't count.
    placeholders = {".gitkeep", "README.md"}
    dataset_dir = ROOT / "datasets"
    contents = [
        p for p in dataset_dir.iterdir()
        if p.name not in placeholders
    ] if dataset_dir.exists() else []

    if not contents:
        sys.exit(
            "[error] datasets/ is empty.\n"
            "        Place your dataset there in YOLO format "
            "(see datasets/README.md), or point configs/"
            "data.yaml at the dataset location on the server."
        )


# ============================================================
# TRAINING
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train the PoolGuard drowning-detection YOLO model"
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base checkpoint: yolov5su.pt, yolov5mu.pt, "
             "yolov8n.pt, yolov8s.pt, ... (default: yolov8n.pt)"
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to the dataset YAML (default: configs/data.yaml)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; lower it if the GPU runs out of "
             "memory (default: 16)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640)"
    )

    parser.add_argument(
        "--device",
        default=None,
        help="'0' for first GPU, '0,1' for two GPUs, 'cpu' to "
             "force CPU (default: auto-detect)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers (default: 8)"
    )

    parser.add_argument(
        "--name",
        default="poolguard",
        help="Run name; also names the exported weights "
             "(default: poolguard)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping: stop if no val improvement for "
             "N epochs (default: 30)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last interrupted run"
    )

    args = parser.parse_args()

    data_config = Path(args.data).resolve()
    preflight(data_config)

    # Imported here so `python train.py --help` works on
    # machines without ultralytics installed.
    from ultralytics import YOLO
    from ultralytics import settings

    # Keep downloaded base checkpoints inside the repo's
    # models/ folder instead of a hidden cache, so server
    # jobs are self-contained.
    settings.update({"weights_dir": str(MODELS_DIR)})

    print(f"[train] Base model : {args.model}")
    print(f"[train] Dataset    : {data_config}")
    print(f"[train] Epochs     : {args.epochs}  "
          f"Batch: {args.batch}  ImgSz: {args.imgsz}")

    # Load the base checkpoint (downloads on first use).
    model = YOLO(args.model)

    # Run training. Ultralytics handles augmentation,
    # LR schedule, checkpoints, and validation per epoch.
    results = model.train(
        data=str(data_config),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        exist_ok=False,      # never overwrite a previous run
    )

    # --------------------------------------------------------
    # EXPORT THE BEST CHECKPOINT
    # --------------------------------------------------------
    # Ultralytics saves best.pt inside the run folder; copy it
    # to weights/ with a stable, descriptive name so the
    # PoolGuard detector can reference it directly.
    # --------------------------------------------------------

    best = Path(results.save_dir) / "weights" / "best.pt"

    if best.exists():
        WEIGHTS_DIR.mkdir(exist_ok=True)
        target = WEIGHTS_DIR / f"{args.name}_best.pt"
        shutil.copy2(best, target)

        print(f"\n[done] Best weights: {target}")
        print("[done] Use in PoolGuard: "
              f"python detector.py --model {target.name}")
    else:
        print("\n[warn] best.pt not found (training may have "
              "been interrupted). Check the runs/ folder.")


if __name__ == "__main__":
    main()
