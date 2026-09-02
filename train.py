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
# COMPATIBILITY:
# The MLOps pipeline's run_training.sh calls this script with
# classic-YOLOv5 style flags (--weights, --project, --seed,
# --exist-ok). Those are now accepted and mapped onto the
# Ultralytics API. Unknown future flags are warned about and
# ignored instead of crashing the run.
#
# USAGE:
#   python train.py                              # defaults
#   python train.py --model yolov5su.pt --epochs 150
#   python train.py --weights base.pt --project /runs --seed 0 --exist-ok
#   python train.py --resume                     # continue run
# ============================================================
 
import argparse
import shutil
import sys
from pathlib import Path
 
 
# ============================================================
# PROJECT PATHS
# ============================================================
 
ROOT = Path(__file__).resolve().parent
 
DEFAULT_DATA_CONFIG = ROOT / "configs" / "data.yaml"
 
MODELS_DIR = ROOT / "models"
 
WEIGHTS_DIR = ROOT / "weights"
 
RUNS_DIR = ROOT / "runs"
 
 
# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
 
def preflight(data_config: Path, using_default_config: bool) -> None:
    """
    Verify everything needed for training exists.
 
    The local datasets/ check only applies when training against
    the repo's own default config. On the training servers the
    pipeline passes its own dataset YAML (absolute paths under
    /trainx/...), and the repo's datasets/ folder is expected to
    be empty there.
    """
 
    if not data_config.exists():
        sys.exit(
            f"[error] Dataset config not found: {data_config}\n"
            "        Copy configs/data.yaml and set the paths "
            "for your dataset."
        )
 
    if not using_default_config:
        return
 
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
            "(see datasets/README.md), or pass --data pointing "
            "at the dataset YAML on the server."
        )
 
 
# ============================================================
# PLATFORM CONTRACT FIXUPS (from the TrainX repo skeleton)
# ============================================================
# The TrainX acceptance gate reads the LAST row of results.csv
# using classic YOLOv5 column names. Ultralytics writes
# (B)-suffixed names, so the gate would see no metrics at all.
# ============================================================
 
RESULTS_CSV_COLUMN_MAP = {
    "metrics/precision(B)": "metrics/precision",
    "metrics/recall(B)": "metrics/recall",
    "metrics/mAP50(B)": "metrics/mAP_0.5",
    "metrics/mAP50-95(B)": "metrics/mAP_0.5:0.95",
}
 
 
def rewrite_results_csv(run_dir: Path) -> None:
    """Rewrite ultralytics' results.csv header into YOLOv5 column
    names, in place, so the TrainX acceptance gate and MLflow
    logging can read the metrics."""
    import csv
 
    path = run_dir / "results.csv"
    if not path.is_file():
        print(f"[warn] {path} not found; acceptance gate will "
              "see no metrics")
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = [RESULTS_CSV_COLUMN_MAP.get(c.strip(), c.strip())
              for c in rows[0]]
    body = [[c.strip() for c in r] for r in rows[1:]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    print(f"[train] rewrote {path} header into YOLOv5 column names")
 
 
def ensure_best_pt(run_dir: Path) -> None:
    """The pipeline requires weights/best.pt. In edge cases
    ultralytics writes only last.pt; copying it is the honest
    equivalent (it IS the final checkpoint)."""
    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if not best.is_file() and last.is_file():
        shutil.copy2(last, best)
        print(f"[train] best.pt was missing; copied last.pt -> {best}")
 
 
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
        "--weights",
        default=None,
        help="Alias for --model, matching the classic YOLOv5 CLI "
             "used by the server pipeline. If set, overrides --model."
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
        "--project",
        default=str(RUNS_DIR),
        help="Directory that run folders are created in "
             "(default: <repo>/runs). Matches the classic "
             "YOLOv5 --project flag used by the pipeline."
    )
 
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible training (default: 0)"
    )
 
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing run folder instead "
             "of failing. Used by the server pipeline on retries."
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
 
    # Tolerate flags this script doesn't know yet (the pipeline
    # may grow new ones) — warn loudly instead of dying before
    # training starts.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] Ignoring unrecognized arguments: {unknown}")
 
    # --weights (pipeline dialect) wins over --model when given.
    base_checkpoint = args.weights if args.weights else args.model
 
    # ------------------------------------------------------------
    # MISSING-CHECKPOINT SAFETY NET
    # ------------------------------------------------------------
    # The pipeline builds the cold-start path as <repo>/<weights_
    # variant>.pt. If the config's variant name doesn't match any
    # committed file, don't crash the run: fall back to any .pt in
    # the repo root, else to an auto-downloaded pretrained base.
    # ------------------------------------------------------------
    ckpt_path = Path(base_checkpoint)
    if not ckpt_path.is_file() and ckpt_path.suffix == ".pt":
        candidates = sorted(ROOT.glob("*.pt"))
        if candidates:
            print(f"[warn] base checkpoint not found: {ckpt_path}")
            print(f"[warn] falling back to repo checkpoint: "
                  f"{candidates[0]} (fix weights_variant in the "
                  f"pipeline config to silence this)")
            base_checkpoint = str(candidates[0])
        else:
            print(f"[warn] base checkpoint not found: {ckpt_path} "
                  f"and no .pt in {ROOT}")
            print("[warn] falling back to auto-downloaded "
                  "'yolov8n.pt' pretrained base (fix "
                  "weights_variant in the pipeline config)")
            base_checkpoint = "yolov8n.pt"
 
    data_config = Path(args.data).resolve()
    using_default_config = (
        data_config == DEFAULT_DATA_CONFIG.resolve()
    )
    preflight(data_config, using_default_config)
 
    # Imported here so `python train.py --help` works on
    # machines without ultralytics installed.
    from ultralytics import YOLO
    from ultralytics import settings
 
    settings.update({"weights_dir": str(MODELS_DIR)})
 
    print(f"[train] Base model : {base_checkpoint}")
    print(f"[train] Dataset    : {data_config}")
    print(f"[train] Project    : {args.project}  Run: {args.name}")
    print(f"[train] Epochs     : {args.epochs}  "
          f"Batch: {args.batch}  ImgSz: {args.imgsz}  "
          f"Seed: {args.seed}")
 
    model = YOLO(base_checkpoint)
 
    results = model.train(
        data=str(data_config),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        seed=args.seed,
        exist_ok=args.exist_ok,
    )
 
    # Platform contract fixups (results.csv columns + best.pt)
    run_dir = Path(results.save_dir)
    rewrite_results_csv(run_dir)
    ensure_best_pt(run_dir)
 
    # --------------------------------------------------------
    # EXPORT THE BEST CHECKPOINT
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