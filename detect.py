# ============================================================
# FILE: detect.py
#
# PROJECT: PoolGuard — Drowning Detection Inference
#
# PURPOSE:
# Run inference with a trained checkpoint on images, folders,
# or video, saving annotated outputs. Called by the MLOps
# pipeline's eval_visual_sample.py (classic YOLOv5 CLI
# dialect) and usable directly by PoolGuard.
#
# USAGE:
#   python detect.py --weights best.pt --source images/
#   python detect.py --model best.pt --source video.mp4 --conf 0.4
# ============================================================
 
import argparse
import sys
from pathlib import Path
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PoolGuard YOLO inference"
    )
 
    parser.add_argument(
        "--weights", "--model",
        dest="weights",
        default=None,
        help="Path to the trained checkpoint (.pt)"
    )
 
    parser.add_argument(
        "--source",
        default=None,
        help="Image, folder, glob, or video to run on"
    )
 
    parser.add_argument(
        "--conf-thres", "--conf",
        dest="conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
 
    parser.add_argument(
        "--iou-thres", "--iou",
        dest="iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)"
    )
 
    parser.add_argument(
        "--imgsz", "--img", "--img-size",
        dest="imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
 
    parser.add_argument(
        "--device",
        default=None,
        help="'0' for GPU, 'cpu' to force CPU (default: auto)"
    )
 
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per image (default: 300)"
    )
 
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Output parent directory (default: runs/detect)"
    )
 
    parser.add_argument(
        "--name",
        default="predict",
        help="Output run folder name (default: predict)"
    )
 
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also save YOLO-format label txt files"
    )
 
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidences in saved txt labels"
    )
 
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing output folder"
    )
 
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] Ignoring unrecognized arguments: {unknown}")
 
    if not args.weights:
        sys.exit("[error] --weights is required (path to best.pt)")
    if not args.source:
        sys.exit("[error] --source is required (image/folder/video)")
 
    weights = Path(args.weights).resolve()
    if not weights.exists():
        sys.exit(f"[error] Checkpoint not found: {weights}")
 
    from ultralytics import YOLO
 
    model = YOLO(str(weights))
 
    print(f"[detect] Model  : {weights}")
    print(f"[detect] Source : {args.source}")
    print(f"[detect] Conf   : {args.conf}  IoU: {args.iou}  "
          f"ImgSz: {args.imgsz}")
 
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )
 
    if results:
        print(f"\n[done] Annotated outputs: {results[0].save_dir}")
 
 
if __name__ == "__main__":
    main()
 