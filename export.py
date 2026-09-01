# ============================================================
# FILE: export.py
#
# PROJECT: PoolGuard — Drowning Detection Model Export
#
# PURPOSE:
# Export a trained checkpoint (best.pt) to deployment formats
# via the Ultralytics API. Called by the MLOps pipeline's
# export_and_quantize.sh, which speaks the classic YOLOv5
# CLI dialect — so both flag styles are accepted here.
#
# USAGE:
#   python export.py --weights best.pt                    # onnx
#   python export.py --weights best.pt --include onnx torchscript
#   python export.py --weights best.pt --include engine --half
#   python export.py --weights best.pt --include tflite --int8
# ============================================================
 
import argparse
import sys
from pathlib import Path
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the PoolGuard YOLO model for deployment"
    )
 
    parser.add_argument(
        "--weights", "--model",
        dest="weights",
        default=None,
        help="Path to the trained checkpoint (.pt) to export"
    )
 
    parser.add_argument(
        "--include", "--format",
        dest="include",
        nargs="+",
        default=["onnx"],
        help="One or more export formats: onnx torchscript "
             "engine tflite openvino coreml ... (default: onnx)"
    )
 
    parser.add_argument(
        "--imgsz", "--img", "--img-size",
        dest="imgsz",
        type=int,
        default=640,
        help="Export image size (default: 640)"
    )
 
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 quantization (supported formats only)"
    )
 
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 quantization (tflite/engine/openvino)"
    )
 
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Dynamic input shapes (onnx/engine)"
    )
 
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify the ONNX graph (default: on)"
    )
 
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (default: ultralytics choice)"
    )
 
    parser.add_argument(
        "--device",
        default=None,
        help="'0' for GPU, 'cpu' to force CPU "
             "(default: auto; engine/TensorRT requires GPU)"
    )
 
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset YAML for INT8 calibration (optional)"
    )
 
    # Tolerate unknown flags from the pipeline instead of dying.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] Ignoring unrecognized arguments: {unknown}")
 
    if not args.weights:
        sys.exit("[error] --weights is required (path to best.pt)")
 
    weights = Path(args.weights).resolve()
    if not weights.exists():
        sys.exit(f"[error] Checkpoint not found: {weights}")
 
    from ultralytics import YOLO
 
    exported = []
    for fmt in args.include:
        fmt = fmt.strip().lower()
        print(f"[export] {weights.name} -> {fmt} "
              f"(imgsz={args.imgsz}, half={args.half}, "
              f"int8={args.int8})")
 
        model = YOLO(str(weights))
        kwargs = dict(
            format=fmt,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            dynamic=args.dynamic,
            simplify=args.simplify,
            device=args.device,
        )
        if args.opset is not None:
            kwargs["opset"] = args.opset
        if args.data is not None:
            kwargs["data"] = args.data
 
        out = model.export(**kwargs)
        exported.append(str(out))
        print(f"[export] wrote: {out}")
 
    print("\n[done] Exported artifacts:")
    for p in exported:
        print(f"  {p}")
 
 
if __name__ == "__main__":
    main()
 