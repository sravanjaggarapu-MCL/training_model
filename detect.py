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
# DUPLICATE SUPPRESSION:
# Standard NMS only removes boxes whose IoU exceeds the NMS
# threshold. A loose box around a tight box on the same person
# can have IoU below that bar and survive (seen in validation
# samples). After prediction we run a second same-class pass
# that drops a lower-confidence box when either:
#   - IoU with a kept box       > --merge-iou   (default 0.35)
#   - overlap / smaller box area > --merge-ios  (default 0.65)
# Tune with --merge-iou / --merge-ios; disable with
# --no-dedupe to see raw model output.
#
# USAGE:
#   python detect.py --weights best.pt --source images/
#   python detect.py --model best.pt --source video.mp4 --conf 0.4
# ============================================================
 
import argparse
import sys
from pathlib import Path
 
 
# ------------------------------------------------------------
# SAME-CLASS DUPLICATE SUPPRESSION (pure python, testable)
# ------------------------------------------------------------
 
def _iou_and_ios(a, b):
    """Return (IoU, intersection-over-smaller) for two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    iou = inter / union if union > 0 else 0.0
    ios = inter / smaller if smaller > 0 else 0.0
    return iou, ios
 
 
def dedupe_indices(boxes_xyxy, confs, classes,
                   merge_iou=0.35, merge_ios=0.65):
    """
    Greedy same-class duplicate suppression.
 
    boxes_xyxy: list of [x1,y1,x2,y2]; confs: list of float;
    classes: list of int. Returns indices to KEEP, ordered by
    descending confidence.
    """
    order = sorted(range(len(confs)),
                   key=lambda i: confs[i], reverse=True)
    keep = []
    for i in order:
        duplicate = False
        for j in keep:
            if classes[i] != classes[j]:
                continue
            iou, ios = _iou_and_ios(boxes_xyxy[i], boxes_xyxy[j])
            if iou > merge_iou or ios > merge_ios:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return keep
 
 
# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
 
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
        "--merge-iou",
        type=float,
        default=0.35,
        help="Post-NMS same-class suppression: drop the lower-"
             "confidence box when IoU exceeds this (default: 0.35)"
    )
 
    parser.add_argument(
        "--merge-ios",
        type=float,
        default=0.65,
        help="Post-NMS same-class suppression: drop the lower-"
             "confidence box when intersection/smaller-box-area "
             "exceeds this (default: 0.65)"
    )
 
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable the post-NMS duplicate suppression pass"
    )
 
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Class-agnostic NMS inside the model (also merges "
             "overlapping boxes of DIFFERENT classes)"
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
          f"ImgSz: {args.imgsz}  Dedupe: "
          f"{'off' if args.no_dedupe else f'iou>{args.merge_iou} or ios>{args.merge_ios}'}")
 
    # Videos: stream through the built-in writer (dedupe pass is
    # image-oriented; native save keeps video output working).
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    is_video = Path(str(args.source)).suffix.lower() in video_exts
 
    if is_video or args.no_dedupe:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            agnostic_nms=args.agnostic_nms,
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
        return
 
    # Images/folders: predict without saving, dedupe, then draw
    # and save the filtered detections ourselves.
    import cv2
 
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        save=False,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )
 
    out_dir = Path(args.project) / args.name
    if out_dir.exists() and not args.exist_ok:
        n = 2
        while (Path(args.project) / f"{args.name}{n}").exists():
            n += 1
        out_dir = Path(args.project) / f"{args.name}{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    if args.save_txt:
        labels_dir.mkdir(exist_ok=True)
 
    total_raw, total_kept = 0, 0
    for r in results:
        n_raw = len(r.boxes)
        total_raw += n_raw
 
        if n_raw > 1:
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clses = [int(c) for c in r.boxes.cls.cpu().tolist()]
            keep = dedupe_indices(xyxy, confs, clses,
                                  args.merge_iou, args.merge_ios)
            if len(keep) < n_raw:
                try:
                    r.boxes = r.boxes[keep]
                except Exception as e:  # ultralytics API drift
                    print(f"[warn] dedupe skipped for "
                          f"{r.path}: {e}")
        total_kept += len(r.boxes)
 
        stem = Path(r.path).stem
        annotated = r.plot()
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), annotated)
 
        if args.save_txt:
            lines = []
            for b in r.boxes:
                cls_id = int(b.cls.item())
                x, y, w, h = b.xywhn[0].tolist()
                line = f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                if args.save_conf:
                    line += f" {b.conf.item():.6f}"
                lines.append(line)
            (labels_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
 
    removed = total_raw - total_kept
    print(f"\n[detect] {total_kept} detections kept "
          f"({removed} duplicate box(es) suppressed)")
    print(f"[done] Annotated outputs: {out_dir.resolve()}")
 
 
if __name__ == "__main__":
    main()