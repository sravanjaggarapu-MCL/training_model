# ============================================================
# FILE: detect.py
#
# PROJECT: PoolGuard — Drowning Detection Inference
#
# PURPOSE:
# Run inference with a trained checkpoint on images, folders,
# or video, saving annotated outputs.
#
# EXTRA BOX REDUCTION:
# 1. Higher confidence threshold removes weak detections.
# 2. Duplicate suppression removes overlapping same-class boxes.
# 3. Minimum box area removes extremely small false detections.
# ============================================================
 
import argparse
import sys
from pathlib import Path
 
 
# ------------------------------------------------------------
# SAME-CLASS DUPLICATE SUPPRESSION
# ------------------------------------------------------------
 
def _iou_and_ios(a, b):
    """
    Return:
        IoU = Intersection over Union
        IoS = Intersection over Smaller box area
    """
 
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
 
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
 
    inter = iw * ih
 
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
 
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
 
    iou = inter / union if union > 0 else 0.0
    ios = inter / smaller if smaller > 0 else 0.0
 
    return iou, ios
 
 
def dedupe_indices(
    boxes_xyxy,
    confs,
    classes,
    merge_iou=0.30,
    merge_ios=0.60
):
    """
    Remove duplicate boxes of the same class.
 
    The highest-confidence box is kept.
 
    A lower-confidence box is removed when it overlaps
    sufficiently with a higher-confidence box.
    """
 
    # Highest confidence first
    order = sorted(
        range(len(confs)),
        key=lambda i: confs[i],
        reverse=True
    )
 
    keep = []
 
    for i in order:
 
        duplicate = False
 
        for j in keep:
 
            # Only compare boxes belonging to the same class
            if classes[i] != classes[j]:
                continue
 
            iou, ios = _iou_and_ios(
                boxes_xyxy[i],
                boxes_xyxy[j]
            )
 
            # Remove lower-confidence duplicate
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
 
    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
 
    parser.add_argument(
        "--weights", "--model",
        dest="weights",
        default=None,
        help="Path to the trained checkpoint (.pt)"
    )
 
    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------
 
    parser.add_argument(
        "--source",
        default=None,
        help="Image, folder, glob, or video to run on"
    )
 
    # --------------------------------------------------------
    # CONFIDENCE THRESHOLD
    # --------------------------------------------------------
 
    parser.add_argument(
        "--conf-thres", "--conf",
        dest="conf",
        type=float,
        default=0.40,
        help="Confidence threshold (default: 0.40)"
    )
 
    # --------------------------------------------------------
    # NMS IOU
    # --------------------------------------------------------
 
    parser.add_argument(
        "--iou-thres", "--iou",
        dest="iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)"
    )
 
    # --------------------------------------------------------
    # DUPLICATE SUPPRESSION
    # --------------------------------------------------------
 
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.30,
        help=(
            "Post-NMS same-class suppression: "
            "remove lower-confidence box when IoU "
            "exceeds this (default: 0.30)"
        )
    )
 
    parser.add_argument(
        "--merge-ios",
        type=float,
        default=0.60,
        help=(
            "Post-NMS same-class suppression: "
            "remove lower-confidence box when "
            "intersection/smaller-box-area exceeds "
            "this (default: 0.60)"
        )
    )
 
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable duplicate suppression"
    )
 
    # --------------------------------------------------------
    # MINIMUM BOX AREA
    # --------------------------------------------------------
 
    parser.add_argument(
        "--min-area",
        type=float,
        default=500.0,
        help=(
            "Minimum bounding-box area in pixels. "
            "Very small boxes are removed. "
            "Set to 0 to disable."
        )
    )
 
    # --------------------------------------------------------
    # NMS OPTIONS
    # --------------------------------------------------------
 
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help=(
            "Class-agnostic NMS inside the model"
        )
    )
 
    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------
 
    parser.add_argument(
        "--imgsz", "--img", "--img-size",
        dest="imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
 
    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
 
    parser.add_argument(
        "--device",
        default=None,
        help="'0' for GPU, 'cpu' to force CPU (default: auto)"
    )
 
    # --------------------------------------------------------
    # MAX DETECTIONS
    # --------------------------------------------------------
 
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum detections per image (default: 100)"
    )
 
    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
 
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
 
    # --------------------------------------------------------
    # PARSE ARGUMENTS
    # --------------------------------------------------------
 
    args, unknown = parser.parse_known_args()
 
    if unknown:
        print(
            f"[warn] Ignoring unrecognized arguments: {unknown}"
        )
 
    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------
 
    if not args.weights:
        sys.exit(
            "[error] --weights is required "
            "(path to best.pt)"
        )
 
    if not args.source:
        sys.exit(
            "[error] --source is required "
            "(image/folder/video)"
        )
 
    if args.conf < 0.0 or args.conf > 1.0:
        sys.exit(
            "[error] Confidence must be between 0 and 1"
        )
 
    if args.min_area < 0:
        sys.exit(
            "[error] --min-area cannot be negative"
        )
 
    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------
 
    weights = Path(args.weights).resolve()
 
    if not weights.exists():
        sys.exit(
            f"[error] Checkpoint not found: {weights}"
        )
 
    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------
 
    from ultralytics import YOLO
 
    model = YOLO(str(weights))
 
    print(f"[detect] Model     : {weights}")
    print(f"[detect] Source    : {args.source}")
    print(f"[detect] Conf      : {args.conf}")
    print(f"[detect] IoU       : {args.iou}")
    print(f"[detect] ImgSz     : {args.imgsz}")
    print(f"[detect] Min Area  : {args.min_area}")
    print(
        f"[detect] Dedupe    : "
        f"{'off' if args.no_dedupe else f'iou>{args.merge_iou} or ios>{args.merge_ios}'}"
    )
 
    # --------------------------------------------------------
    # VIDEO CHECK
    # --------------------------------------------------------
 
    video_exts = {
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm"
    }
 
    is_video = (
        Path(str(args.source)).suffix.lower()
        in video_exts
    )
 
    # --------------------------------------------------------
    # IMAGE / FOLDER PROCESSING
    # --------------------------------------------------------
 
    # We process images ourselves so that we can:
    #
    # 1. Run YOLO
    # 2. Remove low-confidence detections
    # 3. Remove very small detections
    # 4. Remove duplicate boxes
    # 5. Draw only the remaining boxes
    #
    # For video, native YOLO saving is retained.
 
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
            print(
                f"\n[done] Annotated outputs: "
                f"{results[0].save_dir}"
            )
 
        return
 
    # --------------------------------------------------------
    # IMAGE / FOLDER INFERENCE
    # --------------------------------------------------------
 
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
 
    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------
 
    out_dir = Path(args.project) / args.name
 
    if out_dir.exists() and not args.exist_ok:
 
        n = 2
 
        while (
            Path(args.project) /
            f"{args.name}{n}"
        ).exists():
            n += 1
 
        out_dir = (
            Path(args.project) /
            f"{args.name}{n}"
        )
 
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )
 
    labels_dir = out_dir / "labels"
 
    if args.save_txt:
        labels_dir.mkdir(
            parents=True,
            exist_ok=True
        )
 
    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------
 
    total_raw = 0
    total_area_removed = 0
    total_duplicates_removed = 0
    total_kept = 0
 
    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------
 
    for r in results:
 
        n_raw = len(r.boxes)
 
        total_raw += n_raw
 
        if n_raw == 0:
 
            print(
                f"[detect] {r.path}: "
                f"0 detections"
            )
 
            annotated = r.plot()
 
            stem = Path(r.path).stem
 
            cv2.imwrite(
                str(out_dir / f"{stem}.jpg"),
                annotated
            )
 
            continue
 
        # ----------------------------------------------------
        # GET DETECTION DATA
        # ----------------------------------------------------
 
        xyxy = r.boxes.xyxy.cpu().tolist()
 
        confs = r.boxes.conf.cpu().tolist()
 
        clses = [
            int(c)
            for c in r.boxes.cls.cpu().tolist()
        ]
 
        # ----------------------------------------------------
        # REMOVE VERY SMALL BOXES
        # ----------------------------------------------------
 
        area_keep = []
 
        for i, box in enumerate(xyxy):
 
            width = max(
                0.0,
                box[2] - box[0]
            )
 
            height = max(
                0.0,
                box[3] - box[1]
            )
 
            area = width * height
 
            if area >= args.min_area:
                area_keep.append(i)
 
        area_removed = n_raw - len(area_keep)
 
        total_area_removed += area_removed
 
        # ----------------------------------------------------
        # APPLY AREA FILTER
        # ----------------------------------------------------
 
        xyxy_filtered = [
            xyxy[i]
            for i in area_keep
        ]
 
        confs_filtered = [
            confs[i]
            for i in area_keep
        ]
 
        clses_filtered = [
            clses[i]
            for i in area_keep
        ]
 
        # ----------------------------------------------------
        # DUPLICATE SUPPRESSION
        # ----------------------------------------------------
 
        if (
            not args.no_dedupe
            and len(confs_filtered) > 1
        ):
 
            keep_local = dedupe_indices(
                xyxy_filtered,
                confs_filtered,
                clses_filtered,
                args.merge_iou,
                args.merge_ios
            )
 
        else:
 
            keep_local = list(
                range(len(confs_filtered))
            )
 
        duplicates_removed = (
            len(confs_filtered) -
            len(keep_local)
        )
 
        total_duplicates_removed += (
            duplicates_removed
        )
 
        # ----------------------------------------------------
        # CONVERT BACK TO ORIGINAL BOX INDICES
        # ----------------------------------------------------
 
        keep = [
            area_keep[i]
            for i in keep_local
        ]
 
        # ----------------------------------------------------
        # FILTER RESULT BOXES
        # ----------------------------------------------------
 
        try:
 
            r.boxes = r.boxes[keep]
 
        except Exception as e:
 
            print(
                f"[warn] Filtering skipped for "
                f"{r.path}: {e}"
            )
 
        total_kept += len(r.boxes)
 
        # ----------------------------------------------------
        # DRAW FILTERED BOXES
        # ----------------------------------------------------
 
        stem = Path(r.path).stem
 
        annotated = r.plot()
 
        cv2.imwrite(
            str(out_dir / f"{stem}.jpg"),
            annotated
        )
 
        # ----------------------------------------------------
        # SAVE TXT LABELS
        # ----------------------------------------------------
 
        if args.save_txt:
 
            lines = []
 
            for b in r.boxes:
 
                cls_id = int(
                    b.cls.item()
                )
 
                x, y, w, h = (
                    b.xywhn[0].tolist()
                )
 
                line = (
                    f"{cls_id} "
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}"
                )
 
                if args.save_conf:
 
                    line += (
                        f" {b.conf.item():.6f}"
                    )
 
                lines.append(line)
 
            (
                labels_dir /
                f"{stem}.txt"
            ).write_text(
                "\n".join(lines) +
                ("\n" if lines else "")
            )
 
    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------
 
    print(
        f"\n[detect] Raw detections        : "
        f"{total_raw}"
    )
 
    print(
        f"[detect] Small boxes removed   : "
        f"{total_area_removed}"
    )
 
    print(
        f"[detect] Duplicate boxes removed: "
        f"{total_duplicates_removed}"
    )
 
    print(
        f"[detect] Final detections       : "
        f"{total_kept}"
    )
 
    print(
        f"[done] Annotated outputs: "
        f"{out_dir.resolve()}"
    )
 
 
# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
 
if __name__ == "__main__":
    main()