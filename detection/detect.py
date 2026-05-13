"""Run detection inference and output COCO-format predictions JSON.

Supports three inference methods:
  - yolo:  Standard YOLO inference (single-pass, 640x640 resize)
  - rfdetr: RF-DETR inference
  - sahi:  Sliced inference via SAHI (better for small objects)

Usage:
    # Standard YOLO inference
    conda run -n obj-det python detection/detect.py \
        --weights weights/best.pt \
        --gt-json data/splits/split_v7/coco/test/_annotations.coco.json \
        --method yolo \
        --out results/detections/yolo_baseline.json

    # SAHI sliced inference
    conda run -n obj-det python detection/detect.py \
        --weights weights/best.pt \
        --gt-json data/splits/split_v7/coco/test/_annotations.coco.json \
        --method sahi \
        --slice-size 640 --overlap-ratio 0.2 \
        --out results/detections/sahi_640.json

    # With cross-modal NMS
    conda run -n obj-det python detection/detect.py \
        --weights weights/best.pt \
        --gt-json data/splits/split_v7/coco/test/_annotations.coco.json \
        --method yolo \
        --enable-nms --nms-iou-thresh 0.5 \
        --out results/detections/yolo_nms.json

Output:
    COCO-format predictions JSON:
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def run_yolo_inference(weights_path, test_images_dir, conf=0.001, iou=0.6):
    """Run YOLO inference on a directory of images. Returns (predictions, timings_ms)."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model.predict(
        source=test_images_dir,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False,
        stream=True,
    )

    predictions = []
    timings_ms = []
    for r in results:
        img_path = str(Path(r.path).resolve())
        t0 = time.perf_counter()
        # Ultralytics already ran inference; measure post-processing + extraction
        # For accurate per-image timing, we use the speed dict from results
        if hasattr(r, "speed") and r.speed:
            timings_ms.append(sum(r.speed.values()))
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append({
                "_img_path": img_path,
                "category_id": int(box.cls[0]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box.conf[0]),
            })
    return predictions, timings_ms


def run_rfdetr_inference(weights_path, test_coco_json, conf=0.3):
    """Run RF-DETR inference on test set. Returns (predictions, timings_ms)."""
    from rfdetr import RFDETRBase, RFDETRLarge
    from PIL import Image as PILImage

    model_class = RFDETRLarge if "large" in str(weights_path).lower() else RFDETRBase
    model = model_class(pretrain_weights=weights_path)

    coco_gt = COCO(test_coco_json)
    predictions = []
    timings_ms = []

    total = len(coco_gt.dataset["images"])
    for idx, img_info in enumerate(coco_gt.dataset["images"], 1):
        img_path = img_info["file_name"]
        img = PILImage.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        detections = model.predict(img, threshold=conf)
        timings_ms.append((time.perf_counter() - t0) * 1000)

        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                predictions.append({
                    "image_id": img_info["id"],
                    "category_id": int(detections.class_id[i]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(detections.confidence[i]),
                })

        if idx % 100 == 0 or idx == total:
            print(f"  RF-DETR inference: {idx}/{total} images")

    return predictions, timings_ms


def run_sahi_inference(weights_path, test_images_dir, conf=0.001, slice_size=640, overlap_ratio=0.2, model_type="ultralytics"):
    """Run SAHI sliced inference on a directory of images. Returns (predictions, timings_ms).

    model_type: SAHI detector backend — "ultralytics" for YOLO, "huggingface" for DETR-style models,
                or any other type supported by sahi.AutoDetectionModel.
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=weights_path,
        confidence_threshold=conf,
    )

    image_dir = Path(test_images_dir)
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    predictions = []
    timings_ms = []
    total = len(image_files)
    for idx, img_path in enumerate(image_files, 1):
        t0 = time.perf_counter()
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )
        timings_ms.append((time.perf_counter() - t0) * 1000)

        for pred in result.object_prediction_list:
            bbox = pred.bbox
            predictions.append({
                "_img_path": str(img_path.resolve()),
                "category_id": pred.category.id,
                "bbox": [bbox.minx, bbox.miny, bbox.maxx - bbox.minx, bbox.maxy - bbox.miny],
                "score": pred.score.value,
            })

        if idx % 100 == 0 or idx == total:
            print(f"  SAHI inference: {idx}/{total} images")

    return predictions, timings_ms


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def map_paths_to_image_ids(predictions, coco_gt):
    """Convert _img_path-based predictions to image_id-based COCO format."""
    path_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
    mapped = []
    for p in predictions:
        img_id = path_to_id.get(p["_img_path"])
        if img_id is None:
            continue
        mapped.append({
            "image_id": img_id,
            "category_id": p["category_id"],
            "bbox": [round(float(v), 2) for v in p["bbox"]],
            "score": float(p["score"]),
        })
    return mapped


def apply_interclass_nms(predictions, coco_json_path, iou_thresh=0.5):
    """Apply cross-modal NMS per image on COCO-format predictions."""
    from cross_modal_nms import cross_modal_nms

    coco_gt = COCO(coco_json_path)
    name_to_ids = {}
    for cat in coco_gt.dataset["categories"]:
        base = cat["name"].replace("-rgb", "").replace("-thermal", "")
        name_to_ids.setdefault(base, set()).add(cat["id"])
    class_groups = {base: ids for base, ids in name_to_ids.items() if len(ids) > 1}

    if not class_groups:
        print("  Interclass NMS: no cross-modal groups found, skipping.")
        return predictions

    print(f"  Interclass NMS (iou_thresh={iou_thresh}): {class_groups}")

    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image_id"], []).append(p)

    kept = []
    total_before = len(predictions)
    for img_id, preds in by_image.items():
        dets = np.array([
            [p["bbox"][0], p["bbox"][1],
             p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3],
             p["score"], p["category_id"]]
            for p in preds
        ], dtype=np.float32)

        filtered = cross_modal_nms(dets, class_groups, iou_thresh)

        for d in filtered:
            x1, y1, x2, y2 = d[:4]
            kept.append({
                "image_id": img_id,
                "category_id": int(d[5]),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": float(d[4]),
            })

    print(f"  Interclass NMS: {total_before} -> {len(kept)} predictions ({total_before - len(kept)} suppressed)")
    return kept


# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------

def compute_timing_stats(timings_ms):
    """Compute latency statistics from per-image timings."""
    if not timings_ms:
        return None
    arr = np.array(timings_ms)
    # Skip first image (model warmup: CUDA kernel compilation, memory allocation)
    if len(arr) > 1:
        arr = arr[1:]
    return {
        "n_images": len(arr),
        "mean_ms": round(float(np.mean(arr)), 1),
        "median_ms": round(float(np.median(arr)), 1),
        "std_ms": round(float(np.std(arr)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "fps": round(float(1000.0 / np.mean(arr)), 1),
    }


def print_timing_report(stats, method):
    """Print a formatted timing summary."""
    if stats is None:
        return
    print(f"\n{'=' * 50}")
    print(f"  Latency Report — {method}")
    print(f"{'=' * 50}")
    print(f"  Images:     {stats['n_images']}")
    print(f"  Mean:       {stats['mean_ms']:.1f} ms")
    print(f"  Median:     {stats['median_ms']:.1f} ms")
    print(f"  Std:        {stats['std_ms']:.1f} ms")
    print(f"  Min/Max:    {stats['min_ms']:.1f} / {stats['max_ms']:.1f} ms")
    print(f"  P95:        {stats['p95_ms']:.1f} ms")
    print(f"  Throughput: {stats['fps']:.1f} FPS")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run detection inference and output COCO-format predictions.")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--gt-json", required=True, help="COCO GT JSON (for image ID mapping)")
    parser.add_argument("--method", choices=["yolo", "rfdetr", "sahi"], default="yolo",
                        help="Inference method (default: yolo)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (YOLO built-in)")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size (default: 640)")
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="SAHI overlap ratio (default: 0.2)")
    parser.add_argument("--sahi-model-type", default="ultralytics",
                        help="SAHI detector backend: 'ultralytics' for YOLO, 'huggingface' for DETR (default: ultralytics)")
    parser.add_argument("--enable-nms", action="store_true",
                        help="Apply cross-modal NMS to suppress RGB/thermal duplicates")
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5,
                        help="IoU threshold for cross-modal NMS (default: 0.5)")
    parser.add_argument("--out", required=True, help="Output path for predictions JSON")
    args = parser.parse_args()

    coco_gt = COCO(args.gt_json)
    test_images_dir = str(Path(args.gt_json).parent.parent / "images")

    print(f"Method: {args.method} | Weights: {args.weights} | Conf: {args.conf}")

    if args.method == "yolo":
        raw_preds, timings_ms = run_yolo_inference(args.weights, test_images_dir, conf=args.conf, iou=args.iou)
        predictions = map_paths_to_image_ids(raw_preds, coco_gt)
    elif args.method == "rfdetr":
        predictions, timings_ms = run_rfdetr_inference(args.weights, args.gt_json, conf=args.conf)
    elif args.method == "sahi":
        print(f"SAHI: slice_size={args.slice_size}, overlap={args.overlap_ratio}, backend={args.sahi_model_type}")
        raw_preds, timings_ms = run_sahi_inference(
            args.weights, test_images_dir,
            conf=args.conf, slice_size=args.slice_size, overlap_ratio=args.overlap_ratio,
            model_type=args.sahi_model_type,
        )
        predictions = map_paths_to_image_ids(raw_preds, coco_gt)

    timing_stats = compute_timing_stats(timings_ms)
    print_timing_report(timing_stats, args.method)

    if args.enable_nms:
        predictions = apply_interclass_nms(predictions, args.gt_json, args.nms_iou_thresh)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "predictions": predictions,
        "timing": timing_stats,
        "config": {
            "method": args.method,
            "weights": args.weights,
            "conf": args.conf,
            "enable_nms": args.enable_nms,
        },
    }
    if args.method == "sahi":
        output["config"]["slice_size"] = args.slice_size
        output["config"]["overlap_ratio"] = args.overlap_ratio
        output["config"]["sahi_model_type"] = args.sahi_model_type

    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
