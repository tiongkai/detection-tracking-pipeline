"""
Evaluation script. Runs inference on the test split and computes structured metrics.

Usage:
    python pipeline/eval/eval.py --config pipeline/configs/exp_yolo11l.yaml

Outputs to results/<experiment_name>/eval/:
    metrics.json          — all metrics (overall, per-class, per-source, per-domain, matrix)
    metrics_report.md     — human-readable tables
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Core metric utilities
# ---------------------------------------------------------------------------

def predictions_to_coco(predictions: list) -> list:
    """
    Ensure predictions are in COCO result format.
    Each entry: {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    """
    return [
        {
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": [round(float(v), 2) for v in p["bbox"]],
            "score": float(p["score"]),
        }
        for p in predictions
    ]


def compute_map(gt_json_path: str, predictions: list) -> dict:
    """
    Run pycocotools COCOeval on predictions against ground truth.

    Returns dict with keys: mAP50, mAP50_95, per_class_AP50, per_class_AP50_95
    """
    coco_gt = COCO(gt_json_path)
    categories = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    if not predictions:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_AP50": {}, "per_class_AP50_95": {}}

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # Overall
    mAP50_95 = float(evaluator.stats[0])
    mAP50 = float(evaluator.stats[1])

    # Per-class AP from precision array [T, R, K, A, M]
    # T=10 IoU thresholds, R=101 recall points, K=num_classes, A=4 area ranges, M=3 max dets
    precision = evaluator.eval["precision"]
    per_class_AP50 = {}
    per_class_AP50_95 = {}

    for k, cat_id in enumerate(evaluator.params.catIds):
        name = categories.get(cat_id, str(cat_id))
        # AP50: IoU index 0 (0.50), area=all (index 0), maxDets=100 (index 2)
        p50 = precision[0, :, k, 0, 2]
        per_class_AP50[name] = float(np.mean(p50[p50 > -1])) if np.any(p50 > -1) else 0.0
        # AP50-95: mean over all IoU thresholds
        p_all = precision[:, :, k, 0, 2]
        per_class_AP50_95[name] = float(np.mean(p_all[p_all > -1])) if np.any(p_all > -1) else 0.0

    return {
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "per_class_AP50": per_class_AP50,
        "per_class_AP50_95": per_class_AP50_95,
    }


# ---------------------------------------------------------------------------
# Domain classification (from class names)
# ---------------------------------------------------------------------------

# Original v1/v2 class names
_THERMAL_ORIGINAL = {"boat", "human", "outboard motor", "vessel"}
_RGB_ORIGINAL = {"boat-rgb", "human head", "human torso", "outboard motor-rgb", "human-rgb", "vessel-rgb"}


def classify_domain(class_name: str) -> str:
    """
    Classify a class name as 'thermal', 'rgb', or 'unknown'.

    Handles three naming conventions:
      v1/v2 originals : 'boat' → thermal, 'boat-rgb' → rgb, 'human head' → rgb
      v3 standardised : 'boat-thermal' → thermal, 'boat-rgb' → rgb, 'head-rgb' → rgb
      v4 merged       : 'boat' → thermal (via original list), 'head'/'torso' → unknown
                        (merged classes have no domain suffix; use per-source breakdown instead)
    """
    # v3 explicit suffix — highest priority
    if class_name.endswith("-thermal"):
        return "thermal"
    if class_name.endswith("-rgb"):
        return "rgb"
    # v1/v2 / v4 merged names
    if class_name in _THERMAL_ORIGINAL:
        return "thermal"
    if class_name in _RGB_ORIGINAL:
        return "rgb"
    return "unknown"


def has_domain_classes(class_names: list) -> bool:
    """Return True if this experiment has distinguishable thermal/rgb classes."""
    domains = {classify_domain(n) for n in class_names}
    return "thermal" in domains and "rgb" in domains


# ---------------------------------------------------------------------------
# COCO subset helpers
# ---------------------------------------------------------------------------

def _make_coco_subset(coco_gt: COCO, img_ids: set) -> dict:
    images = [i for i in coco_gt.dataset["images"] if i["id"] in img_ids]
    annotations = [a for a in coco_gt.dataset["annotations"] if a["image_id"] in img_ids]
    return {
        "categories": coco_gt.dataset["categories"],
        "images": images,
        "annotations": annotations,
    }


def _filter_coco_by_categories(coco_or_dict, cat_ids: set) -> Optional[dict]:
    if isinstance(coco_or_dict, COCO):
        images = coco_or_dict.dataset["images"]
        annotations = coco_or_dict.dataset["annotations"]
        all_categories = coco_or_dict.dataset["categories"]
    else:
        images = coco_or_dict["images"]
        annotations = coco_or_dict["annotations"]
        all_categories = coco_or_dict["categories"]
    filtered_cats = [c for c in all_categories if c["id"] in cat_ids]
    filtered_anns = [a for a in annotations if a["category_id"] in cat_ids]
    if not filtered_anns:
        return None
    return {"categories": filtered_cats, "images": images, "annotations": filtered_anns}


# ---------------------------------------------------------------------------
# Breakdown metrics
# ---------------------------------------------------------------------------

def compute_breakdown_metrics(
    gt_json_path: str,
    predictions: list,
    split_manifest: pd.DataFrame,
    min_annotations: int = 10,
) -> dict:
    """
    Compute per-source, per-domain, and per-class x per-source metrics.

    Returns nested dict:
        {
          "per_source": {source: {"mAP50": ..., "mAP50_95": ...}},
          "per_domain": {"thermal": {...}, "rgb": {...}},
          "per_class_per_source": {class_name: {source: {"AP50": ..., "AP50_95": ..., "sparse": bool}}},
        }
    """
    import os
    import tempfile

    coco_gt_full = COCO(gt_json_path)
    categories = {c["id"]: c["name"] for c in coco_gt_full.dataset["categories"]}
    sources = list(split_manifest["source"].unique())

    # Build image_id -> source mapping
    img_source = {
        img["id"]: img.get("source", "unknown")
        for img in coco_gt_full.dataset["images"]
    }

    results = {"per_source": {}, "per_domain": {}, "per_class_per_source": {}}

    # --- Per-source ---
    for source in sources:
        source_img_ids = {iid for iid, src in img_source.items() if src == source}
        source_preds = [p for p in predictions if p["image_id"] in source_img_ids]
        if not source_preds:
            results["per_source"][source] = {"mAP50": 0.0, "mAP50_95": 0.0}
            continue
        subset = _make_coco_subset(coco_gt_full, source_img_ids)
        if not subset["images"]:
            results["per_source"][source] = {"mAP50": 0.0, "mAP50_95": 0.0}
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(subset, f)
            tmp_path = f.name
        try:
            m = compute_map(tmp_path, source_preds)
            results["per_source"][source] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"]}
        finally:
            os.unlink(tmp_path)

    # --- Per-domain ---
    # Use classify_domain() which handles v1/v2 originals, v3 -thermal/-rgb suffixes,
    # and v4 merged names. For v4 (no domain suffixes), thermal = merged classes that
    # were originally thermal; rgb = merged classes originally rgb; head/torso = unknown.
    class_names_list = list(categories.values())
    _has_domain = has_domain_classes(class_names_list)
    results["per_domain"]["_has_explicit_domain_classes"] = _has_domain

    for domain in ["thermal", "rgb"]:
        domain_cat_ids = {
            cid for cid, name in categories.items()
            if classify_domain(name) == domain
        }
        domain_preds = [p for p in predictions if p["category_id"] in domain_cat_ids]
        domain_gt = _filter_coco_by_categories(coco_gt_full, domain_cat_ids)
        if domain_gt is None or not domain_preds:
            results["per_domain"][domain] = {"mAP50": 0.0, "mAP50_95": 0.0}
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(domain_gt, f)
            tmp_path = f.name
        try:
            m = compute_map(tmp_path, domain_preds)
            results["per_domain"][domain] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"]}
        finally:
            os.unlink(tmp_path)

    # --- Per-class x per-source matrix ---
    for cat_id, class_name in categories.items():
        results["per_class_per_source"][class_name] = {}
        for source in sources:
            source_img_ids = {iid for iid, src in img_source.items() if src == source}
            cell_preds = [
                p for p in predictions
                if p["image_id"] in source_img_ids and p["category_id"] == cat_id
            ]
            cell_ann_count = sum(
                1 for a in coco_gt_full.dataset["annotations"]
                if a["image_id"] in source_img_ids and a["category_id"] == cat_id
            )
            sparse = cell_ann_count < min_annotations
            if not cell_preds or cell_ann_count == 0:
                results["per_class_per_source"][class_name][source] = {
                    "AP50": 0.0, "AP50_95": 0.0, "sparse": sparse, "n_annotations": cell_ann_count
                }
                continue
            cell_subset = _make_coco_subset(coco_gt_full, source_img_ids)
            cell_gt = _filter_coco_by_categories(cell_subset, {cat_id})
            if cell_gt is None:
                results["per_class_per_source"][class_name][source] = {
                    "AP50": 0.0, "AP50_95": 0.0, "sparse": sparse, "n_annotations": cell_ann_count
                }
                continue
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(cell_gt, f)
                tmp_path = f.name
            try:
                m = compute_map(tmp_path, cell_preds)
                results["per_class_per_source"][class_name][source] = {
                    "AP50": m["mAP50"], "AP50_95": m["mAP50_95"],
                    "sparse": sparse, "n_annotations": cell_ann_count,
                }
            finally:
                os.unlink(tmp_path)

    return results


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def run_yolo_inference(weights_path: str, test_images_dir: str, conf: float = 0.001) -> list:
    """Run YOLO inference on test set. Returns predictions with _img_path field."""
    from ultralytics import YOLO
    from pathlib import Path as P

    model = YOLO(weights_path)
    results = model.predict(
        source=test_images_dir,
        conf=conf,
        iou=0.6,
        save=False,
        verbose=False,
        stream=True,
    )

    predictions = []
    for r in results:
        img_path = str(P(r.path).resolve())
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
    return predictions


def run_rfdetr_inference(weights_path: str, test_coco_json: str, conf: float = 0.3) -> list:
    """Run RF-DETR inference on test set. Returns COCO-format predictions."""
    from rfdetr import RFDETRBase, RFDETRLarge
    from PIL import Image as PILImage

    model_class = RFDETRLarge if "large" in str(weights_path).lower() else RFDETRBase
    model = model_class(pretrain_weights=weights_path)

    coco_gt = COCO(test_coco_json)
    predictions = []

    total = len(coco_gt.dataset["images"])
    for idx, img_info in enumerate(coco_gt.dataset["images"], 1):
        img_path = img_info["file_name"]
        img = PILImage.open(img_path).convert("RGB")
        detections = model.predict(img, threshold=conf)

        # supervision.Detections — access via numpy arrays, not iteration
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

    return predictions


# ---------------------------------------------------------------------------
# Interclass NMS
# ---------------------------------------------------------------------------

def _build_class_groups_from_coco(coco_json_path: str) -> dict:
    """Build cross-modal NMS groups from COCO category names.
    Groups classes that share the same base name after stripping -rgb/-thermal."""
    coco_gt = COCO(coco_json_path)
    name_to_ids = {}
    for cat in coco_gt.dataset["categories"]:
        base = cat["name"].replace("-rgb", "").replace("-thermal", "")
        name_to_ids.setdefault(base, set()).add(cat["id"])
    return {base: ids for base, ids in name_to_ids.items() if len(ids) > 1}


def apply_interclass_nms(predictions: list, coco_json_path: str, iou_thresh: float = 0.5) -> list:
    """Apply cross-modal NMS per image on COCO-format predictions."""
    from pipeline.eval.cross_modal_nms import cross_modal_nms

    class_groups = _build_class_groups_from_coco(coco_json_path)
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
        # Convert COCO [x,y,w,h] to [x1,y1,x2,y2,conf,cls]
        dets = np.array([
            [p["bbox"][0], p["bbox"][1],
             p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3],
             p["score"], p["category_id"]]
            for p in preds
        ], dtype=np.float32)

        filtered = cross_modal_nms(dets, class_groups, iou_thresh)

        # Map back to COCO format
        for d in filtered:
            x1, y1, x2, y2 = d[:4]
            kept.append({
                "image_id": img_id,
                "category_id": int(d[5]),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": float(d[4]),
            })

    print(f"  Interclass NMS: {total_before} → {len(kept)} predictions ({total_before - len(kept)} suppressed)")
    return kept


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_metrics_report(metrics: dict, out_path: Path) -> None:
    lines = ["# Evaluation Report\n"]

    # Overall
    lines += [
        "## Overall\n",
        "| mAP50 | mAP50-95 |",
        "| ----- | -------- |",
        f"| {metrics['overall']['mAP50']:.4f} | {metrics['overall']['mAP50_95']:.4f} |",
        "",
    ]

    # Per source
    lines += ["## Per Source\n", "| Source | mAP50 | mAP50-95 |", "| ------ | ----- | -------- |"]
    for src, m in metrics.get("per_source", {}).items():
        lines.append(f"| {src} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} |")
    lines.append("")

    # Per class
    lines += [
        "## Per Class\n",
        "| Class | Domain | AP50 | AP50-95 |",
        "| ----- | ------ | ---- | ------- |",
    ]
    for cls_name, ap50 in metrics["overall"].get("per_class_AP50", {}).items():
        domain = classify_domain(cls_name)
        ap95 = metrics["overall"]["per_class_AP50_95"].get(cls_name, 0.0)
        lines.append(f"| {cls_name} | {domain} | {ap50:.4f} | {ap95:.4f} |")
    lines.append("")

    # Domain summary
    pd_data = metrics.get("per_domain", {})
    has_domain = pd_data.get("_has_explicit_domain_classes", True)
    lines += ["## Domain Summary\n"]
    if not has_domain:
        lines.append("_Merged-class experiment (v4-style): no explicit thermal/rgb class suffixes. "
                     "Use per-source breakdown (st/willow = thermal, dahua = poor-light RGB) as domain proxy._\n")
    lines += ["| Domain | mAP50 | mAP50-95 |", "| ------ | ----- | -------- |"]
    for domain, m in pd_data.items():
        if not isinstance(m, dict):
            continue  # skip _has_explicit_domain_classes flag
        lines.append(f"| {domain} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} |")
    lines.append("")

    # Per class x per source matrix
    per_cls_src = metrics.get("per_class_per_source", {})
    if per_cls_src:
        sources = list(next(iter(per_cls_src.values())).keys())
        header = "| Class | " + " | ".join(sources) + " |"
        sep = "| ----- | " + " | ".join(["----"] * len(sources)) + " |"
        lines += [
            "## Per Class x Per Source\n",
            "Format: AP50/AP50-95 ([sparse] = fewer than min_annotations annotations)\n",
            header, sep,
        ]
        for cls_name, src_metrics in per_cls_src.items():
            cells = []
            for src in sources:
                m = src_metrics.get(src, {})
                ap50 = m.get("AP50", 0.0)
                ap95 = m.get("AP50_95", 0.0)
                tag = " [sparse]" if m.get("sparse") else ""
                cells.append(f"{ap50:.3f}/{ap95:.3f}{tag}")
            lines.append(f"| {cls_name} | " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--enable-interclass-nms", action="store_true",
                        help="Apply cross-modal NMS to suppress duplicate detections across RGB/thermal class pairs")
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5,
                        help="IoU threshold for interclass NMS (default 0.5)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]
    split_name = config["split"]
    model_type = config["model"]

    split_dir = PROJECT_ROOT / "data" / "splits" / split_name
    test_coco_json = str(split_dir / "coco" / "test" / "_annotations.coco.json")
    manifest_path = split_dir / "split_manifest.csv"
    split_manifest = pd.read_csv(manifest_path)
    test_manifest = split_manifest[split_manifest["split"] == "test"]

    results_dir = PROJECT_ROOT / "results" / exp_name
    eval_dir = results_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    weights_path = str(results_dir / "weights" / "best.pt")

    print(f"Running inference with {weights_path}...")
    if model_type in ("yolo11", "yolo26"):
        test_images_dir = str(split_dir / "test" / "images")
        raw_preds = run_yolo_inference(weights_path, test_images_dir, conf=args.conf)
        # Map file paths to image IDs
        coco_gt = COCO(test_coco_json)
        path_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
        predictions = []
        for p in raw_preds:
            img_id = path_to_id.get(p["_img_path"])
            if img_id is None:
                continue
            predictions.append({
                "image_id": img_id,
                "category_id": p["category_id"],
                "bbox": p["bbox"],
                "score": p["score"],
            })
    elif model_type == "rfdetr":
        for ckpt_name in ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]:
            candidate = results_dir / ckpt_name
            if candidate.exists():
                weights_path = str(candidate)
                break
        print(f"Using checkpoint: {weights_path}")
        predictions = run_rfdetr_inference(weights_path, test_coco_json, conf=0.3)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    predictions = predictions_to_coco(predictions)

    if args.enable_interclass_nms:
        predictions = apply_interclass_nms(predictions, test_coco_json, args.nms_iou_thresh)

    (eval_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))

    print("Computing metrics...")
    overall = compute_map(test_coco_json, predictions)

    breakdown = compute_breakdown_metrics(
        test_coco_json,
        predictions,
        test_manifest,
        min_annotations=config.get("min_annotations_for_sparse", 10),
    )

    metrics = {
        "experiment": exp_name,
        "model": config["model"],
        "size": config["size"],
        "split": split_name,
        "overall": overall,
        **breakdown,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_metrics_report(metrics, eval_dir / "metrics_report.md")
    print(f"Eval complete. Report: {eval_dir / 'metrics_report.md'}")

    # ------------------------------------------------------------------
    # Per-image quality metrics + failure analysis
    # ------------------------------------------------------------------
    print("Computing per-image quality metrics...")
    _run_image_quality_analysis(
        test_coco_json=test_coco_json,
        predictions=predictions,
        eval_dir=eval_dir,
        test_manifest=test_manifest,
    )


def _run_image_quality_analysis(
    test_coco_json: str,
    predictions: list,
    eval_dir: Path,
    test_manifest: pd.DataFrame,
) -> None:
    """Compute per-image metrics and run failure analysis with visualisations."""
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    import cv2 as _cv2
    from pipeline.eval.image_metrics import (
        compute_image_metrics, compute_object_metrics, compute_detection_performance
    )
    from pipeline.eval.failure_analysis import run_failure_analysis

    coco_gt = COCO(test_coco_json)
    gt_by_img = {}
    for ann in coco_gt.dataset["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    pred_by_img = {}
    for p in predictions:
        pred_by_img.setdefault(p["image_id"], []).append(p)

    # Source lookup from manifest
    stem_to_source = dict(zip(test_manifest["filename"], test_manifest["source"]))

    rows = []
    total = len(coco_gt.dataset["images"])
    for idx, img_info in enumerate(coco_gt.dataset["images"], 1):
        img_path = img_info["file_name"]
        stem = Path(img_path).stem
        img_bgr = _cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Image-level metrics
        img_m = compute_image_metrics(img_bgr)

        # Object-level metrics
        gt_boxes = gt_by_img.get(img_info["id"], [])
        gt_boxes_xywh = [ann["bbox"] for ann in gt_boxes]
        obj_m = compute_object_metrics(img_bgr, gt_boxes_xywh)

        # Per-image detection performance
        pred_boxes = pred_by_img.get(img_info["id"], [])
        det_m = compute_detection_performance(gt_boxes, pred_boxes)

        rows.append({
            "image_id": img_info["id"],
            "filename": stem,
            "source": stem_to_source.get(stem, img_info.get("source", "unknown")),
            **img_m,
            **obj_m,
            **det_m,
        })

        if idx % 200 == 0 or idx == total:
            print(f"  {idx}/{total} images processed...")

    img_metrics_df = pd.DataFrame(rows)
    csv_path = eval_dir / "image_metrics.csv"
    img_metrics_df.to_csv(csv_path, index=False)
    print(f"Per-image metrics saved to {csv_path}")

    # Load cached split_metrics.csv if available (generated by compute_split_metrics.py)
    split_dir = Path(test_coco_json).parent.parent.parent  # .../coco/test/.. -> splits/<name>
    split_metrics_csv = split_dir / "split_metrics.csv"
    train_metrics_csv = str(split_metrics_csv) if split_metrics_csv.exists() else None

    print("Running failure analysis and generating visualisations...")
    run_failure_analysis(
        metrics_csv=str(csv_path),
        predictions_json=str(eval_dir / "predictions.json"),
        gt_json=test_coco_json,
        out_dir=str(eval_dir),
        train_metrics_csv=train_metrics_csv,
    )


if __name__ == "__main__":
    main()
