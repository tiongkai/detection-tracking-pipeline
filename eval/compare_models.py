"""
Compare multiple YOLO models on the same holdout set.

Outputs per model:
  - mAP50 / mAP50-95 (via model.val())
  - false_positives/   images with FP boxes drawn (red)
  - false_negatives/   images with missed GT drawn (green dashed)
  - all_annotated/     all images with GT/TP/FP/FN drawn
  - per_image.csv      per-image TP/FP/FN counts

Cross-model outputs:
  - comparison.csv     per-image metrics for all models side by side
  - disagreements/     images where models disagree
  - summary.md         mAP table + aggregate counts

Usage:
    python pipeline/eval/compare_models.py \
        --models model_a.pt model_b.pt \
        --data data.yaml \
        --conf 0.5 \
        --output eval_results/
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_iou_xyxy(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _dedup_preds(pred_boxes, iou_thresh=0.5):
    """
    Remove duplicate predictions via NMS-style IoU dedup.
    Keeps the higher-confidence prediction when two overlap above threshold.
    """
    if not pred_boxes:
        return pred_boxes
    preds = sorted(pred_boxes, key=lambda x: x["conf"], reverse=True)
    keep = []
    for pred in preds:
        is_dup = False
        for kept in keep:
            if pred["class_id"] == kept["class_id"]:
                iou = _box_iou_xyxy(pred["bbox_xyxy"], kept["bbox_xyxy"])
                if iou > iou_thresh:
                    is_dup = True
                    break
        if not is_dup:
            keep.append(pred)
    return keep


def _match_preds(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Greedy match predictions to GT at given IoU threshold.

    Args:
        gt_boxes:   list of {"bbox_xyxy": [x1,y1,x2,y2], "class_id": int}
        pred_boxes: list of {"bbox_xyxy": [x1,y1,x2,y2], "class_id": int, "conf": float}

    Returns:
        tp_pred_idxs: set of pred indices that are TPs
        fp_pred_idxs: set of pred indices that are FPs
        fn_gt_idxs:   set of GT indices that were missed
    """
    preds_sorted = sorted(enumerate(pred_boxes), key=lambda x: x[1]["conf"], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp_pred_idxs = set()
    fp_pred_idxs = set()

    for pi, pred in preds_sorted:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue
            iou = _box_iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh and best_gi >= 0:
            tp_pred_idxs.add(pi)
            gt_matched[best_gi] = True
        else:
            fp_pred_idxs.add(pi)

    fn_gt_idxs = {gi for gi, matched in enumerate(gt_matched) if not matched}
    return tp_pred_idxs, fp_pred_idxs, fn_gt_idxs


# ---------------------------------------------------------------------------
# YOLO label loading
# ---------------------------------------------------------------------------

def _load_gt_yolo(label_path, img_w, img_h):
    """
    Read a YOLO .txt label file (class cx cy w h, normalised).
    Returns list of {"bbox_xyxy": [x1,y1,x2,y2], "class_id": int}.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append({"bbox_xyxy": [x1, y1, x2, y2], "class_id": cls})
    return boxes


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_regression_image(img_path, label_path, label_text, out_path):
    """
    Draw GT boxes with a LOST or GAINED label on the image.

    LOST:   red boxes + "LOST" label — detection regressed
    GAINED: green boxes + "GAINED" label — detection improved
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    gt_boxes = _load_gt_yolo(str(label_path), w, h)

    if label_text == "LOST":
        color = (0, 0, 220)      # red
    else:
        color = (0, 200, 0)      # green

    for gt in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in gt["bbox_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # Label with background for readability
        txt_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx, ty = x1, max(y1 - 8, txt_size[1] + 4)
        cv2.rectangle(img, (tx, ty - txt_size[1] - 4), (tx + txt_size[0] + 4, ty + 4), color, -1)
        cv2.putText(img, label_text, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), img)


def _draw_and_save(img_path, gt_boxes, pred_boxes, tp_idxs, fp_idxs, fn_idxs, out_dirs):
    """
    Draw annotated image and save to relevant directories.

    Colors: GT = green, TP = blue, FP = red, missed GT (FN) = yellow dashed.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    out = img.copy()
    fname = Path(img_path).name

    # Draw GT boxes (green, thin)
    for gi, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = [int(v) for v in gt["bbox_xyxy"]]
        if gi in fn_idxs:
            # Missed GT — yellow, thicker
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 3)
            cv2.putText(out, "MISSED", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Draw predictions
    for pi, pred in enumerate(pred_boxes):
        x1, y1, x2, y2 = [int(v) for v in pred["bbox_xyxy"]]
        if pi in tp_idxs:
            color = (200, 120, 0)  # blue-ish (TP)
            label = f"TP {pred['conf']:.2f}"
        else:
            color = (0, 0, 220)  # red (FP)
            label = f"FP {pred['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, min(y2 + 14, out.shape[0] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save to all_annotated always
    cv2.imwrite(str(out_dirs["all"] / fname), out)

    # Save to false_positives if any FP
    if fp_idxs:
        cv2.imwrite(str(out_dirs["fp"] / fname), out)

    # Save to false_negatives if any FN
    if fn_idxs:
        cv2.imwrite(str(out_dirs["fn"] / fname), out)


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_single_model(model_path, data_yaml, conf, iou_thresh, out_dir, device=0):
    """
    Evaluate a single model: mAP via val(), FP/FN via predict().

    Returns dict with mAP metrics + per-image results.
    """
    from ultralytics import YOLO

    # Use grandparent dir name if file is "best.pt" or "last.pt" to avoid collisions
    p = Path(model_path)
    if p.stem in ("best", "last") and p.parent.name == "weights":
        model_name = p.parent.parent.name
    else:
        model_name = p.stem
    model_dir = Path(out_dir) / model_name
    dirs = {
        "fp": model_dir / "false_positives",
        "fn": model_dir / "false_negatives",
        "all": model_dir / "all_annotated",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --- Step 1: mAP via model.val() ---
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test", device=device, verbose=False)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    print(f"  mAP50: {map50:.4f}  mAP50-95: {map50_95:.4f}")

    # --- Step 2: Predict at fixed conf for FP/FN extraction ---
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_cfg["path"])
    test_key = data_cfg.get("test", data_cfg.get("val", "images"))
    images_dir = data_root / test_key
    labels_dir = data_root / test_key.replace("images", "labels")

    results = model.predict(source=str(images_dir), conf=conf, iou=iou_thresh,
                            save=False, verbose=False, stream=True)

    rows = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for r in results:
        img_path = Path(r.path)
        img_h, img_w = r.orig_shape

        # Load GT
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = _load_gt_yolo(str(label_path), img_w, img_h)

        # Parse predictions
        pred_boxes = []
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_id": int(box.cls[0]),
                    "conf": float(box.conf[0]),
                })

        # Dedup overlapping predictions, then match
        pred_boxes = _dedup_preds(pred_boxes, iou_thresh)
        tp_idxs, fp_idxs, fn_idxs = _match_preds(gt_boxes, pred_boxes, iou_thresh)

        n_tp = len(tp_idxs)
        n_fp = len(fp_idxs)
        n_fn = len(fn_idxs)
        total_tp += n_tp
        total_fp += n_fp
        total_fn += n_fn

        img_prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 1.0
        img_rec = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 1.0
        img_f1 = 2 * img_prec * img_rec / (img_prec + img_rec) if (img_prec + img_rec) > 0 else 0.0

        rows.append({
            "filename": img_path.name,
            "n_gt": len(gt_boxes),
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_fn": n_fn,
            "precision": round(img_prec, 4),
            "recall": round(img_rec, 4),
            "f1": round(img_f1, 4),
        })

        # Draw and save
        _draw_and_save(img_path, gt_boxes, pred_boxes, tp_idxs, fp_idxs, fn_idxs, dirs)

    # Write per_image.csv
    csv_path = model_dir / "per_image.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    fp_count = sum(1 for r in rows if r["n_fp"] > 0)
    fn_count = sum(1 for r in rows if r["n_fn"] > 0)

    print(f"  At conf={conf}: TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Images with FP: {fp_count}  Images with FN: {fn_count}")
    print(f"  Saved to: {model_dir}")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "map50": map50,
        "map50_95": map50_95,
        "val_precision": precision,
        "val_recall": recall,
        "conf_thresh": conf,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "agg_precision": agg_prec,
        "agg_recall": agg_rec,
        "images_with_fp": fp_count,
        "images_with_fn": fn_count,
        "total_images": len(rows),
        "per_image": rows,
        "_data_yaml": data_yaml,
        "_images_dir": images_dir,
        "_labels_dir": labels_dir,
    }


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def build_comparison(all_results, out_dir):
    """Build comparison.csv and disagreements/ from multiple model results."""
    out_dir = Path(out_dir)

    # Build per-image lookup: filename -> {model: row}
    filenames = set()
    model_rows = {}
    for res in all_results:
        name = res["model_name"]
        model_rows[name] = {r["filename"]: r for r in res["per_image"]}
        filenames.update(model_rows[name].keys())

    model_names = [r["model_name"] for r in all_results]

    # Write comparison.csv
    header = ["filename"]
    for mn in model_names:
        header += [f"{mn}_tp", f"{mn}_fp", f"{mn}_fn", f"{mn}_f1"]

    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fname in sorted(filenames):
            row = [fname]
            for mn in model_names:
                r = model_rows[mn].get(fname, {})
                row += [r.get("n_tp", 0), r.get("n_fp", 0),
                        r.get("n_fn", 0), r.get("f1", 0)]
            writer.writerow(row)

    # Pairwise regressions/progressions between consecutive models
    if len(model_names) >= 2:
        # We need GT + image paths to draw regression images
        # Parse from data_yaml stored in results
        data_yaml = all_results[0].get("_data_yaml")
        images_dir = all_results[0].get("_images_dir")
        labels_dir = all_results[0].get("_labels_dir")

        for i in range(len(model_names) - 1):
            prev_mn = model_names[i]
            curr_mn = model_names[i + 1]
            pair_label = f"{prev_mn}_vs_{curr_mn}"
            pair_dir = out_dir / "regressions" / pair_label
            lost_dir = pair_dir / "lost_detections"
            gained_dir = pair_dir / "gained_detections"
            lost_dir.mkdir(parents=True, exist_ok=True)
            gained_dir.mkdir(parents=True, exist_ok=True)

            lost_rows = []
            gained_rows = []

            for fname in sorted(filenames):
                prev_r = model_rows[prev_mn].get(fname, {})
                curr_r = model_rows[curr_mn].get(fname, {})
                prev_fn = int(prev_r.get("n_fn", 0))
                curr_fn = int(curr_r.get("n_fn", 0))

                # Lost: was detected (FN=0) but now missed (FN>0)
                if prev_fn == 0 and curr_fn > 0:
                    lost_rows.append({
                        "filename": fname,
                        f"{prev_mn}_fn": prev_fn,
                        f"{curr_mn}_fn": curr_fn,
                    })
                    if images_dir:
                        _draw_regression_image(
                            images_dir / fname, labels_dir / (Path(fname).stem + ".txt"),
                            "LOST", lost_dir / fname,
                        )

                # Gained: was missed (FN>0) but now detected (FN=0)
                if prev_fn > 0 and curr_fn == 0:
                    gained_rows.append({
                        "filename": fname,
                        f"{prev_mn}_fn": prev_fn,
                        f"{curr_mn}_fn": curr_fn,
                    })
                    if images_dir:
                        _draw_regression_image(
                            images_dir / fname, labels_dir / (Path(fname).stem + ".txt"),
                            "GAINED", gained_dir / fname,
                        )

            # Write CSVs
            if lost_rows:
                csv_path = pair_dir / "lost_detections.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=lost_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(lost_rows)

            if gained_rows:
                csv_path = pair_dir / "gained_detections.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=gained_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(gained_rows)

            print(f"\n{prev_mn} -> {curr_mn}:")
            print(f"  Lost detections: {len(lost_rows)} images")
            print(f"  Gained detections: {len(gained_rows)} images")
            print(f"  Saved to: {pair_dir}")

    # --- Consistent failures ---
    _build_consistent_failures(model_rows, model_names, filenames, out_dir)


def _build_consistent_failures(model_rows, model_names, filenames, out_dir):
    """Find images consistently missed or mispredicted across all models."""
    out_dir = Path(out_dir)
    consist_dir = out_dir / "consistent_failures"
    fn_dir = consist_dir / "always_fn"
    fp_dir = consist_dir / "always_fp"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    always_fn = []
    always_fp = []

    for fname in sorted(filenames):
        rows = [model_rows[mn].get(fname, {}) for mn in model_names]
        fn_counts = [int(r.get("n_fn", 0)) for r in rows]
        fp_counts = [int(r.get("n_fp", 0)) for r in rows]

        if all(fn > 0 for fn in fn_counts):
            always_fn.append({"filename": fname, **{f"{mn}_fn": fn_counts[i] for i, mn in enumerate(model_names)}})
            # Copy annotated image from last (best) model
            src = out_dir / model_names[-1] / "all_annotated" / fname
            if src.exists():
                shutil.copy2(src, fn_dir / fname)

        if all(fp > 0 for fp in fp_counts):
            always_fp.append({"filename": fname, **{f"{mn}_fp": fp_counts[i] for i, mn in enumerate(model_names)}})
            src = out_dir / model_names[-1] / "all_annotated" / fname
            if src.exists():
                shutil.copy2(src, fp_dir / fname)

    # Write CSV
    if always_fn:
        csv_path = consist_dir / "always_fn.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=always_fn[0].keys())
            writer.writeheader()
            writer.writerows(always_fn)

    if always_fp:
        csv_path = consist_dir / "always_fp.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=always_fp[0].keys())
            writer.writeheader()
            writer.writerows(always_fp)

    print(f"Consistent failures: {len(always_fn)} always-FN, {len(always_fp)} always-FP")
    print(f"  Saved to: {consist_dir}")


def write_summary(all_results, out_dir, data_yaml, conf):
    """Write summary.md with mAP table and aggregate stats."""
    out_dir = Path(out_dir)
    lines = [
        "# Model Comparison Summary\n",
        f"- **Dataset**: `{data_yaml}`",
        f"- **Confidence threshold** (for FP/FN): {conf}",
        f"- **IoU threshold**: 0.5\n",
        "## mAP Metrics (model.val)\n",
        "| Model | mAP50 | mAP50-95 | Precision | Recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for res in all_results:
        lines.append(
            f"| {res['model_name']} | {res['map50']:.4f} | {res['map50_95']:.4f} "
            f"| {res['val_precision']:.4f} | {res['val_recall']:.4f} |"
        )

    lines += [
        f"\n## Detection at conf={conf}\n",
        "| Model | TP | FP | FN | Precision | Recall | Images w/ FP | Images w/ FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in all_results:
        lines.append(
            f"| {res['model_name']} | {res['total_tp']} | {res['total_fp']} | {res['total_fn']} "
            f"| {res['agg_precision']:.4f} | {res['agg_recall']:.4f} "
            f"| {res['images_with_fp']} | {res['images_with_fn']} |"
        )

    lines.append("\n## Output Structure\n")
    lines.append("```")
    lines.append(f"{out_dir}/")
    for res in all_results:
        mn = res["model_name"]
        lines.append(f"  {mn}/")
        lines.append(f"    false_positives/   ({res['images_with_fp']} images)")
        lines.append(f"    false_negatives/   ({res['images_with_fn']} images)")
        lines.append(f"    all_annotated/     ({res['total_images']} images)")
        lines.append(f"    per_image.csv")
    lines.append("  comparison.csv")
    lines.append("  disagreements/")
    lines.append("  summary.md")
    lines.append("```")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare YOLO models on a holdout set.")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to .pt model files")
    parser.add_argument("--data", required=True, help="Path to data.yaml (ultralytics format)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for FP/FN (default 0.5)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching (default 0.5)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", type=int, default=0, help="GPU device (default 0)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_path in args.models:
        res = evaluate_single_model(
            model_path=model_path,
            data_yaml=args.data,
            conf=args.conf,
            iou_thresh=args.iou,
            out_dir=str(out_dir),
            device=args.device,
        )
        all_results.append(res)

    if len(all_results) > 1:
        build_comparison(all_results, out_dir)

    write_summary(all_results, out_dir, args.data, args.conf)


if __name__ == "__main__":
    main()
