"""
Per-image quality metrics for failure case analysis.

Computes image-level and object-level metrics that help explain WHY a model
fails on specific images. Called by eval.py after inference.

Outputs a per-image DataFrame with columns:
  image-level:  brightness, contrast_rms, sharpness, noise_level,
                dark_pixel_ratio, overexposed_ratio, dynamic_range,
                edge_density, color_saturation, is_grayscale, color_cast
  object-level: n_gt_boxes, mean_obj_size_px, mean_obj_occupancy,
                mean_obj_brightness, mean_obj_bg_contrast, mean_box_overlap
  model perf:   n_tp, n_fp, n_fn, precision, recall, f1
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Image-level metrics
# ---------------------------------------------------------------------------

def compute_image_metrics(img_bgr: np.ndarray) -> dict:
    """Compute all image-level quality metrics from a BGR image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # Brightness — mean luminance
    brightness = float(np.mean(gray))

    # RMS contrast — std of luminance
    contrast_rms = float(np.std(gray))

    # Sharpness — variance of Laplacian (higher = sharper)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(laplacian))

    # Noise level — median absolute deviation of high-freq component
    # Approximate via difference from box-blurred version
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred
    noise_level = float(np.median(np.abs(residual)) * 1.4826)  # MAD -> sigma estimate

    # Dark / overexposed pixel ratios
    dark_pixel_ratio = float(np.mean(gray < 30 / 255))
    overexposed_ratio = float(np.mean(gray > 240 / 255))

    # Dynamic range
    dynamic_range = float(gray.max() - gray.min())

    # Edge density — Canny edge pixels / total pixels
    gray_u8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_density = float(np.sum(edges > 0) / (h * w))

    # Color saturation — mean HSV-S channel (≈0 for grayscale/thermal)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    color_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

    # Is grayscale — low std across RGB channels
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    rg_diff = np.std(r.astype(np.float32) - g.astype(np.float32))
    gb_diff = np.std(g.astype(np.float32) - b.astype(np.float32))
    is_grayscale = bool((rg_diff < 5.0) and (gb_diff < 5.0))

    # Color cast — how much each channel deviates from mean
    channel_means = [float(np.mean(c)) for c in [r, g, b]]
    overall_mean = np.mean(channel_means)
    color_cast = float(np.max(np.abs(np.array(channel_means) - overall_mean)))

    return {
        "brightness": brightness,
        "contrast_rms": contrast_rms,
        "sharpness": sharpness,
        "noise_level": noise_level,
        "dark_pixel_ratio": dark_pixel_ratio,
        "overexposed_ratio": overexposed_ratio,
        "dynamic_range": dynamic_range,
        "edge_density": edge_density,
        "color_saturation": color_saturation,
        "is_grayscale": is_grayscale,
        "color_cast": color_cast,
        "img_width": w,
        "img_height": h,
    }


# ---------------------------------------------------------------------------
# Object-level metrics (requires GT boxes)
# ---------------------------------------------------------------------------

def compute_object_metrics(img_bgr: np.ndarray, gt_boxes_xywh: list) -> dict:
    """
    Compute per-image object-level metrics from GT bounding boxes.

    gt_boxes_xywh: list of [x_min, y_min, width, height] in absolute pixels (COCO format)
    """
    if not gt_boxes_xywh:
        return {
            "n_gt_boxes": 0,
            "mean_obj_size_px": 0.0,
            "mean_obj_occupancy": 0.0,
            "mean_obj_brightness": 0.0,
            "mean_obj_bg_contrast": 0.0,
            "mean_box_overlap": 0.0,
        }

    h, w = img_bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    sizes, occupancies, brightnesses, contrasts = [], [], [], []

    for x, y, bw, bh in gt_boxes_xywh:
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)
        x2, y2 = min(x + bw, w), min(y + bh, h)
        x, y = max(x, 0), max(y, 0)
        if x2 <= x or y2 <= y:
            continue

        area = (x2 - x) * (y2 - y)
        sizes.append(float(area))
        occupancies.append(float(area / img_area))

        obj_region = gray[y:y2, x:x2]
        brightnesses.append(float(np.mean(obj_region)))

        # Background contrast: compare obj mean to a border strip around it
        pad = 10
        bx1, by1 = max(x - pad, 0), max(y - pad, 0)
        bx2, by2 = min(x2 + pad, w), min(y2 + pad, h)
        bg_region = gray[by1:by2, bx1:bx2]
        bg_mask = np.ones_like(bg_region, dtype=bool)
        # Mask out the object itself
        ry1, ry2 = y - by1, y2 - by1
        rx1, rx2 = x - bx1, x2 - bx1
        bg_mask[ry1:ry2, rx1:rx2] = False
        bg_pixels = bg_region[bg_mask]
        if len(bg_pixels) > 0:
            contrasts.append(float(abs(np.mean(obj_region) - np.mean(bg_pixels))))
        else:
            contrasts.append(0.0)

    # Mean pairwise overlap between GT boxes
    overlaps = []
    boxes = gt_boxes_xywh
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = _box_iou(boxes[i], boxes[j])
            overlaps.append(iou)

    return {
        "n_gt_boxes": len(sizes),
        "mean_obj_size_px": float(np.mean(sizes)) if sizes else 0.0,
        "mean_obj_occupancy": float(np.mean(occupancies)) if occupancies else 0.0,
        "mean_obj_brightness": float(np.mean(brightnesses)) if brightnesses else 0.0,
        "mean_obj_bg_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
        "mean_box_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def _box_iou(box1, box2) -> float:
    """IoU between two COCO-format [x,y,w,h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-image detection performance
# ---------------------------------------------------------------------------

def compute_detection_performance(
    gt_boxes: list,   # list of {"bbox": [x,y,w,h], "category_id": int}
    preds: list,      # list of {"bbox": [x,y,w,h], "score": float, "category_id": int}
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute TP/FP/FN counts per image at given IoU threshold.
    Returns precision, recall, F1 as per-image detection performance proxy.
    """
    if not gt_boxes and not preds:
        return {"n_tp": 0, "n_fp": 0, "n_fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not gt_boxes:
        return {"n_tp": 0, "n_fp": len(preds), "n_fn": 0,
                "precision": 0.0, "recall": 1.0, "f1": 0.0}

    if not preds:
        return {"n_tp": 0, "n_fp": 0, "n_fn": len(gt_boxes),
                "precision": 1.0, "recall": 0.0, "f1": 0.0}

    # Sort predictions by score descending
    preds_sorted = sorted(preds, key=lambda p: p["score"], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp, fp = 0, 0

    for pred in preds_sorted:
        best_iou, best_gt_idx = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            if pred["category_id"] != gt["category_id"]:
                continue
            iou = _box_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = gt_matched.count(False)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"n_tp": tp, "n_fp": fp, "n_fn": fn,
            "precision": precision, "recall": recall, "f1": f1}
