"""
Failure analysis and visualisation for object detection evaluation.

Reads per-image metrics CSV and generates:
  - failure_analysis.md   — correlation table, failure buckets, worst cases
  - visualisations/       — 20 PNG plots

Usage (called by eval.py, or standalone):
    python pipeline/eval/failure_analysis.py --metrics results/<exp>/eval/image_metrics.csv
                                              --predictions results/<exp>/eval/predictions.json
                                              --gt data/splits/<split>/coco/test/_annotations.coco.json
                                              --out results/<exp>/eval/
"""

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


QUALITY_METRICS = [
    "brightness", "contrast_rms", "sharpness", "noise_level",
    "dark_pixel_ratio", "overexposed_ratio", "dynamic_range",
    "edge_density", "color_saturation", "color_cast",
    "mean_obj_size_px", "mean_obj_bg_contrast", "n_gt_boxes",
]

FAILURE_BUCKETS = {
    "dark":          ("brightness",         "lt", 0.15),
    "blurry":        ("sharpness",          "lt", 50.0),
    "low_contrast":  ("contrast_rms",       "lt", 0.06),
    "noisy":         ("noise_level",        "gt", 0.03),
    "small_objects": ("mean_obj_size_px",   "lt", 800.0),
    "low_obj_contrast": ("mean_obj_bg_contrast", "lt", 0.05),
    "cluttered":     ("n_gt_boxes",         "gt", 8),
}

SOURCE_COLORS = {"st": "#e74c3c", "willow": "#f39c12", "dahua": "#3498db"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket_mask(df: pd.DataFrame, metric: str, op: str, thresh: float) -> pd.Series:
    if metric not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    if op == "lt":
        return df[metric] < thresh
    return df[metric] > thresh


def _correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in QUALITY_METRICS:
        if m not in df.columns:
            continue
        valid = df[[m, "f1"]].dropna()
        if len(valid) < 5:
            continue
        rho, pval = spearmanr(valid[m], valid["f1"])
        rows.append({"metric": m, "spearman_rho": round(rho, 3), "p_value": round(pval, 4)})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)


def _load_image(file_name: str) -> np.ndarray:
    """Load image from absolute path stored in COCO file_name field."""
    img = cv2.imread(file_name)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {file_name}")
    return img


def _draw_boxes(img_bgr, gt_boxes, pred_boxes, matched_pred_ids, class_names):
    """
    Draw GT (green solid), TP predictions (blue), FP predictions (red).
    Returns annotated RGB image.
    """
    out = img_bgr.copy()
    # GT boxes — green
    for box in gt_boxes:
        x, y, w, h = [int(v) for v in box["bbox"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cls = class_names[box["category_id"]] if box["category_id"] < len(class_names) else str(box["category_id"])
        cv2.putText(out, cls, (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    # Predictions — blue = TP, red = FP
    for i, pred in enumerate(pred_boxes):
        color = (200, 120, 0) if i in matched_pred_ids else (0, 0, 220)
        x, y, w, h = [int(v) for v in pred["bbox"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{pred['score']:.2f}"
        cv2.putText(out, label, (x, min(y + h + 12, out.shape[0] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Return set of prediction indices that are TPs."""
    from pipeline.eval.image_metrics import _box_iou
    preds_sorted = sorted(enumerate(pred_boxes), key=lambda x: x[1]["score"], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp_pred_ids = set()
    for pi, pred in preds_sorted:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gt_matched[gi] or pred["category_id"] != gt["category_id"]:
                continue
            iou = _box_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_threshold and best_gi >= 0:
            tp_pred_ids.add(pi)
            gt_matched[best_gi] = True
    return tp_pred_ids


# ---------------------------------------------------------------------------
# 20 Visualisations
# ---------------------------------------------------------------------------

def generate_visualisations(
    df: pd.DataFrame,
    out_dir: Path,
    gt_data: dict,
    pred_data: list,
    class_names: list,
    train_df: pd.DataFrame = None,
) -> None:
    """
    df:       per-image test metrics (with F1 column)
    train_df: per-image train metrics from split_metrics.csv (optional but recommended)
    """
    vis_dir = out_dir / "visualisations"
    vis_dir.mkdir(exist_ok=True)

    sources = df["source"].unique().tolist()

    def savefig(name):
        plt.tight_layout()
        plt.savefig(vis_dir / name, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {name}")

    print("Generating visualisations...")

    # ------------------------------------------------------------------
    # 01-04: Train vs Test distribution for key metrics
    # If train_df not available, fall back to test-only by source
    # ------------------------------------------------------------------
    dist_metrics = [
        ("brightness",       "Brightness"),
        ("sharpness",        "Sharpness (Laplacian variance)"),
        ("contrast_rms",     "RMS Contrast"),
        ("color_saturation", "Colour Saturation (0 = grayscale/thermal)"),
    ]
    for i, (metric, title) in enumerate(dist_metrics, start=1):
        fig, ax = plt.subplots(figsize=(8, 4))
        if train_df is not None and metric in train_df.columns:
            # Show train distribution vs test distribution
            train_vals = train_df[metric].dropna()
            test_vals = df[metric].dropna()
            # Normalise to density so scales are comparable
            ax.hist(train_vals, bins=40, density=True, alpha=0.55,
                    label=f"Train (n={len(train_vals)})", color="#2ecc71")
            ax.hist(test_vals, bins=40, density=True, alpha=0.55,
                    label=f"Test (n={len(test_vals)})", color="#e74c3c")
            ax.set_ylabel("Density")
            ax.set_title(f"{title} — Train vs Test")
        else:
            # Fallback: test only, coloured by source
            for src in sources:
                vals = df[df["source"] == src][metric].dropna()
                ax.hist(vals, bins=30, alpha=0.6, label=src,
                        color=SOURCE_COLORS.get(src, "#888"))
            ax.set_ylabel("Images")
            ax.set_title(f"{title} (test set by source)")
        ax.set_xlabel(metric)
        ax.legend()
        savefig(f"0{i}_{metric}_dist.png")

    # ------------------------------------------------------------------
    # 05-09: Metric vs F1 scatter plots (per image)
    # ------------------------------------------------------------------
    scatter_metrics = [
        ("brightness",        "Brightness vs Detection F1"),
        ("sharpness",         "Sharpness vs Detection F1"),
        ("contrast_rms",      "RMS Contrast vs Detection F1"),
        ("mean_obj_bg_contrast", "Object-Background Contrast vs F1"),
        ("mean_obj_size_px",  "Mean Object Size (px) vs F1"),
    ]
    for i, (metric, title) in enumerate(scatter_metrics, start=5):
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for src in sources:
            sub = df[df["source"] == src]
            ax.scatter(sub[metric], sub["f1"], alpha=0.4, s=15,
                       label=src, color=SOURCE_COLORS.get(src, "#888"))
        ax.set_xlabel(metric)
        ax.set_ylabel("Per-image F1 (TP/(TP+FP+FN))")
        ax.set_title(title)
        ax.legend()
        # Trend line
        valid = df[[metric, "f1"]].dropna()
        if len(valid) > 10:
            z = np.polyfit(valid[metric], valid["f1"], 1)
            xr = np.linspace(valid[metric].min(), valid[metric].max(), 100)
            ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1, alpha=0.5, label="trend")
        savefig(f"0{i}_{metric}_vs_f1.png")

    # ------------------------------------------------------------------
    # 10: Spearman correlation heatmap (metrics vs F1)
    # ------------------------------------------------------------------
    corr_df = _correlation_table(df)
    fig, ax = plt.subplots(figsize=(8, max(4, len(corr_df) * 0.4)))
    colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in corr_df["spearman_rho"]]
    bars = ax.barh(corr_df["metric"], corr_df["spearman_rho"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ with per-image F1")
    ax.set_title("Quality Metric Correlation with Detection Performance")
    for bar, (_, row) in zip(bars, corr_df.iterrows()):
        ax.text(bar.get_width() + 0.01 * np.sign(bar.get_width()),
                bar.get_y() + bar.get_height() / 2,
                f"ρ={row['spearman_rho']:.2f}", va="center", fontsize=8)
    savefig("10_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # 11-12: Failure bucket analysis
    # ------------------------------------------------------------------
    bucket_stats = []
    for bucket, (metric, op, thresh) in FAILURE_BUCKETS.items():
        mask = _bucket_mask(df, metric, op, thresh)
        n = mask.sum()
        mean_f1 = df[mask]["f1"].mean() if n > 0 else 0.0
        bucket_stats.append({"bucket": bucket, "n_images": n, "mean_f1": mean_f1})
    bdf = pd.DataFrame(bucket_stats)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(bdf["bucket"], bdf["n_images"], color="#3498db", alpha=0.8)
    ax.set_ylabel("Number of images")
    ax.set_title("Failure Bucket: Image Counts")
    ax.tick_params(axis="x", rotation=30)
    for bar, n in zip(bars, bdf["n_images"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(n), ha="center", fontsize=9)
    savefig("11_failure_bucket_counts.png")

    overall_f1 = df["f1"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    colors_b = ["#e74c3c" if f < overall_f1 * 0.7 else "#f39c12" if f < overall_f1 else "#2ecc71"
                for f in bdf["mean_f1"]]
    ax.bar(bdf["bucket"], bdf["mean_f1"], color=colors_b, alpha=0.85)
    ax.axhline(overall_f1, color="black", linestyle="--", lw=1.5, label=f"Overall F1={overall_f1:.3f}")
    ax.set_ylabel("Mean per-image F1")
    ax.set_title("Failure Bucket: Detection Performance")
    ax.set_ylim(0, min(1.0, overall_f1 * 1.8))
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    savefig("12_failure_bucket_f1.png")

    # ------------------------------------------------------------------
    # 13-16: Sample failure grids (worst 4 images per bucket, 2x2)
    # ------------------------------------------------------------------
    # Build lookup structures
    img_id_to_info = {img["id"]: img for img in gt_data["images"]}
    gt_by_img = {}
    for ann in gt_data["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)
    pred_by_img = {}
    for p in pred_data:
        pred_by_img.setdefault(p["image_id"], []).append(p)

    # Map filename stem -> image_id
    stem_to_imgid = {Path(img["file_name"]).stem: img["id"] for img in gt_data["images"]}

    grid_buckets = [
        ("dark",          "dark",          "Worst Dark Images (brightness < 0.15)"),
        ("blurry",        "blurry",        "Worst Blurry Images (sharpness < 50)"),
        ("low_contrast",  "low_contrast",  "Worst Low-Contrast Images"),
        ("small_objects", "small_objects", "Worst Small-Object Images"),
    ]

    for fig_num, (bucket_key, _, title) in enumerate(grid_buckets, start=13):
        metric, op, thresh = FAILURE_BUCKETS[bucket_key]
        mask = _bucket_mask(df, metric, op, thresh)
        worst = df[mask].nsmallest(4, "f1")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax, (_, row) in zip(axes.flat, worst.iterrows()):
            stem = row["filename"]
            img_id = stem_to_imgid.get(stem)
            try:
                img_info = img_id_to_info[img_id]
                img_bgr = _load_image(img_info["file_name"])
                gt_boxes = gt_by_img.get(img_id, [])
                pred_boxes = pred_by_img.get(img_id, [])
                tp_ids = _match_predictions(gt_boxes, pred_boxes)
                img_rgb = _draw_boxes(img_bgr, gt_boxes, pred_boxes, tp_ids, class_names)
            except Exception:
                img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

            ax.imshow(img_rgb)
            ax.axis("off")
            # Metric info overlay
            src = row.get("source", "?")
            m_val = row.get(metric, 0)
            f1_val = row.get("f1", 0)
            tp, fp, fn = int(row.get("n_tp", 0)), int(row.get("n_fp", 0)), int(row.get("n_fn", 0))
            ax.set_title(
                f"[{src}] {metric}={m_val:.3f}  F1={f1_val:.3f}\n"
                f"TP={tp} FP={fp} FN={fn}",
                fontsize=8, pad=4,
            )

        legend_patches = [
            mpatches.Patch(color=(0, 200 / 255, 0), label="GT box"),
            mpatches.Patch(color=(200 / 255, 120 / 255, 0), label="TP prediction"),
            mpatches.Patch(color=(0, 0, 220 / 255), label="FP prediction"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9)
        savefig(f"{fig_num}_{bucket_key}_failures.png")

    # ------------------------------------------------------------------
    # 17: Train vs Test boxplot per metric (or fallback to source boxplot)
    # ------------------------------------------------------------------
    plot_metrics = ["brightness", "contrast_rms", "sharpness", "color_saturation"]
    plot_metrics = [m for m in plot_metrics if m in df.columns]
    if train_df is not None:
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4 * len(plot_metrics), 5))
        if len(plot_metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, plot_metrics):
            if m not in train_df.columns:
                continue
            data = [train_df[m].dropna().values, df[m].dropna().values]
            bp = ax.boxplot(data, patch_artist=True, labels=["Train", "Test"])
            colors_bp = ["#2ecc71", "#e74c3c"]
            for patch, c in zip(bp["boxes"], colors_bp):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_title(m, fontsize=9)
        fig.suptitle("Image Quality: Train vs Test", fontweight="bold")
    else:
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4 * len(plot_metrics), 5))
        if len(plot_metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, plot_metrics):
            data = [df[df["source"] == s][m].dropna().values for s in sources]
            bp = ax.boxplot(data, patch_artist=True, labels=sources)
            for patch, src in zip(bp["boxes"], sources):
                patch.set_facecolor(SOURCE_COLORS.get(src, "#888"))
                patch.set_alpha(0.7)
            ax.set_title(m, fontsize=9)
            ax.tick_params(axis="x", rotation=20)
        fig.suptitle("Image Quality by Source (test only)", fontweight="bold")
    savefig("17_train_vs_test_boxplot.png")

    # ------------------------------------------------------------------
    # 18: Noise level vs F1
    # ------------------------------------------------------------------
    if "noise_level" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        for src in sources:
            sub = df[df["source"] == src]
            ax.scatter(sub["noise_level"], sub["f1"], alpha=0.4, s=15,
                       label=src, color=SOURCE_COLORS.get(src, "#888"))
        ax.set_xlabel("Noise level (MAD estimate)")
        ax.set_ylabel("Per-image F1")
        ax.set_title("Noise Level vs Detection Performance")
        ax.legend()
        savefig("18_noise_vs_f1.png")

    # ------------------------------------------------------------------
    # 19: Distribution shift bar chart (normalised mean difference per metric)
    # Shows which quality dimensions differ most between train and test
    # ------------------------------------------------------------------
    if train_df is not None:
        shift_metrics = [m for m in QUALITY_METRICS if m in df.columns and m in train_df.columns
                         and m not in ("n_gt_boxes",)]
        shifts = []
        for m in shift_metrics:
            train_vals = train_df[m].dropna()
            test_vals = df[m].dropna()
            if train_vals.std() > 1e-6:
                # Standardised mean difference (Cohen's d-style)
                pooled_std = np.sqrt((train_vals.var() + test_vals.var()) / 2)
                d = (test_vals.mean() - train_vals.mean()) / pooled_std if pooled_std > 1e-6 else 0
            else:
                d = 0
            shifts.append({"metric": m, "shift": round(d, 3),
                           "train_mean": round(train_vals.mean(), 4),
                           "test_mean": round(test_vals.mean(), 4)})
        sdf = pd.DataFrame(shifts).sort_values("shift", key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(9, max(4, len(sdf) * 0.45)))
        colors_s = ["#e74c3c" if v < 0 else "#3498db" for v in sdf["shift"]]
        ax.barh(sdf["metric"], sdf["shift"], color=colors_s, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Standardised mean difference (Test − Train)\n"
                      "Positive = test has higher values; Negative = train has higher")
        ax.set_title("Distribution Shift: Test vs Train\n"
                     "Large |d| = test images are systematically different on this dimension")
        savefig("19_distribution_shift.png")
    else:
        # Fallback: dynamic range distribution by source
        if "dynamic_range" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            for src in sources:
                vals = df[df["source"] == src]["dynamic_range"].dropna()
                ax.hist(vals, bins=30, alpha=0.6, label=src, color=SOURCE_COLORS.get(src, "#888"))
            ax.set_xlabel("Dynamic range (max-min luminance)")
            ax.set_ylabel("Images")
            ax.set_title("Dynamic Range Distribution by Source")
            ax.legend()
            savefig("19_dynamic_range_dist.png")

    # ------------------------------------------------------------------
    # 20: 2x2 overview panel — brightness / contrast / sharpness / saturation vs F1
    # ------------------------------------------------------------------
    panel_metrics = [
        ("brightness",       "Brightness"),
        ("contrast_rms",     "RMS Contrast"),
        ("sharpness",        "Sharpness"),
        ("color_saturation", "Saturation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Key Quality Metrics vs Detection F1 (all sources)", fontsize=13, fontweight="bold")
    for ax, (m, label) in zip(axes.flat, panel_metrics):
        if m not in df.columns:
            continue
        for src in sources:
            sub = df[df["source"] == src]
            ax.scatter(sub[m], sub["f1"], alpha=0.35, s=12,
                       label=src, color=SOURCE_COLORS.get(src, "#888"))
        valid = df[[m, "f1"]].dropna()
        if len(valid) > 10:
            z = np.polyfit(valid[m], valid["f1"], 1)
            xr = np.linspace(valid[m].min(), valid[m].max(), 100)
            ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1.2, alpha=0.6)
        rho, _ = spearmanr(valid[m], valid["f1"]) if len(valid) > 5 else (0, 1)
        ax.set_xlabel(label)
        ax.set_ylabel("F1")
        ax.set_title(f"{label}  (ρ={rho:.2f})", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    savefig("20_metrics_overview_panel.png")

    print(f"All 20 visualisations saved to {vis_dir}")


# ---------------------------------------------------------------------------
# Failure analysis markdown report
# ---------------------------------------------------------------------------

def write_failure_report(
    df: pd.DataFrame,
    out_path: Path,
    train_df: pd.DataFrame = None,
) -> None:
    corr_df = _correlation_table(df)
    overall_f1 = df["f1"].mean()
    overall_prec = df["precision"].mean()
    overall_rec = df["recall"].mean()

    train_note = f" | **Train images:** {len(train_df)}" if train_df is not None else ""
    lines = [
        "# Failure Analysis Report\n",
        f"**Test images:** {len(df)} | "
        f"**Overall F1:** {overall_f1:.3f} | "
        f"**Precision:** {overall_prec:.3f} | "
        f"**Recall:** {overall_rec:.3f}{train_note}\n",
        "---\n",
        "## Strongest Predictors of Failure (Spearman ρ with per-image F1)\n",
        "Computed on test images only. Positive ρ = metric improves performance.\n",
        "| Metric | Spearman ρ | p-value |",
        "| ------ | ---------- | ------- |",
    ]
    for _, row in corr_df.iterrows():
        lines.append(f"| {row['metric']} | {row['spearman_rho']:+.3f} | {row['p_value']:.4f} |")
    lines.append("")

    # Distribution shift section — only when train data is available
    if train_df is not None:
        shift_metrics = [m for m in QUALITY_METRICS if m in df.columns and m in train_df.columns
                         and m not in ("n_gt_boxes",)]
        lines += [
            "## Distribution Shift: Train vs Test\n",
            "Standardised mean difference (Cohen's d). "
            "Large |d| means the test set is systematically different from training on this dimension — "
            "these are your domain gap dimensions.\n",
            "| Metric | Train mean | Test mean | d (shift) | Interpretation |",
            "| ------ | ---------- | --------- | --------- | -------------- |",
        ]
        for m in shift_metrics:
            t_vals = train_df[m].dropna()
            te_vals = df[m].dropna()
            pooled = np.sqrt((t_vals.var() + te_vals.var()) / 2)
            d = (te_vals.mean() - t_vals.mean()) / pooled if pooled > 1e-6 else 0.0
            magnitude = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "LARGE"
            direction = "test > train" if d > 0 else "test < train"
            lines.append(f"| {m} | {t_vals.mean():.3f} | {te_vals.mean():.3f} | "
                         f"{d:+.2f} | {magnitude} ({direction}) |")
        lines.append("")

    lines += ["## Failure Buckets\n",
              "| Bucket | Condition | N images | Mean F1 | vs Overall |",
              "| ------ | --------- | -------- | ------- | ---------- |"]
    for bucket, (metric, op, thresh) in FAILURE_BUCKETS.items():
        mask = _bucket_mask(df, metric, op, thresh)
        n = mask.sum()
        mean_f1 = df[mask]["f1"].mean() if n > 0 else 0.0
        delta = (mean_f1 - overall_f1) / overall_f1 * 100 if overall_f1 > 0 else 0
        cond = f"{metric} {'<' if op == 'lt' else '>'} {thresh}"
        lines.append(f"| {bucket} | {cond} | {n} | {mean_f1:.3f} | {delta:+.1f}% |")
    lines.append("")

    lines += ["## Worst 10 Images (lowest F1)\n",
              "| File | Source | F1 | Brightness | Sharpness | Contrast | TP | FP | FN |",
              "| ---- | ------ | -- | ---------- | --------- | -------- | -- | -- | -- |"]
    worst = df.nsmallest(10, "f1")
    for _, row in worst.iterrows():
        lines.append(
            f"| {row['filename']} | {row.get('source','?')} | {row['f1']:.3f} | "
            f"{row.get('brightness',0):.3f} | {row.get('sharpness',0):.1f} | "
            f"{row.get('contrast_rms',0):.3f} | "
            f"{int(row.get('n_tp',0))} | {int(row.get('n_fp',0))} | {int(row.get('n_fn',0))} |"
        )
    lines.append("")
    lines.append("## Visualisations\n")
    lines.append("See `visualisations/` directory. 20 plots generated:\n")
    vis_descriptions = [
        ("01", "brightness_dist", "Brightness — Train vs Test distribution"),
        ("02", "sharpness_dist", "Sharpness — Train vs Test distribution"),
        ("03", "contrast_rms_dist", "RMS contrast — Train vs Test distribution"),
        ("04", "color_saturation_dist", "Colour saturation — Train vs Test distribution"),
        ("05", "brightness_vs_f1", "Brightness vs per-image F1 scatter"),
        ("06", "sharpness_vs_f1", "Sharpness vs F1 scatter"),
        ("07", "contrast_rms_vs_f1", "RMS contrast vs F1 scatter"),
        ("08", "mean_obj_bg_contrast_vs_f1", "Object-background contrast vs F1"),
        ("09", "mean_obj_size_px_vs_f1", "Mean object size vs F1"),
        ("10", "correlation_heatmap", "Spearman correlation — all metrics vs F1"),
        ("11", "failure_bucket_counts", "Failure bucket image counts"),
        ("12", "failure_bucket_f1", "Failure bucket mean F1"),
        ("13", "dark_failures", "Worst dark image samples (2×2 grid)"),
        ("14", "blurry_failures", "Worst blurry image samples (2×2 grid)"),
        ("15", "low_contrast_failures", "Worst low-contrast samples"),
        ("16", "small_objects_failures", "Worst small-object samples"),
        ("17", "train_vs_test_boxplot", "Train vs Test quality boxplot per metric"),
        ("18", "noise_vs_f1", "Noise level vs F1 scatter"),
        ("19", "distribution_shift", "Distribution shift (Cohen's d) per metric — train vs test"),
        ("20", "metrics_overview_panel", "2×2 summary — top metrics vs F1"),
    ]
    for num, slug, desc in vis_descriptions:
        lines.append(f"- `{num}_{slug}.png` — {desc}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Failure report written to {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_failure_analysis(
    metrics_csv: str,
    predictions_json: str,
    gt_json: str,
    out_dir: str,
    train_metrics_csv: str = None,
) -> None:
    """
    metrics_csv:       per-image test metrics from eval.py
    train_metrics_csv: optional path to data/splits/<split>/split_metrics.csv
                       (generated by compute_split_metrics.py — run once per split)
                       When provided, enables train vs test distribution comparisons.
    """
    out_path = Path(out_dir)
    df = pd.read_csv(metrics_csv)

    train_df = None
    if train_metrics_csv and Path(train_metrics_csv).exists():
        full_split_df = pd.read_csv(train_metrics_csv)
        train_df = full_split_df[full_split_df["split"] == "train"].copy()
        print(f"Loaded train metrics: {len(train_df)} images from {train_metrics_csv}")
    else:
        print("No train metrics CSV found — distribution comparison will be skipped.")
        print("Run: python pipeline/eval/compute_split_metrics.py --split <split_name>")

    with open(predictions_json) as f:
        pred_data = json.load(f)
    with open(gt_json) as f:
        gt_data = json.load(f)

    class_names = [c["name"] for c in sorted(gt_data["categories"], key=lambda x: x["id"])]

    write_failure_report(df, out_path / "failure_analysis.md", train_df=train_df)
    generate_visualisations(df, out_path, gt_data, pred_data, class_names, train_df=train_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-metrics", default=None,
                        help="Path to split_metrics.csv for train vs test comparison")
    args = parser.parse_args()
    run_failure_analysis(args.metrics, args.predictions, args.gt, args.out,
                         train_metrics_csv=args.train_metrics)
