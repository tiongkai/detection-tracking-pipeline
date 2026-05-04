# Evaluation Protocols

## Available Scripts

### 1. `eval.py` — Full Experiment Evaluation (COCO format)

Config-driven evaluation for experiments with COCO JSON ground truth. Runs inference, computes mAP, per-source/domain/class breakdowns, per-image quality metrics, and failure analysis with 20 visualisations.

**Requires**: experiment config YAML, COCO JSON annotations, split manifest CSV.

```bash
python pipeline/eval/eval.py --config pipeline/configs/exp_yolo11l.yaml
```

**Outputs** to `results/<experiment_name>/eval/`:

| File | Description |
| --- | --- |
| `metrics.json` | All metrics (overall, per-class, per-source, per-domain) |
| `metrics_report.md` | Human-readable tables |
| `predictions.json` | Raw predictions in COCO format |
| `image_metrics.csv` | Per-image quality + detection performance |
| `failure_analysis.md` | Correlation table, failure buckets, worst cases |
| `visualisations/` | 20 PNG plots (see failure_analysis.py for descriptions) |

### 2. `compare_models.py` — Multi-Model Comparison (YOLO format)

Compare multiple YOLO models on the same holdout set. Works with standard YOLO datasets (images/ + labels/ with .txt labels and a data.yaml).

```bash
python pipeline/eval/compare_models.py \
    --models model_a.pt model_b.pt model_c.pt \
    --data /path/to/data.yaml \
    --conf 0.5 \
    --iou 0.5 \
    --output /path/to/eval_output/ \
    --device 0
```

**Arguments**:

| Arg | Default | Description |
| --- | --- | --- |
| `--models` | (required) | One or more .pt model paths |
| `--data` | (required) | Path to ultralytics data.yaml |
| `--conf` | 0.5 | Confidence threshold for FP/FN visual extraction |
| `--iou` | 0.5 | IoU threshold for TP/FP/FN matching |
| `--output` | (required) | Output directory |
| `--device` | 0 | GPU device index |

**How it works**:

1. **mAP50 / mAP50-95**: Runs `model.val()` per model (standard COCO eval protocol — sweeps all confidence thresholds internally)
2. **FP/FN extraction**: Runs `model.predict()` at the fixed `--conf` threshold, computes per-image TP/FP/FN via greedy IoU matching, draws annotated images
3. **Cross-model comparison**: Identifies images where models disagree (one detects, another misses)

**Outputs**:

```
output_dir/
  <model_name>/
    false_positives/    images with FP boxes (red)
    false_negatives/    images with missed GT (yellow)
    all_annotated/      all images with GT(green)/TP(blue)/FP(red)/FN(yellow)
    per_image.csv       filename, n_gt, n_tp, n_fp, n_fn, precision, recall, f1
  comparison.csv        per-image metrics for all models side by side
  disagreements/        images where one model detects but another misses
  summary.md            mAP table + aggregate FP/FN counts
```

### 3. `failure_analysis.py` — Failure Analysis + Visualisations

Can be run standalone or is called by eval.py. Generates 20 visualisation plots and a failure report from per-image metrics.

```bash
python pipeline/eval/failure_analysis.py \
    --metrics results/<exp>/eval/image_metrics.csv \
    --predictions results/<exp>/eval/predictions.json \
    --gt data/splits/<split>/coco/test/_annotations.coco.json \
    --out results/<exp>/eval/ \
    --train-metrics data/splits/<split>/split_metrics.csv
```

### 4. `compute_split_metrics.py` — Pre-compute Dataset Quality Metrics

Computes per-image quality metrics (brightness, sharpness, contrast, etc.) for all images in a split. Run once per split; results are cached for train-vs-test distribution comparison in failure analysis.

```bash
python pipeline/eval/compute_split_metrics.py --split split_v1_baseline
```

### 5. `image_metrics.py` — Per-Image Metric Functions (library)

Not a standalone script. Provides:

- `compute_image_metrics(img_bgr)` — brightness, contrast, sharpness, noise, etc.
- `compute_object_metrics(img_bgr, gt_boxes)` — object size, contrast with background
- `compute_detection_performance(gt_boxes, preds)` — TP/FP/FN counts at IoU threshold

## Metric Definitions

### Detection Metrics (Overall)

| Metric | Description |
| --- | --- |
| mAP50 | Mean average precision at IoU threshold 0.5 |
| mAP50-95 | Mean AP averaged across IoU thresholds 0.5, 0.55, ..., 0.95 |

**mAP** is computed via pycocotools COCOeval (sweeps all confidence thresholds to build the precision-recall curve). This is separate from fixed-confidence FP/FN extraction used for per-image analysis.

---

### Per-Image Detection Performance

Computed by greedy matching of predictions to ground truth at IoU >= 0.5. Used as the target variable (F1) in failure analysis.

| Metric | Description |
| --- | --- |
| `n_tp` | True positives — predictions matched to a GT box (IoU >= 0.5, same class) |
| `n_fp` | False positives — predictions with no matching GT box |
| `n_fn` | False negatives — GT boxes with no matching prediction |
| `precision` | TP / (TP + FP) — how many detections are correct |
| `recall` | TP / (TP + FN) — how many objects are found |
| `f1` | 2 * precision * recall / (precision + recall) — primary per-image performance score |

Matching is greedy: predictions sorted by confidence descending, each matched to the highest-IoU unmatched GT of the same class.

---

### Image-Level Quality Metrics

Computed from raw pixel values of each test image. These characterise the visual difficulty of the image independent of the model.

| Metric | Description | Computation |
| --- | --- | --- |
| `brightness` | Mean luminance (0-1) | Mean of grayscale image normalised to [0, 1] |
| `contrast_rms` | RMS contrast | Standard deviation of grayscale luminance |
| `sharpness` | Edge sharpness (higher = sharper) | Variance of Laplacian filter response |
| `noise_level` | Estimated sensor noise sigma | Median absolute deviation of high-frequency residual (image - Gaussian blur), scaled by 1.4826 to estimate sigma |
| `dark_pixel_ratio` | Fraction of very dark pixels | Proportion of pixels with luminance < 30/255 |
| `overexposed_ratio` | Fraction of blown-out pixels | Proportion of pixels with luminance > 240/255 |
| `dynamic_range` | Luminance range (0-1) | max(gray) - min(gray) |
| `edge_density` | Texture/detail density | Canny edge pixels / total pixels (thresholds: 50, 150) |
| `color_saturation` | Colour richness (approx 0 for thermal/grayscale) | Mean of HSV saturation channel, normalised to [0, 1] |
| `is_grayscale` | Whether image is effectively monochrome | True if std(R-G) < 5 and std(G-B) < 5 |
| `color_cast` | Colour channel imbalance | Max absolute deviation of per-channel means from overall mean |
| `img_width` | Image width in pixels | — |
| `img_height` | Image height in pixels | — |

---

### Object-Level Quality Metrics

Computed using ground-truth bounding boxes overlaid on the image. These capture how hard the objects are to detect based on their visual properties.

| Metric | Description | Computation |
| --- | --- | --- |
| `n_gt_boxes` | Number of ground-truth objects in image | Count of GT annotations |
| `mean_obj_size_px` | Mean object area in pixels | Average of (width * height) across all GT boxes |
| `mean_obj_occupancy` | Mean object area as fraction of image | Average of (box area / image area) |
| `mean_obj_brightness` | Mean luminance inside object regions | Average grayscale intensity within GT box crops |
| `mean_obj_bg_contrast` | Object-to-background contrast | For each GT box: abs(mean(object pixels) - mean(10px border strip)), averaged across boxes |
| `mean_box_overlap` | Mean pairwise IoU between GT boxes | Average IoU of all GT box pairs; high values = occlusion/crowding |

---

### Failure Buckets

Predefined thresholds that bin images into interpretable failure categories. Each bucket isolates a specific visual difficulty.

| Bucket | Condition | Rationale |
| --- | --- | --- |
| `dark` | brightness < 0.15 | Night/low-light; objects lack contrast |
| `blurry` | sharpness < 50 | Motion blur or defocus; edges are weak |
| `low_contrast` | contrast_rms < 0.06 | Flat scenes (fog, thermal wash); foreground/background blend |
| `noisy` | noise_level > 0.03 | High sensor noise; small objects lost in noise texture |
| `small_objects` | mean_obj_size_px < 800 | ~28x28 px or smaller; limited feature information |
| `low_obj_contrast` | mean_obj_bg_contrast < 0.05 | Objects are nearly the same luminance as their surroundings |
| `cluttered` | n_gt_boxes > 8 | Many overlapping objects; NMS and crowding challenges |

For each bucket, the failure report shows image count and mean F1 compared to overall F1. A large negative delta (e.g. -40%) means the model systematically struggles in that condition.

---

### Distribution Shift (Train vs Test)

When `split_metrics.csv` is available (generated by `compute_split_metrics.py`), the failure analysis compares the distribution of each quality metric between training and test sets using Cohen's d:

**Cohen's d** = (test_mean - train_mean) / pooled_std

| |d| | Interpretation |
| --- | --- |
| < 0.2 | Negligible shift |
| 0.2 - 0.5 | Small shift |
| 0.5 - 0.8 | Medium shift |
| > 0.8 | **Large shift** — significant domain gap on this dimension |

---

### Correlation Analysis

Spearman rank correlation (rho) between each quality metric and per-image F1, computed on test images only.

- **Positive rho**: higher metric value -> better detection (e.g. sharpness, contrast)
- **Negative rho**: higher metric value -> worse detection (e.g. noise, dark ratio)
- **p-value < 0.05**: statistically significant

The top correlates are the strongest explainers of failure — they tell you which image properties most affect the model.

---

### Visualisations (20 plots)

Generated in `eval/<experiment>/visualisations/`.

| # | Filename | Description |
| --- | --- | --- |
| 01 | `brightness_dist.png` | Brightness distribution — train vs test (or by source) |
| 02 | `sharpness_dist.png` | Sharpness distribution — train vs test |
| 03 | `contrast_rms_dist.png` | RMS contrast distribution — train vs test |
| 04 | `color_saturation_dist.png` | Colour saturation distribution — train vs test |
| 05 | `brightness_vs_f1.png` | Brightness vs per-image F1 scatter (coloured by source) |
| 06 | `sharpness_vs_f1.png` | Sharpness vs F1 scatter |
| 07 | `contrast_rms_vs_f1.png` | RMS contrast vs F1 scatter |
| 08 | `mean_obj_bg_contrast_vs_f1.png` | Object-background contrast vs F1 |
| 09 | `mean_obj_size_px_vs_f1.png` | Mean object size vs F1 |
| 10 | `correlation_heatmap.png` | Spearman rho bar chart — all metrics ranked by correlation with F1 |
| 11 | `failure_bucket_counts.png` | How many images fall into each failure bucket |
| 12 | `failure_bucket_f1.png` | Mean F1 per bucket vs overall F1 baseline |
| 13 | `dark_failures.png` | 2x2 grid of worst dark images with GT/prediction overlays |
| 14 | `blurry_failures.png` | 2x2 grid of worst blurry images |
| 15 | `low_contrast_failures.png` | 2x2 grid of worst low-contrast images |
| 16 | `small_objects_failures.png` | 2x2 grid of worst small-object images |
| 17 | `train_vs_test_boxplot.png` | Boxplot of key metrics — train vs test (or by source) |
| 18 | `noise_vs_f1.png` | Noise level vs F1 scatter |
| 19 | `distribution_shift.png` | Cohen's d bar chart — which dimensions shift most between train and test |
| 20 | `metrics_overview_panel.png` | 2x2 summary panel — brightness, contrast, sharpness, saturation vs F1 |

Box overlays in grid plots (13-16): **green** = GT, **orange** = TP prediction, **blue** = FP prediction.
