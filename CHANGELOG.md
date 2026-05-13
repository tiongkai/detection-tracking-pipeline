# Changelog

## Class-aware matching in tracking eval (2026-05-13)

### Problem

`eval/eval_tracking.py` matched GT to predictions purely by IoU, ignoring class labels. A `boat-rgb` prediction could match a `human-thermal` GT entry if their boxes overlapped above the IoU threshold. In maritime scenes with overlapping boat/human detections (e.g. a person standing on a boat), this silently corrupts metrics.

### Solution

Added class compatibility check to `match_frames()`. Before Hungarian assignment, pairs with incompatible classes are blocked (`cost = 1e6`). Compatibility uses the same base-name grouping as cross-modal NMS — strips `-rgb`/`-thermal` suffixes and compares. So `boat-rgb` can still match `boat-thermal` GT (same object type, different domain), but `boat` can never match `human`.

### Files

#### `eval/eval_tracking.py` (MODIFIED)

- `CLASS_NAMES` — 12-class domain-split name map
- `_base_class(cls_id)` — strips domain suffix to get base object type
- `classes_compatible(cls_a, cls_b)` — returns True if same base type
- `match_frames()` — now extracts `gt_classes` and `pred_classes` per frame, sets `cost[r, c] = 1e6` for incompatible pairs before `linear_sum_assignment`

All metrics (MOTA, IDF1, HOTA, ID switches, fragmentation, MT/ML) flow through `match_frames`, so every metric benefits from this fix. `compute_hota` calls `match_frames` at each IoU threshold and inherits the fix automatically.

---

## Cross-modal NMS (2026-05-04)

### Problem

The 12-class domain-split model (YOLO26l, split_v7) fires both RGB and thermal class variants on the same object in ambiguous lighting — e.g. `boat-rgb` (conf 0.72) and `boat-thermal` (conf 0.65) overlapping on the same boat. YOLO's built-in NMS is per-class, so these pass through as separate detections, creating duplicate tracker IDs for the same physical object.

Observed on ~10% of pt80 twilight/dusk frames (2,807 frames with both `boat-rgb` and `boat-thermal`), and 122 frames in eval/clips.

### Solution

Greedy NMS applied across model classes that map to the same real-world object, run between detection and tracker update.

### Files

#### `pipeline/track/cross_modal_nms.py` (NEW)

- `cross_modal_nms(detections, class_groups, iou_thresh)` — takes (N,6) detection array `[x1,y1,x2,y2,conf,cls]` and a dict mapping base class name to set of model class IDs.
- Within each group, sorts detections by confidence descending; suppresses lower-confidence boxes with IoU > threshold.
- Classes without a cross-modal pair (e.g. `human head`, no thermal counterpart) are left untouched.

#### `pipeline/track/track_video_predict.py` (MODIFIED)

- `build_class_groups(class_names)` — auto-groups model classes by stripping `-rgb`/`-thermal` suffix. E.g. `boat-rgb` (id 0) + `boat-thermal` (id 6) → group `"boat": {0, 6}`.
- NMS runs after detection, before `tracker.update()`, so the tracker only sees deduplicated detections.
- New CLI arguments:
  - `--enable-nms` — enable cross-modal NMS (off by default)
  - `--nms-iou-thresh` — IoU threshold for suppression (default 0.5)

### Usage

```bash
conda run -n boat-tracker python pipeline/track/track_video_predict.py \
    --weights results/yolo26l_split_v7_original_classes/weights/best.pt \
    --source data/eval/pt80_clips \
    --out ./tmp/frame-by-frame-nms/pt80_clips \
    --enable-nms --nms-iou-thresh 0.5 \
    --conf 0.3 --max-coast 10
```
