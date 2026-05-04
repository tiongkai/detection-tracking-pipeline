# Tracker Evaluation Runs

## Scripts

Two tracking scripts exist in `pipeline/track/`:

### `track_video.py` — Standard tracking

Runs YOLO detection on **every frame** and feeds detections to HybridSORT. Only outputs bounding boxes for tracks that are actively matched to a detection. When the detector misses an object, no box is drawn — the track is maintained internally by the tracker but not visualised.

```bash
conda run -n boat-tracker python pipeline/track/track_video.py \
    --weights results/yolo26l_split_v7_original_classes/weights/best.pt \
    --source data/eval/clips --out results/tracker/output \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0
```

### `track_video_predict.py` — Tracking with Kalman prediction for missed detections

Same detection + tracking pipeline, but **also draws Kalman-predicted bounding boxes** when the detector misses an object. This bridges detection gaps so tracked objects remain visible even when the model fails to detect them on some frames.

How it works:
1. After `tracker.update()`, inspects all active tracks across all per-class track lists
2. Tracks matched to a detection → **solid bounding box** (ground truth from detector)
3. Tracks with no detection match (coasting) → **dashed bounding box** labelled `[predicted]`, using the Kalman filter's predicted position
4. When a detection reappears and re-matches the same track ID, the box snaps back to the real detection immediately

Additional features:
- `--det-interval N`: run detection every N frames (default 1). On skipped frames, all tracks coast on Kalman predictions. Useful to test KF quality or reduce compute.
- `--max-coast N`: hide predicted boxes after N frames without a detection match (default 10). Prevents stale predictions from drifting across the frame. The track stays alive internally for `max_age` frames for ReID re-matching.
- `--coast-classes`: only show Kalman predictions for specific classes (substring match). E.g. `--coast-classes boat` limits predictions to boat-rgb and boat-thermal, while other classes only show when detected.
- `--enable-nms`: apply cross-modal NMS between detection and tracker update (see [Interclass NMS](#interclass-nms) below).
- `--nms-iou-thresh`: IoU threshold for cross-modal NMS suppression (default 0.5).

```bash
# Every frame, boat-only Kalman predictions, 10-frame coast limit
conda run -n boat-tracker python pipeline/track/track_video_predict.py \
    --weights results/yolo26l_split_v7_original_classes/weights/best.pt \
    --source data/eval/clips --out results/tracker/output_predict \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0 \
    --det-interval 1 --max-coast 10 --coast-classes boat

# Detection every 2nd frame (test Kalman filter performance)
conda run -n boat-tracker python pipeline/track/track_video_predict.py \
    --weights results/yolo26l_split_v7_original_classes/weights/best.pt \
    --source data/eval/clips --out results/tracker/output_predict_skip \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0 \
    --det-interval 2 --max-coast 10 --coast-classes boat

# With cross-modal NMS enabled
conda run -n boat-tracker python pipeline/track/track_video_predict.py \
    --weights results/yolo26l_split_v7_original_classes/weights/best.pt \
    --source data/eval/pt80_clips --out results/tracker/output_nms \
    --conf 0.3 --iou 0.5 --ema-alpha 1.0 \
    --max-coast 10 --enable-nms --nms-iou-thresh 0.5
```

**Note:** when `--det-interval > 1`, `min_hits` is automatically lowered from 3 to 1 so tracks can be confirmed between detection frames.

### Interclass NMS

The 12-class domain-split model can fire both RGB and thermal class variants on the same object (e.g. `boat-rgb` + `boat-thermal` overlapping). YOLO's built-in NMS is per-class, so these pass through as separate detections. This creates duplicate tracker IDs for the same physical object.

**Where it runs:**
- **Tracking** (`track_video_predict.py`): between detection and `tracker.update()`, via `--enable-nms`. The tracker only sees deduplicated detections.
- **Evaluation** (`pipeline/eval/eval.py`): after inference, before metric computation, via `--enable-interclass-nms`. Affects mAP scores.

**How it works:**
1. `build_class_groups()` strips `-rgb`/`-thermal` suffixes to find cross-modal pairs: `boat-rgb` (id 0) + `boat-thermal` (id 6) → group `"boat": {0, 6}`
2. Within each group, detections are sorted by confidence descending
3. Lower-confidence boxes overlapping above `iou_thresh` are suppressed
4. Classes without a cross-modal pair (e.g. `human head`) are untouched

**Implementation:** `pipeline/track/cross_modal_nms.py` and `pipeline/eval/cross_modal_nms.py` (same module, copied for import convenience).

**Eval pipeline usage:**
```bash
python pipeline/eval/eval.py \
    --config pipeline/configs/exp_yolo26l_v3.yaml \
    --enable-interclass-nms --nms-iou-thresh 0.5
```

**Effectiveness on pt80 clips (IoU thresh 0.5):**
- Frames with both `boat-rgb` + `boat-thermal`: 2,807 → 797 (72% reduction)
- Total boat detections suppressed: 3,468
- Single-boat clips: 100% elimination of duplicate class detections
- Multi-boat clips: partial — remaining duals are separate boats where one is RGB and another is thermal (legitimate), or boxes offset enough that IoU < 0.5

---

## Run History

All runs use **HybridSORT** with `clip_veri.pt` (vehicle ReID), `per_class=True`, `det_thresh=0.3`, `iou_threshold=0.15`.

### Early runs (track_video.py)

| Run | Model | Split | Classes | max_age | EMA alpha | min_hits | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| split_v5 | YOLO26-L | split_v5_merged | 6 (merged) | 30 (1s) | 0.5 | 3 | 2026-04-22 |
| split_v6_merged | YOLO26-L | split_v6_merged | 6 (merged) | 30 (1s) | 0.5 | 3 | 2026-04-23 |
| split_v6_merged_age_90 | YOLO26-L | split_v6_merged | 6 (merged) | 90 (3s) | 0.5 | 3 | 2026-04-23 |
| split_v6_original_classes_age_90 | YOLO26-L | split_v6_original_classes | 12 (domain-split) | 90 (3s) | 0.5 | 3 | 2026-04-23 |
| split_v6_original_classes_age_180_no_ema | YOLO26-L | split_v6_original_classes | 12 (domain-split) | 180 (6s) | 1.0 (off) | 3 | 2026-04-23 |
| yolo26l_split_v7_merged | YOLO26-L | split_v7_merged | 6 (merged) | 180 (6s) | 1.0 (off) | 3 | 2026-04-27 |
| yolo26l_split_v7_original_classes | YOLO26-L | split_v7_original_classes | 12 (domain-split) | 180 (6s) | 1.0 (off) | 3 | 2026-04-27 |

### Kalman prediction runs (track_video_predict.py)

These runs use enhanced ReID settings and Kalman-predicted boxes for missed detections.

| Run | det_interval | max_coast | coast_classes | Date |
| --- | --- | --- | --- | --- |
| tmp/predict_every_frame | 1 | 10 | boat | 2026-04-28 |
| tmp/predict_every_2nd_frame | 2 | 10 | boat | 2026-04-28 |
| tmp/eval (clips + pt80_clips) | 1 | 10 | boat | 2026-04-28 |
| tmp/frame-by-frame-nms (all eval) | 1 | 10 | all | 2026-05-04 |

All Kalman prediction runs use: YOLO26-L `split_v7_original_classes`, `max_age=180`, `ema_alpha=1.0`.

### Interclass NMS runs

| Run | enable_nms | nms_iou_thresh | Source | Clips | Date |
| --- | --- | --- | --- | --- | --- |
| tmp/frame-by-frame-nms/FishingBoat.mp4 | Yes | 0.5 | FishingBoat | 1 | 2026-05-04 |
| tmp/frame-by-frame-nms/clips | Yes | 0.5 | eval/clips | 27 | 2026-05-04 |
| tmp/frame-by-frame-nms/pt80_clips | Yes | 0.5 | eval/pt80_clips | 54 | 2026-05-04 |

## Eval Data

| Source | Location | Clips |
| --- | --- | --- |
| clips | data/eval/clips/ | 27 team eval clips (thermal + RGB) |
| pt80_clips | data/eval/pt80_clips/ | 52 pt80 camera clips |
| standalone | data/eval/FishingBoat.mp4 | 1 RGB fishing boat clip |

## Tracker Configuration

### HybridSORT — track_video.py (standard)

| Parameter | Value | Notes |
| --- | --- | --- |
| reid_weights | clip_veri.pt | CLIP backbone trained on VeRi-776 (vehicle ReID) |
| per_class | True | Separate track IDs per class |
| use_custom_kf | True | 9D Kalman filter (u,v,s,c,r + velocities) |
| det_thresh | 0.3 | Detection confidence threshold |
| iou_threshold | 0.15 | IoU threshold for association |
| min_hits | 3 | Minimum detections before track is confirmed |
| max_age | 180 | 6s ReID recovery window at 30fps |
| alpha | 0.9 | ReID feature EMA (default) |
| longterm_bank_length | 30 | Past features stored per track (default) |
| longterm_reid_weight | 0.0 | Long-term ReID disabled (default) |

### HybridSORT — track_video_predict.py (enhanced ReID)

| Parameter | Value | Notes |
| --- | --- | --- |
| reid_weights | clip_veri.pt | Same ReID model |
| per_class | True | Separate track IDs per class |
| use_custom_kf | True | 9D Kalman filter |
| det_thresh | 0.3 | Detection confidence threshold |
| iou_threshold | 0.15 | IoU threshold for association |
| min_hits | 3 (or 1 if det_interval > 1) | Auto-adjusted for frame skipping |
| max_age | 180 | 6s ReID recovery window |
| alpha | 0.7 | Faster appearance adaptation (was 0.9) |
| longterm_bank_length | 150 | 5s of features stored (was 30) |
| longterm_reid_weight | 0.25 | Long-term ReID enabled (was 0.0) |
| longterm_reid_correction_thresh | 0.5 | More lenient re-matching (was 0.4) |
| longterm_reid_correction_thresh_low | 0.5 | Same for low-score BYTE step (was 0.4) |

### ReID Parameters Reference

| Parameter | Description |
| --- | --- |
| max_age | Frames a track stays alive without detection — sets the ReID recovery window |
| alpha | ReID feature EMA weight. Lower = adapts faster to appearance changes |
| longterm_bank_length | Number of past appearance features stored per track for long-term matching |
| longterm_reid_weight | Weight of long-term feature bank in association cost. 0 = disabled |
| longterm_reid_correction_thresh | Max cosine distance for long-term ReID to correct a match |
| EG_weight_high_score | Weight of short-term ReID in first association step (default 4.6) |
| EG_weight_low_score | Weight of short-term ReID in BYTE low-score association (default 1.3) |

### EMA Box Smoothing

| alpha | Behaviour |
| --- | --- |
| 0.5 | 50% current + 50% history — smooth but introduces lag on moving objects |
| 1.0 | No smoothing — raw tracker output, no lag |

### Detection

| Parameter | Value |
| --- | --- |
| conf | 0.3 |
| iou (NMS) | 0.5 |
| imgsz | 640 (YOLO default) |

## Key Observations

- **split_v5 → split_v6_merged**: v6 has offline augmentation (RGB-only, 6x expansion), v5 has none. Visual tracking quality improved.
- **max_age 30 → 180**: Longer ReID window helps recover tracks after occlusion. Reduces ID switches on boats passing behind structures.
- **EMA 0.5 → 1.0**: Removing EMA eliminates box lag on moving targets. Trade-off: slightly more jitter frame-to-frame, but positions are more accurate.
- **6-class merged vs 12-class domain-split**: The 12-class model (v6_original_classes) had higher val mAP (0.80 vs 0.73) during training.
- **Kalman prediction for missed detections**: Drawing KF-predicted boxes bridges short detection gaps. `max_coast=10` limits drift; predictions are visually distinguished with dashed outlines.
- **Enhanced ReID (longterm_reid_weight 0.0 → 0.25)**: Enabling long-term feature matching improves track ID consistency when objects leave and reappear. The feature bank (150 frames) provides a robust average appearance for re-matching.
- **det_interval=2**: Running detection every other frame roughly halves GPU compute while Kalman prediction fills the gaps. Quality depends on object speed and scene complexity.
- **Cross-modal NMS**: On pt80 twilight clips, the 12-class model fires both `boat-rgb` and `boat-thermal` on ~10% of frames. NMS at IoU 0.5 eliminates 72% of dual-class frames and suppresses 3,468 duplicate boat detections. Remaining duals are mostly multi-boat scenes with legitimate separate objects, or boxes offset enough that IoU < 0.5.
- **Interclass NMS in eval pipeline**: Adding `--enable-interclass-nms` to `eval.py` lets you measure how cross-modal duplicates affect mAP scores. Running with vs without shows the precision impact of duplicate detections on the same object.
