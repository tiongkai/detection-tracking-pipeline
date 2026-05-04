# HybridSORT Tracking & ReID — Parameter Reference

Reference for all HybridSORT parameters in `boxmot` that affect tracking, association, and re-identification. Grouped by function.

Source: `boxmot/trackers/hybridsort/hybridsort.py` (boxmot v15.0.9)

---

## Track Lifecycle

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Frames a track stays alive without a detection match. After this, the track is deleted and its embeddings are lost. Higher = more time for ReID to re-associate, but also more ghost tracks. |
| `min_hits` | 3 | Consecutive matched frames before a track is output. Prevents spurious single-frame detections from creating tracks. Must be 1 if `det_interval > 1` (otherwise KF-only frames reset `hit_streak`). |
| `det_thresh` | 0.7 | Minimum confidence for a detection to enter the first (high-score) association stage. Detections below this but above `low_thresh` go to the BYTE stage. |
| `low_thresh` | 0.1 | Minimum confidence for a detection to enter the BYTE (low-score) association stage. Below this, the detection is discarded entirely. |

**Lifecycle flow:**
1. New detection above `det_thresh` → first association with active tracks
2. Unmatched detection → new `KalmanBoxTracker` created (fresh track ID)
3. Each frame without a match → `time_since_update += 1`
4. `time_since_update > max_age` → track deleted permanently (no gallery, no recovery)

---

## Kalman Filter (Motion Model)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_custom_kf` | True | Use 9D state `[u, v, s, c, r, du, dv, ds, dc]` (center x/y, scale, aspect ratio change, aspect ratio, velocities). False = standard 7D `[u, v, s, r, du, dv, ds]`. |
| `delta_t` | 3 | Number of past frames used to compute velocity estimates for the 4-corner motion model (TCM). Higher = smoother velocity but slower to react to direction changes. |

**What the KF does:**
- Predicts bounding box position when no detection matches (coasting)
- The predicted state is used for IoU computation against new detections
- `convert_x_to_bbox(trk.kf.x)` extracts the predicted bbox from state

---

## Association (Matching Detections to Tracks)

### Stage 1: High-score detections

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iou_threshold` | 0.15 | Minimum IoU (or HMIoU) for a match to be accepted. Matches below this are rejected even if they're the best assignment. |
| `asso_func` | `"hmiou"` | Association function. `hmiou` = height-modulated IoU (penalises height mismatch). Alternatives: `iou`, `giou`, `ciou`, `diou`. |
| `inertia` | 0.05 | Weight of velocity-direction consistency in the TCM (trajectory consistency module). Low = IoU-dominant; high = direction matters more. |
| `TCM_first_step` | True | Enable trajectory consistency module for the first association stage. Adds 4-corner velocity direction costs to the IoU cost matrix. |
| `high_score_matching_thresh` | 0.7 | (Unused in current code — the threshold is applied via `iou_threshold` + long-term ReID correction instead.) |

**Cost matrix (when ReID enabled):**
```
cost = weights[0] * (-(IoU + angle_cost)) + weights[1] * emb_cost + longterm_reid_weight * long_emb_cost
```
Solved with the Hungarian algorithm (`linear_assignment`).

### Stage 2: BYTE (low-score detections)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_byte` | True | Enable BYTE stage — associates low-confidence detections (`low_thresh < conf < det_thresh`) with unmatched tracks from stage 1. |
| `TCM_byte_step` | True | Enable TCM in the BYTE stage. |
| `TCM_byte_step_weight` | 1.0 | Weight of the score difference penalty in BYTE association. |
| `EG_weight_low_score` | 1.3 | Weight of embedding (ReID) cost in the BYTE association. 0 = no ReID in BYTE stage. |

### Stage 3: Last-observation IoU fallback

After stages 1 and 2, any remaining unmatched detections and tracks get one final IoU-only check against each track's `last_observation` (last detected bbox, not KF prediction). No ReID in this stage.

---

## ReID (Re-Identification via Appearance Features)

### Feature Extraction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reid_weights` | — | Path to ReID model weights (e.g. `clip_veri.pt`). The model extracts a D-dimensional embedding vector from each detection crop. |
| `with_reid` | True | Enable ReID. If False, association is purely IoU + motion. |

### Short-term Feature (EMA)

Each track maintains a `smooth_feat` — an exponential moving average of recent embeddings.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.9 | EMA weight for `smooth_feat`. `smooth_feat = alpha * smooth_feat + (1-alpha) * new_feat`. **Higher = more memory of old appearance** (slow adaptation). Lower = adapts faster to appearance changes. Our setting: **0.7**. |
| `adapfs` | False | Adaptive feature smoothing — weights the EMA by detection confidence ratio instead of fixed alpha. |
| `track_thresh` | 0.5 | (Used internally by adapfs logic.) |

**How `smooth_feat` is used:** In stage 1, `embedding_distance(track_features, det_features)` computes cosine distance between each track's `smooth_feat` and each detection's embedding. This becomes the `emb_cost` matrix.

### Long-term Feature Bank

Each track stores up to N past embeddings in a deque (`features`). The mean of this bank represents the track's long-term appearance.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `longterm_bank_length` | 30 | Max embeddings stored per track. At 30fps, default stores ~1s. Our setting: **150** (~5s of appearance history). |
| `with_longterm_reid` | True | Include long-term embedding distance in the stage 1 cost matrix. |
| `longterm_reid_weight` | 0.0 | Weight of long-term embedding cost in stage 1 assignment. **Default 0.0 = disabled.** Our setting: **0.25**. |

**How the long-term bank is used:**
```python
long_track_features = mean(track.features)  # average of all stored embeddings
long_emb_dists = cosine_distance(long_track_features, detection_features)
```
Added to the cost matrix with weight `longterm_reid_weight`.

### Long-term ReID Correction (Post-match Filter)

After Hungarian assignment, matched pairs are checked: if the embedding distance is too high AND IoU is too low, the match is rejected (detection becomes unmatched → new track).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `with_longterm_reid_correction` | True | Enable post-match rejection based on embedding distance. |
| `longterm_reid_correction_thresh` | 0.4 | Max cosine distance for a stage 1 match to survive. If `emb_cost > thresh AND IoU < iou_threshold` → reject the match. Our setting: **0.5** (more permissive). |
| `longterm_reid_correction_thresh_low` | 0.4 | Same threshold applied in the BYTE (low-score) stage. Our setting: **0.5**. |

**Correction logic (stage 1):**
```python
if emb_cost[det, trk] > longterm_reid_correction_thresh and IoU[det, trk] < iou_threshold:
    # reject match — detection goes to unmatched → new track
```

### Embedding-Guided Association Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EG_weight_high_score` | 4.6 | Weight of embedding cost in stage 1 cost matrix. Higher = ReID matters more relative to IoU + motion. |
| `EG_weight_low_score` | 1.3 | Weight of embedding cost in BYTE stage cost matrix. |

---

## Camera Motion Compensation (CMC)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cmc_method` | `"ecc"` | Camera motion compensation method. `ecc` = Enhanced Correlation Coefficient — estimates a 2D affine warp between consecutive frames and applies it to KF states before prediction. Helps when the camera pans/tilts. |

---

## Per-class Tracking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `per_class` | False | Run separate tracker instances per class ID. Each class has its own `active_tracks` list. Prevents cross-class ID switches (e.g. a boat track swapping to a person). Our setting: **True**. |
| `nr_classes` | 80 | Number of classes. Must match model output. Our setting: **12** (domain-split taxonomy). |

**Caveat:** When `per_class=True`, `tracker.active_tracks` after `update()` points to the last class processed. Use `tracker.per_class_active_tracks` dict to iterate all classes.

---

## Troubleshooting: Track Fragmentation (New ID Despite Track Being Alive)

When a detection fails to re-associate with an existing active track, all three association stages have failed and a new track ID is created. Common causes and tuning:

### Cause 1: KF drift → IoU too low after coasting

After several frames without a match, the Kalman prediction drifts away from the true position. When the detection reappears, IoU between predicted box and actual detection drops below `iou_threshold`.

| Parameter | Current | Try | Why |
|-----------|---------|-----|-----|
| `iou_threshold` | 0.15 | **0.05** | Accept weaker spatial overlap after KF drift. This is the hard gate — below this, the match is rejected regardless of ReID score. |

### Cause 2: ReID correction rejecting valid matches

Even when Hungarian assignment pairs a detection to the correct track, the post-match filter rejects pairs where both embedding distance is high AND IoU is low. If appearance shifted (lighting, angle, partial occlusion), cosine distance spikes and the match gets thrown out.

| Parameter | Current | Try | Why |
|-----------|---------|-----|-----|
| `longterm_reid_correction_thresh` | 0.5 | **0.7** | More permissive — allows higher cosine distance before rejecting. The correction logic is `if emb_cost > thresh AND IoU < iou_threshold → reject`. Raising this means only very dissimilar appearances get rejected. |
| `longterm_reid_correction_thresh_low` | 0.5 | **0.7** | Same for the BYTE (low-score) stage. |

### Cause 3: ReID features too stale or too volatile

If `smooth_feat` doesn't represent the object's current appearance well, the embedding cost in the Hungarian assignment will be high, making the optimizer prefer a different pairing or no match at all.

| Parameter | Current | Try | Why |
|-----------|---------|-----|-----|
| `alpha` | 0.7 | **0.5** | `smooth_feat` adapts faster to appearance changes. Useful when lighting or angle shifts gradually — the feature tracks the object's look more closely. |
| `EG_weight_high_score` | 4.6 | **6.0** | Makes ReID dominate over IoU + motion in the cost matrix. If the appearance is good but spatial overlap is poor (KF drift), ReID can override and force the correct match. Trade-off: if ReID is wrong, it'll force incorrect matches too. |

### Cause 4: Per-class boundary (no parameter fix)

With `per_class=True`, a `boat-rgb` track can **never** match a `boat-thermal` detection — they're in completely separate tracker instances. If the model switches class labels on the same object between frames, a new track ID is guaranteed.

**Mitigation:** Cross-modal NMS (`--enable-nms`) suppresses the duplicate class detection before the tracker, so only the higher-confidence class enters tracking. This prevents the label-switching problem but doesn't merge identities retroactively.

### Cause 5: Detection confidence drops below det_thresh

If a detection's confidence drops below `det_thresh` (0.3), it won't enter stage 1. It may still match via BYTE (stage 2) if above `low_thresh` (0.1), but BYTE only uses IoU + optional ReID — no long-term bank.

| Parameter | Current | Try | Why |
|-----------|---------|-----|-----|
| `det_thresh` | 0.3 | **0.4** | Widens the BYTE band (0.1–0.4 instead of 0.1–0.3). More detections get a second-chance association. Trade-off: high-confidence detections lose the stronger stage 1 ReID matching. |

### Recommended tuning order

1. `iou_threshold` 0.15 → 0.05 (cheapest fix, most likely cause)
2. `longterm_reid_correction_thresh` 0.5 → 0.7 (stops post-match rejection)
3. `alpha` 0.7 → 0.5 (better feature tracking)
4. `EG_weight_high_score` 4.6 → 6.0 (let ReID override poor IoU)

---

## Our Current Settings (track_video_predict.py)

```python
HybridSort(
    reid_weights=Path("clip_veri.pt"),
    device="cuda:0",
    half=False,
    per_class=True,
    nr_classes=12,              # 12-class domain-split model
    det_thresh=0.3,             # match --conf
    max_age=180,                # ~6s at 30fps — track stays alive for ReID
    min_hits=3,                 # 1 if det_interval > 1
    iou_threshold=0.15,
    use_custom_kf=True,
    alpha=0.7,                  # faster appearance adaptation (default 0.9)
    longterm_bank_length=150,   # ~5s of stored features (default 30)
    longterm_reid_weight=0.25,  # enable long-term bank (default 0.0 = off)
    with_longterm_reid=True,
    with_longterm_reid_correction=True,
    longterm_reid_correction_thresh=0.5,
    longterm_reid_correction_thresh_low=0.5,
)
```

---

## What's NOT Done (Gaps)

**No dead-track gallery.** When `time_since_update > max_age`, the track is deleted and its embeddings (`smooth_feat`, `features` deque) are garbage-collected. If the same object reappears after `max_age` frames, it gets a brand new track ID. There is no cosine similarity check against previously deleted tracks before assigning a new ID.

**No cross-class ReID.** With `per_class=True`, a `boat-rgb` track cannot be matched to a `boat-thermal` detection via ReID. Cross-modal NMS (separate module) handles suppression before the tracker, but doesn't merge track identities across classes.
