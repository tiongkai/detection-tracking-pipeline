# Task 2 — Tracking Metrics

## Objective

Use relevant metrics to evaluate tracker performance. These metrics allow objective, consistent comparison across tracker configurations (e.g. tuning `iou_threshold`, `alpha`, `longterm_reid_correction_thresh`) and validate whether changes actually improve tracking quality.

Currently we evaluate tracking by visual inspection of output videos. This doesn't scale and can't catch subtle regressions. We need automated metrics.

---

## Tasks

### 2.1 — Background reading (~2 days)

- [ ] **Read up on object detection and tracking**

  Gain background on detection (YOLO, DETR architectures — understand anchor-based vs anchor-free, how NMS works, confidence thresholds) and tracking (SORT → DeepSORT → ByteTrack → HybridSORT progression). This builds intuition for labelling, augmentation, and understanding why tracker parameters exist. We use **HybridSORT** from the `boxmot` library because we need ReID capability.

  Key things to understand:
  - How Kalman filter predicts bounding boxes when detection misses
  - What ReID embeddings are and how cosine similarity matching works
  - What `per_class` tracking means and why cross-modal NMS is needed (see `track.md`)
  - The three association stages in HybridSORT (high-score → BYTE → last-observation fallback)

  Read: `track.md` (parameter reference), `tracker_eval.md` (run history), boxmot repo docs

- [ ] **Identify relevant tracking metrics**

  Read MOT benchmark papers and understand what each metric measures. Focus on which ones are relevant to our use case (maritime surveillance with few objects, long occlusions, cross-domain appearance changes).

  **Core metrics to understand:**

  | Metric | What it measures | Why we care |
  |--------|-----------------|-------------|
  | **MOTA** (Multi-Object Tracking Accuracy) | Combined FP, FN, and ID switch rate | Overall tracking quality — our primary metric |
  | **IDF1** (ID F1 Score) | How consistently the correct ID is maintained over a track's lifetime | Directly measures the track fragmentation problem we're seeing |
  | **HOTA** (Higher Order Tracking Accuracy) | Balanced combination of detection and association quality | Newer metric that separates detection errors from association errors — tells us if problems are from the detector or the tracker |
  | **ID Switches** | Number of times a track ID changes on the same ground-truth object | Directly counts our fragmentation issue |
  | **Fragmentation** | Number of times a ground-truth track is interrupted (gaps) | Measures how often the detector loses an object temporarily |
  | **MT/ML** (Mostly Tracked / Mostly Lost) | % of GT tracks tracked for >80% / <20% of their lifetime | Tells us if we're maintaining tracks for the full duration |

  Deliverable: short writeup (1 page) on which metrics we'll use and why, with references to papers.

  Suggested reading:
  - MOT Challenge evaluation methodology
  - HOTA paper (Luiten et al., 2021)
  - TrackEval repo documentation

  *~1 day*

---

### 2.2 — Ground truth annotation for tracking eval (~5 days)

- [ ] **Create ground-truth tracking annotations for eval clips**

  Metrics need ground truth. Select 3–5 representative clips from `data/eval/` covering different scenarios:

  | Scenario | Suggested clip | Why |
  |----------|---------------|-----|
  | Clear RGB, single boat | `FishingBoat.mp4` | Baseline — should be easy |
  | Thermal, multiple boats | 1 x pt80 dusk clip (023399 series) | Cross-modal NMS territory |
  | Poor lighting, occlusion | 1 x `clips/` dahua segment | Tests ReID through occlusion |
  | Fast camera motion | 1 x `clips/` segment with pan | Tests KF prediction quality |

  For each clip, annotate in MOTChallenge format:
  ```
  <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
  ```

  Use CVAT or similar annotation tool with video tracking mode.

  Deliverable: GT files in `data/eval/gt/` in MOTChallenge format, one per clip.

  *~5 days (annotation is slow — scope to 3 clips and keep each under 30s)*

---

### 2.3 — Tracking output in MOTChallenge format (~1 day)

- [x] **Modify `track_video_predict.py` to output MOTChallenge-format text alongside the video**

  Currently the script only writes annotated mp4 video. Add a `--save-mot` flag that also writes a `.txt` file per clip in MOTChallenge format:

  ```
  <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
  ```

  Include both detection-matched tracks and Kalman-predicted tracks (mark predicted with visibility < 1.0 or a flag). This is the input for TrackEval.

  Deliverable: `--save-mot` flag producing `.txt` files that TrackEval can ingest.

  *~1 day*

---

### 2.4 — Evaluation script using TrackEval (~3 days)

- [x] **Integrate TrackEval for automated metric computation**

  Use the [TrackEval](https://github.com/JonathonLuiten/TrackEval) library (or `py-motmetrics`) to compute MOTA, IDF1, HOTA, ID switches, fragmentation from the MOT-format outputs against ground truth.

  Create a script `eval/eval_tracking.py` that:
  1. Takes a directory of tracker output `.txt` files and a GT directory
  2. Runs TrackEval and outputs a metrics table
  3. Supports `--compare` mode to show two configs side-by-side

  Deliverable: `eval/eval_tracking.py` that produces a table like:
  ```
  | Config           | MOTA | IDF1 | HOTA | IDsw | Frag |
  |------------------|------|------|------|------|------|
  | baseline         | 0.72 | 0.65 | 0.58 | 14   | 23   |
  | lower_iou_thresh | 0.75 | 0.71 | 0.62 | 8    | 19   |
  ```

  Tip: run TrackEval on a standard MOT benchmark example first to learn the expected directory structure and naming conventions before wiring up custom data.

  *~3 days (1 day learning TrackEval on example data + 1 day setup + 1 day testing/debugging)*

---

### 2.5 — Baseline measurement + parameter tuning experiments (~3 days)

#### Background: How HybridSORT association works

When a new frame arrives, the tracker tries to match each detection to an existing track in three stages:

**Stage 1 (high-confidence detections):** Builds a cost matrix combining IoU, 4-corner motion consistency, and ReID cosine similarity. The Hungarian algorithm finds the optimal assignment. After assignment, a post-match filter rejects pairs where both the embedding distance is high AND IoU is low.

```
cost = 1.0 * (-(IoU + angle_cost)) + 4.6 * emb_cost + 0.25 * long_emb_cost
```

**Stage 2 (BYTE — low-confidence detections):** Detections between `low_thresh` (0.1) and `det_thresh` (0.3) get a second-chance association with unmatched tracks from stage 1. Uses IoU + ReID but no long-term feature bank.

**Stage 3 (last-observation fallback):** Remaining unmatched detections and tracks get one final IoU-only check against each track's last real detected position. No ReID.

**If all three stages fail:** The detection creates a new track with a fresh ID. This is the track fragmentation problem — the object is the same, but it gets a new number.

#### The core problem: Track fragmentation

We observe cases where the same object gets assigned a new track ID despite the old track still being alive. For example, a boat is tracked as id1 for frames 1–10, then on frame 11 it suddenly becomes id14. id1 continues coasting on Kalman predictions while id14 starts fresh.

This happens when the detection on frame 11 fails to match id1 in all three association stages. The most common reasons:

1. **KF prediction drifted** — after a few frames of coasting (no detection match), the Kalman filter's predicted box position diverges from reality. When the detection reappears, IoU between prediction and detection is too low.
2. **Appearance changed** — lighting shift, angle change, or partial occlusion causes the ReID embedding to be too different from the track's stored appearance, and the post-match filter rejects the pair.
3. **Cross-modal class switch** — the model labels the same object as `boat-rgb` on one frame and `boat-thermal` on the next. With `per_class=True`, these are completely separate tracker instances that can never match. (Largely mitigated by cross-modal NMS.)

Each experiment below targets one of these causes. The metric to watch is **IDF1** (measures ID consistency over a track's lifetime) and **ID Switches** (counts exactly how many times an ID changes on the same GT object).

---

#### Experiments

- [ ] **Experiment 0: Baseline**

  Run the eval script on all annotated clips with the current settings from `track.md`. Record as the baseline for all comparisons.

  Current config:
  ```
  iou_threshold=0.15, alpha=0.7, longterm_reid_correction_thresh=0.5,
  EG_weight_high_score=4.6, det_thresh=0.3, enable_nms=off
  ```

  *~0.5 day*

- [ ] **Experiment 1: `iou_threshold` tuning**

  **What:** Run with `iou_threshold` = 0.15 (baseline), 0.10, 0.05
  **Why:** This is the hard gate for accepting a match. If IoU between the KF-predicted box and the detection is below this threshold, the match is **always rejected** — regardless of how good the ReID similarity is. After a track coasts for several frames, the Kalman prediction drifts and IoU drops. Lowering this threshold lets the tracker accept spatially weaker matches, giving ReID a chance to confirm the identity.
  **Trade-off:** Too low → false matches between nearby but different objects. Less of a concern in our maritime use case (few objects, far apart) than in crowded pedestrian scenes.
  **Watch for:** IDF1 increase + ID switch decrease = KF drift was causing fragmentation. MOTA decrease = false matches introduced.

  *~0.5 day*

- [ ] **Experiment 2: `longterm_reid_correction_thresh` tuning**

  **What:** Run with `longterm_reid_correction_thresh` = 0.5 (baseline), 0.6, 0.7
  **Why:** After the Hungarian algorithm assigns a detection to a track, this post-match filter double-checks: if `cosine_distance > threshold AND IoU < iou_threshold`, the match is rejected and the detection becomes a new track. At 0.5, any cosine distance above 0.5 combined with poor IoU kills the match. This is aggressive — appearance can change legitimately (lighting gradients, angle changes on a turning boat). Raising the threshold makes the filter more permissive: only very dissimilar appearances get rejected.
  **How it interacts with Experiment 1:** Lowering `iou_threshold` means more matches will have `IoU < iou_threshold`, which means the correction filter fires more often. These two parameters should ideally be tuned together.
  **Watch for:** IDF1 increase + ID switch decrease = the filter was killing valid matches. New false-positive ID merges = threshold is too permissive.

  *~0.5 day*

- [ ] **Experiment 3: `alpha` (ReID EMA weight) tuning**

  **What:** Run with `alpha` = 0.7 (baseline), 0.5, 0.3
  **Why:** Each track maintains a `smooth_feat` — an exponential moving average of past ReID embeddings. The update rule is: `smooth_feat = alpha * smooth_feat + (1-alpha) * new_feat`. At alpha=0.7, the feature is 70% old appearance + 30% new. This means appearance changes slowly in the stored representation. If a boat gradually changes appearance (e.g. dusk lighting transition, rotating angle), the stored feature may lag behind reality, causing high cosine distance when the tracker tries to match. Lower alpha = the stored feature adapts faster to the current appearance.
  **Trade-off:** Too low → the feature becomes volatile, changing drastically frame-to-frame. Brief occlusions or detection noise could corrupt it, making re-association harder after the object reappears.
  **Watch for:** IDF1 increase on clips with gradual appearance change (dusk/twilight clips). Fragmentation increase on clips with occlusion = alpha too low.

  *~0.5 day*

- [ ] **Experiment 4: `EG_weight_high_score` tuning**

  **What:** Run with `EG_weight_high_score` = 4.6 (baseline), 6.0, 8.0
  **Why:** This weight controls how much ReID influences the stage 1 cost matrix relative to IoU + motion. The cost formula is: `cost = 1.0 * (-(IoU + angle_cost)) + EG_weight * emb_cost + 0.25 * long_emb_cost`. At 4.6, ReID already weighs heavily. Increasing it further means that even when IoU is poor (KF drift), a good appearance match can pull the assignment toward the correct track.
  **Trade-off:** If the ReID model produces a false similarity (different objects with similar appearance), a high weight will force an incorrect match. In maritime scenarios (boats look similar), this risk is real.
  **Watch for:** ID switch decrease = ReID is overriding bad IoU for correct matches. MOTA decrease = ReID is forcing wrong matches between similar-looking objects.

  *~0.5 day*

- [ ] **Experiment 5: Interclass NMS impact**

  **What:** Run with and without `--enable-nms --nms-iou-thresh 0.5` on thermal/twilight clips
  **Why:** The 12-class model fires both `boat-rgb` and `boat-thermal` on ambiguous-lighting objects. Without NMS, both detections enter the tracker as separate per-class tracks — guaranteed duplicate IDs for the same physical object. With NMS, the lower-confidence duplicate is suppressed before the tracker sees it. This was measured to reduce dual-class frames by 72% on pt80 twilight clips.
  **Watch for:** ID switch decrease on twilight clips = NMS is preventing cross-class duplicate tracks. No change on clear-domain clips (RGB-only or thermal-only) = expected, NMS only fires when both classes are present.

  *~0.5 day*

- [ ] **Write up findings**

  Document all results in a comparison table. For each experiment, note:
  - Which metric improved, which degraded
  - Whether the improvement was on specific clip types (twilight vs clear, occlusion vs open)
  - Recommended value with justification

  If experiments 1 and 2 both help, run a combined config (e.g. `iou_threshold=0.05 + longterm_reid_correction_thresh=0.7`) and measure whether the improvements stack.

  Deliverable: updated `tracker_eval.md` with metrics tables for each experiment and a final recommended config.

  *~0.5 day*

---

## Time Estimate Summary

| Task | Days |
|------|------|
| 2.1 Background reading + metric selection | 2 |
| 2.2 Ground truth annotation | 5 |
| 2.3 MOTChallenge output format | 1 |
| 2.4 TrackEval integration script | 3 |
| 2.5 Baseline + 5 experiments + writeup | 4 |
| **Total** | **~15 working days** |

Tasks 2.1 and 2.2 are best done sequentially — background reading gives context for consistent annotation decisions. Tasks 2.3 and 2.4 are sequential. Task 2.5 depends on all prior tasks.
