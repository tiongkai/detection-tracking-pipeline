# Degradation-robustness study — findings

Eval set: `eval_videos/wavy-boats/` — 32 Haulover clips, 1080p, MOT GT (boat/human/
outboard-motor). Config: ECC **off** + TRT detector + OSNet-TRT ReID (the real-time
deployment config; verified no accuracy change vs ECC-on on these near-static clips).

**Two probes:** `gtdet` = GT boxes fed as detections (isolates tracking/ReID; Recall ~1
by construction) · `yolo` = real detector (detection robustness). Scoring drops head/
torso and maps thermal→rgb so detections match the 3-class GT. GT is partially
labelled → **Recall** is the trustworthy headline; Precision/MOTA are soft.

Reports in this folder: `summary.md` (tables + full metrics glossary), `per_video_report.md`
(per clip, each corruption vs clean, gtdet+yolo stacked), `per_target_*.md` (per GT
track), `kalman_contribution_yolo.md`, `gallery_compare_clean.md`, `gt_dataset_report.md`,
`metrics.csv` (all raw numbers).

## 1. Detection is the fragile part; tracking is robust
With perfect boxes (`gtdet`), IDF1 stays **0.82–0.88** across *every* corruption. All the
damage is in detection (`yolo` Recall):

| corruption | yolo Recall | vs clean |
|---|---|---|
| clean | 0.669 | — |
| **lowlight s4** | **0.516** | **−23%** |
| invert | 0.605 | −10% |
| grayscale | 0.636 | −5% |
| jpeg s4 | 0.642 | −4% |
| shake s4 | 0.651 | −3% |
| lowlight/jpeg/shake s2 | ~0.67 | flat |

Low-light is by far the worst; JPEG and shake are gentle on recall.

## 2. Invert wrecks ID stability via detection flicker
`yolo` IDsw **67 (clean) → 210 (invert)**, but `gtdet` invert IDsw stays ~10. So inversion
makes the *detector* flicker (constant re-detections → new IDs); the tracker is fine.

## 3. Grayscale is the corruption that touches ReID
Even with perfect boxes, `gtdet` IDF1 **0.853 → 0.824** under grayscale — monochrome
genuinely reduces appearance discrimination (ReID).

## 4. The Kalman filter earns its keep when detection fails — except under shake
KF coasting recall recovery (yolo):

| condition | KF recall gain |
|---|---|
| clean | +8.1 pp |
| **lowlight s4** | **+12.4 pp** |
| **invert** | **+10.4 pp** |
| jpeg s4 | +9.8 pp |
| shake s2 | +6.7 pp |
| **shake s4** | **+4.1 pp** ⬇ |

KF coasting bridges detection dropouts (helps most where detection is worst) — but helps
**least under camera shake**, because erratic motion breaks its constant-velocity
prediction (IoU gate fails → falls back on ReID). → appearance-based re-acquisition
(best-frame ReID / SAM3) is the right tool for shake, not the motion model.

## 5. Best-frame ReID gallery ≈ FIFO on clean
best vs fifo (clean, yolo): IDF1 0.453 → 0.449, HOTA 0.481 → 0.479 — **no improvement**
(slightly worse). On clean video FIFO frames are already good, so "best" adds nothing.
Its hypothesized benefit needs degraded conditions (intermittent clear frames) — not yet run.

## 6. Per-target: the boat tracks clean; similar humans swap; distant targets are missed
- `gtdet` clean: easy clips → all targets clean (1 ID, 0 swaps). Busy clips → boat perfect,
  but **similar humans swap among themselves even with perfect boxes** (pure ReID failure).
- `yolo`: distant/small targets have det_recall ≈ 0 → **missed by detection, not mis-associated**.

## 7. GT long-gap re-split (fairer identity eval)
50% of GT tracks have gaps; 35 exceed `max_age` (180 frames). Splitting GT IDs at gaps >180:
- `gtdet`: IDF1 **0.853 → 0.880** (+2.6 pp), HOTA +1.9 pp — the unfair long-gap penalty was real.
- `yolo`: +0.5 pp only (detector misses those gap-reappearances anyway).
- Detection metrics unchanged (id-agnostic).

## Dataset summary
32 videos · 246 tracks (52 boat / 121 human / 73 motor) · 261,044 boxes · 73,908 frames.

---

## Implications
- **Harden detection for low-light** — it's the dominant failure (recall −23% at s4); the
  tracker is already robust when given boxes.
- **Shake needs appearance-based re-acquisition** (best-frame/SAM3), since KF coasting can't
  help when motion is unpredictable — and that's exactly the wave-bounce case.
- **Grayscale/invert** stress the ReID/detector colour-dependence — relevant if monochrome
  or unusual-colour input is possible.
- **Best-frame gallery** shows no clean-video benefit; only worth pursuing if the degraded-
  condition runs show it helps where frame quality varies.
