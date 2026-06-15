# Pending tasks

Living task list for the detection-tracking pipeline. (`task.md` is the older
intern MOT-metrics task list — kept for history; this file tracks current work.)

Legend: 🔴 blocked · 🟡 ready · 🟢 in progress · ✅ done

---

## Active / ready

### 🟢 Degradation-robustness eval — build (branch `dev/degradation-eval`)
Design: `docs/degradation_eval_design.md`. Working eval set: `eval_videos/wavy-boats/`
(4 Haulover clips, 1080p/60fps, MOT GT in `labels/`, classes 0=boat 2=human).
- [x] Annotated GT videos rendered → `results/wavy_boats_gt/` (via `visualize_gt.py --flat`).
- [x] **GT-as-detections mode** — `track_video_predict.py --gt-dets <labels-dir>`: feeds GT
      boxes as detections (YOLO bypassed), ReID still runs on image crops → isolates pure
      tracking/ReID ability. Smoke (seg1): boat + 2 humans held perfectly; **2 humans
      (GT 2 & 4) swap IDs even with perfect boxes** → pure ReID/association failure.
- [ ] Run GT-dets on all 4 clips + proper IDF1/HOTA via `eval/eval_tracking.py`.
- [ ] `eval/degrade.py` — corruption transforms (low-light, compression, grayscale,
      invert, blur/weather, camera-shake) + box transform for shake. Seeded/deterministic.
- [ ] `track/track_video_predict.py` — add `--degrade --severity --degrade-seed --save-degraded`.
- [ ] `eval/robustness_sweep.py` — corruption×severity matrix → degradation curves + tables.
- [ ] Two probes: (a) **YOLO on degraded** = detector robustness; (b) **GT-dets + degraded
      image** = tracker/ReID robustness in isolation.
- [ ] Headline run: **camera shake × {ECC on, ECC off}**; cross-cut: grayscale/invert × ID retention.
- **Needs from user:** realistic severity anchors — live-stream **bitrate/codec** (for
      compression) and ideally a real **dusk/night** reference (for low-light).

### 🟡 OSNet ReID swap (inference speed)
CLIP ReID (`clip_veri.pt`, 483 MB) is the live-inference bottleneck; OSNet
(`osnet_x0_25_msmt17.pt`, 3 MB) swap planned to hit the 20 FPS target.
- [ ] Run pipeline with `--reid-weights osnet_x0_25_msmt17.pt`, measure FPS gain.
- [ ] Re-eval tracking metrics (IDF1/IDsw) — confirm identity quality holds vs CLIP.
- See memory `project_inference_optimization.md`.

---

## Blocked (awaiting inputs)

### 🔴 Best-frame ReID gallery evaluation (fifo vs best)
Feature built + pushed on `dev/best-frame-reid` (`track/hybridsort_bestframe.py`,
`--reid-gallery best --gallery-k --gallery-diversity`). Paused by user pending new eval set.
- [ ] Run `--reid-gallery fifo` vs `best` on the eval set; compare IDF1 / IDsw / Frag.
- [ ] If best-frame wins → merge `dev/best-frame-reid` → `main`.
- **Blocked on:** the new eval set (ideally the 1–2 s wave-occlusion clips below).

### 🔴 SAM3 hybrid re-acquisition
Plan written on `dev/sam3-hybrid-reacquisition` (`docs/sam3_hybrid_reacquisition_plan.md`).
- [ ] **De-risk experiment (run FIRST):** best-frame ReID vs SAM3 on real occlusion clips.
      Decision gate — if ReID recovers the occlusions, skip SAM3 entirely.
- [ ] Only past the gate: build async SAM3 oracle + reconciliation layer.
- **Blocked on:** a test set of **real 1–2 s wave-occlusion clips with GT IDs through the gap.**

### 🔴 GT pre-annotation — full run
Pre-label a new eval set (boats / people / outboard motors), keeping only tracks of
"boats of interest" (people/motors only if on those boats). 1 clip sanity-tested OK.
- [ ] Full run on `wavy-boats` 1080p via `configs/gt_preannotate.yaml`.
- [ ] Curation tooling: boat-of-interest selection + spatial association of people/motors.
- See `youtube-annotation/annotator/` (SAM3 pre-annotation).

---

## Housekeeping

### 🟡 Branch consolidation
Repo is now a single git-tracked dir (`detection-tracking-pipeline-local`). Branch state:
- `dev/resolve-inference-bottleneck`, `dev/decouple-detection-clean` — merged into `main`
  (stale; can be deleted once confirmed).
- `dev/best-frame-reid` — +2 ahead (gallery feature + repo consolidation). Merge after eval.
- `dev/sam3-hybrid-reacquisition` — +1 (plan doc only). Merge/keep as needed.
- [ ] After best-frame eval: merge to `main`, delete stale branches.

---

## Done
- ✅ Repo consolidation — single git-tracked working tree; `-git` clone retired; artifacts
      gitignored (on `dev/best-frame-reid`).
- ✅ Inference bottleneck — ECC-off + TensorRT → 45.5 FPS (2.2×), no accuracy loss.
- ✅ Model conversion tooling — `export/export_onnx.py`, `export/build_tensorrt.py` + README.
- ✅ Recall headroom sweep (conf × max_coast) — `eval/sweep_recall.py`.
- ✅ SAM3 speed/feasibility study — ~3.8 FPS, sequential, not real-time; encoder-only TRT.
- ✅ Per-frame timing instrumentation (detection / tracking / ReID / total).
- ✅ GT validation + MOT-format export + visualizer.
