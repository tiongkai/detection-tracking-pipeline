# Plan: SAM3 + fast-tracker hybrid for occlusion re-acquisition

**Status:** Planning / parked. Blocked on a **test set of real 1–2 s wave-occlusion
clips**. Pick up once that exists. Author handoff doc — read end to end before coding.

## Problem
HybridSORT fragments track IDs when a boat is **occluded / buried by waves for ~1–2 s**
(it bounces, disappears behind a wave or another hull, reappears). Need stable IDs
through these gaps while keeping **real-time 25 FPS output** (source is 25 FPS, latency
must stay minimal).

## Established facts (measured this cycle — see `youtube-annotation/`)
- SAM3 (`Sam3TrackerVideoModel`, transformers fork) tracks at **~3.8 FPS @ 1008px**
  (forward ~243 ms = 93% of frame; 454M-param ViT encoder is ~all the cost).
  Benchmarks: `youtube-annotation/bench_sam3_speed.py`, `validate_sam3_512.py`.
- SAM3 video tracking is **causally sequential** — frame N needs frame N-1's memory.
  One stream is latency-bound at ~4 FPS; **more GPUs ≠ faster stream**, only more streams.
- Cost is **per-frame, not per-object** (encoder runs once/frame regardless of #objects).
- **TensorRT:** whole model NO (stateful session forward). Encoder YES (exports to ONNX,
  1.85 GB) → ~7–10 FPS @1008 best case. Still not real-time.
- **Low-res (512) is NOT a quick win:** the 1008 feature grid is hard-coded across rotary
  buffers + mask decoder (288=72×4) + memory path. Rotary patch alone is insufficient.
  Would need a low-res-trained checkpoint. Speed-only @518 was 48 ms (~17 FPS) but the
  model can't run *correctly* at 512 without retraining.
- **Memory:** SAM3 uses only ~1.7 GB — bounded `num_maskmem=7` window (1 seed + 6 recent),
  CPU-offloaded frames/memory streamed per frame (`video_storage_device="cpu"`), bf16,
  batch-1. GPU mem is flat regardless of clip length.

## Occlusion-horizon math (drives feasibility)
At 25 FPS source, SAM3 processes ~**every 6th frame**, ~**6-frame (~250 ms) lag**.
Its 6 recent memory slots span **6×6 ≈ 36 frames ≈ 1.4 s** (+ stale seed frame).
- **1 s occlusion → bridged.** 
- **2 s (50 frames) → beyond the recent window** → marginal (only seed left to match).
- **Lever:** raise `num_maskmem` (7→~13 → ~2.9 s horizon). Costs more memory-attention
  compute (small) + GPU mem (have headroom). OOD vs training (7) — **must validate**.

## Architecture — "fast positions, SAM3 identity" (the stronger variant)
Split positions from identity so the SAM3 lag becomes invisible:

- **Fast path — YOLO + HybridSORT, every frame, 25 FPS → the live output**
  (boxes + *provisional* IDs). Never lags. Real detections every frame (no blind KF
  extrapolation across the gap — critical on bouncing water).
- **SAM3 — async background, ~6-frame lag, every-6th-frame → the identity oracle.**
  Maintains a masklet per boat through the occlusion.
- **Reconciliation:** on reappearance, **YOLO re-detects the boat live** (fresh position),
  **SAM3 says "that detection == the boat you lost" → relink the original ID.**
  SAM3 supplies the *identity link*; YOLO supplies the *position*. A label correction
  arriving ~250 ms late is operationally invisible.

ID plumbing: seed SAM3 from HybridSORT's *confirmed* tracks; maintain an ID map; add
newly-detected boats into SAM3's session mid-stream; on HybridSORT loss, match the live
YOLO re-detection to SAM3's masklet (IoU at the aligned frame) → recover the ID.

## Risks / open issues
1. **2 s occlusion vs 1.4 s default horizon** → `num_maskmem` bump (OOD, validate).
2. **SAM3 sees only every 6th frame during the bounce** — exactly the most erratic moment.
   It matches by appearance (robust to position jumps) but sparse frames in chaos is the
   weak spot. Unproven — must measure.
3. **~1 GPU per camera** for the SAM3 path (sequential).
4. SAM3 is *prompted*, not a detector → **YOLO still needed** to discover new boats.

## Do the cheap thing first (likely avoids SAM3 entirely)
Re-acquisition through occlusion is an **identity-through-time** problem, and the
**best-frame ReID gallery** (branch `dev/best-frame-reid`, `track/hybridsort_bestframe.py`,
`--reid-gallery best`) attacks exactly that **at full 25 FPS, no extra GPU**. HybridSORT's
`max_age=180` already keeps a lost track alive **7.2 s** while coasting; on reappearance,
ReID re-matches it — and the best-frame gallery holds the boat's clearest views for a
stronger match. **SAM3's unique value only appears when appearance ReID fails** (near-
identical boats, or appearance changes too much across the gap).

## De-risk experiment (run FIRST, once the test set exists)
On clips with real 1–2 s wave occlusions:
1. **YOLO+HybridSORT + `--reid-gallery best`** — does ID survive the occlusion?
   (cheap, full speed; compare vs `--reid-gallery fifo` baseline). Metrics: IDsw, IDF1,
   Frag, and per-occlusion ID-retention rate.
2. **SAM3 @ every-6th-frame cadence** — can it actually hold the boat through the same
   occlusion? Sweep `num_maskmem` ∈ {7, 13}. Validates the core SAM3 assumption before
   committing a GPU/camera.

**Decision gate:** if (1) recovers the occlusions, ship that — skip SAM3. If not, (2)
tells us whether SAM3 helps and with what `num_maskmem`, justifying the hybrid build.

## Implementation phases (only past the gate)
- **Phase 0 — de-risk** (above). Needs the occlusion test set + GT IDs through the gap.
- **Phase 1 — ReID-only**: tune best-frame gallery (k, diversity, max_age) for these
  occlusions; ship if sufficient.
- **Phase 2 — SAM3 oracle** (if needed): async SAM3 worker (reuse
  `youtube-annotation/annotator/sam3_worker.py` server mode) maintaining masklets for
  confirmed tracks; subsample to latest frame (fixed lag, never queue); ID-map +
  reconciliation layer in the tracking loop; `num_maskmem` tuned from Phase 0.
- **Phase 3 — productionize**: per-camera GPU budgeting; TensorRT the SAM3 *encoder*
  (exportable) to claw back some SAM3 throughput; failure handling when SAM3 also loses it.

## Key references
- SAM3 worker + server protocol: `youtube-annotation/annotator/sam3_worker.py`
- SAM3 benches: `youtube-annotation/bench_sam3_speed.py`, `validate_sam3_512.py`
- Best-frame ReID: branch `dev/best-frame-reid`, `track/hybridsort_bestframe.py`
- SAM3 env: `youtube-annotation/htx-uc6` (uv, py3.12, torch 2.9+cu128, transformers SAM3 fork)
- SAM3 checkpoint: `youtube-annotation/htx-uc6/checkpoints/sam3` (`num_maskmem=7`)
