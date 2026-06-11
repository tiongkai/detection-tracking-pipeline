# Design: degradation-robustness evaluation

**Status:** Design agreed; implementation pending. Build on a new branch
(`dev/degradation-eval`) off `main`.

## Goal
Measure how the detector (YOLOv26-L) and tracker (HybridSORT + ReID) degrade as the
input is corrupted — low light, compression, grayscale/invert, blur/weather, and
random camera motion. Produce **degradation curves** (metric vs severity) per
corruption so we know where each part of the pipeline falls off, and what to harden.

## Core principle — reuse the existing GT
Corruptions split into two buckets by whether they move pixels geometrically:

| Bucket | Corruptions | GT handling |
|---|---|---|
| **Geometry-preserving** | low-light, grayscale, invert, compression, noise, blur, fog, contrast | **Reuse GT as-is** — pixels change, boxes don't move |
| **Geometry-changing** | **camera shake / random motion** | **Apply the same affine to GT boxes** (warp 4 corners → re-fit axis-aligned box) |

No re-labelling either way: geometry-preserving reuses the existing `vws-eval-set`
annotations directly; camera shake applies a *known* synthetic transform, so the GT
transform is exact. **Recall stays the trustworthy headline metric** under partial GT.

## Decisions (settled)
1. **Realistic severity anchoring** — severities are calibrated to real operating
   conditions, not arbitrary 1–5. Needs reference inputs from the user:
   - Compression → anchor to the actual live-stream bitrate / codec (H.264 CRF or
     target kbps). *Action: get target bitrate or a sample stream.*
   - Low-light → anchor Sev3–5 to real dusk/night footage if available.
   - Until anchors are provided, ship provisional ranges (below) and re-scale later.
2. **Camera shake × ECC is the priority probe** — it maps directly to the
   wave-bounce / ego-motion problem and exercises an existing module (ECC / camera
   motion compensation). The headline shake experiment is **severity × {ECC on,
   ECC off}**: does CMC actually recover ego-motion robustness, and by how much?
3. **One corruption at a time** — no compositions for v1. (Realistic multi-corruption
   combos, e.g. night + compression + shake, are a later extension.)

## Corruption catalog (severity 1–5, provisional)

### Low-light / night  *(stresses detector recall)*
Gamma↑ + gain↓ + Poisson (shot) + Gaussian (read) noise.
Sev1 ≈ dusk → Sev5 ≈ near-dark with visible sensor grain.

### Compression  *(stresses small/distant-boat detection)*
JPEG quality {75, 55, 40, 28, 18} and/or H.264 CRF {28, 33, 38, 43, 48}.
JPEG is per-frame (`cv2.imencode`); H.264 needs a re-encode pass (ffmpeg) — see notes.

### Grayscale & invert  *(stresses CLIP ReID identity, not detection)*
Each is binary (severity = on). CLIP ReID embeddings are colour-sensitive, so these
are expected to break *identity* (IDF1/IDsw) more than detection — a good ReID probe.

### Blur / weather  *(stresses against boat shake, sea spray, glare)*
Motion blur (kernel len 3→21), defocus (radius 1→6), fog/haze (atmospheric-light
blend), contrast↓ (glare/overcast).

### Random camera motion  *(priority — stresses ECC / tracker ego-motion)*
- Per-frame 2D translation (+ optional small rotation/scale) from a
  **temporally-smoothed random walk** (real-looking shake, not white-noise flicker).
  Two components: high-freq **jitter** (vibration) + low-freq **drift** (wander).
  Sev1 ≈ ±0.5% frame / ~0.2° → Sev5 ≈ ±5% / ~2°.
- **Pre-zoom-crop ~5%** so shake stays in-frame (no black borders; GT never exits frame).
- Apply the same affine to GT boxes.
- Headline run: shake severity × {ECC on, ECC off}.

## Architecture
- **`eval/degrade.py`** — pure transforms. `apply(frame, type, severity, seed) -> frame`
  for geometry-preserving; for shake also `apply_boxes(boxes, affine) -> boxes`.
  Deterministic given a seed (shake random walk seeded per clip).
- **`track/track_video_predict.py`** flag — `--degrade <type> --severity <n>
  [--degrade-seed N] [--save-degraded DIR]`. Applied on-the-fly in the frame loop
  (no 36 GB of degraded copies written). `--save-degraded` dumps a few sample frames
  for eyeballing.
- **`eval/robustness_sweep.py`** — runs the corruption×severity matrix over
  `vws-eval-set`, scores each run against the existing GT (camera shake: against the
  transformed GT), and emits the curves + tables.

## Metrics & output
- **Recall vs severity** — the headline curve, per corruption.
- Plus IDF1 / HOTA / IDsw for identity effects (esp. grayscale/invert/shake).
- **Relative robustness** = `metric(sev) / metric(clean)` so corruptions sit on one
  comparable axis.
- Dedicated cross-cut rows: **shake × ECC on/off**, **grayscale/invert × ID retention**.

## Implementation notes / gotchas
- **H.264 path** needs an ffmpeg re-encode (per-frame imencode only covers JPEG); decide
  whether to pre-render H.264 clips once per CRF or pipe through ffmpeg on the fly.
- **Camera-shake border**: the ~5% pre-zoom-crop avoids black borders *and* keeps all GT
  in-frame; verify GT boxes that were near the original edge don't get clipped out.
- **Determinism**: seed the shake random walk per clip so reruns (and ECC on/off) see the
  *same* motion — otherwise the ECC comparison isn't controlled.
- **ReID + grayscale/invert**: confirm whether the ReID preprocessing already normalizes
  colour; if so the effect may be muted — worth checking before drawing conclusions.

## Out of scope (v1)
- Geometry-changing corruptions other than translation/rotation shake (no resize/crop/
  perspective — would need richer GT transforms).
- Multi-corruption compositions (deferred).
