# Pending tasks

Living task list for the detection-tracking pipeline. (`task.md` is the older intern
MOT-metrics list — kept for history.) Branch: `dev/degradation-eval`.

Legend: 🔴 blocked · 🟡 ready · 🟢 in progress · ✅ done

---

## Recently done

### ✅ Best-frame ReID on AUGMENTED data — NO benefit (verdict)
Ran yolo + `--gallery best` across all conditions vs fifo → `gallery_best_vs_fifo_augmented.md`.
**Neutral-to-slightly-negative in every condition** (incl degraded; worst shake_s2 IDF1 −0.010).
Failures here are detection dropouts + similar-human ambiguity, not stale gallery frames.
**Don't adopt best-frame for this use case.** Long-paused fifo-vs-best evaluation settled (negative).

### ✅ grayscale_invert corruption
Added (B/W negative). Detection ≈ grayscale (recall 0.64) but no IDsw flicker (60 vs colour-
invert 210); tracking gentler than grayscale (IDF1 0.848). All reports + 10 samples done.

---

## Open follow-ups (degradation study)

### 🟡 shake × ECC-on
KF coasting helps least under shake (+4pp) — does ECC camera-motion-comp rescue it?
- [ ] Re-run shake s2/s4 with ECC ON (no `--no-cmc`), compare vs ECC-off shake.

### 🟡 Re-split GT across all conditions (fair-identity)
resplit@180 gave +2.6pp gtdet IDF1 on clean only. `eval/resplit_gt.py` ready.
- [ ] Re-score all conditions vs `labels_resplit180` → fair-identity column in reports.
- **Decision pending:** make re-split the primary reports, or keep as side experiment?

### 🟡 Severity anchoring (deferred)
- [ ] Anchor compression to real stream bitrate/codec; low-light to a dusk reference.
      (Needs inputs from user.)

---

## Other project tasks

### 🟡 OSNet ReID swap — re-eval on main pipeline
OSNet-TRT is already used in the degradation sweeps (fast). Still to do for the live pipeline:
- [ ] Confirm FPS + IDF1/IDsw on the production eval set vs CLIP; adopt if identity holds.
- See memory `project_inference_optimization.md`.

### 🔴 SAM3 hybrid re-acquisition
Plan on `dev/sam3-hybrid-reacquisition`. Degradation study supports it: under shake the KF
motion model fails (appearance/SAM3 needed).
- [ ] De-risk: best-frame ReID vs SAM3 on real 1–2 s wave-occlusion clips (gate before building).
- **Blocked on:** occlusion test set with GT IDs through the gap.

### 🔴 GT pre-annotation — full run
- [ ] Full pre-label run; boat-of-interest curation tooling. See `youtube-annotation/annotator/`.

### 🟡 Branch cleanup / merge
- [ ] Merge `dev/degradation-eval` → `main` once study is wrapped.
- [ ] Decide fate of `dev/best-frame-reid` (merged into main already), `dev/sam3-hybrid-reacquisition`.

---

## Done
- ✅ **Degradation-robustness study** — 32 clips × 9 corruptions × 2 probes (gtdet/yolo),
  ECC-off+TRT. Reports in `docs/degradation_eval_reports/` (FINDINGS, summary, per_video,
  per_target ×18, kalman, gallery-clean, gt_dataset, per_video_insights, metrics.csv).
  Tooling: degrade.py, robustness_sweep.py, per_video_report.py, per_target_breakdown.py,
  kalman_contribution.py, gallery_compare.py, resplit_gt.py, gt_dataset_report.py, make_samples.py.
- ✅ GT-as-detections mode (`--gt-dets`), annotated GT videos (32), flat-layout `visualize_gt`.
- ✅ Repo consolidation (single git repo); inference bottleneck (ECC-off+TRT 45.5 FPS);
  model conversion tooling; recall headroom sweep; SAM3 speed study; timing instrumentation.
