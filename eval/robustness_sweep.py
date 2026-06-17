"""Corruption-robustness sweep: generate -> infer -> score, then tabulate.

Orchestrates the degradation study end-to-end (the three concerns stay separate
scripts; this just drives them):
  1. generate  : eval/degrade.py writes augmented videos + aligned GT per (kind,severity)
  2. infer     : track/track_video_predict.py runs on each augmented set
                   - probe 'gtdet': --gt-dets  -> isolates tracker/ReID robustness
                   - probe 'yolo' : real detector -> detector robustness
  3. score     : eval/eval_tracking.py (custom) -> IDF1/IDsw/MOTA/HOTA/Recall/Precision

Outputs results/<out>/metrics.csv (per probe×kind×severity×clip) and summary.md
(metric-vs-severity tables per kind, with the clean baseline as severity 0).

Focused default run (per user): kinds {lowlight,jpeg,shake} × severities {2,4} × 4 clips,
both probes, OSNet TRT ReID. Resumable: skips already-generated videos and existing MOT.

    .venv/bin/python eval/robustness_sweep.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from degrade import degrade_video                       # noqa: E402
from eval_tracking import load_mot, evaluate_sequence_custom  # noqa: E402

FOCUS_METRICS = ["Recall", "Precision", "IDF1", "IDsw", "MOTA", "HOTA"]

GLOSSARY = """\
## How to read this report

**Two probes (the core design — they isolate different failure modes):**
- **`gtdet`** — ground-truth boxes are fed to the tracker *as detections* (the detector is
  bypassed). This isolates **tracking / ReID** ability: detection is perfect (Recall ~1.0 by
  construction), so any error is pure identity/association. Best-frame gallery is meaningless
  here (uniform confidence).
- **`yolo`** — the real YOLOv26 detector runs on the (possibly degraded) frame. This measures
  end-to-end / **detector** robustness.

**Severity:** 0 = clean (no corruption); higher = stronger. `grayscale`/`invert` are binary
(reported at a single level). Camera **shake** also warps the GT boxes by the same transform.

**Detector class handling (yolo only):** before scoring, `head`/`torso` predictions are
dropped and thermal classes are mapped to their RGB base, so detections match the
boat/human/motor GT. (Without this, head+torso are ~26k structural false positives that crush
precision.)

**Config:** ECC camera-motion compensation is **OFF** (matches the real-time TRT deployment;
verified to not change accuracy on these near-static-camera clips). ReID = OSNet (TensorRT).
GT is **partially labelled** (not every boat/person is annotated) — this is why *Recall* is the
trustworthy headline and *Precision/MOTA* must be read with caution.

**Metrics:**
- **Recall** = TP / (TP+FN): fraction of GT boxes found. **Most trustworthy here** — extra
  detections can't lower it, so partial GT doesn't bias it. Headline for the detector.
- **Precision** = TP / (TP+FP): *depressed by partial GT* (real-but-unlabelled boats count as
  false positives). Treat as soft, not comparable to literature.
- **IDF1** = identity F1 via a **global 1-to-1 match** of each GT track to the predicted ID it
  overlaps most across the whole clip. Measures how consistently the *correct* identity is held.
  A track that swaps ID once but keeps the new ID for most of the clip is mostly counted
  **correct** — only the shorter (pre-swap) segment is penalized. Headline for identity.
- **IDsw** = ID switches: counted **once** each time a GT track's matched prediction ID changes
  between consecutive frames. A sustained swap = 1 (not re-counted per frame). High on `yolo`
  mostly reflects **detection dropouts** (object buried by a wave → track dies → reappears with
  a new ID), *not* tracker association — compare with the much lower `gtdet` IDsw.
- **MOTA** = 1 − (FN+FP+IDsw)/GT: overall error rate; can go **negative** when FP is high
  (inflated here by partial GT). Soft metric.
- **HOTA** = geometric mean of detection accuracy (DetA) and association accuracy (AssA); a
  balanced single number, less sensitive to the FP/partial-GT issue than MOTA.
- **Frag** = track fragmentations (times a GT track's coverage is interrupted).

**Bottom line:** trust **Recall** (detection robustness) and **IDF1 / HOTA** (identity
robustness); read **Precision / MOTA** as soft (partial GT); **IDsw** counts events and on
`yolo` is dominated by detection dropouts, not association errors.
"""

# Binary corruptions (no severity ladder) — run once.
BINARY_KINDS = {"grayscale", "invert", "grayscale_invert"}

# Detector-class normalization for scoring (per user): ignore head & torso
# (rgb 4,5 + thermal 10,11), and map thermal detections to their rgb base
# (6->0,7->1,8->2,9->3). This is what lets grayscale/invert — which make the
# detector fire thermal-domain classes — still match the rgb GT. GT is already
# rgb {0,2,3} with no head/torso, so this only affects the detector (yolo) probe.
_DROP_CLS = {4, 5, 10, 11}


def normalize_pred(data):
    out = {}
    for fr, entries in data.items():
        kept = []
        for e in entries:
            cls = int(e[6])
            if cls in _DROP_CLS:
                continue
            if cls >= 6:
                cls -= 6                                # thermal -> rgb base
            kept.append(e[:6] + (cls,) + e[7:])
        out[fr] = kept
    return out


def video_stems(video_dir):
    return sorted(p.stem for p in Path(video_dir).iterdir()
                  if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"})


def _gen_one(t):
    src, vpath, kind, sev, seed, gt_in, gt_out = t
    n = degrade_video(src, vpath, kind, sev, seed, gt_in, gt_out)
    return Path(vpath).stem, n


def ensure_generated(kind, sev, videos, labels, aug_root, seed, workers=32):
    """Generate augmented videos+GT for (kind,sev) if missing, in parallel across
    clips (each clip is an independent cv2/numpy job). Returns (vdir, ldir)."""
    out = Path(aug_root) / f"{kind}_s{sev}"
    vdir, ldir = out / "videos", out / "labels"
    vdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for stem in video_stems(videos):
        vpath = vdir / f"{stem}.mp4"
        if vpath.exists():
            continue
        # NB: match by stem via iterdir, not glob — filenames contain '[...]' which
        # glob would interpret as a character class.
        src = next(p for p in Path(videos).iterdir() if p.stem == stem)
        gt_in = Path(labels) / f"{stem}.txt"
        gt_in = str(gt_in) if gt_in.exists() else None
        gt_out = str(ldir / f"{stem}.txt") if gt_in else None
        tasks.append((str(src), str(vpath), kind, sev, seed, gt_in, gt_out))
    if tasks:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        done = 0
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            futs = [ex.submit(_gen_one, t) for t in tasks]
            for f in as_completed(futs):
                stem, n = f.result()
                done += 1
                print(f"    gen {kind} s{sev} [{done}/{len(tasks)}] {stem[:48]}: {n} frames", flush=True)
    return str(vdir), str(ldir)


def run_inference(probe, video_dir, label_dir, out_dir, reid, det_weights, gtdet_weights,
                  expected, no_cmc=False):
    """Run track_video_predict for one probe. Clip-level resume: skip only when all
    `expected` clips already have MOT; otherwise invoke (track_video_predict itself
    skips per-clip via its timing CSV, so partial conditions resume cleanly)."""
    out_dir = Path(out_dir)
    mot_dir = out_dir / "mot"
    existing = len(list(mot_dir.glob("*.txt"))) if mot_dir.exists() else 0
    if expected and existing >= expected:
        print(f"    infer {probe}: SKIP (complete {existing}/{expected})", flush=True)
        return mot_dir
    cmd = [sys.executable, str(ROOT / "track" / "track_video_predict.py"),
           "--source", str(video_dir), "--out", str(out_dir),
           "--reid-weights", str(reid), "--save-mot", "--no-video"]
    if no_cmc:
        cmd += ["--no-cmc"]                              # disable ECC (camera-motion comp)
    if probe == "gtdet":
        cmd += ["--weights", str(gtdet_weights), "--gt-dets", str(label_dir)]
    else:
        cmd += ["--weights", str(det_weights)]
    print(f"    infer {probe}: running ({existing}/{expected} done)...", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "infer.log", "w") as lf:
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=lf, stderr=subprocess.STDOUT)
    return mot_dir


def score(label_dir, mot_dir):
    """Score each clip (GT vs tracker MOT). Returns list of per-clip metric dicts."""
    rows = []
    for gt_file in sorted(Path(label_dir).glob("*.txt")):
        pred_file = Path(mot_dir) / gt_file.name
        if not pred_file.exists():
            continue
        gt_data, pred_data = load_mot(str(gt_file)), normalize_pred(load_mot(str(pred_file)))
        metrics, _ = evaluate_sequence_custom(gt_data, pred_data, 0.5)
        metrics["clip"] = gt_file.stem
        rows.append(metrics)
    return rows


def mean(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", nargs="+", default=["lowlight", "jpeg", "shake"])
    ap.add_argument("--sevs", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--probes", nargs="+", default=["gtdet", "yolo"], choices=["gtdet", "yolo"])
    ap.add_argument("--videos", default="eval_videos/wavy-boats/videos")
    ap.add_argument("--labels", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--aug-root", default="eval_videos/wavy-boats/aug")
    ap.add_argument("--out", default="results/robustness")
    ap.add_argument("--reid", default="export/engines/osnet_x0_25_msmt17.engine")
    ap.add_argument("--det-weights", default="weights/best.engine")
    ap.add_argument("--gtdet-weights", default="weights/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-workers", type=int, default=32,
                    help="parallel processes for augment generation (CPU-bound)")
    ap.add_argument("--no-cmc", action="store_true",
                    help="disable ECC camera-motion compensation (~3x faster; matches the "
                         "ECC-off+TRT real-time deployment config)")
    ap.add_argument("--cleanup-aug", action="store_true",
                    help="delete each condition's augmented videos after inference "
                         "(keeps GT + MOT); bounds disk for large eval sets")
    args = ap.parse_args()

    videos, labels = ROOT / args.videos, ROOT / args.labels
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # conditions: clean baseline (severity 0) + (kind × severity); binary kinds
    # (grayscale/invert) run once at severity 1.
    conditions = [("clean", 0)]
    for k in args.kinds:
        if k in BINARY_KINDS:
            conditions.append((k, 1))
        else:
            conditions += [(k, s) for s in args.sevs]

    all_rows = []  # flat per-clip rows for CSV
    # agg[(probe, kind)] = {sev: {metric: mean}}
    agg = {}

    n_src = len(video_stems(videos))                    # full source clip count
    for cond_kind, sev in conditions:
        cond_name = "clean" if cond_kind == "clean" else f"{cond_kind}_s{sev}"
        if cond_kind == "clean":
            vdir, ldir = str(videos), str(labels)
            label_keys = [(k, 0) for k in args.kinds]   # clean is the sev-0 point for every kind
        else:
            ldir = str(ROOT / args.aug_root / cond_name / "labels")
            vdir = str(ROOT / args.aug_root / cond_name / "videos")
            label_keys = [(cond_kind, sev)]

        # completion reference: GT label files are kept even after --cleanup-aug, so
        # use them (fallback to source count) to decide if regeneration is needed.
        n_lab = len(list(Path(ldir).glob("*.txt"))) if Path(ldir).exists() else 0
        expected = n_lab or n_src
        done = all(
            (out_root / p / cond_name / "mot").exists()
            and len(list((out_root / p / cond_name / "mot").glob("*.txt"))) >= expected
            for p in args.probes)

        if cond_kind != "clean" and not done:
            print(f"[generate] {cond_name}", flush=True)
            ensure_generated(cond_kind, sev, videos, labels, ROOT / args.aug_root,
                             args.seed, args.gen_workers)
            n_lab = len(list(Path(ldir).glob("*.txt")))
            expected = n_lab or n_src

        for probe in args.probes:
            out_dir = out_root / probe / cond_name
            print(f"[{probe}] {cond_name}", flush=True)
            try:
                mot_dir = run_inference(probe, vdir, ldir, out_dir, ROOT / args.reid,
                                        ROOT / args.det_weights, ROOT / args.gtdet_weights,
                                        expected, args.no_cmc)
                rows = score(ldir, mot_dir)
            except Exception as e:
                print(f"    !! {probe} {cond_name} FAILED: {e} (see {out_dir/'infer.log'}); continuing", flush=True)
                continue
            for r in rows:
                r.update({"probe": probe, "kind": cond_kind, "severity": sev})
                all_rows.append(r)
            summary = {m: mean(rows, m) for m in FOCUS_METRICS}
            print("    " + " | ".join(f"{m} {summary[m]:.3f}" for m in FOCUS_METRICS
                                      if summary[m] == summary[m]), flush=True)
            for (k, s) in label_keys:
                agg.setdefault((probe, k), {})[s] = summary

        # free disk: drop this condition's aug videos once both probes are done
        if args.cleanup_aug and cond_kind != "clean":
            import shutil
            shutil.rmtree(Path(ROOT / args.aug_root) / f"{cond_kind}_s{sev}" / "videos",
                          ignore_errors=True)
            print(f"    cleanup: removed {cond_kind}_s{sev}/videos", flush=True)

    # ---- write CSV ----
    keys = sorted({k for r in all_rows for k in r})
    head = ["probe", "kind", "severity", "clip"] + [k for k in keys if k not in
                                                     ("probe", "kind", "severity", "clip")]
    csv_path = out_root / "metrics.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(head) + "\n")
        for r in all_rows:
            f.write(",".join(str(r.get(c, "")) for c in head) + "\n")

    # ---- write markdown summary (metric vs severity per kind/probe) ----
    md = ["# Corruption-robustness sweep\n",
          f"Probes: {args.probes} | kinds: {args.kinds} | severities: {[0]+args.sevs} "
          f"(0 = clean) | ReID: {Path(args.reid).name} | "
          f"ECC: {'off' if args.no_cmc else 'on'}\n",
          GLOSSARY]
    for probe in args.probes:
        focus = ["Recall", "Precision"] if probe == "yolo" else ["IDF1", "MOTA", "IDsw", "HOTA"]
        md.append(f"\n## Probe: {probe}  (metrics: {', '.join(focus)})\n")
        for kind in args.kinds:
            data = agg.get((probe, kind), {})
            sevs = sorted(data)
            md.append(f"\n### {kind}\n")
            md.append("| severity | " + " | ".join(focus) + " |")
            md.append("|" + "---|" * (len(focus) + 1))
            for s in sevs:
                cells = [f"{data[s].get(m, float('nan')):.3f}" for m in focus]
                md.append(f"| {s} | " + " | ".join(cells) + " |")
    (out_root / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\nDone. metrics.csv + summary.md -> {out_root}")


if __name__ == "__main__":
    main()
