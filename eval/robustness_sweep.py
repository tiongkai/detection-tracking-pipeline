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
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from degrade import degrade_video                       # noqa: E402
from eval_tracking import load_mot, evaluate_sequence_custom  # noqa: E402

FOCUS_METRICS = ["Recall", "Precision", "IDF1", "IDsw", "MOTA", "HOTA"]


def video_stems(video_dir):
    return sorted(p.stem for p in Path(video_dir).iterdir()
                  if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"})


def ensure_generated(kind, sev, videos, labels, aug_root, seed):
    """Generate augmented videos+GT for (kind,sev) if missing. Returns (vdir, ldir)."""
    out = Path(aug_root) / f"{kind}_s{sev}"
    vdir, ldir = out / "videos", out / "labels"
    for stem in video_stems(videos):
        vpath = vdir / f"{stem}.mp4"
        if vpath.exists():
            continue
        # NB: match by stem via iterdir, not glob — filenames contain '[...]' which
        # glob would interpret as a character class.
        src = next(p for p in Path(videos).iterdir() if p.stem == stem)
        gt_in = Path(labels) / f"{stem}.txt"
        gt_in = gt_in if gt_in.exists() else None
        gt_out = (ldir / f"{stem}.txt") if gt_in else None
        n = degrade_video(src, vpath, kind, sev, seed, gt_in, gt_out)
        print(f"    gen {kind} s{sev} {stem}: {n} frames", flush=True)
    return str(vdir), str(ldir)


def run_inference(probe, video_dir, label_dir, out_dir, reid, det_weights, gtdet_weights):
    """Run track_video_predict for one probe over a video dir (skip if MOT exists)."""
    mot_dir = Path(out_dir) / "mot"
    if mot_dir.exists() and any(mot_dir.glob("*.txt")):
        print(f"    infer {probe}: SKIP (MOT exists)", flush=True)
        return mot_dir
    cmd = [sys.executable, str(ROOT / "track" / "track_video_predict.py"),
           "--source", str(video_dir), "--out", str(out_dir),
           "--reid-weights", str(reid), "--save-mot", "--no-video"]
    if probe == "gtdet":
        cmd += ["--weights", str(gtdet_weights), "--gt-dets", str(label_dir)]
    else:
        cmd += ["--weights", str(det_weights)]
    print(f"    infer {probe}: running...", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mot_dir


def score(label_dir, mot_dir):
    """Score each clip (GT vs tracker MOT). Returns list of per-clip metric dicts."""
    rows = []
    for gt_file in sorted(Path(label_dir).glob("*.txt")):
        pred_file = Path(mot_dir) / gt_file.name
        if not pred_file.exists():
            continue
        gt_data, pred_data = load_mot(str(gt_file)), load_mot(str(pred_file))
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
    args = ap.parse_args()

    videos, labels = ROOT / args.videos, ROOT / args.labels
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # conditions: clean baseline (severity 0) + (kind × severity)
    conditions = [("clean", 0)] + list(product(args.kinds, args.sevs))

    all_rows = []  # flat per-clip rows for CSV
    # agg[(probe, kind)] = {sev: {metric: mean}}
    agg = {}

    for cond_kind, sev in conditions:
        if cond_kind == "clean":
            vdir, ldir = str(videos), str(labels)
            label_keys = [(k, 0) for k in args.kinds]   # clean is the sev-0 point for every kind
        else:
            print(f"[generate] {cond_kind} s{sev}", flush=True)
            vdir, ldir = ensure_generated(cond_kind, sev, videos, labels, ROOT / args.aug_root, args.seed)
            label_keys = [(cond_kind, sev)]

        for probe in args.probes:
            cond_name = f"{cond_kind}_s{sev}" if cond_kind != "clean" else "clean"
            out_dir = out_root / probe / cond_name
            print(f"[{probe}] {cond_name}", flush=True)
            mot_dir = run_inference(probe, vdir, ldir, out_dir, ROOT / args.reid,
                                    ROOT / args.det_weights, ROOT / args.gtdet_weights)
            rows = score(ldir, mot_dir)
            for r in rows:
                r.update({"probe": probe, "kind": cond_kind, "severity": sev})
                all_rows.append(r)
            summary = {m: mean(rows, m) for m in FOCUS_METRICS}
            print("    " + " | ".join(f"{m} {summary[m]:.3f}" for m in FOCUS_METRICS
                                      if summary[m] == summary[m]), flush=True)
            for (k, s) in label_keys:
                agg.setdefault((probe, k), {})[s] = summary

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
          f"(0 = clean) | ReID: {Path(args.reid).name}\n"]
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
