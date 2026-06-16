"""Compare best-frame vs FIFO ReID gallery on the clean eval set (yolo probe).

The fifo baseline is the existing yolo/clean run from robustness_sweep. This runs
the best-frame gallery on the same clips and scores both, so the only difference is
the gallery policy: keep the last-K embeddings (fifo) vs the top-K highest-confidence
embeddings (best). Best-frame needs varying detection confidence to discriminate, so
this is the yolo probe (real detections), not gtdet (uniform conf).

    .venv/bin/python eval/gallery_compare.py \
        --videos eval_videos/wavy-boats/videos \
        --labels eval_videos/wavy-boats/labels \
        --fifo-mot results/robustness_all/yolo/clean/mot \
        --best-out results/gallery_compare/best_clean
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_tracking import load_mot, evaluate_sequence_custom   # noqa: E402
from robustness_sweep import normalize_pred                    # noqa: E402

METRICS = ["Recall", "Precision", "IDF1", "IDsw", "Frag", "MOTA", "HOTA"]


def score_dir(gt_dir, mot_dir):
    per = []
    for gt_file in sorted(Path(gt_dir).glob("*.txt")):
        pred = Path(mot_dir) / gt_file.name
        if not pred.exists():
            continue
        m, _ = evaluate_sequence_custom(load_mot(str(gt_file)),
                                        normalize_pred(load_mot(str(pred))), 0.5)
        per.append(m)
    n = len(per)
    return {k: (sum(p[k] for p in per) / n if n else float("nan")) for k in METRICS}, n


def run_best(videos, out, reid, det_weights, k, div):
    mot = Path(out) / "mot"
    expected = len([p for p in Path(videos).iterdir() if p.suffix.lower() == ".mp4"])
    if mot.exists() and len(list(mot.glob("*.txt"))) >= expected:
        print(f"best: SKIP (complete {len(list(mot.glob('*.txt')))}/{expected})")
        return mot
    cmd = [sys.executable, str(ROOT / "track" / "track_video_predict.py"),
           "--weights", str(det_weights), "--source", str(videos), "--out", str(out),
           "--reid-weights", str(reid), "--reid-gallery", "best",
           "--gallery-k", str(k), "--gallery-diversity", str(div),
           "--save-mot", "--no-video"]
    print("best: running...", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / "infer.log", "w") as lf:
        subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=lf, stderr=subprocess.STDOUT)
    return mot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="eval_videos/wavy-boats/videos")
    ap.add_argument("--labels", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--fifo-mot", default="results/robustness_all/yolo/clean/mot")
    ap.add_argument("--best-out", default="results/gallery_compare/best_clean")
    ap.add_argument("--reid", default="export/engines/osnet_x0_25_msmt17.engine")
    ap.add_argument("--det-weights", default="weights/best.engine")
    ap.add_argument("--gallery-k", type=int, default=12)
    ap.add_argument("--gallery-diversity", type=float, default=0.10)
    args = ap.parse_args()

    run_best(ROOT / args.videos, ROOT / args.best_out, ROOT / args.reid,
             ROOT / args.det_weights, args.gallery_k, args.gallery_diversity)

    fifo, nf = score_dir(ROOT / args.labels, ROOT / args.fifo_mot)
    best, nb = score_dir(ROOT / args.labels, Path(ROOT / args.best_out) / "mot")

    print(f"\nfifo: {nf} clips | best: {nb} clips (k={args.gallery_k}, div={args.gallery_diversity})")
    print(f"{'metric':<10} {'fifo':>9} {'best':>9} {'delta':>9}")
    md = ["# ReID gallery: best-frame vs FIFO (clean eval set, yolo probe)\n",
          f"k={args.gallery_k}, diversity={args.gallery_diversity}, clips={nf}\n",
          "| metric | fifo | best | delta |", "|---|---|---|---|"]
    for k in METRICS:
        d = best[k] - fifo[k]
        print(f"{k:<10} {fifo[k]:>9.3f} {best[k]:>9.3f} {d:>+9.3f}")
        md.append(f"| {k} | {fifo[k]:.3f} | {best[k]:.3f} | {d:+.3f} |")
    out_md = Path(ROOT / args.best_out).parent / "gallery_compare_clean.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"\n-> {out_md}")


if __name__ == "__main__":
    main()
