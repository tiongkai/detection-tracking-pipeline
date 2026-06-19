"""Compare best-frame vs FIFO ReID gallery across all augmented conditions (yolo probe).

fifo = main sweep (results/robustness_all/yolo); best = results/robustness_gallery_best/yolo.
Both ECC-off, OSNet-TRT, scored against the same GT (clean labels, or aug labels for the
condition). Writes a per-condition fifo-vs-best delta table.

    .venv/bin/python eval/gallery_sweep_compare.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_tracking import load_mot, evaluate_sequence_custom   # noqa: E402
from robustness_sweep import normalize_pred                    # noqa: E402

METRICS = ["Recall", "IDF1", "IDsw", "Frag", "HOTA", "AssA"]
CONDS = ["clean", "lowlight_s2", "lowlight_s4", "jpeg_s2", "jpeg_s4", "shake_s2",
         "shake_s4", "grayscale_s1", "invert_s1", "grayscale_invert_s1"]


def score(gt_dir, mot_dir):
    per = []
    for gt in sorted(Path(gt_dir).glob("*.txt")):
        pr = Path(mot_dir) / gt.name
        if not pr.exists():
            continue
        m, _ = evaluate_sequence_custom(load_mot(str(gt)), normalize_pred(load_mot(str(pr))), 0.5)
        per.append(m)
    n = len(per)
    return {k: (sum(p[k] for p in per) / n if n else 0.0) for k in METRICS}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--aug-root", default="eval_videos/wavy-boats/aug_best")
    ap.add_argument("--fifo", default="results/robustness_all/yolo")
    ap.add_argument("--best", default="results/robustness_gallery_best/yolo")
    ap.add_argument("--out", default="docs/degradation_eval_reports/gallery_best_vs_fifo_augmented.md")
    args = ap.parse_args()

    md = ["# Best-frame vs FIFO ReID gallery — across augmented conditions (yolo)\n",
          "Both ECC-off, OSNet-TRT. Delta = best - fifo. Positive IDF1/HOTA/AssA = best-frame helps.\n",
          "| condition | IDF1 fifo | IDF1 best | dIDF1 | dHOTA | dAssA | dIDsw | dRecall |",
          "|---|---|---|---|---|---|---|---|"]
    print(f"{'condition':20} {'IDF1 fifo':>9} {'best':>7} {'dIDF1':>7} {'dHOTA':>7} {'dAssA':>7} {'dIDsw':>7}")
    for c in CONDS:
        gt = args.labels if c == "clean" else f"{args.aug_root}/{c}/labels"
        f, nf = score(gt, f"{args.fifo}/{c}/mot")
        b, nb = score(gt, f"{args.best}/{c}/mot")
        if nf == 0 or nb == 0:
            print(f"{c:20} (missing: fifo {nf}, best {nb})")
            continue
        d = {k: b[k] - f[k] for k in METRICS}
        print(f"{c:20} {f['IDF1']:9.3f} {b['IDF1']:7.3f} {d['IDF1']:+7.3f} "
              f"{d['HOTA']:+7.3f} {d['AssA']:+7.3f} {d['IDsw']:+7.1f}")
        md.append(f"| {c} | {f['IDF1']:.3f} | {b['IDF1']:.3f} | {d['IDF1']:+.3f} | "
                  f"{d['HOTA']:+.3f} | {d['AssA']:+.3f} | {d['IDsw']:+.1f} | {d['Recall']:+.3f} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
