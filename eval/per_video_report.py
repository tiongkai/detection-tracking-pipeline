"""Per-video detailed metric report, from a sweep's metrics.csv.

One section per clip; one table per clip with a row for each (probe, condition) and
a column for every metric. This is the human-readable rollup of metrics.csv (which
holds the same numbers as raw rows).

    .venv/bin/python eval/per_video_report.py \
        --csv results/robustness_all/metrics.csv \
        --out results/robustness_all/per_video_report.md
"""
import argparse
import csv
from pathlib import Path

# corruption groups: each is shown with the clean baseline first, then its severities
KIND_GROUPS = [("lowlight", [2, 4]), ("jpeg", [2, 4]), ("shake", [2, 4]),
               ("grayscale", [1]), ("invert", [1]), ("grayscale_invert", [1])]
INT_COLS = {"TP", "FP", "FN", "IDsw", "Frag", "MT", "ML", "PT", "GT_tracks",
            "IDTP", "IDFP", "IDFN"}
# column display order (identity-ish first, then detection counts)
METRIC_ORDER = ["GT_tracks", "MT", "PT", "ML", "Recall", "Precision",
                "IDF1", "IDsw", "Frag", "MOTA", "HOTA", "DetA", "AssA",
                "MOTP", "TP", "FP", "FN", "IDTP", "IDFP", "IDFN"]

GLOSSARY_NOTE = (
    "_Recall_ = trustworthy headline (partial GT). _Precision/MOTA_ = soft (partial GT). "
    "_IDF1/HOTA/AssA_ = identity quality (sustained swaps mostly forgiven). _IDsw_ = swap "
    "events (high on yolo = detection dropouts). `gtdet` = perfect boxes (isolates tracking/"
    "ReID); `yolo` = real detector. See summary.md for full glossary.\n"
)


def fmt(col, v):
    if v in ("", None):
        return ""
    try:
        f = float(v)
    except ValueError:
        return str(v)
    return str(int(round(f))) if col in INT_COLS else f"{f:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/robustness_all/metrics.csv")
    ap.add_argument("--out", default="results/robustness_all/per_video_report.md")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    all_cols = rows[0].keys() if rows else []
    metrics = [m for m in METRIC_ORDER if m in all_cols]

    # index: (clip, probe, condname) -> row ; condname is "clean" or "<kind>_s<sev>"
    idx = {}
    clips, probes = set(), set()
    for r in rows:
        name = "clean" if r["kind"] == "clean" else f"{r['kind']}_s{r['severity']}"
        idx[(r["clip"], r["probe"], name)] = r
        clips.add(r["clip"]); probes.add(r["probe"])
    probes = [p for p in ("gtdet", "yolo") if p in probes]

    md = ["# Per-video metric report\n",
          f"{len(clips)} videos · each corruption shown with its clean baseline "
          f"(severity 0)\n", GLOSSARY_NOTE]

    for clip in sorted(clips):
        md.append(f"\n## {clip}\n")
        for kind, sevs in KIND_GROUPS:
            md.append(f"\n### {kind}\n")
            md.append("| condition | probe | " + " | ".join(metrics) + " |")
            md.append("|---|---|" + "---|" * len(metrics))
            for label, cond in [("clean", "clean")] + [(f"{kind}_s{s}", f"{kind}_s{s}") for s in sevs]:
                for probe in probes:                     # gtdet then yolo, stacked
                    r = idx.get((clip, probe, cond))
                    if not r:
                        continue
                    cells = [fmt(m, r.get(m, "")) for m in metrics]
                    md.append(f"| {label} | {probe} | " + " | ".join(cells) + " |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"{len(clips)} videos -> {args.out}")


if __name__ == "__main__":
    main()
