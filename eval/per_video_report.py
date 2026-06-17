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

# display order of conditions (clean first)
COND_ORDER = ["clean", "lowlight_s2", "lowlight_s4", "jpeg_s2", "jpeg_s4",
              "shake_s2", "shake_s4", "grayscale_s1", "invert_s1"]
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


def cond_key(kind, sev):
    name = "clean" if kind == "clean" else f"{kind}_s{sev}"
    return name, (COND_ORDER.index(name) if name in COND_ORDER else 99)


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

    by_clip = {}
    for r in rows:
        name, order = cond_key(r["kind"], r["severity"])
        by_clip.setdefault(r["clip"], []).append((r["probe"], name, order, r))

    md = ["# Per-video metric report\n",
          f"{len(by_clip)} videos · {len(metrics)} metrics · probes gtdet/yolo "
          f"· conditions: {', '.join(COND_ORDER)}\n", GLOSSARY_NOTE]

    for clip in sorted(by_clip):
        md.append(f"\n## {clip}\n")
        md.append("| probe | condition | " + " | ".join(metrics) + " |")
        md.append("|---|---|" + "---|" * len(metrics))
        entries = sorted(by_clip[clip], key=lambda e: (e[0], e[2]))
        for probe, name, _, r in entries:
            cells = [fmt(m, r.get(m, "")) for m in metrics]
            md.append(f"| {probe} | {name} | " + " | ".join(cells) + " |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"{len(by_clip)} videos -> {args.out}")


if __name__ == "__main__":
    main()
