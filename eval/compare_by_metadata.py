"""Compare tracker experiments against baseline for a filtered subset of clips.

Usage:
    conda run -n boat-tracker python eval/compare_by_metadata.py
    conda run -n boat-tracker python eval/compare_by_metadata.py --non-interactive \
        --filter domain=thermal lighting=dark --experiments exp09_nms exp10_nms_alpha0.5 \
        --out results/analysis/thermal_dark
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT       = Path(__file__).parent.parent
META_PATH       = REPO_ROOT / "data/eval/vws_eval_set_metadata.json"
EXPERIMENTS_DIR = REPO_ROOT / "results/experiments"
BASELINE_NAME   = "exp00_baseline"

METRICS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "Precision", "Recall",
           "IDsw", "FP", "FN", "Frag"]
ERROR_METRICS = {"IDsw", "FP", "FN", "Frag"}   # lower = better


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(exp_name):
    path = EXPERIMENTS_DIR / exp_name / "eval" / "tracking_metrics_mot.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def delta_str(d, metric):
    better = (d < 0) if metric in ERROR_METRICS else (d > 0)
    worse  = (d > 0) if metric in ERROR_METRICS else (d < 0)
    symbol = " ↑" if better else (" ↓" if worse else "")
    if isinstance(d, float):
        return f"{d:+.3f}{symbol}"
    return f"{d:+d}{symbol}"


def available_experiments():
    return sorted(
        d.name for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir()
        and d.name != BASELINE_NAME
        and (d / "eval" / "tracking_metrics_mot.json").exists()
    )


def available_filter_values(meta):
    """Return {field: sorted list of unique values} for filterable fields."""
    fields = ["domain", "lighting", "cross_modal", "multi_object", "camera"]
    result = {}
    for f in fields:
        vals = sorted({str(m[f]) for m in meta.values() if m.get(f) is not None})
        result[f] = vals
    return result


def apply_filters(meta, filters):
    """Return list of clip stems matching all filters.

    filters: {field: set of string values to include}
    """
    matched = []
    for filepath, m in meta.items():
        clip_stem = Path(filepath).stem
        ok = True
        for field, allowed in filters.items():
            val = m.get(field)
            if val is None:
                ok = False
                break
            if str(val).lower() not in allowed:
                ok = False
                break
        if ok:
            matched.append(clip_stem)
    return sorted(matched)


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def prompt(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"{msg}{suffix}: ").strip()
    except EOFError:
        print("\nERROR: stdin is not available (conda run blocks interactive input).")
        print("For interactive mode, activate the environment first:\n")
        print("  conda activate boat-tracker")
        print("  python eval/compare_by_metadata.py\n")
        print("Or use non-interactive mode:")
        print("  conda run -n boat-tracker python eval/compare_by_metadata.py --non-interactive \\")
        print("      --filter domain=thermal --experiments exp10_nms_alpha0.5 --out results/analysis/out\n")
        sys.exit(1)
    return raw if raw else default


def pick_from_list(options, label, multi=True):
    """Show numbered list, return selected items."""
    print(f"\n{label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if multi:
        raw = prompt("Select (comma-separated numbers, or Enter to skip)", default="")
    else:
        raw = prompt("Select number", default="1")
    if not raw:
        return []
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(options):
                selected.append(options[idx])
    return selected


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_comparison(baseline_data, exp_data_map, clips):
    """Return list of per-clip comparison dicts."""
    rows = []
    baseline_seqs = baseline_data.get("per_sequence", {})

    for clip in clips:
        b = baseline_seqs.get(clip)
        if b is None:
            print(f"  WARNING: {clip} not in baseline, skipping")
            continue

        row = {"clip": clip, "experiments": {}}
        for exp_name, exp_data in exp_data_map.items():
            e = exp_data.get("per_sequence", {}).get(clip)
            if e is None:
                print(f"  WARNING: {clip} not in {exp_name}, skipping experiment")
                continue
            row["experiments"][exp_name] = {
                m: {"baseline": b.get(m), "experiment": e.get(m),
                    "delta": (e.get(m, 0) - b.get(m, 0))}
                for m in METRICS
            }
        rows.append(row)
    return rows


def write_markdown(rows, exp_names, baseline_name, filters, out_path):
    filter_desc = ", ".join(f"{k}={'+'.join(sorted(v))}" for k, v in filters.items()) or "none"
    lines = [
        f"# Experiment Comparison Report\n",
        f"**Baseline:** {baseline_name}  ",
        f"**Experiments:** {', '.join(exp_names)}  ",
        f"**Filters:** {filter_desc}  ",
        f"**Clips matched:** {len(rows)}\n",
    ]

    for row in rows:
        lines.append(f"\n## {row['clip']}\n")
        header = "| Metric | Baseline |" + "".join(f" {e} | Δ |" for e in exp_names)
        sep    = "|--------|----------|" + "".join("---------|---|" for _ in exp_names)
        lines += [header, sep]
        for m in METRICS:
            b_val = row["experiments"].get(exp_names[0], {}).get(m, {}).get("baseline")
            cells = f"| {m} | {fmt(b_val)} |"
            for exp_name in exp_names:
                ed = row["experiments"].get(exp_name, {}).get(m)
                if ed:
                    cells += f" {fmt(ed['experiment'])} | {delta_str(ed['delta'], m)} |"
                else:
                    cells += " — | — |"
            lines.append(cells)

    # Averages
    if rows:
        lines.append(f"\n## Averages across {len(rows)} clip(s)\n")
        header = "| Metric | Baseline |" + "".join(f" {e} | Δ |" for e in exp_names)
        sep    = "|--------|----------|" + "".join("---------|---|" for _ in exp_names)
        lines += [header, sep]
        for m in METRICS:
            b_vals = [r["experiments"].get(exp_names[0], {}).get(m, {}).get("baseline", 0)
                      for r in rows if exp_names[0] in r["experiments"]]
            b_avg = sum(b_vals) / len(b_vals) if b_vals else 0
            cells = f"| {m} | {fmt(b_avg)} |"
            for exp_name in exp_names:
                e_vals = [r["experiments"].get(exp_name, {}).get(m, {}).get("experiment", 0)
                          for r in rows if exp_name in r["experiments"]]
                d_vals = [r["experiments"].get(exp_name, {}).get(m, {}).get("delta", 0)
                          for r in rows if exp_name in r["experiments"]]
                if e_vals:
                    e_avg = sum(e_vals) / len(e_vals)
                    d_avg = sum(d_vals) / len(d_vals)
                    cells += f" {fmt(e_avg)} | {delta_str(d_avg, m)} |"
                else:
                    cells += " — | — |"
            lines.append(cells)

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Report:  {out_path}")


def write_json(rows, exp_names, baseline_name, filters, out_path):
    out_path.write_text(json.dumps({
        "baseline": baseline_name,
        "experiments": exp_names,
        "filters": {k: sorted(v) for k, v in filters.items()},
        "clip_count": len(rows),
        "clips": rows,
    }, indent=2))
    print(f"  JSON:    {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_interactive():
    print("\n=== Experiment Comparison Tool ===\n")

    # Load metadata
    if not META_PATH.exists():
        sys.exit(f"ERROR: metadata not found at {META_PATH}")
    meta = json.loads(META_PATH.read_text())
    filter_options = available_filter_values(meta)

    # --- Step 1: pick filters ---
    print("Step 1: Select metadata fields to filter on")
    field_names = list(filter_options.keys())
    selected_fields = pick_from_list(field_names, "Available filter fields")

    filters = {}
    for field in selected_fields:
        vals = filter_options[field]
        print(f"\nFilter: {field}")
        chosen = pick_from_list(vals, f"  Available values for '{field}'")
        if chosen:
            filters[field] = {v.lower() for v in chosen}

    clips = apply_filters(meta, filters)
    if not clips:
        sys.exit("No clips matched the selected filters. Exiting.")
    print(f"\n  → {len(clips)} clip(s) matched\n")

    # --- Step 2: pick experiments ---
    print("Step 2: Select experiments to compare against baseline "
          f"({BASELINE_NAME})")
    exps = available_experiments()
    selected_exps = pick_from_list(exps, "Available experiments")
    if not selected_exps:
        sys.exit("No experiments selected. Exiting.")

    # --- Step 3: output directory ---
    filter_slug = "_".join(
        f"{k}-{'_'.join(sorted(v))}" for k, v in sorted(filters.items())
    ) or "all"
    default_out = str(REPO_ROOT / "results/analysis" / filter_slug)
    out_dir = Path(prompt("\nStep 3: Output directory", default=default_out))

    return filters, clips, selected_exps, out_dir


def run_non_interactive(args):
    meta = json.loads(META_PATH.read_text())

    filters = {}
    for f in (args.filter or []):
        if "=" not in f:
            sys.exit(f"ERROR: filter '{f}' must be in field=value format")
        field, val = f.split("=", 1)
        filters.setdefault(field, set()).add(val.lower())

    clips = apply_filters(meta, filters)
    if not clips:
        sys.exit("No clips matched the selected filters. Exiting.")
    print(f"  → {len(clips)} clip(s) matched")

    selected_exps = args.experiments or []
    if not selected_exps:
        sys.exit("No experiments specified. Use --experiments.")

    out_dir = Path(args.out) if args.out else REPO_ROOT / "results/analysis/comparison"
    return filters, clips, selected_exps, out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run without prompts (requires --filter and --experiments)")
    parser.add_argument("--filter", nargs="*", metavar="FIELD=VALUE",
                        help="Metadata filters e.g. domain=thermal lighting=dark")
    parser.add_argument("--experiments", nargs="*", metavar="EXP",
                        help="Experiment names to compare e.g. exp09_nms exp10_nms_alpha0.5")
    parser.add_argument("--out", default=None,
                        help="Output directory")
    args = parser.parse_args()

    if args.non_interactive:
        filters, clips, selected_exps, out_dir = run_non_interactive(args)
    else:
        filters, clips, selected_exps, out_dir = run_interactive()

    # --- Load data ---
    baseline_data = load_metrics(BASELINE_NAME)
    if baseline_data is None:
        sys.exit(f"ERROR: baseline eval not found for {BASELINE_NAME}")

    exp_data_map = {}
    for exp_name in selected_exps:
        data = load_metrics(exp_name)
        if data is None:
            print(f"  WARNING: no eval results for {exp_name}, skipping")
        else:
            exp_data_map[exp_name] = data

    if not exp_data_map:
        sys.exit("No valid experiment data found. Exiting.")

    # --- Build comparison ---
    print(f"\nComparing {len(clips)} clip(s) across "
          f"{len(exp_data_map)} experiment(s)...\n")
    rows = build_comparison(baseline_data, exp_data_map, clips)

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_slug = "_vs_".join(exp_data_map.keys())
    write_markdown(rows, list(exp_data_map.keys()), BASELINE_NAME, filters,
                   out_dir / f"report_{exp_slug}.md")
    write_json(rows, list(exp_data_map.keys()), BASELINE_NAME, filters,
               out_dir / f"comparison_{exp_slug}.json")

    print(f"\nDone. {len(rows)} clips saved to {out_dir}/")


if __name__ == "__main__":
    main()
