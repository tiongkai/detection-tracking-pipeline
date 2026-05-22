"""Interactive experiment comparison tool.

Scans results/experiments/ for exp<NN>_* directories, prompts for which
experiments to compare, and prints a diff table with arrows showing
improvement (green) or regression (red).

Usage:
    python eval/compare_experiments.py
"""
import json
import re
import sys
from pathlib import Path

# ── terminal colours ──────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"

# ── metric metadata ───────────────────────────────────────────────────────────
# (display_name, higher_is_better, is_float)
OVERALL_METRICS = [
    ("HOTA",      True,  True),
    ("DetA",      True,  True),
    ("AssA",      True,  True),
    ("MOTA",      True,  True),
    ("IDF1",      True,  True),
    ("Precision", True,  True),
    ("Recall",    True,  True),
    ("IDsw",      False, False),
    ("Frag",      False, False),
    ("MT",        True,  False),
    ("ML",        False, False),
    ("GT_tracks", None,  False),   # informational only
]

SEQ_METRICS = [
    ("HOTA",  True,  True),
    ("MOTA",  True,  True),
    ("IDF1",  True,  True),
    ("DetA",  True,  True),
    ("AssA",  True,  True),
    ("IDsw",  False, False),
    ("Frag",  False, False),
    ("MT",    True,  False),
    ("ML",    False, False),
]

EXPERIMENTS_DIR = Path(__file__).parent.parent / "results" / "experiments"


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_experiments():
    """Return sorted list of (exp_number, dir_path) tuples."""
    exps = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"exp(\d+)", d.name)
        if m:
            exps.append((int(m.group(1)), d))
    return exps


def load_experiment(exp_dir):
    """Load the first tracking_metrics_*.json found in exp_dir/eval/."""
    eval_dir = exp_dir / "eval"
    jsons = sorted(eval_dir.glob("tracking_metrics_*.json"))
    if not jsons:
        return None
    return json.loads(jsons[0].read_text())


def load_config(exp_dir):
    """Return config description string from the experiment YAML if present."""
    yamls = sorted(exp_dir.glob("*.yaml"))
    if not yamls:
        return ""
    try:
        # minimal parse — just grab name/description lines
        lines = yamls[0].read_text().splitlines()
        info = {}
        for line in lines:
            for key in ("name", "description"):
                if line.startswith(f"{key}:"):
                    info[key] = line.split(":", 1)[1].strip()
        parts = [info.get("name", ""), info.get("description", "")]
        return "  |  ".join(p for p in parts if p)
    except Exception:
        return ""


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_val(val, is_float):
    if is_float:
        return f"{val:.3f}"
    return str(int(val))


def fmt_delta(delta, is_float, higher_is_better):
    """Return coloured delta string with arrow."""
    if delta == 0:
        return "  —"

    if is_float:
        sign = "+" if delta > 0 else ""
        ds = f"{sign}{delta:.3f}"
    else:
        sign = "+" if delta > 0 else ""
        ds = f"{sign}{int(delta)}"

    if higher_is_better is None:
        arrow = ""
        colourise = lambda s: s
    elif (delta > 0) == higher_is_better:
        arrow = " ↑"
        colourise = green
    else:
        arrow = " ↓"
        colourise = red

    return colourise(f"{ds}{arrow}")


def col_width(strings, minimum=6):
    return max(minimum, max(len(s) for s in strings))


# ── printing ──────────────────────────────────────────────────────────────────

def print_overall(exp_data):
    """Print overall metrics comparison table."""
    names  = [d["name"] for d in exp_data]
    data   = [d["data"]["overall"] for d in exp_data]

    # column widths
    metric_w = max(len(m) for m, _, _ in OVERALL_METRICS)
    val_cols  = []
    for i, nd in enumerate(zip(names, data)):
        name, d = nd
        vals = []
        for metric, _, is_float in OVERALL_METRICS:
            vals.append(fmt_val(d.get(metric, 0), is_float))
        val_cols.append(col_width([name] + vals))

    # delta columns (one per consecutive pair, or base vs each)
    base_data = data[0]

    print(bold("\n── Overall ──────────────────────────────────────────────────────"))
    # header
    header = f"  {'Metric':<{metric_w}}"
    for i, (name, w) in enumerate(zip(names, val_cols)):
        tag = f"[exp{exp_data[i]['num']:02d}] {name}"
        header += f"  {tag:<{w}}"
    if len(data) > 1:
        header += f"  {'Δ (vs base)'}"
    print(bold(header))
    print("  " + "─" * (metric_w + sum(w + 2 for w in val_cols) + 20))

    for metric, hib, is_float in OVERALL_METRICS:
        row = f"  {metric:<{metric_w}}"
        for i, (d, w) in enumerate(zip(data, val_cols)):
            v = fmt_val(d.get(metric, 0), is_float)
            row += f"  {v:<{w}}"
        if len(data) > 1:
            base_val = base_data.get(metric, 0)
            # show delta for each non-base experiment
            deltas = []
            for d in data[1:]:
                delta = d.get(metric, 0) - base_val
                deltas.append(fmt_delta(delta, is_float, hib))
            row += "  " + "  ".join(deltas)
        print(row)


def print_per_sequence(exp_data):
    """Print per-sequence comparison for each sequence present in any experiment."""
    # collect all sequence names
    all_seqs = []
    seen = set()
    for d in exp_data:
        for seq in d["data"]["per_sequence"]:
            if seq not in seen:
                all_seqs.append(seq)
                seen.add(seq)

    base_data = exp_data[0]["data"]
    names     = [d["name"] for d in exp_data]

    metric_w = max(len(m) for m, _, _ in SEQ_METRICS)

    for seq in all_seqs:
        print(bold(f"\n── {seq}"))
        header = f"  {'Metric':<{metric_w}}"
        val_ws = []
        for ed in exp_data:
            seq_d = ed["data"]["per_sequence"].get(seq, {})
            vals  = [fmt_val(seq_d.get(m, 0), fl) for m, _, fl in SEQ_METRICS]
            col_label = f"[exp{ed['num']:02d}] {ed['name']}"
            w = col_width([col_label] + vals)
            val_ws.append(w)
            header += f"  {col_label:<{w}}"
        if len(exp_data) > 1:
            header += "  Δ"
        print(bold(header))
        print("  " + "─" * (metric_w + sum(w + 2 for w in val_ws) + 15))

        for metric, hib, is_float in SEQ_METRICS:
            row = f"  {metric:<{metric_w}}"
            vals_by_exp = []
            for ed, w in zip(exp_data, val_ws):
                seq_d = ed["data"]["per_sequence"].get(seq, {})
                v = fmt_val(seq_d.get(metric, 0), is_float)
                vals_by_exp.append(seq_d.get(metric, 0))
                row += f"  {v:<{w}}"
            if len(exp_data) > 1:
                base_val = base_data["per_sequence"].get(seq, {}).get(metric, 0)
                deltas = []
                for val in vals_by_exp[1:]:
                    deltas.append(fmt_delta(val - base_val, is_float, hib))
                row += "  " + "  ".join(deltas)
            print(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    exps = discover_experiments()
    if not exps:
        print(f"No experiments found in {EXPERIMENTS_DIR}")
        sys.exit(1)

    print(bold("\nAvailable experiments:"))
    for num, d in exps:
        cfg = load_config(d)
        has_eval = bool(list((d / "eval").glob("tracking_metrics_*.json"))) if (d / "eval").exists() else False
        status = green("✓ eval") if has_eval else yellow("✗ no eval")
        print(f"  [{num:02d}] {d.name:<45}  {status}  {cfg}")

    print()
    raw = input("Enter experiment numbers to compare (e.g. 0 1  or  0): ").strip()
    if not raw:
        sys.exit(0)

    try:
        chosen = [int(x) for x in raw.split()]
    except ValueError:
        print("Invalid input — expected space-separated integers.")
        sys.exit(1)

    exp_map = {num: d for num, d in exps}
    exp_data = []
    for num in chosen:
        if num not in exp_map:
            print(f"Experiment {num} not found.")
            sys.exit(1)
        d = exp_map[num]
        metrics = load_experiment(d)
        if metrics is None:
            print(f"No eval JSON found for exp{num:02d} — run eval_tracking.py first.")
            sys.exit(1)
        name_m = re.match(r"exp\d+_(.*)", d.name)
        name = name_m.group(1) if name_m else d.name
        exp_data.append({"num": num, "name": name, "data": metrics})

    print_overall(exp_data)

    if len(exp_data) > 0:
        print()
        show_seq = input("Show per-sequence breakdown? [y/N] ").strip().lower()
        if show_seq == "y":
            print_per_sequence(exp_data)

    print()


if __name__ == "__main__":
    main()
