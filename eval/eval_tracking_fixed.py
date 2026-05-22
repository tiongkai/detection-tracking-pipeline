"""Tracking evaluation using TrackEval library. (FIXED VERSION)"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TrackEval Integration
# ---------------------------------------------------------------------------

def ensure_trackeval():
    """Install TrackEval if not available."""
    try:
        import trackeval
        return True
    except ImportError:
        print("TrackEval not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/JonathonLuiten/TrackEval.git"])
            import trackeval
            print("TrackEval installed successfully.")
            return True
        except Exception as e:
            print(f"Failed to install TrackEval: {e}")
            print("Falling back to custom metrics implementation.")
            return False


def prepare_trackeval_format(gt_dir, tracker_dir, tracker_name, temp_dir):
    """Convert your data to TrackEval expected format."""
    temp_dir = Path(temp_dir)
    gt_base = temp_dir / "gt" / "mot_challenge"
    tracker_base = temp_dir / "trackers" / "mot_challenge" / tracker_name / "data"
    
    gt_dir = Path(gt_dir)
    seqs_found = []
    
    for gt_file in gt_dir.rglob("gt.txt"):
        parent = gt_file.parent
        if parent.name == "gt":
            seq_name = parent.parent.name
        else:
            seq_name = parent.name
        dest = gt_base / seq_name / "gt" / "gt.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt_file, dest)
        seqs_found.append(seq_name)
    
    tracker_dir = Path(tracker_dir)
    tracker_base.mkdir(parents=True, exist_ok=True)
    
    matched_seqs = []
    for tracker_file in tracker_dir.glob("*.txt"):
        seq_name = tracker_file.stem
        if seq_name in seqs_found:
            dest = tracker_base / f"{seq_name}.txt"
            shutil.copy2(tracker_file, dest)
            matched_seqs.append(seq_name)
    
    return matched_seqs


def run_trackeval(temp_dir, tracker_name, iou_threshold=0.5, classes_to_eval=None):
    """Run TrackEval and return results."""
    from trackeval import Evaluator, get_dataset, get_default_dataset_config
    
    dataset_config = get_default_dataset_config()
    dataset_config.GT_FOLDER = str(temp_dir / "gt")
    dataset_config.TRACKERS_FOLDER = str(temp_dir / "trackers")
    dataset_config.OUTPUT_FOLDER = str(temp_dir / "output")
    dataset_config.TRACKERS_TO_EVAL = [tracker_name]
    dataset_config.CLASSES_TO_EVAL = classes_to_eval or ["pedestrian"]
    dataset_config.SPLIT_TO_EVAL = "mot_challenge"
    dataset_config.USE_PARALLEL = False
    dataset_config.PRINT_CONFIG = False
    dataset_config.PRINT_ONLY_COMBINED = True
    
    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity", "VACE"],
        "THRESHOLD": iou_threshold,
        "PRINT_CONFIG": False,
    }
    
    dataset = get_dataset(dataset_config)
    evaluator = Evaluator(dataset, metrics_config)
    results = evaluator.evaluate()
    
    return results


def parse_trackeval_results(results, tracker_name):
    """Extract metrics from TrackEval results."""
    metrics = {
        "MOTA": 0.0, "MOTP": 0.0, "IDF1": 0.0, "HOTA": 0.0,
        "DetA": 0.0, "AssA": 0.0, "IDsw": 0, "Frag": 0,
        "MT": 0, "ML": 0, "Precision": 0.0, "Recall": 0.0,
        "TP": 0, "FP": 0, "FN": 0,
    }
    
    if tracker_name not in results:
        return metrics
    
    tracker_res = results[tracker_name]
    
    if "CLEAR" in tracker_res:
        clear = tracker_res["CLEAR"]
        metrics["MOTA"] = clear.get("MOTA", 0.0)
        metrics["MOTP"] = clear.get("MOTP", 0.0)
        metrics["IDsw"] = int(clear.get("IDSW", 0))
        metrics["TP"] = int(clear.get("CLR_TP", 0))
        metrics["FP"] = int(clear.get("CLR_FP", 0))
        metrics["FN"] = int(clear.get("CLR_FN", 0))
        if metrics["TP"] + metrics["FP"] > 0:
            metrics["Precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
        if metrics["TP"] + metrics["FN"] > 0:
            metrics["Recall"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
    
    if "Identity" in tracker_res:
        metrics["IDF1"] = tracker_res["Identity"].get("IDF1", 0.0)
    
    if "HOTA" in tracker_res:
        hota = tracker_res["HOTA"]
        metrics["HOTA"] = hota.get("HOTA", 0.0)
        metrics["DetA"] = hota.get("DetA", 0.0)
        metrics["AssA"] = hota.get("AssA", 0.0)
    
    if "VACE" in tracker_res:
        vace = tracker_res["VACE"]
        metrics["Frag"] = int(vace.get("Frag", 0))
        metrics["MT"] = int(vace.get("MT", 0))
        metrics["ML"] = int(vace.get("ML", 0))
    
    return metrics


# ---------------------------------------------------------------------------
# Custom Metrics Implementation (Fallback)
# ---------------------------------------------------------------------------

def load_mot(path):
    """Load MOT-format file."""
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            frame = int(p[0])
            tid = int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            conf = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else 1
            vis = float(p[8]) if len(p) > 8 else 1.0
            data[frame].append((tid, x, y, w, h, conf, cls, vis))
    return dict(data)


def discover_sequences(gt_dir, tracker_dir):
    """Match GT sequences to tracker output files by clip name.
    
    Handles both:
        - seq_name/gt/gt.txt (standard MOT)
        - seq_name/gt.txt (flat structure)
    """
    gt_dir, tracker_dir = Path(gt_dir), Path(tracker_dir)
    
    gt_seqs = {}
    for gt_file in sorted(gt_dir.rglob("gt.txt")):
        parent = gt_file.parent
        # If parent is named 'gt', go up one more level
        if parent.name == "gt":
            seq_name = parent.parent.name
        else:
            seq_name = parent.name
        gt_seqs[seq_name] = gt_file

    matched = []
    for name, gt_file in gt_seqs.items():
        tf = tracker_dir / f"{name}.txt"
        if tf.exists():
            matched.append((name, gt_file, tf))
            print(f"  ✓ {name}: matched")
        else:
            print(f"  ✗ {name}: no tracker output (expected {tf})")

    if not matched:
        raise FileNotFoundError(
            f"No matching sequences.\n  GT has: {list(gt_seqs.keys())}\n  Tracker dir: {tracker_dir}\n"
            f"Tracker files: {list(tracker_dir.glob('*.txt'))}"
        )
    return matched


# Class compatibility (domain-split taxonomy)
CLASS_NAMES = {
    0: "boat-rgb", 1: "vessel-rgb", 2: "human-rgb",
    3: "outboard motor-rgb", 4: "head-rgb", 5: "torso-rgb",
    6: "boat-thermal", 7: "vessel-thermal", 8: "human-thermal",
    9: "outboard motor-thermal", 10: "head-thermal", 11: "torso-thermal",
}


def _base_class(cls_id):
    name = CLASS_NAMES.get(cls_id, str(cls_id))
    return name.replace("-rgb", "").replace("-thermal", "")


def classes_compatible(cls_a, cls_b):
    return _base_class(cls_a) == _base_class(cls_b)


def iou_matrix(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    a_x2 = a[:, 0] + a[:, 2]
    a_y2 = a[:, 1] + a[:, 3]
    b_x2 = b[:, 0] + b[:, 2]
    b_y2 = b[:, 1] + b[:, 3]

    ix1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    iy1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    ix2 = np.minimum(a_x2[:, None], b_x2[None, :])
    iy2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_frames(gt_data, pred_data, iou_thresh):
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    frame_results = []

    for frame in all_frames:
        gt_entries = [e for e in gt_data.get(frame, []) if e[5] > 0]
        pred_entries = list(pred_data.get(frame, []))

        gt_ids = [e[0] for e in gt_entries]
        pred_ids = [e[0] for e in pred_entries]
        gt_boxes = [(e[1], e[2], e[3], e[4]) for e in gt_entries]
        pred_boxes = [(e[1], e[2], e[3], e[4]) for e in pred_entries]
        gt_classes = [e[6] for e in gt_entries]
        pred_classes = [e[6] for e in pred_entries]

        matches = []
        if gt_boxes and pred_boxes:
            iou = iou_matrix(gt_boxes, pred_boxes)
            cost = 1 - iou
            cost[iou < iou_thresh] = 1e6
            for r in range(len(gt_classes)):
                for c in range(len(pred_classes)):
                    if not classes_compatible(gt_classes[r], pred_classes[c]):
                        cost[r, c] = 1e6
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if iou[r, c] >= iou_thresh:
                    matches.append((gt_ids[r], pred_ids[c], float(iou[r, c])))

        matched_gt = {m[0] for m in matches}
        matched_pred = {m[1] for m in matches}
        fp_ids = [pid for pid in pred_ids if pid not in matched_pred]
        fn_ids = [gid for gid in gt_ids if gid not in matched_gt]

        frame_results.append({
            "frame": frame,
            "matches": matches,
            "fp_ids": fp_ids,
            "fn_ids": fn_ids,
            "n_gt": len(gt_ids),
            "n_pred": len(pred_ids),
        })

    return frame_results


def compute_clear(frame_results):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    sum_iou = 0.0
    prev_match = {}

    for fr in frame_results:
        matches = fr["matches"]
        tp = len(matches)
        fp = len(fr["fp_ids"])
        fn = len(fr["fn_ids"])

        idsw = 0
        cur_match = {}
        for gt_id, pred_id, iou_val in matches:
            cur_match[gt_id] = pred_id
            if gt_id in prev_match and prev_match[gt_id] != pred_id:
                idsw += 1
            sum_iou += iou_val

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_idsw += idsw
        prev_match = cur_match

    n_gt_total = total_tp + total_fn
    mota = 1 - (total_fn + total_fp + total_idsw) / n_gt_total if n_gt_total > 0 else 0
    motp = sum_iou / total_tp if total_tp > 0 else 0

    return {
        "MOTA": float(mota),
        "MOTP": float(motp),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "IDsw": int(total_idsw),
    }


def compute_track_quality(gt_data, frame_results):
    gt_frames = defaultdict(set)
    for frame, entries in gt_data.items():
        for e in entries:
            if e[5] > 0:
                gt_frames[e[0]].add(frame)

    matched_at_frame = defaultdict(set)
    for fr in frame_results:
        for gt_id, _, _ in fr["matches"]:
            matched_at_frame[gt_id].add(fr["frame"])

    n_gt_tracks = len(gt_frames)
    mt = 0
    pt = 0
    ml = 0
    total_frag = 0

    for gt_id, frames in gt_frames.items():
        n_present = len(frames)
        n_matched = len(matched_at_frame.get(gt_id, set()))
        ratio = n_matched / n_present if n_present > 0 else 0

        if ratio > 0.8:
            mt += 1
        elif ratio < 0.2:
            ml += 1
        else:
            pt += 1

        sorted_frames = sorted(frames)
        was_tracked = False
        for f in sorted_frames:
            is_tracked = f in matched_at_frame.get(gt_id, set())
            if was_tracked and not is_tracked:
                total_frag += 1
            was_tracked = is_tracked

    return {
        "Frag": int(total_frag),
        "MT": int(mt),
        "PT": int(pt),
        "ML": int(ml),
        "GT_tracks": int(n_gt_tracks),
    }


def compute_identity(gt_data, pred_data, frame_results):
    pair_tp = defaultdict(int)
    gt_total = defaultdict(int)
    pred_total = defaultdict(int)

    for entries in gt_data.values():
        for e in entries:
            if e[5] > 0:
                gt_total[e[0]] += 1
    for entries in pred_data.values():
        for e in entries:
            pred_total[e[0]] += 1

    for fr in frame_results:
        for gt_id, pred_id, _ in fr["matches"]:
            pair_tp[(gt_id, pred_id)] += 1

    gt_ids = sorted(gt_total.keys())
    pred_ids = sorted(pred_total.keys())

    if not gt_ids or not pred_ids:
        return {"IDF1": 0.0, "IDTP": 0, "IDFP": 0, "IDFN": 0}

    cost = np.zeros((len(gt_ids), len(pred_ids)))
    gi_map = {gid: i for i, gid in enumerate(gt_ids)}
    pi_map = {pid: i for i, pid in enumerate(pred_ids)}
    for (gid, pid), count in pair_tp.items():
        if gid in gi_map and pid in pi_map:
            cost[gi_map[gid], pi_map[pid]] = -count

    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(cost)

    idtp = 0
    for r, c in zip(ri, ci):
        tp_count = pair_tp.get((gt_ids[r], pred_ids[c]), 0)
        idtp += tp_count

    sum_gt = sum(gt_total.values())
    sum_pred = sum(pred_total.values())
    idfn = sum_gt - idtp
    idfp = sum_pred - idtp
    idf1 = 2 * idtp / (sum_gt + sum_pred) if (sum_gt + sum_pred) > 0 else 0

    return {
        "IDF1": float(idf1),
        "IDTP": int(idtp),
        "IDFP": int(idfp),
        "IDFN": int(idfn),
    }


def compute_hota(gt_data, pred_data, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    hota_vals, deta_vals, assa_vals = [], [], []

    for alpha in thresholds:
        frame_results = match_frames(gt_data, pred_data, alpha)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        gt_track_len = defaultdict(int)
        pred_track_len = defaultdict(int)
        pair_tpa = defaultdict(int)

        for fr in frame_results:
            tp = len(fr["matches"])
            total_tp += tp
            total_fp += len(fr["fp_ids"])
            total_fn += len(fr["fn_ids"])
            for gt_id, pred_id, _ in fr["matches"]:
                gt_track_len[gt_id] += 1
                pred_track_len[pred_id] += 1
                pair_tpa[(gt_id, pred_id)] += 1

        denom = total_tp + total_fp + total_fn
        deta = total_tp / denom if denom > 0 else 0
        deta_vals.append(deta)

        if total_tp == 0:
            assa_vals.append(0)
            hota_vals.append(0)
            continue

        ass_sum = 0.0
        for (gid, pid), tpa in pair_tpa.items():
            fpa = pred_track_len[pid] - tpa
            fna = gt_track_len[gid] - tpa
            d = tpa + fpa + fna
            ass_sum += tpa * (tpa / d) if d > 0 else 0

        assa = ass_sum / total_tp
        assa_vals.append(assa)
        hota_vals.append(np.sqrt(deta * assa))

    return {
        "HOTA": float(np.mean(hota_vals)),
        "DetA": float(np.mean(deta_vals)),
        "AssA": float(np.mean(assa_vals)),
    }


def evaluate_sequence_custom(gt_data, pred_data, iou_threshold=0.5):
    frame_results = match_frames(gt_data, pred_data, iou_threshold)
    clear = compute_clear(frame_results)
    quality = compute_track_quality(gt_data, frame_results)
    identity = compute_identity(gt_data, pred_data, frame_results)
    hota = compute_hota(gt_data, pred_data)

    precision = clear["TP"] / (clear["TP"] + clear["FP"]) if (clear["TP"] + clear["FP"]) > 0 else 0
    recall = clear["TP"] / (clear["TP"] + clear["FN"]) if (clear["TP"] + clear["FN"]) > 0 else 0

    return {**clear, **quality, **identity, **hota, "Precision": precision, "Recall": recall}


def evaluate_tracker_custom(gt_dir, tracker_dir, iou_threshold=0.5):
    sequences = discover_sequences(gt_dir, tracker_dir)

    per_seq = {}
    all_gt, all_pred = {}, {}
    frame_offset = 0

    for seq_name, gt_file, tracker_file in sequences:
        gt_data = load_mot(gt_file)
        pred_data = load_mot(tracker_file)
        print(f"  {seq_name}: {len(gt_data)} GT frames, {len(pred_data)} tracker frames")
        per_seq[seq_name] = evaluate_sequence_custom(gt_data, pred_data, iou_threshold)

        max_frame = max(
            max(gt_data.keys(), default=0),
            max(pred_data.keys(), default=0),
        )
        for f, entries in gt_data.items():
            all_gt[f + frame_offset] = entries
        for f, entries in pred_data.items():
            all_pred[f + frame_offset] = entries
        frame_offset += max_frame + 1

    overall = evaluate_sequence_custom(all_gt, all_pred, iou_threshold)
    for key in ("MT", "ML", "PT", "GT_tracks"):
        overall[key] = sum(r[key] for r in per_seq.values())
    
    return {"per_sequence": per_seq, "overall": overall}


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_tracker(gt_dir, tracker_dir, iou_threshold=0.5, use_trackeval=True):
    tracker_name = Path(tracker_dir).name
    
    if use_trackeval and ensure_trackeval():
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                seqs = prepare_trackeval_format(gt_dir, tracker_dir, tracker_name, temp_dir)
                
                if not seqs:
                    print(f"  WARNING: No matching sequences for {tracker_name}")
                    return None
                
                print(f"  Found {len(seqs)} sequences")
                results = run_trackeval(temp_dir, tracker_name, iou_threshold)
                metrics = parse_trackeval_results(results, tracker_name)
                return {"overall": metrics, "per_sequence": {}}
        except Exception as e:
            print(f"  TrackEval failed: {e}")
            print("  Falling back to custom implementation...")
    
    print("  Using custom metrics implementation...")
    return evaluate_tracker_custom(gt_dir, tracker_dir, iou_threshold)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SUMMARY_COLS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDsw", "Frag", "MT", "ML", "Precision", "Recall"]


def _fmt(val):
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def format_table(results_list, names):
    header = "| Config | " + " | ".join(SUMMARY_COLS) + " |"
    sep = "|" + "|".join(["--------"] * (len(SUMMARY_COLS) + 1)) + "|"
    rows = [header, sep]
    for name, res in zip(names, results_list):
        if res is None:
            rows.append(f"| {name} | ERROR |")
            continue
        o = res["overall"]
        cells = " | ".join(_fmt(o.get(c, 0)) for c in SUMMARY_COLS)
        rows.append(f"| {name} | {cells} |")
    return "\n".join(rows)


def format_per_sequence(results, name):
    if not results or not results.get("per_sequence"):
        return ""
    
    cols = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDsw", "Frag", "MT", "ML", "GT_tracks"]
    header = "| Sequence | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["--------"] * (len(cols) + 1)) + "|"
    rows = [f"\n### {name} — Per Sequence\n", header, sep]
    
    for seq_name, m in results["per_sequence"].items():
        cells = " | ".join(_fmt(m.get(c, 0)) for c in cols)
        rows.append(f"| {seq_name} | {cells} |")
    
    return "\n".join(rows)


def write_report(results_list, names, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Tracking Evaluation Report\n", "## Overall\n", format_table(results_list, names), ""]
    
    for name, res in zip(names, results_list):
        lines.append(format_per_sequence(res, name))
        lines.append("")
    
    for name, res in zip(names, results_list):
        if res:
            (out_dir / f"tracking_metrics_{name}.json").write_text(json.dumps(res, indent=2))
    
    df_data = {}
    for name, res in zip(names, results_list):
        if res:
            df_data[name] = res["overall"]
    if df_data:
        pd.DataFrame(df_data).T.to_csv(out_dir / "metrics_summary.csv")
    
    (out_dir / "tracking_report.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {out_dir / 'tracking_report.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker output using TrackEval (or custom metrics).")
    parser.add_argument("--config", default=None,
                        help="Path to experiment config YAML — sets --gt, --tracker, -o, and iou-threshold")
    parser.add_argument("--gt", default=None, help="GT directory (<clip>/gt/gt.txt)")
    parser.add_argument("--tracker", default=None, nargs="+",
                        help="Tracker output directory (<clip>.txt). Multiple for comparison.")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each tracker (default: directory names)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("-o", "--out", default=None, help="Output directory for report files")
    parser.add_argument("--no-trackeval", action="store_true",
                        help="Force use of custom metrics instead of TrackEval")
    args = parser.parse_args()

    gt = args.gt
    tracker_dirs = args.tracker
    out = args.out
    iou_threshold = args.iou_threshold

    if args.config:
        import yaml
        cfg_path = Path(args.config)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        exp_dir = cfg_path.parent
        print(f"Config: {cfg_path}")

        if gt is None:
            gt = cfg.get("gt")
        if tracker_dirs is None:
            tracker_dirs = [str(exp_dir / "mot")]
        if out is None:
            out = str(exp_dir / "eval")
        if iou_threshold is None:
            iou_threshold = cfg.get("eval", {}).get("iou_threshold", 0.5)

    if not gt:
        parser.error("--gt is required (or set via config YAML)")
    if not tracker_dirs:
        parser.error("--tracker is required (or set via config YAML)")

    names = args.names or [Path(t).name for t in tracker_dirs]
    if len(names) != len(tracker_dirs):
        parser.error("--names count must match --tracker count")

    use_trackeval = not args.no_trackeval

    results_list = []
    for name, tdir in zip(names, tracker_dirs):
        print(f"\nEvaluating: {name} ({tdir})")
        result = evaluate_tracker(gt, tdir, iou_threshold, use_trackeval)
        results_list.append(result)

    print("\n" + format_table(results_list, names))
    
    for name, res in zip(names, results_list):
        if res and res.get("per_sequence"):
            print(format_per_sequence(res, name))

    if out:
        write_report(results_list, names, out)


if __name__ == "__main__":
    main()
