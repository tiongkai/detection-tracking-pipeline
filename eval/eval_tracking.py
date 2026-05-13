"""Tracking evaluation. Computes MOT metrics against ground truth.

Both GT and tracker output use MOTChallenge format (comma-separated):
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,<visibility>

    GT:      conf=1 means consider, conf=0 means ignore.
    Tracker: conf=detection confidence. visibility=1.0 for detected, <1.0 for Kalman-predicted.

Directory layout:

    gt_dir/
        <clip_name>/
            gt.txt

    tracker_dir/
        <clip_name>.txt

Usage:
    conda run -n boat-tracker python eval/eval_tracking.py \\
        --gt data/eval/gt \\
        --tracker results/tracker/mot/baseline

    # Compare two configs side-by-side:
    conda run -n boat-tracker python eval/eval_tracking.py \\
        --gt data/eval/gt \\
        --tracker results/tracker/mot/baseline results/tracker/mot/tuned \\
        --names baseline tuned \\
        -o results/tracker/eval
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_mot(path):
    """Load MOT-format file. Returns {frame: [(id, x, y, w, h, conf, cls, vis), ...]}."""
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
    """Match GT sequences to tracker output files by clip name."""
    gt_dir, tracker_dir = Path(gt_dir), Path(tracker_dir)
    gt_seqs = {f.parent.name: f for f in sorted(gt_dir.rglob("gt.txt"))}

    matched = []
    for name, gt_file in gt_seqs.items():
        tf = tracker_dir / f"{name}.txt"
        if tf.exists():
            matched.append((name, gt_file, tf))
        else:
            print(f"  WARNING: no tracker output for '{name}', skipping")

    if not matched:
        raise FileNotFoundError(
            f"No matching sequences.\n  GT has: {list(gt_seqs.keys())}\n  Tracker dir: {tracker_dir}"
        )
    return matched


# ---------------------------------------------------------------------------
# Class compatibility (domain-split taxonomy)
# ---------------------------------------------------------------------------

# 12-class domain-split: same object type across RGB/thermal should match.
CLASS_NAMES = {
    0: "boat-rgb", 1: "vessel-rgb", 2: "human-rgb",
    3: "outboard motor-rgb", 4: "head-rgb", 5: "torso-rgb",
    6: "boat-thermal", 7: "vessel-thermal", 8: "human-thermal",
    9: "outboard motor-thermal", 10: "head-thermal", 11: "torso-thermal",
}


def _base_class(cls_id):
    """Strip domain suffix to get the base object type."""
    name = CLASS_NAMES.get(cls_id, str(cls_id))
    return name.replace("-rgb", "").replace("-thermal", "")


def classes_compatible(cls_a, cls_b):
    """True if two class IDs refer to the same object type (possibly different domains)."""
    return _base_class(cls_a) == _base_class(cls_b)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_matrix(boxes_a, boxes_b):
    """IoU between two sets of (x, y, w, h) boxes. Returns (N, M) array."""
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


# ---------------------------------------------------------------------------
# Frame-level matching (shared by all metrics)
# ---------------------------------------------------------------------------

def match_frames(gt_data, pred_data, iou_thresh):
    """Match GT to predictions per frame via Hungarian assignment at the given IoU threshold.

    Returns per-frame match info used by CLEAR, Identity, and HOTA metrics.
    """
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

        matches = []  # (gt_id, pred_id, iou)
        if gt_boxes and pred_boxes:
            iou = iou_matrix(gt_boxes, pred_boxes)
            cost = 1 - iou
            cost[iou < iou_thresh] = 1e6
            for r in range(len(gt_classes)):
                for c in range(len(pred_classes)):
                    if not classes_compatible(gt_classes[r], pred_classes[c]):
                        cost[r, c] = 1e6
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


# ---------------------------------------------------------------------------
# CLEAR metrics (MOTA, MOTP)
# ---------------------------------------------------------------------------

def compute_clear(frame_results):
    """CLEAR MOT metrics: MOTA, MOTP, FP, FN, ID switches."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    sum_iou = 0.0

    prev_match = {}  # gt_id -> pred_id from previous frame

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


# ---------------------------------------------------------------------------
# Fragmentation + MT/ML
# ---------------------------------------------------------------------------

def compute_track_quality(gt_data, frame_results):
    """Fragmentation count, Mostly Tracked / Partially Tracked / Mostly Lost."""
    # Build per-GT-id timeline: was it matched in each frame it appears?
    gt_frames = defaultdict(list)  # gt_id -> sorted list of frames it appears in
    for entries in gt_data.values():
        for e in entries:
            if e[5] > 0:
                gt_frames[e[0]].append(None)
    # Re-derive from gt_data with actual frame numbers
    gt_frames = defaultdict(set)
    for frame, entries in gt_data.items():
        for e in entries:
            if e[5] > 0:
                gt_frames[e[0]].add(frame)

    matched_at_frame = defaultdict(set)  # gt_id -> set of frames where matched
    for fr in frame_results:
        for gt_id, _, _ in fr["matches"]:
            matched_at_frame[gt_id].add(fr["frame"])

    n_gt_tracks = len(gt_frames)
    mt = 0  # >80% tracked
    pt = 0  # 20-80% tracked
    ml = 0  # <20% tracked
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

        # Fragmentation: count gaps in matched frames for this GT track
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


# ---------------------------------------------------------------------------
# Identity metrics (IDF1)
# ---------------------------------------------------------------------------

def compute_identity(gt_data, pred_data, frame_results):
    """IDF1 — measures how consistently the correct ID is maintained."""
    # For each (gt_id, pred_id) pair that ever matches, count co-occurrences.
    pair_tp = defaultdict(int)  # (gt_id, pred_id) -> frames matched together
    gt_total = defaultdict(int)  # gt_id -> total frames present
    pred_total = defaultdict(int)  # pred_id -> total frames present

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

    # Greedy 1-to-1 assignment of gt_id <-> pred_id to maximize IDTP
    # Build unique gt and pred IDs
    gt_ids = sorted(gt_total.keys())
    pred_ids = sorted(pred_total.keys())

    if not gt_ids or not pred_ids:
        return {"IDF1": 0.0, "IDTP": 0, "IDFP": 0, "IDFN": 0}

    # Cost matrix for Hungarian: maximize pair_tp → minimize -pair_tp
    cost = np.zeros((len(gt_ids), len(pred_ids)))
    gi_map = {gid: i for i, gid in enumerate(gt_ids)}
    pi_map = {pid: i for i, pid in enumerate(pred_ids)}
    for (gid, pid), count in pair_tp.items():
        if gid in gi_map and pid in pi_map:
            cost[gi_map[gid], pi_map[pid]] = -count

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


# ---------------------------------------------------------------------------
# HOTA
# ---------------------------------------------------------------------------

def compute_hota(gt_data, pred_data, thresholds=None):
    """HOTA — Higher Order Tracking Accuracy (Luiten et al., 2021)."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    hota_vals, deta_vals, assa_vals = [], [], []

    for alpha in thresholds:
        frame_results = match_frames(gt_data, pred_data, alpha)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        gt_matches = defaultdict(dict)    # gt_id  -> {frame: pred_id}
        pred_matches = defaultdict(dict)  # pred_id -> {frame: gt_id}
        all_tp_pairs = []

        for fr in frame_results:
            tp = len(fr["matches"])
            total_tp += tp
            total_fp += len(fr["fp_ids"])
            total_fn += len(fr["fn_ids"])
            for gt_id, pred_id, _ in fr["matches"]:
                gt_matches[gt_id][fr["frame"]] = pred_id
                pred_matches[pred_id][fr["frame"]] = gt_id
                all_tp_pairs.append((gt_id, pred_id))

        denom = total_tp + total_fp + total_fn
        deta = total_tp / denom if denom > 0 else 0
        deta_vals.append(deta)

        if total_tp == 0:
            assa_vals.append(0)
            hota_vals.append(0)
            continue

        ass_sum = 0.0
        for gid, pid in all_tp_pairs:
            tpa = sum(1 for p in gt_matches[gid].values() if p == pid)
            fpa = sum(1 for g in pred_matches[pid].values() if g != gid)
            fna = sum(1 for p in gt_matches[gid].values() if p != pid)
            d = tpa + fpa + fna
            ass_sum += tpa / d if d > 0 else 0

        assa = ass_sum / total_tp
        assa_vals.append(assa)
        hota_vals.append(np.sqrt(deta * assa))

    return {
        "HOTA": float(np.mean(hota_vals)),
        "DetA": float(np.mean(deta_vals)),
        "AssA": float(np.mean(assa_vals)),
    }


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(gt_data, pred_data, iou_threshold=0.5):
    """Compute all metrics for a single sequence."""
    frame_results = match_frames(gt_data, pred_data, iou_threshold)
    clear = compute_clear(frame_results)
    quality = compute_track_quality(gt_data, frame_results)
    identity = compute_identity(gt_data, pred_data, frame_results)
    hota = compute_hota(gt_data, pred_data)

    precision = clear["TP"] / (clear["TP"] + clear["FP"]) if (clear["TP"] + clear["FP"]) > 0 else 0
    recall = clear["TP"] / (clear["TP"] + clear["FN"]) if (clear["TP"] + clear["FN"]) > 0 else 0

    return {**clear, **quality, **identity, **hota, "Precision": precision, "Recall": recall}


def evaluate_tracker(gt_dir, tracker_dir, iou_threshold=0.5):
    """Evaluate a tracker across all sequences. Returns per-sequence and overall metrics."""
    sequences = discover_sequences(gt_dir, tracker_dir)

    per_seq = {}
    all_gt, all_pred = {}, {}
    frame_offset = 0

    for seq_name, gt_file, tracker_file in sequences:
        gt_data = load_mot(gt_file)
        pred_data = load_mot(tracker_file)
        print(f"  {seq_name}: {len(gt_data)} GT frames, {len(pred_data)} tracker frames")
        per_seq[seq_name] = evaluate_sequence(gt_data, pred_data, iou_threshold)

        # Merge for overall (offset frames to avoid collisions between sequences)
        max_frame = max(
            max(gt_data.keys(), default=0),
            max(pred_data.keys(), default=0),
        )
        for f, entries in gt_data.items():
            all_gt[f + frame_offset] = entries
        for f, entries in pred_data.items():
            all_pred[f + frame_offset] = entries
        frame_offset += max_frame + 1

    overall = evaluate_sequence(all_gt, all_pred, iou_threshold)
    return {"per_sequence": per_seq, "overall": overall}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SUMMARY_COLS = ["MOTA", "IDF1", "HOTA", "IDsw", "Frag", "MT", "ML", "Precision", "Recall"]


def _fmt(val):
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def format_table(results_list, names):
    """Markdown comparison table across tracker configs."""
    header = "| Config | " + " | ".join(SUMMARY_COLS) + " |"
    sep = "|" + "|".join(["--------"] * (len(SUMMARY_COLS) + 1)) + "|"
    rows = [header, sep]
    for name, res in zip(names, results_list):
        o = res["overall"]
        cells = " | ".join(_fmt(o[c]) for c in SUMMARY_COLS)
        rows.append(f"| {name} | {cells} |")
    return "\n".join(rows)


def format_per_sequence(results, name):
    """Per-sequence breakdown table."""
    cols = ["MOTA", "IDF1", "HOTA", "IDsw", "Frag"]
    header = "| Sequence | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["--------"] * (len(cols) + 1)) + "|"
    rows = [f"\n### {name} — Per Sequence\n", header, sep]
    for seq_name, m in results["per_sequence"].items():
        cells = " | ".join(_fmt(m[c]) for c in cols)
        rows.append(f"| {seq_name} | {cells} |")
    return "\n".join(rows)


def write_report(results_list, names, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Tracking Evaluation Report\n", "## Overall\n", format_table(results_list, names), ""]
    for name, res in zip(names, results_list):
        lines.append(format_per_sequence(res, name))
        lines.append("")

    (out_dir / "tracking_report.md").write_text("\n".join(lines) + "\n")
    for name, res in zip(names, results_list):
        (out_dir / f"tracking_metrics_{name}.json").write_text(json.dumps(res, indent=2))

    print(f"Report:  {out_dir / 'tracking_report.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker output against ground truth.")
    parser.add_argument("--gt", required=True, help="GT directory (<clip>/gt.txt)")
    parser.add_argument("--tracker", required=True, nargs="+",
                        help="Tracker output directory (<clip>.txt). Multiple for comparison.")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each tracker (default: directory names)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("-o", "--out", default=None, help="Output directory for report files")
    args = parser.parse_args()

    names = args.names or [Path(t).name for t in args.tracker]
    if len(names) != len(args.tracker):
        parser.error("--names count must match --tracker count")

    results_list = []
    for name, tdir in zip(names, args.tracker):
        print(f"\nEvaluating: {name} ({tdir})")
        results_list.append(evaluate_tracker(args.gt, tdir, args.iou_threshold))

    print("\n" + format_table(results_list, names))
    for name, res in zip(names, results_list):
        print(format_per_sequence(res, name))

    if args.out:
        write_report(results_list, names, args.out)


if __name__ == "__main__":
    main()
