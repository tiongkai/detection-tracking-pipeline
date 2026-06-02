"""Extract video dimensions for all clips in a directory and save to CSV."""
import argparse
import csv
import glob
import os
from pathlib import Path

import cv2


def get_video_info(path: str) -> dict | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = {
        "clip": Path(path).stem,
        "path": path,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_s"] = round(info["n_frames"] / info["fps"], 2) if info["fps"] > 0 else 0
    info["resolution"] = f"{info['width']}x{info['height']}"
    cap.release()
    return info


def main():
    parser = argparse.ArgumentParser(description="Extract video dimensions for all clips in a directory")
    parser.add_argument("--source", required=True, help="Directory containing video clips")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <source>/video_dimensions.csv)")
    args = parser.parse_args()

    exts = {".mp4", ".mkv", ".avi", ".mov", ".ts"}
    clips = sorted(
        p for p in glob.glob(os.path.join(args.source, "**", "*"), recursive=True)
        if Path(p).suffix.lower() in exts
    )

    if not clips:
        print(f"No video files found in {args.source}")
        return

    rows = []
    for clip in clips:
        info = get_video_info(clip)
        if info:
            rows.append(info)
            print(f"  {info['resolution']:>12}  {info['fps']:>6.2f} fps  {info['n_frames']:>6} frames  {Path(clip).name}")
        else:
            print(f"  ERROR: cannot open {clip}")

    out_path = args.out or os.path.join(args.source, "video_dimensions.csv")
    fieldnames = ["clip", "resolution", "width", "height", "fps", "n_frames", "duration_s", "path"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    resolutions = {}
    for r in rows:
        resolutions.setdefault(r["resolution"], 0)
        resolutions[r["resolution"]] += 1

    print(f"\n{len(rows)} clips saved to {out_path}")
    print("Resolution breakdown:")
    for res, count in sorted(resolutions.items(), key=lambda x: -x[1]):
        print(f"  {res}: {count} clip(s)")


if __name__ == "__main__":
    main()
