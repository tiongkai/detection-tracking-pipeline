"""
Compute per-image quality and object-level metrics for a full split.
Run once per split — results are cached so eval.py can load them for
train vs test distribution comparison in failure analysis.

Usage:
    python pipeline/eval/compute_split_metrics.py --split split_v1_baseline

Writes to:
    data/splits/<split_name>/split_metrics.csv
    (columns: filename, split, source, video_id, <all image_metrics columns>, <object_metrics columns>)
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.eval.image_metrics import compute_image_metrics, compute_object_metrics
from pycocotools.coco import COCO


def compute_metrics_for_split(split_name: str) -> None:
    split_dir = PROJECT_ROOT / "data" / "splits" / split_name
    out_csv = split_dir / "split_metrics.csv"

    if not split_dir.exists():
        raise FileNotFoundError(f"Split not found: {split_dir}")

    manifest = pd.read_csv(split_dir / "split_manifest.csv")
    stem_to_source = dict(zip(manifest["filename"], manifest["source"]))
    stem_to_video = dict(zip(manifest["filename"], manifest["video_id"]))
    stem_to_split = dict(zip(manifest["filename"], manifest["split"]))

    rows = []

    for df_split, coco_split in [("train", "train"), ("val", "valid"), ("test", "test")]:
        coco_json = split_dir / "coco" / coco_split / "_annotations.coco.json"
        if not coco_json.exists():
            print(f"  [SKIP] No COCO JSON for {df_split}")
            continue

        coco = COCO(str(coco_json))
        gt_by_img = {}
        for ann in coco.dataset["annotations"]:
            gt_by_img.setdefault(ann["image_id"], []).append(ann)

        total = len(coco.dataset["images"])
        print(f"\n{df_split}: {total} images")

        for idx, img_info in enumerate(coco.dataset["images"], 1):
            stem = Path(img_info["file_name"]).stem
            img_bgr = cv2.imread(img_info["file_name"])
            if img_bgr is None:
                continue

            img_m = compute_image_metrics(img_bgr)
            gt_boxes = gt_by_img.get(img_info["id"], [])
            obj_m = compute_object_metrics(img_bgr, [ann["bbox"] for ann in gt_boxes])

            rows.append({
                "filename": stem,
                "split": df_split,
                "source": stem_to_source.get(stem, img_info.get("source", "unknown")),
                "video_id": stem_to_video.get(stem, ""),
                **img_m,
                **obj_m,
            })

            if idx % 500 == 0 or idx == total:
                print(f"  {idx}/{total} processed...")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} rows)")

    # Print summary
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        if len(sub):
            print(f"  {sp:5s}: brightness={sub['brightness'].mean():.3f}  "
                  f"saturation={sub['color_saturation'].mean():.3f}  "
                  f"sharpness={sub['sharpness'].mean():.1f}  "
                  f"n={len(sub)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, help="Split name, e.g. split_v1_baseline")
    args = parser.parse_args()
    compute_metrics_for_split(args.split)
