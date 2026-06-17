"""Regenerate N sample augmented clips per corruption, for visual inspection.

The robustness sweep deletes full augmented videos after inference (--cleanup-aug)
to bound disk. This re-creates a small, retained sample set (default 10 clips per
corruption) into <out>/<cond>/{videos,labels}. Deterministic (same seed as the
sweep), so samples match what was evaluated.

    .venv/bin/python eval/make_samples.py \
        --videos eval_videos/wavy-boats/videos --labels eval_videos/wavy-boats/labels \
        --out eval_videos/wavy-boats/aug_samples --n 10
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from degrade import degrade_video                       # noqa: E402
from robustness_sweep import video_stems, BINARY_KINDS, _gen_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="eval_videos/wavy-boats/videos")
    ap.add_argument("--labels", default="eval_videos/wavy-boats/labels")
    ap.add_argument("--out", default="eval_videos/wavy-boats/aug_samples")
    ap.add_argument("--kinds", nargs="+",
                    default=["lowlight", "jpeg", "shake", "grayscale", "invert", "grayscale_invert"])
    ap.add_argument("--sevs", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--n", type=int, default=10, help="clips per corruption")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    videos, labels, out = ROOT / args.videos, ROOT / args.labels, ROOT / args.out
    stems = video_stems(videos)[: args.n]

    conditions = []
    for k in args.kinds:
        conditions += [(k, 1)] if k in BINARY_KINDS else [(k, s) for s in args.sevs]

    tasks = []
    for kind, sev in conditions:
        cond = f"{kind}_s{sev}"
        for stem in stems:
            vpath = out / cond / "videos" / f"{stem}.mp4"
            if vpath.exists():
                continue
            src = next(p for p in Path(videos).iterdir() if p.stem == stem)
            gt_in = Path(labels) / f"{stem}.txt"
            gt_in = str(gt_in) if gt_in.exists() else None
            gt_out = str(out / cond / "labels" / f"{stem}.txt") if gt_in else None
            (out / cond / "videos").mkdir(parents=True, exist_ok=True)
            (out / cond / "labels").mkdir(parents=True, exist_ok=True)
            tasks.append((str(src), str(vpath), kind, sev, args.seed, gt_in, gt_out))

    print(f"{len(conditions)} corruptions x {len(stems)} clips = {len(tasks)} to generate")
    if tasks:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        done = 0
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            for f in as_completed([ex.submit(_gen_one, t) for t in tasks]):
                stem, n = f.result()
                done += 1
                print(f"  [{done}/{len(tasks)}] {stem[:48]}: {n} frames", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
