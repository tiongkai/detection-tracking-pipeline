# Issues — PR #4 "decouple detection model"

Should-fix items from code review (head `f9d4fbb`, base `a9b30c8`).

> **✅ All resolved** in the clean rebase merged to `main` as commit `2f09692`
> (branch `dev/decouple-detection-clean`). PR #4 is superseded — close it manually
> (closing requires the GitHub API / a token, not available on this machine).

## 1. Type-branching defeats the abstraction
`track/track_video.py` `process_clip` branches on the concrete detector type
("RF-DETR uses `threshold`, YOLO uses `conf`"), leaking the implementation that
`BaseDetector` is meant to hide. Normalize parameter names at the interface so
the caller makes a single uniform `detector.predict(frame, conf=conf, iou=iou)`
call with no `isinstance`/string sniffing.

## 2. RF-DETR class names hardcoded to COCO-80
`RFDetrDetector._load_class_names()` returns the COCO map, but this is a
maritime/boat domain with a likely custom-trained model. `class_id` will index
into the wrong labels in the annotated output (YOLO correctly uses
`model.names`). Load names from config instead of shipping a silent mislabel.

## 3. RF-DETR ignores `device`
`RFDetrDetector.__init__` stores `self.device` but `predict()` never uses it and
`to()` is a no-op, so `--device` has no effect for RF-DETR. Wire it into the
model or document that RF-DETR pins its own device.

## 4. `to()` is a no-op everywhere
`to()` is `pass`/attr-set in the base class and both subclasses. Either
implement it or remove it from the interface so it doesn't imply a capability
that doesn't exist.

## 5. Fragile weights-path guessing in `run()`
`f"weights/{model}_{size}.pt"` builds e.g. `weights/rfdetr_base.pt`, but RF-DETR
weights are conventionally `.pth`. Require an explicit `weights` and fail loudly
rather than guessing the path.
