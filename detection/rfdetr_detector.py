"""RF-DETR detector implementation."""

import warnings
from typing import Dict, List, Any, Optional
import numpy as np
import cv2
from PIL import Image

from .base_detector import BaseDetector


# COCO 80-class fallback, used only when no domain class names are supplied.
_COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush',
}


class RFDetrDetector(BaseDetector):
    """RF-DETR detector wrapper."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda:0",
        model_size: str = "base",
        class_names: Optional[Dict[int, str]] = None,
    ):
        """Initialize RF-DETR detector.

        Args:
            weights_path: Path to RF-DETR weights file
            device: Torch device (cuda:0, cpu, etc.)
            model_size: 'base' or 'large'
            class_names: Optional {class_id: name} mapping for the trained
                domain. If omitted, falls back to COCO-80 with a warning — for a
                custom-trained model this will mislabel detections, so pass the
                real mapping (e.g. from the experiment config).
        """
        from rfdetr import RFDETRBase, RFDETRLarge

        self.device = device
        self.model_size = model_size

        if model_size.lower() == 'large':
            self.model = RFDETRLarge(pretrain_weights=weights_path)
        else:
            self.model = RFDETRBase(pretrain_weights=weights_path)

        self.to(device)

        if class_names:
            self._class_names = {int(k): v for k, v in class_names.items()}
        else:
            warnings.warn(
                "RFDetrDetector: no class_names provided; falling back to COCO-80. "
                "Detections from a custom-trained model will be mislabeled. "
                "Pass class_names from your config."
            )
            self._class_names = dict(_COCO_CLASSES)

    def predict(
        self, image: np.ndarray, conf: float = 0.3, iou: float = 0.5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Run RF-DETR inference.

        Args:
            image: BGR image as numpy array
            conf: Confidence threshold (mapped to RF-DETR's ``threshold``)
            iou: Unused by RF-DETR (no NMS step); accepted for interface parity

        Returns:
            List of detection dictionaries
        """
        # Convert BGR (OpenCV) to RGB (PIL expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        detections = self.model.predict(pil_image, threshold=conf)

        results = []
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                results.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(detections.confidence[i]),
                    "class_id": int(detections.class_id[i]),
                })

        return results

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def to(self, device: str):
        """Move model to device (best effort across RF-DETR API versions)."""
        self.device = device
        inner = getattr(self.model, "model", None)
        target = getattr(inner, "model", inner)  # RFDETRBase.model.model is the nn.Module
        if target is not None and hasattr(target, "to"):
            try:
                target.to(device)
            except Exception as e:  # pragma: no cover - depends on rfdetr version
                warnings.warn(f"RFDetrDetector: could not move model to {device}: {e}")
