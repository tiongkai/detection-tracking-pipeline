"""YOLO detector implementation using Ultralytics."""

from typing import Dict, List, Any
import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """YOLO detector wrapper."""

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        """Initialize YOLO detector.

        Args:
            weights_path: Path to YOLO .pt weights file
            device: Torch device (cuda:0, cpu, etc.)
        """
        self.device = device
        self.model = YOLO(weights_path)
        self.model.to(device)
        self._class_names = self.model.names

    def predict(
        self, image: np.ndarray, conf: float = 0.3, iou: float = 0.5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Run YOLO inference.

        Args:
            image: BGR image as numpy array
            conf: Confidence threshold
            iou: NMS IoU threshold
            **kwargs: Optional ``verbose`` (default: False)

        Returns:
            List of detection dictionaries
        """
        verbose = kwargs.get("verbose", False)

        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=verbose,
            device=self.device,
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()

                for box, conf_val, cls_id in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = box
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf_val),
                        "class_id": int(cls_id),
                    })

        return detections

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        self.model.to(device)
