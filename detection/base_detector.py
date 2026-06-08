"""Base detector interface for all detection models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class BaseDetector(ABC):
    """Abstract base class for object detection models.

    All detectors expose a uniform ``predict`` signature so callers never need
    to know the concrete model type. Detector-specific parameters (e.g. YOLO's
    NMS ``iou``) are accepted by name; detectors that don't use a given
    parameter simply ignore it.
    """

    @abstractmethod
    def predict(
        self, image: np.ndarray, conf: float = 0.3, iou: float = 0.5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            conf: Confidence threshold (all detectors)
            iou: NMS IoU threshold (used by detectors that perform NMS; ignored
                otherwise)
            **kwargs: Additional model-specific parameters

        Returns:
            List of detection dictionaries, each containing:
                - bbox: [x1, y1, x2, y2] in pixel coordinates (float)
                - confidence: detection confidence score (float)
                - class_id: integer class ID (int)
        """
        pass

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Return mapping from class_id to class name."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of classes the model can detect."""
        pass

    @abstractmethod
    def to(self, device: str):
        """Move model to the specified device."""
        pass
