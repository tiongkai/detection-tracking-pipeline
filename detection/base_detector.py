"""Base detector interface for all detection models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class BaseDetector(ABC):
    """Abstract base class for object detection models."""
    
    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """Run inference on a single image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            **kwargs: Model-specific parameters (conf, iou, threshold, etc.)
            
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
    
    def to(self, device: str):
        """Move model to specified device."""
        pass