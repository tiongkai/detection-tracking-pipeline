"""RF-DETR detector implementation."""

from typing import Dict, List, Any, Optional
import numpy as np
import cv2
from PIL import Image

from .base_detector import BaseDetector


class RFDetrDetector(BaseDetector):
    """RF-DETR detector wrapper."""
    
    def __init__(self, weights_path: str, device: str = "cuda:0", 
                 model_size: str = "base"):
        """Initialize RF-DETR detector.
        
        Args:
            weights_path: Path to RF-DETR weights file
            device: Torch device (cuda:0, cpu, etc.)
            model_size: 'base' or 'large'
        """
        from rfdetr import RFDETRBase, RFDETRLarge
        
        self.device = device
        self.model_size = model_size
        
        if model_size.lower() == 'large':
            self.model = RFDETRLarge(pretrain_weights=weights_path)
        else:
            self.model = RFDETRBase(pretrain_weights=weights_path)
        
        self._class_names = self._load_class_names()
        
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names for RF-DETR.
        
        RF-DETR uses COCO classes by default. Override if needed.
        """
        # COCO class mapping (80 classes)
        coco_classes = {
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
            79: 'toothbrush'
        }
        
        # For maritime domain, you might want to override with your class mapping
        # You can load from a config file here if needed
        return coco_classes
    
    def predict(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """Run RF-DETR inference.
        
        Args:
            image: BGR image as numpy array
            **kwargs: RF-DETR parameters:
                - threshold: Confidence threshold (default: 0.3)
                - device: Override device for this prediction
                
        Returns:
            List of detection dictionaries
        """
        threshold = kwargs.get('threshold', 0.3)
        
        # Convert BGR (OpenCV) to RGB (PIL expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run inference
        detections = self.model.predict(pil_image, threshold=threshold)
        
        # Convert to standard format
        results = []
        if detections is not None and len(detections) > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                results.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(detections.confidence[i]),
                    'class_id': int(detections.class_id[i])
                })
        
        return results
    
    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names
    
    @property
    def num_classes(self) -> int:
        return len(self._class_names)
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        # RF-DETR device handling - you may need to implement based on RF-DETR API
        # Some versions use .to(device) or handle device in predict