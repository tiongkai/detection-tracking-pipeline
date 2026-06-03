"""Detection model factory for creating detector instances from config."""

from typing import Dict, Any, Optional
from pathlib import Path

from .base_detector import BaseDetector
from .yolo_detector import YOLODetector
from .rfdetr_detector import RFDetrDetector


def create_detector(config: Dict[str, Any]) -> BaseDetector:
    """Create a detector instance from configuration.
    
    Args:
        config: Detection configuration dictionary with keys:
            - model_type: 'yolo' or 'rfdetr'
            - weights: Path to model weights
            - device: Device to run on (default: 'cuda:0')
            - size: For RF-DETR: 'base' or 'large'
            
    Returns:
        BaseDetector instance
        
    Example:
        config = {
            'model_type': 'yolo',
            'weights': 'weights/best.pt',
            'device': 'cuda:0'
        }
        detector = create_detector(config)
        
        config = {
            'model_type': 'rfdetr',
            'weights': 'weights/rfdetr_base.pth',
            'device': 'cuda:0',
            'size': 'base'
        }
        detector = create_detector(config)
    """
    model_type = config.get('model_type', 'yolo').lower()
    weights_path = config.get('weights')
    device = config.get('device', 'cuda:0')
    
    if not weights_path:
        raise ValueError("'weights' must be specified in config")
    
    if model_type == 'yolo':
        return YOLODetector(weights_path, device=device)
    elif model_type == 'rfdetr':
        model_size = config.get('size', 'base')
        return RFDetrDetector(weights_path, device=device, model_size=model_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: 'yolo', 'rfdetr'")


def create_detector_from_config_file(config_path: str) -> BaseDetector:
    """Create detector from YAML config file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        BaseDetector instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract detection config (might be nested under 'detection' key)
    if 'detection' in config:
        detection_config = config['detection']
    else:
        detection_config = config
    
    return create_detector(detection_config)