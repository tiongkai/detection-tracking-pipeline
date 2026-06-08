"""Detection model factory for creating detector instances from config."""

from typing import Dict, Any

from .base_detector import BaseDetector
from .yolo_detector import YOLODetector
from .rfdetr_detector import RFDetrDetector


def create_detector(config: Dict[str, Any]) -> BaseDetector:
    """Create a detector instance from configuration.

    Args:
        config: Detection configuration dictionary with keys:
            - model_type: 'yolo' or 'rfdetr'
            - weights: Path to model weights (required)
            - device: Device to run on (default: 'cuda:0')
            - size: For RF-DETR: 'base' or 'large'
            - class_names: For RF-DETR: optional {class_id: name} mapping

    Returns:
        BaseDetector instance

    Example:
        detector = create_detector({
            'model_type': 'yolo',
            'weights': 'weights/best.pt',
            'device': 'cuda:0',
        })
    """
    model_type = config.get('model_type', 'yolo').lower()
    weights_path = config.get('weights')
    device = config.get('device', 'cuda:0')

    if not weights_path:
        raise ValueError("'weights' must be specified in config")

    if model_type == 'yolo':
        return YOLODetector(weights_path, device=device)
    elif model_type == 'rfdetr':
        return RFDetrDetector(
            weights_path,
            device=device,
            model_size=config.get('size', 'base'),
            class_names=config.get('class_names'),
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: 'yolo', 'rfdetr'"
        )


def create_detector_from_config_file(config_path: str) -> BaseDetector:
    """Create detector from a YAML config file.

    The detection config may be nested under a top-level 'detection' key or be
    the top-level document itself.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        BaseDetector instance
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    detection_config = config.get('detection', config)
    return create_detector(detection_config)
