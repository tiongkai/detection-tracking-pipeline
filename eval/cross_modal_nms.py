import numpy as np


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cross_modal_nms(detections: np.ndarray, class_groups: dict, iou_thresh: float = 0.5) -> np.ndarray:
    """Apply NMS across model classes that belong to the same project class.

    Args:
        detections: (N, 6) array — [x1, y1, x2, y2, conf, cls].
        class_groups: dict mapping project_class_id -> set of model class IDs.
                      e.g. {0: {0, 1, 6, 7}, 1: {2, 8}, ...}
        iou_thresh: IoU threshold for suppression.

    Returns:
        Filtered (M, 6) array with cross-modal duplicates removed.
    """
    if len(detections) == 0:
        return detections

    keep = np.ones(len(detections), dtype=bool)

    all_grouped_cls = set()
    for cls_ids in class_groups.values():
        all_grouped_cls.update(cls_ids)

    for group_cls_ids in class_groups.values():
        if len(group_cls_ids) < 2:
            continue

        group_mask = np.array([int(d[5]) in group_cls_ids for d in detections])
        group_indices = np.where(group_mask)[0]

        if len(group_indices) < 2:
            continue

        group_confs = detections[group_indices, 4]
        order = group_confs.argsort()[::-1]

        for i in range(len(order)):
            idx_i = group_indices[order[i]]
            if not keep[idx_i]:
                continue
            for j in range(i + 1, len(order)):
                idx_j = group_indices[order[j]]
                if not keep[idx_j]:
                    continue
                if _iou(detections[idx_i, :4], detections[idx_j, :4]) > iou_thresh:
                    keep[idx_j] = False

    return detections[keep]
