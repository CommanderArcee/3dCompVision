from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_utils import save_json


def segmentation_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask > 0
    gt = gt_mask > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def save_metrics(
    out_path: Path,
    num_points: int,
    mean_reprojection_error: float,
    classification_accuracy: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "num_points_3d": int(num_points),
        "mean_reprojection_error": float(mean_reprojection_error),
        "classification_accuracy": float(classification_accuracy),
    }
    if extra:
        payload.update(extra)
    save_json(payload, out_path)
