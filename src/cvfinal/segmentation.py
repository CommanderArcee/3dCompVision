from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _largest_contour_mask(binary: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, color=255, thickness=-1)
    return mask


def auto_rect(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x = int(w * 0.15)
    y = int(h * 0.15)
    rw = int(w * 0.7)
    rh = int(h * 0.7)
    return x, y, rw, rh


def generate_mask(
    first_frame_path: Path,
    out_mask_path: Path,
    out_debug_path: Path,
    use_interactive_roi: bool = False,
) -> np.ndarray:
    img = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read frame for segmentation: {first_frame_path}")

    if use_interactive_roi:
        roi = cv2.selectROI("Select Object", img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            roi = auto_rect(img)
    else:
        roi = auto_rect(img)

    mask = np.zeros(img.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, roi, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_mask = _largest_contour_mask(closed)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_debug_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_mask_path), final_mask)

    debug = img.copy()
    debug[final_mask == 0] = (0, 0, 0)
    cv2.imwrite(str(out_debug_path), debug)

    return final_mask
