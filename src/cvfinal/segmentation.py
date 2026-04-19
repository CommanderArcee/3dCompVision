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


def _filtered_contour_mask(binary: np.ndarray, min_area: float) -> np.ndarray:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    mask = np.zeros_like(binary)
    kept = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            cv2.drawContours(mask, [c], -1, color=255, thickness=-1)
            kept += 1
    if kept == 0:
        return _largest_contour_mask(binary)
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
    dilate_iter: int = 2,
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

    final_mask = _segment_with_roi(img, roi, dilate_iter=dilate_iter)

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_debug_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_mask_path), final_mask)

    debug = img.copy()
    debug[final_mask == 0] = (0, 0, 0)
    cv2.imwrite(str(out_debug_path), debug)

    return final_mask


def _segment_with_roi(img: np.ndarray, roi: tuple[int, int, int, int], dilate_iter: int) -> np.ndarray:

    mask = np.zeros(img.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, roi, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    binary = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    h, w = closed.shape[:2]
    min_area = 0.002 * h * w
    final_mask = _filtered_contour_mask(closed, min_area=min_area)

    if dilate_iter > 0:
        final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iter)

    return final_mask


def generate_masks_for_frames(
    frame_paths: list[Path],
    out_dir: Path,
    use_interactive_roi: bool = False,
    dilate_iter: int = 2,
) -> list[np.ndarray]:
    if not frame_paths:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    first_img = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise FileNotFoundError(f"Could not read frame for segmentation: {frame_paths[0]}")

    if use_interactive_roi:
        roi = cv2.selectROI("Select Object", first_img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            roi = auto_rect(first_img)
    else:
        roi = auto_rect(first_img)

    masks: list[np.ndarray] = []
    for idx, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        m = _segment_with_roi(img, roi, dilate_iter=dilate_iter)
        masks.append(m)

        cv2.imwrite(str(out_dir / f"mask_{idx:04d}.png"), m)
        debug = img.copy()
        debug[m == 0] = (0, 0, 0)
        cv2.imwrite(str(out_dir / f"mask_debug_{idx:04d}.png"), debug)

    # Compatibility outputs for existing checks
    if masks:
        cv2.imwrite(str(out_dir / "mask.png"), masks[0])
        dbg0 = first_img.copy()
        dbg0[masks[0] == 0] = (0, 0, 0)
        cv2.imwrite(str(out_dir / "mask_debug.png"), dbg0)

    return masks
