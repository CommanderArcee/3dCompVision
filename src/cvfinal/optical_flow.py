from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import list_images, load_gray


def save_dense_optical_flow(frame_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = list_images(frame_dir)
    if len(frames) < 2:
        return []

    saved: list[Path] = []
    prev = load_gray(frames[0])

    for i in range(1, len(frames)):
        curr = load_gray(frames[i])
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out_path = out_dir / f"flow_{i-1:04d}_{i:04d}.png"
        cv2.imwrite(str(out_path), bgr)
        saved.append(out_path)
        prev = curr

    return saved
