from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import list_images, load_color


def preprocess_frames(input_dir: Path, output_dir: Path, overlay_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for image_path in list_images(input_dir):
        color = load_color(image_path)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, threshold1=80, threshold2=160)

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(color, 0.8, edges_bgr, 0.6, 0.0)

        processed_name = image_path.stem + "_proc.png"
        overlay_name = image_path.stem + "_overlay.png"

        processed_path = output_dir / processed_name
        overlay_path = overlay_dir / overlay_name

        cv2.imwrite(str(processed_path), enhanced)
        cv2.imwrite(str(overlay_path), overlay)
        saved.append(processed_path)

    return saved


def mask_background(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = gray.copy()
    masked[mask == 0] = 0
    return masked
