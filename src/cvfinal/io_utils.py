from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def list_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in exts])


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float64)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(pts)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def chunked(iterable: Iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
