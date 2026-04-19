from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    data_root: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/output")
    frame_step: int = 15
    frame_resize_w: int = 800
    frame_resize_h: int = 600
    ratio_test: float = 0.75
    min_pnp_points: int = 8
    reproj_threshold_px: float = 2.0
    random_seed: int = 42
    kmeans_clusters: int = 50
    svm_test_size: float = 0.3


def ensure_dirs(cfg: PipelineConfig) -> None:
    for path in [
        cfg.data_root,
        cfg.raw_dir,
        cfg.interim_dir,
        cfg.processed_dir,
        cfg.output_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
