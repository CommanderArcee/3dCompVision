from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_ply_points(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        end_idx = lines.index("end_header")
    except ValueError as e:
        raise RuntimeError(f"Invalid PLY file (missing end_header): {path}") from e

    if end_idx + 1 >= len(lines):
        return np.zeros((0, 3), dtype=np.float64)

    pts = np.loadtxt(lines[end_idx + 1 :], dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts[:, :3]


def main() -> None:
    parser = argparse.ArgumentParser(description="View ASCII PLY point cloud with matplotlib")
    parser.add_argument("--ply", type=Path, required=True, help="Path to .ply file")
    parser.add_argument("--max-points", type=int, default=50000, help="Max points to render")
    parser.add_argument("--point-size", type=float, default=1.0, help="Scatter point size")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save plot image")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    args = parser.parse_args()

    points = load_ply_points(args.ply)
    if len(points) == 0:
        raise RuntimeError(f"No points found in file: {args.ply}")

    if len(points) > args.max_points:
        idx = np.linspace(0, len(points) - 1, args.max_points).astype(int)
        points = points[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=args.point_size, c=points[:, 2], cmap="viridis")
    ax.set_title(args.ply.name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=220)
        print(f"Saved plot to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
