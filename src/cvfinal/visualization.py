from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def pca_align(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_3d) == 0:
        return points_3d, np.zeros(3), np.eye(3)

    mean = points_3d.mean(axis=0)
    centered = points_3d - mean
    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    order = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, order]
    rotated = centered @ eig_vecs
    return rotated.real, mean, eig_vecs.real


def save_point_cloud_plot(points: np.ndarray, out_path: Path, title: str = "Point Cloud") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=points[:, 2], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_rotation_gif(points: np.ndarray, out_gif: Path, n_frames: int = 40) -> None:
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_gif.parent / "_gif_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(n_frames):
        angle = (360.0 * i) / n_frames
        frame_path = temp_dir / f"frame_{i:03d}.png"

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=points[:, 2], cmap="plasma")
        ax.view_init(elev=20, azim=angle)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        fig.tight_layout()
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        paths.append(frame_path)

    imgs = [imageio.imread(p) for p in paths]
    imageio.mimsave(out_gif, imgs, duration=0.08)


def overlay_points_on_image(
    image: np.ndarray,
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    out_path: Path,
) -> None:
    out = image.copy()
    if len(points_3d) > 0:
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        proj = (P @ homo.T).T
        uv = proj[:, :2] / proj[:, 2:3]
        for x, y in uv:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < out.shape[0] and 0 <= xi < out.shape[1]:
                cv2.circle(out, (xi, yi), 1, (0, 255, 0), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
