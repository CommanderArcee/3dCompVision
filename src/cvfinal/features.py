from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_utils import load_gray
from .preprocessing import mask_background


@dataclass
class FeaturePack:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass
class PairMatches:
    idx_a: int
    idx_b: int
    feat_idx_a: np.ndarray
    feat_idx_b: np.ndarray
    points_a: np.ndarray
    points_b: np.ndarray
    good_matches: list[cv2.DMatch]


def create_sift() -> cv2.SIFT:
    return cv2.SIFT_create()


def detect_features(gray: np.ndarray, mask: np.ndarray | None = None) -> FeaturePack:
    sift = create_sift()
    if mask is not None:
        gray = mask_background(gray, mask)

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.zeros((0, 128), dtype=np.float32)
    return FeaturePack(keypoints=keypoints, descriptors=descriptors)


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray, ratio_test: float = 0.75) -> list[cv2.DMatch]:
    if len(desc_a) == 0 or len(desc_b) == 0:
        return []

    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)
    return good


def get_match_points(
    kps_a: list[cv2.KeyPoint],
    kps_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    pts_a = np.float32([kps_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in matches])
    return pts_a, pts_b


def draw_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    kps_a: list[cv2.KeyPoint],
    kps_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    out_path: Path,
    max_draw: int = 100,
) -> None:
    to_draw = matches[:max_draw]
    vis = cv2.drawMatches(
        img_a,
        kps_a,
        img_b,
        kps_b,
        to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def pairwise_matches(
    image_paths: list[Path],
    mask: np.ndarray | list[np.ndarray] | None,
    ratio_test: float,
    out_dir: Path,
) -> tuple[list[FeaturePack], list[PairMatches]]:
    features: list[FeaturePack] = []
    grays = [load_gray(p) for p in image_paths]
    masks: list[np.ndarray | None] = [None] * len(grays)
    if isinstance(mask, list):
        for i in range(min(len(mask), len(grays))):
            masks[i] = mask[i]
    elif mask is not None:
        masks = [mask] * len(grays)

    for i, g in enumerate(grays):
        features.append(detect_features(g, mask=masks[i]))

    pairs: list[PairMatches] = []
    for i in range(len(grays) - 1):
        f1 = features[i]
        f2 = features[i + 1]
        good = match_descriptors(f1.descriptors, f2.descriptors, ratio_test=ratio_test)
        idx_a = np.asarray([m.queryIdx for m in good], dtype=np.int32) if good else np.zeros((0,), dtype=np.int32)
        idx_b = np.asarray([m.trainIdx for m in good], dtype=np.int32) if good else np.zeros((0,), dtype=np.int32)
        pts1, pts2 = get_match_points(f1.keypoints, f2.keypoints, good) if good else (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

        if len(good) >= 8:
            F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.99)
            if F is not None and inlier_mask is not None:
                keep = inlier_mask.ravel().astype(bool)
                good = [m for m, k in zip(good, keep) if k]
                idx_a = idx_a[keep]
                idx_b = idx_b[keep]
                pts1 = pts1[keep]
                pts2 = pts2[keep]

        draw_matches(
            grays[i],
            grays[i + 1],
            f1.keypoints,
            f2.keypoints,
            good,
            out_dir / f"matches_{i:04d}_{i+1:04d}.png",
        )
        pairs.append(PairMatches(i, i + 1, idx_a, idx_b, pts1, pts2, good))
    return features, pairs
