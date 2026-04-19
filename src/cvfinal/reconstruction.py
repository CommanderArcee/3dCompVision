from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import FeaturePack, PairMatches


@dataclass
class CameraPose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class ReconstructionResult:
    points_3d: np.ndarray
    poses: list[CameraPose]
    observations: list[dict]
    mean_reprojection_error: float


def make_intrinsics(width: int, height: int, focal: float | None = None) -> np.ndarray:
    if focal is None:
        focal = float(max(width, height)) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def _reprojection_error(P: np.ndarray, point3d: np.ndarray, point2d: np.ndarray) -> float:
    homog = np.append(point3d, 1.0)
    proj = P @ homog
    if abs(proj[2]) < 1e-9:
        return float("inf")
    proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj - point2d))


def _triangulate_pair(
    K: np.ndarray,
    pose_a: CameraPose,
    pose_b: CameraPose,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pts_a) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=bool)

    P_a = projection_matrix(K, pose_a.R, pose_a.t)
    P_b = projection_matrix(K, pose_b.R, pose_b.t)

    points4d = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
    points3d = (points4d[:3] / points4d[3]).T

    X_a = (pose_a.R @ points3d.T + pose_a.t.reshape(3, 1)).T
    X_b = (pose_b.R @ points3d.T + pose_b.t.reshape(3, 1)).T
    positive_depth = (X_a[:, 2] > 0.05) & (X_b[:, 2] > 0.05)

    err_a = np.array([_reprojection_error(P_a, p3d, p2d) for p3d, p2d in zip(points3d, pts_a)])
    err_b = np.array([_reprojection_error(P_b, p3d, p2d) for p3d, p2d in zip(points3d, pts_b)])
    reproj_ok = (err_a < 8.0) & (err_b < 8.0)

    finite_ok = np.isfinite(points3d).all(axis=1)
    keep = positive_depth & reproj_ok & finite_ok
    return points3d, keep


def _best_initial_pair(pair_matches: list[PairMatches]) -> int:
    best_idx = -1
    best_score = -1
    for i, pair in enumerate(pair_matches):
        score = len(pair.points_a)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0 or best_score < 8:
        raise RuntimeError("No frame pair has enough inlier matches to initialize reconstruction")
    return best_idx


def run_incremental_sfm(
    image_shape: tuple[int, int],
    features: list[FeaturePack],
    pair_matches: list[PairMatches],
    reproj_threshold_px: float = 6.0,
) -> tuple[ReconstructionResult, np.ndarray]:
    h, w = image_shape
    K = make_intrinsics(w, h)

    n_frames = len(features)
    if n_frames < 2 or not pair_matches:
        return (
            ReconstructionResult(
                points_3d=np.zeros((0, 3), dtype=np.float64),
                poses=[CameraPose(np.eye(3), np.zeros(3))],
                observations=[],
                mean_reprojection_error=0.0,
            ),
            K,
        )

    init_idx = _best_initial_pair(pair_matches)
    init_pair = pair_matches[init_idx]

    E, e_mask = cv2.findEssentialMat(
        init_pair.points_a,
        init_pair.points_b,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.8,
    )
    if E is None or e_mask is None:
        raise RuntimeError("Essential matrix estimation failed for initialization")

    _, R_init, t_init, pose_mask = cv2.recoverPose(E, init_pair.points_a, init_pair.points_b, K)
    inliers = (e_mask.ravel() > 0) & (pose_mask.ravel() > 0)

    pts_a = init_pair.points_a[inliers]
    pts_b = init_pair.points_b[inliers]
    idx_a = init_pair.feat_idx_a[inliers]
    idx_b = init_pair.feat_idx_b[inliers]

    poses_opt: list[CameraPose | None] = [None] * n_frames
    poses_opt[init_pair.idx_a] = CameraPose(np.eye(3), np.zeros(3))
    poses_opt[init_pair.idx_b] = CameraPose(R_init, t_init.ravel())

    points3d, keep = _triangulate_pair(K, poses_opt[init_pair.idx_a], poses_opt[init_pair.idx_b], pts_a, pts_b)

    map3d: list[np.ndarray] = []
    feature_to_point: dict[tuple[int, int], int] = {}
    observations: list[dict] = []

    for p3d, ok, fa, fb, p2a, p2b in zip(points3d, keep, idx_a, idx_b, pts_a, pts_b):
        if not ok:
            continue
        pid = len(map3d)
        map3d.append(p3d)
        feature_to_point[(init_pair.idx_a, int(fa))] = pid
        feature_to_point[(init_pair.idx_b, int(fb))] = pid
        observations.append({"point_id": pid, "frame": init_pair.idx_a, "xy": p2a.tolist()})
        observations.append({"point_id": pid, "frame": init_pair.idx_b, "xy": p2b.tolist()})

    # Propagate poses and triangulate new points using consecutive pairs.
    for pair in pair_matches:
        a = pair.idx_a
        b = pair.idx_b

        pose_a = poses_opt[a]
        if pose_a is None:
            continue

        # Estimate pose of b if missing via PnP from known 3D-2D correspondences.
        if poses_opt[b] is None:
            obj_pts = []
            img_pts = []
            for fa, fb, p2b in zip(pair.feat_idx_a, pair.feat_idx_b, pair.points_b):
                key_a = (a, int(fa))
                if key_a in feature_to_point:
                    obj_pts.append(map3d[feature_to_point[key_a]])
                    img_pts.append(p2b)

            if len(obj_pts) >= 8:
                obj_np = np.asarray(obj_pts, dtype=np.float64)
                img_np = np.asarray(img_pts, dtype=np.float64)
                ok, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    obj_np,
                    img_np,
                    K,
                    None,
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=4.0,
                    confidence=0.99,
                    iterationsCount=200,
                )
                if ok and inliers_pnp is not None and len(inliers_pnp) >= 6:
                    R_b, _ = cv2.Rodrigues(rvec)
                    poses_opt[b] = CameraPose(R_b, tvec.ravel())

        pose_b = poses_opt[b]
        if pose_b is None:
            continue

        # Link existing tracks and triangulate fresh points.
        new_pts_a = []
        new_pts_b = []
        new_idx_a = []
        new_idx_b = []

        for fa, fb, p2a, p2b in zip(pair.feat_idx_a, pair.feat_idx_b, pair.points_a, pair.points_b):
            key_a = (a, int(fa))
            key_b = (b, int(fb))

            has_a = key_a in feature_to_point
            has_b = key_b in feature_to_point

            if has_a and not has_b:
                pid = feature_to_point[key_a]
                feature_to_point[key_b] = pid
                observations.append({"point_id": pid, "frame": b, "xy": p2b.tolist()})
                continue

            if has_a or has_b:
                continue

            new_pts_a.append(p2a)
            new_pts_b.append(p2b)
            new_idx_a.append(int(fa))
            new_idx_b.append(int(fb))

        if new_pts_a:
            tri_pts, tri_keep = _triangulate_pair(
                K,
                pose_a,
                pose_b,
                np.asarray(new_pts_a, dtype=np.float32),
                np.asarray(new_pts_b, dtype=np.float32),
            )
            for p3d, ok, fa, fb, p2a, p2b in zip(tri_pts, tri_keep, new_idx_a, new_idx_b, new_pts_a, new_pts_b):
                if not ok:
                    continue
                pid = len(map3d)
                map3d.append(p3d)
                feature_to_point[(a, int(fa))] = pid
                feature_to_point[(b, int(fb))] = pid
                observations.append({"point_id": pid, "frame": a, "xy": np.asarray(p2a).tolist()})
                observations.append({"point_id": pid, "frame": b, "xy": np.asarray(p2b).tolist()})

    # Fill missing poses with nearest known pose for downstream visualization.
    last_pose = CameraPose(np.eye(3), np.zeros(3))
    poses: list[CameraPose] = []
    for p in poses_opt:
        if p is not None:
            last_pose = p
        poses.append(last_pose)

    points_np = np.asarray(map3d, dtype=np.float64)
    if len(points_np) == 0:
        return ReconstructionResult(points_np, poses, observations, 0.0), K

    # Robust per-point filtering using median reprojection error across observations.
    point_errors: dict[int, list[float]] = {}
    all_errors: list[float] = []
    for obs in observations:
        pid = int(obs["point_id"])
        frame = int(obs["frame"])
        if pid >= len(points_np) or frame >= len(poses):
            continue
        P = projection_matrix(K, poses[frame].R, poses[frame].t)
        err = _reprojection_error(P, points_np[pid], np.asarray(obs["xy"], dtype=np.float64))
        if not np.isfinite(err):
            continue
        point_errors.setdefault(pid, []).append(err)
        all_errors.append(err)

    mean_err = float(np.mean(all_errors)) if all_errors else 0.0

    if reproj_threshold_px > 0 and point_errors:
        keep_mask = np.zeros(len(points_np), dtype=bool)
        for pid, errs in point_errors.items():
            med = float(np.median(errs))
            if med <= reproj_threshold_px:
                keep_mask[pid] = True
        kept = points_np[keep_mask]
        if len(kept) >= 20:
            points_np = kept

    return ReconstructionResult(points_np, poses, observations, mean_err), K
