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
        focal = float(width)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    if len(pts1) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3d = (points4d[:3] / points4d[3]).T
    return points3d


def _reprojection_error(P: np.ndarray, point3d: np.ndarray, point2d: np.ndarray) -> float:
    homog = np.append(point3d, 1.0)
    proj = P @ homog
    proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj - point2d))


def run_incremental_sfm(
    image_shape: tuple[int, int],
    features: list[FeaturePack],
    pair_matches: list[PairMatches],
    reproj_threshold_px: float = 2.0,
) -> tuple[ReconstructionResult, np.ndarray]:
    h, w = image_shape
    K = make_intrinsics(w, h)

    if not pair_matches:
        return (
            ReconstructionResult(
                points_3d=np.zeros((0, 3)),
                poses=[CameraPose(np.eye(3), np.zeros(3))],
                observations=[],
                mean_reprojection_error=0.0,
            ),
            K,
        )

    first = pair_matches[0]
    if len(first.points_a) < 8:
        raise RuntimeError("Not enough matches in first pair to estimate essential matrix")

    E, inlier_mask = cv2.findEssentialMat(first.points_a, first.points_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed")

    _, R1, t1, pose_mask = cv2.recoverPose(E, first.points_a, first.points_b, K)

    poses: list[CameraPose] = [CameraPose(np.eye(3), np.zeros(3)), CameraPose(R1, t1.ravel())]

    inliers = (inlier_mask.ravel() > 0) & (pose_mask.ravel() > 0)
    pts0 = first.points_a[inliers]
    pts1 = first.points_b[inliers]

    P0 = projection_matrix(K, poses[0].R, poses[0].t)
    P1 = projection_matrix(K, poses[1].R, poses[1].t)

    points3d = triangulate(P0, P1, pts0, pts1)

    observations: list[dict] = []
    point_ids: list[int] = []

    for p3d, p2d_0, p2d_1 in zip(points3d, pts0, pts1):
        pid = len(point_ids)
        point_ids.append(pid)
        observations.append({"point_id": pid, "frame": 0, "xy": p2d_0.tolist()})
        observations.append({"point_id": pid, "frame": 1, "xy": p2d_1.tolist()})

    frame_key_to_point: dict[tuple[int, tuple[int, int]], int] = {}
    for pid, p2d_1 in enumerate(pts1):
        key = (1, (int(round(p2d_1[0])), int(round(p2d_1[1]))))
        frame_key_to_point[key] = pid

    all_points = points3d.tolist()

    for i in range(1, len(pair_matches)):
        pair = pair_matches[i]
        prev_frame = pair.idx_a
        curr_frame = pair.idx_b

        pts_prev = pair.points_a
        pts_curr = pair.points_b

        obj_points = []
        img_points = []
        matched_pid = []

        for p_prev, p_curr in zip(pts_prev, pts_curr):
            key_prev = (prev_frame, (int(round(p_prev[0])), int(round(p_prev[1]))))
            if key_prev in frame_key_to_point:
                pid = frame_key_to_point[key_prev]
                obj_points.append(all_points[pid])
                img_points.append(p_curr)
                matched_pid.append(pid)

        if len(obj_points) >= 8:
            obj_points_np = np.asarray(obj_points, dtype=np.float64)
            img_points_np = np.asarray(img_points, dtype=np.float64)

            ok, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                obj_points_np,
                img_points_np,
                K,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok:
                continue

            R_new, _ = cv2.Rodrigues(rvec)
            t_new = tvec.ravel()
            poses.append(CameraPose(R_new, t_new))
        else:
            poses.append(poses[-1])

        P_prev = projection_matrix(K, poses[prev_frame].R, poses[prev_frame].t)
        P_curr = projection_matrix(K, poses[curr_frame].R, poses[curr_frame].t)

        for p_prev, p_curr in zip(pts_prev, pts_curr):
            key_prev = (prev_frame, (int(round(p_prev[0])), int(round(p_prev[1]))))
            key_curr = (curr_frame, (int(round(p_curr[0])), int(round(p_curr[1]))))

            if key_curr in frame_key_to_point:
                continue

            if key_prev in frame_key_to_point:
                pid = frame_key_to_point[key_prev]
                frame_key_to_point[key_curr] = pid
                observations.append({"point_id": pid, "frame": curr_frame, "xy": p_curr.tolist()})
                continue

            tri = triangulate(P_prev, P_curr, p_prev.reshape(1, 2), p_curr.reshape(1, 2))
            if len(tri) == 0:
                continue
            point = tri[0]
            if not np.isfinite(point).all():
                continue
            pid = len(all_points)
            all_points.append(point.tolist())
            frame_key_to_point[key_prev] = pid
            frame_key_to_point[key_curr] = pid
            observations.append({"point_id": pid, "frame": prev_frame, "xy": p_prev.tolist()})
            observations.append({"point_id": pid, "frame": curr_frame, "xy": p_curr.tolist()})

    points_np = np.asarray(all_points, dtype=np.float64)

    errors = []
    for obs in observations:
        pid = obs["point_id"]
        frame = obs["frame"]
        xy = np.asarray(obs["xy"], dtype=np.float64)
        if frame >= len(poses):
            continue
        P = projection_matrix(K, poses[frame].R, poses[frame].t)
        err = _reprojection_error(P, points_np[pid], xy)
        if np.isfinite(err):
            errors.append(err)

    mean_err = float(np.mean(errors)) if errors else 0.0

    if reproj_threshold_px > 0 and errors:
        keep = np.ones(len(points_np), dtype=bool)
        point_errs: dict[int, list[float]] = {}
        for obs in observations:
            pid = obs["point_id"]
            frame = obs["frame"]
            if frame >= len(poses):
                continue
            P = projection_matrix(K, poses[frame].R, poses[frame].t)
            err = _reprojection_error(P, points_np[pid], np.asarray(obs["xy"]))
            point_errs.setdefault(pid, []).append(err)
        for pid, errs in point_errs.items():
            if np.mean(errs) > reproj_threshold_px:
                keep[pid] = False
        points_np = points_np[keep]

    result = ReconstructionResult(
        points_3d=points_np,
        poses=poses,
        observations=observations,
        mean_reprojection_error=mean_err,
    )
    return result, K
