from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cvfinal.classification import run_bovw_svm
from cvfinal.config import PipelineConfig, ensure_dirs
from cvfinal.evaluation import save_metrics
from cvfinal.features import pairwise_matches
from cvfinal.io_utils import list_images, load_color, load_gray, reset_dir, save_ply
from cvfinal.optical_flow import save_dense_optical_flow
from cvfinal.preprocessing import preprocess_frames
from cvfinal.reconstruction import run_incremental_sfm
from cvfinal.segmentation import generate_mask
from cvfinal.visualization import (
    overlay_points_on_image,
    pca_align,
    save_point_cloud_plot,
    save_rotation_gif,
)
from scripts.extract_frames import extract_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full CV final project pipeline")
    parser.add_argument("--video", type=Path, default=Path("data/raw/objectron_sample.MOV"))
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--classification-root", type=Path, default=Path("data/interim/classification"))
    parser.add_argument("--enable-optical-flow", action="store_true")
    parser.add_argument("--interactive-roi", action="store_true")
    parser.add_argument("--frame-step", type=int, default=15)
    parser.add_argument("--frame-width", type=int, default=800)
    parser.add_argument("--frame-height", type=int, default=600)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--kmeans-clusters", type=int, default=50)
    parser.add_argument("--svm-test-size", type=float, default=0.3)
    parser.add_argument("--reproj-threshold", type=float, default=15.0)
    parser.add_argument("--disable-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig(
        frame_step=args.frame_step,
        frame_resize_w=args.frame_width,
        frame_resize_h=args.frame_height,
        ratio_test=args.ratio_test,
        kmeans_clusters=args.kmeans_clusters,
        svm_test_size=args.svm_test_size,
        reproj_threshold_px=args.reproj_threshold,
    )
    ensure_dirs(cfg)

    frame_dir = cfg.interim_dir / "frames"
    proc_dir = cfg.processed_dir / "frames"
    overlay_dir = cfg.output_dir / "preprocess_overlays"
    match_dir = cfg.output_dir / "matches"
    flow_dir = cfg.output_dir / "optical_flow"
    seg_dir = cfg.output_dir / "segmentation"
    recon_dir = cfg.output_dir / "reconstruction"
    cls_dir = cfg.output_dir / "classification"
    metrics_path = cfg.output_dir / "metrics.json"

    for d in [proc_dir, overlay_dir, match_dir, flow_dir, seg_dir, recon_dir, cls_dir]:
        reset_dir(d)

    if args.frames_dir is None:
        print("[Step 0] Extracting frames...")
        reset_dir(frame_dir)
        n_frames = extract_frames(args.video, frame_dir, cfg.frame_step, cfg.frame_resize_w, cfg.frame_resize_h)
        if n_frames < 2:
            raise RuntimeError("Need at least 2 extracted frames for reconstruction")
        print(f"Extracted {n_frames} frames")
    else:
        frame_dir = args.frames_dir
        n_frames = len(list_images(frame_dir))
        if n_frames < 2:
            raise RuntimeError("Need at least 2 input frames in --frames-dir")
        print(f"[Step 0] Using existing frame directory with {n_frames} frames")

    print("[Step 1] Preprocessing frames...")
    preprocess_frames(frame_dir, proc_dir, overlay_dir)

    frame_paths = list_images(frame_dir)
    proc_paths = list_images(proc_dir)
    if len(proc_paths) < 2:
        raise RuntimeError("Need at least 2 processed frames")

    print("[Step 2] Generating segmentation mask...")
    mask_path = seg_dir / "mask.png"
    mask_debug = seg_dir / "mask_debug.png"
    mask = generate_mask(
        first_frame_path=frame_paths[0],
        out_mask_path=mask_path,
        out_debug_path=mask_debug,
        use_interactive_roi=args.interactive_roi,
    )
    if args.disable_mask:
        mask = None

    print("[Step 3] Feature extraction and matching...")
    features, pairs = pairwise_matches(proc_paths, mask, cfg.ratio_test, match_dir)
    print(f"Matched {len(pairs)} consecutive frame pairs")

    print("[Step 4+5] Incremental SfM reconstruction...")
    sample = load_gray(proc_paths[0])
    recon, K = run_incremental_sfm(sample.shape, features, pairs, cfg.reproj_threshold_px)

    sparse_ply = recon_dir / "sparse_cloud.ply"
    save_ply(recon.points_3d, sparse_ply)

    print("[Step 6] Optical flow...")
    if args.enable_optical_flow:
        save_dense_optical_flow(proc_dir, flow_dir)

    print("[Step 7] PCA alignment...")
    rotated, mean, eig = pca_align(recon.points_3d)
    rotated_ply = recon_dir / "rotated_cloud.ply"
    save_ply(rotated, rotated_ply)

    print("[Step 8] SVM classification (BoVW)...")
    cls_result = run_bovw_svm(
        dataset_root=args.classification_root,
        output_dir=cls_dir,
        n_clusters=cfg.kmeans_clusters,
        test_size=cfg.svm_test_size,
        random_seed=cfg.random_seed,
    )
    (cls_dir / "classification_report.txt").write_text(cls_result.report_text, encoding="utf-8")

    print("[Step 9+10] Evaluation and visualization...")
    save_point_cloud_plot(recon.points_3d, recon_dir / "sparse_cloud.png", "Sparse Point Cloud")
    save_point_cloud_plot(rotated, recon_dir / "rotated_cloud.png", "PCA Aligned Point Cloud")
    save_rotation_gif(rotated, recon_dir / "rotating_cloud.gif")

    ref_img = load_color(frame_paths[0])
    if recon.poses:
        overlay_points_on_image(
            image=ref_img,
            points_3d=recon.points_3d,
            K=K,
            R=recon.poses[0].R,
            t=recon.poses[0].t,
            out_path=recon_dir / "overlay_projection.png",
        )

    save_metrics(
        out_path=metrics_path,
        num_points=len(recon.points_3d),
        mean_reprojection_error=recon.mean_reprojection_error,
        classification_accuracy=cls_result.accuracy,
        extra={
            "classification_report_path": str(cls_dir / "classification_report.txt"),
            "confusion_matrix_path": str(cls_result.confusion_matrix_path),
            "sparse_cloud_path": str(sparse_ply),
            "rotated_cloud_path": str(rotated_ply),
            "pca_mean": mean.tolist() if len(mean) else [],
            "pca_eigenvectors": eig.tolist() if len(eig) else [],
        },
    )

    print("[Step 11] Pipeline complete.")
    print(f"Outputs available under: {cfg.output_dir}")


if __name__ == "__main__":
    main()
