from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(metrics: dict) -> str:
    return f"""# Computer Vision Final Project Report Draft

## 1. Title
Multi-Technique 3D Reconstruction from a Rotating Video: A Comprehensive Computer Vision Pipeline

## 2. Problem Statement
This project reconstructs a sparse 3D point cloud from monocular object-centric video while integrating classical CV tasks including preprocessing, segmentation, feature extraction, geometric reconstruction, optical flow visualization, and object classification.

## 3. Dataset Details
- Primary video dataset: Objectron (sample clip)
- Classification dataset: COIL-100 (subset)
- Additional verification sequence: ordered COIL object rotation frames

## 4. Preprocessing
- CLAHE histogram equalization
- Canny edge detection
- Edge-overlay visualization

## 5. Tasks Performed
- Segmentation: GrabCut + morphology + contour mask
- Feature extraction: SIFT + FLANN + Lowe ratio test
- 3D geometry: Essential matrix, pose recovery, triangulation, incremental PnP
- Motion analysis: Dense optical flow
- Recognition: BoVW + SVM with confusion matrix
- Shape analysis: PCA alignment of 3D cloud

## 6. Environment
- Language: Python
- Package manager: uv
- Core libraries: OpenCV, NumPy, SciPy, scikit-learn, matplotlib

## 7. Results (Auto-filled)
- Number of 3D points: {metrics.get('num_points_3d', 'N/A')}
- Mean reprojection error: {metrics.get('mean_reprojection_error', 'N/A')}
- Classification accuracy: {metrics.get('classification_accuracy', 'N/A')}

### Output Files
- Confusion matrix: {metrics.get('confusion_matrix_path', 'N/A')}
- Classification report: {metrics.get('classification_report_path', 'N/A')}
- Sparse cloud: {metrics.get('sparse_cloud_path', 'N/A')}
- Rotated cloud: {metrics.get('rotated_cloud_path', 'N/A')}

## 8. Program Proof Checklist
- Include screenshots for each module execution and output artifact.
- Include command history used for dataset download and pipeline run.

## 9. Relevance to Subject
The project is fully Computer Vision focused and uses image processing, feature extraction, geometric vision, motion estimation, segmentation, and recognition topics from the syllabus.

## 10. References
1. Hartley & Zisserman, Multiple View Geometry, 2nd ed.
2. Forsyth & Ponce, Computer Vision: A Modern Approach.
3. Objectron (CVPR 2021).
4. COIL-100 Technical Report (1996).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report draft markdown from metrics")
    parser.add_argument("--metrics", type=Path, default=Path("data/output/metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/PROJECT_REPORT_DRAFT.md"))
    args = parser.parse_args()

    metrics = {}
    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    report = build_report(metrics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote report draft to {args.out}")


if __name__ == "__main__":
    main()
