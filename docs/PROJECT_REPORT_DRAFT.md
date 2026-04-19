# Computer Vision Final Project Report Draft

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
- Number of 3D points: 5
- Mean reprojection error: 3988.902620342679
- Classification accuracy: 1.0

### Output Files
- Confusion matrix: data/output/classification/confusion_matrix.png
- Classification report: data/output/classification/classification_report.txt
- Sparse cloud: data/output/reconstruction/sparse_cloud.ply
- Rotated cloud: data/output/reconstruction/rotated_cloud.ply

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
