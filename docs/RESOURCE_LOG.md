# Resource and Technique Log

This file records every external dataset/reference and every major technique used in the implementation.

## Datasets

1. **Google Objectron**
   - Type: Real-world object-centric videos
   - Usage: Primary verification for frame extraction, preprocessing, feature matching, SfM/PnP reconstruction
   - Link: https://github.com/google-research-datasets/Objectron

2. **COIL-100 (Columbia Object Image Library)**
   - Type: Multi-view real object images
   - Usage: Classification (BoVW + SVM), confusion matrix, auxiliary reconstruction verification
   - Link: https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php

3. **ALOI (Amsterdam Library of Object Images)**
   - Type: Real object images with viewpoint and illumination variations
   - Usage: Optional robustness verification
   - Link: https://aloi.science.uva.nl/

## Core Techniques in Code

1. CLAHE histogram equalization (`cv2.createCLAHE`)
2. Canny edge detection (`cv2.Canny`)
3. GrabCut segmentation (`cv2.grabCut`)
4. Morphological closing (`cv2.morphologyEx`)
5. SIFT keypoints and descriptors (`cv2.SIFT_create`)
6. FLANN matching + Lowe ratio test
7. Essential matrix estimation with RANSAC (`cv2.findEssentialMat`)
8. Relative pose recovery (`cv2.recoverPose`)
9. Triangulation (`cv2.triangulatePoints`)
10. PnP RANSAC (`cv2.solvePnPRansac`)
11. Dense optical flow (Farneback)
12. PCA point-cloud alignment
13. BoVW + SVM classification (`sklearn`)
14. Confusion matrix and classification report

## Implementation Usage Trace

- `src/cvfinal/preprocessing.py`
  - CLAHE histogram equalization
  - Canny edge detection
- `src/cvfinal/segmentation.py`
  - GrabCut segmentation
  - Morphological closing
  - Largest contour mask filtering
- `src/cvfinal/features.py`
  - SIFT feature extraction
  - FLANN matching and Lowe ratio test
- `src/cvfinal/reconstruction.py`
  - Essential matrix estimation (RANSAC)
  - Pose recovery
  - Triangulation
  - Incremental PnP-RANSAC
- `src/cvfinal/optical_flow.py`
  - Dense Farneback optical flow
- `src/cvfinal/classification.py`
  - Bag of Visual Words (KMeans)
  - SVM classification
  - Confusion matrix and classification report
- `src/cvfinal/visualization.py`
  - PCA point-cloud axis alignment
  - 3D plotting and GIF generation

## References (Theory)

1. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004.
2. D. A. Forsyth and J. Ponce, *Computer Vision: A Modern Approach*, 2011.
