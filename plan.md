Title: Multi‑Technique 3D Reconstruction from a Rotating Video: A Comprehensive Computer Vision Pipeline

Abstract:
We present a complete Structure‑from‑Motion (SfM) pipeline that reconstructs a sparse 3D point cloud of an object from a monocular video. The project integrates 12+ classical and modern CV techniques including SIFT feature extraction, epipolar geometry, triangulation, PnP pose estimation, dense optical flow, PCA, SVM classification, segmentation, edge detection, histogram equalization, and morphological filtering. The output is a 3D point cloud visualized in Python, plus quantitative metrics (reprojection error, classification accuracy). The work covers all 6 syllabus units and uses 10+ lab experiments, demonstrating a holistic understanding of computer vision.

Implementation Plan (LLM‑Friendly Steps)
Below are numbered steps. Each step is a self-contained coding task. You can ask an LLM to generate code for each step sequentially.

Step 0: Setup & Data Preparation
Create project folder structure: images/, output/, scripts/

Install: opencv-python, opencv-contrib-python, numpy, matplotlib, scikit-learn, scipy

Write a script extract_frames.py that reads a video file (input.mp4), extracts every 15th frame, resizes to 800x600, saves as frame_0001.jpg, etc.

Output: 30–50 frames in images/ folder.

Step 1: Preprocessing – Histogram Equalization & Edge Overlay
For each frame:

Convert to grayscale.

Apply CLAHE (cv2.createCLAHE) to enhance contrast.

Run Canny edge detection (cv2.Canny).

Overlay edges on original image (optional: save overlay image).

Save processed frames in processed/ folder.

Lab used: Exp 3 (histogram equalization), Exp 4/5 (Canny).

Step 2: Object Segmentation (Mask Generation)
For the first frame only (or semi-automatically):

Let user select a bounding box around object (or use GrabCut with a rectangle).

Apply GrabCut (cv2.grabCut) to separate object from background.

Apply morphological closing (cv2.morphologyEx) to fill holes.

Find largest contour (cv2.findContours) and create a binary mask.

Save mask as mask.png.

For subsequent frames, use the same mask (or track mask using optical flow – optional).

Lab used: Exp 2 (contours), Exp 5 (morphology), Exp 7 (segmentation).

Step 3: Feature Extraction & Matching (SIFT)
For each consecutive frame pair (i, i+1):

Load processed grayscale images.

Apply the object mask: zero out background features.

Detect SIFT keypoints and descriptors: sift.detectAndCompute().

Match using FLANN matcher (or BFMatcher with k=2).

Apply Lowe’s ratio test (0.75).

Visualize and save match image for report.

Store good matches for all pairs.

Lab used: Exp 8 (SIFT features).

Step 4: Camera Intrinsics & First Two‑View Reconstruction
Define camera intrinsic matrix K (assume focal length = image width, principal point at center).

For the first pair (frame 0 and 1):

Extract matched point coordinates.

Compute Essential Matrix: E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC).

Recover pose: _, R, t, mask_pose = cv2.recoverPose(E, points1, points2, K).

Build projection matrices: P1 = K @ [I | 0], P2 = K @ [R | t].

Triangulate: points4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).

Convert to 3D: points3D = points4D[:3] / points4D[3].

Store 3D points and corresponding 2D observations for each point (track).

Lab used: Unit 2 (epipolar geometry), Unit 4 (SfM).

Step 5: Incremental Reconstruction (PnP for new frames)
For each new frame i (from 2 to N-1):

Match frame i with frame i-1 (using Step 3).

Find which existing 3D points are visible in frame i by checking 2D‑3D correspondences.

Run PnP RANSAC: success, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, K, distCoeffs=None).

Convert rvec to rotation matrix R via cv2.Rodrigues.

Build projection matrix P_new = K @ [R | tvec].

Find unmatched 2D points in frame i that are not yet triangulated.

For each such point, find its match in frame i-1 and triangulate using P_prev and P_new.

Append new 3D points to global cloud.

After each frame, optionally filter outliers by reprojection error (>2 pixels).

Lab used: Unit 4 (motion estimation), Unit 2 (3D vision).

Step 6: Dense Optical Flow Visualization (Optional but impressive)
For each consecutive frame pair (grayscale):

Compute Farneback optical flow: flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0).

Convert flow to color: hsv = np.zeros_like(prev_frame); set Hue = angle, Value = magnitude.

Save color flow image for each pair.

This is not used in reconstruction but shown as a separate output in PPT.

Lab used: Unit 4 (optical flow) – not explicitly in lab but mentioned in theory.

Step 7: PCA on 3D Point Cloud
Collect all 3D points into array (N, 3).

Center the points: subtract mean.

Compute covariance matrix: cov = np.cov(points_centered.T).

Compute eigenvalues and eigenvectors: eig_vals, eig_vecs = np.linalg.eig(cov).

Sort eigenvectors by eigenvalue (principal axes).

Rotate point cloud so that principal axes align with X, Y, Z.

Save rotated cloud as output_rotated.ply.

Lab used: Exp 8 (PCA).

Step 8: Object Classification using SVM (with Confusion Matrix)
For each frame, extract a global descriptor:

Option A: Bag‑of‑visual‑words from SIFT descriptors (cluster into 50 clusters using K‑means, then histogram).

Option B: Use PCA‑reduced SIFT descriptors (flatten all descriptors of a frame, reduce to 100 dims).

Label each frame with object class (e.g., “mug” or “toy” – you need at least 2 classes; record two different objects).

Train SVM (sklearn.svm.SVC) with 70% frames, test on 30%.

Generate classification report: precision, recall, f1‑score.

Generate confusion matrix using sklearn.metrics.confusion_matrix and plot with matplotlib.

Lab used: Exp 9 (SVM, KNN).

Step 9: Quantitative Evaluation & Metrics
Mean reprojection error: For each 3D point, project it into all frames where it was observed. Compute Euclidean distance between projected point and original 2D feature. Average over all observations.

Number of 3D points: Total points in final cloud.

Classification accuracy: From Step 8.

Optional – Segmentation IoU: If you have ground truth masks for a few frames, compute Intersection‑over‑Union.

Create a table with these numbers for your PPT.

Step 10: Visualization & Output Generation
Save final point cloud as .ply file (use open3d or simple ASCII format).

Generate a 3D scatter plot using matplotlib:

Rotate the view and save multiple angles as images.

Create a GIF of rotating point cloud using imageio.

Overlay point cloud on one of the original images (project points onto image plane) to show alignment.

Save all output images in output/ folder.

Step 11: Final Script Integration & Documentation
Combine all steps into one master script run_full_pipeline.py with flags to enable/disable optional modules.

Add clear comments and print statements for each major step.

Write a README.md with:

Project description

Setup instructions

How to run

Output description

Ensure the script works from start to end without manual intervention (except bounding box selection for segmentation).

