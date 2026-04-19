from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .features import create_sift
from .io_utils import list_images, load_gray


@dataclass
class ClassificationResult:
    report_text: str
    accuracy: float
    confusion_matrix_path: Path


def _collect_images_by_class(root: Path) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img in list_images(class_dir):
            image_paths.append(img)
            labels.append(class_dir.name)
    return image_paths, labels


def _build_bovw_descriptors(image_paths: list[Path], k: int, random_seed: int) -> np.ndarray:
    sift = create_sift()
    all_desc: list[np.ndarray] = []
    per_img_desc: list[np.ndarray] = []

    for p in image_paths:
        gray = load_gray(p)
        _, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            desc = np.zeros((1, 128), dtype=np.float32)
        all_desc.append(desc)
        per_img_desc.append(desc)

    stacked = np.vstack(all_desc).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    kmeans.fit(stacked)

    hists = np.zeros((len(image_paths), k), dtype=np.float32)
    for i, desc in enumerate(per_img_desc):
        words = kmeans.predict(desc.astype(np.float32))
        for w in words:
            hists[i, w] += 1.0
        hists[i] /= max(1.0, hists[i].sum())

    return hists


def run_bovw_svm(
    dataset_root: Path,
    output_dir: Path,
    n_clusters: int = 50,
    test_size: float = 0.3,
    random_seed: int = 42,
) -> ClassificationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths, labels = _collect_images_by_class(dataset_root)
    if len(set(labels)) < 2:
        raise RuntimeError("Need at least two classes for SVM classification")

    X = _build_bovw_descriptors(image_paths, n_clusters, random_seed)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=random_seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    accuracy = float(np.mean(y_test == y_pred))

    cm_path = output_dir / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    disp.figure_.tight_layout()
    disp.figure_.savefig(cm_path, dpi=200)
    plt.close(disp.figure_)

    return ClassificationResult(report_text=report, accuracy=accuracy, confusion_matrix_path=cm_path)
