from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests


COIL100_URL = "https://www.cs.columbia.edu/CAVE/databases/pub/coil-100/coil-100.zip"

# Public Objectron video URL patterns vary by mirror. This script tries known candidates.
OBJECTRON_VIDEO_CANDIDATES = [
    "https://storage.googleapis.com/objectron/videos/train/chair/batch-1/0.mp4",
    "https://storage.googleapis.com/objectron/videos/train/cup/batch-1/0.mp4",
    "https://storage.googleapis.com/objectron/videos/train/camera/batch-1/0.mp4",
]


def download_file(url: str, out_path: Path, timeout: int = 60) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except requests.RequestException:
        return False


def download_coil100(data_root: Path) -> Path:
    zip_path = data_root / "raw" / "coil-100.zip"
    ok = download_file(COIL100_URL, zip_path)
    if not ok:
        raise RuntimeError(f"Failed to download COIL-100 from {COIL100_URL}")

    extract_dir = data_root / "raw" / "coil-100"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def prepare_coil_classification(
    coil_root: Path,
    out_dir: Path,
    class_ids: list[int],
    max_imgs_per_class: int = 72,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid in class_ids:
        class_name = f"obj{cid}"
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        pattern = f"obj{cid}__*.png"
        files = sorted((coil_root / "coil-100").glob(pattern))
        for p in files[:max_imgs_per_class]:
            shutil.copy2(p, class_dir / p.name)


def download_objectron_video(data_root: Path) -> Path:
    out = data_root / "raw" / "objectron_sample.mp4"
    for url in OBJECTRON_VIDEO_CANDIDATES:
        if download_file(url, out):
            return out
    raise RuntimeError("Failed to download Objectron sample video from known URLs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--coil-classes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="COIL object IDs to include for classification",
    )
    args = parser.parse_args()

    data_root = args.data_root

    print("Downloading COIL-100...")
    coil_raw = download_coil100(data_root)
    cls_out = data_root / "interim" / "classification"
    prepare_coil_classification(coil_raw, cls_out, args.coil_classes)
    print(f"Prepared COIL-100 classification subset at: {cls_out}")

    print("Downloading Objectron sample video...")
    video_path = download_objectron_video(data_root)
    print(f"Saved Objectron sample at: {video_path}")


if __name__ == "__main__":
    main()
