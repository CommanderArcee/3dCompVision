from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, step: int, width: int, height: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sampled frames from a video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    n = extract_frames(args.video, args.out, args.step, args.width, args.height)
    print(f"Saved {n} frames to {args.out}")


if __name__ == "__main__":
    main()
