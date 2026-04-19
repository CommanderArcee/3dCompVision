from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _angle_key(path: Path) -> int:
    m = re.search(r"__(\d+)\.png$", path.name)
    if not m:
        return 0
    return int(m.group(1))


def prepare_sequence(class_dir: Path, out_dir: Path, step: int = 5) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(class_dir.glob("*.png"), key=_angle_key)
    selected = imgs[::step]
    for i, p in enumerate(selected):
        shutil.copy2(p, out_dir / f"frame_{i:04d}.png")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ordered frame sequence from COIL class")
    parser.add_argument("--class-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    n = prepare_sequence(args.class_dir, args.out, args.step)
    print(f"Prepared {n} ordered frames at {args.out}")


if __name__ == "__main__":
    main()
