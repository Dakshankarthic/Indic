"""
Prepare higher-resolution page images for OCR.

Option A - Upscale existing 512px test images (quick fallback, limited gain):
  python scripts/prepare_hires_images.py --src org_img-20260701T115420Z-3-001/org_img --scale 3

Option B - After downloading full scans, copy/rename them into org_img_hires/
  Expected names: page_1.jpg, page_2.jpg, ... (same as test set)

Check output dimensions:
  python scripts/prepare_hires_images.py --check-only --dst org_img_hires
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")


def upscale_folder(src: Path, dst: Path, scale: int, limit: int = 0) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("page_*.jpg"))
    if limit > 0:
        files = files[:limit]

    print(f"Upscaling {len(files)} images from {src} -> {dst} ({scale}x)")
    for i, path in enumerate(files, 1):
        out = dst / path.name
        if out.exists() and out.stat().st_size > 0:
            continue
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            upscaled = img.resize((w * scale, h * scale), Image.LANCZOS)
            upscaled.save(out, quality=95)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    sample = dst / files[0].name if files else None
    if sample and sample.exists():
        with Image.open(sample) as im:
            print(f"Sample output size: {im.size}")


def check_folder(dst: Path) -> None:
    files = sorted(dst.glob("page_*.jpg"))
    if not files:
        print(f"No page_*.jpg in {dst}")
        return
    sizes: dict[tuple[int, int], int] = {}
    for p in files:
        with Image.open(p) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
    print(f"{dst}: {len(files)} images")
    for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"  {size[0]}x{size[1]}: {count} files")


def main() -> None:
    base = Path(r"d:\indic_challenge")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=str(base / "org_img-20260701T115420Z-3-001" / "org_img"),
    )
    parser.add_argument("--dst", default=str(base / "org_img_hires"))
    parser.add_argument("--scale", type=int, default=3, help="3 => 512 becomes 1536")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    dst = Path(args.dst)
    if args.check_only:
        check_folder(dst)
        return

    upscale_folder(Path(args.src), dst, args.scale, args.limit)


if __name__ == "__main__":
    main()
