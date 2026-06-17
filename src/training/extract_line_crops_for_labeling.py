"""
Extract Line Crops from DINO Output for Manual Labeling.

This script reads the temp_dino_regions.json from Step 1,
crops each detected text line from the original images, 
preprocesses them, and saves them as individual images.

It also creates a template CSV (labels.csv) where you manually 
type the correct transcription for each line crop.

Usage:
    python extract_line_crops_for_labeling.py --json paddle_results/temp_dino_regions.json --output labeling_data
"""

import os
import sys
import cv2
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def preprocess_crop(crop):
    """Clean up line crop for better readability during labeling."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 8
    )
    
    padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    return padded


def main():
    parser = argparse.ArgumentParser(description="Extract DINO line crops for labeling")
    parser.add_argument("--json", required=True, help="Path to temp_dino_regions.json")
    parser.add_argument("--output", default="labeling_data", help="Output directory")
    parser.add_argument("--max_lines", type=int, default=500, help="Max lines to extract (default 500)")
    parser.add_argument("--min_width", type=int, default=100, help="Skip lines narrower than this")
    parser.add_argument("--min_height", type=int, default=15, help="Skip lines shorter than this")
    args = parser.parse_args()

    out_dir = Path(args.output)
    crops_dir = out_dir / "line_crops"
    crops_raw_dir = out_dir / "line_crops_raw"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops_raw_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, 'r') as f:
        all_data = json.load(f)

    csv_rows = []
    line_count = 0

    print(f"Extracting line crops from {len(all_data)} images...")
    
    for item in tqdm(all_data, desc="Extracting"):
        img_path = item["img_path"]
        img_w = item["img_w"]
        img_h = item["img_h"]

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not read {img_path}")
            continue

        img_stem = Path(img_path).stem

        for li, ld in enumerate(item.get("lines_data", [])):
            if line_count >= args.max_lines:
                break

            lx1, ly1, lx2, ly2 = ld['bbox']
            w = lx2 - lx1
            h = ly2 - ly1

            if w < args.min_width or h < args.min_height:
                continue

            if ld.get('is_marginalia', False):
                continue

            pad = 4
            cy1 = max(0, ly1 - pad)
            cy2 = min(img_h, ly2 + pad)
            cx1 = max(0, lx1 - pad)
            cx2 = min(img_w, lx2 + pad)

            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            fname_raw = f"{img_stem}_line{li:03d}.jpg"
            cv2.imwrite(str(crops_raw_dir / fname_raw), crop)

            processed = preprocess_crop(crop)
            fname = f"{img_stem}_line{li:03d}.jpg"
            cv2.imwrite(str(crops_dir / fname), processed)

            csv_rows.append({
                "file_name": fname,
                "text": "",  # <-- YOU FILL THIS IN
                "source_image": Path(img_path).name,
                "line_index": li,
                "bbox": f"{lx1},{ly1},{lx2},{ly2}"
            })
            line_count += 1

        if line_count >= args.max_lines:
            break

    csv_path = out_dir / "labels.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text", "source_image", "line_index", "bbox"])
        writer.writeheader()
        writer.writerows(csv_rows)

    gt_path = out_dir / "rec_gt_train.txt"
    with open(gt_path, 'w', encoding='utf-8') as f:
        for row in csv_rows:
            f.write(f"line_crops/{row['file_name']}\t{row['text']}\n")

    print(f"\n{'='*60}")
    print(f"DONE! Extracted {line_count} line crops.")
    print(f"{'='*60}")
    print(f"\n  Clean crops:     {crops_dir}")
    print(f"  Raw crops:       {crops_raw_dir}")
    print(f"  Label CSV:       {csv_path}")
    print(f"  PaddleOCR GT:    {gt_path}")
    print(f"\n  NEXT STEP:")
    print(f"  1. Open {csv_path} in Excel/editor")
    print(f"  2. Look at each image in {crops_dir}")
    print(f"  3. Type the correct Devanagari text in the 'text' column")
    print(f"  4. Run: python prepare_paddleocr_finetuning.py --labeled_csv {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
