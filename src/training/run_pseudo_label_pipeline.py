import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import shutil

# Add src/pipeline to path to import pipeline modules properly
pipeline_dir = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.append(str(pipeline_dir))

from dino_layout_step1 import (
    load_dino_model, extract_patch_features, cluster_text_mask, binarize, 
    detect_lines_from_mask, detect_words_and_chars_in_line, get_line_polygon
)
from opencv_layout_refinement import (
    detect_page_frame, detect_damage_holes, detect_text_regions, classify_marginalia
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_6_channel_mask(img_w, img_h, page_frame, damage_regions, text_regions, marginalia_regions, illustrations, lines_data):
    """
    Creates a 6-channel mask:
    0: text_region
    1: marginalia/notes
    2: illustration/diagram
    3: page_frame
    4: damage/hole
    5: text_line
    """
    m0 = np.zeros((img_h, img_w), dtype=np.uint8)
    m1 = np.zeros((img_h, img_w), dtype=np.uint8)
    m2 = np.zeros((img_h, img_w), dtype=np.uint8)
    m3 = np.zeros((img_h, img_w), dtype=np.uint8)
    m4 = np.zeros((img_h, img_w), dtype=np.uint8)
    m5 = np.zeros((img_h, img_w), dtype=np.uint8)

    # 3. page_frame (Channel 3)
    if page_frame and 'polygon' in page_frame and len(page_frame['polygon']) >= 3:
        pts = np.array(page_frame['polygon'], dtype=np.int32)
        cv2.fillPoly(m3, [pts], 255)
    elif page_frame and 'bbox' in page_frame:
        x1, y1, x2, y2 = page_frame['bbox']
        cv2.rectangle(m3, (x1, y1), (x2, y2), 255, -1)

    # 0. text_region (Channel 0)
    for region in text_regions:
        x1, y1, x2, y2 = region['bbox']
        cv2.rectangle(m0, (x1, y1), (x2, y2), 255, -1)

    # 1. marginalia/notes (Channel 1)
    for region in marginalia_regions:
        x1, y1, x2, y2 = region['bbox']
        cv2.rectangle(m1, (x1, y1), (x2, y2), 255, -1)

    # 2. illustration/diagram (Channel 2)
    for illus in illustrations:
        if 'polygon' in illus and len(illus['polygon']) >= 3:
            pts = np.array(illus['polygon'], dtype=np.int32)
            cv2.fillPoly(m2, [pts], 255)
        elif 'bbox' in illus:
            x1, y1, x2, y2 = illus['bbox']
            cv2.rectangle(m2, (x1, y1), (x2, y2), 255, -1)

    # 4. damage/hole (Channel 4)
    for hole in damage_regions:
        if 'polygon' in hole and len(hole['polygon']) >= 3:
            pts = np.array(hole['polygon'], dtype=np.int32)
            cv2.fillPoly(m4, [pts], 255)
        elif 'bbox' in hole:
            x1, y1, x2, y2 = hole['bbox']
            cv2.rectangle(m4, (x1, y1), (x2, y2), 255, -1)

    # 5. text_line (Channel 5)
    for line in lines_data:
        if 'polygon' in line and len(line['polygon']) >= 3:
            pts = np.array(line['polygon'], dtype=np.int32)
            cv2.fillPoly(m5, [pts], 255)
        else:
            x1, y1, x2, y2 = line['bbox']
            cv2.rectangle(m5, (x1, y1), (x2, y2), 255, -1)

    mask = np.stack([m0, m1, m2, m3, m4, m5], axis=-1)
    return mask

def check_confidence(lines_data, text_mask):
    """
    Confidence Filter:
    - Reject if < 3 text lines (likely blank/cover)
    - Reject if > 50% of page is text mask (clustering failure)
    """
    if len(lines_data) < 3:
        return False
    
    text_ratio = np.sum(text_mask) / (text_mask.shape[0] * text_mask.shape[1])
    if text_ratio > 0.5:
        return False

    return True

def main():
    base_dir = Path(__file__).resolve().parents[2]
    # We will process a subset first for speed if needed, but the script supports all
    img_dirs = [
        base_dir / "olai_suvadi_images",
        base_dir / "ramcharitmanas"
    ]
    
    out_dir = base_dir / "training_data_pseudo"
    img_out_dir = out_dir / "images"
    mask_out_dir = out_dir / "masks"
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    all_images = []
    for d in img_dirs:
        if d.exists():
            all_images.extend(list(d.glob("**/*.jpg")) + list(d.glob("**/*.png")))
    
    print(f"Found {len(all_images)} total images in unlabeled pool.")

    print(f"Loading DINOv2 model on {DEVICE}...")
    model = load_dino_model()

    success_count = 0
    
    # Process 800 images to fit within a 3-hour total pipeline time
    MAX_IMAGES = 800
    np.random.seed(42) # Reproducible subset
    if len(all_images) > MAX_IMAGES:
        all_images = np.random.choice(all_images, MAX_IMAGES, replace=False)

    for img_path in tqdm(all_images, desc="Generating Pseudo-labels"):
        out_mask_path = mask_out_dir / f"{img_path.stem}.npz"
        if out_mask_path.exists():
            success_count += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Downscale large images slightly to speed up pipeline if necessary, but DINO handles it
        h, w = img.shape[:2]

        feat_grid, _, _ = extract_patch_features(model, img)
        text_mask, _ = cluster_text_mask(feat_grid, img)
        binary = binarize(img)
        
        lines_data, mask_full, binary_masked = detect_lines_from_mask(text_mask, binary, h, w)
        
        if not check_confidence(lines_data, text_mask):
            continue
        
        # Add line polygons
        for ld in lines_data:
            lx1, ly1, lx2, ly2 = ld['bbox']
            line_roi = binary_masked[ly1:ly2, lx1:lx2]
            ld['polygon'] = get_line_polygon(line_roi, lx1, ly1)

        # Refinement steps
        page_frame_dict, leaf_mask = detect_page_frame(img)
        damage_regions = detect_damage_holes(img, leaf_mask)
        text_regions_raw = detect_text_regions(binary_masked)
        text_regions, marginalia_regions = classify_marginalia(text_regions_raw, w)
        illustrations = [] # Currently no specific illustration detector

        # Generate 6-channel mask
        mask = create_6_channel_mask(
            w, h, page_frame_dict, damage_regions, text_regions, 
            marginalia_regions, illustrations, lines_data
        )
        
        # Save Mask (.npz)
        out_mask_path = mask_out_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_mask_path, mask=mask)
        
        # Copy Image
        out_img_path = img_out_dir / img_path.name
        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)
            
        success_count += 1

    print(f"\nPseudo-labeling complete! Successfully generated {success_count} high-confidence training pairs.")
    print(f"Images saved to: {img_out_dir}")
    print(f"Masks saved to: {mask_out_dir}")

if __name__ == "__main__":
    main()
