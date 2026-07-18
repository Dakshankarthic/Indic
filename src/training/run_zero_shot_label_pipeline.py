import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import shutil
import concurrent.futures

pipeline_dir = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.append(str(pipeline_dir))

from dino_layout_step1 import (
    load_dino_model, extract_patch_features, cluster_text_mask, binarize, 
    detect_lines_from_mask, detect_words_and_chars_in_line, get_line_polygon,
    is_printed_page, detect_illustrations_from_binary
)
from opencv_layout_refinement import (
    detect_page_frame, detect_damage_holes, detect_text_regions, classify_marginalia
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_6_channel_mask(img_w, img_h, page_frame, damage_regions, text_regions, marginalia_regions, illustrations, lines_data):
    m0 = np.zeros((img_h, img_w), dtype=np.uint8)
    m1 = np.zeros((img_h, img_w), dtype=np.uint8)
    m2 = np.zeros((img_h, img_w), dtype=np.uint8)
    m3 = np.zeros((img_h, img_w), dtype=np.uint8)
    m4 = np.zeros((img_h, img_w), dtype=np.uint8)
    m5 = np.zeros((img_h, img_w), dtype=np.uint8)

    if page_frame and 'polygon' in page_frame and len(page_frame['polygon']) >= 3:
        pts = np.array(page_frame['polygon'], dtype=np.int32)
        cv2.fillPoly(m3, [pts], 255)
    elif page_frame and 'bbox' in page_frame:
        x1, y1, x2, y2 = page_frame['bbox']
        cv2.rectangle(m3, (x1, y1), (x2, y2), 255, -1)

    for region in text_regions:
        x1, y1, x2, y2 = region['bbox']
        cv2.rectangle(m0, (x1, y1), (x2, y2), 255, -1)

    for region in marginalia_regions:
        x1, y1, x2, y2 = region['bbox']
        cv2.rectangle(m1, (x1, y1), (x2, y2), 255, -1)

    for illus in illustrations:
        if 'polygon' in illus and len(illus['polygon']) >= 3:
            pts = np.array(illus['polygon'], dtype=np.int32)
            cv2.fillPoly(m2, [pts], 255)
        elif 'bbox' in illus:
            x1, y1, x2, y2 = illus['bbox']
            cv2.rectangle(m2, (x1, y1), (x2, y2), 255, -1)

    for hole in damage_regions:
        if 'polygon' in hole and len(hole['polygon']) >= 3:
            pts = np.array(hole['polygon'], dtype=np.int32)
            cv2.fillPoly(m4, [pts], 255)
        elif 'bbox' in hole:
            x1, y1, x2, y2 = hole['bbox']
            cv2.rectangle(m4, (x1, y1), (x2, y2), 255, -1)

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
    if len(lines_data) < 3:
        return False
    text_ratio = np.sum(text_mask) / (text_mask.shape[0] * text_mask.shape[1])
    if text_ratio > 0.5:
        return False
    return True

global_model = None

def process_image_worker(args):
    global global_model
    img_path, mask_out_dir, img_out_dir, viz_dir, save_viz = args
    
    out_mask_path = mask_out_dir / f"{img_path.stem}.npz"
    if out_mask_path.exists():
        return 1

    img = cv2.imread(str(img_path))
    if img is None: return 0
    
    if global_model is None:
        global_model = load_dino_model()
        
    h, w = img.shape[:2]
    is_printed = is_printed_page(img)

    feat_grid, _, _ = extract_patch_features(global_model, img)
    text_mask, _ = cluster_text_mask(feat_grid, img, is_printed)
    binary = binarize(img)
    
    illus_mask, illustrations = detect_illustrations_from_binary(binary, w, h)
    
    mask_full = cv2.resize(text_mask * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    text_only_mask = cv2.bitwise_and(mask_full, cv2.bitwise_not(illus_mask))
    text_mask_binary = (text_only_mask > 0).astype(np.uint8)
    
    lines_data, mask_full, binary_masked = detect_lines_from_mask(text_mask_binary, binary, h, w)
    
    if not check_confidence(lines_data, text_mask):
        return 0
    
    for ld in lines_data:
        lx1, ly1, lx2, ly2 = ld['bbox']
        line_roi = binary_masked[ly1:ly2, lx1:lx2]
        ld['polygon'] = get_line_polygon(line_roi, lx1, ly1)

    page_frame_dict, leaf_mask = detect_page_frame(img)
    damage_regions = detect_damage_holes(img, leaf_mask)
    text_regions_raw = detect_text_regions(binary_masked)
    text_regions, marginalia_regions = classify_marginalia(text_regions_raw, w)

    mask = create_6_channel_mask(
        w, h, page_frame_dict, damage_regions, text_regions, 
        marginalia_regions, illustrations, lines_data
    )
    
    np.savez_compressed(out_mask_path, mask=mask)
    
    out_img_path = img_out_dir / img_path.name
    if not out_img_path.exists():
        shutil.copy2(img_path, out_img_path)
        
    if save_viz:
        overlay = img.copy()
        overlay[text_only_mask > 0] = [0, 0, 255]
        overlay[illus_mask > 0] = [255, 100, 0]
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        for ld in lines_data:
            lx1, ly1, lx2, ly2 = ld['bbox']
            cv2.rectangle(blended, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
        cv2.imwrite(str(viz_dir / img_path.name), blended)
        
    return 1

def main():
    import multiprocessing
    multiprocessing.freeze_support()
    
    base_dir = Path(__file__).resolve().parents[2]
    img_dirs = [
        base_dir / "datasets" / "ramcharitmanas",
        base_dir / "olai_suvadi_images"
    ]
    
    out_dir = base_dir / "training_data_zero_shot"
    img_out_dir = out_dir / "images"
    mask_out_dir = out_dir / "masks"
    viz_dir = base_dir / "dino_viz_review"
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    for d in img_dirs:
        if d.exists():
            all_images.extend(list(d.glob("**/*.jpg")) + list(d.glob("**/*.png")))
    
    print(f"Found {len(all_images)} total images in unlabeled pool.")
    
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    args_list = []
    for i, img_path in enumerate(all_images):
        save_viz = (i < 20)
        args_list.append((img_path, mask_out_dir, img_out_dir, viz_dir, save_viz))

    success_count = 0
    print("Starting sequential processing (this is safer and often faster on a single GPU)...")
    
    for args in tqdm(args_list, desc="Generating Zero-shot labels"):
        success_count += process_image_worker(args)

    print(f"\nZero-shot labeling complete! Successfully generated {success_count} high-confidence training pairs.")
    print(f"Images saved to: {img_out_dir}")
    print(f"Masks saved to: {mask_out_dir}")

if __name__ == "__main__":
    main()
