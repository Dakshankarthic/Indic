"""Full visualization: Lines + Words + Illustrations on a single image.

Character-level boxes are intentionally disabled by default. Devanagari glyphs are
not clean one-box-per-Unicode-character shapes because matras, halants, conjuncts,
and the shirorekha overlap. Word and line boxes are much more reliable for the
challenge video and for human correction in Aletheia.
"""
import cv2
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "src/pipeline")
from dino_layout_step1 import (
    load_dino_model, extract_patch_features, binarize,
    cluster_text_mask, is_printed_page, detect_illustrations_from_binary,
    detect_lines_from_mask, detect_words_and_chars_in_line
)

INPUT_DIR = r"D:\indic_challenge\org_img-20260701T115420Z-3-001\org_img"
OUTPUT_DIR = r"D:\indic_challenge\final_visual"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SHOW_CHARACTER_BOXES = False

def overlap_ratio(line_bbox, illus_bbox):
    lx1, ly1, lx2, ly2 = line_bbox
    ix1, iy1, ix2, iy2 = illus_bbox
    ox1, oy1 = max(lx1, ix1), max(ly1, iy1)
    ox2, oy2 = min(lx2, ix2), min(ly2, iy2)
    if ox1 >= ox2 or oy1 >= oy2: return 0.0
    overlap_area = (ox2 - ox1) * (oy2 - oy1)
    line_area = max(1, (lx2 - lx1) * (ly2 - ly1))
    return overlap_area / line_area

# Get pages 1-40 sorted numerically
all_images = list(Path(INPUT_DIR).glob("*.jpg"))
def page_num(p):
    try: return int(p.stem.split("_")[1])
    except: return 9999
all_images.sort(key=page_num)
images = [p for p in all_images if page_num(p) <= 40]
print(f"Processing {len(images)} images (pages 1-40)")

print("Loading DINO model...")
model = load_dino_model()

for img_path in tqdm(images, desc="Processing"):
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    h, w = img.shape[:2]
    feat_grid, _, _ = extract_patch_features(model, img)
    printed = is_printed_page(img)
    text_mask, _ = cluster_text_mask(feat_grid, img, printed)
    binary = binarize(img)
    
    # Detect illustrations
    illus_mask, illus_regions = detect_illustrations_from_binary(binary, w, h)
    
    # Detect lines
    lines_data, mask_full, binary_masked = detect_lines_from_mask(
        text_mask, binary, h, w, is_printed=printed
    )
    
    # Filter lines overlapping with illustrations
    filtered_lines = []
    for ld in lines_data:
        dominated = False
        for ir in illus_regions:
            if overlap_ratio(ld["bbox"], ir["bbox"]) > 0.3:
                dominated = True
                break
        if not dominated:
            filtered_lines.append(ld)
    
    # --- Build visualization ---
    vis = img.copy()
    
    # 1) Text mask overlay in RED/PINK
    text_overlay = np.zeros_like(vis)
    text_overlay[:, :, 2] = 255
    text_overlay[:, :, 1] = 80
    text_overlay[:, :, 0] = 80
    mask_bool = mask_full > 0
    mask_bool_clean = mask_bool & (illus_mask == 0)
    vis[mask_bool_clean] = cv2.addWeighted(vis, 0.4, text_overlay, 0.6, 0)[mask_bool_clean]
    
    # 2) Illustration overlay in BLUE
    illus_bool = illus_mask > 0
    if np.any(illus_bool):
        illus_overlay = np.zeros_like(vis)
        illus_overlay[:, :, 0] = 255
        illus_overlay[:, :, 1] = 120
        vis[illus_bool] = cv2.addWeighted(vis, 0.4, illus_overlay, 0.6, 0)[illus_bool]
    
    # 3) Illustration bounding boxes in thick CYAN
    for ir in illus_regions:
        ix1, iy1, ix2, iy2 = ir["bbox"]
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (255, 150, 0), 3)
    
    # 4) For each line: detect words. Character boxes are optional/debug only.
    for i, line in enumerate(filtered_lines):
        lx1, ly1, lx2, ly2 = line["bbox"]
        
        # LINE box in GREEN
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (lx1, ly1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Use the proper word/char detection with shirorekha ablation
        words, chars_by_word = detect_words_and_chars_in_line(binary, line)
        
        for wi, word in enumerate(words):
            wx1, wy1, wx2, wy2 = word['bbox']
            # WORD box in BLUE
            cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
            
            if SHOW_CHARACTER_BOXES:
                word_chars = chars_by_word.get(str(wi), [])
                for ch in word_chars:
                    cx1, cy1, cx2, cy2 = ch['bbox']
                    cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)
    
    # Label
    label = "PRINTED" if printed else "HANDWRITTEN"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, "GREEN=Line BLUE=Word CYAN=Illus", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    out_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), vis)

print(f"Done! {len(images)} visualizations saved to {OUTPUT_DIR}")
