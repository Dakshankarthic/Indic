import os
import sys
import cv2
import torch
import numpy as np
import pytesseract
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

# Add paths
sys.path.append(r"D:\indic_challenge")
sys.path.append(r"D:\indic_challenge\src\pipeline")

from src.pipeline.dino_layout_step1 import (
    load_dino_model, extract_patch_features, is_printed_page, 
    cluster_text_mask, binarize, detect_lines_from_mask,
    detect_words_and_chars_in_line, detect_page_frame,
    detect_damage_holes, detect_text_regions, classify_marginalia
)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_DIR = r'D:\indic_challenge\tessdata'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Initializing DINOv2 Foundation Model on {DEVICE} for Live Pitch Evaluation...")
dino_model = load_dino_model()

def process_single_image(image_path, output_dir=r"D:\indic_challenge\pitch_output"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    img_name = Path(image_path).stem
    H, W = img.shape[:2]
    print(f"\n=======================================================")
    print(f"[*] LIVE INFERENCE ON MANUSCRIPT: {img_name} ({W}x{H})")
    print(f"=======================================================")

    # 1. DINO Layout & Geometry Feature Extraction
    feat_grid, _, _ = extract_patch_features(dino_model, img)
    printed = is_printed_page(img)
    print(f"[*] Stage 1 Substrate Classification: {'Printed Paper' if printed else 'Palm Leaf / Ancient Manuscript'}")

    text_mask, _ = cluster_text_mask(feat_grid, img, printed)
    binary = binarize(img)
    lines_data, mask_full, binary_masked = detect_lines_from_mask(text_mask, binary, H, W, is_printed=printed)
    
    # 2. Page Frame, Damage Holes, Text Regions
    page_frame_dict, leaf_mask = detect_page_frame(img)
    damage_regions = detect_damage_holes(img, leaf_mask)
    text_regions_raw = detect_text_regions(binary_masked)
    text_regions, marginalia_regions = classify_marginalia(text_regions_raw, W)
    
    print(f"[*] Stage 1 Detection: {len(lines_data)} TextLines, {len(text_regions)} TextRegions, {len(damage_regions)} Damage Holes")

    # 3. Stage 2 & 3: Line-level OCR, Words, Akshara Extraction
    img_vis = img.copy()
    all_extracted_text = []
    total_words = 0
    total_chars = 0

    # Draw Damage Holes (Yellow)
    for dm in damage_regions:
        pts = np.array(dm.get('polygon', []), dtype=np.int32)
        if len(pts) > 2:
            cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 200, 220), thickness=2)

    # Process Lines
    for ld in lines_data:
        words, chars_by_word = detect_words_and_chars_in_line(binary_masked, ld)
        ld['words'] = words
        ld['chars'] = chars_by_word
        
        # Line bbox
        poly = np.array(ld['polygon'], dtype=np.int32)
        lx, ly, lw, lh = cv2.boundingRect(poly)
        line_crop = img[max(0, ly):min(H, ly+lh), max(0, lx):min(W, lx+lw)]
        
        # OCR Line
        try:
            line_text = pytesseract.image_to_string(line_crop, config=f'--tessdata-dir {TESSDATA_DIR} -l san --psm 7').strip()
        except Exception:
            line_text = ""
            
        if line_text:
            all_extracted_text.append(line_text)
            
        # Draw Line (Green)
        if len(poly) > 2:
            cv2.polylines(img_vis, [poly], isClosed=True, color=(0, 200, 0), thickness=2)
            
        # Draw Words (Blue) & Aksharas (Red)
        for w in words:
            total_words += 1
            w_poly = np.array(w['polygon'], dtype=np.int32)
            if len(w_poly) > 2:
                cv2.polylines(img_vis, [w_poly], isClosed=True, color=(255, 120, 0), thickness=1)
                
        for clist in chars_by_word:
            for c in clist:
                total_chars += 1
                c_poly = np.array(c['polygon'], dtype=np.int32)
                if len(c_poly) > 2:
                    cv2.polylines(img_vis, [c_poly], isClosed=True, color=(0, 0, 220), thickness=1)

    # 4. Save Visual Result with Legend
    out_visual_path = os.path.join(output_dir, f"{img_name}_annotated_output.jpg")
    cv2.imwrite(out_visual_path, img_vis)

    # 5. Build Standard PRImA PAGE-XML 2013
    root = ET.Element("PcGts", xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15")
    page_el = ET.SubElement(root, "Page", imageFilename=Path(image_path).name, imageWidth=str(W), imageHeight=str(H))
    
    for r_idx, reg in enumerate(text_regions):
        reg_el = ET.SubElement(page_el, "TextRegion", id=f"r_{r_idx}", type="paragraph")
        pts_str = " ".join([f"{pt[0]},{pt[1]}" for pt in reg.get("polygon", [[0,0],[W,0],[W,H],[0,H]])])
        ET.SubElement(reg_el, "Coords", points=pts_str)
        
    for l_idx, ld in enumerate(lines_data):
        line_el = ET.SubElement(page_el, "TextLine", id=f"l_{l_idx}")
        pts_str = " ".join([f"{pt[0]},{pt[1]}" for pt in ld.get("polygon", [])])
        ET.SubElement(line_el, "Coords", points=pts_str)
        if l_idx < len(all_extracted_text):
            text_equiv = ET.SubElement(line_el, "TextEquiv")
            unicode_el = ET.SubElement(text_equiv, "Unicode")
            unicode_el.text = all_extracted_text[l_idx]

    out_xml_path = os.path.join(output_dir, f"{img_name}_page.xml")
    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    with open(out_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"\n[+] LIVE INFERENCE COMPLETE:")
    print(f"    - Output Visual Image: {out_visual_path}")
    print(f"    - Output PAGE-XML:     {out_xml_path}")
    print(f"    - TextLines Extracted: {len(lines_data)}")
    print(f"    - Words Segmented:     {total_words}")
    print(f"    - Glyphs (Aksharas):   {total_chars}")
    print(f"    - Damage Holes Found:  {len(damage_regions)}")
    print(f"    - Extracted Text Sample:")
    for l in all_extracted_text[:4]:
        print(f"      > {l}")
    print(f"=======================================================\n")
    return out_visual_path, out_xml_path, len(lines_data), total_words, total_chars

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_single_image(sys.argv[1])
    else:
        test_img = r"D:\indic_challenge\beamer\page_1_input.jpg"
        if os.path.exists(test_img):
            process_single_image(test_img)
