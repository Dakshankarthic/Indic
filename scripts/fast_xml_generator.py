"""
Fast XML Generator (Post-Surya Cache)
Reads bounding boxes via DINO and text from the pre-computed Surya Cache.
Generates full PAGE-XML hierarchy (Lines, Words, Characters) instantly.

Usage:
    python scripts/fast_xml_generator.py
"""
import sys
import os
import cv2
import json
import time
import numpy as np
import pytesseract
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

# Paths
ROOT = Path(r"d:\indic_challenge")
INPUT_DIR = ROOT / "org_img-20260701T115420Z-3-001" / "org_img"
OUTPUT_XML_DIR = ROOT / "final_competition_xmls"
OUTPUT_VISUAL_DIR = ROOT / "final_results"
TESSDATA_DIR = ROOT / "tessdata"
CACHE_FILE = ROOT / "surya_text_cache.json"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.makedirs(OUTPUT_XML_DIR, exist_ok=True)
os.makedirs(OUTPUT_VISUAL_DIR, exist_ok=True)

# ── Load custom pipeline modules ──────────────────────────────
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "pipeline"))
from src.pipeline.dino_layout_step1 import (
    load_dino_model, extract_patch_features, is_printed_page,
    cluster_text_mask, binarize as dino_binarize,
    detect_illustrations_from_binary, detect_lines_from_mask,
    detect_lines_in_region
)
from surya_xml_pipeline import (
    make_akshara_boxes, score_devanagari_ocr, binarize, create_xml
)

def main():
    print(f"Reading cached text from {CACHE_FILE}")
    if not CACHE_FILE.exists():
        print("ERROR: Cache file not found! Wait for OCR to finish.")
        return
        
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        surya_cache = json.load(f)

    print("Loading DINOv2 model for fast bounding box generation...")
    dino_model = load_dino_model()

    image_files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    print(f"\nProcessing {len(image_files)} pages instantly using cached text...")
    start_time = time.time()

    for page_idx, filename in enumerate(image_files):
        stem = filename.split('.')[0]
        if stem not in surya_cache:
            continue
            
        print(f"[{page_idx+1}/{len(image_files)}] Generating XML for {filename}...")
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        objects = []
        dino_illus = []
        dino_lines = []

        # ── 1. DINO Layout Detection ─────────────────────────────
        try:
            feat_grid, _, _ = extract_patch_features(dino_model, img)
            printed = is_printed_page(img)
            text_mask, _ = cluster_text_mask(feat_grid, img, printed)
            binary = dino_binarize(img)
            illus_mask, illus_regions = detect_illustrations_from_binary(binary, w, h)
            lines_data, mask_full, binary_masked = detect_lines_from_mask(
                text_mask, binary, h, w, is_printed=printed
            )

            for illus in illus_regions:
                ix1, iy1, ix2, iy2 = illus['bbox']
                objects.append({"name": "illustration", "xmin": ix1, "ymin": iy1, "xmax": ix2, "ymax": iy2})
                dino_illus.append((ix1, iy1, ix2, iy2))

            for ld in lines_data:
                lx1, ly1, lx2, ly2 = ld['bbox']
                line_area = (lx2 - lx1) * (ly2 - ly1)
                is_inside_illus = False
                for (ix1, iy1, ix2, iy2) in dino_illus:
                    inter_x1, inter_y1 = max(lx1, ix1), max(ly1, iy1)
                    inter_x2, inter_y2 = min(lx2, ix2), min(ly2, iy2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        if (inter_x2 - inter_x1) * (inter_y2 - inter_y1) > 0.5 * line_area:
                            is_inside_illus = True
                            break
                if not is_inside_illus:
                    objects.append({"name": "text_line", "xmin": lx1, "ymin": ly1, "xmax": lx2, "ymax": ly2})
                    dino_lines.append((lx1, ly1, lx2, ly2))
        except Exception as e:
            print(f"  DINO failed: {e}")

        # ── 1b. FALLBACK ─────────────────────────
        expected_lines = max(5, h // 40)
        avg_line_width = sum(x2 - x1 for x1, _, x2, _ in dino_lines) / len(dino_lines) if dino_lines else 0
        lines_are_fragments = avg_line_width < w * 0.4 and len(dino_lines) > 0
        too_few_lines = len(dino_lines) < expected_lines * 0.5

        if too_few_lines or lines_are_fragments:
            if lines_are_fragments:
                objects = [o for o in objects if o["name"] != "text_line"]
                dino_lines.clear()
            fallback_binary = binarize(img)
            fallback_lines = detect_lines_in_region(fallback_binary, 0, 0, w, h, w, h, 0)
            for ld in fallback_lines:
                lx1, ly1, lx2, ly2 = ld['bbox']
                if not any(abs(ly1 - dl[1]) < 10 for dl in dino_lines):
                    objects.append({"name": "text_line", "xmin": lx1, "ymin": ly1, "xmax": lx2, "ymax": ly2})
                    dino_lines.append((lx1, ly1, lx2, ly2))

        # ── 2. Get Text From Cache ────────────────────────────
        surya_text = surya_cache[stem]
        surya_page_lines = [l.strip() for l in surya_text.split('\n') if l.strip()]

        # ── 3. Assign Surya text to detected lines ───────────────
        line_indices = list(range(len(dino_lines)))
        line_indices.sort(key=lambda i: dino_lines[i][1])

        for rank, li in enumerate(line_indices):
            count = 0
            for obj in objects:
                if obj["name"] == "text_line":
                    if count == li:
                        if rank < len(surya_page_lines):
                            obj["text"] = surya_page_lines[rank]
                        break
                    count += 1

        # ── 4. Tesseract for word bounding boxes ─────────────────
        img_for_tess = img.copy()
        for (ix1, iy1, ix2, iy2) in dino_illus:
            cv2.rectangle(img_for_tess, (ix1, iy1), (ix2, iy2), (255, 255, 255), -1)

        word_boxes = []
        word_items = []
        char_boxes = []
        line_idx = 0

        for obj in objects:
            if obj["name"] != "text_line": continue
            lx1, ly1, lx2, ly2 = dino_lines[line_idx]
            line_crop = img_for_tess[ly1:ly2, lx1:lx2]
            try:
                config = f'--tessdata-dir {TESSDATA_DIR} -l san --psm 7'
                line_data = pytesseract.image_to_data(line_crop, config=config, output_type=pytesseract.Output.DICT)

                surya_line_text = obj.get("text", "")
                surya_words = surya_line_text.split() if surya_line_text else []
                surya_word_idx = 0

                for i in range(len(line_data["level"])):
                    if line_data["level"][i] != 5: continue
                    tess_word = line_data["text"][i].strip()
                    if not tess_word: continue

                    ww, wh = int(line_data["width"][i]), int(line_data["height"][i])
                    if ww <= 1 or wh <= 1: continue

                    wx, wy = lx1 + int(line_data["left"][i]), ly1 + int(line_data["top"][i])
                    word_box = (wx, wy, wx + ww, wy + wh)

                    word_text = surya_words[surya_word_idx] if surya_word_idx < len(surya_words) else tess_word
                    surya_word_idx += 1

                    objects.append({"name": "word", "text": word_text, "xmin": word_box[0], "ymin": word_box[1], "xmax": word_box[2], "ymax": word_box[3]})
                    word_boxes.append(word_box)
                    word_items.append({"text": word_text, "bbox": word_box})
            except Exception:
                pass
            line_idx += 1

        # ── 5. Character/Glyph boxes from ink projection ─────────
        try:
            for word in word_items:
                wx1, wy1, wx2, wy2 = word["bbox"]
                crop = img_for_tess[wy1:wy2, wx1:wx2]
                if crop.size == 0: continue
                for char_text, cx1, cy1, cx2, cy2 in make_akshara_boxes(crop, word["text"], word["bbox"]):
                    objects.append({"name": "character", "text": char_text, "xmin": cx1, "ymin": cy1, "xmax": cx2, "ymax": cy2})
                    char_boxes.append((cx1, cy1, cx2, cy2))
        except Exception:
            pass

        # ── 6. Generate PAGE-XML ─────────────────────────────────
        create_xml(filename, img.shape, objects)

        # ── 7. Draw visual overlay ───────────────────────────────
        img_vis = img.copy()
        for (cx1, cy1, cx2, cy2) in char_boxes: cv2.rectangle(img_vis, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)
        for (wx1, wy1, wx2, wy2) in word_boxes: cv2.rectangle(img_vis, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        for (lx1, ly1, lx2, ly2) in dino_lines: cv2.rectangle(img_vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
        for (ix1, iy1, ix2, iy2) in dino_illus: cv2.rectangle(img_vis, (ix1, iy1), (ix2, iy2), (255, 255, 0), 3)

        cv2.imwrite(os.path.join(OUTPUT_VISUAL_DIR, filename), img_vis)

        rate = (page_idx + 1) / (time.time() - start_time)
        eta = (len(image_files) - page_idx - 1) / rate / 60
        print(f"  Rate={rate:.2f}pg/s ETA={eta:.1f}min")

    print("\nCOMPLETE! All XMLs and Visuals Generated!")

if __name__ == "__main__":
    main()
