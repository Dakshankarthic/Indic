"""
Surya OCR → PAGE-XML Pipeline
Combines DINO layout detection with Surya OCR for text recognition.
Outputs PAGE-XML with full hierarchy (TextRegion → TextLine → Word → Glyph)
and visual overlays with all 4 layers.

Usage:
    python scripts/surya_xml_pipeline.py
    python scripts/surya_xml_pipeline.py --pages 40
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import torch
import numpy as np
import pytesseract
import re
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image

# Force llama.cpp backend before importing Surya
os.environ['SURYA_INFERENCE_BACKEND'] = 'llamacpp'
os.environ['VLLM_GPU_TYPE'] = '2070'
os.environ['LLAMA_CPP_BINARY'] = r'd:\indic_challenge\llama_bin\llama-server.exe'

# Paths
INPUT_DIR = r"D:\indic_challenge\org_img-20260701T115420Z-3-001\org_img"
OUTPUT_XML_DIR = r"D:\indic_challenge\surya_competition_xmls"
OUTPUT_VISUAL_DIR = r"D:\indic_challenge\surya_visual"
TESSDATA_DIR = r"D:\indic_challenge\tessdata"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.makedirs(OUTPUT_XML_DIR, exist_ok=True)
os.makedirs(OUTPUT_VISUAL_DIR, exist_ok=True)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

# ── Devanagari character splitting (from pipeline_master) ──────────────
DEVANAGARI_MARKS = set(chr(c) for c in range(0x093A, 0x094F + 1))
DEVANAGARI_MARKS.update(chr(c) for c in range(0x0951, 0x0957 + 1))
DEVANAGARI_MARKS.update(chr(c) for c in range(0x0962, 0x0963 + 1))
DEVANAGARI_VIRAMA = "\u094d"
ZERO_WIDTH_MARKS = {"\u200c", "\u200d", "\ufeff"}


def split_devanagari_aksharas(text):
    """Group Devanagari text into visual akshara-like units."""
    units = []
    current = ""
    join_next = False
    for ch in text:
        if ch in ZERO_WIDTH_MARKS:
            if current:
                current += ch
            continue
        if ch.isspace():
            if current:
                units.append(current)
                current = ""
            join_next = False
            continue
        if not current:
            current = ch
            join_next = ch == DEVANAGARI_VIRAMA
            continue
        if ch in DEVANAGARI_MARKS:
            current += ch
            join_next = ch == DEVANAGARI_VIRAMA
            continue
        if join_next:
            current += ch
            join_next = False
            continue
        units.append(current)
        current = ch
        join_next = ch == DEVANAGARI_VIRAMA
    if current:
        units.append(current)
    merged = []
    for unit in units:
        has_base = any(0x0900 <= ord(ch) <= 0x097F and ch not in DEVANAGARI_MARKS for ch in unit)
        if merged and not has_base:
            merged[-1] += unit
        else:
            merged.append(unit)
    return merged


def coarsen_units_to_width(units, available_width, min_unit_width):
    if not units:
        return []
    max_units = max(1, int(available_width // max(1, min_unit_width)))
    if len(units) <= max_units:
        return units
    chunks = []
    for group in np.array_split(np.array(units, dtype=object), max_units):
        chunks.append("".join(group.tolist()))
    return chunks


def make_akshara_boxes(word_img, word_text, word_box):
    """Create Devanagari akshara/glyph boxes."""
    wx1, wy1, wx2, wy2 = word_box
    units = split_devanagari_aksharas(word_text)
    if not units:
        return []
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), iterations=1)
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return []
    col_ink = np.sum(binary > 0, axis=0).astype(np.float32)
    ink_cols = np.where(col_ink > 0)[0]
    if len(ink_cols) == 0:
        step = max(1, w / len(units))
        return [(unit, int(wx1 + i * step), wy1, int(wx1 + (i + 1) * step), wy2)
                for i, unit in enumerate(units)]
    left, right = int(ink_cols[0]), int(ink_cols[-1]) + 1
    available_width = right - left
    min_width = max(4, int(h * 0.35), int(w * 0.035))
    units = coarsen_units_to_width(units, available_width, min_width)
    split_points = np.linspace(left, right, len(units) + 1).astype(int)
    smooth = cv2.GaussianBlur(col_ink.reshape(1, -1), (5, 1), 0).flatten()
    refined = [int(split_points[0])]
    for boundary in split_points[1:-1]:
        radius = max(2, min_width // 2)
        lo = max(refined[-1] + min_width, int(boundary) - radius)
        hi = min(right - min_width, int(boundary) + radius)
        if hi > lo:
            local = smooth[lo:hi + 1]
            refined.append(lo + int(np.argmin(local)))
        else:
            refined.append(int(boundary))
    refined.append(int(split_points[-1]))
    split_points = np.array(refined, dtype=int)
    boxes = []
    for i, unit in enumerate(units):
        x1, x2 = int(split_points[i]), int(split_points[i + 1])
        if x2 - x1 < 2:
            continue
        pad_x = 1
        lx1, lx2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
        segment = binary[:, lx1:lx2]
        if np.count_nonzero(segment) < max(2, h // 4):
            continue
        rows = np.where(np.sum(segment > 0, axis=1) > 0)[0]
        if len(rows) > 0:
            ly1 = max(0, int(rows[0]) - 1)
            ly2 = min(h, int(rows[-1]) + 2)
        else:
            continue
        boxes.append((unit, wx1 + lx1, wy1 + ly1, wx1 + lx2, wy1 + ly2))
    return boxes


def extract_text_from_result(result):
    """Extract plain text from Surya OCR result."""
    lines = []
    if hasattr(result, 'blocks'):
        for block in result.blocks:
            if hasattr(block, 'html'):
                text = re.sub(r'<[^>]+>', '', block.html).strip()
                if text:
                    lines.append(text)
            elif hasattr(block, 'lines'):
                for line in block.lines:
                    if line.text.strip():
                        lines.append(line.text.strip())
    return '\n'.join(lines)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[«»""]', '', text)
    return text


def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def preprocess_for_surya(pil_img):
    """Enhanced preprocessing for Surya OCR."""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    upscaled = cv2.resize(thresh, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))


def score_devanagari_ocr(text):
    if not text:
        return -1_000_000
    allowed_punctuation = set(" ।॥.,;:-–—()[]{}*+\"'‌‍")
    score = 0
    for ch in text:
        code = ord(ch)
        if ch.isspace():
            score += 1
        elif 0x0900 <= code <= 0x097F:
            score += 5
        elif ch in allowed_punctuation:
            score += 1
        else:
            score -= 12
    for noisy in ["_", "|", "\\", "/", "॑", "॒", "ऽऽ"]:
        score -= text.count(noisy) * 8
    score += min(len(text.split()), 12) * 2
    return score


def create_xml(filename, img_shape, objects):
    """Generate PAGE-XML with full hierarchy."""
    h, w = img_shape[:2]
    d = img_shape[2] if len(img_shape) > 2 else 3

    ns_page = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
    schema_loc = ("http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
                  "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd")

    pcgts = ET.Element("PcGts", {
        "xmlns": ns_page, "xmlns:xsi": ns_xsi, "xsi:schemaLocation": schema_loc
    })
    metadata = ET.SubElement(pcgts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Surya OCR + DINO Pipeline"
    ET.SubElement(metadata, "Created").text = "2026-07-07T12:00:00"
    ET.SubElement(metadata, "LastChange").text = "2026-07-07T12:00:00"

    page = ET.SubElement(pcgts, "Page", {
        "imageFilename": filename, "imageWidth": str(w), "imageHeight": str(h)
    })

    def clean_box(obj):
        x1 = max(0, min(int(obj["xmin"]), w - 1))
        x2 = max(0, min(int(obj["xmax"]), w - 1))
        y1 = max(0, min(int(obj["ymin"]), h - 1))
        y2 = max(0, min(int(obj["ymax"]), h - 1))
        return x1, y1, x2, y2

    def points_str(box):
        x1, y1, x2, y2 = box
        return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"

    def center_in(inner, outer):
        ix1, iy1, ix2, iy2 = inner
        ox1, oy1, ox2, oy2 = outer
        cx, cy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
        return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

    def add_text_equiv(parent, text):
        if text:
            te = ET.SubElement(parent, "TextEquiv")
            ET.SubElement(te, "Unicode").text = text

    line_objs = [o for o in objects if o["name"] == "text_line"]
    word_objs = [o for o in objects if o["name"] == "word"]
    char_objs = [o for o in objects if o["name"] == "character"]

    for li, line in enumerate(line_objs):
        line_box = clean_box(line)
        region = ET.SubElement(page, "TextRegion", {"id": f"text_region_{li:04d}"})
        ET.SubElement(region, "Coords", {"points": points_str(line_box)})
        text_line = ET.SubElement(region, "TextLine", {"id": f"line_{li:04d}"})
        ET.SubElement(text_line, "Coords", {"points": points_str(line_box)})

        line_words = sorted(
            [w for w in word_objs if center_in(clean_box(w), line_box)],
            key=lambda w: clean_box(w)[0]
        )
        for wi, word in enumerate(line_words):
            word_box = clean_box(word)
            word_el = ET.SubElement(text_line, "Word", {"id": f"word_{li:04d}_{wi:04d}"})
            ET.SubElement(word_el, "Coords", {"points": points_str(word_box)})

            word_chars = sorted(
                [c for c in char_objs if center_in(clean_box(c), word_box)],
                key=lambda c: clean_box(c)[0]
            )
            for ci, char in enumerate(word_chars):
                char_box = clean_box(char)
                glyph = ET.SubElement(word_el, "Glyph",
                                     {"id": f"glyph_{li:04d}_{wi:04d}_{ci:04d}"})
                ET.SubElement(glyph, "Coords", {"points": points_str(char_box)})
                add_text_equiv(glyph, char.get("text", ""))

            add_text_equiv(word_el, word.get("text", ""))

        add_text_equiv(text_line, line.get("text", ""))

    region_idx = 0
    for obj in objects:
        if obj["name"] == "illustration":
            region_idx += 1
            region = ET.SubElement(page, "GraphicRegion",
                                   {"id": f"graphic_region_{region_idx:04d}"})
            ET.SubElement(region, "Coords", {"points": points_str(clean_box(obj))})

    xml_str = minidom.parseString(ET.tostring(pcgts)).toprettyxml(indent="  ")
    xml_str = os.linesep.join([s for s in xml_str.splitlines() if s.strip()])

    out_path = os.path.join(OUTPUT_XML_DIR, filename.replace('.jpg', '.xml'))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Surya OCR → PAGE-XML Pipeline")
    parser.add_argument("--pages", type=int, default=40, help="Number of pages to process")
    args = parser.parse_args()

    PAGE_LIMIT = args.pages

    # ── Load models ──────────────────────────────────────────────
    sys.path.append(r"D:\indic_challenge")
    sys.path.append(r"D:\indic_challenge\src\pipeline")
    from src.pipeline.dino_layout_step1 import (
        load_dino_model, extract_patch_features, is_printed_page,
        cluster_text_mask, binarize as dino_binarize,
        detect_illustrations_from_binary, detect_lines_from_mask,
        detect_lines_in_region
    )

    print("Loading DINOv2 model...")
    dino_model = load_dino_model()

    print("Loading Surya OCR model...")
    from surya.recognition import RecognitionPredictor
    rec_predictor = RecognitionPredictor()

    OCR_LANG_CANDIDATES = ["san", "hin+san", "san+hin"]

    # ── Get image list ───────────────────────────────────────────
    image_files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )[:PAGE_LIMIT]

    print(f"\nProcessing {len(image_files)} pages with Surya OCR + DINO layout")
    print(f"Output XML: {OUTPUT_XML_DIR}")
    print(f"Output Visual: {OUTPUT_VISUAL_DIR}\n")

    start_time = time.time()

    for page_idx, filename in enumerate(image_files):
        img_path = os.path.join(INPUT_DIR, filename)
        print(f"[{page_idx+1}/{len(image_files)}] Processing {filename}...")

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
                objects.append({"name": "illustration",
                                "xmin": ix1, "ymin": iy1, "xmax": ix2, "ymax": iy2})
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
                    objects.append({"name": "text_line",
                                    "xmin": lx1, "ymin": ly1, "xmax": lx2, "ymax": ly2})
                    dino_lines.append((lx1, ly1, lx2, ly2))
        except Exception as e:
            print(f"  DINO failed: {e}")

        # ── 1b. FALLBACK: Fragment / too-few-lines check ─────────
        expected_lines = max(5, h // 40)
        avg_line_width = 0
        if dino_lines:
            avg_line_width = sum(x2 - x1 for x1, _, x2, _ in dino_lines) / len(dino_lines)
        lines_are_fragments = avg_line_width < w * 0.4 and len(dino_lines) > 0
        too_few_lines = len(dino_lines) < expected_lines * 0.5

        if too_few_lines or lines_are_fragments:
            reason = "fragmented" if lines_are_fragments else "too few"
            print(f"  WARNING: {reason} lines ({len(dino_lines)}, "
                  f"avg width {avg_line_width:.0f}/{w}px). Using projection fallback.")
            if lines_are_fragments:
                objects = [o for o in objects if o["name"] != "text_line"]
                dino_lines.clear()
            fallback_binary = binarize(img)
            fallback_lines = detect_lines_in_region(fallback_binary, 0, 0, w, h, w, h, 0)
            for ld in fallback_lines:
                lx1, ly1, lx2, ly2 = ld['bbox']
                if not any(abs(ly1 - dl[1]) < 10 for dl in dino_lines):
                    objects.append({"name": "text_line",
                                    "xmin": lx1, "ymin": ly1, "xmax": lx2, "ymax": ly2})
                    dino_lines.append((lx1, ly1, lx2, ly2))
            print(f"  After fallback: {len(dino_lines)} lines")

        # ── 2. Surya OCR on full page ────────────────────────────
        pil_img = Image.open(img_path)
        pil_preprocessed = preprocess_for_surya(pil_img)
        try:
            surya_results = rec_predictor([pil_preprocessed])
            surya_text = extract_text_from_result(surya_results[0])
            surya_text = clean_text(surya_text)
            surya_page_lines = [l.strip() for l in surya_text.split('\n') if l.strip()]
        except Exception as e:
            print(f"  Surya OCR failed: {e}")
            surya_page_lines = []

        # ── 3. Assign Surya text to detected lines ───────────────
        # Sort lines top-to-bottom
        line_indices = list(range(len(dino_lines)))
        line_indices.sort(key=lambda i: dino_lines[i][1])

        for rank, li in enumerate(line_indices):
            line_obj = None
            count = 0
            for obj in objects:
                if obj["name"] == "text_line":
                    if count == li:
                        line_obj = obj
                        break
                    count += 1
            if line_obj and rank < len(surya_page_lines):
                line_obj["text"] = surya_page_lines[rank]

        # ── 4. Tesseract for word bounding boxes ─────────────────
        img_for_tess = img.copy()
        for (ix1, iy1, ix2, iy2) in dino_illus:
            cv2.rectangle(img_for_tess, (ix1, iy1), (ix2, iy2), (255, 255, 255), -1)

        word_boxes = []
        word_items = []
        char_boxes = []
        line_idx = 0

        for obj in objects:
            if obj["name"] != "text_line":
                continue
            lx1, ly1, lx2, ly2 = dino_lines[line_idx]
            line_crop = img_for_tess[ly1:ly2, lx1:lx2]
            try:
                # Use Tesseract only for word BOUNDING BOXES, not text
                best_lang = "san"
                best_score = -1_000_000
                for lang in OCR_LANG_CANDIDATES:
                    config = f'--tessdata-dir {TESSDATA_DIR} -l {lang} --psm 7'
                    try:
                        t = pytesseract.image_to_string(line_crop, config=config).strip()
                    except Exception:
                        t = ""
                    s = score_devanagari_ocr(t)
                    if s > best_score:
                        best_lang, best_score = lang, s

                config = f'--tessdata-dir {TESSDATA_DIR} -l {best_lang} --psm 7'
                line_data = pytesseract.image_to_data(
                    line_crop, config=config, output_type=pytesseract.Output.DICT
                )

                # Get Surya words for this line
                surya_line_text = obj.get("text", "")
                surya_words = surya_line_text.split() if surya_line_text else []
                surya_word_idx = 0

                for i in range(len(line_data["level"])):
                    if line_data["level"][i] != 5:
                        continue
                    tess_word = line_data["text"][i].strip()
                    if not tess_word:
                        continue

                    ww = int(line_data["width"][i])
                    wh = int(line_data["height"][i])
                    if ww <= 1 or wh <= 1:
                        continue

                    wx = lx1 + int(line_data["left"][i])
                    wy = ly1 + int(line_data["top"][i])
                    word_box = (wx, wy, wx + ww, wy + wh)

                    # Use Surya text if available, else fall back to Tesseract
                    if surya_word_idx < len(surya_words):
                        word_text = surya_words[surya_word_idx]
                        surya_word_idx += 1
                    else:
                        word_text = tess_word

                    objects.append({
                        "name": "word", "text": word_text,
                        "xmin": word_box[0], "ymin": word_box[1],
                        "xmax": word_box[2], "ymax": word_box[3]
                    })
                    word_boxes.append(word_box)
                    word_items.append({"text": word_text, "bbox": word_box})

            except Exception as e:
                pass
            line_idx += 1

        # ── 5. Character/Glyph boxes from ink projection ─────────
        try:
            for word in word_items:
                wx1, wy1, wx2, wy2 = word["bbox"]
                crop = img_for_tess[wy1:wy2, wx1:wx2]
                if crop.size == 0:
                    continue
                for char_text, cx1, cy1, cx2, cy2 in make_akshara_boxes(
                        crop, word["text"], word["bbox"]):
                    objects.append({
                        "name": "character", "text": char_text,
                        "xmin": cx1, "ymin": cy1, "xmax": cx2, "ymax": cy2
                    })
                    char_boxes.append((cx1, cy1, cx2, cy2))
        except Exception as e:
            print(f"  Glyph box generation failed: {e}")

        # ── 6. Generate PAGE-XML ─────────────────────────────────
        xml_path = create_xml(filename, img.shape, objects)

        # ── 7. Draw visual overlay ───────────────────────────────
        img_vis = img.copy()

        # Red = Characters/Glyphs (thin)
        for (cx1, cy1, cx2, cy2) in char_boxes:
            cv2.rectangle(img_vis, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)

        # Blue = Words
        for (wx1, wy1, wx2, wy2) in word_boxes:
            cv2.rectangle(img_vis, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)

        # Green = Lines
        for (lx1, ly1, lx2, ly2) in dino_lines:
            cv2.rectangle(img_vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

        # Cyan = Illustrations
        for (ix1, iy1, ix2, iy2) in dino_illus:
            cv2.rectangle(img_vis, (ix1, iy1), (ix2, iy2), (255, 255, 0), 3)

        vis_path = os.path.join(OUTPUT_VISUAL_DIR, filename)
        cv2.imwrite(vis_path, img_vis)

        # ── Progress ─────────────────────────────────────────────
        elapsed = time.time() - start_time
        rate = (page_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(image_files) - page_idx - 1) / rate if rate > 0 else 0
        print(f"  Lines={len(dino_lines)} Words={len(word_boxes)} "
              f"Chars={len(char_boxes)} | "
              f"Rate={rate:.2f}pg/s ETA={eta/60:.1f}min")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE! Processed {len(image_files)} pages in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"PAGE-XML: {OUTPUT_XML_DIR}")
    print(f"Visuals:  {OUTPUT_VISUAL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
