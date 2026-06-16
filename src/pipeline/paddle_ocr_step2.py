import os
os.environ["FLAGS_use_mkldnn"] = "0"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import argparse
import json
import numpy as np
from datetime import datetime
from lxml import etree
from tqdm import tqdm
from paddleocr import PaddleOCR
from aletheia_utils import calculate_human_effort_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== 1. INITIALIZATION ====================

def init_paddle_ocr(use_gpu=True):
    """Initialize PaddleOCR with optimized parameters for ancient Devanagari manuscripts.
    
    Key fixes vs old pipeline:
    - rec_image_shape='3, 48, 320' (standard PaddleOCR recommendation, avoids aspect distortion)
    - drop_score=0.4 (filters out garbage while keeping borderline reads)
    - det=True in ocr.ocr() calls (let PaddleOCR handle detection internally)
    """
    common_kwargs = dict(
        use_angle_cls=False,
        lang='hi',
        show_log=False,
        drop_score=0.4,
        rec_image_shape='3, 48, 320',
        use_gpu=False,
        enable_mkldnn=False,
        rec_batch_num=1,
    )
    if use_gpu:
        try:
            gpu_kwargs = {**common_kwargs, 'use_gpu': True}
            ocr = PaddleOCR(**gpu_kwargs)
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            ocr.ocr(dummy, det=False, rec=True)
            print("Successfully initialized PaddleOCR on GPU.")
            return ocr
        except Exception as e:
            print(f"Failed to initialize PaddleOCR on GPU: {e}. Falling back to CPU...")
    
    ocr = PaddleOCR(**common_kwargs)
    print("Successfully initialized PaddleOCR on CPU.")
    return ocr

# ==================== 2. HELPER FUNCTIONS ====================

def binarize(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary

def preprocess_for_ocr(crop):
    """Gentle preprocessing for ancient manuscript OCR.
    
    PaddleOCR recognition models were trained on natural images with ink-on-paper 
    contrast. Aggressive binarization DESTROYS the gray-level stroke variations, 
    diacritics (matras), and shirorekha context that the model uses to recognize 
    Devanagari characters. 
    
    This function does MINIMAL processing:
    1. Convert to grayscale
    2. Mild denoising to suppress palm-leaf texture
    3. CLAHE to enhance faded ink contrast
    4. Normalize intensity range to improve model consistency
    5. Convert back to 3-channel BGR
    
    NO binarization. NO connected-component filtering. NO morphological ops.
    """
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    
    # 1. Mild denoising — just enough to suppress palm leaf texture grain
    #    h=3 is gentle; old code used h=12 which was smearing ink strokes
    denoised = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
    
    # 2. CLAHE contrast enhancement — clipLimit=2.0 is balanced
    #    Old code used clipLimit=4.0 which was over-amplifying noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 3. Normalize intensity range to 0-255 for model consistency
    #    This helps the model see consistent ink contrast regardless of lighting
    min_val, max_val = enhanced.min(), enhanced.max()
    if max_val > min_val:
        normalized = ((enhanced.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = enhanced
    
    # 4. Convert to 3-channel BGR for PaddleOCR
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

# ==================== 4. PAGE-XML GENERATION ====================

def bbox_to_coords(x1, y1, x2, y2):
    return f"{int(x1)},{int(y1)} {int(x2)},{int(y1)} {int(x2)},{int(y2)} {int(x1)},{int(y2)}"

def poly_to_coords(poly):
    return " ".join([f"{int(x)},{int(y)}" for x, y in poly])

def generate_pagexml(img_name, img_w, img_h, item, out_path):
    nsmap = {None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15", "xsi": "http://www.w3.org/2001/XMLSchema-instance"}
    pcgts = etree.Element("PcGts", nsmap=nsmap)
    pcgts.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd")
    
    metadata = etree.SubElement(pcgts, "Metadata")
    etree.SubElement(metadata, "Creator").text = "Hybrid DINOv2 + PaddleOCR Pipeline"
    now = datetime.now().isoformat()
    etree.SubElement(metadata, "Created").text = now
    etree.SubElement(metadata, "LastChange").text = now

    page = etree.SubElement(pcgts, "Page", imageFilename=img_name, imageWidth=str(img_w), imageHeight=str(img_h))

    if 'page_frame' in item:
        pf = item['page_frame']
        etree.SubElement(page, "Border").append(etree.Element("Coords", points=bbox_to_coords(*pf['bbox'])))

    for i, tr in enumerate(item.get('text_regions', [])):
        region = etree.SubElement(page, "TextRegion", id=f"tr_{i}", type="paragraph")
        etree.SubElement(region, "Coords", points=bbox_to_coords(*tr['bbox']))

    for i, mr in enumerate(item.get('marginalia_regions', [])):
        region = etree.SubElement(page, "TextRegion", id=f"marg_{i}", type="marginalia")
        etree.SubElement(region, "Coords", points=bbox_to_coords(*mr['bbox']))

    for i, dr in enumerate(item.get('damage_regions', [])):
        noise = etree.SubElement(page, "NoiseRegion", id=f"damage_{i}")
        etree.SubElement(noise, "Coords", points=poly_to_coords(dr['polygon']))

    from collections import defaultdict
    regions_map = defaultdict(list)
    for ld in item.get('lines_data', []):
        r_idx = ld.get('region_idx', 1)
        regions_map[r_idx].append(ld)

    for r_idx in sorted(regions_map.keys()):
        region_lines = regions_map[r_idx]
        region_el = page.find(f".//{{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}}TextRegion[@id='tr_{r_idx-1}']")
        if region_el is None:
            region_el = etree.SubElement(page, "TextRegion", id=f"tr_auto_{r_idx}", type="paragraph")

        for li, ld in enumerate(region_lines, 1):
            lx1, ly1, lx2, ly2 = ld['bbox']
            
            if ld.get('is_marginalia', False):
                parent_region = etree.SubElement(page, "TextRegion", id=f"marg_auto_{r_idx}_{li}", type="marginalia")
                etree.SubElement(parent_region, "Coords", points=bbox_to_coords(lx1, ly1, lx2, ly2))
            else:
                parent_region = region_el
                
            line_el = etree.SubElement(parent_region, "TextLine", id=f"r_{r_idx}_l{li}")
            
            etree.SubElement(line_el, "Coords", points=bbox_to_coords(lx1, ly1, lx2, ly2))
            
            baseline_y = ly1 + int((ly2 - ly1) * 0.75)
            etree.SubElement(line_el, "Baseline", points=f"{lx1},{baseline_y} {lx2},{baseline_y}")

            for wi, wd in enumerate(ld.get('words', []), 1):
                word_el = etree.SubElement(line_el, "Word", id=f"r_{r_idx}_l{li}_w{wi}")
                if 'polygon' in wd and wd['polygon']:
                    etree.SubElement(word_el, "Coords", points=poly_to_coords(wd['polygon']))
                elif 'bbox' in wd:
                    etree.SubElement(word_el, "Coords", points=bbox_to_coords(*wd['bbox']))

                char_group = str(wi - 1)
                for ci, cd in enumerate(ld.get('chars', {}).get(char_group, []), 1):
                    g_el = etree.SubElement(word_el, "Glyph", id=f"r_{r_idx}_l{li}_w{wi}_g{ci}")
                    if 'polygon' in cd and cd['polygon']:
                        etree.SubElement(g_el, "Coords", points=poly_to_coords(cd['polygon']))
                    else:
                        etree.SubElement(g_el, "Coords", points=bbox_to_coords(*cd['bbox']))

                if wd.get('text'):
                    te = etree.SubElement(word_el, "TextEquiv")
                    etree.SubElement(te, "Unicode").text = wd['text']

            if ld.get('text'):
                te = etree.SubElement(line_el, "TextEquiv")
                etree.SubElement(te, "Unicode").text = ld['text']

    tree = etree.ElementTree(pcgts)
    tree.write(str(out_path), pretty_print=True, xml_declaration=True, encoding="utf-8")

# ==================== 5. VISUALIZATION ====================

def draw_viz(img, item, out_path):
    viz = img.copy()

    # Colors (BGR)
    color_frame = (255, 0, 255) # Magenta
    color_text_region = (0, 255, 0) # Green
    color_marginalia = (255, 255, 0) # Cyan
    color_damage = (0, 0, 255) # Red
    color_line = (255, 128, 0) # Blue-ish/Cyan
    color_char = (128, 0, 255) # Pink/Purple
    
    if 'page_frame' in item:
        x1, y1, x2, y2 = item['page_frame']['bbox']
        cv2.rectangle(viz, (x1, y1), (x2, y2), color_frame, 2)
        cv2.putText(viz, "page/frame", (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_frame, 2)

    for tr in item.get('text_regions', []):
        x1, y1, x2, y2 = tr['bbox']
        cv2.rectangle(viz, (x1, y1), (x2, y2), color_text_region, 2)
        cv2.putText(viz, "text regions", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text_region, 2)

    for mr in item.get('marginalia_regions', []):
        x1, y1, x2, y2 = mr['bbox']
        cv2.rectangle(viz, (x1, y1), (x2, y2), color_marginalia, 2)
        cv2.putText(viz, "marginalia/notes", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_marginalia, 2)

    for dr in item.get('damage_regions', []):
        pts = np.array(dr['polygon'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], isClosed=True, color=color_damage, thickness=2)
        cv2.putText(viz, "damage/holes", tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_damage, 2)

    for ld in item.get('lines_data', []):
        lx1, ly1, lx2, ly2 = ld['bbox']
        # Line in box
        cv2.rectangle(viz, (lx1, ly1), (lx2, ly2), color_line, 1)
        
        # Character in polygon
        for char_list in ld.get('chars', {}).values():
            for cd in char_list:
                glyph_poly = cd.get('polygon', [])
                if glyph_poly:
                    pts_g = np.array(glyph_poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(viz, [pts_g], isClosed=True, color=color_char, thickness=1)
                    
        for wd in ld.get('words', []):
            if 'polygon' in wd and wd['polygon']:
                pts = wd['polygon']
                wx1 = min([p[0] for p in pts])
                wx2 = max([p[0] for p in pts])
                wy2 = max([p[1] for p in pts])
            elif 'bbox' in wd:
                wx1, wy1, wx2, wy2 = wd['bbox']
            
            # Underline the DINO word with BLACK
            cv2.line(viz, (int(wx1), int(wy2)), (int(wx2), int(wy2)), (0, 0, 0), 2)

    cv2.imwrite(str(out_path), viz)

# ==================== 6. MAIN PIPELINE STEP 2 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Input temp_dino_regions.json")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--gpu", action="store_true", help="Try using GPU for PaddleOCR")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP 2] Initializing OCR Engine...")
    ocr_engine = init_paddle_ocr(use_gpu=args.gpu)

    with open(args.json, 'r') as f:
        all_data = json.load(f)

    total_lines = 0
    total_words = 0
    total_e = 0.0

    print("[STEP 2] Running OCR Recognition on DINO Words...")
    for item in tqdm(all_data, desc="PaddleOCR Transcription"):
        img_path = item["img_path"]
        img_w = item["img_w"]
        img_h = item["img_h"]
        lines_data = item["lines_data"]

        img = cv2.imread(img_path)
        if img is None: continue

        for ld in lines_data:
            lx1, ly1, lx2, ly2 = ld['bbox']
            
            # Expand line crop for OCR context
            pad = 6
            cy1 = max(0, ly1 - pad)
            cy2 = min(img_h, ly2 + pad)
            cx1 = max(0, lx1 - pad)
            cx2 = min(img_w, lx2 + pad)
            
            line_crop = img[cy1:cy2, cx1:cx2]
            
            if line_crop.size > 0 and line_crop.shape[0] > 5 and line_crop.shape[1] > 5:
                # Gentle preprocessing — preserve gray-level ink information
                line_crop_processed = preprocess_for_ocr(line_crop)
                
                # Use det=False, rec=True for pre-cropped line
                # The line-level context (including shirorekha) helps PaddleOCR
                res = ocr_engine.ocr(line_crop_processed, det=False, rec=True)
                
                if res and res[0] and len(res[0]) > 0:
                    text = res[0][0][0]
                    conf = res[0][0][1]
                    # Only keep results with reasonable confidence
                    if float(conf) >= 0.4:
                        ld['text'] = text
                        ld['confidence'] = float(conf)
                    else:
                        ld['text'] = ''
                        ld['confidence'] = float(conf)
                else:
                    ld['text'] = ''
                    ld['confidence'] = 0.0
            else:
                ld['text'] = ''
                ld['confidence'] = 0.0
            
            # DINO word geometry stays untouched — we only add text at line level
            total_words += len(ld.get('words', []))
            total_lines += 1

        # Output PAGE-XML
        img_name = Path(img_path).name
        xml_name = out_dir / (Path(img_name).stem + ".xml")
        generate_pagexml(img_name, img_w, img_h, item, xml_name)

        # Visualization
        viz_name = out_dir / (Path(img_name).stem + "_viz.jpg")
        draw_viz(img, item, viz_name)

        # Evaluate Human Effort
        binary = binarize(img)
        effort_res = calculate_human_effort_score(lines_data, binary)
        xml_e = effort_res["score"]
        total_e += xml_e

    print("\n[STEP 2] Done! Processed {} images.".format(len(all_data)))
    print(f"Total Lines: {total_lines}, Total Words: {total_words}")
    print(f"Total Simulated Human Effort Score (E): {total_e}")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
