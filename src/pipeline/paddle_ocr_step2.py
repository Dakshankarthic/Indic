import os
os.environ["FLAGS_use_mkldnn"] = "0"
import cv2
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from lxml import etree
from tqdm import tqdm
from paddleocr import PaddleOCR
from shapely.geometry import Polygon
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def bbox_to_coords(x1, y1, x2, y2):
    return f"{int(x1)},{int(y1)} {int(x2)},{int(y1)} {int(x2)},{int(y2)} {int(x1)},{int(y2)}"

def poly_to_coords(poly):
    return " ".join([f"{int(x)},{int(y)}" for x, y in poly])

def generate_pagexml(img_name, img_w, img_h, lines_data, damage_regions, out_path):
    nsmap = {
        None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    pcgts = etree.Element("PcGts", nsmap=nsmap)
    pcgts.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd")
    metadata = etree.SubElement(pcgts, "Metadata")
    etree.SubElement(metadata, "Creator").text = "2-Step DINOv2 + PaddleOCR (IoU Merge) Pipeline"
    now = datetime.now().isoformat()
    etree.SubElement(metadata, "Created").text = now
    etree.SubElement(metadata, "LastChange").text = now

    page = etree.SubElement(pcgts, "Page",
                            imageFilename=img_name,
                            imageWidth=str(img_w),
                            imageHeight=str(img_h))

    for d_idx, damage in enumerate(damage_regions, 1):
        noise_el = etree.SubElement(page, "NoiseRegion", id=f"damage_{d_idx}")
        etree.SubElement(noise_el, "Coords", points=poly_to_coords(damage['polygon']))

    from collections import defaultdict
    regions_map = defaultdict(list)
    for ld in lines_data:
        r_idx = ld.get('region_idx', 1)
        regions_map[r_idx].append(ld)

    for r_idx in sorted(regions_map.keys()):
        region_lines = regions_map[r_idx]
        is_marginalia = any(ld.get('is_marginalia', False) for ld in region_lines)
        region_type = "marginalia" if is_marginalia else "paragraph"

        rx1 = min([ld['region_bbox'][0] for ld in region_lines])
        ry1 = min([ld['region_bbox'][1] for ld in region_lines])
        rx2 = max([ld['region_bbox'][0] + ld['region_bbox'][2] for ld in region_lines])
        ry2 = max([ld['region_bbox'][1] + ld['region_bbox'][3] for ld in region_lines])

        region_el = etree.SubElement(page, "TextRegion", id=f"r_{r_idx}", type=region_type)
        etree.SubElement(region_el, "Coords", points=bbox_to_coords(rx1, ry1, rx2, ry2))

        for li, ld in enumerate(region_lines, 1):
            line_el = etree.SubElement(region_el, "TextLine", id=f"r_{r_idx}_l{li}")
            
            line_poly = ld['polygon']
            etree.SubElement(line_el, "Coords", points=poly_to_coords(line_poly))
            
            # Write Baseline if we have one (simple baseline across the bottom)
            lx1, ly1, lx2, ly2 = ld['bbox']
            baseline_y = int(ly1 + (ly2 - ly1) * 0.75)
            etree.SubElement(line_el, "Baseline", points=f"{int(lx1)},{baseline_y} {int(lx2)},{baseline_y}")

            for wi, wd in enumerate(ld.get('words', []), 1):
                word_el = etree.SubElement(line_el, "Word", id=f"r_{r_idx}_l{li}_w{wi}")
                etree.SubElement(word_el, "Coords", points=poly_to_coords(wd['polygon']))
                
                # PRImA Glyphs inside Words
                char_group = str(wi - 1)
                for ci, char_dict in enumerate(ld.get('chars', {}).get(char_group, []), 1):
                    glyph_el = etree.SubElement(word_el, "Glyph", id=f"r_{r_idx}_l{li}_w{wi}_c{ci}")
                    etree.SubElement(glyph_el, "Coords", points=poly_to_coords(char_dict['polygon']))

            # Only append TextEquiv if PaddleOCR successfully mapped text!
            if ld.get('text'):
                te = etree.SubElement(line_el, "TextEquiv")
                etree.SubElement(te, "Unicode").text = ld['text']

    tree = etree.ElementTree(pcgts)
    tree.write(str(out_path), pretty_print=True, xml_declaration=True, encoding="utf-8")

def draw_viz(img, lines_data, damage_regions, out_path):
    viz = img.copy()

    for damage in damage_regions:
        pts = np.array(damage['polygon'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    for ld in lines_data:
        # Green for mapped lines, Red for unmapped lines
        color = (0, 255, 0) if ld.get('text') else (0, 0, 255)
        line_poly = ld['polygon']
        pts = np.array(line_poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], isClosed=True, color=color, thickness=3)

        for wd in ld.get('words', []):
            word_poly = wd['polygon']
            w_pts = np.array(word_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz, [w_pts], isClosed=True, color=(255, 200, 50), thickness=1)
            
    cv2.imwrite(str(out_path), viz)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Input temp_dino_regions.json")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(args.json, 'r') as f:
        all_data = json.load(f)

    print("Initializing PaddleOCR with Devanagari model...")
    ocr = PaddleOCR(use_angle_cls=False, lang='hi', use_gpu=False, enable_mkldnn=False)

    total_lines = sum(len(d['lines_data']) for d in all_data)
    total_paddle_mapped = 0

    pbar = tqdm(all_data, desc="Paddle IoU Mapping")
    for data in pbar:
        img_path = data['img_path']
        img_w = data['img_w']
        img_h = data['img_h']
        lines_data = data['lines_data']
        damage_regions = data.get('damage_regions', [])

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1. Run FULL PaddleOCR on the image to get perfect text strings
        paddle_res = ocr.ocr(img, det=True, rec=True)
        paddle_boxes = []
        if paddle_res is not None and len(paddle_res) > 0 and paddle_res[0] is not None:
            for line in paddle_res[0]:
                paddle_boxes.append({
                    'polygon': Polygon(line[0]),
                    'text': line[1][0],
                    'conf': line[1][1],
                    'center_y': sum([pt[1] for pt in line[0]]) / 4,
                    'center_x': sum([pt[0] for pt in line[0]]) / 4
                })

        # 2. Map Paddle strings to DINO polygons via Intersection-over-Minimum-Area & Center Matching
        mapped_count = 0
        for ld in lines_data:
            ld['text'] = "" # Default to empty
            try:
                dino_poly = Polygon(ld['polygon'])
                if not dino_poly.is_valid:
                    dino_poly = dino_poly.buffer(0)
                
                matched_pboxes = []
                
                for pbox in paddle_boxes:
                    if not pbox['polygon'].is_valid:
                        p_poly = pbox['polygon'].buffer(0)
                    else:
                        p_poly = pbox['polygon']
                        
                    intersection = dino_poly.intersection(p_poly).area
                    min_area = min(dino_poly.area, p_poly.area)
                    io_min = intersection / min_area if min_area > 0 else 0
                    
                    dino_min_y = min([p[1] for p in ld['polygon']])
                    dino_max_y = max([p[1] for p in ld['polygon']])
                    
                    if io_min > 0.3 or (dino_min_y <= pbox['center_y'] <= dino_max_y and intersection > 0):
                        matched_pboxes.append(pbox)
                            
                # If matches found, sort by x-coordinate and concatenate text
                if matched_pboxes:
                    matched_pboxes.sort(key=lambda b: b['center_x'])
                    ld['text'] = " ".join([b['text'] for b in matched_pboxes])
                    mapped_count += 1
            except Exception as e:
                pass # Ignore geometric errors

        total_paddle_mapped += mapped_count
        pbar.set_postfix({"Mapped": f"{mapped_count}/{len(lines_data)}"})

        img_name = Path(img_path).name
        out_xml = out_path / f"{Path(img_name).stem}.xml"
        generate_pagexml(img_name, img_w, img_h, lines_data, damage_regions, out_xml)

        out_viz = out_path / f"{Path(img_name).stem}_viz.jpg"
        draw_viz(img, lines_data, damage_regions, out_viz)

    print(f"\n[STEP 2] Done! Generated valid XMLs mapped with PaddleOCR Text ({total_paddle_mapped}/{total_lines} lines).")

if __name__ == "__main__":
    main()
