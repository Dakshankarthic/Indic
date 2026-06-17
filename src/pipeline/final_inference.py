import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from scipy.signal import find_peaks
import sys

# Add training directory to path so we can import the model
sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from unet_model import UNet
from polygon_refiner import process_unet_outputs

# Map model class indices to PAGE-XML region types
CLASS_MAPPING = {
    0: "TextRegion",
    1: "Marginalia",
    2: "GraphicRegion", # illustrations
    3: "PageFrame",
    4: "NoiseRegion", # damage/holes
    5: "TextLine"
}

def binarize(image):
    """Adaptive binarization optimized for palm leaves."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    # Remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def clip_line_to_region(pts, rx1, ry1, rx2, ry2):
    """Clip a polygon to the text region bounds."""
    clipped = []
    for x, y in pts:
        cx = max(rx1, min(rx2, x))
        cy = max(ry1, min(ry2, y))
        clipped.append([cx, cy])
    return clipped

def find_seams(energy, start_ys):
    """Dynamic programming to find paths of least resistance between text lines."""
    h, w = energy.shape
    seams = []
    
    for start_y in start_ys:
        dp = np.full((h, w), np.inf, dtype=np.float32)
        paths = np.zeros((h, w), dtype=np.int32)
        
        search_range = 30 # search window around the valley
        min_y = max(0, start_y - search_range)
        max_y = min(h - 1, start_y + search_range)
        dp[min_y:max_y+1, 0] = energy[min_y:max_y+1, 0]
        
        for x in range(1, w):
            for y in range(min_y, max_y + 1):
                y_prev_min = max(min_y, y - 1)
                y_prev_max = min(max_y, y + 1)
                
                prev_costs = dp[y_prev_min:y_prev_max+1, x-1]
                if len(prev_costs) == 0: continue
                min_idx = np.argmin(prev_costs)
                
                dp[y, x] = energy[y, x] + prev_costs[min_idx]
                paths[y, x] = y_prev_min + min_idx
                
        # Backtrack
        end_y_costs = dp[min_y:max_y+1, w-1]
        if np.all(np.isinf(end_y_costs)):
            continue
        end_y = min_y + np.argmin(end_y_costs)
        
        seam = []
        curr_y = end_y
        for x in range(w-1, -1, -1):
            seam.append((x, curr_y))
            curr_y = paths[curr_y, x]
        
        seam.reverse()
        seams.append(seam)
        
    return seams

def extract_line_rect(line_binary, offset_x, offset_y, region_bounds=None):
    """Create a tight 4-point rectangular polygon around ink in a text line."""
    h, w = line_binary.shape
    
    cols = np.where(np.sum(line_binary, axis=0) > 0)[0]
    rows = np.where(np.sum(line_binary, axis=1) > 0)[0]
    
    if len(cols) == 0 or len(rows) == 0:
        return []
    
    x1 = offset_x + int(cols[0])
    x2 = offset_x + int(cols[-1])
    y1 = offset_y + int(rows[0])
    y2 = offset_y + int(rows[-1])
    
    # Clip to region bounds
    if region_bounds is not None:
        rx1, ry1, rx2, ry2 = region_bounds
        x1 = max(rx1, x1)
        y1 = max(ry1, y1)
        x2 = min(rx2, x2)
        y2 = min(ry2, y2)
    
    if x2 <= x1 or y2 <= y1:
        return []
    
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def detect_lines_in_region(binary, rx, ry, rw, rh, region_bounds=None):
    """Extract individual text lines using projection profile + simple rectangles."""
    roi = binary[ry:ry+rh, rx:rx+rw]
    
    if roi.size == 0 or rw < 10 or rh < 10:
        return []
    
    # Horizontal projection profile
    proj = np.sum(roi, axis=1).astype(np.float64) / 255.0
    proj_smooth = np.convolve(proj, np.ones(10)/10, mode='same')
    
    peaks, _ = find_peaks(proj_smooth, distance=20, prominence=200)
    if len(peaks) < 2:
        # Single line fallback
        rect = extract_line_rect(roi, rx, ry, region_bounds)
        return [rect] if rect else []
    
    # Find valleys between peaks
    valleys = []
    for i in range(len(peaks) - 1):
        valley_y = peaks[i] + np.argmin(proj_smooth[peaks[i]:peaks[i+1]])
        valleys.append(valley_y)
    
    # Line boundaries
    boundaries = [0] + valleys + [rh]
    
    lines = []
    for i in range(len(boundaries) - 1):
        y1, y2 = boundaries[i], boundaries[i+1]
        if y2 - y1 < 8:
            continue
        line_roi = roi[y1:y2, :]
        if np.sum(line_roi) > 255 * 10:
            rect = extract_line_rect(line_roi, rx, ry + y1, region_bounds)
            if rect:
                lines.append(rect)
    
    return lines

def detect_words_and_glyphs(binary, line_rect):
    """Detect words and individual characters (glyphs) inside a text line.
    
    Uses connected component bounding boxes for characters.
    Groups characters into words based on horizontal gap analysis.
    Returns: list of words, where each word is {'rect': [...], 'glyphs': [...]}
    """
    x1, y1 = line_rect[0]
    x2, y2 = line_rect[2]
    
    line_roi = binary[y1:y2, x1:x2]
    if line_roi.size == 0 or line_roi.shape[0] < 3 or line_roi.shape[1] < 3:
        return []
    
    # Find connected components (each ink blob = one character/akshara)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_roi, connectivity=8)
    
    # Collect valid character bounding boxes (filter noise)
    min_char_area = 15  # minimum pixels for a character
    char_boxes = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_char_area:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        # Convert to absolute coordinates
        abs_x1 = x1 + cx
        abs_y1 = y1 + cy
        abs_x2 = x1 + cx + cw
        abs_y2 = y1 + cy + ch
        char_boxes.append({
            'rect': [[abs_x1, abs_y1], [abs_x2, abs_y1], [abs_x2, abs_y2], [abs_x1, abs_y2]],
            'cx': abs_x1 + cw // 2,
            'left': abs_x1,
            'right': abs_x2
        })
    
    if not char_boxes:
        return []
    
    # Sort characters left-to-right
    char_boxes.sort(key=lambda c: c['left'])
    
    # Group into words by detecting large horizontal gaps
    if len(char_boxes) < 2:
        # Single character = single word
        word_rect = char_boxes[0]['rect']
        return [{'rect': word_rect, 'glyphs': [c['rect'] for c in char_boxes]}]
    
    # Calculate gaps between consecutive characters
    gaps = []
    for i in range(len(char_boxes) - 1):
        gap = char_boxes[i+1]['left'] - char_boxes[i]['right']
        gaps.append(max(0, gap))
    
    # Word boundary threshold: gaps larger than median * 2.5 (or at least 8px)
    if gaps:
        median_gap = np.median(gaps)
        word_gap_threshold = max(8, median_gap * 2.5)
    else:
        word_gap_threshold = 8
    
    # Group characters into words
    words = []
    current_word_chars = [char_boxes[0]]
    
    for i in range(len(gaps)):
        if gaps[i] > word_gap_threshold:
            # End current word, start new one
            words.append(current_word_chars)
            current_word_chars = [char_boxes[i+1]]
        else:
            current_word_chars.append(char_boxes[i+1])
    words.append(current_word_chars)  # last word
    
    # Build word-level bounding boxes
    result = []
    for word_chars in words:
        all_x1 = min(c['rect'][0][0] for c in word_chars)
        all_y1 = min(c['rect'][0][1] for c in word_chars)
        all_x2 = max(c['rect'][2][0] for c in word_chars)
        all_y2 = max(c['rect'][2][1] for c in word_chars)
        word_rect = [[all_x1, all_y1], [all_x2, all_y1], [all_x2, all_y2], [all_x1, all_y2]]
        glyph_rects = [c['rect'] for c in word_chars]
        result.append({'rect': word_rect, 'glyphs': glyph_rects})
    
    return result

def create_page_xml(image_path, width, height, regions, line_polys, line_words, out_path):
    """Generate PAGE-XML with full hierarchy: TextRegion > TextLine > Word > Glyph."""
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    schema_loc = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
    
    def pts_str(poly):
        return " ".join([f"{int(p[0])},{int(p[1])}" for p in poly])
    
    root = ET.Element("PcGts", xmlns=ns)
    root.set("xmlns:xsi", xsi)
    root.set("xsi:schemaLocation", schema_loc)
    
    metadata = ET.SubElement(root, "Metadata")
    creator = ET.SubElement(metadata, "Creator")
    creator.text = "AutoAnn-Indic-Pipeline"
    created = ET.SubElement(metadata, "Created")
    created.text = datetime.now().isoformat()
    last_change = ET.SubElement(metadata, "LastChange")
    last_change.text = datetime.now().isoformat()
    
    page = ET.SubElement(root, "Page")
    page.set("imageFilename", Path(image_path).name)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))
    
    # Text Regions with full hierarchy
    for i, poly in enumerate(regions['text_regions']):
        region = ET.SubElement(page, "TextRegion")
        region.set("id", f"region_text_{i}")
        coords = ET.SubElement(region, "Coords")
        coords.set("points", pts_str(poly))
        
        # Add text lines to this region
        region_pts = np.array(poly, dtype=np.float32)
        for j, lpoly in enumerate(line_polys):
            line_pts = np.array(lpoly, dtype=np.float32)
            cx, cy = np.mean(line_pts, axis=0)
            if cv2.pointPolygonTest(region_pts, (float(cx), float(cy)), False) >= 0:
                tl = ET.SubElement(region, "TextLine")
                tl.set("id", f"line_{i}_{j}")
                tcoords = ET.SubElement(tl, "Coords")
                tcoords.set("points", pts_str(lpoly))
                
                # Add Words and Glyphs inside this TextLine
                words = line_words.get(j, [])
                for k, word in enumerate(words):
                    w_elem = ET.SubElement(tl, "Word")
                    w_elem.set("id", f"word_{i}_{j}_{k}")
                    wcoords = ET.SubElement(w_elem, "Coords")
                    wcoords.set("points", pts_str(word['rect']))
                    
                    # Add Glyphs (characters) inside this Word
                    for g, glyph_rect in enumerate(word['glyphs']):
                        g_elem = ET.SubElement(w_elem, "Glyph")
                        g_elem.set("id", f"glyph_{i}_{j}_{k}_{g}")
                        gcoords = ET.SubElement(g_elem, "Coords")
                        gcoords.set("points", pts_str(glyph_rect))

    # Other regions
    def add_regions(region_list, tag_name, id_prefix):
        for i, poly in enumerate(region_list):
            if len(poly) < 3: continue
            r = ET.SubElement(page, tag_name)
            r.set("id", f"{id_prefix}_{i}")
            c = ET.SubElement(r, "Coords")
            c.set("points", pts_str(poly))
            
    add_regions(regions['marginalia'], "TextRegion", "region_margin")
    add_regions(regions['illustrations'], "GraphicRegion", "region_illus")
    add_regions(regions['page_frame'], "Border", "border")
    add_regions(regions['damage_holes'], "NoiseRegion", "noise")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = UNet(n_classes=6).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("Model weights loaded.")

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list(in_dir.glob("*.jpg"))
    from tqdm import tqdm
    for img_path in tqdm(img_paths, desc="Processing Images"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        regions = process_unet_outputs(probs, w, h)

        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        if len(regions['page_frame']) > 0:
            for pf_poly in regions['page_frame']:
                pts = np.array(pf_poly, dtype=np.int32)
                cv2.fillPoly(leaf_mask, [pts], 255)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, leaf_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

        binary = binarize(img)
        binary = cv2.bitwise_and(binary, leaf_mask)

        all_line_polys = []
        all_line_words = {}  # maps line index -> list of words
        for poly in regions['text_regions']:
            pts = np.array(poly, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(w - rx, rw)
            rh = min(h - ry, rh)
            region_bounds = (rx, ry, rx + rw, ry + rh)
            lines = detect_lines_in_region(binary, rx, ry, rw, rh, region_bounds=region_bounds)
            for line_rect in lines:
                line_idx = len(all_line_polys)
                all_line_polys.append(line_rect)
                # Detect words and glyphs inside this line
                words = detect_words_and_glyphs(binary, line_rect)
                all_line_words[line_idx] = words

        out_xml_path = out_dir / f"{img_path.stem}.xml"
        create_page_xml(str(img_path), w, h, regions, all_line_polys, all_line_words, str(out_xml_path))

    print(f"Inference complete! PAGE-XML files saved to {out_dir}")

if __name__ == "__main__":
    main()
