import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from lxml import etree
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.cluster import KMeans
import warnings
import easyocr

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 1. DINOv2 LOADING & FEATURE EXTRACTION ─────────────────────────

def load_dino_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.to(DEVICE)
    model.eval()
    return model

def extract_patch_features(model, img_bgr, patch_size=14):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = (img_tensor - torch.tensor(mean)) / torch.tensor(std)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = model.forward_features(img_tensor)['x_norm_patchtokens']
    
    features = features.squeeze(0).cpu().numpy()
    proc_h = new_h // patch_size
    proc_w = new_w // patch_size
    feat_grid = features.reshape(proc_h, proc_w, -1)
    
    return feat_grid, proc_h, proc_w

def cluster_text_mask(feat_grid, img_bgr, n_clusters=3):
    h, w, c = feat_grid.shape
    features_flat = feat_grid.reshape(-1, c)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    labels = kmeans.fit_predict(features_flat)
    label_grid = labels.reshape(h, w)
    
    # Identify the text cluster based on highest variance
    variances = []
    for i in range(n_clusters):
        mask = (label_grid == i)
        if np.sum(mask) < 10:
            variances.append(0)
            continue
        cluster_pixels = feat_grid[mask]
        variances.append(np.var(cluster_pixels))
        
    text_cluster_idx = np.argmax(variances)
    text_mask = (label_grid == text_cluster_idx).astype(np.uint8)
    return text_mask, label_grid

# ─── 2. DINO REGION DETECTION ───────────────────────────────────────

def extract_text_regions_from_mask(text_mask, orig_h, orig_w):
    mask_full = cv2.resize(text_mask * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_dilated = cv2.dilate(mask_full, kernel_dilate)

    kw = int(orig_w // 15)
    kh = int(orig_h // 35)
    if kw % 2 == 0: kw += 1
    if kh % 2 == 0: kh += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 0.005 * (orig_w * orig_h)
    regions = []
    for c in contours:
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rw * rh > min_area:
            pad = 20
            rx1 = max(0, rx - pad)
            ry1 = max(0, ry - pad)
            rx2 = min(orig_w, rx + rw + pad)
            ry2 = min(orig_h, ry + rh + pad)
            regions.append((rx1, ry1, rx2 - rx1, ry2 - ry1))

    if not regions:
        regions = [(0, 0, orig_w, orig_h)]
        
    regions.sort(key=lambda r: r[1])
    return regions, mask_dilated

def create_underline_polygon(start_pt, end_pt, thickness=4):
    xs, ys = start_pt
    xe, ye = end_pt
    return [(int(xs), int(ys - thickness)), (int(xe), int(ye - thickness)), 
            (int(xe), int(ye + thickness)), (int(xs), int(ys + thickness))]

# ─── 3. EASYOCR INFERENCE ──────────────────────────────────────────

def process_easyocr_on_regions(reader, img, regions):
    lines_data = []
    num_words = 0
    
    for r_idx, (rx, ry, rw, rh) in enumerate(regions, 1):
        roi = img[ry:ry+rh, rx:rx+rw]
        
        # Run EasyOCR
        # Returns list of tuples: (bounding_box, text, confidence)
        results = reader.readtext(roi)
        
        for (box, text, prob) in results:
            if prob < 0.1:
                continue
                
            # Translate coordinates back to the original image
            # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (usually Top-Left, Top-Right, Bottom-Right, Bottom-Left)
            abs_box = [(int(pt[0] + rx), int(pt[1] + ry)) for pt in box]
            
            words_text = text.split(" ")
            words_text = [w for w in words_text if w.strip()]
            if not words_text:
                continue
                
            words = []
            
            # Calculate the baseline vector (Bottom-Left to Bottom-Right)
            bl_left = abs_box[3]
            bl_right = abs_box[2]
            
            dx = bl_right[0] - bl_left[0]
            dy = bl_right[1] - bl_left[1]
            num_w = len(words_text)
            
            gap = 0.05
            for wi, w_text in enumerate(words_text):
                start_f = (wi + (gap if wi > 0 else 0)) / num_w
                end_f = (wi + 1 - (gap if wi < num_w - 1 else 0)) / num_w
                
                x_start = bl_left[0] + dx * start_f
                y_start = bl_left[1] + dy * start_f
                x_end = bl_left[0] + dx * end_f
                y_end = bl_left[1] + dy * end_f
                
                word_poly = create_underline_polygon((x_start, y_start), (x_end, y_end), thickness=4)
                
                words.append({
                    'text': w_text,
                    'polygon': word_poly
                })
                
            lines_data.append({
                'polygon': abs_box,
                'bbox': (
                    min(pt[0] for pt in abs_box), 
                    min(pt[1] for pt in abs_box),
                    max(pt[0] for pt in abs_box),
                    max(pt[1] for pt in abs_box)
                ),
                'text': text,
                'region_idx': r_idx,
                'words': words
            })
            num_words += num_w
            
    return lines_data, num_words

# ─── 4. PAGE-XML GENERATION ─────────────────────────────────────────

def bbox_to_coords(x1, y1, x2, y2):
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"

def poly_to_coords(poly):
    return " ".join([f"{x},{y}" for x, y in poly])

def generate_pagexml(img_name, img_w, img_h, lines_data, out_path):
    nsmap = {
        None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    pcgts = etree.Element("PcGts", nsmap=nsmap)
    metadata = etree.SubElement(pcgts, "Metadata")
    etree.SubElement(metadata, "Creator").text = "DINOv2 + EasyOCR Pipeline"
    now = datetime.now().isoformat()
    etree.SubElement(metadata, "Created").text = now
    etree.SubElement(metadata, "LastChange").text = now

    page = etree.SubElement(pcgts, "Page",
                            imageFilename=img_name,
                            imageWidth=str(img_w),
                            imageHeight=str(img_h))

    from collections import defaultdict
    regions_map = defaultdict(list)
    for ld in lines_data:
        r_idx = ld.get('region_idx', 1)
        regions_map[r_idx].append(ld)

    for r_idx in sorted(regions_map.keys()):
        region_lines = regions_map[r_idx]
        
        rx1 = min([ld['bbox'][0] for ld in region_lines])
        ry1 = min([ld['bbox'][1] for ld in region_lines])
        rx2 = max([ld['bbox'][2] for ld in region_lines])
        ry2 = max([ld['bbox'][3] for ld in region_lines])
        
        pad = 10
        rx1 = max(0, rx1 - pad)
        ry1 = max(0, ry1 - pad)
        rx2 = min(img_w, rx2 + pad)
        ry2 = min(img_h, ry2 + pad)

        region_el = etree.SubElement(page, "TextRegion", id=f"r_{r_idx}", type="paragraph")
        etree.SubElement(region_el, "Coords", points=bbox_to_coords(rx1, ry1, rx2, ry2))

        for li, ld in enumerate(region_lines, 1):
            line_el = etree.SubElement(region_el, "TextLine", id=f"r_{r_idx}_l{li}")
            
            line_poly = ld['polygon']
            etree.SubElement(line_el, "Coords", points=poly_to_coords(line_poly))
            
            te = etree.SubElement(line_el, "TextEquiv")
            etree.SubElement(te, "Unicode").text = ld['text']

            for wi, wd in enumerate(ld.get('words', []), 1):
                word_el = etree.SubElement(line_el, "Word", id=f"r_{r_idx}_l{li}_w{wi}")
                etree.SubElement(word_el, "Coords", points=poly_to_coords(wd['polygon']))
                w_te = etree.SubElement(word_el, "TextEquiv")
                etree.SubElement(w_te, "Unicode").text = wd['text']

    tree = etree.ElementTree(pcgts)
    tree.write(str(out_path), pretty_print=True, xml_declaration=True, encoding="utf-8")

def draw_viz(img, lines_data, out_path):
    viz = img.copy()
    for ld in lines_data:
        line_poly = ld['polygon']
        pts = np.array(line_poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(viz, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        for wd in ld.get('words', []):
            word_poly = wd['polygon']
            w_pts = np.array(word_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz, [w_pts], isClosed=True, color=(255, 180, 0), thickness=3)

    cv2.imwrite(str(out_path), viz)

# ─── 5. MAIN PIPELINE ─────────────────────────────────────────────

def process_image(dino_model, reader, img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read {img_path}")

    h, w = img.shape[:2]

    # Step 1: DINOv2 feature extraction & Region Mask
    feat_grid, _, _ = extract_patch_features(dino_model, img)
    text_mask, _ = cluster_text_mask(feat_grid, img)
    regions, _ = extract_text_regions_from_mask(text_mask, h, w)

    # Step 2: EasyOCR lines and text
    lines_data, num_words = process_easyocr_on_regions(reader, img, regions)

    # Step 3: Output XML and Visualization
    out_xml = output_dir / f"{img_path.stem}.xml"
    generate_pagexml(img_path.name, w, h, lines_data, out_xml)

    out_viz = output_dir / f"{img_path.stem}_viz.jpg"
    draw_viz(img, lines_data, out_viz)

    return len(lines_data), num_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image or folder")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading DINOv2 model on {DEVICE}...")
    dino_model = load_dino_model()
    
    print("Initializing EasyOCR with Hindi/Devanagari model...")
    # Disable verbose to prevent UnicodeEncodeError in Windows terminal during model download
    reader = easyocr.Reader(['hi', 'en'], gpu=(DEVICE == "cuda"), verbose=False)

    if in_path.is_file():
        images = [in_path]
    else:
        images = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")))

    print(f"Processing {len(images)} images...")
    total_lines, total_words = 0, 0

    pbar = tqdm(images, desc="Processing")
    for img_path in pbar:
        try:
            lines_count, words_count = process_image(dino_model, reader, img_path, out_path)
            total_lines += lines_count
            total_words += words_count
        except Exception as e:
            print(f"\nError on {img_path.name}: {e}")

    print(f"\nDone! {total_lines} lines, {total_words} words across {len(images)} images.")

if __name__ == "__main__":
    main()
