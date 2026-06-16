import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from lxml import etree
import sys
from tqdm import tqdm

# Add training dir to path to import UNet
sys.path.append(str(Path(__file__).resolve().parents[1] / "training"))
from unet_model import UNet
from polygon_refiner import process_unet_outputs

def points_to_string(points):
    """Converts a list of [x, y] coordinates to PAGE-XML Coords string 'x,y x,y ...'"""
    return " ".join([f"{int(x)},{int(y)}" for x, y in points])

def binarize(img_bgr):
    """Adaptive threshold binarization for text detection."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 5)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

def smooth_polygon(poly, epsilon_factor=0.01):
    """Smooth a jagged polygon using convex hull + approxPolyDP."""
    pts = np.array(poly, dtype=np.int32)
    if len(pts) < 3:
        return poly
    hull = cv2.convexHull(pts)
    epsilon = epsilon_factor * cv2.arcLength(hull, True)
    smoothed = cv2.approxPolyDP(hull, epsilon, True)
    return smoothed.reshape(-1, 2).tolist()

def clip_line_to_region(line_poly, rx1, ry1, rx2, ry2):
    """Clip a text line polygon so it stays within the text region bounds."""
    clipped = []
    for x, y in line_poly:
        cx = max(rx1, min(rx2, x))
        cy = max(ry1, min(ry2, y))
        clipped.append([cx, cy])
    return clipped

def extract_line_polygon(binary_roi, offset_x, offset_y, region_bounds=None, epsilon_factor=0.003):
    """Create a clean rectangular polygon for a text line."""
    h, w = binary_roi.shape
    # Find actual ink extent
    cols = np.where(np.sum(binary_roi, axis=0) > 0)[0]
    rows = np.where(np.sum(binary_roi, axis=1) > 0)[0]
    
    if len(cols) > 0 and len(rows) > 0:
        x1 = offset_x + cols[0]
        x2 = offset_x + cols[-1]
        y1 = offset_y + rows[0]
        y2 = offset_y + rows[-1]
    else:
        x1, y1 = offset_x, offset_y
        x2, y2 = offset_x + w, offset_y + h
    
    # Clip to region bounds
    if region_bounds is not None:
        rx1, ry1, rx2, ry2 = region_bounds
        x1 = max(rx1, x1)
        y1 = max(ry1, y1)
        x2 = min(rx2, x2)
        y2 = min(ry2, y2)
    
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def detect_lines_in_region(binary, rx, ry, rw, rh, region_bounds=None):
    """Use horizontal projection profile to split a text region into individual lines."""
    roi = binary[ry:ry + rh, rx:rx + rw]
    if roi.size == 0 or rw < 10 or rh < 10:
        return []

    h_proj = np.sum(roi, axis=1).astype(np.float64) / 255.0
    smooth_size = max(5, rh // 40)
    if smooth_size % 2 == 0:
        smooth_size += 1
    h_proj_smooth = cv2.GaussianBlur(h_proj.reshape(-1, 1), (1, smooth_size), 0).flatten()

    max_val = np.max(h_proj_smooth)
    if max_val == 0:
        return []
    h_proj_norm = h_proj_smooth / max_val

    text_rows = np.where(h_proj_norm > 0.05)[0]
    if len(text_rows) == 0:
        return []
    text_start, text_end = text_rows[0], text_rows[-1]
    proj_region = h_proj_norm[text_start:text_end + 1]

    min_distance = max(8, len(proj_region) // 30)
    valleys = []
    for i in range(min_distance, len(proj_region) - min_distance):
        left_max = np.max(proj_region[max(0, i - min_distance):i])
        right_max = np.max(proj_region[i + 1:min(len(proj_region), i + min_distance + 1)])
        local_val = proj_region[i]
        peak_avg = (left_max + right_max) / 2
        if peak_avg > 0 and local_val < peak_avg * 0.85:
            window_half = min_distance // 2
            local_window = proj_region[max(0, i - window_half):min(len(proj_region), i + window_half + 1)]
            if local_val <= np.min(local_window) + 0.01:
                valleys.append(i)

    merged_valleys = []
    if valleys:
        cluster = [valleys[0]]
        for v in valleys[1:]:
            if v - cluster[-1] < min_distance:
                cluster.append(v)
            else:
                merged_valleys.append(min(cluster, key=lambda idx: proj_region[idx]))
                cluster = [v]
        merged_valleys.append(min(cluster, key=lambda idx: proj_region[idx]))

    boundaries = [0] + merged_valleys + [len(proj_region)]
    line_polys = []
    for i in range(len(boundaries) - 1):
        r_start, r_end = boundaries[i], boundaries[i + 1]
        if r_end - r_start < 5:
            continue
        abs_y1 = ry + text_start + r_start
        abs_y2 = ry + text_start + r_end
        line_strip = binary[abs_y1:abs_y2, rx:rx + rw]
        v_proj = np.sum(line_strip, axis=0)
        cols = np.where(v_proj > 0)[0]
        if len(cols) < 5:
            continue
        lx1 = rx + cols[0]
        lx2 = rx + cols[-1]
        line_roi = binary[abs_y1:abs_y2, lx1:lx2]
        poly = extract_line_polygon(line_roi, lx1, abs_y1, region_bounds=region_bounds)
        line_polys.append(poly)

    return line_polys

def create_page_xml(img_path, img_w, img_h, regions_dict, line_polys, out_path):
    """Generates PAGE-XML conforming to 2013 schema."""
    namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    nsmap = {None: namespace}
    
    PcGts = etree.Element("{%s}PcGts" % namespace, nsmap=nsmap)
    Metadata = etree.SubElement(PcGts, "{%s}Metadata" % namespace)
    Creator = etree.SubElement(Metadata, "{%s}Creator" % namespace)
    Creator.text = "AutoAnn-Indic-UNet"
    Created = etree.SubElement(Metadata, "{%s}Created" % namespace)
    Created.text = "2026-06-16T00:00:00"
    LastChange = etree.SubElement(Metadata, "{%s}LastChange" % namespace)
    LastChange.text = "2026-06-16T00:00:00"
    
    Page = etree.SubElement(PcGts, "{%s}Page" % namespace, 
                            imageFilename=Path(img_path).name, 
                            imageWidth=str(img_w), 
                            imageHeight=str(img_h))

    # Page frame (Border)
    if len(regions_dict['page_frame']) > 0:
        largest_frame = max(regions_dict['page_frame'], key=lambda p: cv2.contourArea(np.array(p)))
        Border = etree.SubElement(Page, "{%s}Border" % namespace)
        etree.SubElement(Border, "{%s}Coords" % namespace, points=points_to_string(largest_frame))

    # Text Regions with TextLines nested inside
    for i, poly in enumerate(regions_dict['text_regions']):
        smooth_poly = smooth_polygon(poly)
        TextRegion = etree.SubElement(Page, "{%s}TextRegion" % namespace, id=f"text_region_{i}", type="text_region")
        etree.SubElement(TextRegion, "{%s}Coords" % namespace, points=points_to_string(smooth_poly))
        
        region_pts = np.array(smooth_poly, dtype=np.float32)
        for j, lp in enumerate(line_polys):
            line_pts = np.array(lp)
            cx, cy = np.mean(line_pts, axis=0)
            if cv2.pointPolygonTest(region_pts, (float(cx), float(cy)), False) >= 0:
                TextLine = etree.SubElement(TextRegion, "{%s}TextLine" % namespace, id=f"line_{i}_{j}")
                etree.SubElement(TextLine, "{%s}Coords" % namespace, points=points_to_string(lp))

    # Marginalia
    for i, poly in enumerate(regions_dict['marginalia']):
        smooth_poly = smooth_polygon(poly)
        TextRegion = etree.SubElement(Page, "{%s}TextRegion" % namespace, id=f"marginalia_{i}", type="marginalia")
        etree.SubElement(TextRegion, "{%s}Coords" % namespace, points=points_to_string(smooth_poly))

    # Illustrations
    for i, poly in enumerate(regions_dict['illustrations']):
        GraphicRegion = etree.SubElement(Page, "{%s}GraphicRegion" % namespace, id=f"illustration_{i}")
        etree.SubElement(GraphicRegion, "{%s}Coords" % namespace, points=points_to_string(poly))

    # Damage/Holes
    for i, poly in enumerate(regions_dict['damage_holes']):
        NoiseRegion = etree.SubElement(Page, "{%s}NoiseRegion" % namespace, id=f"damage_{i}")
        etree.SubElement(NoiseRegion, "{%s}Coords" % namespace, points=points_to_string(poly))

    tree = etree.ElementTree(PcGts)
    tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input directory of images")
    parser.add_argument('--output', type=str, required=True, help="Output directory for PAGE-XML files")
    parser.add_argument('--model', type=str, required=True, help="Path to UNet weights")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    model = UNet(n_channels=3, n_classes=6, bilinear=False).to(device)
    
    if Path(args.model).exists():
        model.load_state_dict(torch.load(args.model, map_location=device))
        print("Model weights loaded.")
    else:
        print("WARNING: Model weights not found. Using untrained weights.")
    
    model.eval()

    image_files = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png"))
    for img_path in tqdm(image_files, desc="Processing Images"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        # Preprocess for UNet
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # UNet Inference (for regions, page_frame, damage, etc.)
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(img_tensor)
            else:
                outputs = model(img_tensor)
            probs = torch.sigmoid(outputs[0]).float().cpu().numpy()

        # Get region-level polygons from UNet
        regions = process_unet_outputs(probs, w, h)

        # --- Create a mask of the actual leaf area (exclude dark scanner background) ---
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

        # Mask the binary image
        binary = binarize(img)
        binary = cv2.bitwise_and(binary, leaf_mask)

        # --- TEXT LINE DETECTION using projection profile at full resolution ---
        all_line_polys = []
        for poly in regions['text_regions']:
            pts = np.array(poly, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(w - rx, rw)
            rh = min(h - ry, rh)
            region_bounds = (rx, ry, rx + rw, ry + rh)
            lines = detect_lines_in_region(binary, rx, ry, rw, rh, region_bounds=region_bounds)
            all_line_polys.extend(lines)

        # Export XML
        out_xml_path = out_dir / f"{img_path.stem}.xml"
        create_page_xml(str(img_path), w, h, regions, all_line_polys, str(out_xml_path))

    print(f"Inference complete! PAGE-XML files saved to {out_dir}")

if __name__ == "__main__":
    main()

