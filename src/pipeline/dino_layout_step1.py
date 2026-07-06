import cv2
import torch
import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
import warnings

from opencv_layout_refinement import (
    detect_page_frame, 
    detect_damage_holes, 
    detect_text_regions, 
    classify_marginalia
)

warnings.filterwarnings("ignore", category=UserWarning)

PATCH_SIZE = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dino_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    model = model.to(DEVICE)
    model.eval()
    return model

def extract_patch_features(model, img_bgr, max_dim=1024):
    h, w = img_bgr.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)
    new_h = max((scaled_h // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
    new_w = max((scaled_w // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
    img_resized = cv2.resize(img_bgr, (new_w, new_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_rgb - mean) / std
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        features = model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"]
    n_patches_h = new_h // PATCH_SIZE
    n_patches_w = new_w // PATCH_SIZE
    feat = patch_tokens[0].cpu().numpy()
    feat_grid = feat.reshape(n_patches_h, n_patches_w, -1)
    return feat_grid, new_h, new_w

def is_printed_page(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border = np.concatenate([gray[0:5, :].flatten(), gray[-5:, :].flatten(),
                             gray[:, 0:5].flatten(), gray[:, -5:].flatten()])
    return np.mean(border) > 200

def detect_illustrations_from_binary(binary_img, orig_w, orig_h):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    illus_mask = np.zeros_like(binary_img)
    illus_regions = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < (orig_w * orig_h) * 0.03: continue
        
        roi = binary_img[y:y+h, x:x+w]
        if roi.size == 0: continue
        
        ink_density = np.sum(roi > 0) / roi.size
        aspect = float(w) / h if h > 0 else 1.0
        
        h_proj = np.sum(roi, axis=1).astype(np.float64) / 255
        max_val = np.max(h_proj) if np.max(h_proj) > 0 else 1
        h_proj_norm = h_proj / max_val
        
        valleys = 0
        in_peak = False
        for val in h_proj_norm:
            if val > 0.2: in_peak = True
            elif val < 0.03 and in_peak: valleys += 1; in_peak = False
        
        lines_per_100px = valleys / (h / 100.0) if h > 0 else 0
        
        is_illus = False
        if ink_density > 0.30 and lines_per_100px < 1.5:
            is_illus = True
        if ink_density > 0.25 and 0.3 < aspect < 3.0 and valleys < 3:
            is_illus = True
            
        if is_illus:
            cv2.drawContours(illus_mask, [cnt], -1, 255, -1)
            illus_regions.append({'bbox': (int(x), int(y), int(x+w), int(y+h))})
            
    return illus_mask, illus_regions

def cluster_text_mask(feat_grid, img, is_printed):
    ph, pw, fd = feat_grid.shape
    features_flat = feat_grid.reshape(-1, fd)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (pw, ph), interpolation=cv2.INTER_AREA)
    
    if is_printed:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_flat)
        label_grid = labels.reshape(ph, pw)
        mean_0 = gray_small[label_grid == 0].mean()
        mean_1 = gray_small[label_grid == 1].mean()
        text_label = 0 if mean_0 < mean_1 else 1
        text_mask = (label_grid == text_label).astype(np.uint8)
    else:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_flat)
        label_grid = labels.reshape(ph, pw)
        borders = np.concatenate([label_grid[0,:], label_grid[-1,:], 
                                  label_grid[:,0], label_grid[:,-1]])
        bg_label = np.bincount(borders).argmax()
        remaining = [l for l in range(3) if l != bg_label]
        if len(remaining) == 2:
            mean_a = gray_small[label_grid == remaining[0]].mean()
            mean_b = gray_small[label_grid == remaining[1]].mean()
            text_label = remaining[0] if mean_a < mean_b else remaining[1]
        else:
            text_label = remaining[0]
        text_mask = (label_grid == text_label).astype(np.uint8)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
    return text_mask, label_grid

def extract_polygon_hull(roi, offset_x, offset_y, epsilon_factor=0.005):
    h, w = roi.shape
    pad = 10
    padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(padded, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [(offset_x, offset_y), (offset_x+w, offset_y), (offset_x+w, offset_y+h), (offset_x, offset_y+h)]
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    poly = cv2.approxPolyDP(largest_contour, epsilon, True)
    poly_points = []
    prev_pt = None
    for p in poly:
        x = max(0, min(w, p[0][0] - pad))
        y = max(0, min(h, p[0][1] - pad))
        pt = (int(x + offset_x), int(y + offset_y))
        if pt != prev_pt:
            poly_points.append(pt)
            prev_pt = pt
    if len(poly_points) > 1 and poly_points[-1] == poly_points[0]:
        poly_points.pop()
    if len(poly_points) < 3:
        return [(offset_x, offset_y), (offset_x+w, offset_y), (offset_x+w, offset_y+h), (offset_x, offset_y+h)]
    return poly_points

def get_line_polygon(roi, offset_x, offset_y, epsilon_factor=0.003):
    h, w = roi.shape
    pad = 10
    padded = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(max(50, w * 0.05)), 5))
    dilated = cv2.dilate(padded, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [(offset_x, offset_y), (offset_x+w, offset_y), (offset_x+w, offset_y+h), (offset_x, offset_y+h)]
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    poly = cv2.approxPolyDP(largest_contour, epsilon, True)
    poly_points = []
    prev_pt = None
    for p in poly:
        x = max(0, min(w, p[0][0] - pad))
        y = max(0, min(h, p[0][1] - pad))
        pt = (int(x + offset_x), int(y + offset_y))
        if pt != prev_pt:
            poly_points.append(pt)
            prev_pt = pt
    if len(poly_points) > 1 and poly_points[-1] == poly_points[0]:
        poly_points.pop()
    if len(poly_points) < 3:
        return [(offset_x, offset_y), (offset_x+w, offset_y), (offset_x+w, offset_y+h), (offset_x, offset_y+h)]
    return poly_points

def detect_damage_and_holes(binary, mask_full):
    noise_mask = cv2.bitwise_and(binary, cv2.bitwise_not(mask_full))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(noise_mask, connectivity=8)
    damage_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1000:
            cx, cy, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            c_roi = noise_mask[cy:cy+ch, cx:cx+cw]
            poly = extract_polygon_hull(c_roi, cx, cy)
            damage_regions.append({'bbox': (int(cx), int(cy), int(cx+cw), int(cy+ch)), 'polygon': poly})
    return damage_regions

def detect_lines_in_region(binary, rx, ry, rw, rh, orig_w, orig_h, region_idx):
    roi = binary[ry:ry+rh, rx:rx+rw]
    if roi.size == 0 or rw < 10 or rh < 10: return []
    h_proj = np.sum(roi, axis=1).astype(np.float64) / 255
    smooth_size = max(5, rh // 40)
    if smooth_size % 2 == 0: smooth_size += 1
    h_proj_smooth = cv2.GaussianBlur(h_proj.reshape(-1, 1), (1, smooth_size), 0).flatten()
    max_val = np.max(h_proj_smooth)
    if max_val == 0: return []
    h_proj_norm = h_proj_smooth / max_val
    text_rows = np.where(h_proj_norm > 0.05)[0]
    if len(text_rows) == 0: return []
    text_start, text_end = text_rows[0], text_rows[-1]
    proj_region = h_proj_norm[text_start:text_end+1]
    valleys = []
    min_distance = max(5, len(proj_region) // 50)  # Allow closer lines for dense printed text
    for i in range(min_distance, len(proj_region) - min_distance):
        window_half = min_distance // 2
        left_max = np.max(proj_region[max(0, i - min_distance):i])
        right_max = np.max(proj_region[i+1:min(len(proj_region), i + min_distance + 1)])
        local_val = proj_region[i]
        peak_avg = (left_max + right_max) / 2
        
        # Require a deeper valley to avoid splitting giant Devanagari characters
        threshold = peak_avg * 0.25  # More sensitive valley detection for printed Devanagari
        
        if peak_avg > 0 and local_val < threshold:
            local_window = proj_region[max(0, i - window_half):min(len(proj_region), i + window_half + 1)]
            if local_val <= np.min(local_window) + 0.01: valleys.append(i)
    merged_valleys = []
    if valleys:
        cluster = [valleys[0]]
        for v in valleys[1:]:
            if v - cluster[-1] < min_distance: cluster.append(v)
            else:
                merged_valleys.append(min(cluster, key=lambda idx: proj_region[idx]))
                cluster = [v]
        merged_valleys.append(min(cluster, key=lambda idx: proj_region[idx]))
    boundaries = [0] + merged_valleys + [len(proj_region)]
    lines_data = []
    for i in range(len(boundaries) - 1):
        r_start, r_end = boundaries[i], boundaries[i + 1]
        if r_end - r_start < 5: continue
        abs_y1, abs_y2 = ry + text_start + r_start, ry + text_start + r_end
        line_strip = binary[abs_y1:abs_y2, rx:rx+rw]
        v_proj = np.sum(line_strip, axis=0)
        cols = np.where(v_proj > 0)[0]
        if len(cols) < 5: continue
        lx1, lx2 = rx + cols[0], rx + cols[-1]
        line_roi = binary[abs_y1:abs_y2, lx1:lx2]
        is_marginalia = line_width = lx2 - lx1 < orig_w * 0.15 and (lx1 < orig_w * 0.15 or lx2 > orig_w * 0.85)
        lines_data.append({
            'bbox': (int(lx1), int(abs_y1), int(lx2), int(abs_y2)), 
            'is_marginalia': is_marginalia,
            'region_idx': region_idx,
            'region_bbox': (int(rx), int(ry), int(rw), int(rh))
        })
    return lines_data

def merge_overlapping_regions(regions):
    """Merge regions that overlap vertically into larger combined regions."""
    if not regions:
        return regions
    regions = sorted(regions, key=lambda r: r[1])  # sort by y
    merged = [list(regions[0])]
    for rx, ry, rw, rh in regions[1:]:
        prev = merged[-1]
        prev_y2 = prev[1] + prev[3]
        if ry < prev_y2 + 20:  # 20px tolerance
            new_x = min(prev[0], rx)
            new_y = min(prev[1], ry)
            new_x2 = max(prev[0] + prev[2], rx + rw)
            new_y2 = max(prev_y2, ry + rh)
            merged[-1] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
        else:
            merged.append([rx, ry, rw, rh])
    return [tuple(r) for r in merged]

def detect_columns(binary_masked, rx, ry, rw, rh):
    """Detect column boundaries using vertical projection profile."""
    roi = binary_masked[ry:ry+rh, rx:rx+rw]
    if roi.size == 0: return [(rx, ry, rw, rh)]
    
    v_proj = np.sum(roi, axis=0).astype(np.float64) / 255
    smooth_k = max(5, rw // 50)
    if smooth_k % 2 == 0: smooth_k += 1
    v_smooth = cv2.GaussianBlur(v_proj.reshape(1, -1), (smooth_k, 1), 0).flatten()
    
    max_val = np.max(v_smooth)
    if max_val == 0: return [(rx, ry, rw, rh)]
    v_norm = v_smooth / max_val
    
    # Find wide vertical gaps (columns separated by empty space)
    min_gap_width = rw // 15  # At least ~6% of region width
    in_gap = v_norm < 0.02
    gaps = []
    gap_start = None
    for i in range(len(in_gap)):
        if in_gap[i]:
            if gap_start is None: gap_start = i
        else:
            if gap_start is not None:
                gap_width = i - gap_start
                if gap_width > min_gap_width:
                    gaps.append((gap_start, i))
                gap_start = None
    
    if not gaps:
        return [(rx, ry, rw, rh)]
    
    # Build column boundaries
    columns = []
    prev_end = 0
    for gs, ge in gaps:
        if gs > prev_end + 20:
            columns.append((rx + prev_end, ry, gs - prev_end, rh))
        prev_end = ge
    if prev_end < rw - 20:
        columns.append((rx + prev_end, ry, rw - prev_end, rh))
    
    return columns if columns else [(rx, ry, rw, rh)]

def detect_lines_from_mask(text_mask, binary, orig_h, orig_w, is_printed=False):
    mask_full = cv2.resize(text_mask * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    if is_printed:
        # For printed pages: smaller horizontal dilation to avoid bridging columns
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (orig_w // 15, 5))  # Less vertical dilation to prevent merging lines
    else:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (orig_w // 8, 15))
    
    mask_dilated = cv2.dilate(mask_full, kernel_dilate)
    binary_masked = cv2.bitwise_and(binary, mask_dilated)
    
    if is_printed:
        # For printed: use vertical dilation to connect lines within columns, not across
        kw = max(3, int(orig_w // 20))
        kh = int(orig_h // 10)
    else:
        kw = int(orig_w // 3)
        kh = int(orig_h // 15)
    if kw % 2 == 0: kw += 1
    if kh % 2 == 0: kh += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.001 * (orig_w * orig_h)  # Catch smaller text regions
    regions = []
    for c in contours:
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rw * rh > min_area: regions.append((rx, ry, rw, rh))
    regions = merge_overlapping_regions(regions)
    
    if not is_printed:
        # Original behavior for palm leaves: merge narrow regions
        if regions and any(rw < 0.5 * orig_w for _, _, rw, _ in regions):
            min_x = min(r[0] for r in regions)
            min_y = min(r[1] for r in regions)
            max_x = max(r[0] + r[2] for r in regions)
            max_y = max(r[1] + r[3] for r in regions)
            regions = [(min_x, min_y, max_x - min_x, max_y - min_y)]
    
    if not regions: regions = [(0, 0, orig_w, orig_h)]
    regions.sort(key=lambda r: r[1])
    
    all_lines = []
    r_idx = 0
    for (rx, ry, rw, rh) in regions:
        if is_printed:
            # Split region into columns first
            columns = detect_columns(binary_masked, rx, ry, rw, rh)
        else:
            columns = [(rx, ry, rw, rh)]
        
        for (cx, cy, cw, ch) in columns:
            r_idx += 1
            pad_y, pad_x = 15, 30
            ry_new = max(0, cy - pad_y)
            rh_new = min(orig_h - ry_new, ch + 2 * pad_y)
            rx_new = max(0, cx - pad_x)
            rw_new = min(orig_w - rx_new, cw + 2 * pad_x)
            all_lines.extend(detect_lines_in_region(binary_masked, rx_new, ry_new, rw_new, rh_new, orig_w, orig_h, r_idx))
    
    all_lines.sort(key=lambda b: b['bbox'][1])
    return all_lines, mask_full, binary_masked

def ablate_shirorekha(binary_roi):
    if binary_roi.shape[0] < 10 or binary_roi.shape[1] < 10: return binary_roi
    ablated_roi = binary_roi.copy()
    H, W = ablated_roi.shape
    horiz_proj = np.sum(ablated_roi, axis=1)
    search_limit = int(H * 0.45)
    if search_limit < 3: return ablated_roi
    peak_y = np.argmax(horiz_proj[:search_limit])
    if horiz_proj[peak_y] > W * 0.25 * 255:
        thickness = max(2, int(H * 0.06))
        ablated_roi[max(0, peak_y - thickness):min(H, peak_y + thickness + 1), :] = 0
    return ablated_roi

def detect_words_and_chars_in_line(binary, line_dict):
    lx1, ly1, lx2, ly2 = line_dict['bbox']
    roi = binary[ly1:ly2, lx1:lx2]
    if roi.size == 0: return [], {}
    ablated_roi = ablate_shirorekha(roi)
    H, W = ly2 - ly1, lx2 - lx1
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ablated_roi, connectivity=8)
    min_cc_area = max(6, int(H * 0.5))
    ccs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_cc_area: continue
        cx, cy, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if cw > W * 0.85 and ch < H * 0.25: continue
        ccs.append({'local_bbox': (int(cx), int(cy), int(cx+cw), int(cy+ch)), 'label_id': i, 'centroid_x': float(centroids[i][0])})
    
    if not ccs:
        poly = extract_polygon_hull(roi, lx1, ly1)
        return [{'bbox': (int(lx1), int(ly1), int(lx2), int(ly2)), 'polygon': poly}], {"0": [{'bbox': (int(lx1), int(ly1), int(lx2), int(ly2)), 'polygon': poly}]}
    
    ccs.sort(key=lambda c: c['local_bbox'][0])
    glyphs = []
    for cc in ccs:
        cx1, cy1, cx2, cy2 = cc['local_bbox']
        cc_roi = roi[cy1:cy2, cx1:cx2].copy()
        cc_mask = (labels[cy1:cy2, cx1:cx2] == cc['label_id']).astype(np.uint8) * 255
        dilated_cc = cv2.dilate(cc_mask, np.ones((7, 1), np.uint8), iterations=1)
        restored_cc_mask = cv2.bitwise_and(cc_roi, dilated_cc)
        poly = extract_polygon_hull(restored_cc_mask, lx1 + cx1, ly1 + cy1, epsilon_factor=0.01)
        glyphs.append({'bbox': (int(lx1+cx1), int(ly1+cy1), int(lx1+cx2), int(ly1+cy2)), 'polygon': poly})
        
    avg_char_width = np.mean([g['bbox'][2] - g['bbox'][0] for g in glyphs])
    word_gap_threshold = max(8, avg_char_width * 0.6, H * 0.12)
    word_groups = [[0]]
    for i in range(1, len(glyphs)):
        if glyphs[i]['bbox'][0] - glyphs[i-1]['bbox'][2] > word_gap_threshold: word_groups.append([i])
        else: word_groups[-1].append(i)
        
    words = []
    chars_by_word = {}
    for wi, group in enumerate(word_groups):
        group_glyphs = [glyphs[i] for i in group]
        wx1, wy1 = min(g['bbox'][0] for g in group_glyphs), min(g['bbox'][1] for g in group_glyphs)
        wx2, wy2 = max(g['bbox'][2] for g in group_glyphs), max(g['bbox'][3] for g in group_glyphs)
        word_poly = extract_polygon_hull(binary[wy1:wy2, wx1:wx2], wx1, wy1, epsilon_factor=0.008)
        words.append({'bbox': (int(wx1), int(wy1), int(wx2), int(wy2)), 'polygon': word_poly})
        chars_by_word[str(wi)] = group_glyphs
    return words, chars_by_word

def binarize(img_bgr):
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")))
    
    print(f"[STEP 1] Loading DINOv2 model on {DEVICE}...")
    model = load_dino_model()
    
    all_data = []
    for img_path in tqdm(images, desc="DINO Layout"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        feat_grid, _, _ = extract_patch_features(model, img)
        printed = is_printed_page(img)
        text_mask, _ = cluster_text_mask(feat_grid, img, printed)
        binary = binarize(img)
        lines_data, mask_full, binary_masked = detect_lines_from_mask(text_mask, binary, h, w, is_printed=printed)
        for ld in lines_data:
            words, chars_by_word = detect_words_and_chars_in_line(binary_masked, ld)
            ld['words'] = words
            ld['chars'] = chars_by_word
        page_frame_dict, leaf_mask = detect_page_frame(img)
        page_frame = page_frame_dict
        
        damage_regions = detect_damage_holes(img, leaf_mask)
        
        text_regions_raw = detect_text_regions(binary_masked)
        text_regions, marginalia_regions = classify_marginalia(text_regions_raw, w)
        
        illustrations = [] # Placeholder if no obvious illustrations
        
        all_data.append({
            "img_path": str(img_path),
            "img_w": w,
            "img_h": h,
            "lines_data": lines_data,
            "damage_regions": damage_regions,
            "text_regions": text_regions,
            "page_frame": page_frame,
            "marginalia_regions": marginalia_regions,
            "illustrations": illustrations
        })
        
    with open(out_json, 'w') as f:
        json.dump(all_data, f, cls=NumpyEncoder)
    print(f"[STEP 1] Geometry saved to {out_json}")

if __name__ == "__main__":
    main()
