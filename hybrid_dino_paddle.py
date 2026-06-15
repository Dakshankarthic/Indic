import os
# Suppress paddle MKL-DNN warnings
os.environ["FLAGS_use_mkldnn"] = "0"

# Import paddle/paddleocr first to avoid pybind11 collision with torch (generic_type already registered error)
from paddleocr import PaddleOCR
import paddle

import cv2
import torch
import argparse
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from lxml import etree
from tqdm import tqdm
from PIL import ImageFile
from sklearn.cluster import KMeans
from aletheia_utils import calculate_human_effort_score


warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

PATCH_SIZE = 14
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 1. INITIALIZATION ====================

def load_dino_model():
    """Load DINOv2 ViT-B/14."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    model = model.to(DEVICE)
    model.eval()
    return model

def init_paddle_ocr(use_gpu=True):
    """Initializes PaddleOCR on GPU, falls back to CPU if setup/run fails."""
    if use_gpu:
        try:
            ocr = PaddleOCR(use_angle_cls=False, lang='hi', use_gpu=True, enable_mkldnn=False, show_log=False)
            # Dummy test to verify CUDA works
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            ocr.ocr(dummy, det=True, rec=True)
            print("Successfully initialized PaddleOCR on GPU.")
            return ocr
        except Exception as e:
            print(f"Failed to initialize PaddleOCR on GPU: {e}. Falling back to CPU...")
            
    ocr = PaddleOCR(use_angle_cls=False, lang='hi', use_gpu=False, enable_mkldnn=False, show_log=False)
    print("Successfully initialized PaddleOCR on CPU.")
    return ocr

# ==================== 2. DINOv2 LAYOUT DETECTION ====================

def extract_patch_features(model, img_bgr):
    h, w = img_bgr.shape[:2]
    new_h = (h // PATCH_SIZE) * PATCH_SIZE
    new_w = (w // PATCH_SIZE) * PATCH_SIZE
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

def cluster_text_mask(feat_grid, img, n_clusters=3):
    ph, pw, fd = feat_grid.shape
    features_flat = feat_grid.reshape(-1, fd)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
    labels = kmeans.fit_predict(features_flat)
    label_grid = labels.reshape(ph, pw)

    borders = np.concatenate([
        label_grid[0, :], label_grid[-1, :],
        label_grid[:, 0], label_grid[:, -1]
    ])
    scanner_bg_label = np.bincount(borders).argmax()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (pw, ph), interpolation=cv2.INTER_AREA)

    all_labels = set(range(n_clusters))
    remaining_labels = list(all_labels - {scanner_bg_label})

    if len(remaining_labels) == 2:
        label_a, label_b = remaining_labels
        mean_a = gray_small[label_grid == label_a].mean()
        mean_b = gray_small[label_grid == label_b].mean()
        text_label = label_a if mean_a < mean_b else label_b
    elif len(remaining_labels) == 1:
        text_label = remaining_labels[0]
    else:
        text_label = scanner_bg_label

    text_mask = (label_grid == text_label).astype(np.uint8)
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
        x = p[0][0] - pad
        y = p[0][1] - pad
        x = max(0, min(w, x))
        y = max(0, min(h, y))
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
            cx = stats[i, cv2.CC_STAT_LEFT]
            cy = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            c_roi = noise_mask[cy:cy+ch, cx:cx+cw]
            poly = extract_polygon_hull(c_roi, cx, cy)
            damage_regions.append({'bbox': (cx, cy, cx+cw, cy+ch), 'polygon': poly})
    return damage_regions

def detect_lines_in_region(binary, rx, ry, rw, rh, orig_w, orig_h, region_idx):
    roi = binary[ry:ry+rh, rx:rx+rw]
    if roi.size == 0 or rw < 10 or rh < 10:
        return []

    h_proj = np.sum(roi, axis=1).astype(np.float64) / 255
    smooth_size = max(5, rh // 40)
    if smooth_size % 2 == 0:
        smooth_size += 1
    h_proj_smooth = cv2.GaussianBlur(h_proj.reshape(-1, 1), (1, smooth_size), 0).flatten()

    max_val = np.max(h_proj_smooth)
    if max_val == 0:
        return []

    h_proj_norm = h_proj_smooth / max_val
    text_rows = np.where(h_proj_norm > 0.02)[0]
    if len(text_rows) == 0:
        return []

    text_start = text_rows[0]
    text_end = text_rows[-1]
    proj_region = h_proj_norm[text_start:text_end+1]

    valleys = []
    min_distance = max(8, len(proj_region) // 30)

    for i in range(min_distance, len(proj_region) - min_distance):
        window_half = min_distance // 2
        left_max = np.max(proj_region[max(0, i - min_distance):i])
        right_max = np.max(proj_region[i+1:min(len(proj_region), i + min_distance + 1)])
        local_val = proj_region[i]

        peak_avg = (left_max + right_max) / 2
        if peak_avg > 0 and local_val < peak_avg * 0.85:
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
                best = min(cluster, key=lambda idx: proj_region[idx])
                merged_valleys.append(best)
                cluster = [v]
        best = min(cluster, key=lambda idx: proj_region[idx])
        merged_valleys.append(best)

    boundaries = [0] + merged_valleys + [len(proj_region)]
    lines_data = []

    for i in range(len(boundaries) - 1):
        r_start = boundaries[i]
        r_end = boundaries[i + 1]
        if r_end - r_start < 5:
            continue

        abs_y1 = ry + text_start + r_start
        abs_y2 = ry + text_start + r_end

        line_strip = binary[abs_y1:abs_y2, rx:rx+rw]
        v_proj = np.sum(line_strip, axis=0)
        cols = np.where(v_proj > 0)[0]
        if len(cols) < 5:
            continue

        lx1, lx2 = rx + cols[0], rx + cols[-1]
        line_roi = binary[abs_y1:abs_y2, lx1:lx2]
        line_poly = extract_polygon_hull(line_roi, lx1, abs_y1, epsilon_factor=0.003)
        
        is_marginalia = False
        line_width = lx2 - lx1
        if line_width < orig_w * 0.15 and (lx1 < orig_w * 0.15 or lx2 > orig_w * 0.85):
            is_marginalia = True

        lines_data.append({
            'bbox': (lx1, abs_y1, lx2, abs_y2), 
            'polygon': line_poly,
            'is_marginalia': is_marginalia,
            'region_idx': region_idx,
            'region_bbox': (rx, ry, rw, rh)
        })

    return lines_data

def detect_lines_from_mask(text_mask, binary, orig_h, orig_w):
    mask_full = cv2.resize(text_mask * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_dilated = cv2.dilate(mask_full, kernel_dilate)
    binary_masked = cv2.bitwise_and(binary, mask_dilated)

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
            regions.append((rx, ry, rw, rh))

    if not regions:
        regions = [(0, 0, orig_w, orig_h)]

    regions.sort(key=lambda r: r[1])

    all_lines = []
    for r_idx, (rx, ry, rw, rh) in enumerate(regions, 1):
        pad_y = 15
        ry_new = max(0, ry - pad_y)
        rh_new = min(orig_h - ry_new, rh + 2 * pad_y)
        
        pad_x = 30
        rx_new = max(0, rx - pad_x)
        rw_new = min(orig_w - rx_new, rw + 2 * pad_x)

        lines = detect_lines_in_region(binary_masked, rx_new, ry_new, rw_new, rh_new, orig_w, orig_h, r_idx)
        all_lines.extend(lines)

    all_lines.sort(key=lambda b: b['bbox'][1])
    return all_lines, mask_full, binary_masked

# ==================== 3. SHIROREKHA & GLYPH EXTRACTION ====================

def ablate_shirorekha(binary_roi):
    if binary_roi.shape[0] < 10 or binary_roi.shape[1] < 10:
        return binary_roi

    ablated_roi = binary_roi.copy()
    H, W = ablated_roi.shape
    horiz_proj = np.sum(ablated_roi, axis=1)
    
    search_limit = int(H * 0.45)
    if search_limit < 3:
        return ablated_roi
        
    peak_y = np.argmax(horiz_proj[:search_limit])
    if horiz_proj[peak_y] > W * 0.25 * 255:
        thickness = max(2, int(H * 0.06))
        y_start = max(0, peak_y - thickness)
        y_end = min(H, peak_y + thickness + 1)
        ablated_roi[y_start:y_end, :] = 0
        
    return ablated_roi

def detect_line_glyphs(binary, line_dict):
    lx1, ly1, lx2, ly2 = line_dict['bbox']
    roi = binary[ly1:ly2, lx1:lx2]
    if roi.size == 0:
        return []

    ablated_roi = ablate_shirorekha(roi)
    H = ly2 - ly1
    W = lx2 - lx1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ablated_roi, connectivity=8
    )

    min_cc_area = max(6, int(H * 0.5))
    ccs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_cc_area:
            continue
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if cw > W * 0.85 and ch < H * 0.25:
            continue
        ccs.append({
            'local_bbox': (cx, cy, cx + cw, cy + ch),
            'label_id': i,
            'centroid_x': centroids[i][0],
            'area': area,
        })

    if not ccs:
        return []

    ccs.sort(key=lambda c: c['local_bbox'][0])

    glyphs = []
    for cc in ccs:
        cx1, cy1, cx2, cy2 = cc['local_bbox']
        cc_roi = roi[cy1:cy2, cx1:cx2].copy()
        cc_mask = (labels[cy1:cy2, cx1:cx2] == cc['label_id']).astype(np.uint8) * 255
        
        kernel = np.ones((7, 1), np.uint8)
        dilated_cc = cv2.dilate(cc_mask, kernel, iterations=1)
        restored_cc_mask = cv2.bitwise_and(cc_roi, dilated_cc)
        
        poly = extract_polygon_hull(restored_cc_mask, lx1 + cx1, ly1 + cy1, epsilon_factor=0.01)
        
        glyphs.append({
            'bbox': (lx1 + cx1, ly1 + cy1, lx1 + cx2, ly1 + cy2),
            'polygon': poly,
            'centroid_x': lx1 + cc['centroid_x'],
        })

    return glyphs

# ==================== 4. FALLBACK DINO CC WORD SEGMENTATION ====================

def detect_words_and_chars_in_line(binary, line_dict):
    """Fallback word clustering if OCR detection yields no matches for this line."""
    lx1, ly1, lx2, ly2 = line_dict['bbox']
    roi = binary[ly1:ly2, lx1:lx2]
    if roi.size == 0:
        return [], {}

    glyphs = detect_line_glyphs(binary, line_dict)
    if not glyphs:
        # Fallback to single word representing entire line
        line_poly = extract_polygon_hull(roi, lx1, ly1)
        word = {'bbox': (lx1, ly1, lx2, ly2), 'polygon': line_poly, 'text': ""}
        return [word], {0: [{'bbox': (lx1, ly1, lx2, ly2), 'polygon': line_poly}]}

    H = ly2 - ly1
    avg_char_width = np.mean([g['bbox'][2] - g['bbox'][0] for g in glyphs])
    word_gap_threshold = max(8, avg_char_width * 0.6, H * 0.12)

    word_groups = [[0]]
    for i in range(1, len(glyphs)):
        prev_right = glyphs[i - 1]['bbox'][2]
        curr_left = glyphs[i]['bbox'][0]
        gap = curr_left - prev_right
        if gap > word_gap_threshold:
            word_groups.append([i])
        else:
            word_groups[-1].append(i)

    words = []
    chars_by_word = {}
    for wi, group in enumerate(word_groups):
        group_glyphs = [glyphs[i] for i in group]
        wx1 = min(g['bbox'][0] for g in group_glyphs)
        wy1 = min(g['bbox'][1] for g in group_glyphs)
        wx2 = max(g['bbox'][2] for g in group_glyphs)
        wy2 = max(g['bbox'][3] for g in group_glyphs)

        word_roi = binary[wy1:wy2, wx1:wx2]
        word_poly = extract_polygon_hull(word_roi, wx1, wy1, epsilon_factor=0.008)
        
        words.append({
            'bbox': (wx1, wy1, wx2, wy2),
            'polygon': word_poly,
            'text': ""
        })
        chars_by_word[wi] = group_glyphs

    return words, chars_by_word

# ==================== 5. HYBRID HYBRID OCR ALIGNMENTS ====================

def map_paddle_boxes_to_dino_lines(lines_data, paddle_boxes):
    for ld in lines_data:
        ld['paddle_boxes'] = []

    for pbox in paddle_boxes:
        p_coords = pbox['box']
        pxs = [pt[0] for pt in p_coords]
        pys = [pt[1] for pt in p_coords]
        px1, py1, px2, py2 = min(pxs), min(pys), max(pxs), max(pys)
        py_center = (py1 + py2) / 2
        px_center = (px1 + px2) / 2
        
        best_line = None
        best_score = float('inf')
        
        for ld in lines_data:
            lx1, ly1, lx2, ly2 = ld['bbox']
            lh = ly2 - ly1
            
            h_overlap = max(0, min(lx2, px2) - max(lx1, px1))
            if h_overlap <= 0:
                continue
                
            ly_center = (ly1 + ly2) / 2
            v_overlap = max(0, min(ly2, py2) - max(ly1, py1))
            dist = abs(py_center - ly_center)
            
            if dist < lh * 0.75 or v_overlap > 0.3 * min(lh, py2 - py1):
                if dist < best_score:
                    best_score = dist
                    best_line = ld
                    
        if best_line is not None:
            best_line['paddle_boxes'].append(pbox)

def split_paddle_box_into_words(pbox):
    text = pbox['text'].strip()
    pts = pbox['box']
    words = text.split()
    if len(words) <= 1:
        return [{'coords': pts, 'text': text}]
        
    lengths = [len(w) for w in words]
    total_len = sum(lengths) + len(words) - 1
    if total_len == 0:
        return [{'coords': pts, 'text': text}]
        
    # Coordinates mapping
    p1, p2, p3, p4 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2]), np.array(pts[3])
    
    split_words = []
    curr_len = 0
    for w in words:
        t_start = curr_len / total_len
        t_end = (curr_len + len(w)) / total_len
        
        pt_top_start = (1 - t_start) * p1 + t_start * p2
        pt_top_end = (1 - t_end) * p1 + t_end * p2
        
        pt_bottom_start = (1 - t_start) * p4 + t_start * p3
        pt_bottom_end = (1 - t_end) * p4 + t_end * p3
        
        w_quad = [
            [float(pt_top_start[0]), float(pt_top_start[1])],
            [float(pt_top_end[0]), float(pt_top_end[1])],
            [float(pt_bottom_end[0]), float(pt_bottom_end[1])],
            [float(pt_bottom_start[0]), float(pt_bottom_start[1])]
        ]
        
        split_words.append({
            'coords': w_quad,
            'text': w
        })
        curr_len += len(w) + 1
        
    return split_words

def extract_word_polygon(w_quad, binary_img, lx1, ly1, lx2, ly2):
    xs = [pt[0] for pt in w_quad]
    ys = [pt[1] for pt in w_quad]
    wx1, wy1, wx2, wy2 = min(xs), min(ys), max(xs), max(ys)
    
    ix1 = max(lx1, int(wx1))
    iy1 = max(ly1, int(wy1))
    ix2 = min(lx2, int(wx2))
    iy2 = min(ly2, int(wy2))
    
    if ix2 > ix1 and iy2 > iy1:
        word_roi = binary_img[iy1:iy2, ix1:ix2]
        if word_roi.size > 0 and np.sum(word_roi) > 50:
            poly = extract_polygon_hull(word_roi, ix1, iy1, epsilon_factor=0.008)
            if len(poly) >= 3:
                return poly
                
    return [tuple(pt) for pt in w_quad]

def assign_glyphs_to_words(glyphs, split_words):
    chars_by_word = {i: [] for i in range(len(split_words))}
    
    for glyph in glyphs:
        g_cx = glyph['centroid_x']
        
        best_w_idx = None
        min_dist = float('inf')
        
        for idx, w in enumerate(split_words):
            wx1, _, wx2, _ = w['bbox']
            
            if wx1 <= g_cx <= wx2:
                best_w_idx = idx
                break
                
            dist = min(abs(g_cx - wx1), abs(g_cx - wx2))
            if dist < min_dist:
                min_dist = dist
                best_w_idx = idx
                
        if best_w_idx is not None:
            chars_by_word[best_w_idx].append(glyph)
            
    return chars_by_word

# ==================== 6. BINARIZATION & PROCESSING ====================

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

def process_hybrid_image(dino_model, ocr_engine, img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read {img_path}")

    h, w = img.shape[:2]

    # Step 1: DINOv2 feature extraction
    feat_grid, _, _ = extract_patch_features(dino_model, img)

    # Step 2: Cluster features -> text mask
    text_mask, _ = cluster_text_mask(feat_grid, img, n_clusters=3)

    # Step 3: Binarize image
    binary = binarize(img)

    # Step 4: Detect text lines from mask
    lines_data, mask_full, binary_masked = detect_lines_from_mask(text_mask, binary, h, w)

    # Step 5: Run PaddleOCR on full image
    paddle_res = ocr_engine.ocr(img, det=True, rec=True)
    paddle_boxes = []
    if paddle_res is not None and len(paddle_res) > 0 and paddle_res[0] is not None:
        for item in paddle_res[0]:
            paddle_boxes.append({
                'box': item[0],
                'text': item[1][0],
                'conf': item[1][1]
            })

    # Step 6: Map Paddle OCR boxes to DINO Lines
    map_paddle_boxes_to_dino_lines(lines_data, paddle_boxes)

    # Step 7: Build Word and Glyph structures
    total_words = 0
    for ld in lines_data:
        lx1, ly1, lx2, ly2 = ld['bbox']
        
        # Sort matched OCR boxes from left to right
        ld['paddle_boxes'].sort(key=lambda pb: sum([pt[0] for pt in pb['box']]) / 4)
        
        if ld['paddle_boxes']:
            # 1. OCR Split-Word generation
            line_words = []
            line_text_parts = []
            
            for pbox in ld['paddle_boxes']:
                split_w_list = split_paddle_box_into_words(pbox)
                
                for sw in split_w_list:
                    # Bounding box of split word
                    xs = [pt[0] for pt in sw['coords']]
                    ys = [pt[1] for pt in sw['coords']]
                    wx1, wy1, wx2, wy2 = min(xs), min(ys), max(xs), max(ys)
                    
                    # Extract polygon hull inside this word space
                    word_poly = extract_word_polygon(sw['coords'], binary_masked, lx1, ly1, lx2, ly2)
                    
                    line_words.append({
                        'bbox': (wx1, wy1, wx2, wy2),
                        'polygon': word_poly,
                        'text': sw['text']
                    })
                    line_text_parts.append(sw['text'])
            
            # Combine text transcription for the entire line
            ld['words'] = line_words
            ld['text'] = " ".join(line_text_parts)
            total_words += len(line_words)
            
            # 2. Extract DINO line glyphs and associate them to OCR word boxes
            glyphs = detect_line_glyphs(binary_masked, ld)
            ld['chars'] = assign_glyphs_to_words(glyphs, line_words)
            
        else:
            # Fallback to DINO only
            words, chars_by_word = detect_words_and_chars_in_line(binary_masked, ld)
            ld['words'] = words
            ld['chars'] = chars_by_word
            ld['text'] = "" # Empty transcription
            total_words += len(words)

    # Step 8: Detect Damage/Holes
    damage_regions = detect_damage_and_holes(binary, mask_full)

    # Step 9: Output XML and visual validation
    out_xml = output_dir / f"{img_path.stem}.xml"
    generate_pagexml(img_path.name, w, h, lines_data, damage_regions, out_xml)

    # Step 10: Calculate Simulated Human Effort Score (E)
    effort = calculate_human_effort_score(lines_data, binary)

    out_viz = output_dir / f"{img_path.stem}_viz.jpg"
    draw_viz(img, mask_full, lines_data, out_viz)

    return len(lines_data), total_words, effort

# ==================== 7. PAGE-XML GENERATION ====================

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
    pcgts.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation", 
              "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd")
    
    metadata = etree.SubElement(pcgts, "Metadata")
    etree.SubElement(metadata, "Creator").text = "Hybrid DINOv2 + PaddleOCR Pipeline"
    now = datetime.now().isoformat()
    etree.SubElement(metadata, "Created").text = now
    etree.SubElement(metadata, "LastChange").text = now

    page = etree.SubElement(pcgts, "Page",
                            imageFilename=img_name,
                            imageWidth=str(img_w),
                            imageHeight=str(img_h))

    # Group lines by region
    from collections import defaultdict
    regions_map = defaultdict(list)
    for ld in lines_data:
        r_idx = ld.get('region_idx', 1)
        regions_map[r_idx].append(ld)

    if not lines_data:
        region = etree.SubElement(page, "TextRegion", id="r1", type="paragraph")
        etree.SubElement(region, "Coords", points=bbox_to_coords(0, 0, img_w, img_h))

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
            lx1, ly1, lx2, ly2 = ld['bbox']
            
            if ld.get('is_marginalia', False):
                parent_region = etree.SubElement(page, "TextRegion", id=f"r_marg_{r_idx}_{li}", type="marginalia")
                etree.SubElement(parent_region, "Coords", points=bbox_to_coords(lx1, ly1, lx2, ly2))
            else:
                parent_region = region_el
                
            line_el = etree.SubElement(parent_region, "TextLine", id=f"r_{r_idx}_l{li}")
            
            # 1. Coords
            line_poly = ld.get('polygon', [(lx1, ly1), (lx2, ly1), (lx2, ly2), (lx1, ly2)])
            etree.SubElement(line_el, "Coords", points=poly_to_coords(line_poly))
            
            # 2. Baseline
            baseline_y = ly1 + int((ly2 - ly1) * 0.75)
            etree.SubElement(line_el, "Baseline", points=f"{lx1},{baseline_y} {lx2},{baseline_y}")

            # 3. Words
            for wi, wd in enumerate(ld.get('words', []), 1):
                word_poly = wd.get('polygon', [])
                word_el = etree.SubElement(line_el, "Word", id=f"r_{r_idx}_l{li}_w{wi}")
                
                if word_poly and len(word_poly) >= 3:
                    etree.SubElement(word_el, "Coords", points=poly_to_coords(word_poly))
                else:
                    wx1, wy1, wx2, wy2 = wd['bbox']
                    etree.SubElement(word_el, "Coords", points=bbox_to_coords(wx1, wy1, wx2, wy2))

                # Glyphs inside Word
                char_group = wi - 1
                for ci, cd in enumerate(ld.get('chars', {}).get(char_group, []), 1):
                    glyph_poly = cd.get('polygon', [])
                    g_el = etree.SubElement(word_el, "Glyph", id=f"r_{r_idx}_l{li}_w{wi}_g{ci}")
                    if glyph_poly and len(glyph_poly) >= 3:
                        etree.SubElement(g_el, "Coords", points=poly_to_coords(glyph_poly))
                    else:
                        cx1, cy1, cx2, cy2 = cd['bbox']
                        etree.SubElement(g_el, "Coords", points=bbox_to_coords(cx1, cy1, cx2, cy2))

                # TextEquiv of Word
                if wd.get('text'):
                    te = etree.SubElement(word_el, "TextEquiv")
                    etree.SubElement(te, "Unicode").text = wd['text']

            # 4. TextEquiv of TextLine
            if ld.get('text'):
                te = etree.SubElement(line_el, "TextEquiv")
                etree.SubElement(te, "Unicode").text = ld['text']

    # Damage regions
    for di, dd in enumerate(damage_regions, 1):
        noise_el = etree.SubElement(page, "NoiseRegion", id=f"damage_{di}")
        etree.SubElement(noise_el, "Coords", points=poly_to_coords(dd['polygon']))

    tree = etree.ElementTree(pcgts)
    tree.write(str(out_path), pretty_print=True, xml_declaration=True, encoding="utf-8")

# ==================== 8. VISUALIZATION ====================

def draw_viz(img, mask_full, lines_data, out_path):
    viz = img.copy()

    # Semi-transparent overlay of DINO text mask (green tint)
    mask_color = np.zeros_like(viz)
    mask_color[:, :, 1] = mask_full
    viz = cv2.addWeighted(viz, 0.7, mask_color, 0.3, 0)

    # BGR harmonized color scheme
    # DINO lines: multi-color borders
    colors = [
        (0,255,0), (255,255,0), (0,255,255), (255,0,255),
        (128,255,0), (0,128,255), (255,128,0), (0,255,128)
    ]

    for li, ld in enumerate(lines_data):
        c = colors[li % len(colors)]
        
        # Draw line polygon
        line_poly = ld.get('polygon', [])
        if line_poly and len(line_poly) >= 3:
            pts = np.array(line_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz, [pts], isClosed=True, color=c, thickness=2)
        else:
            lx1, ly1, lx2, ly2 = ld['bbox']
            cv2.rectangle(viz, (lx1, ly1), (lx2, ly2), c, 2)
            
        lx1, ly1 = ld['bbox'][0], ld['bbox'][1]
        cv2.putText(viz, f"L{li+1}", (lx1, ly1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        # Draw OCR-aligned Words (Blue)
        for wd in ld.get('words', []):
            word_poly = wd.get('polygon', [])
            if word_poly and len(word_poly) >= 3:
                pts = np.array(word_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(viz, [pts], isClosed=True, color=(255, 120, 0), thickness=1)
            else:
                wx1, wy1, wx2, wy2 = wd['bbox']
                cv2.rectangle(viz, (int(wx1), int(wy1)), (int(wx2), int(wy2)), (255, 120, 0), 1)

        # Draw DINO Glyphs (Red)
        for char_list in ld.get('chars', {}).values():
            for cd in char_list:
                glyph_poly = cd.get('polygon', [])
                if glyph_poly and len(glyph_poly) >= 3:
                    pts = np.array(glyph_poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(viz, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
                else:
                    cx1, cy1, cx2, cy2 = cd['bbox']
                    cv2.rectangle(viz, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)

    cv2.imwrite(str(out_path), viz)

# ==================== 9. MAIN PIPELINE ====================

def main():
    parser = argparse.ArgumentParser(description="Hybrid DINOv2 + PaddleOCR Pipeline")
    parser.add_argument("--input", required=True, help="Input image folder or path")
    parser.add_argument("--output", required=True, help="Output results folder")
    parser.add_argument("--gpu", action="store_true", help="Try using GPU for PaddleOCR")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading DINOv2 model on {DEVICE}...")
    dino_model = load_dino_model()
    print("DINOv2 loaded.")

    print("Initializing PaddleOCR...")
    # Default to args.gpu, if GPU initialization fails it falls back to CPU automatically
    ocr_engine = init_paddle_ocr(use_gpu=args.gpu)

    if in_path.is_file():
        images = [in_path]
    else:
        images = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")))

    print(f"Processing {len(images)} images...")
    total_lines = 0
    total_words = 0
    total_effort = 0

    pbar = tqdm(images, desc="Processing")
    for img_path in pbar:
        try:
            lines_count, words_count, effort = process_hybrid_image(
                dino_model, ocr_engine, img_path, out_path
            )
            total_lines += lines_count
            total_words += words_count
            total_effort += effort['score']
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")

    print(f"\nDone! Processed {len(images)} images.")
    print(f"Total Lines: {total_lines}, Total Words: {total_words}")
    print(f"Simulated Human Effort Score (E): {total_effort:.1f}")
    print(f"Outputs written to: {out_path}")

if __name__ == "__main__":
    main()
