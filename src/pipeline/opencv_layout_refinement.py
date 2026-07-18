import cv2
import numpy as np

def detect_page_frame(img_bgr):
    """Finds the palm leaf boundary."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = img_bgr.shape[:2]
        return {'bbox': [0, 0, w, h], 'polygon': []}, np.ones((h, w), dtype=np.uint8) * 255
        
    leaf_contour = max(contours, key=cv2.contourArea)
    
    epsilon = 0.005 * cv2.arcLength(leaf_contour, True)
    approx = cv2.approxPolyDP(leaf_contour, epsilon, True)
    
    x, y, w, h = cv2.boundingRect(leaf_contour)
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [leaf_contour], -1, 255, -1)
    
    return {
        'bbox': [int(x), int(y), int(x+w), int(y+h)],
        'polygon': approx.reshape(-1, 2).tolist()
    }, mask

def detect_damage_holes(img_bgr, leaf_mask):
    """Finds binder string holes inside the leaf.
    
    Real palm-leaf binder holes are:
    - Large (500-8000 px area, typically 20-50px diameter)
    - Nearly perfectly circular (circularity > 0.85)
    - Roughly square aspect ratio (width ≈ height)
    Previous thresholds (area>50, circ>0.6) were far too loose and
    picked up every ink blob and letter stroke as 'damage'.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    inside_leaf = cv2.bitwise_and(thresh, leaf_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inside_leaf = cv2.morphologyEx(inside_leaf, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(inside_leaf, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 8000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.85:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(w) / h if h > 0 else 0
                if 0.5 < aspect < 2.0:
                    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                    holes.append({
                        'bbox': [int(x), int(y), int(x+w), int(y+h)],
                        'polygon': approx.reshape(-1, 2).tolist()
                    })
    return holes

def detect_text_regions(binary_text_mask):
    """Smears text to find large contiguous text blocks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    smeared = cv2.dilate(binary_text_mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Ignore tiny noise
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'area': float(area)
            })
    return regions

def classify_marginalia(text_regions, img_w):
    """Separates main text regions from marginalia."""
    if not text_regions:
        return [], []
        
    regions = sorted(text_regions, key=lambda x: x['area'], reverse=True)
    
    main_regions = []
    marginalia_regions = []
    
    main_block = regions[0]
    main_regions.append(main_block)
    
    for reg in regions[1:]:
        if reg['area'] < 0.3 * main_block['area']:
            x1, y1, x2, y2 = reg['bbox']
            if x1 < img_w * 0.2 or x2 > img_w * 0.8:
                marginalia_regions.append(reg)
            else:
                main_regions.append(reg)
        else:
            main_regions.append(reg)
            
    main_out = [{'bbox': r['bbox']} for r in main_regions]
    marg_out = [{'bbox': r['bbox']} for r in marginalia_regions]
    return main_out, marg_out
