import cv2
import numpy as np

def detect_page_frame(img_bgr):
    """Finds the palm leaf boundary."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Palm leaves are usually lighter than the dark scanner background.
    # Otsu's thresholding to separate leaf from background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = img_bgr.shape[:2]
        return {'bbox': [0, 0, w, h], 'polygon': []}, np.ones((h, w), dtype=np.uint8) * 255
        
    # Assume largest contour is the leaf
    leaf_contour = max(contours, key=cv2.contourArea)
    
    # Simplify polygon
    epsilon = 0.005 * cv2.arcLength(leaf_contour, True)
    approx = cv2.approxPolyDP(leaf_contour, epsilon, True)
    
    x, y, w, h = cv2.boundingRect(leaf_contour)
    
    # Create mask of the leaf for hole detection
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
    # The holes are dark (background) inside the bright leaf.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Only look inside the leaf
    inside_leaf = cv2.bitwise_and(thresh, leaf_mask)
    
    # Morphological close to merge nearby noise but keep real holes intact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inside_leaf = cv2.morphologyEx(inside_leaf, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(inside_leaf, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Real string holes are large — at least 500px area
        if 500 < area < 8000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # Must be very circular (real holes are punched, not torn)
            if circularity > 0.85:
                x, y, w, h = cv2.boundingRect(cnt)
                # Aspect ratio check: holes are roughly square
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
    # Binary mask is assumed to have text as 255 (white)
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
        
    # Sort regions by area
    regions = sorted(text_regions, key=lambda x: x['area'], reverse=True)
    
    main_regions = []
    marginalia_regions = []
    
    main_block = regions[0]
    main_regions.append(main_block)
    
    for reg in regions[1:]:
        # If it's much smaller than main block
        if reg['area'] < 0.3 * main_block['area']:
            x1, y1, x2, y2 = reg['bbox']
            # And it's near the edges (left or right 20%)
            if x1 < img_w * 0.2 or x2 > img_w * 0.8:
                marginalia_regions.append(reg)
            else:
                main_regions.append(reg)
        else:
            main_regions.append(reg)
            
    # Clean up the output to match DINO format (just bbox)
    main_out = [{'bbox': r['bbox']} for r in main_regions]
    marg_out = [{'bbox': r['bbox']} for r in marginalia_regions]
    return main_out, marg_out
