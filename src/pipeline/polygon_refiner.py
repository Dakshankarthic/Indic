import cv2
import numpy as np

def refine_mask_to_polygons(binary_mask, min_area=100, epsilon_factor=0.002):
    """
    Takes a binary mask (H, W), applies morphological cleanup, 
    and returns a list of tight polygons using cv2.approxPolyDP.
    This minimizes the Human Effort (E) score by reducing redundant vertices.
    """
    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Fill small gaps
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    # Remove pixel noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # Shrink wrap: small epsilon for tight boundary
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Ensure PAGE-XML validity (min 4 vertices)
        if len(approx) < 3:
            continue
            
        if len(approx) == 3:
            # Add a fake 4th vertex if it's a triangle to make PAGE-XML happy if needed
            # For now, just accept it, PAGE-XML allows 3+ points for a polygon.
            pass
            
        # Flatten to list of [x, y]
        poly = approx.reshape(-1, 2).tolist()
        polygons.append(poly)
        
    return polygons

def process_unet_outputs(unet_probs, w, h):
    """
    unet_probs: (6, H, W) numpy array of probabilities [0, 1]
    w, h: Original image dimensions
    Returns a dictionary of polygons for each class
    """
    # Resize probabilities back to original image size
    probs_resized = np.zeros((6, h, w), dtype=np.float32)
    for c in range(6):
        channel_prob = np.ascontiguousarray(unet_probs[c])
        probs_resized[c] = cv2.resize(channel_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        
    # Thresholding
    binary_masks = (probs_resized > 0.5).astype(np.uint8) * 255
    
    results = {
        'text_regions': refine_mask_to_polygons(binary_masks[0]),
        'marginalia': refine_mask_to_polygons(binary_masks[1]),
        'illustrations': refine_mask_to_polygons(binary_masks[2]),
        'page_frame': refine_mask_to_polygons(binary_masks[3], min_area=5000, epsilon_factor=0.005),
        'damage_holes': refine_mask_to_polygons(binary_masks[4]),
        'text_lines': refine_mask_to_polygons(binary_masks[5])
    }
    
    return results
