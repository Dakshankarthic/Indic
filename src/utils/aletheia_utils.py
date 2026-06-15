import subprocess
import cv2
import numpy as np
import os

def open_in_aletheia(image_path, xml_path, aletheia_path=r"C:\Users\DK11\Downloads\Aletheia_4.1.1109\Aletheia 4.1\Aletheia.exe"):
    """
    Opens the generated PAGE-XML and image in Aletheia for visual inspection.
    """
    if not os.path.exists(xml_path):
        print(f"[Aletheia] Error: XML file not found at {xml_path}")
        return False
    
    try:
        # Aletheia.exe accepts image and xml paths as arguments
        cmd = [aletheia_path, str(image_path), str(xml_path)]
        print(f"\n[Aletheia] Launching Aletheia: {' '.join(cmd)}")
        subprocess.Popen(cmd)
        return True
    except Exception as e:
        print(f"\n[Aletheia] Failed to launch Aletheia: {e}")
        print("[Aletheia] Please ensure Aletheia.exe is accessible at the provided path.")
        return False

def calculate_human_effort_score(lines_data, binary_img, 
                                 w_poly=1.0, w_frag=2.0, w_fn=5.0):
    """
    Simulates Human Effort Score (E) to correct annotations.
    E = (w_poly * V) + (w_frag * F) + (w_fn * M)
    """
    total_vertices = 0
    total_fragments = 0
    total_missing = 0

    h, w = binary_img.shape
    
    # Create a mask of our detected areas to find missing ink
    detected_mask = np.zeros_like(binary_img)

    for line in lines_data:
        lx1, ly1, lx2, ly2 = line['bbox']
        
        # We output simple bounding boxes (4 vertices)
        # To correct a tight polygon around curved text, a human adds vertices.
        line_vertices = 4
        
        # Check fragmentation within the line box
        roi = binary_img[ly1:ly2, lx1:lx2]
        if roi.size > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
            # Count small noise components (fragments) that might need manual deletion
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 10: # Tiny fragment
                    total_fragments += 1
                elif area > 100:
                    # Complex strokes need more polygon vertices to trace tightly
                    line_vertices += 2
                    
        total_vertices += line_vertices
        
        # Mark as detected
        cv2.rectangle(detected_mask, (lx1, ly1), (lx2, ly2), 255, -1)

    # Calculate Missing Regions (False Negatives)
    # Ink pixels that fall OUTSIDE our detected line bounding boxes
    missed_ink = cv2.bitwise_and(binary_img, cv2.bitwise_not(detected_mask))
    
    # Connected components on missed ink
    num_missed, _, missed_stats, _ = cv2.connectedComponentsWithStats(missed_ink, connectivity=8)
    for i in range(1, num_missed):
        area = missed_stats[i, cv2.CC_STAT_AREA]
        # Only count significant missing chunks
        if area > 50:
            total_missing += 1

    # Calculate final E score
    E = (w_poly * total_vertices) + (w_frag * total_fragments) + (w_fn * total_missing)
    
    return {
        "score": E,
        "vertices_penalty": total_vertices,
        "fragments_penalty": total_fragments,
        "missing_penalty": total_missing
    }
