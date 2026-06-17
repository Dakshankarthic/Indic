import xml.etree.ElementTree as ET
import numpy as np
import cv2
import argparse
from pathlib import Path

def parse_page_xml(xml_path):
    """Parse PAGE-XML and return polygons for TextRegions."""
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = ET.parse(xml_path)
    regions = []
    
    for region in tree.findall('.//pc:TextRegion', ns):
        coords_str = region.find('pc:Coords', ns).attrib['points']
        pts = np.array([[int(p.split(',')[0]), int(p.split(',')[1])] for p in coords_str.split(' ')])
        regions.append(pts)
        
    return regions

def calculate_iou(poly1, poly2, img_shape=(3000, 3000)):
    """Calculate Intersection over Union (IoU) of two polygons."""
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    
    cv2.fillPoly(mask1, [poly1], 1)
    cv2.fillPoly(mask2, [poly2], 1)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0

def estimate_human_effort(pred_xml, gt_xml):
    """
    Estimates the Human Effort (E) score.
    In PRImA layout evaluation, effort is penalized for:
    1. Missing regions
    2. False positive regions
    3. Jagged polygons (requiring vertex deletion/movement)
    
    A lower score is better (0 = Perfect).
    """
    pred_regions = parse_page_xml(pred_xml)
    gt_regions = parse_page_xml(gt_xml)
    
    effort_score = 0.0
    
    print(f"Evaluating {Path(pred_xml).name} against Ground Truth...")
    print(f"Predicted Regions: {len(pred_regions)} | Ground Truth Regions: {len(gt_regions)}")
    
    region_diff = abs(len(pred_regions) - len(gt_regions))
    effort_score += region_diff * 50.0  # 50 clicks penalty for entirely missed/extra region
    
    for p_poly in pred_regions:
        best_iou = 0
        best_gt = None
        for g_poly in gt_regions:
            iou = calculate_iou(p_poly, g_poly)
            if iou > best_iou:
                best_iou = iou
                best_gt = g_poly
                
        if best_iou < 0.5:
            effort_score += 30.0 
        else:
            vertex_penalty = abs(len(p_poly) - len(best_gt)) * 0.5 # 0.5 clicks per vertex adjustment
            effort_score += vertex_penalty
            
            iou_penalty = (1.0 - best_iou) * 100.0
            effort_score += iou_penalty
            
    print(f"\nEstimated Human Effort Score: {effort_score:.2f} (Lower is better)")
    if effort_score < 10:
        print("Rating: EXCEPTIONAL (Negligible human correction needed)")
    elif effort_score < 50:
        print("Rating: GREAT (Minor adjustments needed)")
    else:
        print("Rating: NEEDS WORK (Significant human editing required)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Human Effort Score")
    parser.add_argument("--pred", required=True, help="Path to Predicted PAGE-XML")
    parser.add_argument("--gt", required=True, help="Path to Ground Truth PAGE-XML")
    args = parser.parse_args()
    
    estimate_human_effort(args.pred, args.gt)
