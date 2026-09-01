import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.append(r"d:\indic_challenge")
sys.path.append(r"d:\indic_challenge\src\pipeline")

from src.pipeline.pipeline_master import (
    dino_model, extract_patch_features, is_printed_page,
    cluster_text_mask, binarize, detect_illustrations_from_binary,
    detect_lines_from_mask, process_page
)

def run_instant_inference(image_path, output_dir=r"d:\indic_challenge\pitch_results"):
    os.makedirs(output_dir, exist_ok=True)
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {image_path}")
        return None

    print(f"[*] Processing Pitch Image: {img_path.name}")
    xml_path, visual_path, stats = process_page(
        str(img_path),
        output_xml_dir=output_dir,
        output_visual_dir=output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[+] Output Visual Overlay: {visual_path}")
    print(f"[+] Output PAGE-XML: {xml_path}")
    print(f"[+] Detection Summary: {stats}")
    return visual_path, xml_path, stats

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_instant_inference(sys.argv[1])
    else:
        print("Ready for image input.")
