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

def create_page_xml(img_path, img_w, img_h, regions_dict, out_path):
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
        # Use the largest page frame
        largest_frame = max(regions_dict['page_frame'], key=lambda p: cv2.contourArea(np.array(p)))
        Border = etree.SubElement(Page, "{%s}Border" % namespace)
        Coords = etree.SubElement(Border, "{%s}Coords" % namespace, points=points_to_string(largest_frame))

    # Text Regions
    for i, poly in enumerate(regions_dict['text_regions']):
        TextRegion = etree.SubElement(Page, "{%s}TextRegion" % namespace, id=f"text_region_{i}", type="text_region")
        etree.SubElement(TextRegion, "{%s}Coords" % namespace, points=points_to_string(poly))
        
        # In PAGE-XML, TextLines are usually inside TextRegions. 
        # For simplicity, if we don't have mapping, we can put TextLines at Page level or group them.
        # Here we map TextLines that fall inside this TextRegion.
        # A simple center-point check:
        for j, line_poly in enumerate(regions_dict['text_lines']):
            line_pts = np.array(line_poly)
            cx, cy = np.mean(line_pts, axis=0)
            if cv2.pointPolygonTest(np.array(poly, dtype=np.float32), (float(cx), float(cy)), False) >= 0:
                TextLine = etree.SubElement(TextRegion, "{%s}TextLine" % namespace, id=f"line_{i}_{j}")
                etree.SubElement(TextLine, "{%s}Coords" % namespace, points=points_to_string(line_poly))

    # Marginalia
    for i, poly in enumerate(regions_dict['marginalia']):
        TextRegion = etree.SubElement(Page, "{%s}TextRegion" % namespace, id=f"marginalia_{i}", type="marginalia")
        etree.SubElement(TextRegion, "{%s}Coords" % namespace, points=points_to_string(poly))

    # Illustrations
    for i, poly in enumerate(regions_dict['illustrations']):
        GraphicRegion = etree.SubElement(Page, "{%s}GraphicRegion" % namespace, id=f"illustration_{i}")
        etree.SubElement(GraphicRegion, "{%s}Coords" % namespace, points=points_to_string(poly))

    # Damage/Holes (NoiseRegion)
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

        # Preprocess
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Infer
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(img_tensor)
            else:
                outputs = model(img_tensor)
                
            probs = torch.sigmoid(outputs[0]).float().cpu().numpy()

        # Process and Refine
        regions = process_unet_outputs(probs, w, h)

        # Export XML
        out_xml_path = out_dir / f"{img_path.stem}.xml"
        create_page_xml(str(img_path), w, h, regions, str(out_xml_path))

    print(f"Inference complete! PAGE-XML files saved to {out_dir}")

if __name__ == "__main__":
    main()
