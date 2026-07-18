import argparse
import Levenshtein
import glob
import os
import json
import xml.etree.ElementTree as ET

def parse_page_xml_text(xml_path):
    """Extracts all Unicode text from a PAGE-XML file, ordered by TextLine."""
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = ET.parse(xml_path)
    text_content = []
    
    # Extract from TextLine to ensure we get line-level text
    for line in tree.findall('.//pc:TextLine', ns):
        # We look for the direct TextEquiv of the TextLine
        text_equiv = line.find('./pc:TextEquiv/pc:Unicode', ns)
        if text_equiv is not None and text_equiv.text:
            text_content.append(text_equiv.text.strip())
            
    return "\n".join(text_content)

def calculate_metrics(pred_text, gt_text):
    """Calculates CER, WER, and Accuracy."""
    # Handle empty strings
    if not gt_text and not pred_text:
        return 0.0, 0.0, 1.0
    if not gt_text:
        return 1.0, 1.0, 0.0
        
    cer = Levenshtein.distance(pred_text, gt_text) / max(len(gt_text), 1)
    
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    wer = Levenshtein.distance(pred_words, gt_words) / max(len(gt_words), 1)
    
    accuracy = max(0.0, 1.0 - cer)
    
    return cer, wer, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CER and WER for PAGE-XML")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted XMLs")
    parser.add_argument("--gt_json", required=False, help="Path to ground truth JSON file")
    parser.add_argument("--gt_dir", required=False, help="Directory containing ground truth XMLs")
    args = parser.parse_args()
    
    total_cer, total_wer, total_acc = 0, 0, 0
    count = 0
    
    print("Evaluating OCR Metrics...")
    
    if args.gt_json:
        with open(args.gt_json, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            
        # Map basename (e.g. page_1) to ground truth text
        gt_map = {}
        for item in gt_data:
            if item.get("images") and len(item["images"]) > 0:
                basename = os.path.splitext(os.path.basename(item["images"][0]))[0]
                gt_map[basename] = item.get("output", "").strip()
                
        pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.xml")))
        
        for pred_path in pred_files:
            basename = os.path.splitext(os.path.basename(pred_path))[0]
            if basename in gt_map:
                pred_text = parse_page_xml_text(pred_path).strip()
                gt_text = gt_map[basename]
                
                cer, wer, acc = calculate_metrics(pred_text, gt_text)
                total_cer += cer
                total_wer += wer
                total_acc += acc
                count += 1
    elif args.gt_dir:
        pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.xml")))
        for pred_path in pred_files:
            basename = os.path.basename(pred_path)
            gt_path = os.path.join(args.gt_dir, basename)
            if os.path.exists(gt_path):
                pred_text = parse_page_xml_text(pred_path).strip()
                gt_text = parse_page_xml_text(gt_path).strip()
                
                cer, wer, acc = calculate_metrics(pred_text, gt_text)
                total_cer += cer
                total_wer += wer
                total_acc += acc
                count += 1
    else:
        print("Must provide either --gt_json or --gt_dir")
        exit(1)
            
    if count > 0:
        print(f"\nFinal Validation Results over {count} images:")
        print(f"Overall Accuracy : {(total_acc/count)*100:.2f}%")
        print(f"CER              : {(total_cer/count)*100:.2f}%")
        print(f"WER              : {(total_wer/count)*100:.2f}%")
    else:
        print("No matching ground truth found! Cannot compute CER/WER.")
