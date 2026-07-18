import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import os
import time
import Levenshtein
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

def main():
    json_path = r'C:\Users\DK11\Downloads\hindi_ocr.json'
    page_name = 'page_100'
    img_path = fr'd:\indic_challenge\org_img-20260701T115420Z-3-001\org_img\{page_name}.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: Could not find image at {img_path}")
        return

    # Load Ground Truth
    with open(json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    gt_text = ""
    for item in gt_data:
        if item.get('images') and f'{page_name}.jpg' in item['images'][0]:
            gt_text = item.get('output', '').strip()
            break
            
    if not gt_text:
        print(f"Warning: No ground truth found for {page_name}")
        
    print(f"Starting Surya OCR test on {page_name}...")
    t0 = time.time()
    
    # Initialize Predictors (this will start Docker/vLLM backend)
    print("Loading models (this may take a few minutes if pulling docker image)...")
    rec_predictor = RecognitionPredictor()
    
    img = Image.open(img_path)
    
    print("Running detection and recognition...")
    predictions = rec_predictor([img])
    
    surya_lines = []
    if predictions and predictions[0] and hasattr(predictions[0], 'blocks'):
        for block in predictions[0].blocks:
            if hasattr(block, 'html'):
                import re
                text = re.sub(r'<[^>]+>', '', block.html).strip() # Strip HTML tags
                if text: surya_lines.append(text)
            elif hasattr(block, 'lines'):
                for line in block.lines:
                    if line.text.strip():
                        surya_lines.append(line.text.strip())
    elif predictions and predictions[0] and hasattr(predictions[0], 'text_lines'):
        for line in predictions[0].text_lines:
            if line.text.strip():
                surya_lines.append(line.text.strip())

    surya_text = '\n'.join(surya_lines)
    
    if gt_text:
        # Character Error Rate
        cer = Levenshtein.distance(surya_text, gt_text) / max(len(gt_text), 1)
        
        # Word Error Rate
        surya_words = surya_text.split()
        gt_words = gt_text.split()
        wer = Levenshtein.distance(surya_words, gt_words) / max(len(gt_words), 1)
    else:
        cer = 0.0
        wer = 0.0
        
    elapsed = time.time() - t0
    print(f'\n--- RESULTS ---')
    print(f'Surya: {len(surya_text)} chars, {len(surya_lines)} lines')
    print(f'CER: {cer*100:.1f}%')
    print(f'WER: {wer*100:.1f}%')
    print(f'Time: {elapsed:.1f}s')
    print(f'\nFirst 5 lines:')
    for l in surya_lines[:5]:
        print(f'  {l[:100]}')

if __name__ == '__main__':
    main()
