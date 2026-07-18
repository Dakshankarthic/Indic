import json
import os
import Levenshtein

def compute_wer(pred_text, gt_text):
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return Levenshtein.distance(pred_words, gt_words) / len(gt_words)

def compute_cer(pred_text, gt_text):
    if len(gt_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    return Levenshtein.distance(pred_text, gt_text) / len(gt_text)

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[«»“”]', '', text)
    return text

def main():
    json_path = r'C:\Users\DK11\Downloads\hindi_ocr.json'
    cache_path = r'd:\indic_challenge\surya_text_cache.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    total_gt_chars = 0
    total_edit_dist_chars = 0
    total_gt_words = 0
    total_edit_dist_words = 0
    
    count = 0
    
    for item in gt_data:
        if item.get('images'):
            fname = os.path.basename(item['images'][0])
            page_name = fname.replace('.jpg', '')
            gt_text = item.get('output', '').strip()
            
            if page_name in cache_data:
                pred_text = cache_data[page_name]
                pred_text = clean_text(pred_text)
                
                total_gt_chars += len(gt_text)
                total_edit_dist_chars += Levenshtein.distance(pred_text, gt_text)
                
                gt_words = gt_text.split()
                pred_words = pred_text.split()
                total_gt_words += len(gt_words)
                total_edit_dist_words += Levenshtein.distance(pred_words, gt_words)
                
                count += 1
                
    if count == 0:
        print("No matching files found between cache and GT.")
        return
        
    micro_cer = total_edit_dist_chars / total_gt_chars if total_gt_chars > 0 else 0
    micro_wer = total_edit_dist_words / total_gt_words if total_gt_words > 0 else 0
    
    print(f"Evaluated {count} pages from Ground Truth:")
    print(f"Overall Accuracy: {(1 - micro_cer) * 100:.2f}%")
    print(f"CER: {micro_cer * 100:.2f}%")
    print(f"WER: {micro_wer * 100:.2f}%")

if __name__ == '__main__':
    main()
