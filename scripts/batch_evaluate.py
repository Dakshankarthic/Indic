"""
Batch Surya OCR Evaluation Script
Runs Surya OCR on all test images and computes CER, WER, and overall accuracy.
For AutoAnn-Indic Challenge submission.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import os
import time
import cv2
import numpy as np
import Levenshtein
from PIL import Image
import re

# Force llama.cpp backend before importing Surya
os.environ['SURYA_INFERENCE_BACKEND'] = 'llamacpp'
os.environ['VLLM_GPU_TYPE'] = '2070'
os.environ['LLAMA_CPP_BINARY'] = r'd:\indic_challenge\llama_bin\llama-server.exe'

from surya.recognition import RecognitionPredictor

def extract_text_from_result(result):
    """Extract plain text from Surya OCR result blocks."""
    lines = []
    if hasattr(result, 'blocks'):
        for block in result.blocks:
            if hasattr(block, 'html'):
                text = re.sub(r'<[^>]+>', '', block.html).strip()
                if text:
                    lines.append(text)
            elif hasattr(block, 'lines'):
                for line in block.lines:
                    if line.text.strip():
                        lines.append(line.text.strip())
    return '\n'.join(lines)

def clean_text(text):
    """Normalize OCR output before CER/WER calculation."""
    # Collapse multiple spaces, strip surrounding whitespace,
    # and remove stray punctuation that never appears in ground truth.
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[«»“”]', '', text)
    return text

def compute_wer(pred_text, gt_text):
    """Compute Word Error Rate."""
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return Levenshtein.distance(pred_words, gt_words) / len(gt_words)

def compute_cer(pred_text, gt_text):
    """Compute Character Error Rate."""
    if len(gt_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    return Levenshtein.distance(pred_text, gt_text) / len(gt_text)

def deskew_image(img_cv):
    """Detect and correct image rotation using minAreaRect."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return img_cv
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_cv, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(pil_img):
    """Apply enhanced deskew, CLAHE, noise reduction, adaptive threshold,
    morphological closing, and aggressive upscaling for better OCR."""
    # Convert to OpenCV BGR
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ---- Deskew -------------------------------------------------
    img_cv = deskew_image(img_cv)

    # ---- Contrast Enhancement (CLAHE) ---------------------------
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ---- Noise Reduction (Bilateral Filter) --------------------
    # Reduces speckle while preserving edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # ---- Adaptive Threshold ------------------------------------
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)

    # ---- Morphological Closing (to mend broken strokes) -------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---- Aggressive Upscaling (6×) -----------------------------
    upscaled = cv2.resize(thresh, None, fx=6.0, fy=6.0,
                         interpolation=cv2.INTER_CUBIC)

    # Convert back to 3‑channel BGR (Surya expects color images)
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    # Return as PIL Image
    return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))

def main():
    json_path = r'C:\Users\DK11\Downloads\hindi_ocr.json'
    img_dir = r'd:\indic_challenge\org_img-20260701T115420Z-3-001\org_img'
    results_path = r'd:\indic_challenge\evaluation_results.json'

    # Load ground truth
    with open(json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    print(f"Total pages in ground truth: {len(gt_data)}")

    # Build GT map: page_name -> text
    gt_map = {}
    for item in gt_data:
        if item.get('images'):
            fname = os.path.basename(item['images'][0])
            page_name = fname.replace('.jpg', '')
            gt_map[page_name] = item.get('output', '').strip()

    print(f"Ground truth entries: {len(gt_map)}")

    # Initialize Surya with deterministic decoding
    print("Initializing Surya OCR (llamacpp backend, deterministic)...")
    rec_predictor = RecognitionPredictor()

    BATCH_SIZE = 4
    page_names = sorted(gt_map.keys(), key=lambda x: int(x.replace('page_', '')))

    all_cer = []
    all_wer = []
    all_results = []
    total_gt_chars = 0
    total_edit_dist_chars = 0
    total_gt_words = 0
    total_edit_dist_words = 0

    start_time = time.time()

    for batch_start in range(0, len(page_names), BATCH_SIZE):
        batch_names = page_names[batch_start:batch_start + BATCH_SIZE]
        batch_images = []
        batch_gt = []
        valid_names = []

        for pname in batch_names:
            img_path = os.path.join(img_dir, f'{pname}.jpg')
            if not os.path.exists(img_path):
                print(f"  SKIP: {pname}.jpg not found")
                continue

            pil_img = Image.open(img_path)
            pil_img = preprocess_image(pil_img)

            batch_images.append(pil_img)
            batch_gt.append(gt_map[pname])
            valid_names.append(pname)

        if not batch_images:
            continue

        try:
            predictions = rec_predictor(batch_images)
        except Exception as e:
            print(f"  ERROR on batch starting {batch_names[0]}: {e}")
            continue

        for i, (pname, gt_text) in enumerate(zip(valid_names, batch_gt)):
            pred_text = extract_text_from_result(predictions[i])
            pred_text = clean_text(pred_text)

            cer = compute_cer(pred_text, gt_text)
            wer = compute_wer(pred_text, gt_text)

            all_cer.append(cer)
            all_wer.append(wer)

            total_gt_chars += len(gt_text)
            total_edit_dist_chars += Levenshtein.distance(pred_text, gt_text)

            gt_words = gt_text.split()
            pred_words = pred_text.split()
            total_gt_words += len(gt_words)
            total_edit_dist_words += Levenshtein.distance(pred_words, gt_words)

            all_results.append({
                'page': pname,
                'cer': round(cer, 4),
                'wer': round(wer, 4),
                'pred_chars': len(pred_text),
                'gt_chars': len(gt_text),
                'pred_lines': len(pred_text.split('\n')) if pred_text else 0,
            })

        processed = batch_start + len(batch_names)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(page_names) - processed) / rate if rate > 0 else 0

        avg_cer = sum(all_cer) / len(all_cer) if all_cer else 0
        avg_wer = sum(all_wer) / len(all_wer) if all_wer else 0

        print(f"  [{processed}/{len(page_names)}] "
              f"CER={avg_cer*100:.1f}% WER={avg_wer*100:.1f}% "
              f"Rate={rate:.1f}pg/s ETA={eta/60:.0f}min")

    total_time = time.time() - start_time

    micro_cer = total_edit_dist_chars / total_gt_chars if total_gt_chars > 0 else 0
    micro_wer = total_edit_dist_words / total_gt_words if total_gt_words > 0 else 0
    macro_cer = sum(all_cer) / len(all_cer) if all_cer else 0
    macro_wer = sum(all_wer) / len(all_wer) if all_wer else 0
    accuracy = (1 - micro_cer) * 100

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Pages processed:     {len(all_cer)} / {len(page_names)}")
    print(f"Total time:          {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"")
    print(f"Micro-averaged CER:  {micro_cer*100:.2f}%")
    print(f"Micro-averaged WER:  {micro_wer*100:.2f}%")
    print(f"Macro-averaged CER:  {macro_cer*100:.2f}%")
    print(f"Macro-averaged WER:  {macro_wer*100:.2f}%")
    print(f"Overall Accuracy:    {accuracy:.2f}%")
    print(f"{'='*60}")

    summary = {
        'model': 'Surya OCR v2 (llamacpp backend, deterministic, enhanced preprocessing)',
        'gpu': 'NVIDIA RTX 2070 8GB',
        'total_pages': len(all_cer),
        'total_time_seconds': round(total_time, 1),
        'micro_cer': round(micro_cer, 4),
        'micro_wer': round(micro_wer, 4),
        'macro_cer': round(macro_cer, 4),
        'macro_wer': round(macro_wer, 4),
        'overall_accuracy': round(accuracy, 2),
        'per_page_results': all_results
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {results_path}")

if __name__ == '__main__':
    main()