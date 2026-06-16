"""Full test: Run all 10 test images through improved PaddleOCR pipeline.

Key improvements:
- rec_image_shape='3, 48, 320' (avoids aspect distortion)
- drop_score=0.5 (filters garbage)
- det_db_thresh reduced for ancient text
"""
import os
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
from paddleocr import PaddleOCR


def preprocess_for_ocr(crop):
    """Gentle preprocessing — preserve gray-level ink information."""
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    denoised = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    min_val, max_val = enhanced.min(), enhanced.max()
    if max_val > min_val:
        normalized = ((enhanced.astype(np.float32) - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = enhanced
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def main():
    print("Initializing improved PaddleOCR (Hindi)...")
    ocr = PaddleOCR(
        use_angle_cls=False, lang='hi', use_gpu=False, enable_mkldnn=False,
        show_log=False, drop_score=0.3, rec_image_shape='3, 48, 320',
        rec_batch_num=1,
    )

    test_dir = "test_10_images"
    images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])

    if not images:
        print("No images found!")
        return

    out_path = "paddle_cpu_output_v2.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for img_name in images:
            img_path = os.path.join(test_dir, img_name)
            out.write(f"\n{'='*60}\n")
            out.write(f"Processing {img_name} with improved PaddleOCR...\n")
            print(f"Processing {img_name}...")

            img = cv2.imread(img_path)
            if img is None:
                out.write("Failed to load image!\n")
                continue

            h, w = img.shape[:2]

            # Full page detection + recognition with improved params
            res = ocr.ocr(img_path, det=True, rec=True)

            if res is None or len(res) == 0 or res[0] is None:
                out.write("No text detected on page.\n")
                continue

            lines = res[0]
            out.write(f"Detected {len(lines)} text lines:\n")

            for idx, line in enumerate(lines):
                box = line[0]
                text_det = line[1][0]
                conf_det = line[1][1]

                # Re-process each crop with gentle preprocessing for better accuracy
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x1, y1 = max(0, int(min(x_coords))), max(0, int(min(y_coords)))
                x2, y2 = min(w, int(max(x_coords))), min(h, int(max(y_coords)))

                if y2 - y1 > 5 and x2 - x1 > 5:
                    line_crop = img[y1:y2, x1:x2]
                    processed = preprocess_for_ocr(line_crop)
                    res2 = ocr.ocr(processed, det=False, rec=True)
                    text_improved = text_det
                    conf_improved = conf_det
                    if res2 and res2[0] and len(res2[0]) > 0:
                        text_improved = res2[0][0][0]
                        conf_improved = float(res2[0][0][1])

                    if conf_improved >= 0.4:
                        out.write(f"  L{idx}: {text_improved} | conf={conf_improved:.2f}\n")
                    else:
                        out.write(f"  L{idx}: [{text_det}] (low conf={conf_improved:.2f}, raw={conf_det:.2f})\n")
                else:
                    out.write(f"  L{idx}: {text_det} | conf={conf_det:.2f}\n")

    print(f"\nDone! Output written to {out_path}")
    print("Compare with paddle_cpu_output.txt to see improvements.")


if __name__ == "__main__":
    main()