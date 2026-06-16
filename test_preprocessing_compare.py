"""Quick comparison: Old destructive binarization vs New gentle preprocessing."""
import os
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ---- OLD preprocess (destructive) ----
def old_preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(denoised)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    min_area = max(10, crop.shape[0] * crop.shape[1] * 0.001)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 255
    padded = cv2.copyMakeBorder(binary, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


# ---- NEW preprocess (gentle) ----
def new_preprocess(crop):
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
    ocr = PaddleOCR(
        use_angle_cls=False, lang='hi', use_gpu=False, enable_mkldnn=False,
        show_log=False, drop_score=0.1, rec_image_shape='3, 48, 320',
        rec_batch_num=1,
    )

    img = cv2.imread("test_10_images/ms1000_1994_0001_web.jpg")
    if img is None:
        print("Image not found!")
        return

    ly1, ly2 = 280, 340
    lx1, lx2 = 260, 1520
    line_crop = img[ly1:ly2, lx1:lx2]

    print("Line crop size:", line_crop.shape)
    print()

    cv2.imwrite("comparison_original.png", line_crop)
    old_processed = old_preprocess(line_crop)
    new_processed = new_preprocess(line_crop)
    cv2.imwrite("comparison_old_binarized.png", old_processed)
    cv2.imwrite("comparison_new_gentle.png", new_processed)

    print("=" * 60)
    print("OLD PREPROCESSING (destructive binarization)")
    print("=" * 60)
    res_old = ocr.ocr(old_processed, det=False, rec=True)
    if res_old and res_old[0] and len(res_old[0]) > 0:
        print("Text:", res_old[0][0][0])
        print("Confidence:", f"{res_old[0][0][1]:.4f}")
    else:
        print("NO RESULT!")

    print()
    print("=" * 60)
    print("NEW PREPROCESSING (gentle enhancement)")
    print("=" * 60)
    res_new = ocr.ocr(new_processed, det=False, rec=True)
    if res_new and res_new[0] and len(res_new[0]) > 0:
        print("Text:", res_new[0][0][0])
        print("Confidence:", f"{res_new[0][0][1]:.4f}")
    else:
        print("NO RESULT!")

    print()
    print("Saved comparison images:")
    print("  comparison_original.png")
    print("  comparison_old_binarized.png")
    print("  comparison_new_gentle.png")


if __name__ == "__main__":
    main()