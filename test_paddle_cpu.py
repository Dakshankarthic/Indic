import os
os.environ["FLAGS_use_mkldnn"] = "0"
from paddleocr import PaddleOCR
import cv2

def main():
    ocr = PaddleOCR(use_angle_cls=False, lang='hi', use_gpu=False, enable_mkldnn=False)
    test_dir = "D:/indic_challenge/test_10_images"
    images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])
    if not images:
        print("No images found!")
        return
    with open("D:/indic_challenge/paddle_cpu_output.txt", "w", encoding="utf-8") as out_file:
        for img_name in images:
            img_path = os.path.join(test_dir, img_name)
            out_file.write(f"\n======================================")
            out_file.write(f"\nProcessing {img_name} with CPU PaddleOCR...\n")
            res = ocr.ocr(img_path, det=True, rec=True)
            if res is None or len(res) == 0 or res[0] is None:
                out_file.write("No results!\n")
                continue
            out_file.write(f"Detected {len(res[0])} text boxes:\n")
            for idx, line in enumerate(res[0]):
                text_val = line[1][0]
                out_file.write(f"Line {idx}: Box: {line[0]}, Text: {text_val}, Conf: {line[1][1]:.2f}\n")
    print("Done! Output written to paddle_cpu_output.txt")



if __name__ == "__main__":
    main()
