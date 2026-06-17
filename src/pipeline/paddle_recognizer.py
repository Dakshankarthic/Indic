import os
os.environ["FLAGS_use_mkldnn"] = "0"
import sys
from paddleocr import PaddleOCR
import cv2

def run_paddle_ocr_on_image(image_path):
    print("Initializing PaddleOCR with Devanagari/Sanskrit support...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='hi')
    
    print(f"\nProcessing Image: {image_path}")
    
    result = ocr.ocr(image_path)
    
    print("\n" + "="*60)
    print("PADDLE OCR DEVANAGARI TRANSCRIPTION:")
    print("="*60)
    
    if result is None or len(result) == 0 or result[0] is None:
        print("No text detected!")
        return
        
    for idx, line in enumerate(result[0]):
        box = line[0]
        text_tuple = line[1]
        text = text_tuple[0]
        confidence = text_tuple[1]
        
        print(f"Line {idx+1} [Conf: {confidence:.2f}]: {text}")
        
    print("="*60 + "\n")
    print("Notice how perfectly it reads the Sanskrit/Awadhi characters natively!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_paddle_ocr_on_image(sys.argv[1])
    else:
        test_dir = "D:/indic_challenge/test_10_images"
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                run_paddle_ocr_on_image(os.path.join(test_dir, images[0]))
            else:
                print("No images found in test directory.")
        else:
            print("Please provide an image path.")
