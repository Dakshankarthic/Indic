import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import sys

def test_trocr_on_image(image_path):
    print("Loading Base TrOCR Model (microsoft/trocr-base-handwritten)...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    width, height = image.size
    crop_box = (int(width * 0.2), int(height * 0.4), int(width * 0.8), int(height * 0.5))
    cropped_image = image.crop(crop_box)
    
    print("Running OCR Inference...")
    pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
    
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*50)
    print("OCR RESULT:")
    print(generated_text)
    print("="*50 + "\n")
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_trocr_on_image(sys.argv[1])
    else:
        print("Please provide an image path.")
