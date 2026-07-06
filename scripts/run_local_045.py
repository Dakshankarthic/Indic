import os, json, time
from pathlib import Path
from PIL import Image
import torch
from surya.ocr import run_ocr
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

def main():
    CACHE_FILE = "d:/indic_challenge/surya_text_cache.json"

    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print("Deleted old bugged cache.")

    print("Loading models directly to your RTX 2070 Super GPU...")
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_processor, rec_model = load_rec_processor(), load_rec_model()

    det_model.to("cuda")
    rec_model.to("cuda")
    print("Models loaded successfully!")

    IMG_DIR = "d:/indic_challenge/org_img-20260701T115420Z-3-001/org_img"
    pages = sorted(Path(IMG_DIR).glob("page_*.jpg"), key=lambda p: int(p.stem.split('_')[1]))
    cache = {}
    t0 = time.time()
    BATCH = 8 # RTX 2070 Super has 8GB VRAM!

    for start in range(0, len(pages), BATCH):
        batch = pages[start:start+BATCH]
        images = [Image.open(p).convert("RGB") for p in batch]
        langs = [["ta", "en"]] * len(images)
        
        try:
            results = run_ocr(images, langs, det_model, det_processor, rec_model, rec_processor)
            for path, result in zip(batch, results):
                text = "\n".join([line.text for line in result.text_lines])
                cache[path.stem] = text
        except Exception as e:
            print(f"Error at {start}: {e}")
            for path in batch:
                cache[path.stem] = ""
        
        done = start + len(batch)
        # Always print so we see progress
        rate = done / (time.time() - t0)
        eta = (len(pages) - done) / rate / 60
        print(f"  [{done}/{len(pages)}] {rate:.2f} pg/s | ETA {eta:.1f} min")
        
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)

    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print("\nDONE!")

if __name__ == "__main__":
    main()
