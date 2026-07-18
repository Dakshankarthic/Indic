"""
=== ALL-IN-ONE KAGGLE NOTEBOOK ===
DINO Layout + Surya OCR → PAGE-XML with full hierarchy
Run this on Kaggle with GPU T4 x2

INSTRUCTIONS:
1. Upload this entire dataset to Kaggle
2. Create a new notebook, set Accelerator to GPU T4 x2
3. Copy-paste this entire file into a single code cell
4. Run it
5. Download the output files from /kaggle/working/output/
"""

import subprocess
import sys
import os

# ============================================================
# STEP 0: Install dependencies
# ============================================================
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "surya-ocr", "opencv-python-headless", "Pillow",
                       "numpy", "lxml"])
print("Dependencies installed!")

# ============================================================
# STEP 1: Imports
# ============================================================
import json
import re
import time
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from lxml import etree

# ============================================================
# STEP 2: Configuration - AUTO-DETECT PATHS
# ============================================================
# Find the dataset path automatically
KAGGLE_INPUT = Path("/kaggle/input")
dataset_dirs = list(KAGGLE_INPUT.iterdir()) if KAGGLE_INPUT.exists() else []

IMG_DIR = None
XML_DIR = None

for d in dataset_dirs:
    if (d / "images").exists():
        IMG_DIR = str(d / "images")
        XML_DIR = str(d / "xmls")
        break
    # Also check if images are directly in the dataset
    jpgs = list(d.glob("page_*.jpg"))
    if len(jpgs) > 100:
        IMG_DIR = str(d)
        break

if IMG_DIR is None:
    # Fallback: try common paths
    for p in ["/kaggle/input/indic-kaggle-package/images",
              "/kaggle/input/kaggle-package/images",
              "/kaggle/input/indic-images/org_img"]:
        if os.path.exists(p):
            IMG_DIR = p
            XML_DIR = p.replace("/images", "/xmls")
            break

if IMG_DIR is None:
    print("ERROR: Could not find images directory!")
    print("Available datasets:", [str(d) for d in dataset_dirs])
    for d in dataset_dirs:
        print(f"  {d}: {list(d.iterdir())[:10]}")
    sys.exit(1)

OUTPUT_DIR = "/kaggle/working/output"
CACHE_FILE = "/kaggle/working/surya_text_cache.json"
FINAL_XML_DIR = os.path.join(OUTPUT_DIR, "final_xmls")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visuals")

os.makedirs(FINAL_XML_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

print(f"Images: {IMG_DIR}")
print(f"XMLs: {XML_DIR}")
print(f"Output: {OUTPUT_DIR}")

BATCH_SIZE = 4

# ============================================================
# STEP 3: Helper Functions
# ============================================================
def extract_text_from_result(result):
    """Extract plain text from Surya OCR result."""
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
    if hasattr(result, 'text_lines') and result.text_lines:
        if not lines:
            for line in result.text_lines:
                if line.text and line.text.strip():
                    lines.append(line.text.strip())
    return '\n'.join(lines)


def deskew_image(img_cv):
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
    h, w = img_cv.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_cv, m, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(pil_img):
    """Enhanced preprocessing for Surya OCR."""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_cv = deskew_image(img_cv)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    upscaled = cv2.resize(thresh, None, fx=6.0, fy=6.0,
                          interpolation=cv2.INTER_CUBIC)
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NSMAP = {'pc': PAGE_NS}


def inject_surya_text_into_xml(xml_path, surya_text, output_path):
    """Inject Surya OCR text into existing PAGE-XML, replacing Tesseract text."""
    tree = etree.parse(xml_path)
    root = tree.getroot()

    surya_lines = [l.strip() for l in surya_text.split('\n') if l.strip()]

    # Find all TextLines
    text_lines = root.findall(f".//{{{PAGE_NS}}}TextLine")

    # Assign Surya text to TextLines (line by line)
    for i, tl in enumerate(text_lines):
        if i < len(surya_lines):
            line_text = surya_lines[i]
        else:
            line_text = ""

        # Update TextLine's TextEquiv/Unicode
        te = tl.find(f"{{{PAGE_NS}}}TextEquiv")
        if te is None:
            te = etree.SubElement(tl, f"{{{PAGE_NS}}}TextEquiv")
        uni = te.find(f"{{{PAGE_NS}}}Unicode")
        if uni is None:
            uni = etree.SubElement(te, f"{{{PAGE_NS}}}Unicode")
        uni.text = line_text

        # Update Word-level text
        words_el = tl.findall(f"{{{PAGE_NS}}}Word")
        line_words = line_text.split() if line_text else []
        for j, w in enumerate(words_el):
            w_te = w.find(f"{{{PAGE_NS}}}TextEquiv")
            if w_te is None:
                w_te = etree.SubElement(w, f"{{{PAGE_NS}}}TextEquiv")
            w_uni = w_te.find(f"{{{PAGE_NS}}}Unicode")
            if w_uni is None:
                w_uni = etree.SubElement(w_te, f"{{{PAGE_NS}}}Unicode")
            if j < len(line_words):
                w_uni.text = line_words[j]
            else:
                w_uni.text = ""

            # Update Glyph-level text
            glyphs_el = w.findall(f"{{{PAGE_NS}}}Glyph")
            word_text = line_words[j] if j < len(line_words) else ""
            for k, g in enumerate(glyphs_el):
                g_te = g.find(f"{{{PAGE_NS}}}TextEquiv")
                if g_te is None:
                    g_te = etree.SubElement(g, f"{{{PAGE_NS}}}TextEquiv")
                g_uni = g_te.find(f"{{{PAGE_NS}}}Unicode")
                if g_uni is None:
                    g_uni = etree.SubElement(g_te, f"{{{PAGE_NS}}}Unicode")
                if k < len(word_text):
                    g_uni.text = word_text[k]
                else:
                    g_uni.text = ""

    # Update TextRegion text (concatenation of its lines)
    for tr in root.findall(f".//{{{PAGE_NS}}}TextRegion"):
        region_lines = []
        for tl in tr.findall(f"{{{PAGE_NS}}}TextLine"):
            te = tl.find(f"{{{PAGE_NS}}}TextEquiv")
            if te is not None:
                uni = te.find(f"{{{PAGE_NS}}}Unicode")
                if uni is not None and uni.text:
                    region_lines.append(uni.text)
        tr_te = tr.find(f"{{{PAGE_NS}}}TextEquiv")
        if tr_te is None:
            tr_te = etree.SubElement(tr, f"{{{PAGE_NS}}}TextEquiv")
        tr_uni = tr_te.find(f"{{{PAGE_NS}}}Unicode")
        if tr_uni is None:
            tr_uni = etree.SubElement(tr_te, f"{{{PAGE_NS}}}Unicode")
        tr_uni.text = '\n'.join(region_lines)

    tree.write(output_path, xml_declaration=True, encoding='UTF-8',
               pretty_print=True)


def parse_coords(coords_el):
    """Parse Coords points into bounding box."""
    if coords_el is None:
        return None
    pts_str = coords_el.get("points", "")
    if not pts_str:
        return None
    points = []
    for p in pts_str.strip().split():
        parts = p.split(",")
        if len(parts) == 2:
            points.append((int(float(parts[0])), int(float(parts[1]))))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def render_visual(img_path, xml_path, output_path):
    """Draw bounding boxes on image from PAGE-XML."""
    img = cv2.imread(img_path)
    if img is None:
        return
    tree = etree.parse(xml_path)
    root = tree.getroot()
    img_vis = img.copy()

    # Red = Glyphs
    for g in root.findall(f".//{{{PAGE_NS}}}Glyph"):
        box = parse_coords(g.find(f"{{{PAGE_NS}}}Coords"))
        if box:
            cv2.rectangle(img_vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
    # Blue = Words
    for w in root.findall(f".//{{{PAGE_NS}}}Word"):
        box = parse_coords(w.find(f"{{{PAGE_NS}}}Coords"))
        if box:
            cv2.rectangle(img_vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    # Green = Lines
    for tl in root.findall(f".//{{{PAGE_NS}}}TextLine"):
        box = parse_coords(tl.find(f"{{{PAGE_NS}}}Coords"))
        if box:
            cv2.rectangle(img_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    # Cyan = Graphics
    for gr in root.findall(f".//{{{PAGE_NS}}}GraphicRegion"):
        box = parse_coords(gr.find(f"{{{PAGE_NS}}}Coords"))
        if box:
            cv2.rectangle(img_vis, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 3)

    cv2.imwrite(output_path, img_vis)


# ============================================================
# STEP 4: Run Surya OCR on all images — DUAL GPU
# ============================================================
import torch
import multiprocessing as mp
from multiprocessing import Manager

NUM_GPUS = torch.cuda.device_count()
print(f"\nDetected {NUM_GPUS} GPU(s): {[torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]}")

# Load existing cache
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding='utf-8') as f:
        cache = json.load(f)

pages = sorted(Path(IMG_DIR).glob("page_*.jpg"),
               key=lambda p: int(p.stem.split('_')[1]))
print(f"Total pages: {len(pages)}")

pending = [p for p in pages if p.stem not in cache]
print(f"Pending OCR: {len(pending)} | Already cached: {len(pages) - len(pending)}\n")


def gpu_worker(gpu_id, page_list, shared_cache, lock, worker_batch_size=4):
    """Run Surya OCR on a subset of pages using a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from surya.recognition import RecognitionPredictor
    rec = RecognitionPredictor()
    print(f"  [GPU {gpu_id}] Model loaded, processing {len(page_list)} pages")

    for start in range(0, len(page_list), worker_batch_size):
        batch = page_list[start:start + worker_batch_size]
        images = []
        for p in batch:
            pil_img = Image.open(p).convert("RGB")
            images.append(preprocess_image(pil_img))

        try:
            preds = rec(images)
            with lock:
                for path, result in zip(batch, preds):
                    shared_cache[path.stem] = extract_text_from_result(result)
        except Exception as e:
            print(f"  [GPU {gpu_id}] ERROR: {e}")
            with lock:
                for path in batch:
                    if path.stem not in shared_cache:
                        shared_cache[path.stem] = ""

        done = start + len(batch)
        if done % 20 == 0 or done >= len(page_list):
            print(f"  [GPU {gpu_id}] {done}/{len(page_list)} done")

    print(f"  [GPU {gpu_id}] FINISHED")


t0 = time.time()

if NUM_GPUS >= 2 and len(pending) > 10:
    print(f"Using {NUM_GPUS} GPUs in parallel!\n")
    manager = Manager()
    shared_cache = manager.dict(cache)
    lock = manager.Lock()

    # Split pages evenly across GPUs
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, p in enumerate(pending):
        chunks[i % NUM_GPUS].append(p)

    for i, chunk in enumerate(chunks):
        print(f"  GPU {i}: {len(chunk)} pages")

    # Use spawn context for CUDA compatibility
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = ctx.Process(target=gpu_worker,
                        args=(gpu_id, chunks[gpu_id], shared_cache, lock, BATCH_SIZE))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results back
    cache.update(dict(shared_cache))
else:
    print("Using single GPU\n")
    from surya.recognition import RecognitionPredictor
    rec_predictor = RecognitionPredictor()
    print("Model loaded!\n")

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        images = []
        for p in batch:
            pil_img = Image.open(p).convert("RGB")
            images.append(preprocess_image(pil_img))

        try:
            preds = rec_predictor(images)
            for path, result in zip(batch, preds):
                cache[path.stem] = extract_text_from_result(result)
        except Exception as e:
            print(f"  ERROR on batch {start}: {e}")
            for path in batch:
                if path.stem not in cache:
                    cache[path.stem] = ""

        done = start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(pending) - done) / rate if rate > 0 else 0
        if done % 20 == 0 or done == len(pending):
            print(f"  [{done}/{len(pending)}] "
                  f"{rate:.2f} pg/s | ETA: {eta/60:.1f} min")

# Save final cache
with open(CACHE_FILE, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)

ocr_time = time.time() - t0
print(f"\nOCR complete! {len(cache)} pages in {ocr_time/60:.1f} min")

# ============================================================
# STEP 5: Inject Surya text into XMLs + render visuals
# ============================================================
print(f"\nInjecting Surya text into XMLs and rendering visuals...")

if XML_DIR and os.path.exists(XML_DIR):
    t1 = time.time()
    xml_files = sorted(Path(XML_DIR).glob("page_*.xml"),
                       key=lambda p: int(p.stem.split('_')[1]))
    print(f"Processing {len(xml_files)} XMLs...")

    for i, xml_path in enumerate(xml_files):
        page_name = xml_path.stem
        surya_text = cache.get(page_name, "")
        out_xml = os.path.join(FINAL_XML_DIR, xml_path.name)
        img_path = os.path.join(IMG_DIR, page_name + ".jpg")

        # Inject text
        try:
            inject_surya_text_into_xml(str(xml_path), surya_text, out_xml)
        except Exception as e:
            print(f"  XML error {page_name}: {e}")
            import shutil
            shutil.copy2(str(xml_path), out_xml)

        # Render visual
        vis_path = os.path.join(VISUAL_DIR, page_name + ".jpg")
        if os.path.exists(img_path):
            try:
                render_visual(img_path, out_xml, vis_path)
            except Exception as e:
                print(f"  Visual error {page_name}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(xml_files)}] done")

    inject_time = time.time() - t1
    print(f"Injection + visuals done in {inject_time:.0f}s")
else:
    print("WARNING: No XML directory found. Only cache file saved.")
    print("Download surya_text_cache.json and inject locally.")

# ============================================================
# STEP 6: Summary
# ============================================================
total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"ALL DONE!")
print(f"Total time: {total_time/60:.1f} min")
print(f"Surya text cache: {CACHE_FILE}")
print(f"Final XMLs: {FINAL_XML_DIR} ({len(os.listdir(FINAL_XML_DIR))} files)")
print(f"Visuals: {VISUAL_DIR} ({len(os.listdir(VISUAL_DIR))} files)")
print(f"{'='*60}")
print(f"\nDownload the output/ folder from /kaggle/working/output/")
print(f"It contains final_xmls/ and visuals/")
