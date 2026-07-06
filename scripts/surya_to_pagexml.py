"""
Run Surya OCR (best method) and inject transcription into PAGE-XML files.

Resumes from cache if interrupted.

  python scripts/surya_to_pagexml.py
  python scripts/surya_to_pagexml.py --inject-only   # XML update from cache only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
os.environ.setdefault("LLAMA_CPP_BINARY", r"d:\indic_challenge\llama_bin\llama-server.exe")

ROOT = Path(r"d:\indic_challenge")
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def extract_text_from_result(result) -> str:
    lines = []
    if hasattr(result, "blocks"):
        for block in result.blocks:
            if hasattr(block, "html"):
                text = re.sub(r"<[^>]+>", "", block.html).strip()
                if text:
                    lines.append(text)
            elif hasattr(block, "lines"):
                for line in block.lines:
                    if line.text.strip():
                        lines.append(line.text.strip())
    if hasattr(result, "text_lines") and result.text_lines:
        if not lines:
            for line in result.text_lines:
                if line.text and line.text.strip():
                    lines.append(line.text.strip())
    return "\n".join(lines)


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
    return cv2.warpAffine(
        img_cv, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_cv = deskew_image(img_cv)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    upscaled = cv2.resize(thresh, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))


def line_top_y(line_el) -> int:
    coords = line_el.find(f"./{{{PAGE_NS}}}Coords")
    if coords is None or not coords.get("points"):
        return 0
    ys = [int(p.split(",")[1]) for p in coords.get("points").split() if "," in p]
    return min(ys) if ys else 0


def set_text_equiv(parent, text: str) -> None:
    for old in parent.findall(f"./{{{PAGE_NS}}}TextEquiv"):
        parent.remove(old)
    if not text:
        return
    te = ET.SubElement(parent, f"{{{PAGE_NS}}}TextEquiv")
    uni = ET.SubElement(te, f"{{{PAGE_NS}}}Unicode")
    uni.text = text


def inject_text_into_xml(xml_path: Path, full_text: str) -> None:
    surya_lines = full_text.split("\n")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    text_lines = root.findall(f".//{{{PAGE_NS}}}TextLine")
    text_lines.sort(key=line_top_y)

    if not text_lines:
        page = root.find(f".//{{{PAGE_NS}}}Page")
        if page is None:
            return
        region = page.find(f"./{{{PAGE_NS}}}TextRegion")
        if region is None:
            region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion")
            region.set("id", "region_text_surya")
            c = ET.SubElement(region, f"{{{PAGE_NS}}}Coords")
            w = int(page.get("imageWidth", "512"))
            h = int(page.get("imageHeight", "512"))
            c.set("points", f"0,0 {w},0 {w},{h} 0,{h}")
        text_lines = []
        for i, line_text in enumerate(surya_lines):
            if not line_text.strip():
                continue
            line_el = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine")
            line_el.set("id", f"line_surya_{i}")
            coords = ET.SubElement(line_el, f"{{{PAGE_NS}}}Coords")
            y1 = min(h - 20, 5 + i * 18)
            coords.set("points", f"0,{y1} {w},{y1} {w},{y1+14} 0,{y1+14}")
            set_text_equiv(line_el, line_text)
    else:
        for i, line_el in enumerate(text_lines):
            text = surya_lines[i] if i < len(surya_lines) else ""
            set_text_equiv(line_el, text)

    metadata = root.find(f"./{{{PAGE_NS}}}Metadata")
    if metadata is not None:
        for tag in ("Creator", "LastChange"):
            el = metadata.find(f"./{{{PAGE_NS}}}{tag}")
            if el is not None:
                el.text = "Surya OCR + PAGE-XML merge"

    ET.indent(tree, space="  ")
    rough = ET.tostring(tree.getroot(), encoding="unicode")
    # Avoid ns0: prefixes — Aletheia needs default namespace tags
    rough = rough.replace(f'xmlns:ns0="{PAGE_NS}"', f'xmlns="{PAGE_NS}"')
    rough = re.sub(r"</?ns0:", lambda m: m.group(0).replace("ns0:", ""), rough)
    from xml.dom import minidom

    xml_str = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    xml_str = "\n".join(line for line in xml_str.splitlines() if line.strip())
    xml_path.write_text(xml_str, encoding="utf-8")


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default=str(ROOT / "org_img-20260701T115420Z-3-001" / "org_img"))
    parser.add_argument("--xml-dir", default=str(ROOT / "final_competition_xmls"))
    parser.add_argument("--cache", default=str(ROOT / "surya_text_cache.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--inject-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    xml_dir = Path(args.xml_dir)
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)

    pages = sorted(img_dir.glob("page_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    if args.limit > 0:
        pages = pages[: args.limit]

    if not args.inject_only:
        from surya.recognition import RecognitionPredictor

        print(f"Loading Surya... ({len(pages)} pages, batch={args.batch_size})")
        rec = RecognitionPredictor()
        pending = [p for p in pages if p.stem not in cache]
        print(f"OCR pending: {len(pending)} | cached: {len(pages) - len(pending)}")

        t0 = time.time()
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            images = [preprocess_image(Image.open(p).convert("RGB")) for p in batch]
            preds = rec(images)
            for path, result in zip(batch, preds):
                cache[path.stem] = extract_text_from_result(result)
            save_cache(cache_path, cache)
            done = start + len(batch)
            print(f"  OCR [{done}/{len(pending)}] elapsed={time.time()-t0:.0f}s")

    print("Injecting text into PAGE-XML...")
    injected = 0
    for path in pages:
        stem = path.stem
        xml_path = xml_dir / f"{stem}.xml"
        if stem not in cache or not xml_path.exists():
            continue
        inject_text_into_xml(xml_path, cache[stem])
        injected += 1
        if injected % 100 == 0:
            print(f"  injected {injected}/{len(pages)}")

    save_cache(cache_path, cache)
    print(f"Done. OCR cache: {cache_path}")
    print(f"Updated {injected} XML files in {xml_dir}")

    # Refresh Aletheia package and visuals after OCR/inject
    try:
        import subprocess

        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, str(root / "scripts" / "prepare_aletheia_submission.py")], check=False)
        subprocess.run(
            [sys.executable, str(root / "scripts" / "render_visuals_from_xml.py"), "--xml-dir", str(xml_dir)],
            check=False,
        )
        print("Refreshed aletheia_submission/ and final_visual/")
    except Exception as e:
        print(f"Note: post-process refresh skipped: {e}")


if __name__ == "__main__":
    main()
