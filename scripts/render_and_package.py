"""
Render visuals from existing PAGE-XML files and prepare Aletheia submission.
No GPU needed - just reads XMLs and draws bounding boxes on images.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import xml.etree.ElementTree as ET
import shutil
import time

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

INPUT_IMG_DIR = r"D:\indic_challenge\org_img-20260701T115420Z-3-001\org_img"
XML_DIR = r"D:\indic_challenge\final_competition_xmls"
VISUAL_DIR = r"D:\indic_challenge\final_visual"
ALETHEIA_DIR = r"D:\indic_challenge\aletheia_submission"

os.makedirs(VISUAL_DIR, exist_ok=True)
os.makedirs(ALETHEIA_DIR, exist_ok=True)


def parse_coords(coords_el):
    """Parse Coords points='x1,y1 x2,y2 ...' into bounding box."""
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


def render_visual(img, xml_path):
    """Draw all bounding boxes from PAGE-XML onto image."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []
    words = []
    glyphs = []
    graphics = []

    # Find all TextLines
    for tl in root.findall(f".//{{{PAGE_NS}}}TextLine"):
        coords = tl.find(f"{{{PAGE_NS}}}Coords")
        box = parse_coords(coords)
        if box:
            lines.append(box)

    # Find all Words
    for w in root.findall(f".//{{{PAGE_NS}}}Word"):
        coords = w.find(f"{{{PAGE_NS}}}Coords")
        box = parse_coords(coords)
        if box:
            words.append(box)

    # Find all Glyphs
    for g in root.findall(f".//{{{PAGE_NS}}}Glyph"):
        coords = g.find(f"{{{PAGE_NS}}}Coords")
        box = parse_coords(coords)
        if box:
            glyphs.append(box)

    # Find all GraphicRegions (illustrations)
    for gr in root.findall(f".//{{{PAGE_NS}}}GraphicRegion"):
        coords = gr.find(f"{{{PAGE_NS}}}Coords")
        box = parse_coords(coords)
        if box:
            graphics.append(box)

    img_vis = img.copy()

    # Draw in order: glyphs (bottom) → words → lines → graphics (top)
    # Red = Characters/Glyphs (thin)
    for (x1, y1, x2, y2) in glyphs:
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Blue = Words
    for (x1, y1, x2, y2) in words:
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Green = Lines
    for (x1, y1, x2, y2) in lines:
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Cyan = Illustrations
    for (x1, y1, x2, y2) in graphics:
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 255, 0), 3)

    return img_vis, len(lines), len(words), len(glyphs), len(graphics)


def main():
    xml_files = sorted(
        [f for f in os.listdir(XML_DIR) if f.endswith('.xml')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    print(f"Rendering visuals for {len(xml_files)} pages...")
    print(f"Source images: {INPUT_IMG_DIR}")
    print(f"XMLs: {XML_DIR}")
    print(f"Visuals output: {VISUAL_DIR}")
    print(f"Aletheia output: {ALETHEIA_DIR}")
    print()

    start = time.time()
    total_lines = 0
    total_words = 0
    total_glyphs = 0

    for i, xml_file in enumerate(xml_files):
        page_name = xml_file.replace('.xml', '')
        img_name = page_name + '.jpg'
        img_path = os.path.join(INPUT_IMG_DIR, img_name)
        xml_path = os.path.join(XML_DIR, xml_file)

        if not os.path.exists(img_path):
            print(f"  SKIP: {img_name} not found")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  SKIP: {img_name} unreadable")
            continue

        # Render visual
        img_vis, n_lines, n_words, n_glyphs, n_graphics = render_visual(img, xml_path)
        vis_path = os.path.join(VISUAL_DIR, img_name)
        cv2.imwrite(vis_path, img_vis)

        total_lines += n_lines
        total_words += n_words
        total_glyphs += n_glyphs

        # Copy XML + raw image to Aletheia submission
        aletheia_xml = os.path.join(ALETHEIA_DIR, xml_file)
        aletheia_img = os.path.join(ALETHEIA_DIR, img_name)
        if not os.path.exists(aletheia_xml) or os.path.getmtime(xml_path) > os.path.getmtime(aletheia_xml):
            shutil.copy2(xml_path, aletheia_xml)
        if not os.path.exists(aletheia_img):
            shutil.copy2(img_path, aletheia_img)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(xml_files) - i - 1) / rate
            print(f"  [{i+1}/{len(xml_files)}] {page_name}: "
                  f"L={n_lines} W={n_words} G={n_glyphs} | "
                  f"{rate:.1f} pg/s, ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE! {len(xml_files)} pages in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total: {total_lines} lines, {total_words} words, {total_glyphs} glyphs")
    print(f"Visuals: {VISUAL_DIR}")
    print(f"Aletheia: {ALETHEIA_DIR}")

    # Verify counts
    vis_count = len([f for f in os.listdir(VISUAL_DIR) if f.endswith('.jpg')])
    aletheia_xml = len([f for f in os.listdir(ALETHEIA_DIR) if f.endswith('.xml')])
    aletheia_jpg = len([f for f in os.listdir(ALETHEIA_DIR) if f.endswith('.jpg')])
    print(f"\nVerification:")
    print(f"  final_visual/: {vis_count} JPGs")
    print(f"  aletheia_submission/: {aletheia_xml} XMLs + {aletheia_jpg} JPGs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
