"""
Create a stunning, presentation-quality visualization for the hackathon evaluator.
Generates multiple views showing the full PAGE-XML hierarchy:
  1. Original image (clean)
  2. TextRegion detection
  3. TextLine detection  
  4. Word + Glyph detection
  5. Full hierarchy overlay
  6. Side-by-side comparison panel
"""
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

# --- Config ---
IMG_PATH = r"d:\indic_challenge\test_10_images\ms100_1993_0000_web.jpg"
XML_PATH = r"d:\indic_challenge\final_results\ms100_1993_0000_web.xml"
OUT_DIR  = r"d:\indic_challenge\presentation"

# Colors (BGR) — carefully chosen for readability on parchment
COLOR_REGION   = (50, 50, 220)      # Deep red
COLOR_LINE     = (30, 200, 30)      # Vivid green
COLOR_WORD     = (220, 160, 30)     # Blue-cyan
COLOR_GLYPH    = (0, 165, 255)      # Orange
COLOR_BG       = (30, 30, 35)       # Near-black background
COLOR_PANEL_BG = (40, 40, 45)       # Slightly lighter panel
COLOR_TEXT     = (230, 230, 230)     # Light text
COLOR_ACCENT   = (0, 200, 255)      # Gold accent

os.makedirs(OUT_DIR, exist_ok=True)

# --- Parse XML ---
ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
tree = ET.parse(XML_PATH)
root = tree.getroot()

def parse_pts(coords_str):
    return np.array([[int(float(p.split(',')[0])), int(float(p.split(',')[1]))]
                     for p in coords_str.split(' ')], dtype=np.int32)

regions_data = []
for r in root.findall('.//pc:TextRegion', ns):
    pts = parse_pts(r.find('pc:Coords', ns).attrib['points'])
    lines_data = []
    for tl in r.findall('pc:TextLine', ns):
        lpts = parse_pts(tl.find('pc:Coords', ns).attrib['points'])
        words_data = []
        for w in tl.findall('pc:Word', ns):
            wpts = parse_pts(w.find('pc:Coords', ns).attrib['points'])
            glyphs_data = []
            for g in w.findall('pc:Glyph', ns):
                gpts = parse_pts(g.find('pc:Coords', ns).attrib['points'])
                glyphs_data.append(gpts)
            words_data.append({'pts': wpts, 'glyphs': glyphs_data})
        lines_data.append({'pts': lpts, 'words': words_data})
    regions_data.append({'pts': pts, 'lines': lines_data})

total_lines = sum(len(r['lines']) for r in regions_data)
total_words = sum(len(l['words']) for r in regions_data for l in r['lines'])
total_glyphs = sum(len(w['glyphs']) for r in regions_data for l in r['lines'] for w in l['words'])

# --- Load original ---
orig = cv2.imread(IMG_PATH)
h, w = orig.shape[:2]

def add_label(img, text, pos, color, font_scale=0.55, thickness=1, bg=True):
    """Draw a label with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), (0,0,0), -1)
        cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+4), color, 1)
    cv2.putText(img, text, (x+1, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_with_alpha(base, overlay, alpha=0.3):
    """Blend overlay on base with transparency."""
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

# ============================
# VIEW 1: Original Clean Image
# ============================
v1 = orig.copy()
add_label(v1, "Original Manuscript", (15, 30), COLOR_TEXT, 0.7, 2)
cv2.imwrite(os.path.join(OUT_DIR, "1_original.jpg"), v1, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 2: TextRegion Detection
# ============================
v2 = orig.copy()
overlay2 = orig.copy()
for i, r in enumerate(regions_data):
    cv2.fillPoly(overlay2, [r['pts']], COLOR_REGION)
v2 = draw_with_alpha(v2, overlay2, 0.25)
for i, r in enumerate(regions_data):
    cv2.polylines(v2, [r['pts']], True, COLOR_REGION, 3, cv2.LINE_AA)
    # Label each region
    cx, cy = np.mean(r['pts'], axis=0).astype(int)
    add_label(v2, f"Region {i}", (r['pts'][0][0]+5, r['pts'][0][1]+20), COLOR_REGION, 0.5)
add_label(v2, f"TextRegion Detection ({len(regions_data)} regions)", (15, 30), COLOR_REGION, 0.7, 2)
cv2.imwrite(os.path.join(OUT_DIR, "2_text_regions.jpg"), v2, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 3: TextLine Detection
# ============================
v3 = orig.copy()
overlay3 = orig.copy()
line_count = 0
for r in regions_data:
    cv2.polylines(v3, [r['pts']], True, COLOR_REGION, 2, cv2.LINE_AA)
    for l in r['lines']:
        line_count += 1
        cv2.fillPoly(overlay3, [l['pts']], COLOR_LINE)
v3 = draw_with_alpha(v3, overlay3, 0.20)
line_count = 0
for r in regions_data:
    cv2.polylines(v3, [r['pts']], True, COLOR_REGION, 2, cv2.LINE_AA)
    for l in r['lines']:
        line_count += 1
        cv2.polylines(v3, [l['pts']], True, COLOR_LINE, 2, cv2.LINE_AA)
        # Line number label on the left
        ly = int(np.mean(l['pts'][:, 1]))
        lx = int(np.min(l['pts'][:, 0]))
        add_label(v3, f"L{line_count}", (lx - 45, ly + 5), COLOR_LINE, 0.4, 1)
add_label(v3, f"TextLine Detection ({total_lines} lines)", (15, 30), COLOR_LINE, 0.7, 2)
cv2.imwrite(os.path.join(OUT_DIR, "3_text_lines.jpg"), v3, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 4: Word + Glyph Detection
# ============================
v4 = orig.copy()
for r in regions_data:
    for l in r['lines']:
        cv2.polylines(v4, [l['pts']], True, (80, 80, 80), 1, cv2.LINE_AA)
        for wi, wd in enumerate(l['words']):
            # Draw underline for words
            wx1 = np.min(wd['pts'][:, 0])
            wx2 = np.max(wd['pts'][:, 0])
            wy_bottom = np.max(wd['pts'][:, 1])
            cv2.line(v4, (wx1, wy_bottom + 2), (wx2, wy_bottom + 2), COLOR_WORD, 3, cv2.LINE_AA)
            for g in wd['glyphs']:
                cv2.polylines(v4, [g], True, COLOR_GLYPH, 1, cv2.LINE_AA)
add_label(v4, f"Word ({total_words}) + Glyph ({total_glyphs}) Detection", (15, 30), COLOR_WORD, 0.7, 2)
cv2.imwrite(os.path.join(OUT_DIR, "4_words_glyphs.jpg"), v4, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 5: Full Hierarchy Overlay (THE HERO IMAGE)
# ============================
v5 = orig.copy()
# Semi-transparent region fill
overlay5 = orig.copy()
for r in regions_data:
    cv2.fillPoly(overlay5, [r['pts']], COLOR_REGION)
v5 = draw_with_alpha(v5, overlay5, 0.12)

# Draw all layers
for r in regions_data:
    cv2.polylines(v5, [r['pts']], True, COLOR_REGION, 3, cv2.LINE_AA)
    for l in r['lines']:
        cv2.polylines(v5, [l['pts']], True, COLOR_LINE, 2, cv2.LINE_AA)
        for wd in l['words']:
            # Draw underline for words
            wx1 = np.min(wd['pts'][:, 0])
            wx2 = np.max(wd['pts'][:, 0])
            wy_bottom = np.max(wd['pts'][:, 1])
            cv2.line(v5, (wx1, wy_bottom + 2), (wx2, wy_bottom + 2), COLOR_WORD, 2, cv2.LINE_AA)
            for g in wd['glyphs']:
                cv2.polylines(v5, [g], True, COLOR_GLYPH, 1, cv2.LINE_AA)

add_label(v5, f"Full PAGE-XML Hierarchy: {len(regions_data)}R | {total_lines}L | {total_words}W | {total_glyphs}G",
          (15, 30), COLOR_ACCENT, 0.65, 2)
cv2.imwrite(os.path.join(OUT_DIR, "5_full_hierarchy.jpg"), v5, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 6: Professional Side-by-Side Panel
# ============================
# Create a 2x2 grid with padding and labels
pad = 20
thumb_w = w // 2
thumb_h = h // 2
panel_w = thumb_w * 2 + pad * 3
panel_h = thumb_h * 2 + pad * 3 + 80  # +80 for title bar

panel = np.full((panel_h, panel_w, 3), COLOR_BG, dtype=np.uint8)

# Title bar
cv2.rectangle(panel, (0, 0), (panel_w, 65), COLOR_PANEL_BG, -1)
title_font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(panel, "AutoAnn-Indic: Automated Layout Analysis for Sanskrit Manuscripts",
            (pad, 35), title_font, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)
cv2.putText(panel, f"ms100_1993_0000_web.jpg  |  {len(regions_data)} Regions  |  {total_lines} Lines  |  {total_words} Words  |  {total_glyphs} Glyphs",
            (pad, 55), title_font, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

# Accent line under title
cv2.line(panel, (pad, 65), (panel_w - pad, 65), COLOR_ACCENT, 2)

thumbnails = [
    (v1, "Original"),
    (v2, f"TextRegions ({len(regions_data)})"),
    (v3, f"TextLines ({total_lines})"),
    (v5, f"Full Hierarchy"),
]

positions = [
    (pad, 80),
    (pad + thumb_w + pad, 80),
    (pad, 80 + thumb_h + pad),
    (pad + thumb_w + pad, 80 + thumb_h + pad),
]

for (src, label), (px, py) in zip(thumbnails, positions):
    thumb = cv2.resize(src, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    # Add subtle border
    cv2.rectangle(thumb, (0, 0), (thumb_w-1, thumb_h-1), (80, 80, 80), 1)
    panel[py:py+thumb_h, px:px+thumb_w] = thumb
    # Label below each thumbnail
    cv2.putText(panel, label, (px + 5, py + thumb_h - 10), title_font, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT_DIR, "6_presentation_panel.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 7: Zoomed Detail Panel (cropped region showing glyph-level detail)
# ============================
# Crop a portion of the full hierarchy image to show glyph quality
# Focus on lines 3-7 roughly
crop_y1, crop_y2 = 200, 550
crop_x1, crop_x2 = 80, 900
detail_crop = v5[crop_y1:crop_y2, crop_x1:crop_x2].copy()

# Scale up 2x for clarity
detail_zoom = cv2.resize(detail_crop, (detail_crop.shape[1]*2, detail_crop.shape[0]*2),
                          interpolation=cv2.INTER_CUBIC)

# Add annotation arrows / callout labels
dh, dw = detail_zoom.shape[:2]
# Title
zoom_panel = np.full((dh + 60, dw, 3), COLOR_BG, dtype=np.uint8)
zoom_panel[50:50+dh, :] = detail_zoom
cv2.putText(zoom_panel, "Zoomed Detail: Word & Glyph Level Segmentation (2x)",
            (10, 30), title_font, 0.6, COLOR_ACCENT, 2, cv2.LINE_AA)
cv2.line(zoom_panel, (10, 45), (dw - 10, 45), COLOR_ACCENT, 1)

cv2.imwrite(os.path.join(OUT_DIR, "7_zoomed_detail.jpg"), zoom_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

# ============================
# VIEW 8: Legend / Key
# ============================
legend_w, legend_h = 500, 280
legend = np.full((legend_h, legend_w, 3), COLOR_BG, dtype=np.uint8)
cv2.putText(legend, "PAGE-XML Hierarchy Legend", (15, 35), title_font, 0.65, COLOR_ACCENT, 2, cv2.LINE_AA)
cv2.line(legend, (15, 50), (legend_w - 15, 50), COLOR_ACCENT, 1)

items = [
    (COLOR_REGION, "TextRegion", f"{len(regions_data)} detected", 3),
    (COLOR_LINE,   "TextLine",   f"{total_lines} detected", 2),
    (COLOR_WORD,   "Word",       f"{total_words} detected", 2),
    (COLOR_GLYPH,  "Glyph",     f"{total_glyphs} detected", 1),
]

for idx, (color, name, count, thick) in enumerate(items):
    y = 85 + idx * 48
    # Color swatch (filled rectangle)
    cv2.rectangle(legend, (25, y-12), (70, y+12), color, -1)
    cv2.rectangle(legend, (25, y-12), (70, y+12), (200,200,200), 1)
    cv2.putText(legend, name, (85, y+5), title_font, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(legend, count, (250, y+5), title_font, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT_DIR, "8_legend.jpg"), legend, [cv2.IMWRITE_JPEG_QUALITY, 95])

print(f"=== Presentation images saved to {OUT_DIR} ===")
print(f"  1_original.jpg          - Clean original manuscript")
print(f"  2_text_regions.jpg      - TextRegion detection overlay")
print(f"  3_text_lines.jpg        - TextLine detection with line numbers")
print(f"  4_words_glyphs.jpg      - Word + Glyph (character) detection")
print(f"  5_full_hierarchy.jpg    - Full PAGE-XML hierarchy (HERO IMAGE)")
print(f"  6_presentation_panel.jpg - 2x2 comparison panel for slides")
print(f"  7_zoomed_detail.jpg     - Zoomed 2x detail of glyph detection")
print(f"  8_legend.jpg            - Color legend / key")
