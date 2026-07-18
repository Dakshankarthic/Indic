import argparse
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOC = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
)


def box_to_poly(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def points(poly):
    return " ".join(f"{int(x)},{int(y)}" for x, y in poly)


def parse_box(obj):
    box = obj.find("bndbox")
    return (
        int(float(box.findtext("xmin", "0"))),
        int(float(box.findtext("ymin", "0"))),
        int(float(box.findtext("xmax", "0"))),
        int(float(box.findtext("ymax", "0"))),
    )


def overlap_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ox1, oy1 = max(ax1, bx1), max(ay1, by1)
    ox2, oy2 = min(ax2, bx2), min(ay2, by2)
    if ox2 <= ox1 or oy2 <= oy1:
        return 0
    return (ox2 - ox1) * (oy2 - oy1)


def choose_parent(child_box, parent_boxes):
    if not parent_boxes:
        return None
    cx = (child_box[0] + child_box[2]) / 2
    cy = (child_box[1] + child_box[3]) / 2
    best_idx = None
    best_score = -1
    for idx, pbox in enumerate(parent_boxes):
        px1, py1, px2, py2 = pbox
        inside = px1 <= cx <= px2 and py1 <= cy <= py2
        score = overlap_area(child_box, pbox)
        if inside:
            score += 10_000_000
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def add_coords(parent, box):
    coords = ET.SubElement(parent, f"{{{PAGE_NS}}}Coords")
    coords.set("points", points(box_to_poly(box)))
    return coords


def add_text_equiv(parent, text):
    if not text:
        return
    text_equiv = ET.SubElement(parent, f"{{{PAGE_NS}}}TextEquiv")
    unicode_el = ET.SubElement(text_equiv, f"{{{PAGE_NS}}}Unicode")
    unicode_el.text = text


def convert_one(xml_path, out_path):
    src = ET.parse(xml_path).getroot()
    filename = src.findtext("filename", f"{xml_path.stem}.jpg")
    size = src.find("size")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    grouped = {"text_line": [], "word": [], "character": [], "illustration": []}
    for obj in src.findall("object"):
        name = obj.findtext("name", "")
        if name not in grouped:
            continue
        grouped[name].append(
            {
                "box": parse_box(obj),
                "text": obj.findtext("text", ""),
            }
        )

    ET.register_namespace("", PAGE_NS)
    ET.register_namespace("xsi", XSI_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts")
    root.set(f"{{{XSI_NS}}}schemaLocation", SCHEMA_LOC)

    metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = "AutoAnn-Indic VOC-to-PAGE converter"
    now = datetime.now().isoformat()
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Created").text = now
    ET.SubElement(metadata, f"{{{PAGE_NS}}}LastChange").text = now

    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page")
    page.set("imageFilename", filename)
    page.set("imageWidth", str(width))
    page.set("imageHeight", str(height))

    text_boxes = [item["box"] for item in grouped["text_line"]]
    if text_boxes:
        x1 = min(b[0] for b in text_boxes)
        y1 = min(b[1] for b in text_boxes)
        x2 = max(b[2] for b in text_boxes)
        y2 = max(b[3] for b in text_boxes)
    else:
        x1, y1, x2, y2 = 0, 0, width, height

    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion")
    region.set("id", "region_text_0")
    add_coords(region, (x1, y1, x2, y2))

    words_by_line = {i: [] for i in range(len(grouped["text_line"]))}
    word_boxes = [item["box"] for item in grouped["word"]]
    for word in grouped["word"]:
        idx = choose_parent(word["box"], text_boxes)
        if idx is not None:
            words_by_line[idx].append(word)

    chars_by_word = {i: [] for i in range(len(grouped["word"]))}
    for char in grouped["character"]:
        idx = choose_parent(char["box"], word_boxes)
        if idx is not None:
            chars_by_word[idx].append(char)

    word_index = {id(item): idx for idx, item in enumerate(grouped["word"])}
    for line_idx, line in enumerate(sorted(grouped["text_line"], key=lambda item: (item["box"][1], item["box"][0]))):
        original_idx = grouped["text_line"].index(line)
        line_el = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine")
        line_el.set("id", f"line_{line_idx}")
        add_coords(line_el, line["box"])

        line_words = sorted(words_by_line.get(original_idx, []), key=lambda item: (item["box"][0], item["box"][1]))
        line_text = " ".join(w["text"] for w in line_words if w["text"])
        add_text_equiv(line_el, line_text)

        for local_word_idx, word in enumerate(line_words):
            global_word_idx = word_index[id(word)]
            word_el = ET.SubElement(line_el, f"{{{PAGE_NS}}}Word")
            word_el.set("id", f"word_{line_idx}_{local_word_idx}")
            add_coords(word_el, word["box"])
            add_text_equiv(word_el, word["text"])

            glyphs = sorted(chars_by_word.get(global_word_idx, []), key=lambda item: (item["box"][0], item["box"][1]))
            for glyph_idx, glyph in enumerate(glyphs):
                glyph_el = ET.SubElement(word_el, f"{{{PAGE_NS}}}Glyph")
                glyph_el.set("id", f"glyph_{line_idx}_{local_word_idx}_{glyph_idx}")
                add_coords(glyph_el, glyph["box"])
                add_text_equiv(glyph_el, glyph["text"])

    for idx, illustration in enumerate(grouped["illustration"]):
        graphic = ET.SubElement(page, f"{{{PAGE_NS}}}GraphicRegion")
        graphic.set("id", f"graphic_{idx}")
        add_coords(graphic, illustration["box"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description="Convert VOC-style annotation XML to PAGE-XML 2013.")
    parser.add_argument("--input", required=True, help="Folder containing VOC-style XML files.")
    parser.add_argument("--output", required=True, help="Folder to write PAGE-XML files.")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    xml_files = sorted(in_dir.glob("*.xml"))
    for xml_path in xml_files:
        convert_one(xml_path, out_dir / xml_path.name)
    print(f"Converted {len(xml_files)} files to {out_dir}")


if __name__ == "__main__":
    main()
