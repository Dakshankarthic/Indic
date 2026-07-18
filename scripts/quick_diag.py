"""Quick diagnostic: pred vs GT for sample pages."""
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")

import Levenshtein


def parse_xml(p):
    ns = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    tree = ET.parse(p)
    lines = []
    for line in tree.findall(".//pc:TextLine", ns):
        te = line.find("./pc:TextEquiv/pc:Unicode", ns)
        if te is not None and te.text:
            lines.append(te.text.strip())
    return "\n".join(lines)


def main():
    gt_json = r"C:\Users\DK11\Downloads\hindi_ocr.json"
    pred_dir = r"d:\indic_challenge\test_improved_xmls"
    img_dir = r"d:\indic_challenge\org_img-20260701T115420Z-3-001\org_img"

    with open(gt_json, encoding="utf-8") as f:
        gt = {
            os.path.splitext(os.path.basename(x["images"][0]))[0]: x["output"].strip()
            for x in json.load(f)
            if x.get("images")
        }

    out = []
    for page in ["page_1", "page_4", "page_10", "page_100"]:
        pred_path = os.path.join(pred_dir, f"{page}.xml")
        img_path = os.path.join(img_dir, f"{page}.jpg")
        g = gt.get(page, "")
        if not os.path.exists(pred_path):
            out.append(f"{page}: no pred xml")
            continue
        pred = parse_xml(pred_path)
        cer = Levenshtein.distance(pred, g) / max(len(g), 1)
        pw = pred.split()
        gw = g.split()
        wer = Levenshtein.distance(pw, gw) / max(len(gw), 1)

        img_info = "missing"
        if os.path.exists(img_path):
            from PIL import Image

            with Image.open(img_path) as im:
                img_info = f"{im.size[0]}x{im.size[1]}"

        tree = ET.parse(pred_path)
        page_el = tree.find(
            ".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page"
        )
        xml_size = (
            f"{page_el.get('imageWidth')}x{page_el.get('imageHeight')}"
            if page_el is not None
            else "?"
        )

        out.append(f"=== {page} ===")
        out.append(
            f"GT chars={len(g)} pred={len(pred)} ratio={len(pred)/max(len(g),1):.2f}"
        )
        out.append(
            f"GT lines={g.count(chr(10))+1} pred lines={pred.count(chr(10))+1 if pred else 0}"
        )
        out.append(f"GT words={len(gw)} pred words={len(pw)}")
        out.append(f"CER={cer*100:.1f}% WER={wer*100:.1f}%")
        out.append(f"Image file: {img_info}  XML page size: {xml_size}")
        out.append("GT first 5 lines:")
        for i, line in enumerate(g.split("\n")[:5]):
            out.append(f"  {i+1}: {line[:100]}")
        out.append("PRED first 5 lines:")
        for i, line in enumerate((pred or "").split("\n")[:5]):
            out.append(f"  {i+1}: {line[:100]}")
        out.append("")

    report = "\n".join(out)
    print(report)
    with open(r"d:\indic_challenge\quick_diag.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
