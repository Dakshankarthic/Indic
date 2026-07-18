"""
Render final_visual overlays from PAGE-XML (lines, words, illustrations).

Matches current submission XML geometry + shows Surya/Tesseract text labels.

  python scripts/render_visuals_from_xml.py
  python scripts/render_visuals_from_xml.py --limit 50
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\indic_challenge")
IMG_DIR = ROOT / "org_img-20260701T115420Z-3-001" / "org_img"
XML_DIR = ROOT / "final_competition_xmls"
OUT_DIR = ROOT / "final_visual"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def parse_points(points_str: str) -> list[tuple[int, int]]:
    pts = []
    for pair in points_str.strip().split():
        if "," not in pair:
            continue
        x, y = pair.split(",", 1)
        pts.append((int(float(x)), int(float(y))))
    return pts


def bbox_from_points(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    if not pts:
        return 0, 0, 0, 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def get_text(el) -> str:
    te = el.find(f"./{{{PAGE_NS}}}TextEquiv/{{{PAGE_NS}}}Unicode")
    return (te.text or "").strip() if te is not None else ""


def render_page(img_path: Path, xml_path: Path, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    vis = img.copy()
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for graphic in root.findall(f".//{{{PAGE_NS}}}GraphicRegion"):
        coords = graphic.find(f"./{{{PAGE_NS}}}Coords")
        if coords is not None and coords.get("points"):
            x1, y1, x2, y2 = bbox_from_points(parse_points(coords.get("points")))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 150, 0), 3)

    for line in root.findall(f".//{{{PAGE_NS}}}TextLine"):
        coords = line.find(f"./{{{PAGE_NS}}}Coords")
        if coords is None or not coords.get("points"):
            continue
        x1, y1, x2, y2 = bbox_from_points(parse_points(coords.get("points")))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = get_text(line)
        if text:
            snippet = text[:40] + ("..." if len(text) > 40 else "")
            cv2.putText(
                vis, snippet, (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
            )

        for word in line.findall(f"./{{{PAGE_NS}}}Word"):
            wcoords = word.find(f"./{{{PAGE_NS}}}Coords")
            if wcoords is None or not wcoords.get("points"):
                continue
            wx1, wy1, wx2, wy2 = bbox_from_points(parse_points(wcoords.get("points")))
            cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (255, 0, 0), 1)

    cv2.putText(
        vis, "GREEN=Line BLUE=Word CYAN=Illus | Surya+PAGE-XML",
        (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=str(XML_DIR))
    parser.add_argument("--img-dir", default=str(IMG_DIR))
    parser.add_argument("--output", default=str(OUT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.output)

    xmls = sorted(xml_dir.glob("page_*.xml"), key=lambda p: int(p.stem.split("_")[1]))
    if args.limit > 0:
        xmls = xmls[: args.limit]

    for xml_path in tqdm(xmls, desc="Rendering visuals"):
        img_path = img_dir / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            continue
        render_page(img_path, xml_path, out_dir / img_path.name)

    print(f"Saved {len(list(out_dir.glob('*.jpg')))} images to {out_dir}")


if __name__ == "__main__":
    main()
