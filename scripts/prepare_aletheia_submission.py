"""
Build Aletheia-ready folder: each page_N.jpg next to page_N.xml (same directory).

Aletheia opens PAGE-XML when the image file named in imageFilename sits beside the XML.

  python scripts/prepare_aletheia_submission.py
  python scripts/prepare_aletheia_submission.py --open page_100   # test one page in Aletheia
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\indic_challenge")
IMG_SRC = ROOT / "org_img-20260701T115420Z-3-001" / "org_img"
XML_SRC = ROOT / "final_competition_xmls"
OUT_DIR = ROOT / "aletheia_submission"
ALETHEIA_EXE = Path(
    r"C:\Users\DK11\Downloads\Aletheia_4.1.1109\Aletheia 4.1\Aletheia.exe"
)


def fix_xml_image_ref(xml_path: Path, image_name: str) -> None:
    """Update imageFilename without rewriting namespaces (Aletheia-safe)."""
    text = xml_path.read_text(encoding="utf-8")
    import re

    new_text, n = re.subn(
        r'(<[^>]*imageFilename=")[^"]*(")',
        rf"\g<1>{image_name}\g<2>",
        text,
        count=1,
    )
    if n:
        xml_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT_DIR))
    parser.add_argument("--open", default="", help="Open one page in Aletheia, e.g. page_100")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(IMG_SRC.glob("page_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    if args.limit > 0:
        images = images[: args.limit]

    copied = 0
    for img in images:
        xml_src = XML_SRC / f"{img.stem}.xml"
        if not xml_src.exists():
            continue
        shutil.copy2(img, out / img.name)
        shutil.copy2(xml_src, out / f"{img.stem}.xml")
        fix_xml_image_ref(out / f"{img.stem}.xml", img.name)
        copied += 1

    print(f"Aletheia package ready: {out}")
    print(f"  Pairs (jpg+xml): {copied}")
    print(f"  Open in Aletheia: File -> Open -> select any .xml in this folder")

    # Fix ns0: namespace prefixes if any source files used ElementTree.write()
    norm = ROOT / "scripts" / "normalize_pagexml_for_aletheia.py"
    if norm.exists():
        subprocess.run([sys.executable, str(norm), "--dir", str(out)], check=False)

    if args.open:
        stem = args.open.replace(".xml", "").replace(".jpg", "")
        img = out / f"{stem}.jpg"
        xml = out / f"{stem}.xml"
        if not img.exists() or not xml.exists():
            print(f"Missing pair for {stem}")
            return
        if ALETHEIA_EXE.exists():
            cmd = [str(ALETHEIA_EXE), str(img), str(xml)]
            print("Launching:", " ".join(cmd))
            subprocess.Popen(cmd)
        else:
            print(f"Aletheia not found at: {ALETHEIA_EXE}")
            print(f"Manual: open {xml} in Aletheia (image must stay in same folder)")


if __name__ == "__main__":
    main()
