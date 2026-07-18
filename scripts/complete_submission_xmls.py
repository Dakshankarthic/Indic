"""
Bootstrap + resume full 1054-page PAGE-XML generation with OCR text.

1. Copy any existing XMLs that already have Unicode text into final_competition_xmls
2. Run pipeline_master.py for remaining pages (skips completed)

Usage:
  python scripts/complete_submission_xmls.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\indic_challenge")
INPUT_DIR = ROOT / "org_img-20260701T115420Z-3-001" / "org_img"
OUTPUT_DIR = ROOT / "final_competition_xmls"
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

SOURCE_DIRS = [
    ROOT / "final_xmls_pagexml",
    ROOT / "final_xmls",
    ROOT / "test_improved_xmls",
]


def has_text(xml_path: Path) -> bool:
    if not xml_path.exists():
        return False
    try:
        tree = ET.parse(xml_path)
        for line in tree.findall(f".//{{{PAGE_NS}}}TextLine"):
            te = line.find(f"./{{{PAGE_NS}}}TextEquiv/{{{PAGE_NS}}}Unicode")
            if te is not None and te.text and te.text.strip():
                return True
    except Exception:
        return False
    return False


def bootstrap_copies() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    images = sorted(INPUT_DIR.glob("page_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))

    for img in images:
        out = OUTPUT_DIR / f"{img.stem}.xml"
        if has_text(out):
            continue
        for src_dir in SOURCE_DIRS:
            candidate = src_dir / f"{img.stem}.xml"
            if has_text(candidate):
                shutil.copy2(candidate, out)
                copied += 1
                break
    return copied


def count_status() -> tuple[int, int, int]:
    images = sorted(INPUT_DIR.glob("page_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    total = len(images)
    with_text = sum(1 for img in images if has_text(OUTPUT_DIR / f"{img.stem}.xml"))
    return total, with_text, total - with_text


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Missing input dir: {INPUT_DIR}")
        sys.exit(1)

    total, done_before, remaining_before = count_status()
    print(f"Before bootstrap: {done_before}/{total} pages with OCR text")

    copied = bootstrap_copies()
    total, done_after, remaining = count_status()
    print(f"Copied {copied} existing XMLs into {OUTPUT_DIR}")
    print(f"After bootstrap: {done_after}/{total} pages with OCR text | Remaining: {remaining}")

    if remaining == 0:
        print("All 1054 pages already have OCR text in output XML.")
        return

    print("\nStarting pipeline for remaining pages...")
    env = os.environ.copy()
    env["AUTOANN_OUTPUT_XML_DIR"] = str(OUTPUT_DIR)
    env["AUTOANN_SKIP_EXISTING"] = "1"
    env["AUTOANN_INPUT_DIR"] = str(INPUT_DIR)

    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "pipeline" / "pipeline_master.py")],
        cwd=str(ROOT),
        env=env,
    )

    total, done_final, left = count_status()
    print(f"\nFinished pipeline exit={result.returncode}")
    print(f"Final: {done_final}/{total} pages with OCR text | Remaining: {left}")

    if done_final > 0:
        print("\nRun evaluation:")
        print(
            "  python src/pipeline/evaluate_cer_wer.py "
            f"--pred_dir {OUTPUT_DIR} "
            r'--gt_json "C:\Users\DK11\Downloads\hindi_ocr.json"'
        )


if __name__ == "__main__":
    main()
