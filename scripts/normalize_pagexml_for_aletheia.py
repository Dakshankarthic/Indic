"""
Normalize PAGE-XML files for Aletheia compatibility.

Aletheia requires default-namespace tags (<PcGts>, <Page>, ...) not ns0: prefixes
introduced by ElementTree.write().

  python scripts/normalize_pagexml_for_aletheia.py --dir aletheia_submission
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def normalize_pagexml_text(text: str) -> str:
    # ElementTree default: ns0 prefix
    text = text.replace(f'xmlns:ns0="{PAGE_NS}"', f'xmlns="{PAGE_NS}"')
    text = re.sub(r"<ns0:([\w]+)", r"<\1", text)
    text = re.sub(r"</ns0:([\w]+)", r"</\1", text)
    text = text.replace("ns0:", "")

    # Ensure standard declaration
    if not text.lstrip().startswith("<?xml"):
        text = '<?xml version="1.0" encoding="utf-8"?>\n' + text

    # Aletheia prefers double-quoted xml declaration style like PRImA samples
    text = text.replace("<?xml version='1.0' encoding='utf-8'?>", '<?xml version="1.0" encoding="utf-8"?>')

    # Remove invalid XML control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    return text


def normalize_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    fixed = normalize_pagexml_text(original)
    if fixed != original:
        path.write_text(fixed, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Folder with PAGE-XML files")
    args = parser.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("*.xml"))
    changed = 0
    for f in files:
        if normalize_file(f):
            changed += 1
    print(f"Normalized {changed}/{len(files)} files in {root}")


if __name__ == "__main__":
    main()
