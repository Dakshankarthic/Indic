"""Print final CER/WER/accuracy summary from all available outputs."""
import json
import sys
from pathlib import Path

import Levenshtein

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\indic_challenge")
GT_JSON = Path(r"C:\Users\DK11\Downloads\hindi_ocr.json")


def parse_xml_text(path: Path) -> str:
    import xml.etree.ElementTree as ET

    ns = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    tree = ET.parse(path)
    lines = []
    for line in tree.findall(".//pc:TextLine", ns):
        te = line.find("./pc:TextEquiv/pc:Unicode", ns)
        if te is not None and te.text:
            lines.append(te.text.strip())
    return "\n".join(lines)


def load_gt() -> dict[str, str]:
    with open(GT_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {
        Path(x["images"][0]).stem: x.get("output", "").strip()
        for x in data
        if x.get("images")
    }


def micro_metrics(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {}
    tc = tw = ed_c = ed_w = 0
    for pred, gt in pairs:
        tc += len(gt)
        tw += len(gt.split())
        ed_c += Levenshtein.distance(pred, gt)
        ed_w += Levenshtein.distance(pred.split(), gt.split())
    cer = ed_c / max(tc, 1)
    wer = ed_w / max(tw, 1)
    return {
        "pages": len(pairs),
        "cer_pct": cer * 100,
        "wer_pct": wer * 100,
        "accuracy_pct": (1 - cer) * 100,
    }


def eval_xml_dir(name: str, pred_dir: Path, gt: dict) -> dict | None:
    if not pred_dir.exists():
        return None
    pairs = []
    empty = 0
    for xml in sorted(pred_dir.glob("*.xml")):
        stem = xml.stem
        if stem not in gt:
            continue
        pred = parse_xml_text(xml).strip()
        if not pred:
            empty += 1
        pairs.append((pred, gt[stem]))
    m = micro_metrics(pairs)
    if m:
        m["name"] = name
        m["empty_predictions"] = empty
        m["gt_total"] = len(gt)
    return m


def main() -> None:
    gt = load_gt()
    print(f"Ground truth pages: {len(gt)}\n")

    rows = []
    for name, sub in [
        ("Tesseract pipeline (final_xmls_pagexml)", "final_xmls_pagexml"),
        ("Improved pipeline (test_improved_xmls)", "test_improved_xmls"),
        ("Layout-only (final_competition_xmls)", "final_competition_xmls"),
    ]:
        m = eval_xml_dir(name, ROOT / sub, gt)
        if m:
            rows.append(m)

    surya_path = ROOT / "surya_predictions.json"
    if surya_path.exists():
        with open(surya_path, encoding="utf-8") as f:
            s = json.load(f)
        rows.append(
            {
                "name": f"Surya OCR ({s.get('pages', 0)} pages only)",
                "pages": s["pages"],
                "cer_pct": s["micro_cer"] * 100,
                "wer_pct": s["micro_wer"] * 100,
                "accuracy_pct": s["overall_accuracy"],
                "empty_predictions": 0,
                "gt_total": len(gt),
            }
        )

    print(f"{'Pipeline':<45} {'Pages':>6} {'CER%':>8} {'WER%':>8} {'Acc%':>8} {'Empty':>6}")
    print("-" * 85)
    for r in rows:
        print(
            f"{r['name']:<45} {r['pages']:>6} {r['cer_pct']:>7.2f}% {r['wer_pct']:>7.2f}% "
            f"{r['accuracy_pct']:>7.2f}% {r.get('empty_predictions', 0):>6}"
        )

    best = max(
        (r for r in rows if r["pages"] >= 100 and r.get("empty_predictions", 0) < r["pages"]),
        key=lambda r: r["accuracy_pct"],
        default=None,
    )
    print()
    if best:
        print("BEST SUBMITTABLE OCR RESULT (most pages with text):")
        print(f"  {best['name']}")
        print(f"  Pages evaluated: {best['pages']} / {best['gt_total']}")
        print(f"  CER:      {best['cer_pct']:.2f}%")
        print(f"  WER:      {best['wer_pct']:.2f}%")
        print(f"  Accuracy: {best['accuracy_pct']:.2f}%")


if __name__ == "__main__":
    main()
