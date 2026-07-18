"""Compare CER/WER across all OCR outputs in the workspace."""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import Levenshtein

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:\indic_challenge")
GT_JSON = Path(r"C:\Users\DK11\Downloads\hindi_ocr.json")
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


def load_gt():
    with open(GT_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {
        Path(x["images"][0]).stem: x.get("output", "").strip()
        for x in data
        if x.get("images")
    }


def parse_xml_text(path: Path) -> str:
    tree = ET.parse(path)
    lines = []
    for line in tree.findall(f".//{{{PAGE_NS}}}TextLine"):
        te = line.find(f"./{{{PAGE_NS}}}TextEquiv/{{{PAGE_NS}}}Unicode")
        if te is not None and te.text:
            lines.append(te.text.strip())
    return "\n".join(lines)


def micro_metrics(pairs):
    if not pairs:
        return None
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
        "cer": cer * 100,
        "wer": wer * 100,
        "accuracy": (1 - cer) * 100,
    }


def eval_xml_dir(name, pred_dir: Path, gt: dict):
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
    if not m:
        return None
    m["name"] = name
    m["empty"] = empty
    return m


def eval_surya_json(path: Path, gt: dict):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data.get("per_page", []):
        stem = item["page"]
        if stem in gt:
            pairs.append((item.get("text", "").strip(), gt[stem]))
    m = micro_metrics(pairs)
    if not m:
        return None
    m["name"] = f"Surya JSON ({path.name})"
    m["empty"] = sum(1 for p, _ in pairs if not p)
    if "micro_cer" in data and not pairs:
        m["cer"] = data["micro_cer"] * 100
        m["wer"] = data["micro_wer"] * 100
        m["accuracy"] = data.get("overall_accuracy", (1 - data["micro_cer"]) * 100)
        m["pages"] = data.get("pages", 0)
    return m


def main():
    gt = load_gt()
    results = []

    for name, sub in [
        ("Tesseract pipeline (final_xmls)", "final_xmls"),
        ("Tesseract pipeline (final_xmls_pagexml)", "final_xmls_pagexml"),
        ("Submission XML (final_competition_xmls)", "final_competition_xmls"),
        ("Improved pipeline (test_improved_xmls)", "test_improved_xmls"),
    ]:
        r = eval_xml_dir(name, ROOT / sub, gt)
        if r:
            results.append(r)

    for sj in ["surya_predictions.json", "evaluation_results.json", "surya_full_predictions.json"]:
        r = eval_surya_json(ROOT / sj, gt)
        if r:
            results.append(r)

    results.sort(key=lambda x: x["cer"])

    print("OCR COMPARISON (lower CER = better)\n")
    print(f"{'Method':<45} {'Pages':>6} {'Empty':>6} {'CER%':>8} {'WER%':>8} {'Acc%':>8}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['name']:<45} {r['pages']:>6} {r.get('empty',0):>6} "
            f"{r['cer']:>7.2f}% {r['wer']:>7.2f}% {r['accuracy']:>7.2f}%"
        )

    if results:
        best = results[0]
        print(f"\nBEST RIGHT NOW: {best['name']}")
        print(f"  CER {best['cer']:.2f}% | WER {best['wer']:.2f}% | Accuracy {best['accuracy']:.2f}%")
        print(f"  Evaluated on {best['pages']} pages")


if __name__ == "__main__":
    main()
