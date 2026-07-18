"""
Run Surya OCR on test images (optionally upscaled) and compute CER/WER.

Usage:
  # Quick test on 10 pages (512px originals)
  python scripts/run_surya_hires.py --limit 10

  # Upscale 512 -> 1536 and run (modest gain; true high-res scans are better)
  python scripts/run_surya_hires.py --upscale 3 --limit 20

  # Full run after placing high-res images in org_img_hires/
  python scripts/run_surya_hires.py --img-dir D:\\indic_challenge\\org_img_hires --output surya_predictions.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import Levenshtein
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
os.environ.setdefault(
    "LLAMA_CPP_BINARY", r"d:\indic_challenge\llama_bin\llama-server.exe"
)


def extract_text(result) -> str:
    lines: list[str] = []
    if hasattr(result, "text_lines") and result.text_lines:
        for line in result.text_lines:
            if line.text and line.text.strip():
                lines.append(line.text.strip())
        return "\n".join(lines)

    if hasattr(result, "blocks"):
        for block in result.blocks:
            if hasattr(block, "html"):
                text = re.sub(r"<[^>]+>", "", block.html).strip()
                if text:
                    lines.append(text)
            elif hasattr(block, "lines"):
                for line in block.lines:
                    if line.text.strip():
                        lines.append(line.text.strip())
    return "\n".join(lines)


def load_gt(gt_json: Path) -> dict[str, str]:
    with open(gt_json, encoding="utf-8") as f:
        data = json.load(f)
    gt_map: dict[str, str] = {}
    for item in data:
        if item.get("images"):
            stem = Path(item["images"][0]).stem
            gt_map[stem] = item.get("output", "").strip()
    return gt_map


def open_image(path: Path, upscale: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if upscale <= 1:
        return img
    w, h = img.size
    return img.resize((w * upscale, h * upscale), Image.LANCZOS)


def compute_metrics(pred: str, gt: str) -> tuple[float, float]:
    cer = Levenshtein.distance(pred, gt) / max(len(gt), 1)
    pred_words = pred.split()
    gt_words = gt.split()
    wer = Levenshtein.distance(pred_words, gt_words) / max(len(gt_words), 1)
    return cer, wer


def main() -> None:
    parser = argparse.ArgumentParser(description="Surya OCR + CER/WER evaluation")
    parser.add_argument(
        "--img-dir",
        default=r"d:\indic_challenge\org_img-20260701T115420Z-3-001\org_img",
        help="Folder with page_*.jpg images",
    )
    parser.add_argument(
        "--gt-json",
        default=r"C:\Users\DK11\Downloads\hindi_ocr.json",
        help="Ground-truth JSON (hindi_ocr.json)",
    )
    parser.add_argument(
        "--output",
        default=r"d:\indic_challenge\surya_predictions.json",
        help="Where to save predictions + metrics",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="Multiply width/height (1=original, 3 turns 512->1536)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = all pages")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    gt_map = load_gt(Path(args.gt_json))
    page_names = sorted(
        gt_map.keys(),
        key=lambda x: int(x.replace("page_", "")) if x.replace("page_", "").isdigit() else x,
    )
    if args.limit > 0:
        page_names = page_names[: args.limit]

    existing = [p for p in page_names if (img_dir / f"{p}.jpg").exists()]
    missing = len(page_names) - len(existing)
    if missing:
        print(f"Warning: {missing} pages missing from {img_dir}")

    from surya.recognition import RecognitionPredictor

    print(f"Images: {img_dir}")
    print(f"Pages: {len(existing)} | Upscale: {args.upscale}x | Batch: {args.batch_size}")
    print("Loading Surya (llamacpp)...")
    rec = RecognitionPredictor()

    per_page: list[dict] = []
    total_gt_chars = 0
    total_edit_chars = 0
    total_gt_words = 0
    total_edit_words = 0
    t0 = time.time()

    for batch_start in range(0, len(existing), args.batch_size):
        batch_names = existing[batch_start : batch_start + args.batch_size]
        images = [open_image(img_dir / f"{n}.jpg", args.upscale) for n in batch_names]
        preds = rec(images)

        for name, result in zip(batch_names, preds):
            pred_text = extract_text(result)
            gt_text = gt_map[name]
            cer, wer = compute_metrics(pred_text, gt_text)

            total_gt_chars += len(gt_text)
            total_edit_chars += Levenshtein.distance(pred_text, gt_text)
            gw, pw = gt_text.split(), pred_text.split()
            total_gt_words += len(gw)
            total_edit_words += Levenshtein.distance(pw, gw)

            per_page.append(
                {
                    "page": name,
                    "cer": round(cer, 4),
                    "wer": round(wer, 4),
                    "pred_chars": len(pred_text),
                    "gt_chars": len(gt_text),
                    "pred_lines": pred_text.count("\n") + 1 if pred_text else 0,
                    "text": pred_text,
                }
            )

        done = batch_start + len(batch_names)
        micro_cer = total_edit_chars / max(total_gt_chars, 1)
        print(
            f"  [{done}/{len(existing)}] micro-CER={micro_cer*100:.1f}% "
            f"elapsed={time.time()-t0:.0f}s"
        )

    micro_cer = total_edit_chars / max(total_gt_chars, 1)
    micro_wer = total_edit_words / max(total_gt_words, 1)
    macro_cer = sum(p["cer"] for p in per_page) / max(len(per_page), 1)
    macro_wer = sum(p["wer"] for p in per_page) / max(len(per_page), 1)

    summary = {
        "model": "Surya OCR (llamacpp)",
        "img_dir": str(img_dir),
        "upscale": args.upscale,
        "pages": len(per_page),
        "micro_cer": round(micro_cer, 4),
        "micro_wer": round(micro_wer, 4),
        "macro_cer": round(macro_cer, 4),
        "macro_wer": round(macro_wer, 4),
        "overall_accuracy": round((1 - micro_cer) * 100, 2),
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_page": per_page,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Pages:            {len(per_page)}")
    print(f"Micro CER:        {micro_cer*100:.2f}%")
    print(f"Micro WER:        {micro_wer*100:.2f}%")
    print(f"Overall accuracy: {(1-micro_cer)*100:.2f}%")
    print(f"Saved:            {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
