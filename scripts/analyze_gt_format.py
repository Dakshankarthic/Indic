"""Analyze hindi_ocr.json ground-truth text structure."""
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

GT = Path(r"C:\Users\DK11\Downloads\hindi_ocr.json")
OUT = Path(r"d:\indic_challenge\gt_format_analysis.txt")

with open(GT, encoding="utf-8") as f:
    data = json.load(f)

line_counts = []
nonempty_line_counts = []
wrapped_line_counts = []  # lines > 80 chars (likely wrapped prose)
verse_like = []  # lines ending with ॥

for item in data:
    out = item.get("output", "")
    lines = out.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    line_counts.append(len(lines))
    nonempty_line_counts.append(len(nonempty))
    wrapped_line_counts.append(sum(1 for ln in nonempty if len(ln) > 80))
    verse_like.append(sum(1 for ln in nonempty if "॥" in ln or ln.strip().startswith("दो०") or ln.strip().startswith("सो०") or ln.strip().startswith("छं०") or ln.strip().startswith("छ०")))

lines = []
lines.append("GROUND TRUTH FORMAT ANALYSIS (hindi_ocr.json)")
lines.append("=" * 60)
lines.append(f"Total pages: {len(data)}")
lines.append(f"Schema: instruction + input + output + images[]")
lines.append(f"Task label: {data[0]['instruction']!r}")
lines.append("")
lines.append("ANSWER: FULL-PAGE transcription stored as ONE string.")
lines.append("Lines are separated by newline (\\n), NOT separate JSON records.")
lines.append("")
lines.append("Statistics (split output by \\n):")
lines.append(f"  Total lines per page:     min={min(line_counts)}, max={max(line_counts)}, median={statistics.median(line_counts):.0f}, mean={statistics.mean(line_counts):.1f}")
lines.append(f"  Non-empty lines per page: min={min(nonempty_line_counts)}, max={max(nonempty_line_counts)}, median={statistics.median(nonempty_line_counts):.0f}")
lines.append(f"  Long wrapped lines (>80c): median={statistics.median(wrapped_line_counts):.0f} per page")
lines.append(f"  Verse-like lines (॥/दो०/छं०): median={statistics.median(verse_like):.0f} per page")
lines.append(f"  Pages with blank lines:   {sum(1 for item in data if '\\n\\n' in item.get('output',''))} / {len(data)}")
lines.append("")
lines.append("Line types mixed on each page:")
lines.append("  - Headers (e.g. * बालकाण्ड * १०१)")
lines.append("  - Poetry verses (दोहा/चौपाई ending with ॥)")
lines.append("  - Commentary prose (टीका paragraphs, often wrapped)")
lines.append("  - Index/table entries (page_4, page_10)")
lines.append("  - Blank lines between blocks")
lines.append("")
lines.append("HOW CER/WER IS COMPUTED IN YOUR SCRIPTS:")
lines.append("  pred_text = join all TextLine Unicode with \\n  OR full-page OCR string")
lines.append("  gt_text   = entire output field for that page")
lines.append("  Compare WHOLE PAGE strings (not line-by-line alignment)")
lines.append("")
lines.append("IMPLICATION:")
lines.append("  - GT is page-level, but internally line-structured via \\n")
lines.append("  - Missing/extra newlines hurt CER")
lines.append("  - Line order matters")
lines.append("  - One merged line can cause large WER/CER spikes")
lines.append("")

for name in ["page_1", "page_10", "page_100", "page_4"]:
    item = next(x for x in data if x["images"][0].endswith(f"{name}.jpg"))
    out = item["output"]
    ls = out.split("\n")
    lines.append("-" * 60)
    lines.append(f"{name}.jpg | {len(ls)} lines | {len(out)} chars | {len(out.split())} words")
    lines.append("Structure preview:")
    for i, ln in enumerate(ls[:15], 1):
        tag = ""
        s = ln.strip()
        if not s:
            tag = "[BLANK]"
        elif "॥" in s:
            tag = "[VERSE]"
        elif s.startswith("दो०") or s.startswith("सो०") or s.startswith("छं०"):
            tag = "[CHAND]"
        elif len(s) > 100:
            tag = "[PROSE]"
        elif s.startswith("*") or s.startswith("#") or s.startswith("["):
            tag = "[HEADER]"
        lines.append(f"  {i:2d} {tag:8s} {s[:90]}")

text = "\n".join(lines)
OUT.write_text(text, encoding="utf-8")
print(text)
print(f"\nSaved: {OUT}")
