"""
Detailed error analysis for Hindi OCR predictions vs ground truth.
Shows per-page breakdowns, common error patterns, and sample diffs.
"""
import json
import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import unicodedata
import re

def parse_page_xml_text(xml_path):
    """Extracts all Unicode text from a PAGE-XML file, ordered by TextLine."""
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = ET.parse(xml_path)
    text_content = []
    for line in tree.findall('.//pc:TextLine', ns):
        text_equiv = line.find('./pc:TextEquiv/pc:Unicode', ns)
        if text_equiv is not None and text_equiv.text:
            text_content.append(text_equiv.text.strip())
    return "\n".join(text_content)

def levenshtein_ops(s1, s2):
    """Returns edit distance and the list of operations (insert, delete, replace, match)."""
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrace to get operations
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('replace', s1[i-1], s2[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('delete', s1[i-1], ''))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('insert', '', s2[j-1]))
            j -= 1
        else:
            break
    return dp[m][n], ops

def char_category(ch):
    """Categorize a character for error analysis."""
    cat = unicodedata.category(ch)
    if '\u0900' <= ch <= '\u097F':
        return 'Devanagari'
    elif ch.isdigit() or '\u0966' <= ch <= '\u096F':
        return 'Digit'
    elif ch.isspace():
        return 'Whitespace'
    elif cat.startswith('P') or cat.startswith('S'):
        return 'Punctuation/Symbol'
    elif ch.isascii():
        return 'ASCII'
    else:
        return 'Other'

def analyze_errors(pred_text, gt_text, max_chars=2000):
    """Analyze character-level errors between prediction and ground truth."""
    # Truncate for performance on very long texts
    p = pred_text[:max_chars]
    g = gt_text[:max_chars]
    
    dist, ops = levenshtein_ops(p, g)
    
    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    category_errors = Counter()
    
    for op_type, ch1, ch2 in ops:
        if op_type == 'replace':
            substitutions[(ch1, ch2)] += 1
            category_errors[f"sub:{char_category(ch1)}->{char_category(ch2)}"] += 1
        elif op_type == 'delete':
            deletions[ch1] += 1
            category_errors[f"del:{char_category(ch1)}"] += 1
        elif op_type == 'insert':
            insertions[ch2] += 1
            category_errors[f"ins:{char_category(ch2)}"] += 1
    
    return {
        'edit_distance': dist,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'category_errors': category_errors,
    }

def word_level_diff(pred_text, gt_text):
    """Show word-level differences."""
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    
    correct = 0
    wrong = []
    
    # Simple word alignment using set comparison
    gt_set = set(gt_words)
    pred_set = set(pred_words)
    
    missing_from_pred = gt_set - pred_set  # in GT but not in pred
    extra_in_pred = pred_set - gt_set      # in pred but not in GT
    
    return missing_from_pred, extra_in_pred

def calculate_cer_wer(pred_text, gt_text):
    """Calculate CER and WER."""
    if not gt_text and not pred_text:
        return 0.0, 0.0
    if not gt_text:
        return 1.0, 1.0
    
    # CER
    cer_dist = levenshtein_ops(pred_text, gt_text)[0]
    cer = cer_dist / max(len(gt_text), 1)
    
    # WER
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    wer_dist = levenshtein_ops(pred_words, gt_words)[0]
    wer = wer_dist / max(len(gt_words), 1)
    
    return cer, wer

def main():
    gt_json_path = r"C:\Users\DK11\Downloads\hindi_ocr.json"
    pred_dir = r"d:\indic_challenge\final_xmls"
    
    print("Loading ground truth...")
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_map = {}
    for item in gt_data:
        if item.get("images") and len(item["images"]) > 0:
            basename = os.path.splitext(os.path.basename(item["images"][0]))[0]
            gt_map[basename] = item.get("output", "").strip()
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.xml")))
    
    print(f"Found {len(pred_files)} predicted XMLs")
    print(f"Ground truth has {len(gt_map)} entries")
    print("=" * 80)
    
    all_subs = Counter()
    all_dels = Counter()
    all_ins = Counter()
    all_cat_errors = Counter()
    per_page = []
    
    total_gt_chars = 0
    total_pred_chars = 0
    total_gt_lines = 0
    total_pred_lines = 0
    empty_pred_count = 0
    
    for pred_path in pred_files:
        basename = os.path.splitext(os.path.basename(pred_path))[0]
        if basename not in gt_map:
            continue
        
        pred_text = parse_page_xml_text(pred_path).strip()
        gt_text = gt_map[basename]
        
        total_gt_chars += len(gt_text)
        total_pred_chars += len(pred_text)
        total_gt_lines += gt_text.count('\n') + 1
        total_pred_lines += pred_text.count('\n') + 1 if pred_text else 0
        
        if not pred_text:
            empty_pred_count += 1
        
        cer, wer = calculate_cer_wer(pred_text, gt_text)
        
        analysis = analyze_errors(pred_text, gt_text)
        all_subs += analysis['substitutions']
        all_dels += analysis['deletions']
        all_ins += analysis['insertions']
        all_cat_errors += analysis['category_errors']
        
        per_page.append({
            'page': basename,
            'cer': cer,
            'wer': wer,
            'gt_len': len(gt_text),
            'pred_len': len(pred_text),
            'gt_lines': gt_text.count('\n') + 1,
            'pred_lines': pred_text.count('\n') + 1 if pred_text else 0,
        })
    
    # ======== REPORT ========
    print("\n" + "=" * 80)
    print("                    DETAILED ERROR ANALYSIS REPORT")
    print("=" * 80)
    
    # 1. Overview
    print(f"\n--- OVERVIEW ---")
    print(f"  Pages evaluated     : {len(per_page)}")
    print(f"  Empty predictions   : {empty_pred_count}")
    print(f"  Total GT chars      : {total_gt_chars:,}")
    print(f"  Total Pred chars    : {total_pred_chars:,}")
    print(f"  Char ratio (P/GT)   : {total_pred_chars/max(total_gt_chars,1):.2f}")
    print(f"  Total GT lines      : {total_gt_lines:,}")
    print(f"  Total Pred lines    : {total_pred_lines:,}")
    
    # 2. Worst pages
    sorted_pages = sorted(per_page, key=lambda x: x['cer'], reverse=True)
    print(f"\n--- TOP 15 WORST PAGES (by CER) ---")
    print(f"  {'Page':<20} {'CER%':>8} {'WER%':>8} {'GT chars':>10} {'Pred chars':>10} {'GT lines':>10} {'Pred lines':>10}")
    print(f"  {'-'*18:<20} {'-'*6:>8} {'-'*6:>8} {'-'*8:>10} {'-'*8:>10} {'-'*8:>10} {'-'*8:>10}")
    for p in sorted_pages[:15]:
        print(f"  {p['page']:<20} {p['cer']*100:>7.1f}% {p['wer']*100:>7.1f}% {p['gt_len']:>10,} {p['pred_len']:>10,} {p['gt_lines']:>10} {p['pred_lines']:>10}")
    
    # 3. Best pages
    sorted_best = sorted(per_page, key=lambda x: x['cer'])
    print(f"\n--- TOP 10 BEST PAGES (by CER) ---")
    print(f"  {'Page':<20} {'CER%':>8} {'WER%':>8} {'GT chars':>10} {'Pred chars':>10}")
    print(f"  {'-'*18:<20} {'-'*6:>8} {'-'*6:>8} {'-'*8:>10} {'-'*8:>10}")
    for p in sorted_best[:10]:
        print(f"  {p['page']:<20} {p['cer']*100:>7.1f}% {p['wer']*100:>7.1f}% {p['gt_len']:>10,} {p['pred_len']:>10,}")
    
    # 4. Error categories
    print(f"\n--- ERROR CATEGORIES (aggregated across all pages) ---")
    for cat, cnt in all_cat_errors.most_common(20):
        print(f"  {cat:<40} : {cnt:>6}")
    
    # 5. Top substitutions (predicted char -> ground truth char)
    print(f"\n--- TOP 25 CHARACTER SUBSTITUTIONS (pred_char -> gt_char) ---")
    for (ch1, ch2), cnt in all_subs.most_common(25):
        ch1_name = unicodedata.name(ch1, repr(ch1)) if ch1.strip() else repr(ch1)
        ch2_name = unicodedata.name(ch2, repr(ch2)) if ch2.strip() else repr(ch2)
        print(f"  '{ch1}' -> '{ch2}'  ({ch1_name} -> {ch2_name})  : {cnt:>5}")
    
    # 6. Top deletions (chars in pred but not needed / chars in GT missed)
    print(f"\n--- TOP 15 MOST DELETED CHARACTERS (in pred, not in GT) ---")
    for ch, cnt in all_dels.most_common(15):
        ch_name = unicodedata.name(ch, repr(ch)) if ch.strip() else repr(ch)
        print(f"  '{ch}' ({ch_name})  : {cnt:>5}")
    
    # 7. Top insertions (chars missing from pred, should be there)
    print(f"\n--- TOP 15 MOST INSERTED CHARACTERS (in GT, missing from pred) ---")
    for ch, cnt in all_ins.most_common(15):
        ch_name = unicodedata.name(ch, repr(ch)) if ch.strip() else repr(ch)
        print(f"  '{ch}' ({ch_name})  : {cnt:>5}")
    
    # 8. CER distribution
    print(f"\n--- CER DISTRIBUTION ---")
    buckets = {'0-10%': 0, '10-20%': 0, '20-30%': 0, '30-40%': 0, '40-50%': 0,
               '50-60%': 0, '60-70%': 0, '70-80%': 0, '80-90%': 0, '90-100%': 0, '>100%': 0}
    for p in per_page:
        c = p['cer'] * 100
        if c <= 10: buckets['0-10%'] += 1
        elif c <= 20: buckets['10-20%'] += 1
        elif c <= 30: buckets['20-30%'] += 1
        elif c <= 40: buckets['30-40%'] += 1
        elif c <= 50: buckets['40-50%'] += 1
        elif c <= 60: buckets['50-60%'] += 1
        elif c <= 70: buckets['60-70%'] += 1
        elif c <= 80: buckets['70-80%'] += 1
        elif c <= 90: buckets['80-90%'] += 1
        elif c <= 100: buckets['90-100%'] += 1
        else: buckets['>100%'] += 1
    
    for bucket, count in buckets.items():
        bar = '█' * count
        print(f"  {bucket:>10} : {count:>4} pages  {bar}")
    
    # 9. Sample side-by-side for the worst page
    print(f"\n--- SAMPLE COMPARISON (worst page: {sorted_pages[0]['page']}) ---")
    worst = sorted_pages[0]['page']
    worst_pred = parse_page_xml_text(os.path.join(pred_dir, worst + '.xml')).strip()
    worst_gt = gt_map[worst]
    
    pred_lines = worst_pred.split('\n')[:10]
    gt_lines = worst_gt.split('\n')[:10]
    
    print(f"\n  PREDICTED (first 10 lines):")
    for i, line in enumerate(pred_lines):
        print(f"    {i+1:>3}: {line[:100]}")
    
    print(f"\n  GROUND TRUTH (first 10 lines):")
    for i, line in enumerate(gt_lines):
        print(f"    {i+1:>3}: {line[:100]}")
    
    # 10. Sample for a mid-range page
    mid_idx = len(sorted_pages) // 2
    mid_page = sorted_pages[mid_idx]['page']
    print(f"\n--- SAMPLE COMPARISON (mid-range page: {mid_page}, CER={sorted_pages[mid_idx]['cer']*100:.1f}%) ---")
    mid_pred = parse_page_xml_text(os.path.join(pred_dir, mid_page + '.xml')).strip()
    mid_gt = gt_map[mid_page]
    
    pred_lines = mid_pred.split('\n')[:8]
    gt_lines = mid_gt.split('\n')[:8]
    
    print(f"\n  PREDICTED (first 8 lines):")
    for i, line in enumerate(pred_lines):
        print(f"    {i+1:>3}: {line[:120]}")
    
    print(f"\n  GROUND TRUTH (first 8 lines):")
    for i, line in enumerate(gt_lines):
        print(f"    {i+1:>3}: {line[:120]}")

if __name__ == "__main__":
    main()
