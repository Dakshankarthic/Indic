"""
download_all_datasets.py
========================
Master script to scrape/download ALL datasets referenced in the
AutoAnn-Indic Challenge PDF (NCVPRIPG 2026).

Datasets covered:
  1. OPenn Indic Manuscripts (expand to items 101-390)
  2. Ramcharitmanas — complete Gita Press 790 + Mool scan
  3. EAP (Endangered Archives Programme) palm-leaf manuscripts
  4. DIVA-HisDB (medieval manuscript layout dataset)
  5. cBAD 2017 (baseline detection dataset from Zenodo)
  6. DocLayNet (IBM — subset download)
  7. PubLayNet (IBM — subset download)
  8. DocBank (metadata/sample)
  9. Indiscapes (Indic manuscript instance segmentation)
 10. CVIT Indic Handwriting word-level datasets
 11. Sanskrit Post-OCR correction dataset

Usage:
    python download_all_datasets.py                   # download everything
    python download_all_datasets.py --source openn     # single source
    python download_all_datasets.py --source ramcharitmanas
    python download_all_datasets.py --list             # list sources
"""

import os
import re
import sys
import json
import time
import zipfile
import argparse
import requests
from urllib.parse import urljoin
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\indic_challenge\datasets")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AutoAnn-Indic-Research/1.0"
}
MAX_RETRIES = 3
POLITE_DELAY = 0.3  # seconds between requests


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def download_file(url, dest, chunk_size=8192, retries=MAX_RETRIES):
    """Download a file with retry logic. Skips if already exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return True
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)
                return True
            elif r.status_code == 404:
                return False
        except Exception as e:
            if attempt == retries - 1:
                print(f"    FAIL after {retries} attempts: {e}")
        time.sleep(1)
    return False


def download_large_file(url, dest, label=""):
    """Download a large file with progress reporting."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  Already exists: {os.path.basename(dest)}")
        return True
    print(f"  Downloading {label or os.path.basename(dest)}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=300, stream=True)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}")
            return False
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    print(f"\r    {pct}% ({mb:.1f} MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════
# Source 1: OPenn Indic Manuscripts (expand beyond item 100)
# ═════════════════════════════════════════════════════════════════════
def download_openn():
    """
    Download OPenn Indic manuscript images (items 101–390).
    Items 1–100 are already in olai_suvadi_images/.
    """
    out_dir = ensure_dir(BASE_DIR / "openn_indic")
    print("\n" + "=" * 60)
    print("SOURCE 1: OPenn Indic Manuscripts (items 101–390)")
    print("=" * 60)

    openn_base = "https://openn.library.upenn.edu/Data/0002"
    total = 0

    for ms_id in range(101, 391):
        item_name = f"mscoll390_item{ms_id}"
        web_url = f"{openn_base}/{item_name}/data/web/"
        try:
            resp = requests.get(web_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            jpgs = re.findall(r'href="([^"]+_web\.jpg)"', resp.text)
            if not jpgs:
                continue
            print(f"  {item_name}: {len(jpgs)} pages")
            for img_name in jpgs:
                url = f"{web_url}{img_name}"
                out = os.path.join(out_dir, f"ms{ms_id}_{img_name}")
                if download_file(url, out):
                    total += 1
                time.sleep(POLITE_DELAY)
        except Exception as e:
            print(f"  {item_name}: error — {e}")
            continue

    print(f"\n  OPenn total new images: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 2: Ramcharitmanas (complete downloads)
# ═════════════════════════════════════════════════════════════════════
def download_ramcharitmanas():
    """
    Complete the Ramcharitmanas downloads:
    - Gita Press 790: pages 50–614 (first 50 already downloaded)
    - Mool scan: pages 30+ (first 30 already downloaded)
    """
    out_dir = ensure_dir(BASE_DIR / "ramcharitmanas")
    print("\n" + "=" * 60)
    print("SOURCE 2: Ramcharitmanas (completing full scans)")
    print("=" * 60)

    total = 0

    # Gita Press 790 — 615 pages total, start from page 50
    print("  --- Gita Press 790 (pages 50–614) ---")
    for i in range(50, 615):
        url = f"https://archive.org/download/GitaPress790/page/n{i}.jpg"
        out = os.path.join(out_dir, f"gp790_page_{i:04d}.jpg")
        if download_file(url, out):
            total += 1
            if (i - 50) % 50 == 0:
                print(f"    Page {i}/614 done...")
        time.sleep(POLITE_DELAY)

    # Mool scan — estimate ~300 pages, start from page 30
    print("  --- Ramcharitmanas Mool (pages 30+) ---")
    consecutive_fails = 0
    for i in range(30, 500):
        url = f"https://archive.org/download/shri-ramcharitmanas-mool/page/n{i}.jpg"
        out = os.path.join(out_dir, f"mool_page_{i:04d}.jpg")
        if download_file(url, out):
            total += 1
            consecutive_fails = 0
            if (i - 30) % 50 == 0:
                print(f"    Page {i} done...")
        else:
            consecutive_fails += 1
            if consecutive_fails > 10:
                print(f"    Stopping at page {i} (10 consecutive 404s)")
                break
        time.sleep(POLITE_DELAY)

    print(f"\n  Ramcharitmanas total new images: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 3: EAP (Endangered Archives Programme)
# ═════════════════════════════════════════════════════════════════════
def download_eap():
    """
    Download palm-leaf manuscript images from EAP.
    Focus on EAP636-4 (Indian manuscripts collection referenced in PDF).
    """
    out_dir = ensure_dir(BASE_DIR / "eap_palmleaf")
    print("\n" + "=" * 60)
    print("SOURCE 3: Endangered Archives Programme (EAP636-4)")
    print("=" * 60)

    # EAP provides IIIF manifests for their collections
    # Try to get the collection page and extract image links
    collection_url = "https://eap.bl.uk/collection/EAP636-4"
    total = 0

    try:
        resp = requests.get(collection_url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  Could not access EAP collection page: HTTP {resp.status_code}")
            print("  Trying IIIF manifest approach...")

        # Try to find item links on the collection page
        item_links = re.findall(r'href="(/archive-file/[^"]+)"', resp.text)
        if not item_links:
            item_links = re.findall(r'href="(/item/[^"]+)"', resp.text)

        print(f"  Found {len(item_links)} items on collection page")

        for link in item_links[:50]:  # Download first 50 items
            item_url = f"https://eap.bl.uk{link}"
            try:
                item_resp = requests.get(item_url, headers=HEADERS, timeout=30)
                # Look for image URLs (IIIF or direct)
                img_urls = re.findall(
                    r'(https://[^"]+\.(?:jpg|jpeg|png|tif))', item_resp.text, re.I
                )
                for img_url in img_urls[:5]:  # Max 5 images per item
                    fname = img_url.split("/")[-1]
                    out = os.path.join(out_dir, fname)
                    if download_file(img_url, out):
                        total += 1
                    time.sleep(POLITE_DELAY)
            except Exception:
                continue

    except Exception as e:
        print(f"  Error accessing EAP: {e}")

    # Also try a few known EAP India collections
    eap_collections = [
        "EAP676",  # Indian manuscript collection
        "EAP729",  # South Asian palm-leaf
    ]
    for coll_id in eap_collections:
        try:
            url = f"https://eap.bl.uk/collection/{coll_id}"
            resp = requests.get(url, headers=HEADERS, timeout=30)
            items = re.findall(r'href="(/archive-file/[^"]+)"', resp.text)
            if not items:
                items = re.findall(r'href="(/item/[^"]+)"', resp.text)
            print(f"  {coll_id}: found {len(items)} items")
            for link in items[:20]:
                item_url = f"https://eap.bl.uk{link}"
                try:
                    item_resp = requests.get(item_url, headers=HEADERS, timeout=30)
                    imgs = re.findall(
                        r'(https://[^"]+\.(?:jpg|jpeg|png))', item_resp.text, re.I
                    )
                    for img_url in imgs[:3]:
                        fname = f"{coll_id}_{img_url.split('/')[-1]}"
                        out = os.path.join(out_dir, fname)
                        if download_file(img_url, out):
                            total += 1
                        time.sleep(POLITE_DELAY)
                except Exception:
                    continue
        except Exception as e:
            print(f"  {coll_id} error: {e}")

    print(f"\n  EAP total images: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 4: DIVA-HisDB
# ═════════════════════════════════════════════════════════════════════
def download_diva_hisdb():
    """
    Download DIVA-HisDB medieval manuscript layout dataset.
    Source: University of Fribourg DIVA group.
    """
    out_dir = ensure_dir(BASE_DIR / "diva_hisdb")
    print("\n" + "=" * 60)
    print("SOURCE 4: DIVA-HisDB (Medieval Manuscript Layout)")
    print("=" * 60)

    # DIVA-HisDB is typically distributed as ZIP files
    # Try the official download page
    info_url = "https://diuf.unifr.ch/main/hisdoc/diva-hisdb"
    alt_urls = [
        "https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/CB55/img-CB55.zip",
        "https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/CS18/img-CS18.zip",
        "https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/CS863/img-CS863.zip",
    ]

    total = 0

    # Try direct download links
    for url in alt_urls:
        fname = url.split("/")[-1]
        dest = os.path.join(out_dir, fname)
        print(f"  Trying: {fname}")
        if download_large_file(url, dest, fname):
            total += 1
            # Extract if zip
            if fname.endswith(".zip") and os.path.getsize(dest) > 1000:
                try:
                    with zipfile.ZipFile(dest, "r") as z:
                        z.extractall(out_dir)
                    print(f"    Extracted {fname}")
                except Exception as e:
                    print(f"    Extract failed: {e}")

    # Also try the GT (ground truth) files
    gt_urls = [
        "https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/CB55/pixel-level-gt-CB55.zip",
        "https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/CS18/pixel-level-gt-CS18.zip",
        "https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/CS863/pixel-level-gt-CS863.zip",
    ]
    for url in gt_urls:
        fname = url.split("/")[-1]
        dest = os.path.join(out_dir, fname)
        print(f"  Trying GT: {fname}")
        if download_large_file(url, dest, fname):
            total += 1
            if fname.endswith(".zip") and os.path.getsize(dest) > 1000:
                try:
                    with zipfile.ZipFile(dest, "r") as z:
                        z.extractall(out_dir)
                    print(f"    Extracted {fname}")
                except Exception as e:
                    print(f"    Extract failed: {e}")

    print(f"\n  DIVA-HisDB files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 5: cBAD 2017 (Zenodo)
# ═════════════════════════════════════════════════════════════════════
def download_cbad():
    """
    Download cBAD 2017 baseline detection dataset from Zenodo.
    Record ID: 835441
    """
    out_dir = ensure_dir(BASE_DIR / "cbad_2017")
    print("\n" + "=" * 60)
    print("SOURCE 5: cBAD 2017 (Baseline Detection, Zenodo)")
    print("=" * 60)

    # Use Zenodo API to get download links
    api_url = "https://zenodo.org/api/records/835441"
    total = 0

    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])
            print(f"  Found {len(files)} files on Zenodo")
            for f in files:
                fname = f.get("key", f.get("filename", "unknown"))
                url = f.get("links", {}).get("self", "")
                if not url:
                    # Try alternate key
                    url = f"https://zenodo.org/records/835441/files/{fname}?download=1"
                size_mb = f.get("size", 0) / (1024 * 1024)
                print(f"  File: {fname} ({size_mb:.1f} MB)")

                dest = os.path.join(out_dir, fname)
                if download_large_file(url, dest, fname):
                    total += 1
                    # Extract zip files
                    if fname.endswith(".zip") and os.path.getsize(dest) > 1000:
                        try:
                            with zipfile.ZipFile(dest, "r") as z:
                                z.extractall(out_dir)
                            print(f"    Extracted {fname}")
                        except Exception as e:
                            print(f"    Extract failed: {e}")
        else:
            print(f"  Zenodo API returned HTTP {resp.status_code}")
            # Fallback: try direct download
            fallback_url = "https://zenodo.org/records/835441/files/Train-cBAD-ICDAR2017.zip?download=1"
            dest = os.path.join(out_dir, "Train-cBAD-ICDAR2017.zip")
            if download_large_file(fallback_url, dest):
                total += 1

    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n  cBAD 2017 files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 6: DocLayNet (IBM — sample subset)
# ═════════════════════════════════════════════════════════════════════
def download_doclaynet():
    """
    Download DocLayNet sample/small subset from GitHub.
    Full dataset is ~28GB; we download just the sample + annotations.
    """
    out_dir = ensure_dir(BASE_DIR / "doclaynet")
    print("\n" + "=" * 60)
    print("SOURCE 6: DocLayNet (IBM — sample subset)")
    print("=" * 60)

    total = 0

    # DocLayNet provides a small sample on GitHub
    github_urls = [
        ("https://raw.githubusercontent.com/DS4SD/DocLayNet/main/README.md", "README.md"),
    ]

    for url, fname in github_urls:
        dest = os.path.join(out_dir, fname)
        if download_file(url, dest):
            total += 1

    # Try to download the small subset from HuggingFace (more accessible)
    # DocLayNet small is ~1.6GB
    hf_url = "https://huggingface.co/datasets/ds4sd/DocLayNet-small/resolve/main/DocLayNet_small.zip"
    dest = os.path.join(out_dir, "DocLayNet_small.zip")
    print("  Attempting DocLayNet-small from HuggingFace (~1.6GB)...")
    if download_large_file(hf_url, dest, "DocLayNet_small.zip"):
        total += 1
        if os.path.getsize(dest) > 10000:
            try:
                with zipfile.ZipFile(dest, "r") as z:
                    z.extractall(out_dir)
                print("    Extracted DocLayNet_small.zip")
            except Exception as e:
                print(f"    Extract failed: {e}")

    # Save dataset info
    info = {
        "source": "DocLayNet (IBM Research)",
        "paper": "https://arxiv.org/abs/2206.01062",
        "github": "https://github.com/DS4SD/DocLayNet",
        "full_dataset_url": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
        "classes": ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
                     "Page-header", "Picture", "Section-header", "Table", "Text", "Title"],
        "annotation_format": "COCO JSON",
        "note": "Full dataset is ~28GB. Only small subset downloaded."
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  DocLayNet files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 7: PubLayNet (IBM — sample subset)
# ═════════════════════════════════════════════════════════════════════
def download_publaynet():
    """
    Download PubLayNet sample data.
    Full dataset is ~74GB; we download just the sample + annotations.
    """
    out_dir = ensure_dir(BASE_DIR / "publaynet")
    print("\n" + "=" * 60)
    print("SOURCE 7: PubLayNet (IBM — sample/annotations)")
    print("=" * 60)

    total = 0

    # PubLayNet annotations are much smaller than images
    # Try to get just the labels/annotations
    label_urls = [
        ("https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz", "labels.tar.gz"),
    ]

    for url, fname in label_urls:
        dest = os.path.join(out_dir, fname)
        print(f"  Downloading {fname}...")
        if download_large_file(url, dest, fname):
            total += 1

    # Save dataset info
    info = {
        "source": "PubLayNet (IBM Research)",
        "paper": "https://arxiv.org/abs/1908.07836",
        "github": "https://github.com/ibm-aur-nlp/PubLayNet",
        "classes": ["text", "title", "list", "table", "figure"],
        "annotation_format": "COCO JSON",
        "total_images": 360000,
        "note": "Full dataset is ~74GB. Only labels/annotations downloaded. Get images from: https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/"
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  PubLayNet files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 8: DocBank (metadata + sample)
# ═════════════════════════════════════════════════════════════════════
def download_docbank():
    """
    Download DocBank dataset info and sample.
    Full dataset requires request; we save info and sample what's public.
    """
    out_dir = ensure_dir(BASE_DIR / "docbank")
    print("\n" + "=" * 60)
    print("SOURCE 8: DocBank (metadata + sample)")
    print("=" * 60)

    total = 0

    # DocBank README from GitHub
    readme_url = "https://raw.githubusercontent.com/doc-analysis/DocBank/main/README.md"
    dest = os.path.join(out_dir, "README.md")
    if download_file(readme_url, dest):
        total += 1

    # Save dataset info
    info = {
        "source": "DocBank (Microsoft Research Asia + PKU)",
        "paper": "https://arxiv.org/abs/2006.01038",
        "github": "https://github.com/doc-analysis/DocBank",
        "classes": ["abstract", "author", "caption", "date", "equation", "figure",
                     "footer", "list", "paragraph", "reference", "section", "table", "title"],
        "annotation_format": "Custom TXT (token-level labels)",
        "total_documents": 500000,
        "note": "Full dataset requires filling a form at: https://github.com/doc-analysis/DocBank. Download instructions saved.",
        "download_instructions": (
            "1. Go to https://github.com/doc-analysis/DocBank\n"
            "2. Fill the data request form\n"
            "3. Download the 500K document images + annotations"
        )
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  DocBank info saved: {total} files")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 9: Indiscapes (ICDAR 2019)
# ═════════════════════════════════════════════════════════════════════
def download_indiscapes():
    """
    Download Indiscapes dataset — instance segmentation for Indic manuscripts.
    This is the MOST RELEVANT reference dataset for the challenge.
    """
    out_dir = ensure_dir(BASE_DIR / "indiscapes")
    print("\n" + "=" * 60)
    print("SOURCE 9: Indiscapes (Indic Manuscript Instance Segmentation)")
    print("=" * 60)

    total = 0

    # The paper and project page
    paper_url = "https://cvit.iiit.ac.in/images/ConferencePapers/2019/ICDAR_19_Indiscapes_camera_ready.pdf"
    dest = os.path.join(out_dir, "Indiscapes_ICDAR2019.pdf")
    print("  Downloading paper PDF...")
    if download_file(paper_url, dest):
        total += 1

    # Try to get the dataset from CVIT project page
    project_url = "https://cvit.iiit.ac.in/research/projects/cvit-projects/indiscapes"
    print("  Checking CVIT project page for dataset links...")
    try:
        resp = requests.get(project_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            # Look for download links
            links = re.findall(r'href="([^"]*(?:download|dataset|data|zip|tar)[^"]*)"', resp.text, re.I)
            print(f"  Found {len(links)} potential download links")
            for link in links:
                if not link.startswith("http"):
                    link = urljoin(project_url, link)
                fname = link.split("/")[-1].split("?")[0]
                if not fname:
                    fname = "indiscapes_data.zip"
                dest = os.path.join(out_dir, fname)
                if download_large_file(link, dest, fname):
                    total += 1
    except Exception as e:
        print(f"  Error: {e}")

    # Try the Indiscapes2 GitHub (newer version)
    github_urls = [
        "https://raw.githubusercontent.com/ihdia/Indiscapes2/main/README.md",
    ]
    for url in github_urls:
        fname = "Indiscapes2_README.md"
        dest = os.path.join(out_dir, fname)
        if download_file(url, dest):
            total += 1

    # Save dataset info
    info = {
        "source": "Indiscapes (CVIT, IIIT-H)",
        "paper": "https://arxiv.org/abs/1912.07025",
        "project_page": "https://cvit.iiit.ac.in/research/projects/cvit-projects/indiscapes",
        "classes": [
            "Boundary", "Physical Degradation", "Page Boundary",
            "Character Line Segment", "Character Component",
            "Picture", "Decorator", "Library Marker",
            "Boundary Line", "Median Line"
        ],
        "annotation_format": "COCO-style instance segmentation",
        "note": "Most relevant dataset for AutoAnn-Indic. Has 10 class labels for Indic manuscripts.",
        "relevance": "HIGH — only dataset with instance segmentation annotations for historical Indic manuscripts"
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Indiscapes files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 10: CVIT Indic Handwriting
# ═════════════════════════════════════════════════════════════════════
def download_cvit_indic_hw():
    """
    Download IIIT-H CVIT Indic Handwriting word-level datasets.
    """
    out_dir = ensure_dir(BASE_DIR / "cvit_indic_hw")
    print("\n" + "=" * 60)
    print("SOURCE 10: CVIT Indic Handwriting (word-level)")
    print("=" * 60)

    total = 0

    # CVIT project page
    project_url = "https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data"
    print("  Checking CVIT Indic HW project page...")
    try:
        resp = requests.get(project_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            # Look for dataset download links
            links = re.findall(r'href="([^"]*(?:download|dataset|data|zip|tar|gz)[^"]*)"', resp.text, re.I)
            print(f"  Found {len(links)} potential download links")
            for link in links[:10]:
                if not link.startswith("http"):
                    link = urljoin(project_url, link)
                fname = link.split("/")[-1].split("?")[0]
                if not fname or len(fname) < 3:
                    continue
                dest = os.path.join(out_dir, fname)
                print(f"  Downloading: {fname}")
                if download_large_file(link, dest, fname):
                    total += 1
        else:
            print(f"  HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

    # Save dataset info
    info = {
        "source": "CVIT, IIIT-H — Indic Handwriting Data",
        "project_page": "https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data",
        "description": "Word-level handwriting datasets for Devanagari and other Indic scripts",
        "scripts_covered": ["Devanagari", "Bangla", "Telugu", "Malayalam", "Kannada",
                           "Tamil", "Gujarati", "Gurumukhi", "Odia", "Urdu"],
        "annotation_format": "Word images + text labels",
        "note": "Useful as a baseline/pretraining reference for Indic script recognition"
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  CVIT Indic HW files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Source 11: Sanskrit Post-OCR Correction Dataset
# ═════════════════════════════════════════════════════════════════════
def download_sanskrit_postocr():
    """
    Download the Sanskrit Post-OCR text correction dataset.
    Paper: Findings of EMNLP 2022.
    """
    out_dir = ensure_dir(BASE_DIR / "sanskrit_postocr")
    print("\n" + "=" * 60)
    print("SOURCE 11: Sanskrit Post-OCR Correction Dataset")
    print("=" * 60)

    total = 0

    # ACL Anthology paper page
    paper_url = "https://aclanthology.org/2022.findings-emnlp.466/"
    print("  Checking ACL Anthology page for dataset...")
    try:
        resp = requests.get(paper_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            # Look for PDF and dataset links
            pdf_links = re.findall(r'href="([^"]*\.pdf)"', resp.text)
            for link in pdf_links[:1]:
                if not link.startswith("http"):
                    link = urljoin(paper_url, link)
                dest = os.path.join(out_dir, "sanskrit_postocr_paper.pdf")
                if download_file(link, dest):
                    total += 1

            # Look for GitHub/dataset links
            all_links = re.findall(r'href="([^"]*(?:github|data|code)[^"]*)"', resp.text, re.I)
            for link in all_links:
                print(f"  Found link: {link}")
    except Exception as e:
        print(f"  Error: {e}")

    # The dataset is typically on GitHub — try common patterns
    github_candidates = [
        "https://github.com/ayushbits/Sanskrit-Post-OCR",
        "https://github.com/maheshwarianukriti/Sanskrit-Post-OCR",
    ]
    for gh_url in github_candidates:
        try:
            resp = requests.get(gh_url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                print(f"  Found GitHub repo: {gh_url}")
                # Download the repo as zip
                zip_url = gh_url + "/archive/refs/heads/main.zip"
                dest = os.path.join(out_dir, "repo.zip")
                if download_large_file(zip_url, dest, "Sanskrit Post-OCR repo"):
                    total += 1
                    if os.path.getsize(dest) > 1000:
                        try:
                            with zipfile.ZipFile(dest, "r") as z:
                                z.extractall(out_dir)
                            print("    Extracted repo")
                        except Exception as e:
                            print(f"    Extract failed: {e}")
                break
        except Exception:
            continue

    # Save dataset info
    info = {
        "source": "Sanskrit Post-OCR Text Correction",
        "paper": "https://aclanthology.org/2022.findings-emnlp.466/",
        "description": "Benchmark dataset for post-OCR text correction in Sanskrit",
        "annotation_format": "Text pairs (OCR output → corrected text)",
        "note": "Useful for downstream text correction pipelines after HTR/OCR"
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Sanskrit Post-OCR files downloaded: {total}")
    return total


# ═════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═════════════════════════════════════════════════════════════════════
SOURCES = {
    "openn": ("OPenn Indic Manuscripts (101-390)", download_openn),
    "ramcharitmanas": ("Ramcharitmanas (complete)", download_ramcharitmanas),
    "eap": ("Endangered Archives Programme", download_eap),
    "diva_hisdb": ("DIVA-HisDB", download_diva_hisdb),
    "cbad": ("cBAD 2017 (Zenodo)", download_cbad),
    "doclaynet": ("DocLayNet (subset)", download_doclaynet),
    "publaynet": ("PubLayNet (annotations)", download_publaynet),
    "docbank": ("DocBank (info)", download_docbank),
    "indiscapes": ("Indiscapes", download_indiscapes),
    "cvit_hw": ("CVIT Indic Handwriting", download_cvit_indic_hw),
    "sanskrit": ("Sanskrit Post-OCR", download_sanskrit_postocr),
}


def main():
    parser = argparse.ArgumentParser(description="Download all AutoAnn-Indic challenge datasets")
    parser.add_argument("--source", type=str, default=None,
                        help="Download only this source (e.g., openn, ramcharitmanas, cbad)")
    parser.add_argument("--list", action="store_true", help="List available sources")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable sources:")
        for key, (desc, _) in SOURCES.items():
            print(f"  {key:20s} — {desc}")
        return

    ensure_dir(BASE_DIR)

    if args.source:
        if args.source not in SOURCES:
            print(f"Unknown source: {args.source}")
            print(f"Available: {', '.join(SOURCES.keys())}")
            return
        desc, func = SOURCES[args.source]
        print(f"\n{'#' * 60}")
        print(f"  Downloading: {desc}")
        print(f"{'#' * 60}")
        count = func()
        print(f"\n{'#' * 60}")
        print(f"  DONE — {desc}: {count} items")
        print(f"{'#' * 60}")
    else:
        print(f"\n{'#' * 60}")
        print(f"  AutoAnn-Indic: Downloading ALL {len(SOURCES)} Dataset Sources")
        print(f"{'#' * 60}")

        results = {}
        for key, (desc, func) in SOURCES.items():
            try:
                count = func()
                results[key] = count
            except Exception as e:
                print(f"\n  ERROR in {key}: {e}")
                results[key] = -1

        print(f"\n\n{'=' * 60}")
        print("  SUMMARY OF ALL DOWNLOADS")
        print(f"{'=' * 60}")
        for key, count in results.items():
            desc = SOURCES[key][0]
            status = f"{count} items" if count >= 0 else "FAILED"
            print(f"  {desc:45s} : {status}")
        print(f"{'=' * 60}")
        print(f"  Output directory: {BASE_DIR}")


if __name__ == "__main__":
    main()
