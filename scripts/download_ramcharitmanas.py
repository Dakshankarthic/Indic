"""
Download Ramcharitmanas page images from Internet Archive.
Downloads individual page JPEGs directly (no need to download the full 380MB ZIP).
"""
import os
import requests
import time

OUTPUT_DIR = r"D:\indic_challenge\ramcharitmanas\images_for_labeling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Internet Archive serves individual page images via their IIIF-like endpoint
# Format: https://ia803104.us.archive.org/BookReader/BookReaderImages.php?zip=/23/items/GitaPress790/790_..._jp2.zip&file=790_..._jp2/790_..._0001.jp2&id=GitaPress790&scale=1&rotate=0

# Simpler approach: use the "leaf" image API
# https://archive.org/download/GitaPress790/page/n{PAGE_NUM}/mode/1up
# Or even simpler: direct JP2 download from the item

SOURCES = [
    {
        "name": "GitaPress790",
        "server": "ia803104.us.archive.org",
        "total_pages": 615,
        "prefix": "gp790"
    },
    {
        "name": "shri-ramcharitmanas-mool",
        "server": None,  # will check metadata first
        "total_pages": None,
        "prefix": "mool"
    }
]

def download_page_image(item_id, page_num, output_path):
    """Download a single page image from Internet Archive using their rendering API."""
    # This URL returns a JPEG of the page
    url = f"https://archive.org/download/{item_id}/page/n{page_num}.jpg"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AutoAnn-Indic-Research/1.0"}
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True  # Already downloaded
    
    try:
        resp = requests.get(url, headers=headers, timeout=30, stream=True)
        if resp.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
        else:
            print(f"  Failed page {page_num}: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  Error page {page_num}: {e}")
        return False

def main():
    # Download Gita Press 790 - first 50 pages as a starter set for labeling
    # (615 total pages, but you only need ~30-50 for initial YOLO training)
    item_id = "GitaPress790"
    num_pages = 50  # Download first 50 pages for now
    
    print(f"=== Downloading {num_pages} pages from Gita Press 790 Ramcharitmanas ===")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    success = 0
    for i in range(num_pages):
        page_num = i  # 0-indexed
        out_file = os.path.join(OUTPUT_DIR, f"gp790_page_{i:04d}.jpg")
        print(f"  Downloading page {i+1}/{num_pages}...", end=" ", flush=True)
        if download_page_image(item_id, page_num, out_file):
            size_kb = os.path.getsize(out_file) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success += 1
        else:
            print("FAILED")
        time.sleep(0.5)  # Be polite to the server
    
    print(f"\nDone! Downloaded {success}/{num_pages} pages.")
    
    # Also download from the Mool scan
    item_id2 = "shri-ramcharitmanas-mool"
    num_pages2 = 30
    
    print(f"\n=== Downloading {num_pages2} pages from Ramcharitmanas Mool ===")
    
    success2 = 0
    for i in range(num_pages2):
        out_file = os.path.join(OUTPUT_DIR, f"mool_page_{i:04d}.jpg")
        print(f"  Downloading page {i+1}/{num_pages2}...", end=" ", flush=True)
        if download_page_image(item_id2, i, out_file):
            size_kb = os.path.getsize(out_file) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            success2 += 1
        else:
            print("FAILED")
        time.sleep(0.5)
    
    print(f"\nDone! Downloaded {success2}/{num_pages2} pages.")
    print(f"\n=== TOTAL: {success + success2} images ready for labeling in Roboflow ===")
    print(f"Folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
