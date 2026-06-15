"""
Download OLAI SUVADI (palm-leaf manuscript) images from OPenn.
Correct approach: scrape the /data/web/ directory to find actual image filenames.
"""
import os
import requests
import re
import time

OUTPUT_DIR = r"D:\indic_challenge\olai_suvadi_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 AutoAnn-Indic/1.0"}
BASE = "https://openn.library.upenn.edu/Data/0002"

# We will download manuscripts from mscoll390_item1 to item100
MS_IDS = list(range(1, 101))

def get_image_list(ms_id):
    """Scrape the web directory to get actual image filenames."""
    url = f"{BASE}/mscoll390_item{ms_id}/data/web/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        # Extract .jpg filenames (not .xmp)
        jpgs = re.findall(r'href="([^"]+_web\.jpg)"', resp.text)
        return jpgs
    except:
        return []

def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for c in r.iter_content(8192):
                    f.write(c)
            return True
    except:
        pass
    return False

def main():
    print(f"Downloading olai suvadi images to: {OUTPUT_DIR}\n")
    total = 0
    
    for ms_id in MS_IDS:
        print(f"--- Manuscript mscoll390_item{ms_id} ---")
        images = get_image_list(ms_id)
        if not images:
            print(f"  No images found or 404. Skipping.\n")
            continue
        
        print(f"  Found {len(images)} page images. Downloading ALL...")
        for img_name in images:  # Download all pages for the manuscript
            url = f"{BASE}/mscoll390_item{ms_id}/data/web/{img_name}"
            out = os.path.join(OUTPUT_DIR, f"ms{ms_id}_{img_name}")
            
            if download(url, out):
                kb = os.path.getsize(out) / 1024
                print(f"  OK: {img_name} ({kb:.0f} KB)")
                total += 1
            else:
                print(f"  FAIL: {img_name}")
            time.sleep(0.2)
        print()
    
    print(f"\n=== DONE: {total} olai suvadi images downloaded ===")
    print(f"Folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
