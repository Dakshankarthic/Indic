# Indic Manuscript Analysis — Comprehensive Project Analysis

## 1. Executive Summary

This project is a multi-pronged pipeline for **automated layout analysis of historical Indic palm-leaf manuscripts**, primarily written in Tamil and Devanagari scripts. It uses **DINOv2** (a self-supervised Vision Transformer from Meta AI) as its core feature extractor, supplemented by classical computer vision techniques, to detect text regions, lines, words, and individual glyphs. The project also includes infrastructure for training a custom **PyTorch U-Net** segmentation model using Kraken-generated pseudo-labels, and provides evaluation via **PAGE-XML** output compatible with the **Aletheia** ground-truth editor.

---

## 2. Project Architecture

```
                    ┌──────────────────────────┐
                    │   Manuscript Images (.jpg)│
                    │   1,521 olai suvadi pages  │
                    │   6,354 external images    │
                    └────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ dino_text_      │   │ text_detection_ │   │ pipeline.py     │
│ detection.py    │   │ pipeline.py     │   │ (DINOv2 ViT-L)  │
│ (DINOv2 ViT-B)  │   │ (Classical CV)  │   │ Tiled attention │
│ K-Means k=3     │   │ Sauvola binariz.│   │ Otsu threshold  │
│ CC-based glyphs │   │ DP-based chars  │   │ Skeleton baselines│
└───────┬─────────┘   └───────┬─────────┘   └───────┬─────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   PAGE-XML       │
                    │   + Visualization │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │ results/   │ │results_full│ │results_pipe│
     │ (10 pages) │ │ (27 pages) │ │line/ (10)  │
     └────────────┘ └────────────┘ └────────────┘
```

### Training Sub-System
```
┌───────────────────┐     ┌───────────────┐     ┌───────────────┐
│ Kraken blla       │ ──▶ │ Pseudo-Label  │ ──▶ │ Custom U-Net  │
│ Layout Analysis   │     │ .npz Masks    │     │ Training      │
└───────────────────┘     └───────────────┘     └───────────────┘
```

---

## 3. File-by-File Analysis

### 3.1 Core Pipeline Scripts

#### `dino_text_detection.py` (722 lines) — **Primary Production Pipeline**
The flagship pipeline. Uses DINOv2 ViT-B/14 as a frozen feature extractor.

| Stage | Method | Details |
|-------|--------|---------|
| Feature Extraction | DINOv2 ViT-B/14 | Resizes image to multiple of 14px, extracts 768-dim patch tokens |
| Text Segmentation | K-Means (k=3) | Clusters patches into text/leaf/dark-background using DINO features + intensity heuristics |
| Line Detection | Horizontal projection + valley detection | *Relative* valley detection handles Devanagari shirorekha; morphological closing finds text regions |
| Word Detection | Connected Components (CC) | Groups glyphs by horizontal gap threshold (avg_char_width × 0.6) |
| Character Detection | CC analysis within words | Filters shirorekha bars (width > 85% of line, height < 25% of line) |
| Polygon Extraction | `cv2.approxPolyDP` | Tight contours for lines, words, and glyphs instead of bounding boxes |
| Damage Detection | `detect_damage_and_holes()` | Finds ink blobs outside the DINO text mask |
| Output | PAGE-XML 2019-07-15 | Full hierarchy: TextRegion → TextLine → Word → Glyph |

**Key innovations:**
- Uses DINOv2 features + pixel intensity to disambiguate text from leaf (ink is always darker)
- `extract_polygon_hull()` with padding and dilation ensures tight, non-degenerate polygon contours
- Marginalia detection via width/position heuristics
- Integrates `calculate_human_effort_score()` from `aletheia_utils.py`

#### `text_detection_pipeline.py` (612 lines) — **Classical CV Baseline**
A pure classical computer vision approach for comparison.

| Stage | Method |
|-------|--------|
| Preprocessing | Gaussian blur + adaptive Gaussian threshold (Sauvola-like, blockSize=51, C=15) |
| Page Frame | Convex hull of largest dilated contour |
| Line Detection | Horizontal projection + relative valley detection (identical to DINO pipeline) |
| Word Detection | Vertical projection within each line; minimum gap = max(8px, 0.12×line_height) |
| Character Detection | DP-based vertical projection valley finding with min_char_width constraint |

**Key differences from DINO pipeline:**
- No semantic feature extraction — relies purely on binarized pixel intensities
- Uses page frame detection instead of DINO mask for region finding
- Character detection uses a dynamic programming approach to find optimal split points with cost minimization
- Does NOT produce polygon contours (only bounding boxes)
- Does NOT compute Human Effort Score

#### `pipeline.py` (304 lines) — **DINOv2 ViT-L Tiled Pipeline**
An alternative DINOv2 approach using the larger ViT-L/14 model with **tiled processing** to avoid OOM errors.

| Stage | Method |
|-------|--------|
| Feature Extraction | Tiled DINOv2 ViT-L/14 (1008px tiles, 252px overlap), L2-norm of patch tokens as attention |
| Text Segmentation | Otsu threshold on attention map (with anti-inversion fix) |
| Line Detection | `skimage.measure.label` + `regionprops`, filtered for `area > 100` and `width > height` |
| Baseline Detection | `skimage.morphology.skeletonize` on the component mask, simplified to ~10 points |
| Macro Region | `cv2.convexHull` of all line points, then `cv2.minAreaRect` |

**Key differences:**
- Uses ViT-L (larger backbone) instead of ViT-B
- Attention map via L2-norm of patch tokens (not K-Means clustering)
- No word or character detection — outputs only TextRegion and TextLine with baselines
- Tiled processing with overlap for large images; downscales images > 1500px
- Uses `shapely` for polygon simplification and validation

### 3.2 Supporting Scripts

#### `aletheia_utils.py` (85 lines)
- **`open_in_aletheia()`**: Launches the Aletheia GUI with image + XML for manual review
- **`calculate_human_effort_score()`**: Simulates the correction effort E = (w_poly × V) + (w_frag × F) + (w_fn × M)
  - V: polygon vertex count per line (proxy for adjustment complexity)
  - F: tiny CC fragments needing manual deletion
  - M: missed ink regions (false negatives requiring manual annotation)
  - Default weights: w_poly=1.0, w_frag=2.0, w_fn=5.0

#### `check_xml.py` — XML validation tool
#### `fix_xml.py` — XML repair/fix utilities
#### `edit_labels.py` — YOLO/PAGE label format editing
#### `create_reference.py` — Reference annotation creation
#### `download_olai_suvadi.py`, `download_ramcharitmanas.py`, `download_all_datasets.py` — Dataset downloaders

### 3.3 Training Infrastructure

#### `unet_model.py` (120 lines)
A standard U-Net from scratch with:
- 4 downsampling stages (64→128→256→512→1024)
- 4 upsampling stages with skip connections
- 3-channel output: Background/Region | Line Boundary | Baseline
- Trilinear or transposed-conv upsampling options

#### `dataset.py` (62 lines)
PyTorch `Dataset` for `ManuscriptDataset`:
- Loads RGB images and 3-channel `.npz` masks
- Resizes to 512×512
- Converts to CHW float32 tensors

#### `generate_pseudo_labels.py` (121 lines)
Uses **Kraken's blla** (Baseline Layout Analysis) to generate ground-truth segmentation masks:
- Channel 0: Text Regions (filled polygons)
- Channel 1: Text Line boundaries (filled polygons)
- Channel 2: Baselines (drawn as 3px wide lines)
- Saves as `.npz` files for U-Net training

#### `kraken_pipeline.py` — Kraken blla pipeline wrapper
#### `debug_kmeans.py` — Debugging tool for K-Means clustering visualization

---

## 4. Dataset Composition

### 4.1 Primary Dataset: Olai Suvadi (Palm Leaf Manuscripts)
| Metric | Value |
|--------|-------|
| Total images | **1,521** |
| Unique manuscripts | **21** |
| Average pages/manuscript | ~72 |
| Largest manuscript | ms1_1985 (900 pages — 59% of dataset) |
| Significant manuscripts | ms4_3166 (271), ms2_6374 (86), ms3_2278 (53) |
| Single-page manuscripts | ms1600_3375 (2), ms100_1993 (3) |

**Full manuscript distribution:**
```
ms1_1985:   900   ms10_6345:   25   ms100_1993:   3
ms2_6374:    86   ms101_1996:   6   ms200_3623:  25
ms3_2278:    53   ms300_2810:  25   ms400_5733:  10
ms4_3166:   271   ms500_5740:  25   ms1000_1994:  7
                   ms1100_6222: 15   ms1200_5426:  6
                   ms1300_5476: 25   ms1400_4141:  9
                   ms1600_3375:  2   ms1700_2662: 13
                   ms1800_2974:  5   ms1900_3204:  5
                   ms2000_4039:  5
```

### 4.2 External Datasets
| Dataset | Images | Status |
|---------|--------|--------|
| openn_indic | 3,248 | Downloaded (jpg) |
| cbad_2017 | 2,035 | Downloaded (jpg) |
| ramcharitmanas | 1,035 | Downloaded (jpg) |
| yolo_dataset | 36 | Downloaded (jpg) |
| cvit_indic_hw | 0 | Empty |
| diva_hisdb | 0 | Empty |
| eap_palmleaf | 0 | Empty |
| indiscapes | 0 | Empty |
| sanskrit_postocr | 0 | Empty |

### 4.3 Test Subset
- `test_10_images/`: 10 images (3 from ms100_1993, 7 from ms1000_1994)

### 4.4 Additional Images
- `olai_manuscript.jpg` — Single reference image
- `olai_suvadi_images/eap_palmleaf/` — Subdirectory (EAP collection)
- `olai_suvadi_images/internet_archive_palmleaf/` — Subdirectory (IA collection)
- `ramcharitmanas/` — Additional manuscript images

---

## 5. Output Results Summary

| Output Directory | XMLs | Visualizations | Pipeline Used |
|-----------------|------|----------------|---------------|
| `results/` | 10 | 10 | `dino_text_detection.py` (10 test images) |
| `results_full/` | 27 | 27 | `dino_text_detection.py` (27 images from 4 manuscripts) |
| `results_pipeline/` | 10 | 10 | `text_detection_pipeline.py` (10 test images) |

### Sample Output Quality (ms1000_1994_0006_web, 1800×797px):
- **7 TextLines** detected (L1–L7)
- **~180 Words** across all lines
- **~180+ Glyphs** with individual bbox coordinates
- Line 1 (header): single word with 11 glyphs
- Line 2 (dense body text): 40 words with individual glyphs
- Line 7 (footer): 2 words with multi-character glyphs
- Baselines computed at y = 75% of line height

---

## 6. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Feature Extractor | DINOv2 (ViT-B/14, ViT-L/14) | PyTorch Hub |
| Clustering | scikit-learn KMeans | — |
| Image Processing | OpenCV | 4.10 |
| Binarization | Adaptive Gaussian (Sauvola-like) | OpenCV |
| Numerics | NumPy | — |
| XML Generation | lxml | 5.3 |
| Polygon Handling | shapely | 2.0 |
| Skeletonization | scikit-image | 0.24 |
| Pseudo-Labeling | Kraken blla | — |
| Deep Learning | PyTorch | ≥2.0 |
| Object Detection | Ultralytics YOLO | 8.3 |
| Visualization | PIL, OpenCV | — |
| Progress | tqdm | 4.67 |
| Evaluation | Aletheia 4.1 (Windows GUI) | — |

---

## 7. Key Design Decisions & Rationale

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| **Frozen DINOv2 backbone** | No fine-tuning needed; leverages self-supervised pretraining on diverse images | Cannot adapt to manuscript-specific domain shift |
| **K-Means k=3** | Manuscripts have 3 visual zones: ink (text), leaf (substrate), dark (scanner bg) | Over/under-segmentation possible on degraded pages |
| **Relative valley detection** | Devanagari shirorekha means projection never drops to zero between lines | May miss very tight lines; requires parameter tuning |
| **CC-based character detection** | Indic scripts have distinct stroke patterns; avoids complex segmentation | Shirorekha segments appear as single large CCs |
| **Polygon contours** (DINO pipeline) | Tight polygons better for Aletheia correction workflow than bounding boxes | More complex XML; requires validation |
| **Bounding boxes** (classical pipeline) | Simpler, faster, more compatible | Less precise for curved/cursive text |
| **ViT-B vs ViT-L** | ViT-B fits in 2GB GPU; ViT-L needs tiling for large images | ViT-L may produce better features |
| **PAGE-XML 2019-07-15** | Standard format for document layout analysis; Aletheia-compatible | Complex namespace handling |

---

## 8. Strengths

1. **Multi-approach architecture**: Three distinct pipelines (DINO-ViT-B, DINO-ViT-L, Classical CV) provide robustness and comparative analysis capability
2. **Full hierarchy output**: TextRegion → TextLine → Word → Glyph to character level
3. **Polygon-level precision**: Tight contours via `approxPolyDP` rather than crude bounding boxes
4. **Handles Indic script challenges**: Relative valley detection for shirorekha, CC filtering for header lines
5. **Quantitative evaluation**: Human Effort Score enables comparison between pipeline configurations
6. **Aletheia integration**: Direct launch into the standard ground-truth editor for manual correction
7. **Training-ready**: Pseudo-labeling via Kraken + custom U-Net for potential supervised improvement
8. **Large dataset**: 1,521 palm-leaf images across 21 manuscripts + 6,354 external Indic images

---

## 9. Limitations & Areas for Improvement

1. **Class imbalance**: ms1_1985 alone is 59% of the primary dataset; model likely overfits to this manuscript's style
2. **MS1_1985 first 27 pages mislabeled**: The file listing shows ms1_1985 followed immediately by 27 ms1_1985-like names that are likely ms1_1985 pages, not separate subdirectories—but the structure is incorrect
3. **No supervised training completed**: U-Net architecture and dataset loader exist but training loop is not implemented
4. **ViT-B vs ViT-L inconsistency**: Three pipelines use different backbones and algorithms; results are not directly comparable
5. **No validation metrics**: The Human Effort Score is a simulation, not measured against ground truth
6. **No test/train split**: No organized evaluation protocol
7. **Hardcoded paths**: `aletheia_utils.py` contains an absolute path to Aletheia on DK11's machine
8. **Fragment filtering may be aggressive**: Connected components < 6px or < 0.5×line_height are filtered, potentially removing diacritics or small glyphs (matras, anusvaras)
9. **Multiple subdirectories not organized**: `eap_palmleaf/`, `internet_archive_palmleaf/` inside `olai_suvadi_images/` are not part of the 1,521 count
10. **GPU memory concerns**: ViT-L tiled pipeline with OOM retry logic is fragile

---

## 10. Dependency Graph

```
dino_text_detection.py
├── torch (DINOv2 ViT-B/14)
├── sklearn (KMeans)
├── cv2 (OpenCV)
├── lxml (PAGE-XML)
├── numpy
├── PIL
├── tqdm
└── aletheia_utils.py
    └── subprocess (Aletheia.exe)

text_detection_pipeline.py
├── cv2 (OpenCV)
├── lxml (PAGE-XML)
├── numpy
├── PIL
└── tqdm

pipeline.py
├── torch (DINOv2 ViT-L/14)
├── shapely (polygon ops)
├── skimage (otsu, label, regionprops, skeletonize)
├── cv2 (OpenCV)
├── lxml (PAGE-XML)
└── tqdm

unet_model.py
└── torch (nn.Module)

dataset.py
├── torch (Dataset)
└── numpy

generate_pseudo_labels.py
├── kraken.blla
└── cv2
```

---

## 11. Suggested Next Steps

1. **Implement U-Net training loop**: Complete the supervised pipeline using Kraken pseudo-labels
2. **Create train/val/test split**: Stratified by manuscript to test generalization
3. **Evaluate against ground truth**: Annotate a small set of pages manually and compute IoU, precision, recall
4. **Consolidate pipelines**: Choose the best approach (likely DINO-ViT-B with K-Means) and optimize it
5. **Process full 1,521-image dataset**: Currently only 27+10=37 pages have been processed
6. **Add multi-script support**: Test on Devanagari (Ramcharitmanas), Telugu, Kannada from external datasets
7. **Improve word segmentation for Tamil**: Current CC-based approach may not handle the continuous nature of Tamil script well (no shirorekha = different gap heuristics needed)
8. **Add confidence scores**: Per-line/per-glyph confidence from DINO feature distances to cluster centroids

---

## 12. File Inventory (Complete)

| File | Lines | Purpose |
|------|-------|---------|
| `dino_text_detection.py` | 722 | Main DINOv2 ViT-B pipeline (K-Means + CC) |
| `text_detection_pipeline.py` | 612 | Classical CV pipeline (projection profiles + DP) |
| `pipeline.py` | 304 | DINOv2 ViT-L tiled pipeline (attention + Otsu) |
| `aletheia_utils.py` | 85 | Aletheia launcher + Human Effort Score |
| `unet_model.py` | 120 | Custom PyTorch U-Net (3-class output) |
| `dataset.py` | 62 | PyTorch Dataset for Manuscript images + masks |
| `generate_pseudo_labels.py` | 121 | Kraken blla pseudo-label generator |
| `kraken_pipeline.py` | — | Kraken blla wrapper |
| `check_xml.py` | — | XML validation |
| `fix_xml.py` | — | XML repair |
| `edit_labels.py` | — | Label format editing |
| `create_reference.py` | — | Reference annotation creation |
| `debug_kmeans.py` | — | K-Means debugging/visualization |
| `download_*.py` | — | Dataset downloaders (3 files) |
| `extract_pdf.py` | — | PDF extraction utility |
| `requirements.txt` | 10 | Python dependencies |
| `README.md` | 112 | Project documentation |
| `challenge_text.txt` | — | Challenge description |
| `pdf_text.txt` | — | Extracted PDF text |
| Scratch files | 7 files | Debugging/experimentation scripts |
