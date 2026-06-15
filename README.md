# DINOv2 Layout Analysis for Indic Manuscripts

Automated text line, word, and character detection for historical Indic palm-leaf manuscripts using **DINOv2** (Vision Transformer) as the sole backbone.

## Architecture

```
Input Image
    │
    ▼
┌──────────────────────────────┐
│  DINOv2 ViT-B/14 Backbone    │   ← Frozen pretrained model
│  Extract patch token features │   ← 768-dim feature per 14×14 patch
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  K-Means Feature Clustering   │   ← Unsupervised: text vs background
│  k=3 (text, leaf, dark bg)    │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Text Mask → Projection       │   ← Horizontal projection on DINO mask
│  Valley Detection → Lines     │   ← Relative valley detection
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Word Detection               │   ← Vertical projection within lines
│  Character Detection          │   ← Connected components within words
└──────────────────────────────┘
    │
    ▼
PAGE-XML + Visualization
```

## How It Works

### 1. DINOv2 Feature Extraction
Each image is fed through a frozen **DINOv2 ViT-B/14** model. The model divides the image into 14×14 pixel patches and produces a 768-dimensional feature vector for each patch. These features encode rich semantic information about what each patch contains (text, background, damage, etc.).

### 2. Unsupervised Text Segmentation
We apply **K-Means clustering (k=3)** on the patch features to separate:
- **Text regions** — ink on the manuscript
- **Leaf surface** — blank palm leaf background
- **Dark background** — the scanner/camera background

The text cluster is identified automatically as the one most concentrated in the center of the image.

### 3. Text Line Detection
The binary text mask is projected horizontally (summing pixels per row). **Valley detection** on this projection identifies gaps between text lines. We use *relative* valley detection (not absolute threshold) to handle Devanagari's shirorekha connecting line.

### 4. Word & Character Detection
- **Words**: Vertical projection within each text line finds inter-word gaps
- **Characters**: Connected component analysis within each word isolates individual glyphs

### 5. Output
- **PAGE-XML** with full hierarchy: `TextRegion → TextLine → Word → Glyph`
- **Visualization** with colored line boxes, word boxes, and character boxes

## Installation

```bash
pip install torch torchvision opencv-python numpy scikit-learn lxml tqdm pillow
```

## Usage

```bash
# Single image
python dino_text_detection.py --input path/to/image.jpg --output results/

# Folder of images
python dino_text_detection.py --input path/to/images/ --output results/
```

## Output Format

For each input image `foo.jpg`, the pipeline generates:
- `foo.xml` — PAGE-XML with TextRegion, TextLine, Word, Glyph hierarchy
- `foo_viz.jpg` — Color-coded visualization:
  - 🟢 **Green overlay**: The dense text block identified by DINOv2, refined with a pixel-perfect binarized ink mask to perfectly hug the text strokes.
  - 🔲 **Large multi-colored rectangles** (Pink, Blue, Yellow, etc.): The individual **Text Lines** (L1, L2, L3...).
  - 🟧 **Orange thin boxes**: Individual **Words** separated by gaps within a line.
  - 🟥 **Red thin boxes**: Individual **Characters / Glyphs** (found using connected component analysis of the ink strokes).
## Project Structure

```
indic_challenge/
├── dino_text_detection.py    # Main DINOv2 pipeline
├── text_detection_pipeline.py # Classical CV baseline (for comparison)
├── README.md                  # This file
├── olai_suvadi_images/        # Input dataset (1,521 images)
├── test_10_images/            # 10-image test subset
└── results/                   # Output XML + visualizations
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| DINOv2 ViT-B/14 | Smaller than ViT-L but still powerful features; fits in GPU memory |
| K-Means k=3 | Manuscripts have 3 visual zones: text, leaf, dark background |
| Relative valley detection | Devanagari shirorekha means projection never drops to zero between lines |
| Connected components for chars | Indic scripts have distinct stroke patterns per glyph |

## Requirements

- Python 3.8+
- PyTorch with CUDA (GPU recommended)
- ~2GB GPU memory for ViT-B/14
