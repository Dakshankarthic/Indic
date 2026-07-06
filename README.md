# AutoAnn-Indic Challenge — NCVPRIPG 2026 Submission


**Video Demo:** [YouTube Link](https://youtu.be/ckIwNORNI4I)

---

## Pipeline Overview

Our solution uses a geometry-first annotation approach with three stages:

### Stage 1 — Layout Detection (DINOv2 + U-Net)

- **DINOv2 Feature Extraction:** A frozen DINOv2 Vision Transformer extracts patch-level features from each page. K-Means clustering separates text regions from background automatically — adapting between printed pages (white margins) and palm leaf manuscripts (dark background).
- **U-Net Segmentation:** A custom U-Net model trained on 3,000+ document images predicts six channels: text region, marginalia, illustration, page frame, damage/holes, and text lines. The damage channel is critical for palm leaf manuscripts where holes and stains must not be confused with text.
- **Column Classification:** A binary classification algorithm analyzes vertical projection gaps to separate multi-column layouts, preserving correct reading order.
- **Illustration Detection:** Ink density and horizontal projection (valley counting) analysis on binary images classifies high-density blocks with no line structure as `GraphicRegion`.

### Stage 2 — Text Recognition (Surya OCR)

- **Surya OCR v2:** State-of-the-art transformer-based OCR model for Devanagari and English text recognition.
- **Preprocessing Pipeline:** Images undergo adaptive thresholding, CLAHE contrast enhancement, bilateral filtering, automatic deskew correction, and 6× upscaling before OCR inference.
- **GPU Optimized:** Batch processing (batch size 8) on NVIDIA RTX 2070 Super (8 GB VRAM) with float16 precision.

### Stage 3 — Word and Character Segmentation (OpenCV + Tesseract)

- **Word Bounding Boxes:** Tesseract (`--psm 7`) provides word-level bounding box coordinates.
- **Character (Akshara) Splitting:** A custom Devanagari akshara splitter uses ink density projection to locate natural gaps between connected characters, generating precise `<Glyph>` bounding boxes without an extra AI model.
- **Morphological Refinement:** Adaptive thresholding and morphological operations handle stains, uneven lighting, and degraded paper.

---

## Evaluation Metrics

Evaluated on all 1,054 test pages:

| Metric              | Value   |
|---------------------|---------|
| **Overall Accuracy** | 84.50%  |
| **CER**             | 15.32%  |
| **WER**             | 9.97%   |

### Evaluation Scripts

- `src/pipeline/evaluate_cer_wer.py` — Computes CER and WER using Levenshtein distance against ground truth.
- `src/pipeline/evaluate_human_score.py` — Estimates human annotation effort required to correct the output.

---

## Output Format

The pipeline outputs standard **PAGE-XML 2013** files with the full hierarchy:

```
PcGts → Page → TextRegion → TextLine → Word → Glyph
```

- **TextRegion:** Detected text blocks with polygon coordinates.
- **TextLine:** Individual lines with Unicode text.
- **Word:** Word-level bounding boxes with text content.
- **Glyph:** Character-level bounding boxes for each akshara.
- **GraphicRegion:** Illustrations and non-text areas.

All output is fully compatible with the **Aletheia** ground-truth editor.

---

## Directory Structure

```
indic_challenge/
├── src/pipeline/           # Core pipeline code
│   ├── pipeline_master.py  # Main entry point
│   ├── dino_layout_step1.py
│   ├── evaluate_cer_wer.py
│   └── evaluate_human_score.py
├── scripts/                # Utility scripts
│   ├── run_local_045.py    # GPU-optimized Surya OCR runner
│   ├── fast_xml_generator.py
│   ├── surya_xml_pipeline.py
│   └── batch_evaluate.py
├── submissionxml/          # Final PAGE-XML output (1054 files)
├── surya_visual/           # Visual overlays with bounding boxes
└── README.md
```

---

## How to Run

```bash
# Step 1: Run Surya OCR (GPU required)
python scripts/run_local_045.py

# Step 2: Generate PAGE-XML with bounding boxes
python scripts/fast_xml_generator.py

# Step 3: Evaluate CER/WER (requires ground truth JSON)
python src/pipeline/evaluate_cer_wer.py --pred_dir submissionxml --gt_json path/to/ground_truth.json
```

---

## Requirements

- Python 3.12
- NVIDIA GPU (8 GB+ VRAM recommended)
- Tesseract OCR with Sanskrit/Hindi tessdata
- Key packages: `surya-ocr`, `torch`, `opencv-python`, `scikit-learn`, `Levenshtein`, `Pillow`
