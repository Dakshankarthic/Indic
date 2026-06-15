# Indic Manuscript Layout & OCR Pipeline

This repository contains a robust pipeline for detecting complex layouts in ancient Indic palm-leaf manuscripts (Tamil and Devanagari) and extracting their text.

It uniquely combines **DINOv2** (a self-supervised Vision Transformer from Meta AI) for physical layout extraction with **PaddleOCR** for text transcription.

## 🚀 The Hybrid Pipeline Architecture

Because ancient manuscripts feature complex structures and skewed baselines, standard OCR engines fail to correctly parse their geometric layout. To solve this, we use a decoupled **2-Step pipeline** using **Intersection-over-Union (IoU)** matching.

### Step 1: DINOv2 Layout Detection (`dino_layout_step1.py`)
Runs on your GPU. It extracts every single Region, Line, Word, and Character (glyph) using unsupervised clustering and connected components over DINOv2's semantic patch features. It saves this perfect geometrical structure into a JSON file without attempting to transcribe it.

### Step 2: PaddleOCR IoU Mapping (`paddle_ocr_step2.py`)
Runs on the CPU. It runs PaddleOCR in its native detection+recognition mode on the full image. It mathematically compares Paddle's bounding boxes to DINO's precise line polygons using Shapely IoU (Intersection-over-Union). When a match is found, Paddle's perfect Devanagari text is injected directly into DINO's complex `<TextEquiv>` nodes.

## 🛠️ Usage

Simply run the batch script to execute both steps consecutively:
```cmd
run_full_paddle_pipeline.bat
```

This will:
1. Process all images in the `test_10_images` folder.
2. Generate intermediate geometric data in `temp_dino_regions.json`.
3. Output final, PRImA-compliant **PAGE-XML** files and visualization images into the `paddle_results/` folder.

## 🧠 TrOCR Fine-Tuning

PaddleOCR's DB Detector may occasionally miss highly dense or degraded lines. The pipeline is designed to leave the `<TextEquiv>` tags empty for those missed lines while still providing perfect geometric polygons.

You can use the provided script to extract the successfully mapped crops and pseudo-labels to train a custom **TrOCR-FFTCA** model!

1. Extract Ground Truth crops:
   ```cmd
   python generate_pseudo_labels_from_dino.py
   ```
2. Train the TrOCR-FFTCA model:
   ```cmd
   python train_trocr.py
   ```
3. Evaluate the TrOCR model:
   ```cmd
   python test_trocr.py
   ```
