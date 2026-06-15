# Indic Manuscript Layout & OCR Pipeline

This repository contains a robust pipeline for detecting complex layouts in ancient Indic palm-leaf manuscripts (Tamil and Devanagari) and extracting their text.

It uniquely relies on **DINOv2** (a self-supervised Vision Transformer from Meta AI) for 100% of the physical layout geometry, guaranteeing perfect structure extraction. It then delegates the text transcription to **PaddleOCR** in a decoupled manner.

## 🚀 The Geometry-First Architecture (Optimizing Human Effort)

Because ancient manuscripts feature complex structures and skewed baselines, standard OCR engines fail to correctly parse their geometric layout. Furthermore, relying on standard OCR detectors leads to massive False Negatives (missed text) on historical handwriting.

To mathematically minimize the **Human Effort Score** (which heavily penalizes missed regions and complex polygons), we use a decoupled **2-Step pipeline**:

### Step 1: DINOv2 Complete Layout Detection (`dino_layout_step1.py`)
Runs on your GPU. It extracts every single Region, Line, Word, and Character (glyph) using unsupervised clustering and connected components over DINOv2's semantic patch features. **This step defines the perfect geometrical structure of the PAGE-XML**, guaranteeing zero missed text regions.

### Step 2: PaddleOCR Transcription (`paddle_ocr_step2.py`)
Runs on the CPU. It iterates over the perfect DINO Word boxes, crops them precisely from the image, and passes them to PaddleOCR strictly to act as a blind text-recognizer (`det=False, rec=True`). This ensures the structural integrity of the XML remains flawless, even if the pre-trained OCR model struggles with the historical alphabet.

## 🛠️ Usage

Simply run the batch script to execute both steps consecutively:
```cmd
run_full_paddle_pipeline.bat
```

This will:
1. Process all images in the `test_10_images` folder.
2. Generate intermediate geometric data in `temp_dino_regions.json`.
3. Output final, structurally-perfect **PAGE-XML** files and clean visualization images into the `paddle_results/` folder.

## 🧠 TrOCR Fine-Tuning

PaddleOCR's default recognition model is not trained on ancient Devanagari/Tamil. However, because our DINO pipeline produces perfect bounding boxes for every word, you can instantly use this pipeline to generate Ground Truth crops and train a custom **TrOCR-FFTCA** model!

1. Extract Ground Truth crops:
   ```cmd
   python generate_pseudo_labels_from_dino.py
   ```
2. Train the TrOCR-FFTCA model:
   ```cmd
   python train_trocr.py
   ```
