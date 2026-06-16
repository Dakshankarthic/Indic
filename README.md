# AutoAnn-Indic Challenge: Layout Analysis Pipeline

This repository contains our entry for the NCVPRIPG 2026 "AutoAnn-Indic" Document AI challenge. Our solution uses a cutting-edge hybrid approach combining **DINOv2** (zero-shot feature extraction) and a custom **U-Net** (6-class semantic segmentation) to achieve exceptional performance on historical Indian palm leaf manuscripts.

## Challenge Constraints & Strategy
The challenge strictly prohibits using external datasets for training. Therefore, we use a **Zero-Shot Self-Training Strategy**:
1. We use a frozen DINOv2 Vision Transformer + K-Means clustering to automatically generate "pseudo-labels" for the unlabelled pool of 1,601 manuscript images.
2. We train our U-Net model from scratch on these pseudo-labels.
3. We post-process the U-Net outputs using aggressive OpenCV polygon shrink-wrapping to minimize the "Human Effort" evaluation score.

## Architecture Pipeline

### Phase 1: Pseudo-Label Generation
`src/training/run_pseudo_label_pipeline.py`
Scans the unlabeled pool (`olai_suvadi_images` and `ramcharitmanas`) and leverages DINOv2 to detect generic layout geometry. It filters out low-confidence pages and generates 6-channel `.npz` masks containing:
- `text_region`
- `marginalia/notes`
- `illustration/diagram`
- `page_frame`
- `damage/hole`
- `text_line`

### Phase 2: U-Net Fine-Tuning
`src/training/train_unet.py`
Trains a custom 6-class U-Net on the pseudo-labels generated in Phase 1. Utilizes `torch.amp` (mixed precision) for fast training and `DiceBCELoss` to handle extreme class imbalances (e.g., small holes vs large text regions).

### Phase 3: Polygon Refinement
`src/pipeline/polygon_refiner.py`
Post-processes the raw U-Net logits using morphological closing and `cv2.approxPolyDP`. By carefully tuning the `epsilon` value, we generate mathematically tight polygons with fewer vertices, drastically reducing the time required for human annotators to fix mistakes.

### Phase 4: Final Inference & Export
`src/pipeline/final_inference.py`
Runs the trained U-Net on a test set and directly exports the predictions to the strict **PAGE-XML 2013** format required by the challenge.

---

## How to Run the Training Pipeline

We have bundled the 2.5-hour pipeline into a single batch file. It will automatically run Phase 1, Phase 2, and Phase 4 in sequence.

```bash
# Ensure you are in the project root directory
.\run_full_unet_pipeline.bat
```

> [!TIP]
> This pipeline processes 800 images to fit within a 3-hour constraint on an RTX 2070 SUPER. You can modify `MAX_IMAGES` in `run_pseudo_label_pipeline.py` if you wish to process the full 1,601 image pool.

## OCR Text Recognition
Currently, PaddleOCR is inadequate for 500-year-old cursive Devanagari. We have created preparation scripts (`prepare_paddleocr_finetuning.py` and `extract_line_crops_for_labeling.py`) to extract line crops. An automated LLM-assisted OCR fine-tuning strategy is documented in `ocr_implementation_plan.md`.
