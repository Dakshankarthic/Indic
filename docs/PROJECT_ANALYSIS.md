# AutoAnn-Indic Technical Analysis

## Context & Constraints

The "AutoAnn-Indic" competition is graded on two distinct axes:
1. **Annotation Quality**: Strict IoU and F1-scores comparing our predicted polygon bounds against hidden ground truth.
2. **Human Effort Score**: The time it takes a human annotator to correct our predictions inside the Aletheia software tool.

Additionally, we are strictly forbidden from training on external datasets. We may only use the provided seed dataset and the unlabeled pool (1,601 manuscript pages).

## Evolving Strategy

### Deprecated Approach: Monolithic PaddleOCR
Our initial attempt relied heavily on `PaddleOCR` and `DINOv2` wrapped into a monolithic pipeline. We discovered three critical flaws:
1. PaddleOCR (`det=True`) failed to properly bound the complex, overlapping layout of historical palm-leaf manuscripts.
2. PaddleOCR's text recognition (`rec`) produced hallucinated transcriptions because it was pre-trained on modern printed Hindi fonts, rather than 500-year-old cursive script.
3. The raw polygon output contained hundreds of jagged vertices per text line, severely impacting the Human Effort Score.

### Current Approach: Segmented 24-Hour Strategy
To win the challenge under strict hardware limits (RTX 2070 SUPER, 8GB VRAM) and dataset limits, we have split the problem into two tracks:

#### Track 1: Layout Analysis (U-Net)
Instead of forcing PaddleOCR to do layout detection, we rely on semantic segmentation.
- We use a pre-trained **DINOv2** Vision Transformer to perform zero-shot clustering on the unlabeled pool. This gives us noisy, macro-level region boxes.
- We run OpenCV morphological operations to extract exact structural classes (`page_frame`, `damage/hole`, `marginalia`).
- We combine these into a 6-channel mask and use it as a "pseudo-label" to train a custom **U-Net** architecture from scratch.
- **Polygon Shrink-Wrapping**: The U-Net outputs pixel-level probabilities. To minimize the Human Effort score, we pass these probabilities through `cv2.approxPolyDP` with a highly aggressive epsilon factor (`0.002`). This guarantees the tightest possible polygon with the fewest possible vertices.

#### Track 2: OCR/Transcription (LLM Auto-Labeling)
Because PaddleOCR's `rec` model is untrained on this dialect/script, we must fine-tune it. Since manual labeling of 500 text crops is unfeasible within a 24-hour competition window, we have designed an automated training data generation pipeline:
- We extract the exact text line crops found by the U-Net.
- We feed these crops into a powerful Vision-LLM (e.g., Google Gemini 1.5 Pro) to automatically generate transcriptions.
- We use these AI-generated transcriptions to fine-tune a specialized TrOCR or PaddleOCR recognition model for 20 minutes, giving us a highly accurate domain-specific reader without any human typing effort.

## Current Pipeline Execution
The complete Track 1 (Layout Analysis) pipeline runs entirely on-device and completes in ~2.5 hours via `run_full_unet_pipeline.bat`. Output is strictly formatted to PAGE-XML 2013 standards as required by the challenge.
