@echo off
echo ========================================================
echo DINOv2 + PADDLE OCR DECOUPLED PIPELINE
echo ========================================================

echo.
echo [1/2] Running DINOv2 Layout Detection...
python dino_layout_step1.py --input test_10_images --output temp_dino_regions.json

if %errorlevel% neq 0 (
    echo DINOv2 Layout Detection Failed!
    exit /b %errorlevel%
)

echo.
echo [2/2] Running PaddleOCR Transcription...
python paddle_ocr_step2.py --json temp_dino_regions.json --output paddle_results

if %errorlevel% neq 0 (
    echo PaddleOCR Transcription Failed!
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo SUCCESS! Pipeline finished successfully.
echo XMLs and Visualizations saved to: paddle_results\
echo ========================================================
pause
