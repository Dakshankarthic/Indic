@echo off
echo ========================================================
echo AUTOANN-INDIC: 2.5-HOUR UNET TRAINING PIPELINE
echo ========================================================
echo.

echo [1/3] Generating pseudo-labels for 800 images (Est. 1.1 hours)
python src\training\run_pseudo_label_pipeline.py
if %errorlevel% neq 0 (
    echo Pseudo-label generation failed!
    exit /b %errorlevel%
)
echo.

echo [2/3] Training U-Net Layout Model for 20 epochs (Est. 1.5 hours)
python src\training\train_unet.py
if %errorlevel% neq 0 (
    echo U-Net Training failed!
    exit /b %errorlevel%
)
echo.

echo [3/3] Running Final Inference on test images
python src\pipeline\final_inference.py --input test_10_images --output final_results --model models\unet\unet_best.pth
if %errorlevel% neq 0 (
    echo Final Inference failed!
    exit /b %errorlevel%
)
echo.

echo ========================================================
echo PIPELINE COMPLETE!
echo Check final_results\ for the PAGE-XML outputs.
echo ========================================================
