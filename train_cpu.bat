@echo off
chcp 65001 >nul
echo ================================================
echo nnU-Net training script (CPU version - Windows)
echo ================================================
echo.

REM Set environment variables
set nnUNet_raw=data\nnUNet_raw
set nnUNet_preprocessed=data\nnUNet_preprocessed
set nnUNet_results=models

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check nnU-Net
python -c "import nnunetv2" >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] nnU-Net not installed
    echo Please run: pip install nnunetv2
    pause
    exit /b 1
)

REM Check data
if not exist "data\nnUNet_raw\Dataset501_BUSI\dataset.json" (
    echo [Error] Data not prepared
    echo Please run: python scripts\01_preprocess.py
    pause
    exit /b 1
)

echo [1/2] Preprocessing data...
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

if %errorlevel% neq 0 (
    echo [Error] Preprocessing failed
    pause
    exit /b 1
)

echo.
echo [2/2] Training model (CPU mode)...
echo Training time: approx 8-12 hours (CPU)
echo Suggestion: run overnight
echo.

REM CPU training: --device cpu -num_gpus 0
nnUNetv2_train 501 2d 0 --device cpu -num_gpus 0

echo.
echo ================================================
echo Training complete!
echo ================================================
echo.
echo Next steps:
echo   Launch Web UI: python scripts\05_web_app.py
echo   Single image prediction: python scripts\04_predict.py ^<image_path^>
echo.

pause