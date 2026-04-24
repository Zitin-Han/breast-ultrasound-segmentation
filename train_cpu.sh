#!/bin/bash
# nnU-Net training script (CPU version - Linux/Mac)

echo "================================================"
echo "nnU-Net training script (CPU version)"
echo "================================================"
echo ""

# Set environment variables
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="models"

# Check Python
if ! command -v python &> /dev/null; then
    echo "[Error] Python not found. Please install Python 3.8+"
    exit 1
fi

# Check nnU-Net
python -c "import nnunetv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[Error] nnU-Net not installed"
    echo "Please run: pip install nnunetv2"
    exit 1
fi

# Check data
if [ ! -f "data/nnUNet_raw/Dataset501_BUSI/dataset.json" ]; then
    echo "[Error] Data not prepared"
    echo "Please run: python scripts/01_preprocess.py"
    exit 1
fi

echo "[1/2] Preprocessing data..."
python scripts/02_train_cpu.py --fold 0

if [ $? -ne 0 ]; then
    echo "[Error] Preprocessing failed"
    exit 1
fi

echo ""
echo "[2/2] Training model (CPU mode)..."
echo "Training time: approx 8-12 hours (CPU)"
echo "Suggestion: run overnight"
echo ""

# CPU training: --device cpu -num_gpus 0
nnUNetv2_train 501 2d 0 --device cpu -num_gpus 0

echo ""
echo "================================================"
echo "Training complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  Launch Web UI: streamlit run scripts/05_web_app.py"
echo "  Single image prediction: python scripts/04_predict.py <image_path>"