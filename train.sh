#!/bin/bash
# nnU-Net 训练脚本 (Linux/Mac)

echo "================================================"
echo "nnU-Net 训练脚本"
echo "================================================"
echo ""

# 设置环境变量
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="models"

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "[错误] 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

# 检查 nnU-Net
python -c "import nnunetv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[错误] nnU-Net 未安装"
    echo "请运行：pip install nnunetv2"
    exit 1
fi

# 检查数据
if [ ! -f "data/nnUNet_raw/Dataset501_BUSI/dataset.json" ]; then
    echo "[错误] 数据未准备好"
    echo "请先运行：python scripts/01_preprocess.py"
    exit 1
fi

echo "[1/2] 预处理数据..."
python scripts/02_train.py --fold 0

if [ $? -ne 0 ]; then
    echo "[错误] 预处理失败"
    exit 1
fi

echo ""
echo "[2/2] 训练模型..."
echo "训练时间：约 1-2 小时（GPU）"
echo ""

# 运行训练
nnUNetv2_train 501 2d 0

echo ""
echo "================================================"
echo "训练完成！"
echo "================================================"
echo ""
echo "下一步："
echo "  启动 Web 界面：streamlit run scripts/05_web_app.py"
echo "  单图预测：python scripts/04_predict.py <图像路径>"
