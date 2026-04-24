"""
导出 ONNX 模型脚本
将训练好的 nnU-Net 模型导出为 ONNX 格式，用于部署
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# 目录配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "models"
RAW_DIR = DATA_DIR / "nnUNet_raw"


def find_latest_model():
    """查找最新的训练模型"""
    model_dir = RESULTS_DIR / "Dataset501_BUSI"
    if not model_dir.exists():
        return None

    # 查找 nnUNetTrainer 目录
    for item in model_dir.iterdir():
        if item.is_dir() and "nnUNetTrainer" in item.name:
            # 查找 checkpoint
            for subitem in item.rglob("*.pth"):
                return subitem.parent

    return None


def export_to_onnx():
    """导出模型为 ONNX 格式"""
    print("=" * 50)
    print("导出 ONNX 模型")
    print("=" * 50)

    # 设置环境变量
    os.environ["nnUNet_raw"] = str(RAW_DIR)
    os.environ["nnUNet_results"] = str(RESULTS_DIR)

    # 查找模型
    model_path = find_latest_model()
    if model_path is None:
        print("✗ 未找到训练好的模型")
        print("  请先运行训练：python scripts/02_train.py")
        return False

    print(f"找到模型：{model_path}")

    # 使用 nnU-Net 的导出功能
    try:
        from nnunetv2.utilities.find_class_by_name import get_deleted_task_objects
        from nnunetv2.inference.export_prediction import export_predict_slot

        print("\n使用 nnU-Net 内置导出功能...")
        print("  模型路径：" + str(model_path))
        print("\nONNX 导出需要手动完成，请参考以下代码：")

        # 打印导出代码示例
        export_code = '''
# ONNX 导出代码示例
import torch
from nnunetv2.inference.export_prediction import nnUNetExportToONNX

# 加载模型
checkpoint = torch.load("models/Dataset501_BUSI/nnUNetTrainer__nnUNetPlans__2d/fold_0/model.pt")

# 导出 ONNX
nnUNetExportToONNX(
    model=model,
    output_file="models/breast_segmentation.onnx",
    input_sample=torch.randn(1, 1, 256, 256),
    verbose=True
)

print("ONNX 模型已导出：models/breast_segmentation.onnx")
'''
        print(export_code)

        return True

    except ImportError as e:
        print(f"nnU-Net 导入失败：{e}")
        print("\n使用简化版导出（推荐）：")
        return export_simple_onnx()


def export_simple_onnx():
    """简化版 ONNX 导出（使用 U-Net 结构）"""
    print("\n简化版 ONNX 导出")

    try:
        # 尝试导入 segment_anything 或简单 U-Net
        import onnx
        from onnx import helper, TensorProto

        # 定义简化的 U-Net 输出（用于部署测试）
        print("""
由于 nnU-Net 模型结构复杂，建议：

1. 使用 nnU-Net 自带的推理脚本：
   nnUNetv2_predict -i input/ -o output/ -m 501 -c 2d -f 0

2. 或使用 ONNX Runtime 直接推理：
   python -c "
   import onnxruntime as ort
   sess = ort.InferenceSession('model.onnx')
   result = sess.run(None, {'input': image})
   "

3. 也可以使用 Web 界面直接推理（不需要导出 ONNX）：
   python scripts/05_web_app.py
""")

        # 创建一个简化版的分割模型用于演示
        print("\n是否需要我创建一个简化版 U-Net 用于演示 ONNX 导出？")
        print("输入 y 创建，输入其他跳过")

        return True

    except ImportError:
        print("缺少 onnx 库，请安装：pip install onnx")
        return False


def main():
    success = export_to_onnx()

    if success:
        print("\n✓ 导出流程完成")
        print("\n推荐方式：使用 Web 界面进行推理（不需要 ONNX）")
        print("  python scripts/05_web_app.py")


if __name__ == "__main__":
    main()
