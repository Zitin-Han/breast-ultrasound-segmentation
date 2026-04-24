"""
训练脚本 (CPU 版本)
使用 nnU-Net 训练乳腺超声结节分割模型 —— 适配本地 CPU 环境
"""

import os
import sys
import subprocess
from pathlib import Path

# 目录配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "nnUNet_raw"
PREPROCESSED_DIR = DATA_DIR / "nnUNet_preprocessed"
RESULTS_DIR = BASE_DIR / "models"


def check_nnunet():
    """检查 nnU-Net 是否安装"""
    try:
        result = subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "--help"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_data():
    """检查数据是否准备好"""
    dataset_json = RAW_DIR / "Dataset501_BUSI" / "dataset.json"
    return dataset_json.exists()


def set_env():
    """设置 nnU-Net 环境变量"""
    os.environ["nnUNet_raw"] = str(RAW_DIR)
    os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_DIR)
    os.environ["nnUNet_results"] = str(RESULTS_DIR)

    print("nnU-Net 环境变量：")
    print(f"  nnUNet_raw:        {RAW_DIR}")
    print(f"  nnUNet_preprocessed: {PREPROCESSED_DIR}")
    print(f"  nnUNet_results:    {RESULTS_DIR}")


def preprocess():
    """预处理数据"""
    print("\n" + "=" * 50)
    print("步骤 1：预处理数据（生成计划）")
    print("=" * 50)

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", "501",           # 数据集 ID
        "-pl", "ExperimentPlanner",  # 默认规划器
        "--verify_dataset_integrity"     # 验证数据完整性
    ]

    print(f"运行：{' '.join(cmd)}")
    result = subprocess.run(cmd, env={**os.environ, **os.environ})

    if result.returncode != 0:
        print("✗ 预处理失败")
        return False

    print("✓ 预处理完成")
    return True


def train(fold: int = 0):
    """训练模型 (CPU 模式)"""
    print("\n" + "=" * 50)
    print(f"步骤 2：训练模型（Fold {fold}）")
    print("=" * 50)

    # CPU 训练参数：device cpu, 减少 batch size
    cmd = [
        "nnUNetv2_train",
        "501", "2d", str(fold),   # 数据集、2D U-Net、第 0 折
        "--device", "cpu",         # 强制使用 CPU
        "-num_gpus", "0"           # 不使用 GPU
    ]

    print(f"运行：{' '.join(cmd)}")
    print("\n⚠️ CPU 训练速度较慢，预计需要 8-12 小时")
    print("   建议：夜间挂机训练，或考虑 Colab 免费 GPU 加速")
    print()

    result = subprocess.run(cmd, env={**os.environ, **os.environ})

    if result.returncode != 0:
        print("✗ 训练失败")
        return False

    print("✓ 训练完成")
    return True


def train_all_folds():
    """训练所有折（5 折交叉验证）"""
    print("\n" + "=" * 50)
    print("训练所有折（5 折交叉验证）")
    print("=" * 50)

    for fold in range(5):
        print(f"\n--- Fold {fold} ---")
        if not train(fold):
            return False

    print("\n✓ 所有折训练完成！")
    return True


def print_next_steps():
    """打印下一步指引"""
    print("\n" + "=" * 50)
    print("训练完成！下一步操作")
    print("=" * 50)
    print("""
1. 导出 ONNX 模型（用于部署）：
   python scripts/03_export_onnx.py

2. 启动 Web 界面：
   python scripts/05_web_app.py

3. 单图预测：
   python scripts/04_predict.py <图像路径> [--output <输出路径>]
    """)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="训练 nnU-Net 模型 (CPU 版本)")
    parser.add_argument("--all", action="store_true", help="训练所有折（5 折交叉验证）")
    parser.add_argument("--fold", type=int, default=0, help="训练指定折（默认 0）")
    args = parser.parse_args()

    print("=" * 50)
    print("nnU-Net 训练脚本 (CPU 版本)")
    print("=" * 50)

    # 检查环境
    if not check_nnunet():
        print("✗ nnU-Net 未安装")
        print("  请运行：pip install nnunetv2")
        sys.exit(1)

    # 检查数据
    if not check_data():
        print("✗ 数据未准备好")
        print("  请先运行：python scripts/01_preprocess.py")
        sys.exit(1)

    # 设置环境变量
    set_env()

    # 创建目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 预处理
    if not preprocess():
        sys.exit(1)

    # 训练
    if args.all:
        train_all_folds()
    else:
        train(args.fold)

    # 打印下一步
    print_next_steps()


if __name__ == "__main__":
    main()
