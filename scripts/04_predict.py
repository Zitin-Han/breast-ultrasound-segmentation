"""
推理预测脚本
对单张图像进行分割预测
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

# 目录配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "models"


def load_image(img_path: Path) -> np.ndarray:
    """加载图像"""
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def predict_with_nnunet(image: np.ndarray) -> np.ndarray:
    """使用 nnU-Net 预测"""
    try:
        # 设置环境变量
        os.environ["nnUNet_raw"] = str(DATA_DIR / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(DATA_DIR / "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = str(RESULTS_DIR)

        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
        from nnunetv2.utilities.file_tree_utilities import search_for_splits
        import tempfile

        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # 保存输入图像
            img_pil = Image.fromarray(image)
            img_pil.save(input_dir / "case_0000_0000.png")

            # 预测
            predict_from_raw_data(
                dataset_name_or_id="501",
                configuration="2d",
                fold="0",
                input_folder=str(input_dir),
                output_folder=str(output_dir),
                save_probabilities=False,
                overwrite=True
            )

            # 读取结果
            result_path = output_dir / "case_0000.png"
            if result_path.exists():
                result = Image.open(result_path)
                return np.array(result)

        return None

    except Exception as e:
        print(f"nnU-Net 预测失败：{e}")
        return None


def predict_with_simple_unet(image: np.ndarray) -> np.ndarray:
    """使用简化版 U-Net 预测（备用方案）"""
    try:
        import torch
        import torch.nn as nn

        # 简单的 U-Net 结构（用于演示）
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                # 简化的编码器-解码器结构
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 2, 1)  # 2 类：背景/结节
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        # 检查是否有训练好的模型
        model_path = RESULTS_DIR / "Dataset501_BUSI"
        if not model_path.exists():
            print("⚠ 未找到训练模型，使用阈值分割演示")
            # 使用简单的阈值分割作为演示
            binary = (image > np.mean(image) + np.std(image)).astype(np.uint8)
            return binary * 2  # 标签 2 表示结节

        return None

    except Exception as e:
        print(f"简化版预测失败：{e}")
        return None


def create_visualization(image: np.ndarray, mask: np.ndarray, output_path: Path):
    """创建可视化图像"""
    # 转换为 PIL Image
    if len(image.shape) == 2:
        img_color = Image.fromarray(image).convert('RGB')
    else:
        img_color = Image.fromarray(image)

    # 创建掩码叠加
    overlay = img_color.copy()
    overlay_draw = ImageDraw.Draw(overlay)

    # 在掩码区域绘制红色轮廓
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    # 简化可视化：直接用颜色标记
    mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 128)
    mask_rgb = Image.merge('RGB', [mask_pil, mask_pil, mask_pil])

    # 叠加
    overlay = Image.blend(overlay, mask_rgb, alpha=0.3)

    # 保存
    overlay.save(output_path)
    print(f"✓ 可视化已保存：{output_path}")


def save_mask(mask: np.ndarray, output_path: Path):
    """保存掩码"""
    mask_img = Image.fromarray(mask.astype(np.uint8) * 127)
    mask_img.save(output_path)
    print(f"✓ 掩码已保存：{output_path}")


def print_statistics(mask: np.ndarray):
    """打印统计信息"""
    lesion_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    lesion_ratio = lesion_pixels / total_pixels * 100

    # 判断类别
    if np.any(mask == 2):
        category = "恶性"
    elif np.any(mask == 1):
        category = "良性"
    else:
        category = "无结节"

    print(f"\n分割结果统计：")
    print(f"  类别：{category}")
    print(f"  结节面积：{lesion_pixels} 像素")
    print(f"  占图像比例：{lesion_ratio:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="乳腺结节分割预测")
    parser.add_argument("input", help="输入图像路径")
    parser.add_argument("-o", "--output", default="output.png", help="输出掩码路径")
    parser.add_argument("-v", "--visualize", action="store_true", help="生成可视化图像")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"✗ 输入文件不存在：{input_path}")
        sys.exit(1)

    print("=" * 50)
    print("乳腺超声结节分割预测")
    print("=" * 50)
    print(f"输入图像：{input_path}")

    # 加载图像
    image = load_image(input_path)
    print(f"图像尺寸：{image.shape}")

    # 预测
    print("\n正在预测...")
    mask = predict_with_nnunet(image)

    if mask is None:
        print("nnU-Net 预测失败，尝试简化版...")
        mask = predict_with_simple_unet(image)

    if mask is None:
        print("✗ 预测失败")
        sys.exit(1)

    # 保存掩码
    save_mask(mask, output_path)

    # 统计信息
    print_statistics(mask)

    # 可视化
    if args.visualize:
        vis_path = output_path.with_name(f"{output_path.stem}_overlay.png")
        create_visualization(image, mask, vis_path)


if __name__ == "__main__":
    main()
