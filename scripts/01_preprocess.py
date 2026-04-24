"""
数据预处理脚本
将 BUSI 数据转换为 nnU-Net 格式
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# 目录配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BUSI_DIR = DATA_DIR / "BUSI"
NNUNET_RAW = DATA_DIR / "nnUNet_raw" / "Dataset501_BUSI"


def create_directories():
    """创建 nnU-Net 目录结构"""
    for subdir in ["imagesTr", "labelsTr"]:
        (NNUNET_RAW / subdir).mkdir(parents=True, exist_ok=True)
    print(f"✓ 创建目录：{NNUNET_RAW}")


def load_image(img_path: Path) -> np.ndarray:
    """加载图像为灰度数组"""
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def find_mask(img_path: Path) -> Path | None:
    """查找对应的掩码文件"""
    parent = img_path.parent

    # 尝试常见的掩码命名方式
    for suffix in ["_mask", "_gt", "_seg", "_annotation"]:
        stem = img_path.stem.replace("_mask", "").replace("_gt", "")
        candidates = [
            parent / f"{stem}{suffix}.png",
            parent / f"{stem.replace(' ', '_')}{suffix}.png",
            parent / f"{img_path.stem}{suffix}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    # 尝试在同目录下找同名掩码
    for file in parent.glob(f"{img_path.stem}*.png"):
        if "_mask" in file.name or "_gt" in file.name or "mask" in file.name:
            return file

    return None


def load_mask(mask_path: Path, target_label: int = 1) -> np.ndarray:
    """加载掩码并二值化"""
    if not mask_path.exists():
        return None

    mask = Image.open(mask_path)
    if mask.mode != 'L':
        mask = mask.convert('L')
    mask_arr = np.array(mask)

    # 二值化
    binary_mask = (mask_arr > 0).astype(np.uint8)
    return binary_mask


def convert_dataset():
    """转换 BUSI 数据到 nnU-Net 格式"""
    if not BUSI_DIR.exists():
        print(f"✗ 错误：未找到 BUSI 数据目录 {BUSI_DIR}")
        print("  请先运行：python scripts/00_download_data.py")
        return None

    categories = {
        "0_Normal": 0,
        "1_Benign": 1,
        "2_Malignant": 2
    }

    case_id = 0
    all_images = []

    print("\n开始转换数据...")

    for category, label_id in categories.items():
        category_dir = BUSI_DIR / category
        if not category_dir.exists():
            print(f"  ⚠ 跳过：{category}（目录不存在）")
            continue

        # 获取所有图像文件
        png_files = sorted(category_dir.glob("*.png"))
        image_files = [f for f in png_files if "_mask" not in f.name]

        print(f"\n处理 {category}：{len(image_files)} 张图像")

        for img_path in tqdm(image_files, desc=category):
            case_name = f"case_{case_id:04d}"

            # 加载图像并强制转灰度（nnU-Net 要求单通道）
            img_gray = load_image(img_path)
            img_dest = NNUNET_RAW / "imagesTr" / f"{case_name}_0000.png"
            Image.fromarray(img_gray, mode='L').save(img_dest)

            # 处理掩码
            mask_path = find_mask(img_path)
            if mask_path and mask_path.exists():
                mask_arr = load_mask(mask_path, label_id)
            else:
                # 无掩码时创建全零掩码
                img = load_image(img_path)
                mask_arr = np.zeros_like(img)

            mask_img = Image.fromarray(mask_arr * label_id, mode='L')
            mask_dest = NNUNET_RAW / "labelsTr" / f"{case_name}.png"
            mask_img.save(mask_dest)

            all_images.append({
                "case_id": case_id,
                "category": category,
                "label": label_id,
                "has_lesion": label_id > 0
            })

            case_id += 1

    return all_images


def create_dataset_json(total_cases: int):
    """创建 nnU-Net 数据集描述文件"""
    dataset_json = {
        "channel_names": {
            "0": "ultrasound"
        },
        "labels": {
            "background": 0,
            "benign": 1,
            "malignant": 2
        },
        "numTraining": total_cases,
        "file_ending": ".png",
        "name": "BUSI",
        "description": (
            "Breast Ultrasound Images Dataset for nodule segmentation. "
            "Contains 580 images with expert-annotated masks. "
            "Classes: Normal (133), Benign (375), Malignant (132)."
        ),
        "reference": "Al-Dhabyani W, et al. Data in Brief 28 (2020): 104863.",
        "license": "CC BY 4.0",
        "tensorImageSize": "2D",
        "modality": {
            "0": "US"
        }
    }

    json_path = NNUNET_RAW / "dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    print(f"✓ 创建 dataset.json：{json_path}")


def print_statistics(all_images: list):
    """打印统计信息"""
    total = len(all_images)
    with_lesion = sum(1 for img in all_images if img["has_lesion"])

    print("\n" + "=" * 50)
    print("转换统计")
    print("=" * 50)
    print(f"  总图像数：{total}")
    print(f"  有结节：{with_lesion}（{with_lesion/total*100:.1f}%）")
    print(f"  无结节：{total - with_lesion}（{(total-with_lesion)/total*100:.1f}%）")
    print("=" * 50)


def main():
    print("=" * 50)
    print("BUSI → nnU-Net 格式转换")
    print("=" * 50)

    # 创建目录
    create_directories()

    # 转换数据
    all_images = convert_dataset()
    if all_images is None:
        return

    # 创建 dataset.json
    create_dataset_json(len(all_images))

    # 打印统计
    print_statistics(all_images)

    print("\n✓ 预处理完成！")
    print("\n下一步：训练模型")
    print("  Windows: train.bat")
    print("  Linux/Mac: bash train.sh")


if __name__ == "__main__":
    main()
