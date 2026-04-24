"""
数据下载指引脚本
BUSI 数据集下载地址：https://data.mendeley.com/datasets/wty94exzgg/3
"""

import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

# 目录配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BUSI_DIR = DATA_DIR / "BUSI"


def print_download_guide():
    """打印下载指引"""
    guide = """
╔════════════════════════════════════════════════════════════════════╗
║                      BUSI 数据集下载指引                            ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. 访问以下网址：                                                  ║
║     https://data.mendeley.com/datasets/wty94exzgg/3                ║
║                                                                    ║
║  2. 点击右上角 "Download" 按钮                                      ║
║                                                                    ║
║  3. 下载完成后，将 ZIP 文件放到 data/ 目录下                         ║
║                                                                    ║
║  4. 运行本脚本进行解压：                                            ║
║     python scripts/00_download_data.py <zip文件路径>               ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

数据集构成：
  - 0_Normal/     : 133 张（无结节）
  - 1_Benign/     : 375 张（良性结节）
  - 2_Malignant/  : 132 张（恶性结节）
  合计：580 张乳腺超声图像 + 对应分割掩码
"""
    print(guide)


def extract_dataset(zip_path: str):
    """解压数据集"""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"错误：文件不存在 {zip_path}")
        return False

    print(f"正在解压：{zip_path}")

    # 创建目录
    BUSI_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 提取到临时目录
            extract_dir = DATA_DIR / "temp_extract"
            zip_ref.extractall(extract_dir)

            # 移动文件到正确位置
            for item in extract_dir.iterdir():
                if item.is_dir():
                    # 检查是否是数据文件夹
                    for subdir in ["0_Normal", "1_Benign", "2_Malignant", "Normal", "Benign", "Malignant"]:
                        if subdir in item.name:
                            dest = BUSI_DIR / item.name
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)
                            print(f"  ✓ 已提取：{item.name}")

            # 清理临时目录
            shutil.rmtree(extract_dir)

        print(f"\n✓ 解压完成！数据位于：{BUSI_DIR}")
        return True

    except Exception as e:
        print(f"解压失败：{e}")
        return False


def verify_data():
    """验证数据是否完整"""
    expected = {
        "0_Normal": 133,
        "1_Benign": 375,
        "2_Malignant": 132
    }

    print("\n验证数据完整性...")
    all_ok = True

    for folder, expected_count in expected.items():
        folder_path = BUSI_DIR / folder
        if not folder_path.exists():
            print(f"  ✗ 缺失：{folder}")
            all_ok = False
            continue

        files = list(folder_path.glob("*.png"))
        # 过滤掩码文件
        image_files = [f for f in files if "_mask" not in f.name and "_gt" not in f.name]

        if len(image_files) >= expected_count * 0.9:  # 允许 10% 误差
            print(f"  ✓ {folder}: {len(image_files)} 张图像")
        else:
            print(f"  ✗ {folder}: 仅 {len(image_files)} 张（期望 {expected_count}）")
            all_ok = False

    return all_ok


def main():
    import sys

    BUSI_DIR.mkdir(parents=True, exist_ok=True)

    # 检查数据是否已存在
    if verify_data():
        print("\n✓ 数据已就绪，可以开始预处理！")
        print("  运行：python scripts/01_preprocess.py")
        return

    # 如果有命令行参数，尝试解压
    if len(sys.argv) > 1:
        if extract_dataset(sys.argv[1]):
            verify_data()
        return

    # 打印下载指引
    print_download_guide()


if __name__ == "__main__":
    main()
