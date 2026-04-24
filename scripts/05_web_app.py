"""
Streamlit Web 界面
上传乳腺超声图像，自动返回分割结果
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch

# 设置页面配置
st.set_page_config(
    page_title="乳腺超声结节分割",
    page_icon="🩺",
    layout="wide"
)

# 目录配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "models"


def load_image(img_file) -> np.ndarray:
    """从上传文件加载图像"""
    img = Image.open(img_file)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def predict_segmentation(image: np.ndarray) -> np.ndarray:
    """执行分割预测"""
    try:
        # 设置环境变量
        os.environ["nnUNet_raw"] = str(DATA_DIR / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(DATA_DIR / "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = str(RESULTS_DIR)

        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
        import shutil

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
        st.error(f"nnU-Net 预测失败：{e}")
        return None


def simple_threshold_predict(image: np.ndarray) -> np.ndarray:
    """简单的阈值分割（备用方案）"""
    # 计算阈值
    threshold = np.mean(image) + 0.5 * np.std(image)
    binary = (image > threshold).astype(np.uint8)

    # 形态学处理
    from scipy import ndimage
    binary = ndimage.binary_opening(binary, iterations=1)
    binary = ndimage.binary_closing(binary, iterations=1)

    return binary.astype(np.uint8) * 2  # 标签 2


def create_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """创建叠加可视化"""
    # 彩色图像
    img_rgb = Image.fromarray(image).convert('RGB')

    # 创建掩码颜色
    mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # 良性 = 绿色
    mask_colored[mask == 1] = [0, 255, 0]
    # 恶性 = 红色
    mask_colored[mask == 2] = [255, 0, 0]

    mask_img = Image.fromarray(mask_colored)

    # 叠加
    overlay = Image.blend(img_rgb, mask_img, alpha=0.4)

    return overlay


def calculate_statistics(mask: np.ndarray) -> dict:
    """计算统计信息"""
    lesion_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    lesion_ratio = lesion_pixels / total_pixels * 100

    # 判断类别
    if np.any(mask == 2):
        category = "⚠️ 疑似恶性"
        category_color = "red"
    elif np.any(mask == 1):
        category = "✓ 良性可能"
        category_color = "green"
    else:
        category = "✓ 无明显结节"
        category_color = "blue"

    return {
        "category": category,
        "category_color": category_color,
        "lesion_pixels": lesion_pixels,
        "lesion_ratio": lesion_ratio,
        "image_size": total_pixels
    }


def main():
    # 标题
    st.title("🩺 乳腺超声结节分割系统")
    st.markdown("---")

    # 侧边栏
    st.sidebar.header("说明")
    st.sidebar.info("""
    上传乳腺超声图像，系统将自动检测并分割结节区域。

    **支持格式**：PNG, JPG, DICOM
    **输出**：结节位置 + 分类（良性/恶性）
    **免责声明**：本系统仅供辅助参考，不替代医生诊断
    """)

    st.sidebar.header("使用方法")
    st.sidebar.markdown("""
    1. 上传乳腺超声图像
    2. 点击"开始分析"按钮
    3. 查看分割结果和统计信息
    """)

    # 主界面
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 上传图像")
        uploaded_file = st.file_uploader(
            "选择乳腺超声图像",
            type=['png', 'jpg', 'jpeg', 'dcm']
        )

        if uploaded_file:
            st.image(uploaded_file, caption="上传的图像", use_column_width=True)

    with col2:
        st.subheader("📊 分析结果")

        if uploaded_file:
            if st.button("🔍 开始分析", type="primary"):
                with st.spinner("正在分析图像..."):
                    # 加载图像
                    image = load_image(uploaded_file)

                    # 预测
                    mask = predict_segmentation(image)

                    if mask is None:
                        st.warning("nnU-Net 模型未找到，使用简单阈值分割演示")
                        mask = simple_threshold_predict(image)

                    # 创建可视化
                    overlay = create_overlay(image, mask)

                    # 显示结果
                    st.image(overlay, caption="分割结果", use_column_width=True)

                    # 统计信息
                    stats = calculate_statistics(mask)

                    st.markdown("---")
                    st.markdown(f"### 诊断结果：<span style='color:{stats['category_color']}'>{stats['category']}</span>",
                              unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("结节面积", f"{stats['lesion_pixels']} 像素")
                    with col_b:
                        st.metric("占图像比例", f"{stats['lesion_ratio']:.2f}%")

                    st.markdown("---")
                    st.caption("⚠️ 本结果仅供辅助参考，不替代专业医生诊断")
        else:
            st.info("👆 请先上传图像")

    # 底部信息
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        基于 nnU-Net 的医学图像分割系统 | 仅供辅助参考
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
