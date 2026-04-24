# 乳腺超声结节分割项目

## 环境要求
- Python 3.8+
- NVIDIA GPU（RTX 3060+）或 CPU
- CUDA 11.7+（如用 GPU）

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载数据
从以下地址下载 BUSI 数据集：
- https://data.mendeley.com/datasets/wty94exzgg/3
- 解压到 `data/BUSI/` 目录

期望结构：
```
data/BUSI/
├── 0_Normal/      # 133 张
├── 1_Benign/      # 375 张
└── 2_Malignant/   # 132 张
```

### 3. 预处理数据
```bash
python scripts/01_preprocess.py
```

### 4. 训练模型
```bash
# Windows
train.bat

# Linux/Mac
bash train.sh
```

### 5. 启动 Web 界面
```bash
python scripts/05_web_app.py
```
访问 http://localhost:8501

## 项目流程
1. 下载 BUSI 数据集
2. 转换为 nnU-Net 格式
3. 训练 nnU-Net 模型
4. 导出 ONNX 模型
5. 启动 Web 界面进行预测

## 评估指标
- Dice Score ≥ 0.85（测试集）
- 推理时间 < 500ms（GPU）
