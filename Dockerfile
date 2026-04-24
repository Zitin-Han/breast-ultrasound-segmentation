FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

LABEL maintainer="kangjia"
LABEL description="Breast Ultrasound Nodule Segmentation with nnU-Net v2"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set nnU-Net environment variables
ENV nnUNet_raw=/workspace/data/nnUNet_raw \
    nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed \
    nnUNet_results=/workspace/models

WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nnunetv2 explicitly (ensure latest)
RUN pip install --no-cache-dir nnunetv2>=2.2.0

# Copy project code
COPY scripts/ ./scripts/
COPY models/simple_unet.py ./models/simple_unet.py
COPY train.sh .

# Create data directories
RUN mkdir -p /workspace/data/BUSI \
    /workspace/data/nnUNet_raw \
    /workspace/data/nnUNet_preprocessed \
    /workspace/models \
    /workspace/output

# Default command: show usage
CMD ["bash", "-c", "echo 'Breast Ultrasound Segmentation Docker Container' && echo '' && echo 'Usage:' && echo '  1. Download data:  python scripts/00_download_data.py <busi.zip>' && echo '  2. Preprocess:     python scripts/01_preprocess.py' && echo '  3. Plan & preprocess: nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity' && echo '  4. Train:           nnUNetv2_train 501 2d 0' && echo '  5. Predict:         nnUNetv2_predict -i input/ -o output/ -d 501 -c 2d -f 0' && echo '' && echo 'Or run the full pipeline:' && echo '  bash train.sh' && echo '' && echo 'For web UI:' && echo '  streamlit run scripts/05_web_app.py --server.port 8501 --server.address 0.0.0.0' && exec bash"]
