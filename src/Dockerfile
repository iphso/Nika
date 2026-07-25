FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    libglx0 \
    libegl1 \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libegl1-mesa-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
                numpy \
                pillow \
                scikit-image \
                piq \
                pandas \
                torch_dct \
                tensorboard \
                opencv-python \
                imageio[ffmpeg] \
                imageio-ffmpeg \
                scikit-video \
                fvcore

EXPOSE 6006
