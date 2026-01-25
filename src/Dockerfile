FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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

RUN pip install numpy \
                torch \
                torchvision \
                pillow \
                scikit-image \
                piq \
                pandas \
                torch_dct \
                tensorboard \
                opencv-python \
                imageio[ffmpeg] \
                imageio-ffmpeg \
                scikit-video

EXPOSE 6006
