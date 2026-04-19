ARG CUDA_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

WORKDIR /app

COPY . /app

ARG JAX_VERSION=0.9.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libjpeg-dev \
    zlib1g-dev \
    libglx0 \
    libegl1 \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libegl1-mesa-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    cuda-cupti-13-0 \
    cuda-nvcc-13-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/app-venv && \
    /opt/app-venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/app-venv/bin/pip install \
      numpy \
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

# Separate JAX environment. This uses the CUDA 13 plugin packages that matched the
# working local setup.
RUN python3 -m venv /opt/jax13-venv && \
    /opt/jax13-venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/jax13-venv/bin/pip install \
      "jax==${JAX_VERSION}" \
      "jaxlib==${JAX_VERSION}" \
      "jax-cuda13-plugin==${JAX_VERSION}" \
      "jax-cuda13-pjrt==${JAX_VERSION}"

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/cuda-13.0/extras/CUPTI/lib64:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
ENV PATH=/opt/app-venv/bin:${PATH}

EXPOSE 6006
