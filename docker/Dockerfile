FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace

# Install JAX with CUDA 12 support and project dependencies
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    "jax[cuda12]" \
    flax \
    optax \
    gymnax \
    matplotlib \
    numpy \
    pandas \
    kaggle \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["python", "run_standalone.py"]
