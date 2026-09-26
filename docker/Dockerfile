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

# Dependencies come from requirements-cuda.txt (generated list + jax[cuda12]); copied first for layer caching
COPY requirements.txt requirements-cuda.txt ./
RUN python -m pip install --upgrade pip setuptools wheel     && python -m pip install --no-cache-dir -r requirements-cuda.txt

CMD ["python", "run_standalone.py"]
