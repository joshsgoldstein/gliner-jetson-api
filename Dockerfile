# Minimal Jetson base: CUDA 12.6 + cuDNN 9, no pre-installed ML packages.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

# Install Python, pip, and system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libsndfile1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy API requirements
COPY gliner-api/requirements.txt /app/requirements.txt

# Install Jetson aarch64 CUDA PyTorch wheel (not available on PyPI)
RUN pip3 install --no-cache-dir --no-deps \
      "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"

# Install API runtime deps
RUN pip3 install --no-cache-dir --default-timeout=100 -r /app/requirements.txt

# Install Jetson aarch64 ONNX Runtime GPU wheel (not available on PyPI)
RUN pip3 uninstall -y onnxruntime || true \
    && pip3 install --no-cache-dir "numpy<2" \
    && pip3 install --no-cache-dir \
      "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"

# Install GLiNER + GLiNER2 without deps (avoids pulling x86 CPU-only torch)
RUN pip3 install --no-cache-dir gliner --no-deps \
    && pip3 install --no-cache-dir gliner2 --no-deps \
    && pip3 uninstall -y onnxruntime || true

# Re-pin Jetson CUDA torch wheel — earlier pip installs may have
# pulled in a CPU-only PyTorch as a transitive dependency.
RUN pip3 install --no-cache-dir --force-reinstall --no-deps \
      "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"

# Copy application code
COPY gliner-api/app.py /app/app.py
COPY gliner-api/download-model.py /app/download-model.py
COPY gliner-api/models /app/models

EXPOSE 8012

# tini properly handles SIGTERM and reaps zombie processes as PID 1
ENTRYPOINT ["tini", "--"]
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8012", "--loop", "asyncio"]
