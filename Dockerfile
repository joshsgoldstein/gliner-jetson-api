# syntax=docker/dockerfile:1.7
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
    libopenblas0 \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Jetson wheel URLs
ENV TORCH_WHL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
    TORCH_WHL_FILE="torch-2.8.0-cp310-cp310-linux_aarch64.whl"

# Install Jetson aarch64 CUDA PyTorch wheel with persistent build cache.
# If cached wheel is corrupt/incomplete, re-download automatically.
RUN --mount=type=cache,target=/var/cache/jetson-wheels \
    WHEEL_PATH="/var/cache/jetson-wheels/${TORCH_WHL_FILE}" && \
    if ! pip3 install --no-cache-dir --no-deps "$WHEEL_PATH" 2>/dev/null; then \
      rm -f "$WHEEL_PATH" && \
      curl -fL --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
        "$TORCH_WHL_URL" -o "${WHEEL_PATH}.tmp" && \
      mv "${WHEEL_PATH}.tmp" "$WHEEL_PATH" && \
      pip3 install --no-cache-dir --no-deps "$WHEEL_PATH"; \
    fi

# Copy API requirements after torch install so requirements edits don't
# invalidate the large torch wheel layer.
COPY gliner-api/requirements.txt /app/requirements.txt
RUN test -s /app/requirements.txt

# Install API runtime deps
RUN pip3 install --no-cache-dir --default-timeout=100 -r /app/requirements.txt

# Install GLiNER + GLiNER2 without deps (avoids pulling x86 CPU-only torch)
RUN pip3 install --no-cache-dir gliner --no-deps \
    && pip3 install --no-cache-dir gliner2 --no-deps \
    && pip3 uninstall -y onnxruntime onnxruntime-gpu || true

# Patch gliner package to lazy-load GLiNER from .model so importing `gliner`
# for shared modules (used by gliner2) doesn't hard-require onnxruntime.
RUN cat > /usr/local/lib/python3.10/dist-packages/gliner/__init__.py <<'PY'
__version__ = "0.2.24"

from .config import GLiNERConfig
from .infer_packing import (
    PackedBatch,
    InferencePackingConfig,
    unpack_spans,
    pack_requests,
)


def __getattr__(name):
    if name == "GLiNER":
        from .model import GLiNER

        return GLiNER
    raise AttributeError(name)


__all__ = [
    "GLiNER",
    "GLiNERConfig",
    "InferencePackingConfig",
    "PackedBatch",
    "pack_requests",
    "unpack_spans",
]
PY

# Re-pin Jetson CUDA torch wheel and numpy<2 LAST — earlier pip installs
# may have pulled in CPU-only PyTorch or NumPy 2.x as transitive deps.
RUN --mount=type=cache,target=/var/cache/jetson-wheels \
    WHEEL_PATH="/var/cache/jetson-wheels/${TORCH_WHL_FILE}" \
    && if ! pip3 install --no-cache-dir --force-reinstall --no-deps "$WHEEL_PATH" 2>/dev/null; then \
      rm -f "$WHEEL_PATH" && \
      curl -fL --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
        "$TORCH_WHL_URL" -o "${WHEEL_PATH}.tmp" && \
      mv "${WHEEL_PATH}.tmp" "$WHEEL_PATH" && \
      pip3 install --no-cache-dir --force-reinstall --no-deps "$WHEEL_PATH"; \
    fi \
    && pip3 install --no-cache-dir "numpy<2"

# Copy application code
COPY gliner-api/app.py /app/app.py
COPY gliner-api/download-model.py /app/download-model.py
COPY gliner-api/models /app/models

EXPOSE 8012

# tini properly handles SIGTERM and reaps zombie processes as PID 1
ENTRYPOINT ["tini", "--"]
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8012", "--loop", "asyncio"]
