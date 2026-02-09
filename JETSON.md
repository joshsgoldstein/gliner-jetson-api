# Jetson Build Notes

Building for NVIDIA Jetson (aarch64) requires careful dependency management
because standard PyPI wheels are compiled for x86_64 and won't work. This
document explains every Jetson-specific step in the Dockerfile and why it's
necessary.

## Base Image: `dustynv/l4t-pytorch:r36.2.0`

This is a community Jetson image from the
[jetson-containers](https://github.com/dusty-nv/jetson-containers) project.
It ships with CUDA, Python 3.10, and an older PyTorch pre-installed. We use it
as a starting point because it has the correct NVIDIA driver integration for
Jetson, but we replace most of the ML packages it bundles.

> **Future improvement:** Switch to `nvcr.io/nvidia/l4t-jetpack:r36.4.0` — a
> leaner NVIDIA base image with CUDA + cuDNN but no pre-installed Python ML
> packages. This would eliminate most of the cleanup steps below.

## cuDNN 9 (`libcudnn9-cuda-12`)

The base image ships with cuDNN **8**, but PyTorch 2.8.0 (built for CUDA 12.6)
requires cuDNN **9**. The NVIDIA CUDA apt repository is added manually to install
`libcudnn9-cuda-12` since the base image doesn't include this repo.

## PyTorch: Jetson aarch64 CUDA Wheel

```dockerfile
RUN pip3 install --no-cache-dir --no-deps \
      "https://pypi.jetson-ai-lab.io/jp6/cu126/.../torch-2.8.0-cp310-cp310-linux_aarch64.whl"
```

Standard `pip install torch` pulls an **x86_64 CPU-only** wheel from PyPI, which
won't use the Jetson's GPU. Instead we install NVIDIA's official aarch64 CUDA
wheel from [pypi.jetson-ai-lab.io](https://pypi.jetson-ai-lab.io). The `--no-deps`
flag prevents pip from pulling in transitive dependencies that could conflict
with the Jetson environment.

**This wheel is installed twice** — once early (so other packages can detect
torch during install) and once at the very end with `--force-reinstall`. This
is because intermediate `pip install` commands (e.g. `transformers`, `accelerate`)
will silently pull in a CPU-only PyTorch from PyPI as a dependency, overwriting
the Jetson CUDA wheel. Re-pinning it last ensures the final image has the
correct GPU-enabled PyTorch.

## ONNX Runtime: Jetson GPU Wheel

```dockerfile
RUN pip3 install --no-cache-dir \
      "https://pypi.jetson-ai-lab.io/jp6/cu126/.../onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"
```

Same situation as PyTorch — the standard `onnxruntime` or `onnxruntime-gpu` from
PyPI is x86_64-only. NVIDIA provides aarch64 GPU wheels for Jetson at
[pypi.jetson-ai-lab.io](https://pypi.jetson-ai-lab.io). We also explicitly
uninstall any CPU `onnxruntime` that may have been pulled in by dependencies.

## GLiNER / GLiNER2: `--no-deps`

```dockerfile
RUN pip3 install --no-cache-dir gliner --no-deps \
    && pip3 install --no-cache-dir gliner2 --no-deps
```

`gliner` and `gliner2` are installed with `--no-deps` because their declared
dependencies include `torch`, `transformers`, etc. Letting pip resolve these
would pull in x86_64 CPU-only wheels from PyPI, overwriting the Jetson-specific
wheels we already installed. All actual dependencies are installed separately
with the correct Jetson-compatible versions.

## Removing torchvision / torchaudio

```dockerfile
RUN pip3 uninstall -y torchvision torchaudio || true
```

The base image bundles `torchvision` and `torchaudio` compiled for the
**older** PyTorch that shipped with it. After we upgrade PyTorch to 2.8.0,
these become incompatible — their C++ operators don't match the new PyTorch ABI.
When `transformers` tries to import `torchvision` (via its image processing
utils), it crashes with `RuntimeError: operator torchvision::nms does not exist`.
GLiNER2 doesn't need either package, so removing them is the cleanest fix.

## `libsndfile1`

The base image includes Python audio packages (`soundfile`, `librosa`) as
leftover transitive dependencies. These require the `libsndfile` system library
at runtime. Rather than uninstalling the Python packages (which may break other
things), we install the missing system lib.

## `tini` (init process)

Python running as PID 1 inside a container doesn't handle `SIGTERM` properly —
it ignores the signal, so `docker stop` has to wait 10 seconds then `SIGKILL`
the process. `tini` runs as PID 1 instead and properly forwards signals to
uvicorn, enabling graceful shutdown.

## `numpy < 2`

NumPy 2.x introduced breaking ABI changes. Many packages in the Jetson
ecosystem (PyTorch, ONNX Runtime, transformers) are compiled against NumPy 1.x.
Pinning `numpy < 2` prevents silent segfaults from ABI mismatches.

## `MODEL_PRELOAD=0`

On Jetson, loading the model during uvicorn's startup event can trigger
`malloc(): invalid size (unsorted)` crashes. Setting `MODEL_PRELOAD=0` defers
model loading until the first API request, after uvicorn is fully initialized.
The first request will be slow (model load), but subsequent requests are fast.

## `--loop asyncio` (uvicorn)

Uvicorn defaults to `uvloop` if available, but `uvloop` can conflict with
CUDA/PyTorch on Jetson. Explicitly using `--loop asyncio` avoids this.

## No `--reload` (uvicorn)

Uvicorn's `--reload` flag forks processes to watch for file changes. Forking
after CUDA initialization corrupts GPU state, causing `malloc()` crashes. Never
use `--reload` with GPU models.
