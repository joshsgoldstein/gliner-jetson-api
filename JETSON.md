# Jetson Build Notes

This project runs GLiNER2 on NVIDIA Jetson (aarch64) with Docker.

These notes document **exactly** what was required to get a stable container
on JetPack 6, and why each choice exists.

## Final Base Image

`nvcr.io/nvidia/l4t-jetpack:r36.4.0`

Why:
- Includes CUDA 12.6 + cuDNN 9 already
- Matches JetPack/L4T runtime better than older community images
- Avoids most manual CUDA/cuDNN patching

We moved away from `dustynv/l4t-pytorch:r36.2.0` because of repeated CUDA/cuDNN
and binary mismatch issues.

## Python and System Packages

Installed in Dockerfile:
- `python3-pip`
- `git`
- `libsndfile1`
- `libopenblas0`
- `tini`
- `curl`

Why:
- `libsndfile1`: required by transitive audio deps (`soundfile`) that appear in HF/ML stacks
- `libopenblas0`: required by PyTorch at import time even when running GPU inference
- `tini`: proper PID 1 signal handling (`docker stop` works cleanly)
- `curl`: used for robust wheel downloads with retry

## Jetson-Specific Wheels (Critical)

Use NVIDIA Jetson wheel index for aarch64 CUDA builds:
- Torch: `torch-2.8.0-cp310-cp310-linux_aarch64.whl` (cu126)

Why:
- Default PyPI resolution can install wrong/CPU variants or incompatible binaries
- We explicitly pin known-good Jetson wheels

## BuildKit Wheel Cache + Retry

Large wheel downloads from `pypi.jetson-ai-lab.io` were flaky/timeouting.

What we do:
- Use `# syntax=docker/dockerfile:1.7`
- Use `RUN --mount=type=cache,target=/var/cache/jetson-wheels`
- Download via `curl --retry ...`
- Install from cached wheel path
- If cached wheel is corrupt, auto-delete and re-download

Why:
- Faster rebuilds
- Survives transient network issues
- Avoids broken cache poisoning

## Layer Ordering for Cache Stability

Torch wheel install is intentionally placed **before** copying `requirements.txt`.

Why:
- Editing Python requirements should not invalidate the expensive torch wheel layer

## NumPy Pinning

Pin `numpy<2` (in requirements and repinned at the end of Dockerfile).

Why:
- Torch/Jetson native modules compiled against NumPy 1.x ABI
- NumPy 2.x causes runtime warnings/errors and potential instability

## `gliner` / `gliner2` Installation Strategy

Install with `--no-deps`:
- `pip install gliner --no-deps`
- `pip install gliner2 --no-deps`

Why:
- Prevents pip from overriding pinned Jetson torch with incompatible resolver outcomes

## ONNX Runtime Decision (Important)

Current image removes ONNX Runtime packages:
- `pip uninstall -y onnxruntime onnxruntime-gpu`

Why:
- In this container stack, ONNX Runtime caused native crashes (`free(): invalid pointer`) after inference
- `gliner` imports ONNX modules eagerly, so we patched import behavior (next section)

## `gliner` Lazy-Import Patch

We patch `/usr/local/lib/python3.10/dist-packages/gliner/__init__.py` during build
so `GLiNER` is lazy-loaded from `.model` via `__getattr__`.

Why:
- Prevents hard import-time requirement on `onnxruntime`
- Lets GLiNER2/PyTorch path run stably without ONNX Runtime in this container

## Uvicorn / Startup Settings

- Use `--loop asyncio`
- Use `MODEL_PRELOAD=0` in Docker runs

Why:
- `MODEL_PRELOAD=0` avoids some startup allocator issues seen on Jetson
- First request pays model-load cost; subsequent requests are faster

## Concurrency Notes

API endpoints run with `asyncio.to_thread(...)` so event loop remains responsive.

Practical behavior:
- Multiple requests can be accepted concurrently
- GPU inference is still effectively serialized by model/device execution

## Operational Notes

- Prefer `make docker-build` for cached builds
- Use `make docker-build-no-cache` only when necessary
- BuildKit cache is required for wheel caching behavior

## Known Working Signals

When healthy, logs should show:
- `Uvicorn running on http://0.0.0.0:8012`
- First inference returns `200 OK`
- `GLiNER2 loaded on GPU (Orin)`
- No `free(): invalid pointer` / `malloc()` crashes

