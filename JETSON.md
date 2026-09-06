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



## JetPack 7 (L4T r39) Compatibility

**This JetPack 6 image runs unmodified on a JetPack 7 host. Do not "port" it.**

Verified on `jarvita-agx` (L4T **R39.2.0**, JetPack 7.2, Ubuntu 24.04, driver
default runtime `nvidia`):

```
torch 2.8.0 | cuda 12.6 | arch_list ['sm_87'] | is_available True | Orin
GLiNER2 loaded on GPU (Orin)
```

The container ships its own CUDA 12.6 userspace; the newer JP7 driver on the host
is forward-compatible with it. Nothing in the image needs to match the host's
CUDA 13.

### Do NOT switch to the `sbsa/cu130` wheel index

This is the trap. When JetPack 7 landed, `pypi.jetson-ai-lab.io` had **no `jp7`
index** — only `jp6/{cu126,cu128,cu129}` and `sbsa/cu130`. `sbsa/cu130` looks like
the obvious JP7 successor: it has CUDA 13 `cp312` aarch64 wheels matching JP7's
Python 3.12. It is the wrong index.

`sbsa` = Server Base System Architecture — Grace Hopper / Grace Blackwell. Those
wheels are compiled for **sm_90 / sm_100**, not Orin's **sm_87**. Torch imports
fine, `torch.cuda.is_available()` returns `True`, and it even prints
`device: Orin` — then the first real kernel launch dies with:

```
CUDA error: no kernel image is available for execution on the device
```

Getting that far also requires chasing undeclared system deps the sbsa build
needs and the CUDA base image lacks — for the record, so nobody re-derives them:

- `libnvpl-lapack0`, `libnvpl-blas0` — sbsa torch links NVPL, not OpenBLAS
- `cuda-cupti-13-0`
- `libcudss0-cuda-13` — and it installs to
  `/usr/lib/aarch64-linux-gnu/libcudss/13/`, which is **not** on the linker path;
  needs an `/etc/ld.so.conf.d` entry + `ldconfig`
- `libnuma1`

All of that work still ends at `cudaErrorNoKernelImageForDevice`. It is a dead
end for Orin.

### Also note

- There is **no `nvcr.io/nvidia/l4t-jetpack:r39.*` tag** — NVIDIA did not ship an
  L4T JetPack base image for r39. `r36.4.0` remains the base here.
- If you ever do need a JP7-native rebuild, the wheels must come from a Jetson
  index that publishes **sm_87** builds. Check for a `jp7` index appearing on
  `pypi.jetson-ai-lab.io` before assuming `sbsa` will work.
- Pip against a devpi index needs the `+simple/` path
  (`https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/`). Without it pip silently
  falls back to PyPI and installs a **CPU-only** wheel that reports `2.9.1+cpu`.


## Why gliner / gliner2 are version-pinned

They are installed with `--no-deps` (to stop pip resolving an x86 CPU-only torch
over the Jetson CUDA wheel). The cost of `--no-deps` is that **new upstream
dependencies are silently dropped** — the build succeeds and the failure only
shows up at container startup.

This already happened once: `gliner2` 1.2.4 → **1.3.2** added a `peft`
dependency. An unpinned rebuild produced an image that built cleanly and then
died on boot with:

```
ModuleNotFoundError: No module named 'peft'
```

So: `gliner` and `gliner2` are pinned in the Dockerfile, and their runtime deps
(`peft`, `accelerate`, `transformers`, …) are listed explicitly in
`requirements.txt`. When bumping either pin, check the new release's `Requires:`
list (`pip3 show gliner2`) and add anything new to `requirements.txt` in the same
commit.
