# GLiNER2 API (Jetson)

A FastAPI server for [GLiNER2](https://github.com/urchade/GLiNER) multi-task
information extraction, built specifically for NVIDIA Jetson (JetPack 6 / L4T R36).

Supports entity extraction, text classification, structured JSON extraction,
and combined multi-task schemas — all running on-device with GPU acceleration.

## Quick Start (Local)

```bash
make install
make run-no-preload
```

API will be at: `http://localhost:8000`

For development with auto-reload (CPU-only, restarts on file changes):

```bash
make dev
```

## Quick Start (Docker)

```bash
make docker-build-no-cache  # full rebuild (first time or after Dockerfile changes)
make docker-build            # cached rebuild
make docker-run              # foreground with GPU
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check, shows if model is loaded |
| GET | `/version` | GLiNER2 package version |
| POST | `/extract_entities` | Named entity extraction |
| POST | `/classify_text` | Text classification (single/multi-label) |
| POST | `/extract_structured` | Structured JSON extraction |
| POST | `/extract_multitask` | Combined entity + classification + structure |

## Example

```bash
curl -X POST http://localhost:8000/extract_entities \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient received 400mg ibuprofen for severe headache at 2 PM.",
    "labels": ["medication", "dosage", "symptom", "time"]
  }'
```

Response:

```json
{
  "entities": {
    "medication": ["ibuprofen"],
    "dosage": ["400mg"],
    "symptom": ["headache"],
    "time": ["2 PM"]
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `fastino/gliner2-large-v1` | HuggingFace model ID |
| `MODEL_DIR` | `./models` | Local model storage directory |
| `MODEL_PRELOAD` | `1` | Set to `0` to defer model loading until first request |

## Concurrency

Inference endpoints use `asyncio.to_thread()` to run model inference off the
main event loop. This keeps the server responsive to health checks and new
connections while a request is being processed.

However, the model itself is **not thread-safe for concurrent GPU inference** —
requests are effectively serialized through the GIL and CUDA stream ordering.
For production workloads requiring true parallel inference, consider running
multiple uvicorn workers (each with its own model copy) or using a dedicated
inference server like Triton.

## Jetson Notes

This project requires Jetson-specific builds of PyTorch, ONNX Runtime, and
cuDNN that differ from standard x86 packages. The Dockerfile handles all of
this automatically, but if you need to understand or modify the build, see
[JETSON.md](JETSON.md) for a detailed explanation of every workaround and why
it's needed.
