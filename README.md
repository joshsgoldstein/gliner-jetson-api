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
| `MAX_CONCURRENT_INFERENCES` | `1` | Maximum in-flight inference tasks across endpoints |
| `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | `10` | Time to wait for an inference slot before returning busy |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Per-request inference timeout |
| `MAX_TEXT_CHARS` | `20000` | Maximum accepted `text` length |
| `MAX_LABELS` | `256` | Maximum labels for list/dict label payloads |
| `MAX_SCHEMA_FIELDS` | `256` | Maximum schema fields for structured payloads |

## Concurrency

Inference endpoints use `asyncio.to_thread()` to run model inference off the
main event loop. This keeps the server responsive to health checks and new
connections while a request is being processed.

To keep behavior stable under load, the API now includes:

- lock-protected model initialization (prevents duplicate first-load races)
- bounded inference concurrency with a semaphore
- request timeouts for long-running inference calls
- payload size/shape limits for text, labels, and schema fields

Status behavior under load or oversized payloads:

- `413` when `text` exceeds `MAX_TEXT_CHARS`
- `503` when no inference slot is available within `INFERENCE_ACQUIRE_TIMEOUT_SECONDS`
- `504` when inference exceeds `REQUEST_TIMEOUT_SECONDS`

For higher throughput, tune `MAX_CONCURRENT_INFERENCES` carefully for your
Jetson memory budget, or scale with multiple workers (each worker keeps its own
model copy).

## Jetson Notes

This project requires Jetson-specific builds of PyTorch, ONNX Runtime, and
cuDNN that differ from standard x86 packages. The Dockerfile handles all of
this automatically, but if you need to understand or modify the build, see
[JETSON.md](JETSON.md) for a detailed explanation of every workaround and why
it's needed.
