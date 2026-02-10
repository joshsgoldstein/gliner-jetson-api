# AGENTS.md

## Project Overview

This repository hosts a FastAPI service that wraps GLiNER2 for on-device NLP inference
on Jetson-class hardware. The API supports:

- Entity extraction (`/extract_entities`)
- Text classification (`/classify_text`)
- Structured extraction (`/extract_structured`)
- Multi-task schema extraction (`/extract_multitask`)

The server entrypoint is `app.py`.

## Local Development

- Install dependencies: `make install`
- Run without preload: `make run-no-preload`
- Run with autoreload (dev): `make dev`

Default API base URL: `http://localhost:8000`.

## Docker Development

- Full rebuild: `make docker-build-no-cache`
- Cached rebuild: `make docker-build`
- Run with GPU: `make docker-run`

## Runtime Behavior Notes

- Model is cached as a singleton in-process.
- Model init is lock-protected to avoid duplicate first-load races.
- Inference is bounded by a semaphore to avoid resource thrash on Jetson.
- Inference executes in worker threads (`asyncio.to_thread`) so the event loop
  remains responsive.

## Environment Variables

Core model settings:

- `MODEL_ID` (default: `fastino/gliner2-large-v1`)
- `MODEL_DIR` (default: `./models`)
- `MODEL_PRELOAD` (default: `1`)

Concurrency and safety controls:

- `MAX_CONCURRENT_INFERENCES` (default: `1`)
- `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` (default: `10`)
- `REQUEST_TIMEOUT_SECONDS` (default: `120`)
- `MAX_TEXT_CHARS` (default: `20000`)
- `MAX_LABELS` (default: `256`)
- `MAX_SCHEMA_FIELDS` (default: `256`)

## Editing Guidelines

- Keep endpoint payload contracts backward compatible unless explicitly changing API behavior.
- Validate input shape/size before inference.
- Prefer reusing the shared inference wrapper for new inference endpoints.
- Avoid adding heavy startup work beyond model preload.
- Keep logs actionable; include operation names and timing.

## Verification Checklist

When modifying inference or request handling logic:

1. Start the API and hit `/health` and `/version`.
2. Run at least one request against each inference endpoint you changed.
3. Check timeout/saturation paths with low semaphore settings.
4. Confirm no new lints are introduced.
