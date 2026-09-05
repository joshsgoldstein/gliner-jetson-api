# GLiNER2 Jetson API — Documentation

Self-hosted GLiNER2 information-extraction service running on `jarvita-agx`, an
NVIDIA Jetson AGX Orin 64GB.

**Base URL:** `http://192.168.1.177:8013` (container port `8012` → host `8013`)

```bash
curl -s http://192.168.1.177:8013/health/deep
```

`/health/deep` runs a real inference and returns `503` when the model is not
answering, so `curl -f` against it is a complete readiness check. Plain
`/health` is cheaper but reports `ok` for a wedged worker.

## Contents

| Document | What is in it |
|---|---|
| [wiki.md](wiki.md) | What GLiNER2 and GLiNER2.5 are, why run them on-device, the twelve routes, boundary-only capabilities, service architecture, model comparison, homelab fit, upstream links |
| [api.md](api.md) | All 12 endpoints with copy-pasteable curl and verified response bodies, the shared inference options, error semantics, and configuration |
| [runbook.md](runbook.md) | Deploy, start/stop, `/health/deep` monitoring, env var reference, batch sizing and the OOM bound, model swap via `MODEL_ID`, rollback, logs, common failures, throughput and memory sizing |
| [examples.ipynb](examples.ipynb) | Runnable notebook: every endpoint via `requests`, the inference options, relation extraction, and a live batched-vs-sequential timing comparison |

## Also in this repo

| Document | What is in it |
|---|---|
| [../README.md](../README.md) | Project overview and local development |
| [../JETSON.md](../JETSON.md) | Jetson build decisions, wheel indexes, and the traps to avoid |
| [../AGENTS.md](../AGENTS.md) | Contributor and agent guidelines |

## Quick orientation

- **Twelve endpoints.** Health and identity: `GET /health`, `GET /health/deep`,
  `GET /version`. Single-document inference: `POST /extract_entities`,
  `/classify_text`, `/extract_structured`, `/extract_multitask`,
  `/extract_relations`. Batched: `POST /extract_entities_batch`,
  `/classify_text_batch`, `/extract_relations_batch`,
  `/extract_multitask_batch`. FastAPI auto-docs at `/docs`, `/redoc`,
  `/openapi.json`.
- **Five routes need a GLiNER2.5 `boundary` checkpoint** — `/extract_relations`
  and all four batch routes, plus `schema_config.relations`. On a span model
  they return `501`. The default `MODEL_ID` is a boundary checkpoint, so they
  work as deployed.
- **Every inference route takes the same optional knobs:** `threshold`,
  `include_confidence`, `include_spans`, `max_len`, `overlap_policy`, and
  `batch_size` on the batch routes. `include_confidence` and `include_spans`
  change the response shape. See
  [api.md](api.md#inference-options).
- **Unknown keys are a `400`**, in the payload and inside `schema_config`, with
  the allowed keys listed in the response. They used to be silently ignored.
- **Model:** set by `MODEL_ID`, default `fastino/gliner2.5-base-v1`. Weights live
  on the host at `/ssd/gliner/models`. `GET /version` returns `model_revision`,
  a stable weight fingerprint to stamp onto extracted rows.
- **Concurrency:** one inference at a time by default; overload returns `503`,
  not a crash. Batch instead of raising it — measured **10.7x** per document for
  one `/extract_multitask_batch` call over three sequential single calls.
- **Monitoring:** poll `/health/deep`. It runs a real forward pass that
  deliberately bypasses the inference semaphore, so a wedged worker is
  distinguishable from a merely busy one.
- **Platform:** a JetPack 6 image running on a JetPack 7 host, on purpose. Read
  [../JETSON.md](../JETSON.md) before changing the build.
