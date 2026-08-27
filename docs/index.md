# GLiNER2 Jetson API — Documentation

Self-hosted GLiNER2 information-extraction service running on `jarvita-agx`, an
NVIDIA Jetson AGX Orin 64GB.

**Base URL:** `http://192.168.1.177:8013` (container port `8012` → host `8013`)

```bash
curl -s http://192.168.1.177:8013/health
```

## Contents

| Document | What is in it |
|---|---|
| [wiki.md](wiki.md) | What GLiNER2 and GLiNER2.5 are, why run them on-device, service architecture, model family comparison, homelab fit, upstream links |
| [api.md](api.md) | Every endpoint with copy-pasteable curl and real response bodies, plus error semantics and configuration |
| [runbook.md](runbook.md) | Deploy, start/stop, health checks, model swap via `MODEL_ID`, rollback, log inspection, common failures, memory sizing |
| [examples.ipynb](examples.ipynb) | Runnable notebook: every endpoint via `requests`, batch-vs-sequential timing, relation extraction |

## Also in this repo

| Document | What is in it |
|---|---|
| [../README.md](../README.md) | Project overview and local development |
| [../JETSON.md](../JETSON.md) | Jetson build decisions, wheel indexes, and the traps to avoid |
| [../AGENTS.md](../AGENTS.md) | Contributor and agent guidelines |

## Quick orientation

- **Endpoints:** `GET /health`, `GET /version`, `POST /extract_entities`,
  `POST /classify_text`, `POST /extract_structured`, `POST /extract_multitask`.
  FastAPI auto-docs at `/docs`, `/redoc`, `/openapi.json`.
- **Model:** set by `MODEL_ID`, default `fastino/gliner2.5-base-v1`. Weights live
  on the host at `/ssd/gliner/models`.
- **Concurrency:** one inference at a time by default; overload returns `503`,
  not a crash.
- **Platform:** a JetPack 6 image running on a JetPack 7 host, on purpose. Read
  [../JETSON.md](../JETSON.md) before changing the build.
