# GLiNER2 on Jetson — Overview

Background and orientation for the self-hosted GLiNER2 inference service on
`jarvita-agx`. For operating it, see [runbook.md](runbook.md); for calling it,
see [api.md](api.md).

## What GLiNER2 is

GLiNER2 is a family of small **encoder-based information extraction models**
from Fastino. Where a generative LLM is asked to produce extraction results as
text and then parsed, GLiNER2 predicts spans and labels directly from an encoder,
which is why it fits in a few hundred million parameters instead of a few
billion.

The defining property is **zero-shot schema at inference time**. The label set,
the classification classes, and the structured-output fields are all supplied in
the request. There is no fine-tuning step and no per-task model. Ask for
`["medication", "dosage", "symptom", "time"]` on one request and
`["company", "person", "location"]` on the next, against the same loaded weights.

It handles four task shapes, all through the same model:

| Task | What it does |
|---|---|
| Entity extraction | Zero-shot NER against a caller-supplied label set |
| Classification | Assign a label from caller-supplied classes |
| Structured extraction | Fill a declared record schema from free text |
| Multi-task | All of the above against one text in a single forward pass |

The multi-task path is the interesting one: entities, a classification, and a
structured record come back from one pass, rather than three round trips.

Paper: [arXiv:2507.18546](https://arxiv.org/abs/2507.18546).

## GLiNER2.5 and the boundary architecture

GLiNER2.5 was released 2026-08-22. The architectural change is how spans are
proposed.

- **GLiNER2 (`span`)** enumerates candidate spans over a fixed span-width grid
  and scores them. Cost scales with the grid.
- **GLiNER2.5 (`boundary`)** predicts start and end positions and pairs them
  sparsely. No fixed width ceiling, and less wasted scoring.

This matters operationally because **the loaders are not interchangeable**.
A GLiNER2.5 checkpoint declares `"architecture": "boundary"` in its
`config.json` and must be loaded with `AutoExtractor.from_pretrained()`. The
legacy `GLiNER2.from_pretrained()` span loader will not dispatch it. `app.py`
reads `config.json` at load time and picks the right loader, so callers never
see this — but it is the first thing to check when a 2.5 model misbehaves.

The `gliner2` library at version 2.0.0 (released 2026-08-24) supports both
architectures and requires `transformers<5`.

GLiNER2.5 also adds **relation extraction** — given a text and a set of relation
names, it returns typed pairs. Verified on this box with the 2.5 model:

Input text:

> Satya Nadella, CEO of Microsoft, met Sam Altman of OpenAI in Seattle to
> discuss the Azure partnership.

Relations requested: `["works_for", "met_with", "located_in"]`

```json
{
  "relation_extraction": {
    "works_for": [["Satya Nadella", "Microsoft"]],
    "met_with": [["Satya Nadella", "Sam Altman"]],
    "located_in": [["Sam Altman", "Seattle"], ["OpenAI", "Seattle"]]
  }
}
```

Note `located_in` returning two pairs, one of which (`Sam Altman` → `Seattle`)
is a person and one (`OpenAI` → `Seattle`) an organization. The model is
extracting what the sentence supports, not applying a type constraint you did
not give it.

Relation extraction is exposed as `POST /extract_relations`. It requires a
boundary checkpoint: on a span model the endpoint returns **501** naming the
architecture actually loaded, rather than failing obscurely.

## Model family

All models are from Fastino and licensed Apache-2.0.

| Model | Params | Encoder | Language | Architecture |
|---|---|---|---|---|
| `fastino/gliner2-base-v1` | 205M | DeBERTa-v3-base | English | span |
| `fastino/gliner2-large-v1` | 340M | DeBERTa-v3-large | English | span |
| `fastino/gliner2.5-small-v1` | 74M | DeBERTa-v3-xsmall | English | boundary |
| `fastino/gliner2.5-base-v1` | 194M | DeBERTa-v3-base | English | boundary |
| `fastino/gliner2.5-multi-v1` | 287M | mDeBERTa-v3-base | Multilingual | boundary |

Notes:

- `gliner2.5-multi-v1` is the only multilingual option, via the mDeBERTa
  encoder. Its tokenizer requires `protobuf` installed, or load fails with an
  `ImportError`.
- `gliner2.5-small-v1` at 74M is the option to reach for when the box is under
  memory pressure or latency budget is tight.
- On-disk sizes measured here: `gliner2-large-v1` 1.9 GB,
  `gliner2.5-multi-v1` 1.1 GB.

### Measured comparison on this hardware

Average of 5 runs on the AGX Orin GPU, with another LLM sharing the box.
**Indicative, not a rigorous evaluation** — the box was not quiesced and the
sample is small.

| Test | `gliner2.5-multi` | `gliner2-large` |
|---|---|---|
| clinical entity extraction | 117.6 ms | 171.9 ms |
| business entity extraction | 172.0 ms | 149.5 ms |
| spanish entity extraction | 103.1 ms | 170.8 ms |
| french entity extraction | 107.9 ms | 164.8 ms |
| classification | 79.6 ms | 111.2 ms |

`gliner2.5-multi` was faster on four of five tests despite being the
multilingual model, which is consistent with the boundary architecture doing
less work than a span grid — but with a shared box and 5 runs, do not read
precise ratios into these.

Batching, same conditions, `gliner2.5-multi`, 16 identical documents: 384 ms
batched (24.0 ms/doc) versus 1607 ms sequential (100.4 ms/doc) — a **4.2x**
per-document speedup.

Memory: `gliner2-large-v1` measured a 4.28 GiB container footprint idle and
4.69 GiB at peak. Both models loaded into one process measured 3.11 GB of GPU
allocation.

### Accuracy caveat

Outputs from the two models were **identical** on the clinical, business, and
Spanish tests. On the French test they diverged: `gliner2.5-multi` returned
`"PDG de Renault"` as the company where `gliner2-large` returned `"Renault"` —
the 2.5 model swept the job title into the span.

**The sample was four sentences.** That is nowhere near enough to conclude
anything about relative accuracy. It is enough to justify one practice: when
swapping `MODEL_ID`, re-check outputs against your own texts rather than
assuming the newer model is a drop-in improvement. Span boundaries in particular
can shift in ways that break downstream string matching.

## Why run it on-device

- **Latency.** Sub-200 ms per document on local hardware, with no network hop
  and no queue behind someone else's traffic.
- **Data stays home.** Clinical notes, personal documents, and anything else you
  would not paste into a hosted API never leave the LAN. This is the main reason
  the service exists.
- **No per-token cost.** Extraction over a large corpus has a fixed hardware
  cost rather than a bill that scales with volume.
- **The model fits.** 74M–340M parameters is a fundamentally different resource
  class from a generative LLM. It coexists with other workloads on one Orin
  rather than monopolizing it.
- **Right tool for the shape of the job.** Extraction with a known schema does
  not need a generative model. An encoder that emits spans directly does not
  hallucinate values that were not in the text, and does not need output parsing
  or JSON-mode coercion.

The tradeoff is real: GLiNER2 does extraction and classification, not reasoning,
summarization, or generation. It is a component, not a replacement for the LLM
on the same box.

## Service architecture

```
client (LAN)
  │  HTTP  http://192.168.1.177:8013
  ▼
host jarvita-agx  ── AGX Orin 64GB, JetPack 7.2 / L4T R39.2.0, Ubuntu 24.04
  │  Docker 29.1.3, default runtime nvidia, data-root on /ssd
  │  port 8013 → 8012
  ▼
container gliner-api  ── image built on nvcr.io/nvidia/l4t-jetpack:r36.4.0 (JetPack 6)
  │  Python 3.10, torch 2.8.0+cu126, sm_87
  │  tini (PID 1) → uvicorn --loop asyncio → FastAPI (app.py)
  │      ├── validation: text length, label count, schema field count
  │      ├── asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
  │      └── asyncio.to_thread(...) → gliner2 model
  ▼
model  ── GLiNER2 (span) or AutoExtractor (boundary), selected from config.json
  │  weights bind-mounted from /ssd/gliner/models → /app/models
  ▼
Orin iGPU (unified memory, shared with other workloads on the box)
```

Design points worth knowing:

- **One model per process, held as a singleton.** Load is lock-protected so
  simultaneous first requests cannot each trigger a download and load.
- **Inference runs off the event loop** via `asyncio.to_thread()`, so `/health`
  and new connections stay responsive while the GPU is busy.
- **Concurrency is bounded by a semaphore**, default width 1. The GPU serializes
  the work anyway; the semaphore makes that explicit and turns overload into a
  clean `503` instead of memory thrash.
- **Everything is bounded**: text length, label count, schema field count, slot
  acquisition, and inference duration all have limits with defined status codes.
  See [api.md](api.md#errors).
- **Weights live on the host SSD.** Replacing the container does not
  re-download several GB.

### The JetPack 6-on-JetPack 7 situation

The image is built on `nvcr.io/nvidia/l4t-jetpack:r36.4.0` — a JetPack 6 base —
and runs unmodified on a JetPack 7 host. This is correct and deliberate: the
container carries its own CUDA 12.6 userspace, and the JP7 driver is
forward-compatible with it. Nothing in the image needs to match the host's
CUDA 13.

There is also no `nvcr.io/nvidia/l4t-jetpack:r39.*` tag to port to.

The trap, recorded so nobody re-derives it: the `sbsa/cu130` wheel index looks
like the natural JetPack 7 successor and is not. `sbsa` is Grace/Grace-Blackwell —
`sm_90`/`sm_100` — and those wheels fail on Orin's `sm_87` with
`cudaErrorNoKernelImageForDevice`, after importing cleanly and reporting
`torch.cuda.is_available() == True`. Full write-up in [../JETSON.md](../JETSON.md).

## Where it fits in the homelab

`jarvita-agx` already runs a generative LLM. GLiNER2 sits alongside it as the
cheap, deterministic extraction layer:

- **Pre-processing for RAG and indexing.** Pull entities and structured fields
  off documents at ingest so they can be filtered and faceted, instead of
  relying on embedding similarity alone.
- **Routing and triage.** Classify incoming text — a note, a message, a
  document — before deciding whether it needs the expensive model at all.
- **Structured capture from unstructured sources.** Declare a schema, get
  records back. Receipts, notes, feeds, scraped pages.
- **Multilingual extraction** without a second model or a translation hop, via
  `gliner2.5-multi-v1`.
- **Guardrail on generative output.** Run extraction over what the LLM produced
  and check the claimed values actually appear in the source.

Resource-wise it is a good neighbor: roughly 4.3–4.7 GiB of container footprint
for the large model on a 64GB unified-memory box. Its costs are bounded by
design — a semaphore of 1, a text cap, and a request timeout — so it cannot
starve the LLM sharing the same memory pool.

## Upstream sources

| Resource | Link |
|---|---|
| GLiNER2 source | https://github.com/fastino-ai/GLiNER2 |
| Fastino models on Hugging Face | https://huggingface.co/fastino |
| GLiNER2 paper | https://arxiv.org/abs/2507.18546 |

## In-repo references

| Document | Contents |
|---|---|
| [index.md](index.md) | Documentation landing page |
| [api.md](api.md) | Endpoint reference, curl examples, error semantics |
| [runbook.md](runbook.md) | Deploy, health, model swap, rollback, failures, sizing |
| [examples.ipynb](examples.ipynb) | Runnable Python client examples |
| [../README.md](../README.md) | Project overview, local development |
| [../JETSON.md](../JETSON.md) | Jetson build decisions and traps |
| [../AGENTS.md](../AGENTS.md) | Contributor and agent guidelines |
