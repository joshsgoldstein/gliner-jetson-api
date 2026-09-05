# GLiNER2 Jetson API — learning path

A seven-notebook path through the self-hosted GLiNER2 service on `jarvita-agx`
(NVIDIA Jetson AGX Orin 64GB), ordered so each notebook builds on the one before
it. Every code cell runs against the live deployment; every output committed here
was captured from a real run, not written by hand.

Start at [01](01-getting-started.ipynb) and work down. If you only need a
reference rather than a tutorial, use [`../api.md`](../api.md) instead — this
path is here to teach the *why*.

## The path

| # | Notebook | What it covers | Prerequisite | Time |
|---|---|---|---|---|
| 01 | [Getting started](01-getting-started.ipynb) | Configuring `BASE_URL`; the three GET routes and why `/health` lies about a wedged worker; `model_revision` as provenance; span vs boundary architecture; your first extraction and how to read it | none | ~10 min |
| 02 | [Extraction and classification](02-extraction-and-classification.ipynb) | Zero-shot entity extraction across domains; steering labels with descriptions; span boundaries moving between checkpoints; `/classify_text` and why a bare list of labels is rejected; several classification tasks in one pass | 01 | ~20 min |
| 03 | [Structured extraction and multi-task](03-structured-and-multitask.ipynb) | `::` field specs; why records always come back as a list; `/extract_multitask` and the `schema_config` nesting rule; `choices` for closed vocabularies and the confident-guess problem | 01, 02 | ~25 min |
| 04 | [Tuning and response shapes](04-tuning-and-response-shapes.ipynb) | The six inference options; what a `threshold` sweep actually trades away; how `include_confidence` / `include_spans` change your parsing code; verifying character offsets; `max_len` / `overlap_policy`; why a typo'd option is a 400 | 01–03 | ~25 min |
| 05 | [Relation extraction](05-relations.ipynb) | Why a boundary checkpoint is what makes relations possible; `[head, tail]` pairs; the shared confidence score; `schema_config.relations`; and a validator, because relation arguments are not type-checked for you | 01–04 | ~20 min |
| 06 | [Batching and throughput](06-batching-and-throughput.ipynb) | Why naive concurrency does nothing; why batching helps anyway on a saturated GPU; the four batch routes; the two batch bounds and how to discover them live; a measured batched-vs-sequential comparison; two dead ends | 01–05 | ~30 min |
| 07 | [Operating it in production](07-operating-in-production.ipynb) | The full error taxonomy and what each code tells you to *do*; retry with backoff and jitter; what to alert on versus trend; surviving a model swap; a pre-production checklist | 01–06 | ~25 min |

Notebooks 05 and 06 need a GLiNER2.5 **boundary** checkpoint. They detect this
and skip cleanly rather than failing — see [below](#if-boundary-only-routes-return-501).

## Setup

Only dependency is `requests`:

```bash
pip install requests
```

To run the notebooks rather than just read them, you also need Jupyter:

```bash
pip install jupyterlab
jupyter lab
```

### Reaching the service

The default target is the `jarvita-agx` LAN address, `http://192.168.1.177:8013`
(container port `8012` published on host `8013`). Confirm it is up before you
start:

```bash
curl -fs http://192.168.1.177:8013/health/deep >/dev/null && echo up
```

`/health/deep` is the right check because it runs a real inference — plain
`/health` returns `ok` for a worker whose model has wedged. Notebook 01 explains
why that distinction matters.

### Setting `GLINER_BASE_URL`

Every notebook reads the same environment variable, so one export redirects the
whole path:

```bash
export GLINER_BASE_URL=http://localhost:8013
jupyter lab
```

| Deployment | Base URL |
|---|---|
| `jarvita-agx` on the LAN | `http://192.168.1.177:8013` (default) |
| Container on the local host | `http://localhost:8013` |
| `make run` / `make dev` locally | `http://localhost:8125` |

Set it **before** starting Jupyter — the kernel inherits the environment of the
process that launched it, so exporting it in a terminal after the server is
already running will not reach the notebook. If you would rather not restart,
override it in the first code cell instead:

```python
BASE_URL = "http://localhost:8013"
```

## If the service is down

Symptoms and what they mean:

| What you see | Likely cause | What to do |
|---|---|---|
| `ConnectionError` / connection refused | Container not running, or wrong host/port | `docker ps` on the Jetson; check you used `8013`, not `8012` |
| Connection times out, no response | Not on the same LAN, or the box is off | Ping `192.168.1.177`; check Tailscale if you are remote |
| `503` with `"status": "degraded"` on `/health/deep` | Model loaded but not answering — a wedged worker | Restart the container; see [`../runbook.md`](../runbook.md) |
| `503` on inference routes only, `/health/deep` fine | Backpressure — the single inference slot is busy | Normal under load. Retry with backoff (notebook 07), or batch (notebook 06) |
| `"device": "cpu"` on `/health` | GPU not visible to the container | A fault, not a mode. See [`../runbook.md`](../runbook.md) |
| `"loaded": false` long after startup | `MODEL_PRELOAD=0` and nothing has been requested yet | Send one request; it will be slow, then fine |

Restart and log commands live in [`../runbook.md`](../runbook.md).

### If boundary-only routes return 501

Five of the twelve routes need a GLiNER2.5 **boundary** checkpoint:
`/extract_relations` and all four `*_batch` routes, plus `schema_config.relations`.
On a span model they return `501` naming the architecture actually loaded.

Check which you have:

```bash
curl -s http://192.168.1.177:8013/health | python3 -c "import json,sys; print(json.load(sys.stdin)['architecture'])"
```

The default `MODEL_ID` (`fastino/gliner2.5-base-v1`) is a boundary checkpoint, so
this works as deployed. If it prints `span`, notebooks 05 and 06 will skip their
boundary-only cells and still run end to end — but you will not see the outputs
they are about. Change `MODEL_ID` and restart to follow along fully.

## Notes on the numbers

Several notebooks measure throughput and latency. `jarvita-agx` is a **shared
box**, and the measurements were taken alongside whatever else was running on it.
Where a figure appears, the notebook says so and gives the sample size.

Treat every timing here as an order of magnitude, not a benchmark. Notebook 06
shows the same comparison producing 38.4x on one run and 76.0x minutes later —
not because anything improved, but because the un-batched baseline is the part
that absorbs contention. Run the cells on your own deployment and use your own
numbers.

## Related documentation

| Need | Document |
|---|---|
| All 12 endpoints, verified bodies, full error table, configuration | [`../api.md`](../api.md) |
| Deploy, monitoring, env vars, batch sizing, model swap, rollback | [`../runbook.md`](../runbook.md) |
| What GLiNER2/2.5 is, boundary-only capabilities, model comparison | [`../wiki.md`](../wiki.md) |
| The terse reference notebook this path was expanded from | [`../examples.ipynb`](../examples.ipynb) |
| Documentation index | [`../index.md`](../index.md) |
| Jetson build decisions and the `sbsa/cu130` trap | [`../../JETSON.md`](../../JETSON.md) |
