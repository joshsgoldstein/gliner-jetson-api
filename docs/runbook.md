# Operations Runbook

Day-to-day operation of the GLiNER2 inference service on `jarvita-agx`.

## Facts you need before touching anything

| Item | Value |
|---|---|
| Host | `jarvita-agx`, `192.168.1.177` |
| Hardware | NVIDIA Jetson AGX Orin 64GB |
| OS / stack | Ubuntu 24.04, JetPack 7.2, L4T R39.2.0 |
| Docker | 29.1.3, default runtime `nvidia`, data-root on `/ssd` |
| Image | `joshsgoldstein/gliner-api:latest` (built on `nvcr.io/nvidia/l4t-jetpack:r36.4.0`, Python 3.10, torch 2.8.0+cu126, `sm_87`) |
| Container name | `gliner-api` |
| Ports | container `8012` → host `8013` |
| Base URL | `http://192.168.1.177:8013` |
| Model cache (host) | `/ssd/gliner/models`, bind-mounted to `/app/models` |

The image is a **JetPack 6 image running on a JetPack 7 host**, and that is
intentional. It carries its own CUDA 12.6 userspace and the JP7 driver is
forward-compatible with it. Do not "port" it. There is no
`nvcr.io/nvidia/l4t-jetpack:r39.*` tag to port it to. See
[../JETSON.md](../JETSON.md) for the full reasoning.

---

## Deploy

Run all commands on `jarvita-agx`.

```bash
ssh jarvita-agx
```

### First-time deploy

Make sure the model cache directory exists on the SSD, not on the eMMC:

```bash
sudo mkdir -p /ssd/gliner/models
sudo chown -R "$USER":"$USER" /ssd/gliner
```

Start the container:

```bash
docker run -d \
  --name gliner-api \
  --runtime nvidia \
  --restart unless-stopped \
  -p 8013:8012 \
  -e MODEL_ID=fastino/gliner2.5-multi-v1 \
  -v /ssd/gliner/models:/app/models \
  joshsgoldstein/gliner-api:latest
```

On first start with a model that is not yet in `/ssd/gliner/models`, the
container downloads it from Hugging Face. On-disk sizes: `gliner2-large-v1` is
1.9 GB, `gliner2.5-multi-v1` is 1.1 GB. Watch the download in the logs before
expecting the service to answer.

### Redeploy an updated image

```bash
docker pull joshsgoldstein/gliner-api:latest
docker rm -f gliner-api
docker run -d --name gliner-api --runtime nvidia --restart unless-stopped \
  -p 8013:8012 -e MODEL_ID=fastino/gliner2.5-multi-v1 \
  -v /ssd/gliner/models:/app/models joshsgoldstein/gliner-api:latest
```

Because the model cache is a bind mount on the host, replacing the container
does not re-download weights.

Before replacing a working image, record what you are replacing so you can roll
back:

```bash
docker inspect --format '{{.Image}}' gliner-api
docker images --digests joshsgoldstein/gliner-api
```

### Build the image on the box

Building happens from the repo checkout, and requires BuildKit for the wheel
cache mounts.

```bash
cd ~/gliner-jetson-api
make docker-build          # cached
make docker-build-no-cache # full rebuild; only when necessary
```

Read [../JETSON.md](../JETSON.md) before changing anything in the `Dockerfile`.
The wheel index, the `--no-deps` installs, the `numpy<2` re-pin, and the ONNX
Runtime removal are all load-bearing.

---

## Start / stop / restart

```bash
docker stop gliner-api
docker start gliner-api
docker restart gliner-api
```

`tini` is PID 1 in the image, so `docker stop` delivers `SIGTERM` cleanly and
the container exits without being killed.

Status:

```bash
docker ps --filter name=gliner-api
docker inspect --format '{{.State.Status}} restarts={{.RestartCount}}' gliner-api
```

`--restart unless-stopped` means the container comes back after a host reboot
and after a crash. A climbing `RestartCount` is a crash loop — go to the logs.

---

## Health check

Liveness:

```bash
curl -s http://192.168.1.177:8013/health
```

Expected on a healthy box: `"status":"ok"`, `"loaded":true`, `"device":"cuda"`,
`"gpu":"Orin"`. `"device":"cpu"` on a Jetson is a fault, not a degraded mode —
see [Common failures](#common-failures).

Installed library version:

```bash
curl -s http://192.168.1.177:8013/version
```

Functional check — proves the model actually runs a forward pass, which
`/health` does not:

```bash
curl -s -X POST http://192.168.1.177:8013/extract_entities \
  -H "Content-Type: application/json" \
  -d '{"text":"Patient received 400mg ibuprofen for severe headache at 2 PM.",
       "labels":["medication","dosage","symptom","time"]}'
```

Expected:

```json
{"entities":{"medication":["ibuprofen"],"dosage":["400mg"],"symptom":["headache"],"time":["2 PM"]}}
```

One-liner for a scripted check:

```bash
curl -sf http://192.168.1.177:8013/health | grep -q '"loaded":true' \
  && echo OK || echo FAIL
```

With `MODEL_PRELOAD=0`, `loaded` is `false` until the first inference request.
Use the functional check above, not `loaded`, as readiness in that
configuration.

---

## Swapping the model

The model is chosen entirely by the `MODEL_ID` environment variable, which is
fixed at container creation. Swapping means recreating the container.

```bash
docker rm -f gliner-api

docker run -d \
  --name gliner-api \
  --runtime nvidia \
  --restart unless-stopped \
  -p 8013:8012 \
  -e MODEL_ID=fastino/gliner2.5-base-v1 \
  -v /ssd/gliner/models:/app/models \
  joshsgoldstein/gliner-api:latest
```

Confirm the swap took:

```bash
curl -s http://192.168.1.177:8013/health
```

`model_id` and `model_path` should reflect the new model. On builds that report
`architecture`, GLiNER2.5 models show `"architecture":"boundary"` and
`"model_class":"AutoExtractor"`; GLiNER2 models show `"span"` and `"GLiNER2"`.

### Pre-warm weights before the swap

To avoid a cold container sitting through a multi-GB download while it is
already receiving traffic, pull the weights first:

```bash
docker run --rm \
  -e MODEL_ID=fastino/gliner2.5-multi-v1 \
  -v /ssd/gliner/models:/app/models \
  joshsgoldstein/gliner-api:latest \
  python3 /app/download-model.py
```

`download-model.py` is a no-op if `/app/models/<MODEL_ID>` is already populated.

### Which models exist

Full comparison in [wiki.md](wiki.md#model-family). Short version:

| `MODEL_ID` | Params | Arch | Language | On-disk |
|---|---|---|---|---|
| `fastino/gliner2-base-v1` | 205M | span | English | — |
| `fastino/gliner2-large-v1` | 340M | span | English | 1.9 GB |
| `fastino/gliner2.5-small-v1` | 74M | boundary | English | — |
| `fastino/gliner2.5-base-v1` | 194M | boundary | English | — |
| `fastino/gliner2.5-multi-v1` | 287M | boundary | Multilingual | 1.1 GB |

### Prerequisites for GLiNER2.5 models

GLiNER2.5 checkpoints declare `"architecture": "boundary"` in `config.json` and
**must** be loaded with `AutoExtractor.from_pretrained()`. The legacy
`GLiNER2.from_pretrained()` span loader will not dispatch them. `app.py` reads
`config.json` and picks the loader, so this is automatic — but only if the image
was built with a `gliner2` release that exposes `AutoExtractor`. Both loaders
are available in `gliner2` 2.0.0, which requires `transformers<5`.

Additionally, `gliner2.5-multi-v1` uses an mDeBERTa tokenizer that requires
`protobuf` to be installed. Without it, model load fails with an `ImportError`
at tokenizer construction — not a download or CUDA error. If you are swapping to
the multilingual model for the first time on a given image, verify:

```bash
docker exec gliner-api pip3 show protobuf
```

If it is missing, the fix belongs in `requirements.txt` and a rebuild, not in a
`docker exec pip install` that vanishes on the next redeploy.

---

## Rollback

### Roll back the model

Recreate the container with the previous `MODEL_ID`. Because both models' weights
persist in `/ssd/gliner/models`, this is fast — no download.

```bash
docker rm -f gliner-api
docker run -d --name gliner-api --runtime nvidia --restart unless-stopped \
  -p 8013:8012 -e MODEL_ID=fastino/gliner2.5-base-v1 \
  -v /ssd/gliner/models:/app/models joshsgoldstein/gliner-api:latest
curl -s http://192.168.1.177:8013/health
```

### Roll back the image

Run a specific digest or tag rather than `latest`:

```bash
docker images --digests joshsgoldstein/gliner-api
docker rm -f gliner-api
docker run -d --name gliner-api --runtime nvidia --restart unless-stopped \
  -p 8013:8012 -e MODEL_ID=fastino/gliner2.5-base-v1 \
  -v /ssd/gliner/models:/app/models \
  joshsgoldstein/gliner-api@sha256:<digest>
```

This is why you record the digest before a redeploy. Do not prune images on this
box casually — `docker image prune -a` removes your rollback target, and
rebuilding the Jetson image is slow.

### Roll back a config change

Config lives only in `-e` flags on the `docker run` line. There is no config
file to revert. Keep the working `docker run` invocation in the repo or in your
shell history and re-run it.

---

## Logs

```bash
docker logs -f gliner-api            # follow
docker logs --tail 200 gliner-api    # recent
docker logs --since 15m gliner-api   # windowed
```

### Signals of a healthy start

```
Uvicorn running on http://0.0.0.0:8012
GLiNER2 loaded on GPU (Orin).
```

or, for a GLiNER2.5 checkpoint:

```
Loading model from disk (architecture=boundary)...
AutoExtractor loaded on GPU (Orin).
```

Absence of `free(): invalid pointer` / `malloc()` messages is also a healthy
signal — those indicate the ONNX Runtime crash class documented in
[../JETSON.md](../JETSON.md).

### Per-request timing

Every inference logs a timing breakdown:

```
Completed extract_entities total_s=0.181 queue_wait_s=0.000 model_get_s=0.000 infer_s=0.180
```

Read it as:

| Field | Meaning | What a high value tells you |
|---|---|---|
| `queue_wait_s` | Time waiting for a semaphore slot | Saturation. Requests are queuing behind the GPU |
| `model_get_s` | Time in `_get_model()` | Non-zero only on the first request after a cold start with `MODEL_PRELOAD=0` |
| `infer_s` | Actual forward pass | Long documents, a larger model, or GPU contention from another workload |
| `total_s` | End to end | — |

Grep for saturation:

```bash
docker logs --since 1h gliner-api | grep -E 'queue_wait_s=[1-9]'
```

Count errors:

```bash
docker logs --since 1h gliner-api | grep -cE ' (4[0-9]{2}|5[0-9]{2}) '
```

Docker's default `json-file` driver keeps growing. If this container has been up
for months, check and cap it:

```bash
du -sh "$(docker inspect --format '{{.LogPath}}' gliner-api)"
```

---

## Common failures

### `/health` reports `"device":"cpu"` on the Jetson

Inference still works, just slowly. The GPU was not visible to the container.

Check, in order:

```bash
docker inspect --format '{{.HostConfig.Runtime}}' gliner-api   # expect: nvidia
docker exec gliner-api python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected inside the container: `2.8.0 12.6 True`.

- Runtime is not `nvidia` → recreate the container with `--runtime nvidia`.
- Torch reports a `+cpu` suffix → the image was built against the wrong wheel
  index. This is the pip-fell-back-to-PyPI failure described in
  [../JETSON.md](../JETSON.md). Rebuild.

### `CUDA error: no kernel image is available for execution on the device`

The image was built with `sbsa/cu130` wheels. Those are Grace/Blackwell
`sm_90`/`sm_100` builds and will never run on Orin's `sm_87`, no matter how much
you patch around the missing system libraries. Torch will import,
`torch.cuda.is_available()` will return `True`, and it will even print `Orin` —
and then die on the first kernel launch.

Rebuild from the `jp6/cu126` wheel pinned in the `Dockerfile`. Do not chase this
one; it is a dead end, documented at length in [../JETSON.md](../JETSON.md).

### `ModuleNotFoundError` on startup (e.g. `peft`)

`gliner` and `gliner2` are installed with `--no-deps` so pip cannot resolve an
x86 CPU-only torch over the Jetson CUDA wheel. The cost is that new upstream
dependencies are silently dropped: the build succeeds and the container dies at
import.

Fix: add the missing package to `requirements.txt` and rebuild. When bumping the
`gliner2` pin, check `pip3 show gliner2` for the new `Requires:` list and add
anything new in the same commit.

### `ImportError` while loading `gliner2.5-multi-v1`

The mDeBERTa tokenizer needs `protobuf`. Add it to `requirements.txt` and
rebuild. See [Prerequisites for GLiNER2.5 models](#prerequisites-for-gliner25-models).

### GLiNER2.5 model loads as a GLiNER2 and misbehaves or fails to dispatch

`config.json` declares `"architecture": "boundary"`, which requires
`AutoExtractor.from_pretrained()`. Confirm what the service actually did:

```bash
curl -s http://192.168.1.177:8013/health
docker logs gliner-api | grep 'Loading model from disk'
```

If `architecture` reads `span` for a 2.5 model, `app.py` could not read
`config.json` — it logs a warning and falls back to `span`. Check the mounted
model directory is complete:

```bash
docker exec gliner-api ls /app/models/fastino/gliner2.5-multi-v1
```

If the field is missing from `/health` entirely, the running image predates
GLiNER2.5 dispatch. Rebuild.

### `503 Server is busy`

Expected backpressure, not a fault. With `MAX_CONCURRENT_INFERENCES=1` the
service serializes GPU work and rejects anything that cannot get a slot within
`INFERENCE_ACQUIRE_TIMEOUT_SECONDS` (default 10).

Options, in order of preference:

1. Have the client retry with jitter. This is the correct answer for a
   single-GPU box.
2. Raise `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` to let clients wait longer.
3. Raise `MAX_CONCURRENT_INFERENCES` — but only if the box has the headroom, and
   knowing GPU execution is still effectively serialized at the device.
4. Use `POST /extract_entities_batch`. Sixteen documents in one batched call
   measured 4.2x faster per document than sixteen sequential calls (see
   [Batching](#batching)). Requires a boundary checkpoint; batch size is capped
   by `MAX_BATCH_SIZE` (default 64).

### `504 ... timed out after 120.0s`

A single inference exceeded `REQUEST_TIMEOUT_SECONDS`. Usually a very long
document. Split the input; raise the timeout only if you have measured that the
work genuinely needs it.

### `413 'text' exceeds MAX_TEXT_CHARS`

Client-side problem. Chunk the document to under 20000 characters, or raise
`MAX_TEXT_CHARS` — knowing that longer texts cost proportionally more GPU time
and push you toward the 504 above.

### First request after start is very slow

Expected if the container runs with `MODEL_PRELOAD=0` (the setting used by the
`make docker-run*` targets, chosen to avoid startup allocator issues seen on
Jetson). The first request pays the load cost. To move that cost to startup
instead, set `MODEL_PRELOAD=1` and accept a longer container start.

### Container restarts in a loop

```bash
docker inspect --format 'restarts={{.RestartCount}} exit={{.State.ExitCode}}' gliner-api
docker logs --tail 100 gliner-api
```

Most crash loops here are one of: a missing Python dependency (see above), an
OOM kill, or a corrupt/partial model download. For the last one, delete the
model directory on the host and let it re-download:

```bash
docker rm -f gliner-api
sudo rm -rf /ssd/gliner/models/fastino/<model-name>
# then re-run the docker run command
```

### Service unreachable from the LAN

```bash
# on jarvita-agx — does it answer locally?
curl -s http://127.0.0.1:8013/health
docker port gliner-api
```

If it answers on the host but not from the LAN, the problem is network/firewall,
not the service. If it does not answer on the host either, check `docker ps` —
the container may be stopped or the publish mapping may be missing.

---

## Memory and sizing on a shared box

`jarvita-agx` is a 64GB AGX Orin with unified CPU/GPU memory, and it is shared —
these numbers were measured with another LLM resident on the same box. Treat
them as indicative of this deployment, not as a rigorous benchmark.

### Measured footprint

| Measurement | Value |
|---|---|
| `gliner2-large-v1` container footprint, idle | 4.28 GiB |
| `gliner2-large-v1` container footprint, peak | 4.69 GiB |
| Both models loaded together, GPU allocated | 3.11 GB |
| `gliner2-large-v1` on disk | 1.9 GB |
| `gliner2.5-multi-v1` on disk | 1.1 GB |

The gap between GPU-allocated and container footprint is the Python process,
torch, and CUDA context. Budget by container footprint, not by weight size.

### Measured latency

Average of 5 runs per test, on GPU, with another LLM sharing the box. Indicative
only — this is not a controlled evaluation.

| Test | `gliner2.5-multi` | `gliner2-large` |
|---|---|---|
| clinical entity extraction | 117.6 ms | 171.9 ms |
| business entity extraction | 172.0 ms | 149.5 ms |
| spanish entity extraction | 103.1 ms | 170.8 ms |
| french entity extraction | 107.9 ms | 164.8 ms |
| classification | 79.6 ms | 111.2 ms |

### Batching

Same conditions, `gliner2.5-multi`, 16 identical documents:

| Mode | Total | Per document |
|---|---|---|
| Batched | 384 ms | 24.0 ms |
| Sequential | 1607 ms | 100.4 ms |

**4.2x speedup.** On a box with one GPU and a semaphore width of 1, batching is
the highest-leverage throughput change available. Prefer it over raising
`MAX_CONCURRENT_INFERENCES`.

This was measured against the model directly, and is reachable over HTTP via
`POST /extract_entities_batch` on a boundary checkpoint. Prefer batching over
raising `MAX_CONCURRENT_INFERENCES` for bulk work: the batch call holds a single
inference slot for the whole set instead of contending for the semaphore per
document.

### Watching memory live

```bash
docker stats --no-stream gliner-api
sudo tegrastats --interval 1000     # host-wide, includes GPU
```

### Sizing guidance

- Keep `MAX_CONCURRENT_INFERENCES=1` unless you have measured headroom. Raising
  it multiplies peak activation memory without buying much throughput, since the
  GPU serializes the work anyway.
- Uvicorn workers each hold their own model copy. `WORKERS=2` roughly doubles
  the footprint. On a shared box this is almost never the right lever.
- If you need both a 2.x and a 2.5 model available, running them as two
  containers on two host ports costs roughly two container footprints. Loading
  both into one process measured 3.11 GB of GPU allocation, but this service
  holds one model per process by design.
- Leave real headroom for whatever else is resident. Unified memory means an
  LLM's KV cache and this service's activations compete for the same pool.

---

## Related documents

- [api.md](api.md) — endpoint reference and error semantics
- [wiki.md](wiki.md) — what GLiNER2 is and why it runs here
- [examples.ipynb](examples.ipynb) — runnable client examples
- [../JETSON.md](../JETSON.md) — build-level Jetson decisions and traps
- [../README.md](../README.md) — project overview and local development
