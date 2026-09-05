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

Under that policy the service has run 7+ days continuously on this box. A short
uptime you did not cause is a signal: check `RestartCount` and the logs before
assuming it is fine.

---

## Health check

### `/health/deep` — the one monitoring should poll

```bash
curl -s http://192.168.1.177:8013/health/deep
```

Verified response on a healthy box:

```json
{"status":"ok","model_id":"fastino/gliner2.5-base-v1","model_path":"./models/fastino/gliner2.5-base-v1","loaded":true,"architecture":"boundary","model_class":"BoundaryExtractor","device":"cuda","gpu":"Orin","max_concurrent_inferences":1,"inflight":0,"saturated":false,"probe":{"ok":true,"latency_ms":125.2,"result":{"entities":{"company":["Apple"]}}}}
```

`/health/deep` always runs a real inference — it is `/health?probe=1` under a
fixed path. On failure or timeout it returns **`503`** with
`"status":"degraded"` and `"probe":{"ok":false,...,"error":"..."}`. So the
scripted check is just `curl -f`:

```bash
curl -sf http://192.168.1.177:8013/health/deep >/dev/null && echo OK || echo DEGRADED
```

Two properties make this the right monitoring target rather than plain
`/health`:

1. **A loaded model is not a working one.** `/health` reports `"status":"ok"`
   for a process whose inference path is wedged, and will keep doing so for
   hours. Only a forward pass proves the service works.
2. **The probe deliberately bypasses the inference semaphore.** With
   `MAX_CONCURRENT_INFERENCES=1`, a probe that queued behind a hung request
   would hang too, and a wedged worker would look identical to a busy one.
   Bypassing the semaphore keeps them distinguishable.

Measured on this box: while a 48-document batch held the only inference slot,
`/health/deep` returned `200` in **1.56 s** while a normal inference request
queued for **4.08 s**. Single measurement against one saturating batch — enough
to show the bypass works, not a latency SLO.

The probe is bounded by `HEALTH_PROBE_TIMEOUT_SECONDS` (default 5) and runs
against `HEALTH_PROBE_TEXT` (default `Apple is based in Cupertino.`) with the
fixed label `["company"]`. Set `HEALTH_PROBE_TEXT` to something in your own
domain if the default English business sentence is not representative — but keep
it short, because every poll pays its latency.

Suggested alerting:

| Condition | Meaning | Action |
|---|---|---|
| `503` from `/health/deep` | Inference wedged, or slower than the probe timeout | Check logs, then restart the container |
| `200` but `probe.latency_ms` climbing well over baseline (110–130 ms here) | GPU contention from another workload on the box | `tegrastats`; check what else is resident |
| `200` with `"saturated": true` sustained | Legitimately busy, not broken | Move the client to a batch route before touching concurrency |

Do **not** poll `/health/deep` aggressively. Each poll is a real forward pass on
a box with one GPU. Once every 30–60 s is plenty.

### `/health` — cheap liveness

```bash
curl -s http://192.168.1.177:8013/health
```

Same body without `probe`, and it never touches the GPU. This is the right
choice for a container `HEALTHCHECK` (frequent, must not consume GPU) and the
wrong choice for monitoring.

Expected on a healthy box: `"status":"ok"`, `"loaded":true`, `"device":"cuda"`,
`"gpu":"Orin"`. `"device":"cpu"` on a Jetson is a fault, not a degraded mode —
see [Common failures](#common-failures).

Two load fields worth watching:

| Field | Read it as |
|---|---|
| `inflight` | Requests holding an inference slot right now |
| `saturated` | `inflight >= max_concurrent_inferences`. `true` means the next request queues and will `503` after `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` |

`max_concurrent_inferences` reports the **live** semaphore width — whatever
`-e MAX_CONCURRENT_INFERENCES` the container was created with, not necessarily
the `1` default. Read it from `/health` rather than assuming it.

### Version and provenance

```bash
curl -s http://192.168.1.177:8013/version
```

Verified response:

```json
{"gliner2":"2.0.0","model_id":"fastino/gliner2.5-base-v1","model_revision":"4a3138e2432c24b4","architecture":"boundary","model_class":"BoundaryExtractor","torch":"2.8.0"}
```

`model_revision` is a 16-hex-char fingerprint of the weights on disk, derived
from `MODEL_ID` plus the size and mtime of `model.safetensors`. It is stable
across restarts and changes when the weights are replaced, which makes it the
value to stamp onto extracted rows: without it, extractions from two different
checkpoints are indistinguishable once stored. It is a local fingerprint — do
not compare it across hosts, because mtime is a local fact.

Record it before and after any model swap or image redeploy:

```bash
curl -s http://192.168.1.177:8013/version | tee /tmp/gliner-version-before.json
```

### Functional check

Beyond the probe, a real request in your own shape:

```bash
curl -s -X POST http://192.168.1.177:8013/extract_entities \
  -H "Content-Type: application/json" \
  -d '{"text":"Patient received 400mg ibuprofen for severe headache at 2 PM.",
       "labels":["medication","dosage","symptom","time"]}'
```

Verified on `fastino/gliner2.5-base-v1`:

```json
{"entities":{"medication":["ibuprofen"],"dosage":["400mg"],"symptom":["severe headache"],"time":["2 PM"]}}
```

Span boundaries are model-specific — this checkpoint returns
`"severe headache"` where an earlier one returned `"headache"`. Do not treat an
exact-string diff after a model swap as a fault without checking that first.

With `MODEL_PRELOAD=0`, `loaded` is `false` until the first inference request.
Use `/health/deep` or the functional check, not `loaded`, as readiness in that
configuration.

---

## Configuration

Every knob is an environment variable set with `-e` on the `docker run` line.
There is no config file. Changing any of them means recreating the container.

| Variable | Default | What it controls |
|---|---|---|
| `MODEL_ID` | `fastino/gliner2.5-base-v1` | Which checkpoint is served. A span checkpoint makes the five boundary-only routes return `501` |
| `MODEL_DIR` | `./models` | Weights location inside the container (bind-mounted from `/ssd/gliner/models`) |
| `MODEL_PRELOAD` | `1` | `0` defers the load to the first request |
| `MAX_CONCURRENT_INFERENCES` | `1` | Semaphore width |
| `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | `10` | Slot wait before `503` |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Per-request inference budget before `504` |
| `MAX_TEXT_CHARS` | `20000` | Per-document character cap; `413` |
| `MAX_LABELS` | `256` | Label / relation / class list cap; `400` |
| `MAX_SCHEMA_FIELDS` | `256` | Schema field cap; `400` |
| `MAX_BATCH_SIZE` | `64` | Documents per batch request; `413`. Also the ceiling `batch_size` is clamped to |
| `STRUCTURED_DEFAULT_THRESHOLD` | `0.7` | `/extract_structured` only. At 0.5 a 3+ field schema emits a duplicate, span-shifted record |
| `MAX_BATCH_CHARS` | `40000` | **Total** characters summed across a batch; `413` |
| `HEALTH_PROBE_TIMEOUT_SECONDS` | `5` | Probe budget before `/health/deep` returns `503 degraded` |
| `HEALTH_PROBE_TEXT` | `Apple is based in Cupertino.` | Text the probe runs against, with the fixed label `["company"]` |

### Batch sizing and the OOM bound

`MAX_BATCH_SIZE` and `MAX_BATCH_CHARS` are two separate bounds and you need both.

**`MAX_BATCH_SIZE` alone does not bound memory.** 64 documents is a small batch
of tweets and a very large batch of contracts; the cap counts documents, and
activation memory scales with characters. 64 documents of `MAX_TEXT_CHARS`
(20000) each is 1.28M characters in one forward pass, and that is what OOMed
this box. `MAX_BATCH_CHARS=40000` is the bound that actually holds — roughly
6.4x below that worst case, or 64 documents averaging ~3100 characters.

Either overrun is a `413` and neither is a crash. A batch can be well inside the
document cap and still be rejected on characters:

```bash
python3 -c "import json; print(json.dumps({'texts': ['a'*19000]*15, 'labels': ['person']}))" \
  | curl -s -X POST http://192.168.1.177:8013/extract_entities_batch \
      -H "Content-Type: application/json" --data @-
```

```json
{"detail":"batch totals 285000 chars, exceeding MAX_BATCH_CHARS=40000. Split the batch."}
```

The number in that message is the **live configured value**, not the table
default above. Confirm what the running container actually enforces before
telling a client what to chunk to:

```bash
docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' gliner-api \
  | grep -E 'MAX_BATCH|MAX_TEXT|MAX_CONCURRENT|HEALTH_PROBE'
```

Guidance:

- **Chunk clients against both bounds.** Accumulate documents until either
  `len(batch) == MAX_BATCH_SIZE` or the running character total would exceed
  `MAX_BATCH_CHARS`, then flush. Chunking on count alone will `413` on long
  documents.
- **Raising `MAX_BATCH_CHARS` is the OOM lever, not `MAX_BATCH_SIZE`.** If you
  raise it, watch `docker stats` and `tegrastats` through a full-size batch
  before trusting it, and remember the LLM sharing this box competes for the
  same unified-memory pool.
- **`batch_size` in the payload is a different thing.** It is the model's
  internal sub-batch, not the request cap, and it is silently clamped to
  `MAX_BATCH_SIZE`. Leave it unset unless you are tuning memory inside a large
  batch.

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

`model_id` and `model_path` should reflect the new model. GLiNER2.5 models show
`"architecture":"boundary"` and a boundary class in `model_class` —
`"BoundaryExtractor"` on the current image; GLiNER2 models show `"span"` and
`"GLiNER2"`.

Then confirm the weights actually changed, and record the new fingerprint:

```bash
curl -s http://192.168.1.177:8013/version
curl -sf http://192.168.1.177:8013/health/deep >/dev/null && echo PROBE_OK || echo PROBE_FAILED
```

`model_revision` should differ from what you recorded before the swap. If it
does not, the container came up on the old weights — check `MODEL_ID` on the
running container with `docker inspect`.

Swapping **to a span checkpoint** disables five routes: `/extract_relations`,
the three other `*_batch` routes, and `/extract_multitask_batch` all begin
returning `501`, and `schema_config.relations` stops being accepted. Check what
your clients call before swapping down from a 2.5 model.

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
BoundaryExtractor loaded on GPU (Orin).
```

The class name in that line is `type(model).__name__` for whatever
`AutoExtractor.from_pretrained()` dispatched to, so it names the concrete
extractor (`BoundaryExtractor`), not `AutoExtractor`. `/health` and `/version`
report the same value in `model_class`.

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

1. **Move the client to a batch route.** This is the highest-leverage change by
   a wide margin — see [Batching](#batching). A batch holds one inference slot
   for the whole set instead of contending for the semaphore per document, so
   the `503` stops happening rather than being retried around. Requires a
   boundary checkpoint.
2. Have the client retry with jitter. Correct for traffic you cannot batch.
3. Raise `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` to let clients wait longer.
4. Raise `MAX_CONCURRENT_INFERENCES` — last resort, only with measured headroom,
   and knowing GPU execution is still effectively serialized at the device.

Confirm it is really saturation and not a wedge before tuning anything:

```bash
curl -s http://192.168.1.177:8013/health | python3 -m json.tool | grep -E 'inflight|saturated'
curl -sf http://192.168.1.177:8013/health/deep >/dev/null && echo PROBE_OK || echo PROBE_FAILED
```

`saturated: true` with `PROBE_OK` is a busy box. `PROBE_FAILED` is a wedged one
— restart it.

### `/health/deep` returns `503` with `"status":"degraded"`

The process is up and `/health` still says `ok`, but a real forward pass did not
complete within `HEALTH_PROBE_TIMEOUT_SECONDS`. Because the probe bypasses the
semaphore, this is **not** explained by the box being busy — a busy box still
answers the probe (measured: 1.56 s under a saturating 48-document batch).

Read `probe.error` first:

```bash
curl -s http://192.168.1.177:8013/health/deep | python3 -m json.tool
```

| `probe.error` | Likely cause |
|---|---|
| `TimeoutError` | Inference wedged, or the GPU is genuinely stuck behind another workload |
| A CUDA error | See the `no kernel image` entry above, or a GPU fault — check `dmesg` and `tegrastats` |
| A model/tokenizer exception | Incomplete or corrupt weights in the bind mount |

If it is a `TimeoutError` and nothing else on the box explains it, restart the
container. Capture `docker logs --tail 200 gliner-api` first — a wedge that
recurs is worth a bug, and the restart destroys the evidence.

A probe that is merely *slow* (returns `200`, latency well above the 110–130 ms
baseline) usually means GPU contention, not a fault. Check what else is
resident before restarting anything.

### `501 ... requires a GLiNER2.5 boundary checkpoint`

The route exists but the loaded model cannot serve it. Five routes are
boundary-only: `/extract_relations`, `/extract_entities_batch`,
`/classify_text_batch`, `/extract_relations_batch`, `/extract_multitask_batch` —
plus `schema_config.relations` on the multitask routes.

```bash
curl -s http://192.168.1.177:8013/health | grep -o '"architecture":"[a-z]*"'
```

`"span"` means a GLiNER2 checkpoint is loaded and the `501` is correct
behaviour, not a fault. Either point the client at the non-batch equivalents or
swap `MODEL_ID` to a 2.5 checkpoint.

`"boundary"` with a `501` anyway means the architecture was detected after the
route's check — check the logs for the `Could not read architecture` warning,
which means `config.json` was unreadable and `app.py` fell back to `span`.

### `504 ... timed out after 120.0s`

A single inference exceeded `REQUEST_TIMEOUT_SECONDS`. Usually a very long
document. Split the input; raise the timeout only if you have measured that the
work genuinely needs it.

### `413 'text' exceeds MAX_TEXT_CHARS`

Client-side problem. Chunk the document to under 20000 characters, or raise
`MAX_TEXT_CHARS` — knowing that longer texts cost proportionally more GPU time
and push you toward the 504 above.

### `413 batch totals N chars, exceeding MAX_BATCH_CHARS`

Also client-side, and a different bound from `MAX_BATCH_SIZE`. The batch was
within the document count and still too large in characters. Split it. See
[Batch sizing and the OOM bound](#batch-sizing-and-the-oom-bound) for why both
bounds exist and which one to raise.

### `400 Unknown payload key(s): [...]`

Not a regression — the service validates payload keys strictly, and
`schema_config` keys too. A key that used to be silently ignored is now a `400`
listing exactly what the route accepts:

```json
{"detail":"Unknown payload key(s): ['treshold']. Allowed: ['include_confidence', 'include_spans', 'labels', 'max_len', 'overlap_policy', 'text', 'threshold']."}
```

This usually surfaces right after a client upgrade, and the cause is almost
always a typo or a task field sent at the top level instead of nested under
`schema_config`. Read the `Allowed:` list; it is generated from the route's own
key set, so it is never stale.

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

Three checkpoints, same inputs, average of 5 runs on GPU with another LLM
sharing the box:

| `MODEL_ID` | Avg latency | On disk |
|---|---|---|
| `fastino/gliner2.5-base-v1` | 83 ms | 748 MB |
| `fastino/gliner2.5-multi-v1` | 94 ms | 1.1 GB |
| `fastino/gliner2-large-v1` | 109 ms | 1.9 GB |

`gliner2.5-base` is the fastest and the smallest of the three, which is why it
is the current default. On the same run, `gliner2.5-multi` produced **duplicate,
garbled records on structured extraction** and under-recalled English entities;
`gliner2.5-base` did neither. **Caveat: six sentences.** That is not a rigorous
evaluation, and it is not evidence about the multilingual model's multilingual
behaviour — it is a reason to re-check outputs against your own texts whenever
you change `MODEL_ID`.

Older per-task numbers, average of 5 runs, same shared-box conditions:

| Test | `gliner2.5-multi` | `gliner2-large` |
|---|---|---|
| clinical entity extraction | 117.6 ms | 171.9 ms |
| business entity extraction | 172.0 ms | 149.5 ms |
| spanish entity extraction | 103.1 ms | 170.8 ms |
| french entity extraction | 107.9 ms | 164.8 ms |
| classification | 79.6 ms | 111.2 ms |

### Batching

On a box with one GPU and a semaphore width of 1, batching is the
highest-leverage throughput change available. Prefer it over raising
`MAX_CONCURRENT_INFERENCES`: a batch call holds a single inference slot for the
whole set instead of contending for the semaphore per document.

`gliner2.5-multi`, 16 identical documents, model-level:

| Mode | Total | Per document |
|---|---|---|
| Batched | 384 ms | 24.0 ms |
| Sequential | 1607 ms | 100.4 ms |

**4.2x.**

#### Use `/extract_multitask_batch`, not three batch calls

Measured over 32 documents on `fastino/gliner2.5-base-v1`:

| Approach | Per document | Documents/min |
|---|---|---|
| 3 sequential single calls per document (entities, classification, structure) | 274.9 ms | 218 |
| 1 `/extract_multitask_batch` call | 25.8 ms | 2326 |

**10.7x.** One run over 32 documents on a shared box — indicative of the order
of magnitude, not a controlled benchmark.

The gain has two independent sources, and it is worth keeping them apart when
advising a client:

1. **Batching** amortizes per-call and per-request overhead across documents.
2. **Multitask** runs entities + classification + structure in one forward pass
   rather than three.

Calling `/extract_entities_batch`, `/classify_text_batch` and
`/extract_structured` separately buys (1) but not (2). If a client wants more
than one task shape from the same documents, put them in one `schema_config` and
send one `/extract_multitask_batch` request. `schema_config` also accepts
`relations` on a boundary checkpoint, so relation extraction rides along for
free in the same pass.

#### Schema caching is not an optimization

Reusing a built schema across batch calls was measured at **127.41 ms** against
**127.42 ms** for rebuilding it, over 16 documents. That is a 0.01 ms
difference — noise. Do not add a schema cache, and do not accept a change that
adds one on performance grounds.

### Watching memory live

```bash
docker stats --no-stream gliner-api
sudo tegrastats --interval 1000     # host-wide, includes GPU
```

### Sizing guidance

- Keep `MAX_CONCURRENT_INFERENCES=1` unless you have measured headroom. Raising
  it multiplies peak activation memory without buying much throughput, since the
  GPU serializes the work anyway. Batching is the lever; concurrency is not.
- **`MAX_BATCH_CHARS` is the real memory bound on batch requests**, not
  `MAX_BATCH_SIZE` — 64 documents of 20000 characters each is what OOMed this
  box. Raise the character cap only after watching `docker stats` through a
  full-size batch. See
  [Batch sizing and the OOM bound](#batch-sizing-and-the-oom-bound).
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
