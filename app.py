import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, TYPE_CHECKING

import torch
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from gliner2 import GLiNER2

load_dotenv()

# Reduce tokenizer thread noise and native threading contention.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --- Logger ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gliner2-server")

# --- Settings ---
MODEL_ID = os.getenv("MODEL_ID", "fastino/gliner2.5-base-v1")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_ID)
MODEL_PRELOAD = os.getenv("MODEL_PRELOAD", "1").lower() not in {"0", "false", "no"}
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCES", "1")))
INFERENCE_ACQUIRE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_ACQUIRE_TIMEOUT_SECONDS", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_TEXT_CHARS = max(1, int(os.getenv("MAX_TEXT_CHARS", "20000")))
MAX_LABELS = max(1, int(os.getenv("MAX_LABELS", "256")))
MAX_SCHEMA_FIELDS = max(1, int(os.getenv("MAX_SCHEMA_FIELDS", "256")))
MAX_BATCH_SIZE = max(1, int(os.getenv("MAX_BATCH_SIZE", "64")))
# Total characters across a batch. MAX_BATCH_SIZE alone does not bound memory:
# 64 documents of MAX_TEXT_CHARS each is what OOMs the box.
MAX_BATCH_CHARS = max(1, int(os.getenv("MAX_BATCH_CHARS", "200000")))
HEALTH_PROBE_TIMEOUT_SECONDS = float(os.getenv("HEALTH_PROBE_TIMEOUT_SECONDS", "5"))
HEALTH_PROBE_TEXT = os.getenv("HEALTH_PROBE_TEXT", "Apple is based in Cupertino.")

app = FastAPI(title="GLiNER2 API (Jetson)")
_model: Any = None
_model_arch: str | None = None
_model_init_lock = threading.Lock()
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
_inflight = 0


def _ensure_model_downloaded() -> None:
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    if not os.listdir(LOCAL_MODEL_PATH):
        log.info("Downloading model %s to %s ...", MODEL_ID, LOCAL_MODEL_PATH)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_MODEL_PATH,
            local_dir_use_symlinks=False,
        )
        log.info("Download complete.")
    else:
        log.info("Using existing local model at: %s", LOCAL_MODEL_PATH)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _model_revision() -> str | None:
    """Stable fingerprint of the loaded weights, for provenance stamping."""
    weights = os.path.join(LOCAL_MODEL_PATH, "model.safetensors")
    try:
        stat = os.stat(weights)
        seed = f"{MODEL_ID}:{stat.st_size}:{int(stat.st_mtime)}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    except OSError:
        return None


def _detect_architecture() -> str:
    """GLiNER2.5 checkpoints declare architecture 'boundary'; GLiNER2 ones are 'span'.

    The loaders are not interchangeable: GLiNER2.from_pretrained is the legacy span
    loader and will not dispatch a boundary checkpoint.
    """
    config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            arch = (json.load(fh).get("architecture") or "").strip().lower()
        return arch or "span"
    except (OSError, ValueError) as exc:
        log.warning("Could not read architecture from %s (%s); assuming 'span'.", config_path, exc)
        return "span"


def _get_model() -> Any:
    global _model, _model_arch
    if _model is not None:
        return _model

    # Prevent duplicate model download/load when multiple requests arrive together.
    with _model_init_lock:
        if _model is None:
            _ensure_model_downloaded()
            arch = _detect_architecture()
            log.info("Loading model from disk (architecture=%s)...", arch)

            if arch == "boundary":
                from gliner2 import AutoExtractor

                # AutoExtractor places the model itself; no .to() afterwards.
                _model = AutoExtractor.from_pretrained(LOCAL_MODEL_PATH, map_location=DEVICE)
            else:
                from gliner2 import GLiNER2

                _model = GLiNER2.from_pretrained(LOCAL_MODEL_PATH)
                if DEVICE == "cuda":
                    _model = _model.to(DEVICE)

            _model_arch = arch
            if DEVICE == "cuda":
                log.info("%s loaded on GPU (%s).", type(_model).__name__, torch.cuda.get_device_name(0))
            else:
                log.info("%s loaded on CPU.", type(_model).__name__)
            _model.eval()
    return _model


@app.on_event("startup")
def startup_event() -> None:
    if MODEL_PRELOAD:
        _get_model()


async def _probe_inference(timeout: float) -> Any:
    """Run a tiny inference WITHOUT taking an inference slot.

    Deliberate: with MAX_CONCURRENT_INFERENCES=1 a probe that queues behind a
    hung request hangs too, so a wedged worker and a busy one look identical.
    Bypassing the semaphore keeps the two distinguishable.
    """
    model = await asyncio.wait_for(asyncio.to_thread(_get_model), timeout=timeout)
    return await asyncio.wait_for(
        asyncio.to_thread(model.extract_entities, HEALTH_PROBE_TEXT, ["company"]),
        timeout=timeout,
    )


@app.get("/health")
async def health(probe: bool = False) -> Dict[str, Any]:
    """Liveness, and with ?probe=1 an actual inference.

    A loaded model is not a working one: the process can be up and serving
    /health while inference is wedged. Monitoring should call ?probe=1 so a
    stuck worker surfaces immediately rather than hours later.
    """
    body: Dict[str, Any] = {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_path": LOCAL_MODEL_PATH,
        "loaded": _model is not None,
        "architecture": _model_arch,
        "model_class": type(_model).__name__ if _model is not None else None,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "max_concurrent_inferences": MAX_CONCURRENT_INFERENCES,
    }

    body["inflight"] = _inflight
    body["saturated"] = _inflight >= MAX_CONCURRENT_INFERENCES

    if not probe:
        return body

    started = time.monotonic()
    try:
        result = await _probe_inference(HEALTH_PROBE_TIMEOUT_SECONDS)
    except Exception as exc:  # TimeoutError, model errors
        body["status"] = "degraded"
        body["probe"] = {
            "ok": False,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise HTTPException(status_code=503, detail=body)

    body["probe"] = {
        "ok": True,
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "result": result,
    }
    return body


@app.get("/health/deep")
async def health_deep() -> Dict[str, Any]:
    """Always probes. Equivalent to /health?probe=1; point monitoring here."""
    return await health(probe=True)


@app.get("/version")
async def version() -> Dict[str, Any]:
    """Library *and* model identity.

    Downstream consumers stamp this onto every extracted row, so it has to
    identify which model produced a result -- otherwise extractions from
    different checkpoints are silently indistinguishable once stored.
    """
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    try:
        gliner2_version = pkg_version("gliner2")
    except PackageNotFoundError:
        gliner2_version = "unknown"

    return {
        "gliner2": gliner2_version,
        "model_id": MODEL_ID,
        "model_revision": _model_revision(),
        "architecture": _model_arch,
        "model_class": type(_model).__name__ if _model is not None else None,
        "torch": torch.__version__,
    }

# --- Endpoints ---

def _validate_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Provide non-empty 'text'.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"'text' exceeds MAX_TEXT_CHARS={MAX_TEXT_CHARS}.",
        )
    return text


def _validate_labels(labels: Any) -> Any:
    if isinstance(labels, list):
        if not labels:
            raise HTTPException(status_code=400, detail="Provide non-empty 'labels'.")
        if len(labels) > MAX_LABELS:
            raise HTTPException(status_code=400, detail=f"'labels' exceeds MAX_LABELS={MAX_LABELS}.")
        return labels
    if isinstance(labels, dict):
        if not labels:
            raise HTTPException(status_code=400, detail="Provide non-empty 'labels'.")
        if len(labels) > MAX_LABELS:
            raise HTTPException(status_code=400, detail=f"'labels' exceeds MAX_LABELS={MAX_LABELS}.")
        return labels
    raise HTTPException(status_code=400, detail="Provide 'labels' as list or dict.")


def _validate_schema(schema: Any) -> Dict[str, Any]:
    if not isinstance(schema, dict) or not schema:
        raise HTTPException(status_code=400, detail="Provide non-empty 'schema' object.")
    if len(schema) > MAX_SCHEMA_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"'schema' exceeds MAX_SCHEMA_FIELDS={MAX_SCHEMA_FIELDS}.",
        )
    return schema


SCHEMA_CONFIG_KEYS = {"entities", "classification", "structure", "relations"}


def _validate_schema_config(config: Any) -> Dict[str, Any]:
    if not isinstance(config, dict) or not config:
        raise HTTPException(status_code=400, detail="Provide non-empty 'schema_config' object.")

    # Previously an unrecognised key was silently ignored, so a typo -- or a
    # 'relations' block against a build that did not support it -- looked like a
    # successful extraction that was quietly missing data.
    _reject_unknown_keys(config, SCHEMA_CONFIG_KEYS, where="schema_config")

    relations = config.get("relations")
    if relations is not None:
        if not isinstance(relations, list):
            raise HTTPException(status_code=400, detail="'schema_config.relations' must be a list.")
        if len(relations) > MAX_LABELS:
            raise HTTPException(
                status_code=400,
                detail=f"'schema_config.relations' exceeds MAX_LABELS={MAX_LABELS}.",
            )

    entities = config.get("entities")
    if entities is not None:
        if not isinstance(entities, list):
            raise HTTPException(status_code=400, detail="'schema_config.entities' must be a list.")
        if len(entities) > MAX_LABELS:
            raise HTTPException(
                status_code=400,
                detail=f"'schema_config.entities' exceeds MAX_LABELS={MAX_LABELS}.",
            )

    classification = config.get("classification")
    if classification is not None:
        if not isinstance(classification, dict):
            raise HTTPException(status_code=400, detail="'schema_config.classification' must be an object.")
        cls_labels = classification.get("labels", [])
        if not isinstance(cls_labels, list):
            raise HTTPException(status_code=400, detail="'schema_config.classification.labels' must be a list.")
        if len(cls_labels) > MAX_LABELS:
            raise HTTPException(
                status_code=400,
                detail=f"'schema_config.classification.labels' exceeds MAX_LABELS={MAX_LABELS}.",
            )

    structure = config.get("structure")
    if structure is not None:
        if not isinstance(structure, dict):
            raise HTTPException(status_code=400, detail="'schema_config.structure' must be an object.")
        fields = structure.get("fields", [])
        if not isinstance(fields, list):
            raise HTTPException(status_code=400, detail="'schema_config.structure.fields' must be a list.")
        if len(fields) > MAX_SCHEMA_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"'schema_config.structure.fields' exceeds MAX_SCHEMA_FIELDS={MAX_SCHEMA_FIELDS}.",
            )
    return config


# --- Inference options (forwarded to gliner2) -------------------------------
# Every gliner2 extract/classify/batch method accepts these. They used to be
# accepted by the API and silently dropped, which meant callers could not tune
# precision at all and got no confidence or offsets back.

INFERENCE_OPTION_KEYS = {
    "threshold", "include_confidence", "include_spans", "max_len", "overlap_policy",
}


def _reject_unknown_keys(payload: Dict[str, Any], allowed: set, where: str = "payload") -> None:
    """Fail loudly on typos. A 400 beats a plausible-looking partial result."""
    unknown = set(payload) - allowed
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown {where} key(s): {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}."
            ),
        )


def _inference_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and collect the optional inference knobs. Bad values 400."""
    opts: Dict[str, Any] = {}

    if "threshold" in payload:
        value = payload["threshold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail="'threshold' must be a number in [0, 1].")
        if not 0.0 <= float(value) <= 1.0:
            raise HTTPException(status_code=400, detail="'threshold' must be in [0, 1].")
        opts["threshold"] = float(value)

    for flag in ("include_confidence", "include_spans"):
        if flag in payload:
            if not isinstance(payload[flag], bool):
                raise HTTPException(status_code=400, detail=f"'{flag}' must be a boolean.")
            opts[flag] = payload[flag]

    if "max_len" in payload and payload["max_len"] is not None:
        value = payload["max_len"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise HTTPException(status_code=400, detail="'max_len' must be a positive integer.")
        opts["max_len"] = value

    if "overlap_policy" in payload and payload["overlap_policy"] is not None:
        value = payload["overlap_policy"]
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail="'overlap_policy' must be a non-empty string.")
        opts["overlap_policy"] = value

    return opts


async def _run_inference(op_name: str, infer_fn: Callable[["GLiNER2"], Any]) -> Any:
    acquired = False
    loop = asyncio.get_running_loop()
    started = loop.time()
    queue_wait_s = 0.0
    model_get_s = 0.0
    infer_s = 0.0

    try:
        await asyncio.wait_for(
            _inference_semaphore.acquire(),
            timeout=INFERENCE_ACQUIRE_TIMEOUT_SECONDS,
        )
        acquired = True
        globals()["_inflight"] += 1
        acquired_at = loop.time()
        queue_wait_s = acquired_at - started
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Server is busy; no inference slot available within "
                f"{INFERENCE_ACQUIRE_TIMEOUT_SECONDS}s."
            ),
        ) from exc

    try:
        model_get_started = loop.time()
        model = await asyncio.to_thread(_get_model)
        model_get_s = loop.time() - model_get_started

        infer_started = loop.time()
        result = await asyncio.wait_for(
            asyncio.to_thread(infer_fn, model),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        infer_s = loop.time() - infer_started
        return result
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"{op_name} timed out after {REQUEST_TIMEOUT_SECONDS}s.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Error in %s", op_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if acquired:
            globals()["_inflight"] -= 1
            _inference_semaphore.release()
        total_s = loop.time() - started
        log.info(
            "Completed %s total_s=%.3f queue_wait_s=%.3f model_get_s=%.3f infer_s=%.3f",
            op_name,
            total_s,
            queue_wait_s,
            model_get_s,
            infer_s,
        )


@app.post("/extract_entities")
async def extract_entities(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "text": "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
                "labels": ["company", "person", "product", "location"]
            }
        }
    )
):
    """
    Extract entities from text using GLiNER2.
    
    Payload:
    - text: str
    - labels: List[str] OR Dict[str, str] (label -> description)
    """
    _reject_unknown_keys(payload, {"text", "labels"} | INFERENCE_OPTION_KEYS)
    text = _validate_text(payload.get("text"))
    labels = _validate_labels(payload.get("labels"))
    opts = _inference_options(payload)

    log.info(f"Extracting entities for text len={len(text)} opts={opts}")
    return await _run_inference(
        "extract_entities",
        lambda model: model.extract_entities(text, labels, **opts),
    )


@app.post("/classify_text")
async def classify_text(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "text": "This laptop has amazing performance but terrible battery life!",
                "labels": ["positive", "negative", "neutral"]
            }
        }
    )
):
    """
    Classify text using GLiNER2.
    
    Payload:
    - text: str
    - labels: List[str] OR Dict (complex config)
    """
    _reject_unknown_keys(payload, {"text", "labels"} | INFERENCE_OPTION_KEYS)
    text = _validate_text(payload.get("text"))
    labels = _validate_labels(payload.get("labels"))
    opts = _inference_options(payload)

    log.info(f"Classifying text len={len(text)} opts={opts}")
    return await _run_inference(
        "classify_text",
        lambda model: model.classify_text(text, labels, **opts),
    )


@app.post("/extract_structured")
async def extract_structured(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "text": "Goldman Sachs processed a $2.5M equity trade for Tesla Inc.",
                "schema": {
                    "transaction": [
                        "broker::str::Financial institution",
                        "amount::str::Transaction amount",
                        "security::str::Stock name"
                    ]
                }
            }
        }
    )
):
    """
    Extract structured data using GLiNER2.
    
    Payload:
    - text: str
    - schema: Dict defining the structure
    """
    _reject_unknown_keys(payload, {"text", "schema"} | INFERENCE_OPTION_KEYS)
    text = _validate_text(payload.get("text"))
    schema = _validate_schema(payload.get("schema"))
    opts = _inference_options(payload)

    log.info(f"Extracting structured data for text len={len(text)} opts={opts}")
    return await _run_inference(
        "extract_structured",
        lambda model: model.extract_json(text, schema, **opts),
    )


@app.post("/extract_multitask")
async def extract_multitask(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "text": "Contract between Company A and Company B.",
                "schema_config": {
                    "entities": ["company", "date"],
                    "classification": {"name": "contract_type", "labels": ["service", "nda"]},
                    "structure": {
                        "name": "terms",
                        "fields": [
                            {"name": "parties", "dtype": "list"},
                            {"name": "fee", "dtype": "str"}
                        ]
                    }
                }
            }
        }
    )
):
    """
    Multi-task extraction using GLiNER2 schema builder.
    
    Payload:
    - text: str
    - schema_config: Dict defining entities, classification, and structure.
      Example structure for schema_config:
      {
          "entities": ["list", "of", "labels"],
          "classification": { "name": "cls_name", "labels": ["l1", "l2"] },
          "structure": {
              "name": "struct_name",
              "fields": [
                  { "name": "f1", "dtype": "str", "choices": [...] },
                  ...
              ]
          }
      }
    """
    _reject_unknown_keys(payload, {"text", "schema_config"} | INFERENCE_OPTION_KEYS)
    text = _validate_text(payload.get("text"))
    config = _validate_schema_config(payload.get("schema_config"))
    opts = _inference_options(payload)

    log.info(f"Running multi-task extraction for text len={len(text)} opts={opts}")

    def _extract(model: Any) -> Any:
        return model.extract(text, _build_schema(model, config), **opts)

    return await _run_inference("extract_multitask", _extract)


def _build_schema(model: Any, config: Dict[str, Any]) -> Any:
    """Translate a validated schema_config into a gliner2 Schema."""
    schema = model.create_schema()

    if "entities" in config:
        schema.entities(config["entities"])

    if "classification" in config:
        cls_conf = config["classification"]
        schema.classification(cls_conf.get("name", "classification"), cls_conf.get("labels", []))

    if "relations" in config:
        schema.relations(config["relations"])

    if "structure" in config:
        struct_conf = config["structure"]
        struct = schema.structure(struct_conf.get("name", "structure"))
        for field in struct_conf.get("fields", []):
            # field(name, dtype="str", description=None, choices=None)
            struct.field(
                name=field.get("name"),
                dtype=field.get("dtype", "str"),
                description=field.get("description"),
                choices=field.get("choices"),
            )

    return schema


# --- GLiNER2.5 (boundary architecture) only ---

def _require_boundary(feature: str) -> None:
    """These capabilities exist only on GLiNER2.5 'boundary' checkpoints."""
    if _model_arch != "boundary":
        raise HTTPException(
            status_code=501,
            detail=(
                f"'{feature}' requires a GLiNER2.5 boundary checkpoint; "
                f"loaded model '{MODEL_ID}' has architecture '{_model_arch}'. "
                "Set MODEL_ID to e.g. fastino/gliner2.5-multi-v1."
            ),
        )


def _batch_size(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "batch_size" not in payload:
        return {}
    value = payload["batch_size"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HTTPException(status_code=400, detail="'batch_size' must be a positive integer.")
    return {"batch_size": min(value, MAX_BATCH_SIZE)}


def _validate_texts(texts: Any) -> list:
    if not isinstance(texts, list) or not texts:
        raise HTTPException(status_code=400, detail="Provide non-empty 'texts' list.")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"'texts' exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}.",
        )
    validated = [_validate_text(t) for t in texts]

    total_chars = sum(len(t) for t in validated)
    if total_chars > MAX_BATCH_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"batch totals {total_chars} chars, exceeding "
                f"MAX_BATCH_CHARS={MAX_BATCH_CHARS}. Split the batch."
            ),
        )
    return validated


@app.post("/extract_relations")
async def extract_relations(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "text": "Satya Nadella, CEO of Microsoft, met Sam Altman of OpenAI in Seattle.",
                "relations": ["works_for", "met_with", "located_in"]
            }
        }
    )
):
    """
    Extract typed relation triples. Requires a GLiNER2.5 boundary checkpoint.

    Payload:
    - text: str
    - relations: List[str] of relation type names
    """
    _get_model()
    _require_boundary("extract_relations")
    _reject_unknown_keys(payload, {"text", "relations"} | INFERENCE_OPTION_KEYS)
    text = _validate_text(payload.get("text"))
    relations = _validate_labels(payload.get("relations"))
    opts = _inference_options(payload)

    log.info(f"Extracting relations for text len={len(text)} opts={opts}")
    return await _run_inference(
        "extract_relations",
        lambda model: model.extract_relations(text, relations, **opts),
    )


@app.post("/extract_entities_batch")
async def extract_entities_batch(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "texts": [
                    "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
                    "Satya Nadella leads Microsoft from Redmond."
                ],
                "labels": ["company", "person", "location"]
            }
        }
    )
):
    """
    Batched entity extraction — markedly higher throughput than one call per
    document (measured ~4x on GLiNER2.5-multi). Requires a boundary checkpoint.

    Payload:
    - texts: List[str]
    - labels: List[str] OR Dict[str, str]
    """
    _get_model()
    _require_boundary("extract_entities_batch")
    _reject_unknown_keys(payload, {"texts", "labels", "batch_size"} | INFERENCE_OPTION_KEYS)
    texts = _validate_texts(payload.get("texts"))
    labels = _validate_labels(payload.get("labels"))
    opts = _inference_options(payload)
    opts.update(_batch_size(payload))

    log.info(f"Batch extracting entities for {len(texts)} texts opts={opts}")
    return await _run_inference(
        "extract_entities_batch",
        lambda model: {"results": model.batch_extract_entities(texts, labels, **opts)},
    )


@app.post("/classify_text_batch")
async def classify_text_batch(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "texts": ["The battery is terrible.", "Works flawlessly, love it."],
                "labels": {"sentiment": ["positive", "negative", "neutral"]}
            }
        }
    )
):
    """
    Batched text classification. Requires a boundary checkpoint.

    Payload: texts (list), labels, plus optional threshold / include_confidence /
    include_spans / max_len / overlap_policy / batch_size.
    """
    _get_model()
    _require_boundary("classify_text_batch")
    _reject_unknown_keys(payload, {"texts", "labels", "batch_size"} | INFERENCE_OPTION_KEYS)
    texts = _validate_texts(payload.get("texts"))
    labels = _validate_labels(payload.get("labels"))
    opts = _inference_options(payload)
    opts.update(_batch_size(payload))

    log.info(f"Batch classifying {len(texts)} texts opts={opts}")
    return await _run_inference(
        "classify_text_batch",
        lambda model: {"results": model.batch_classify_text(texts, labels, **opts)},
    )


@app.post("/extract_relations_batch")
async def extract_relations_batch(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "texts": [
                    "Satya Nadella, CEO of Microsoft, met Sam Altman of OpenAI.",
                    "Tim Cook runs Apple from Cupertino."
                ],
                "relations": ["works_for", "met_with"]
            }
        }
    )
):
    """
    Batched relation extraction. Requires a boundary checkpoint.

    Payload: texts (list), relations (list), plus the optional inference knobs.
    """
    _get_model()
    _require_boundary("extract_relations_batch")
    _reject_unknown_keys(payload, {"texts", "relations", "batch_size"} | INFERENCE_OPTION_KEYS)
    texts = _validate_texts(payload.get("texts"))
    relations = _validate_labels(payload.get("relations"))
    opts = _inference_options(payload)
    opts.update(_batch_size(payload))

    log.info(f"Batch extracting relations for {len(texts)} texts opts={opts}")
    return await _run_inference(
        "extract_relations_batch",
        lambda model: {"results": model.batch_extract_relations(texts, relations, **opts)},
    )


@app.post("/extract_multitask_batch")
async def extract_multitask_batch(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "texts": ["Apple CEO Tim Cook announced record revenue in Cupertino."],
                "schema_config": {
                    "entities": ["company", "person"],
                    "classification": {"name": "sentiment", "labels": ["positive", "negative"]}
                }
            }
        }
    )
):
    """
    Batched multi-task extraction: one schema applied across many documents.
    Requires a boundary checkpoint.

    Payload: texts (list), schema_config, plus the optional inference knobs.
    """
    _get_model()
    _require_boundary("extract_multitask_batch")
    _reject_unknown_keys(payload, {"texts", "schema_config", "batch_size"} | INFERENCE_OPTION_KEYS)
    texts = _validate_texts(payload.get("texts"))
    config = _validate_schema_config(payload.get("schema_config"))
    opts = _inference_options(payload)
    opts.update(_batch_size(payload))

    log.info(f"Batch multi-task extraction for {len(texts)} texts opts={opts}")

    def _extract(model: Any) -> Any:
        schema = _build_schema(model, config)
        return {"results": model.batch_extract(texts, schema, **opts)}

    return await _run_inference("extract_multitask_batch", _extract)
