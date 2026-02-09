import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
MODEL_ID = os.getenv("MODEL_ID", "fastino/gliner2-large-v1")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_ID)
MODEL_PRELOAD = os.getenv("MODEL_PRELOAD", "1").lower() not in {"0", "false", "no"}

app = FastAPI(title="GLiNER2 API for JetPack 6")
_model: "GLiNER2 | None" = None


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


def _get_model() -> "GLiNER2":
    global _model
    if _model is None:
        from gliner2 import GLiNER2
        _ensure_model_downloaded()
        log.info("Loading GLiNER2 model from disk...")
        _model = GLiNER2.from_pretrained(LOCAL_MODEL_PATH)
        if DEVICE == "cuda":
            _model = _model.to(DEVICE)
            log.info("GLiNER2 loaded on GPU (%s).", torch.cuda.get_device_name(0))
        else:
            log.info("GLiNER2 loaded on CPU.")
        _model.eval()
    return _model


@app.on_event("startup")
def startup_event() -> None:
    if MODEL_PRELOAD:
        _get_model()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_path": LOCAL_MODEL_PATH,
        "loaded": _model is not None,
    }


@app.get("/version")
async def version() -> Dict[str, Any]:
    import gliner2

    return {
        "gliner2": getattr(gliner2, "__version__", "unknown"),
    }

# --- Endpoints ---

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
    text = payload.get("text")
    labels = payload.get("labels")
    
    if not text:
        raise HTTPException(status_code=400, detail="Provide 'text'.")
    if not labels:
        raise HTTPException(status_code=400, detail="Provide 'labels' (list or dict).")

    log.info(f"Extracting entities for text len={len(text)}")
    
    try:
        model = _get_model()
        result = await asyncio.to_thread(model.extract_entities, text, labels)
        return result
    except Exception as e:
        log.error(f"Error in extract_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    text = payload.get("text")
    labels = payload.get("labels")
    
    if not text:
        raise HTTPException(status_code=400, detail="Provide 'text'.")
    if not labels:
        raise HTTPException(status_code=400, detail="Provide 'labels'.")
        
    log.info(f"Classifying text len={len(text)}")
    
    try:
        model = _get_model()
        result = await asyncio.to_thread(model.classify_text, text, labels)
        return result
    except Exception as e:
        log.error(f"Error in classify_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    text = payload.get("text")
    schema = payload.get("schema")
    
    if not text:
        raise HTTPException(status_code=400, detail="Provide 'text'.")
    if not schema:
        raise HTTPException(status_code=400, detail="Provide 'schema'.")
        
    log.info(f"Extracting structured data for text len={len(text)}")
    
    try:
        model = _get_model()
        result = await asyncio.to_thread(model.extract_json, text, schema)
        return result
    except Exception as e:
        log.error(f"Error in extract_structured: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    text = payload.get("text")
    config = payload.get("schema_config")
    
    if not text:
        raise HTTPException(status_code=400, detail="Provide 'text'.")
    if not config:
        raise HTTPException(status_code=400, detail="Provide 'schema_config'.")
        
    log.info(f"Running multi-task extraction for text len={len(text)}")
    
    try:
        model = _get_model()
        # Build schema using the helper method from the model
        schema = model.create_schema()
        
        if "entities" in config:
            schema.entities(config["entities"])
            
        if "classification" in config:
            cls_conf = config["classification"]
            schema.classification(cls_conf.get("name", "classification"), cls_conf.get("labels", []))
            
        if "structure" in config:
            struct_conf = config["structure"]
            s = schema.structure(struct_conf.get("name", "structure"))
            for field in struct_conf.get("fields", []):
                # Apply field definition
                # field(self, name, dtype="str", description=None, choices=None)
                s.field(
                    name=field.get("name"),
                    dtype=field.get("dtype", "str"),
                    description=field.get("description"),
                    choices=field.get("choices")
                )
                
        results = await asyncio.to_thread(model.extract, text, schema)
        return results

    except Exception as e:
        log.error(f"Error in extract_multitask: {e}")
        raise HTTPException(status_code=500, detail=str(e))
