"""Download a GLiNER2 model from HuggingFace Hub to a local directory."""

import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

model_id = os.getenv("MODEL_ID", "fastino/gliner2-large-v1")
local_model_path = os.path.join("./models", model_id)

os.makedirs(local_model_path, exist_ok=True)

if not os.listdir(local_model_path):
    print(f"Downloading {model_id} to {local_model_path} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_model_path,
        local_dir_use_symlinks=False,
    )
    print("Download complete.")
else:
    print(f"Model already exists at {local_model_path}")
