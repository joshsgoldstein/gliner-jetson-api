.PHONY: help venv install install-full run run-no-preload dev docker-build docker-build-no-cache docker-run docker-run-bg docker-logs docker-stop docker-cudnn

VENV_DIR ?= .venv
PYTHON ?= $(VENV_DIR)/bin/python
HOST ?= 0.0.0.0
PORT ?= 8125
LOOP ?= asyncio
WORKERS ?= 1
MODEL_ID ?= fastino/gliner2-large-v1
IMAGE_NAME ?= joshsgoldstein/gliner-api
IMAGE_TAG ?= $(IMAGE_NAME):latest
DOCKER_PORT ?= 8012
DOCKER_CONTEXT ?= ..
DOCKER_WORKERS ?= 1
MAX_CONCURRENT_INFERENCES ?= 1
TORCH_WHL_URL ?= https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl
WHEEL_CACHE_DIR ?= .cache/jetson-wheels

help:
	@echo "Targets:"
	@echo "  venv           Create venv and upgrade pip tooling"
	@echo "  install        Install API requirements into the venv"
	@echo "  install-full   Jetson local install via scripts/setup_venv.sh (with wheel cache)"
	@echo "  run            Start API with model preload"
	@echo "  run-no-preload Start API without model preload"
	@echo "                 (set WORKERS=N for multi-process workers)"
	@echo "  dev            Start API with auto-reload"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-build-no-cache Build Docker image without cache"
	@echo "  docker-run     Run Docker container (foreground)"
	@echo "  docker-run-bg  Run Docker container (background)"
	@echo "                 (set DOCKER_WORKERS=N and/or MAX_CONCURRENT_INFERENCES=N)"
	@echo "  docker-logs    Tail Docker logs"
	@echo "  docker-stop    Stop Docker container"
	@echo "  docker-cudnn   Copy host cuDNN libs into gliner-api/cudnn"

venv:
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

install:
	$(PYTHON) -m pip install -r requirements.txt

install-full:
	VENV_DIR="$(VENV_DIR)" TORCH_WHL_URL="$(TORCH_WHL_URL)" WHEEL_CACHE_DIR="$(WHEEL_CACHE_DIR)" \
		bash scripts/setup_venv.sh

run:
	MODEL_ID=$(MODEL_ID) $(PYTHON) -m uvicorn app:app --host $(HOST) --port $(PORT) --loop $(LOOP) --workers $(WORKERS)

run-no-preload:
	MODEL_ID=$(MODEL_ID) MODEL_PRELOAD=0 $(PYTHON) -m uvicorn app:app --host $(HOST) --port $(PORT) --loop $(LOOP) --workers $(WORKERS)

dev:
	MODEL_ID=$(MODEL_ID) MODEL_PRELOAD=0 $(PYTHON) -m uvicorn app:app --host $(HOST) --port $(PORT) --loop $(LOOP) --reload

docker-build:
	docker build -t $(IMAGE_TAG) -f Dockerfile $(DOCKER_CONTEXT)

docker-build-no-cache:
	docker build --no-cache -t $(IMAGE_TAG) -f Dockerfile $(DOCKER_CONTEXT)

docker-run:
	docker run --rm -it \
		--runtime nvidia \
		-p $(PORT):$(DOCKER_PORT) \
		-e MODEL_ID=$(MODEL_ID) \
		-e MODEL_PRELOAD=0 \
		-e MAX_CONCURRENT_INFERENCES=$(MAX_CONCURRENT_INFERENCES) \
		-v $(PWD)/models:/app/models \
		$(IMAGE_TAG) \
		python3 -m uvicorn app:app --host 0.0.0.0 --port $(DOCKER_PORT) --loop $(LOOP) --workers $(DOCKER_WORKERS)

docker-run-bg:
	docker run -d --rm \
		--name $(IMAGE_NAME) \
		--runtime nvidia \
		-p $(PORT):$(DOCKER_PORT) \
		-e MODEL_ID=$(MODEL_ID) \
		-e MODEL_PRELOAD=0 \
		-e MAX_CONCURRENT_INFERENCES=$(MAX_CONCURRENT_INFERENCES) \
		-v $(PWD)/models:/app/models \
		$(IMAGE_TAG) \
		python3 -m uvicorn app:app --host 0.0.0.0 --port $(DOCKER_PORT) --loop $(LOOP) --workers $(DOCKER_WORKERS)

docker-logs:
	docker logs -f $(IMAGE_NAME)

docker-stop:
	docker stop $(IMAGE_NAME) || true

docker-cudnn:
	mkdir -p cudnn
	cp -a /usr/lib/aarch64-linux-gnu/libcudnn.so.9* cudnn/ || true
