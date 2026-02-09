.PHONY: help install run run-no-preload dev docker-build docker-run docker-run-bg docker-logs docker-stop docker-cudnn

PYTHON ?= ../venv/bin/python
HOST ?= 0.0.0.0
PORT ?= 8125
LOOP ?= asyncio
MODEL_ID ?= fastino/gliner2-large-v1
IMAGE_NAME ?= joshsgoldstein/gliner-api
IMAGE_TAG ?= $(IMAGE_NAME):latest
DOCKER_PORT ?= 8012
DOCKER_CONTEXT ?= ..

help:
	@echo "Targets:"
	@echo "  install        Install API requirements into the venv"
	@echo "  run            Start API with model preload"
	@echo "  run-no-preload Start API without model preload"
	@echo "  dev            Start API with auto-reload"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container (foreground)"
	@echo "  docker-run-bg  Run Docker container (background)"
	@echo "  docker-logs    Tail Docker logs"
	@echo "  docker-stop    Stop Docker container"
	@echo "  docker-cudnn   Copy host cuDNN libs into gliner-api/cudnn"

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	MODEL_ID=$(MODEL_ID) $(PYTHON) -m uvicorn app:app --host $(HOST) --port $(PORT) --loop $(LOOP)

run-no-preload:
	MODEL_ID=$(MODEL_ID) MODEL_PRELOAD=0 $(PYTHON) -m uvicorn app:app --host $(HOST) --port $(PORT) --loop $(LOOP)

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
		-v $(PWD)/models:/app/models \
		$(IMAGE_TAG)

docker-run-bg:
	docker run -d --rm \
		--name $(IMAGE_NAME) \
		--runtime nvidia \
		-p $(PORT):$(DOCKER_PORT) \
		-e MODEL_ID=$(MODEL_ID) \
		-e MODEL_PRELOAD=0 \
		-v $(PWD)/models:/app/models \
		$(IMAGE_TAG)

docker-logs:
	docker logs -f $(IMAGE_NAME)

docker-stop:
	docker stop $(IMAGE_NAME) || true

docker-cudnn:
	mkdir -p cudnn
	cp -a /usr/lib/aarch64-linux-gnu/libcudnn.so.9* cudnn/ || true
