.DEFAULT_GOAL := help

VENV    := venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn

# Load ANTHROPIC_API_KEY (and any overrides) from .env when present.
-include .env
export

.PHONY: help setup run infra infra-down infra-logs documents chats worker ui ingest query

help: ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## create venv and install all service dependencies
	python3 -m venv $(VENV)
	$(PIP) install -r services/documents/requirements.txt \
	               -r services/rag_worker/requirements.txt \
	               -r services/chats/requirements.txt

run: infra ## start infra + all services + UI in one terminal (Ctrl-C stops everything)
	@trap 'kill 0' INT TERM; \
	$(UVICORN) services.documents.app:app --port 8001 --reload & \
	$(UVICORN) services.chats.app:app --port 8002 --reload & \
	$(PY) -m services.rag_worker.worker & \
	$(PY) -m http.server 3000 -d ui & \
	echo "documents :8001 | chats :8002 | worker | ui http://localhost:3000"; \
	wait

infra: ## start MinIO, Redis, Postgres, ChromaDB (docker compose)
	docker compose up -d

infra-down: ## stop infrastructure containers
	docker compose down

infra-logs: ## tail infrastructure logs
	docker compose logs -f

documents: ## run documents service on :8001
	$(UVICORN) services.documents.app:app --port 8001 --reload

chats: ## run chats service on :8002 (needs ANTHROPIC_API_KEY, e.g. via .env)
	$(UVICORN) services.chats.app:app --port 8002 --reload

worker: ## run the RAG worker (consumes minio:events queue)
	$(PY) -m services.rag_worker.worker

ui: ## serve the chat UI on http://localhost:3000
	$(PY) -m http.server 3000 -d ui

ingest: ## simple-rag: ingest sample data into its local chroma_db
	$(PY) simple-rag/ingest.py

query: ## simple-rag: similarity search, usage: make query Q="your question"
	$(PY) simple-rag/ingest.py --query "$(Q)"
