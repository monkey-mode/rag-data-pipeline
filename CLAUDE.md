# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Early-stage. `simple-rag/` contains a minimal working ingestion pipeline (read file → LangChain chunking → ChromaDB with its default local embedding function — no API keys needed). The microservice architecture below is the target design; build toward it and update this file as tooling lands.

## What This Project Is

A fullstack hands-on RAG (Retrieval-Augmented Generation) data pipeline built as Python microservices. The core design principle is **separation of responsibilities** between accepting uploads and processing them:

1. **Upload service** — an async Python API service that handles file uploads. It stores the raw uploaded file in **MinIO** (S3-compatible object storage) and enqueues a processing task. It does NOT do chunking or embedding itself; it returns quickly after the upload is persisted.
2. **Ingestion worker** — consumes the queued task asynchronously, fetches the file from MinIO, chunks it using **LangChain** document loaders/text splitters, generates embeddings, and stores the chunks + embeddings in **ChromaDB** (the vector database).

## Tech Stack

- **Python** microservices (async — use `async def` endpoints and async clients where available)
- **LangChain** for document loading, chunking/splitting, and the RAG retrieval chain
- **ChromaDB** as the vector store
- **MinIO** for raw file/object storage
- Async task execution decouples upload from processing (task queue/worker pattern)

## Architectural Rules

- Keep the upload path and the processing path in separate services/modules — the upload handler must never block on chunking or embedding.
- The file in MinIO is the source of truth for raw uploads; ChromaDB holds only derived data (chunks, embeddings, metadata). Reprocessing should be possible from MinIO alone.
- Pass references (bucket/object key, document ID) through the task queue, not file contents.
- Infrastructure dependencies (MinIO, ChromaDB, the queue broker) should run via Docker Compose for local development.

## Commands

### simple-rag (minimal ingestion pipeline)

```bash
cd simple-rag
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # setup
.venv/bin/python ingest.py                       # ingest data/*.txt|*.md into ./chroma_db
.venv/bin/python ingest.py --query "question"    # test similarity search
```

The full microservice stack (Docker Compose for MinIO/ChromaDB, upload service, worker) has no tooling yet — record those commands here when added.
