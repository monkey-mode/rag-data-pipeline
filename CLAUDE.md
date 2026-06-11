# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project overview, architecture diagram, layout, API reference, and commands live in the README — single source of truth:

@README.md

## Architectural Rules

- Keep the upload path and the processing path in separate services — the upload handler must never block on chunking or embedding.
- The file in MinIO is the source of truth; ChromaDB holds only derived data. Reprocessing must be possible from MinIO alone (the worker deletes a document's old chunks before re-upserting).
- Pass references (bucket/object key, document ID) through the queue, not file contents.
- Embedding consistency: the worker and the chats service must use the same embedding model (currently Chroma's default all-MiniLM-L6-v2; the chats service wraps it in `ChromaDefaultEmbeddings`). Changing it requires re-embedding everything (use the reprocess endpoint per document).
- Each service has its own `requirements.txt`; for local dev they all install into the root `venv/`.
- LLM calls go through `langchain-anthropic` (`ChatAnthropic`, model `claude-opus-4-8`) inside the LCEL chain in `services/chats/app.py`.

## Gotchas

- A moved venv is broken (scripts hard-code paths) — recreate instead of `mv`.
- MinIO's Redis `access` notification format wraps records inconsistently; `worker.event_records()` normalizes the shapes — don't assume `{"Records": [...]}`.
- ChromaDB returns metadata dicts with non-deterministic key order; sort keys when displaying.
- If the worker is down when an upload lands, the event waits in the Redis list (good). If the worker crashes mid-event, the event is lost and the row stays `uploading` — recover with `POST /documents/{id}/reprocess`, which pushes a synthetic event onto the same queue.
- The chats service fails fast at startup without `ANTHROPIC_API_KEY` (set it in `.env`; the Makefile auto-loads it).
