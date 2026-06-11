"""Chats service: retrieval-only RAG queries against ChromaDB.

Embeds the question (same default embedding function the worker used), returns
the most similar chunks, and logs each exchange in its own ``chats`` table.
LLM answer generation can be layered on top later without changing this flow.

Run:
    uvicorn services.chats.app:app --port 8002 --reload
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import chromadb
import psycopg
from fastapi import FastAPI
from pydantic import BaseModel

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          varchar(36) PRIMARY KEY,
    question    text NOT NULL,
    sources     jsonb NOT NULL,
    created_at  timestamptz NOT NULL
)
"""

db: psycopg.Connection
collection: chromadb.Collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, collection
    db = psycopg.connect(DATABASE_URL, autocommit=True)
    db.execute(SCHEMA)
    chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    yield
    db.close()


app = FastAPI(title="chats-service", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    n_results: int = 4


class Source(BaseModel):
    document_id: str
    source: str
    chunk_index: int
    distance: float
    text: str


class QueryResponse(BaseModel):
    chat_id: str
    question: str
    sources: list[Source]


class ChatLogEntry(BaseModel):
    id: str
    question: str
    sources: list[Source]
    created_at: datetime


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    results = collection.query(query_texts=[body.question], n_results=body.n_results)

    sources = [
        Source(
            document_id=meta["document_id"],
            source=meta["source"],
            chunk_index=meta["chunk_index"],
            distance=dist,
            text=doc,
        )
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    chat_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO chats (id, question, sources, created_at) VALUES (%s, %s, %s, %s)",
        (
            chat_id,
            body.question,
            json.dumps([s.model_dump() for s in sources]),
            datetime.now(timezone.utc),
        ),
    )
    return QueryResponse(chat_id=chat_id, question=body.question, sources=sources)


@app.get("/chats", response_model=list[ChatLogEntry])
def list_chats() -> list[ChatLogEntry]:
    rows = db.execute(
        "SELECT id, question, sources, created_at FROM chats ORDER BY created_at DESC"
    ).fetchall()
    return [
        ChatLogEntry(id=r[0], question=r[1], sources=r[2], created_at=r[3]) for r in rows
    ]
