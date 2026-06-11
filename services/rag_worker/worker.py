"""RAG worker: consume MinIO bucket events from Redis, chunk + embed + store, stamp status.

MinIO is configured (docker-compose.yml) to RPUSH every put-event onto the Redis
list ``minio:events``. This worker BLPOPs that list as a queue: for each uploaded
object it downloads the file from MinIO, chunks it with LangChain, stores chunks
in ChromaDB (default local embedding function), and stamps the document row in
Postgres: processing -> ready (or failed).

Run:
    python -m services.rag_worker.worker
"""

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote_plus

import chromadb
import psycopg
import redis
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from minio import Minio

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EVENTS_KEY = os.getenv("MINIO_EVENTS_KEY", "minio:events")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}

minio_client = Minio(
    MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def set_status(db: psycopg.Connection, doc_id: str, status: str, **fields) -> None:
    sets = ", ".join(f"{name} = %s" for name in fields)
    sql = f"UPDATE documents SET status = %s, updated_at = now(){', ' + sets if sets else ''} WHERE id = %s"
    db.execute(sql, (status, *fields.values(), doc_id))


def process_object(db: psycopg.Connection, collection, bucket: str, key: str) -> None:
    # Object keys are "<document_id>/<filename>" (see documents service).
    doc_id, _, filename = key.partition("/")
    print(f"processing {key}")
    set_status(db, doc_id, "processing")
    try:
        suffix = Path(filename).suffix.lower()
        loader_cls = LOADERS.get(suffix)
        if loader_cls is None:
            raise ValueError(f"unsupported file type: {suffix!r}")

        with tempfile.TemporaryDirectory() as tmp:
            local_path = str(Path(tmp) / Path(filename).name)
            minio_client.fget_object(bucket, key, local_path)
            documents = loader_cls(local_path).load()

        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("no text could be extracted")

        # Re-ingest safely: drop any chunks from a previous run of this document.
        existing = collection.get(where={"document_id": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        ids, texts, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc_id}:{i}")
            texts.append(chunk.page_content)
            metadata = {"document_id": doc_id, "source": filename, "chunk_index": i}
            if "page" in chunk.metadata:
                metadata["page"] = chunk.metadata["page"]
            metadatas.append(metadata)
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        set_status(db, doc_id, "ready", chunk_count=len(chunks), error=None)
        print(f"ready {key}: {len(chunks)} chunks")
    except Exception as exc:  # noqa: BLE001 — stamp the failure, keep consuming
        set_status(db, doc_id, "failed", error=str(exc)[:2000])
        print(f"failed {key}: {exc}")


def event_records(raw: bytes) -> list[dict]:
    """Pull S3 event records out of a queue entry, whatever envelope MinIO used.

    Seen shapes: a bare list of records, {"Event": [...], "EventTime": ...}
    (Redis access format), and {"Records": [...]} (plain S3 notification).
    A record is any dict carrying both "eventName" and "s3".
    """

    def collect(node):
        if isinstance(node, list):
            for item in node:
                yield from collect(item)
        elif isinstance(node, dict):
            if "eventName" in node and "s3" in node:
                yield node
            else:
                for key in ("Event", "Records"):
                    if key in node:
                        yield from collect(node[key])

    return list(collect(json.loads(raw)))


def main() -> None:
    queue = redis.Redis.from_url(REDIS_URL, socket_timeout=10)
    db = psycopg.connect(DATABASE_URL, autocommit=True)
    chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    print(f"worker listening on redis list '{EVENTS_KEY}'", flush=True)
    while True:
        # Short blocking pop in a loop: the server answers within 5s (item or nil),
        # so the client's socket timeout never trips on an idle queue.
        item = queue.blpop(EVENTS_KEY, timeout=5)
        if item is None:
            continue
        _, raw = item
        try:
            records = event_records(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"skipping unparseable event: {exc}", flush=True)
            continue
        for record in records:
            if not record.get("eventName", "").startswith("s3:ObjectCreated"):
                continue
            s3 = record["s3"]
            bucket = s3["bucket"]["name"]
            key = unquote_plus(s3["object"]["key"])
            process_object(db, collection, bucket, key)


if __name__ == "__main__":
    main()
