"""Documents service: presigned-URL upload flow + document status tracking.

The client never sends file bytes through this API. It asks for an upload slot,
receives a presigned MinIO URL, and PUTs the file directly to MinIO. The RAG
worker picks the file up from the bucket notification and stamps the status here.

Run:
    uvicorn services.documents.app:app --port 8001 --reload
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import chromadb
import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from minio import Minio
from pydantic import BaseModel
from sqlalchemy import DateTime, String, Integer, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "documents")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EVENTS_KEY = os.getenv("MINIO_EVENTS_KEY", "minio:events")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
UPLOAD_URL_TTL = timedelta(hours=1)
DOWNLOAD_URL_TTL = timedelta(minutes=15)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    object_key: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(32), default="uploading")
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


engine = create_async_engine(DATABASE_URL)
new_session = async_sessionmaker(engine, expire_on_commit=False)

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


redis_queue: aioredis.Redis
chroma_collection: chromadb.Collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_queue, chroma_collection
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    redis_queue = aioredis.from_url(REDIS_URL)
    chroma = await asyncio.to_thread(chromadb.HttpClient, host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_collection = await asyncio.to_thread(chroma.get_or_create_collection, COLLECTION_NAME)
    yield
    await redis_queue.aclose()
    await engine.dispose()


app = FastAPI(title="documents-service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class CreateDocumentRequest(BaseModel):
    filename: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    object_key: str
    status: str
    error: str | None
    chunk_count: int | None
    created_at: datetime
    updated_at: datetime


class CreateDocumentResponse(DocumentResponse):
    upload_url: str


def to_response(row: DocumentRow) -> DocumentResponse:
    return DocumentResponse(
        id=row.id,
        filename=row.filename,
        object_key=row.object_key,
        status=row.status,
        error=row.error,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@app.post("/documents", response_model=CreateDocumentResponse)
async def create_document(body: CreateDocumentRequest) -> CreateDocumentResponse:
    doc_id = str(uuid.uuid4())
    object_key = f"{doc_id}/{body.filename}"
    now = datetime.now(timezone.utc)

    row = DocumentRow(
        id=doc_id,
        filename=body.filename,
        object_key=object_key,
        status="uploading",
        created_at=now,
        updated_at=now,
    )
    async with new_session() as session:
        session.add(row)
        await session.commit()

    upload_url = minio_client.presigned_put_object(BUCKET, object_key, expires=UPLOAD_URL_TTL)
    return CreateDocumentResponse(upload_url=upload_url, **to_response(row).model_dump())


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents() -> list[DocumentResponse]:
    async with new_session() as session:
        rows = (
            await session.execute(
                select(DocumentRow).order_by(DocumentRow.created_at.desc())
            )
        ).scalars()
        return [to_response(r) for r in rows]


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str) -> DocumentResponse:
    async with new_session() as session:
        row = await session.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")
        return to_response(row)


class DownloadResponse(BaseModel):
    download_url: str


@app.get("/documents/{doc_id}/download", response_model=DownloadResponse)
async def download_document(doc_id: str) -> DownloadResponse:
    async with new_session() as session:
        row = await session.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")

    url = await asyncio.to_thread(
        minio_client.presigned_get_object, BUCKET, row.object_key, expires=DOWNLOAD_URL_TTL
    )
    return DownloadResponse(download_url=url)


class ChunkDetail(BaseModel):
    id: str
    chunk_index: int
    text: str
    char_count: int
    metadata: dict


class ChunksResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    chunks: list[ChunkDetail]


@app.get("/documents/{doc_id}/chunks", response_model=ChunksResponse)
async def get_document_chunks(doc_id: str) -> ChunksResponse:
    """Inspect the derived chunks ChromaDB holds for a document.

    Read-only view for the backoffice: returns each chunk's text and metadata,
    ordered by chunk_index. The file in MinIO remains the source of truth; this
    is purely the embedded/queryable projection of it.
    """
    async with new_session() as session:
        row = await session.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")

    result = await asyncio.to_thread(
        chroma_collection.get,
        where={"document_id": doc_id},
        include=["documents", "metadatas"],
    )

    chunks: list[ChunkDetail] = []
    for chunk_id, text, metadata in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        metadata = metadata or {}
        text = text or ""
        chunks.append(
            ChunkDetail(
                id=chunk_id,
                chunk_index=metadata.get("chunk_index", 0),
                text=text,
                char_count=len(text),
                # Chroma returns keys in non-deterministic order — sort for stable display.
                metadata=dict(sorted(metadata.items())),
            )
        )
    chunks.sort(key=lambda c: c.chunk_index)

    return ChunksResponse(
        document_id=doc_id,
        filename=row.filename,
        status=row.status,
        chunk_count=len(chunks),
        chunks=chunks,
    )


class ScatterPoint(BaseModel):
    x: float
    y: float
    z: float
    document_id: str
    source: str
    chunk_index: int
    text: str


class ScatterResponse(BaseModel):
    count: int
    method: str
    # fraction of total variance captured by each of the 3 plotted axes
    explained_variance: list[float]
    points: list[ScatterPoint]


def _pca_3d(vectors: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project N×D embeddings onto their top 3 principal components.

    Returns the N×3 coordinates and the fraction of total variance each of the
    three axes captures. Pads to 3 columns when there are fewer than 3 usable
    components (e.g. only one or two chunks exist).
    """
    centered = vectors - vectors.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)

    k = min(3, components.shape[0])
    coords = centered @ components[:k].T
    if k < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - k)))

    variance = singular_values**2
    total = float(variance.sum())
    explained = (variance[:3] / total).tolist() if total > 0 else []
    explained += [0.0] * (3 - len(explained))
    return coords, explained


@app.get("/embeddings/scatter", response_model=ScatterResponse)
async def embeddings_scatter() -> ScatterResponse:
    """3D projection of every stored chunk embedding, for the backoffice map.

    Pulls all vectors from ChromaDB and reduces them (384-d → 3-d) with PCA so
    the UI can render an interactive scatter. Read-only and derived: nothing is
    written back. Chunks are colored by document client-side via document_id.
    """
    result = await asyncio.to_thread(
        chroma_collection.get, include=["embeddings", "metadatas", "documents"]
    )

    embeddings = np.asarray(result["embeddings"], dtype=float)
    if embeddings.size == 0:
        return ScatterResponse(
            count=0, method="pca", explained_variance=[0.0, 0.0, 0.0], points=[]
        )

    coords, explained = _pca_3d(embeddings)

    points: list[ScatterPoint] = []
    for (x, y, z), metadata, text in zip(
        coords, result["metadatas"], result["documents"]
    ):
        metadata = metadata or {}
        preview = " ".join((text or "").split())[:160]
        points.append(
            ScatterPoint(
                x=float(x),
                y=float(y),
                z=float(z),
                document_id=metadata.get("document_id", ""),
                source=metadata.get("source", "?"),
                chunk_index=metadata.get("chunk_index", 0),
                text=preview,
            )
        )

    return ScatterResponse(
        count=len(points), method="pca", explained_variance=explained, points=points
    )


@app.post("/documents/{doc_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(doc_id: str) -> DocumentResponse:
    """Re-enqueue an already-uploaded document for chunking/embedding.

    Pushes a synthetic ObjectCreated event onto the same Redis list MinIO
    publishes to, so the worker handles it identically to a fresh upload.
    Useful after changing chunking settings, or when an event was lost.
    """
    async with new_session() as session:
        row = await session.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")
        row.status = "queued"
        row.error = None
        row.updated_at = datetime.now(timezone.utc)
        await session.commit()

    event = {
        "Records": [
            {
                "eventName": "s3:ObjectCreated:Reprocess",
                # worker unquotes keys, so quote like MinIO does
                "s3": {"bucket": {"name": BUCKET}, "object": {"key": quote_plus(row.object_key, safe="/")}},
            }
        ]
    }
    await redis_queue.rpush(EVENTS_KEY, json.dumps(event))
    return to_response(row)


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """Delete a document everywhere: ChromaDB chunks, MinIO object, Postgres row."""
    async with new_session() as session:
        row = await session.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")

        chunks = await asyncio.to_thread(chroma_collection.get, where={"document_id": doc_id})
        if chunks["ids"]:
            await asyncio.to_thread(chroma_collection.delete, ids=chunks["ids"])

        await asyncio.to_thread(minio_client.remove_object, BUCKET, row.object_key)

        await session.delete(row)
        await session.commit()

    return {"deleted": doc_id, "chunks_removed": len(chunks["ids"])}
