"""Documents service: presigned-URL upload flow + document status tracking.

The client never sends file bytes through this API. It asks for an upload slot,
receives a presigned MinIO URL, and PUTs the file directly to MinIO. The RAG
worker picks the file up from the bucket notification and stamps the status here.

Run:
    uvicorn services.documents.app:app --port 8001 --reload
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
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
UPLOAD_URL_TTL = timedelta(hours=1)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="documents-service", lifespan=lifespan)


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
