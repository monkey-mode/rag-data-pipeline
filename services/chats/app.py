"""Chats service: full RAG queries composed with LCEL.

The query flow is one LCEL graph:

    {question, k}
        │
        ├─ docs:    RunnableLambda(retrieve)          ── ChromaDB similarity search
        └─ question: itemgetter("question")
        │
        .assign(answer =
            RunnablePassthrough.assign(context=format_docs)
            | ChatPromptTemplate
            | ChatAnthropic
            | StrOutputParser()
        )

Requires ANTHROPIC_API_KEY. Each exchange (question, answer, sources) is logged
in the service's own ``chats`` table.

Run:
    uvicorn services.chats.app:app --port 8002 --reload
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from operator import itemgetter

import chromadb
import psycopg
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from pydantic import BaseModel

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-8")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id          varchar(36) PRIMARY KEY,
    question    text NOT NULL,
    answer      text,
    sources     jsonb NOT NULL,
    created_at  timestamptz NOT NULL
);
ALTER TABLE chats ADD COLUMN IF NOT EXISTS answer text;
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the provided context from the user's "
            "documents. If the context does not contain the answer, say you don't "
            "know. Mention the source filename(s) you used.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain adapter for Chroma's default embedding function.

    The RAG worker stores vectors with Chroma's default EF (all-MiniLM-L6-v2,
    local ONNX). Queries must be embedded with the same model or distances are
    meaningless, so wrap that EF in the LangChain Embeddings interface instead
    of pulling in a second embedding model.
    """

    def __init__(self) -> None:
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


db: psycopg.Connection
rag_chain: Runnable
vectorstore: Chroma


def format_docs(docs_with_scores: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{doc.metadata['source']} chunk {doc.metadata['chunk_index']}]\n{doc.page_content}"
        for doc, _ in docs_with_scores
    )


def retrieve(inputs: dict) -> list[tuple[Document, float]]:
    return vectorstore.similarity_search_with_score(inputs["question"], k=inputs["k"])


def build_chain() -> Runnable:
    llm = ChatAnthropic(model=LLM_MODEL, max_tokens=1024)

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        question=itemgetter("question"),
        docs=RunnableLambda(retrieve),
    ).assign(answer=answer_chain)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, rag_chain, vectorstore
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required — export it before starting")

    db = psycopg.connect(DATABASE_URL, autocommit=True)
    db.execute(SCHEMA)

    chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vectorstore = Chroma(
        client=chroma,
        collection_name=COLLECTION_NAME,
        embedding_function=ChromaDefaultEmbeddings(),
    )
    rag_chain = build_chain()
    yield
    db.close()


app = FastAPI(title="chats-service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


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
    answer: str
    sources: list[Source]


class ChatLogEntry(BaseModel):
    id: str
    question: str
    answer: str | None
    sources: list[Source]
    created_at: datetime


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    result = rag_chain.invoke({"question": body.question, "k": body.n_results})

    sources = [
        Source(
            document_id=doc.metadata["document_id"],
            source=doc.metadata["source"],
            chunk_index=doc.metadata["chunk_index"],
            distance=score,
            text=doc.page_content,
        )
        for doc, score in result["docs"]
    ]

    chat_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO chats (id, question, answer, sources, created_at) VALUES (%s, %s, %s, %s, %s)",
        (
            chat_id,
            body.question,
            result["answer"],
            json.dumps([s.model_dump() for s in sources]),
            datetime.now(timezone.utc),
        ),
    )
    return QueryResponse(
        chat_id=chat_id, question=body.question, answer=result["answer"], sources=sources
    )


@app.get("/chats", response_model=list[ChatLogEntry])
def list_chats() -> list[ChatLogEntry]:
    rows = db.execute(
        "SELECT id, question, answer, sources, created_at FROM chats ORDER BY created_at DESC"
    ).fetchall()
    return [
        ChatLogEntry(id=r[0], question=r[1], answer=r[2], sources=r[3], created_at=r[4])
        for r in rows
    ]
