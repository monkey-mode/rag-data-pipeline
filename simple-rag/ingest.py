"""Simple RAG ingestion: read files from data/, chunk with LangChain, store in ChromaDB.

Usage:
    python ingest.py                      # ingest all .txt/.md files in data/
    python ingest.py --query "question"   # test retrieval against the stored chunks
"""

import argparse
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def read_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Read all .txt and .md files from the data directory."""
    files = sorted(p for p in data_dir.iterdir() if p.suffix in {".txt", ".md"})
    return [(p.name, p.read_text(encoding="utf-8")) for p in files]


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def get_collection() -> chromadb.Collection:
    # Uses Chroma's default embedding function (all-MiniLM-L6-v2, runs locally).
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def ingest() -> None:
    collection = get_collection()
    documents = read_documents(DATA_DIR)
    if not documents:
        print(f"No .txt/.md files found in {DATA_DIR}")
        return

    for filename, text in documents:
        chunks = chunk_text(text)
        collection.upsert(
            ids=[f"{filename}:{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
        )
        print(f"Ingested {filename}: {len(chunks)} chunks")

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


def query(question: str, n_results: int = 3) -> None:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)
    print(f"Top {n_results} chunks for: {question!r}\n")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"[{meta['source']} chunk {meta['chunk_index']}] (distance {dist:.4f})")
        print(doc)
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="run a similarity search instead of ingesting")
    args = parser.parse_args()

    if args.query:
        query(args.query)
    else:
        ingest()
