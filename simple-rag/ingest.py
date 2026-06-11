"""Simple RAG ingestion: load files from data/ with LangChain, chunk, store in ChromaDB.

Usage:
    python ingest.py                      # ingest all supported files in data/
    python ingest.py --query "question"   # test retrieval against the stored chunks
"""

import argparse
from pathlib import Path

import chromadb
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


def load_documents(data_dir: Path) -> list[Document]:
    """Load all supported files as LangChain Documents (one or more per file)."""
    documents = []
    for path in sorted(data_dir.iterdir()):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        documents.extend(loader_cls(str(path)).load())
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split Documents into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def get_collection() -> chromadb.Collection:
    # Uses Chroma's default embedding function (all-MiniLM-L6-v2, runs locally).
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def ingest() -> None:
    documents = load_documents(DATA_DIR)
    if not documents:
        print(f"No supported files ({', '.join(sorted(LOADERS))}) found in {DATA_DIR}")
        return

    chunks = split_documents(documents)

    ids, texts, metadatas = [], [], []
    chunks_per_file: dict[str, int] = {}
    for chunk in chunks:
        source = Path(chunk.metadata["source"]).name
        index = chunks_per_file.get(source, 0)
        chunks_per_file[source] = index + 1

        ids.append(f"{source}:{index}")
        texts.append(chunk.page_content)
        metadata = {"source": source, "chunk_index": index}
        if "page" in chunk.metadata:  # PyPDFLoader yields one Document per page
            metadata["page"] = chunk.metadata["page"]
        metadatas.append(metadata)
        print(f"Prepared chunk {ids[-1]} (source: {source}, length: {len(texts[-1])} chars)")
        print(f" Metadata: {metadata}")
        print("-" * 60)

    collection = get_collection()
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    for source, count in chunks_per_file.items():
        print(f"Ingested {source}: {count} chunks")
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
