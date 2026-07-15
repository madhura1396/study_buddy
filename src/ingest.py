"""Read files from DATA_DIR, chunk them, embed them, and store in ChromaDB."""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Return a list of {"text": ..., "source": ...} for every supported file."""
    readers = {".txt": read_txt, ".pdf": read_pdf}
    documents = []
    for path in sorted(data_dir.iterdir()):
        reader = readers.get(path.suffix.lower())
        if reader is None:
            continue
        text = reader(path)
        if text.strip():
            documents.append({"text": text, "source": path.name})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split each document's text into overlapping chunks with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{doc['source']}::chunk{i}",
                    "text": piece,
                    "source": doc["source"],
                    "chunk_index": i,
                }
            )
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def run_ingestion() -> int:
    """Load, chunk, embed, and upsert all documents in DATA_DIR. Returns chunk count."""
    documents = load_documents()
    if not documents:
        print(f"No .txt or .pdf files found in {DATA_DIR}")
        return 0

    chunks = chunk_documents(documents)
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
    )

    print(f"Ingested {len(documents)} document(s) -> {len(chunks)} chunk(s).")
    return len(chunks)


if __name__ == "__main__":
    run_ingestion()
