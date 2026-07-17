"""Read files from DATA_DIR (including subfolders), chunk them, embed them,
and store in ChromaDB.

Files may live directly in DATA_DIR or in a subfolder, e.g.
DATA_DIR/Linear Regression/notes.docx. A file's subfolder name becomes its
"category" metadata; files directly in DATA_DIR are grouped under
"Uncategorized". A file's modification time becomes "uploaded_at" metadata,
letting the UI show recently-added files first.
"""

import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
)

MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")
DOCX_HEADING_LEVEL_RE = re.compile(r"Heading (\d+)")
UNCATEGORIZED = "Uncategorized"


def parse_txt_sections(path: Path) -> list[dict]:
    """Split plain text into sections using Markdown headers ("# Heading") as
    section markers. A nested header (##, ###) keeps its parent header in the
    heading path (joined by " > "), since a subsection's body text often
    relies on context named only in its parent heading, not repeated in the
    subsection itself. Text with no headers becomes a single section."""
    sections = [{"heading": None, "text": ""}]
    stack: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = MD_HEADER_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            stack[level - 1 :] = [match.group(2).strip()]
            sections.append({"heading": " > ".join(stack), "text": ""})
        else:
            sections[-1]["text"] += line + "\n"
    return sections


def parse_docx_sections(path: Path) -> list[dict]:
    """Use Word's own paragraph styles (Heading 1/2/3, ...) as section markers.
    A nested heading (e.g. a Heading 3 under a Heading 2) keeps its parent
    heading in the heading path, for the same reason as parse_txt_sections."""
    document = Document(str(path))
    sections = [{"heading": None, "text": ""}]
    stack: list[str] = []
    for paragraph in document.paragraphs:
        style_name = paragraph.style.name if paragraph.style else ""
        level_match = DOCX_HEADING_LEVEL_RE.match(style_name)
        if level_match:
            level = int(level_match.group(1))
            stack[level - 1 :] = [paragraph.text.strip()]
            sections.append({"heading": " > ".join(stack), "text": ""})
        else:
            sections[-1]["text"] += paragraph.text + "\n"
    return sections


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Return a list of {"source", "category", "uploaded_at", "rel_path",
    "sections": [{"heading", "text"}, ...]} for every supported file found
    anywhere under data_dir, including subfolders."""
    parsers = {".txt": parse_txt_sections, ".docx": parse_docx_sections}
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        parser = parsers.get(path.suffix.lower())
        if parser is None:
            continue
        sections = [s for s in parser(path) if s["text"].strip()]
        if not sections:
            continue
        rel_path = path.relative_to(data_dir)
        category = rel_path.parent.as_posix() if rel_path.parent != Path(".") else UNCATEGORIZED
        documents.append(
            {
                "source": path.name,
                "category": category,
                "uploaded_at": path.stat().st_mtime,
                "rel_path": rel_path.as_posix(),
                "sections": sections,
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split each document's sections into chunks, sub-splitting only sections
    that exceed CHUNK_SIZE. The heading is prefixed onto the embedded/stored
    text (so its keywords help retrieval match) and also kept as its own
    metadata field (so callers can reliably read it back without parsing)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for doc in documents:
        chunk_index = 0
        for section in doc["sections"]:
            text = section["text"].strip()
            heading = section["heading"] or ""
            pieces = [text] if len(text) <= CHUNK_SIZE else splitter.split_text(text)
            for piece in pieces:
                stored_text = f"Section: {heading}\n{piece}" if heading else piece
                chunks.append(
                    {
                        "id": f"{doc['rel_path']}::chunk{chunk_index}",
                        "text": stored_text,
                        "source": doc["source"],
                        "category": doc["category"],
                        "uploaded_at": doc["uploaded_at"],
                        "chunk_index": chunk_index,
                        "heading": heading,
                    }
                )
                chunk_index += 1
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
        print(f"No .txt or .docx files found in {DATA_DIR}")
        return 0

    chunks = chunk_documents(documents)
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "heading": c["heading"],
                "category": c["category"],
                "uploaded_at": c["uploaded_at"],
            }
            for c in chunks
        ],
    )

    print(f"Ingested {len(documents)} document(s) -> {len(chunks)} chunk(s).")
    return len(chunks)


if __name__ == "__main__":
    run_ingestion()
