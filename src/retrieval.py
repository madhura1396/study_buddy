"""Given a query string, return the top-k most relevant chunks from ChromaDB."""

from src.ingest import get_collection


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    hits = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )
    return hits
