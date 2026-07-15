"""Manual retrieval quality check: enter a query, see the top-k chunks returned.

Run from the project root:
    python -m scripts.test_retrieval
"""

from src.retrieval import retrieve


def main():
    print("Type a query to test retrieval (empty line to quit).\n")
    while True:
        query = input("query> ").strip()
        if not query:
            break

        hits = retrieve(query, top_k=5)
        if not hits:
            print("  (no results — did you run `python -m src.ingest` first?)\n")
            continue

        for rank, hit in enumerate(hits, start=1):
            print(f"\n[{rank}] {hit['source']} (chunk {hit['chunk_index']}, distance={hit['distance']:.4f})")
            print(f"    {hit['text']}")
        print()


if __name__ == "__main__":
    main()
