"""Measure retrieval/chunking quality with a small hand-labeled eval set.

For each question, checks whether any retrieved chunk in the top-k both comes
from the expected source file and contains the expected phrase — i.e.
recall@k. Matching on phrase content (rather than chunk_index) keeps cases
valid even as CHUNK_SIZE/CHUNK_OVERLAP change and chunk boundaries shift.

Edit scripts/eval_cases.json to add or update cases as you add documents.

Run from the project root:
    python -m scripts.eval_retrieval
"""

import json
from pathlib import Path

from src.retrieval import retrieve

EVAL_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"


def load_cases() -> list[dict]:
    with open(EVAL_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_eval(top_k: int = 8) -> None:
    cases = load_cases()
    if not cases:
        print(f"No eval cases found in {EVAL_CASES_PATH}")
        return

    hits_count = 0
    for case in cases:
        question = case["question"]
        expected = case["expected"]

        results = retrieve(question, top_k=top_k)

        found = any(
            e["source"] == r["source"] and e["contains"].lower() in r["text"].lower()
            for e in expected
            for r in results
        )
        hits_count += found

        status = "PASS" if found else "FAIL"
        print(f"[{status}] {question}")
        if not found:
            expected_desc = [(e["source"], e["contains"]) for e in expected]
            retrieved_desc = [(r["source"], r["chunk_index"]) for r in results]
            print(f"    expected one of: {expected_desc}")
            print(f"    got:             {retrieved_desc}")

    total = len(cases)
    print(f"\nRecall@{top_k}: {hits_count}/{total} ({hits_count / total:.0%})")


if __name__ == "__main__":
    run_eval()
