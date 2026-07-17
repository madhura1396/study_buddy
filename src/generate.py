"""Given a question, retrieve grounding chunks and ask Groq for a cited answer."""

from groq import Groq, GroqError

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.retrieval import retrieve

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question thoroughly using only "
    "the numbered context blocks provided, in clear natural prose — explain "
    "concepts fully rather than giving a terse one-liner, and include "
    "relevant detail, examples, or nuance, but ONLY when that detail is "
    "explicitly present in the context. Being thorough means fully "
    "explaining what the context actually says, not adding specifics, "
    "directions, or generalizations the context doesn't state — if the "
    "context leaves something unspecified (e.g. it doesn't say which "
    "direction a value indicates, or doesn't confirm a claim generalizes "
    "to a different scenario), say it's unspecified rather than filling "
    "the gap yourself. For every claim, cite the block's number in square "
    "brackets right after the claim, like [1] or [2][3] for multiple "
    "sources, and only make claims a cited block actually supports — never "
    "write out the source filename or chunk number inline, just the "
    "bracketed number(s); the full source list is shown separately below "
    "your answer. If the context doesn't contain enough information to "
    "answer, say so explicitly instead of guessing. End your answer with "
    "one short, relevant follow-up question that invites the student to go "
    "deeper on the topic (e.g. a related concept also covered in the "
    "context) — but only ask about things the context can actually "
    "support."
)


class MissingAPIKeyError(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


def get_client() -> Groq:
    if not GROQ_API_KEY:
        raise MissingAPIKeyError(
            "GROQ_API_KEY not set. Create a .env file at the project root with "
            "GROQ_API_KEY=<your key> (get one from https://console.groq.com/keys)."
        )
    return Groq(api_key=GROQ_API_KEY)


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] source={hit['source']} chunk={hit['chunk_index']}\n{hit['text']}"
        )
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, top_k: int = 8) -> dict:
    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {
            "answer": (
                "No relevant study material found — has `python -m src.ingest` "
                "been run, and does data/sample_docs contain documents?"
            ),
            "sources": [],
            "hits": [],
        }

    client = get_client()
    prompt = build_prompt(question, hits)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    except GroqError as e:
        raise GenerationError(f"Groq API request failed: {e}") from e

    answer = response.choices[0].message.content
    sources = [{"source": h["source"], "chunk_index": h["chunk_index"]} for h in hits]
    return {"answer": answer, "sources": sources, "hits": hits}
