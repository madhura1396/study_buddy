"""Generate multiple-choice quizzes and flashcards from ingested chapters,
and persist attempt history so results carry across sessions.
"""

import json
import re
import time
import uuid
from typing import Optional

from groq import GroqError

from src.config import DATA_DIR, GROQ_MODEL
from src.generate import MissingAPIKeyError, get_client
from src.ingest import get_collection

QUIZ_HISTORY_PATH = DATA_DIR / ".quiz_history.json"

GROUNDED_MCQ_SYSTEM_PROMPT = (
    "You are a study quiz generator. Given study material, write {n} "
    "multiple-choice questions that test understanding of it. Each question "
    "must have exactly 4 options with exactly one correct answer. Respond "
    "with ONLY a JSON array, no other text, in this exact shape: "
    '[{{"question": "...", "options": ["...", "...", "...", "..."], '
    '"correct_index": 0, "explanation": "one sentence on why"}}]. '
    "Base every question strictly on the material provided — do not invent "
    "facts beyond it."
)

GROUNDED_FLASHCARD_SYSTEM_PROMPT = (
    "You are a study flashcard generator. Given study material, write {n} "
    "flashcards testing recall of its key concepts. Respond with ONLY a "
    'JSON array, no other text, in this exact shape: [{{"front": "a '
    'question or term", "back": "the answer or definition"}}]. Base every '
    "card strictly on the material provided — do not invent facts beyond it."
)

# Interview-style prompts are deliberately NOT grounded in any provided
# text — they draw on the model's general knowledge of the topic, the way
# a real interviewer would ask beyond whatever one set of notes happens to
# cover (e.g. edge cases, "why" questions, comparisons to related concepts).
INTERVIEW_MCQ_SYSTEM_PROMPT = (
    "You are an experienced technical interviewer. Given a topic, write {n} "
    "multiple-choice questions of the kind commonly asked about this topic "
    "in real technical interviews — draw on your general knowledge of the "
    "subject, not limited to any specific document. Cover a mix of "
    "fundamentals, edge cases, and \"why\"/\"when would you\" questions "
    "typical of interview prep. Each question must have exactly 4 options "
    "with exactly one correct answer. Respond with ONLY a JSON array, no "
    'other text, in this exact shape: [{{"question": "...", "options": '
    '["...", "...", "...", "..."], "correct_index": 0, "explanation": "one '
    'sentence on why"}}].'
)

INTERVIEW_FLASHCARD_SYSTEM_PROMPT = (
    "You are an experienced technical interviewer. Given a topic, write {n} "
    "flashcards covering the questions and concepts commonly asked about "
    "this topic in real technical interviews — draw on your general "
    "knowledge of the subject, not limited to any specific document. "
    'Respond with ONLY a JSON array, no other text, in this exact shape: '
    '[{{"front": "an interview question or term", "back": "a strong '
    'answer"}}].'
)


class GenerationFailedError(RuntimeError):
    pass


def get_chunks_for_file(category: str, source: str) -> list[dict]:
    """Return this file's chunks (text + heading), in original order."""
    result = get_collection().get(
        where={"$and": [{"category": category}, {"source": source}]}
    )
    rows = sorted(
        zip(result["documents"], result["metadatas"]), key=lambda r: r[1]["chunk_index"]
    )
    return [{"text": doc, "heading": meta.get("heading", "")} for doc, meta in rows]


def build_scope_text(
    category: str, sources: list[str], headings: Optional[list[str]] = None
) -> str:
    """Concatenate chunk text for the selected files, optionally narrowed to
    only chunks whose heading is in `headings` (chapter-level selection)."""
    parts = []
    for source in sources:
        for chunk in get_chunks_for_file(category, source):
            if headings and chunk["heading"] not in headings:
                continue
            parts.append(chunk["text"])
    return "\n\n".join(parts)


def _parse_json_array(text: str) -> list[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise GenerationFailedError(f"Could not parse quiz generator output: {text[:300]}")


def _generate(system_prompt_template: str, user_content: str, n: int) -> list[dict]:
    if not user_content.strip():
        raise GenerationFailedError("No content in the selected scope to generate from.")
    try:
        client = get_client()
    except MissingAPIKeyError:
        raise
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_template.format(n=n)},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
    except GroqError as e:
        raise GenerationFailedError(f"Groq API request failed: {e}") from e
    return _parse_json_array(response.choices[0].message.content)


def generate_mcqs(scope_text: str, n: int = 5, grounded: bool = True) -> list[dict]:
    if grounded:
        return _generate(GROUNDED_MCQ_SYSTEM_PROMPT, f"Study material:\n{scope_text}", n)
    return _generate(INTERVIEW_MCQ_SYSTEM_PROMPT, f"Topic: {scope_text}", n)


def generate_flashcards(scope_text: str, n: int = 5, grounded: bool = True) -> list[dict]:
    if grounded:
        return _generate(GROUNDED_FLASHCARD_SYSTEM_PROMPT, f"Study material:\n{scope_text}", n)
    return _generate(INTERVIEW_FLASHCARD_SYSTEM_PROMPT, f"Topic: {scope_text}", n)


def _load_history() -> list[dict]:
    if not QUIZ_HISTORY_PATH.exists():
        return []
    return json.loads(QUIZ_HISTORY_PATH.read_text())


def _save_history(history: list[dict]) -> None:
    QUIZ_HISTORY_PATH.write_text(json.dumps(history, indent=2))


def save_attempt(kind: str, scope_label: str, score: int, total: int, missed: list[str]) -> None:
    """Record one completed quiz or flashcard session."""
    history = _load_history()
    history.append(
        {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "scope_label": scope_label,
            "score": score,
            "total": total,
            "missed": missed,
            "timestamp": time.time(),
        }
    )
    _save_history(history)


def get_history() -> list[dict]:
    return sorted(_load_history(), key=lambda a: a["timestamp"], reverse=True)
