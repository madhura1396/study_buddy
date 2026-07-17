"""Score generated answers on quality and citation faithfulness, using an
LLM judge (Groq) — a second layer on top of scripts/eval_retrieval.py, which
only checks whether the right chunks were retrieved, not whether the final
answer is good or accurately grounded in them.

Quality: does the answer cover the facts scripts/eval_cases.json says a good
answer should convey (derived from each case's "expected" phrases)?

Faithfulness: for every [N] citation the answer actually uses, is that claim
really supported by the corresponding cited chunk's text, or fabricated?

Run from the project root:
    python -m scripts.eval_answers
"""

import json
import re
from pathlib import Path

from src.config import GROQ_MODEL
from src.generate import generate_answer, get_client

EVAL_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"

QUALITY_JUDGE_PROMPT = (
    "You are grading a study assistant's answer to a question. You will be "
    "given the question, the assistant's answer, and a list of facts a good "
    "answer should accurately convey. Score the answer's quality from 1 "
    "(poor - missing or wrong) to 5 (excellent - complete and accurate) "
    'based on how well it covers those facts. Respond with ONLY a JSON '
    'object: {"score": <1-5>, "reasoning": "<one sentence>"}.'
)

FAITHFULNESS_JUDGE_PROMPT = (
    "You are checking a study assistant's answer for hallucination. You "
    "will be given the answer text (with bracketed citations like [1], "
    "[2]) and the exact source chunks referenced by each citation number. "
    "Determine whether every factual claim in the answer is actually "
    "supported by its cited chunk(s) — flag anything stated confidently "
    'that is not backed by the cited text. Respond with ONLY a JSON '
    'object: {"faithful": true/false, "reasoning": "<one sentence>"}.'
)


def _parse_judge_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"error": "could not parse judge output", "raw": text}


def _ask_judge(system_prompt: str, user_prompt: str) -> dict:
    client = get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return _parse_judge_json(response.choices[0].message.content)


def judge_quality(question: str, answer: str, must_cover: list[str]) -> dict:
    facts = "\n".join(f"- {fact}" for fact in must_cover)
    prompt = f"Question: {question}\n\nAnswer: {answer}\n\nFacts a good answer should cover:\n{facts}"
    return _ask_judge(QUALITY_JUDGE_PROMPT, prompt)


def judge_faithfulness(answer: str, hits: list[dict]) -> dict:
    cited_indices = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer)))
    valid_indices = [i for i in cited_indices if 1 <= i <= len(hits)]
    if not valid_indices:
        return {"faithful": None, "reasoning": "No citations found in the answer."}

    cited_context = "\n\n".join(f"[{i}] {hits[i - 1]['text']}" for i in valid_indices)
    prompt = f"Answer:\n{answer}\n\nCited source chunks:\n{cited_context}"
    return _ask_judge(FAITHFULNESS_JUDGE_PROMPT, prompt)


def run_eval() -> None:
    with open(EVAL_CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    quality_scores = []
    faithful_pass = 0
    faithful_total = 0

    for case in cases:
        question = case["question"]
        must_cover = [e["contains"] for e in case["expected"]]

        result = generate_answer(question)
        answer, hits = result["answer"], result["hits"]

        quality = judge_quality(question, answer, must_cover)
        score = quality.get("score")
        if isinstance(score, (int, float)):
            quality_scores.append(score)

        faithfulness = judge_faithfulness(answer, hits)
        is_faithful = faithfulness.get("faithful")
        if is_faithful is not None:
            faithful_total += 1
            faithful_pass += bool(is_faithful)

        print(f"\n[{question}]")
        print(f"  quality:    {score}/5 — {quality.get('reasoning', quality.get('raw', ''))}")
        print(f"  faithful:   {is_faithful} — {faithfulness.get('reasoning', faithfulness.get('raw', ''))}")

    if quality_scores:
        avg = sum(quality_scores) / len(quality_scores)
        print(f"\nAverage quality: {avg:.1f}/5 ({len(quality_scores)} cases)")
    if faithful_total:
        print(f"Faithfulness: {faithful_pass}/{faithful_total} ({faithful_pass / faithful_total:.0%})")


if __name__ == "__main__":
    run_eval()
