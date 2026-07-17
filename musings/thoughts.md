# Musings: problems hit while building the retrieval/generation pipeline

Running log of real problems encountered, their root causes, and what could
be tried next. Written as we went, not retroactively cleaned up.

---

## 1. Study docs were .docx, but ingest.py only read .txt/.pdf

**Problem:** `python -m src.ingest` silently ingested 0 chunks once real study
docs were dropped into `data/sample_docs/` — every file was `.docx`.

**Root cause:** `load_documents()` only had readers registered for `.txt` and
`.pdf`; unsupported extensions were skipped without a warning.

**Fix:** added `python-docx` and a `.docx` reader/parser.

**Future enhancement:** `load_documents()` could log which files were skipped
and why, instead of silently dropping them — would have surfaced this
instantly instead of needing a manual `ls` to notice.

---

## 2. No way to know if chunking/retrieval was actually any good

**Problem:** the only feedback loop was eyeballing `scripts/test_retrieval.py`
output — no repeatable, scorable signal for whether a chunking change helped
or hurt.

**Root cause:** there was no eval harness at all.

**Fix:** `scripts/eval_retrieval.py` + `scripts/eval_cases.json` — a small
hand-labeled question set scored as recall@k.

**Future enhancement:** 13 cases is a small, easily-overfit sample. A bigger,
more diverse eval set (covering all docs and more question phrasings) would
make chunking decisions much more trustworthy than they currently are.

---

## 3. First eval design used chunk_index as ground truth — invalid across configs

**Problem:** a chunk-size sweep (500 vs 800 vs 1000) showed recall crashing
from 85% to 31% to 23% as chunk size grew. Looked like bigger chunks were
catastrophically worse.

**Root cause:** the eval matched on `(source, chunk_index)`. Chunk indices
are just a sequential counter recomputed from scratch every time you
re-chunk — chunk #5 under one config isn't the same content as chunk #5
under another. The ground truth itself silently went stale, not the
retrieval.

**Fix:** switched matching to `(source, distinctive content phrase)` — a
substring check against the returned chunk text — which stays valid no
matter how chunk boundaries shift.

**Future enhancement:** none needed here, but worth remembering as a general
lesson: any eval ground truth tied to an implementation-detail index (not
content) will silently rot the next time that implementation changes.

---

## 4. CHUNK_OVERLAP=50 was splitting answers away from their headings

**Problem:** with the fixed eval, `CHUNK_SIZE=500 / CHUNK_OVERLAP=50` scored
77% (10/13) — two questions ("key linear regression formulas", "why square
errors not absolute value") retrieved the wrong document entirely.

**Root cause:** `RecursiveCharacterTextSplitter` cuts purely on character
count, blind to document structure. A heading and the content explaining it
could land in different chunks, and 50 characters of overlap often wasn't
enough to keep them stitched together.

**Fix:** raised `CHUNK_OVERLAP` to 100 (chunk size unchanged) — recall went
to 92% (12/13). More overlap margin meant more chunks near a boundary ended
up self-contained instead of half-orphaned.

**Future enhancement:** this was superseded by structure-aware chunking (see
below), which addresses the same root cause more directly — overlap was
always a blunt workaround for not respecting document structure.

---

## 5. Structure-aware chunking, take 1: heading-in-metadata-only tanked recall

**Problem:** after switching to section-first chunking (split on `Heading
1/2/3` docx styles before any character-based sub-splitting), and storing
the section heading as a ChromaDB metadata field *without* duplicating it
into the embedded chunk text (per the original spec — "not duplicated into
the embedded content") — recall dropped from 92% to 62%.

**Root cause:** ChromaDB's vector search only ever looks at the embedded
`documents` text, never at metadata. Headings like "L - LINEARITY" or "KEY
FORMULAS" carried exactly the keywords the eval questions were phrased
around. Moving those words out of the embedded text into metadata-only made
them invisible to retrieval, even though a human (or the LLM at generation
time) could still see them fine.

**Fix:** decided to embed the heading + body together (`"Section: {heading}\n
{body}"` as the stored/embedded text), while still keeping `heading` as its
own metadata field for reliable structured access. This is a deliberate
deviation from the original "don't duplicate into embedded content"
constraint, traded for retrieval actually working. Recall recovered to 77%.

**Future enhancement:** if strict metadata/content separation is ever
required again (e.g. for a stricter provenance/audit reason), the correct
way to do it without sacrificing recall is to embed `heading + body` via a
manually-computed embedding vector while storing only `body` as the
retrievable/displayed text — ChromaDB supports passing precomputed
`embeddings=` separately from `documents=`. Not implemented here to keep
scope contained.

---

## 6. Nested headings orphaned subsections from the context that explained them

**Problem:** even after fixing #5, one question regressed:
"why do we square errors instead of absolute value" stopped matching
`LR1.docx`'s "Reason 1: Mathematical Tractability" section.

**Root cause:** that section is a `Heading 3` nested inside a `Heading 2`
("6. Why Squaring Instead of Absolute Value?"). The parser initially treated
every heading level as a flat, independent section boundary — so "Reason 1"
became its own chunk containing none of the words "squaring" or "absolute
value," which only appeared in its *parent* heading, one level up and now
discarded.

**Fix:** track a heading stack per document and store the full breadcrumb
path (`"6. Why Squaring Instead of Absolute Value? > Reason 1: Mathematical
Tractability"`) rather than just the leaf heading. This carries parent
context down into child chunks. Fixed this specific case.

**Future enhancement:** breadcrumbs can get long for deeply nested docs
(4+ heading levels) and start eating into the effective `CHUNK_SIZE` budget
for actual content. Worth capping breadcrumb depth or truncating older
ancestors if this becomes a problem on future documents.

---

## 7. Recall plateaued at 77% — root cause turned out to be the embedding model itself

**Problem:** after the breadcrumb fix, recall stayed at 77% (10/13), and
raising `top_k` from 5 to 7 or 10 didn't recover the two remaining failures.

**Root cause, case A ("what are the key linear regression formulas"):** the
target chunk is almost entirely math notation (`Y = β₀ + β₁X + ε`, `SSE =
Σ(yᵢ - ŷᵢ)²`, etc.) with very little natural-language prose. `all-MiniLM-
L6-v2` builds a chunk embedding by mean-pooling token embeddings; symbol-
heavy tokens (Greek letters, summation signs) don't carry much semantic
weight for a model trained mostly on natural sentence pairs, so they dilute
the pooled vector away from matching a natural-language question. This
chunk failed even in the 92%-recall run — it's a persistent weak point, not
something the restructuring caused.

**Root cause, case B ("do predictor variables need to be normally
distributed"):** the correct chunk explicitly contains the answer ("No
assumptions about distributi[on]"), but it didn't even appear in the top 15
–30 results. Two compounding reasons: (1) the chunk is again heavy with
notation and ASCII-art arrows illustrating an equation, which dilutes its
embedding the same way as case A; (2) answering the question actually
requires an inferential step ("the systematic part is deterministic →
therefore X doesn't need a normality assumption") that isn't a paraphrase of
the question — bi-encoder embedding models are good at topical/lexical
similarity, not multi-step logical inference, and MiniLM-L6 is a small,
6-layer distilled model with limited semantic depth to begin with.

**Fix:** none applied — flagged as a modeling limitation rather than a
chunking bug, and confirmed `top_k` increases don't paper over it (one
target was rank 12, the other wasn't found even at rank 30).

**Future enhancement (the real next steps, in order of effort):**
1. Swap `all-MiniLM-L6-v2` for a stronger embedding model (e.g.
   `all-mpnet-base-v2`) — better semantic quality at the cost of slower
   embedding and a larger model download. Same sentence-transformers API,
   drop-in change in `src/config.py` + re-ingest.
2. Add an LLM-based re-ranking pass: retrieve a wider candidate set (e.g.
   top 15 by cosine distance), then have an LLM judge/reorder them by actual
   relevance to the question before picking the final top-k for the prompt.
   This is the standard fix for exactly the "requires inference, not just
   similarity" failure mode in case B.
3. Preprocess math-heavy chunks to carry a short plain-language paraphrase
   alongside the notation (e.g. auto-generated or manually authored), so the
   embedded text has natural-language signal to match against even when the
   source content is dense with symbols.
4. Grow the eval set — 13 cases is small enough that any one hard case (like
   the two above) swings the score by ~8 points. A larger, more
   representative set would make it clearer whether these are edge cases or
   a systemic pattern worth prioritizing.

---

## 8. Streamlit app: couldn't ask a second question

**Problem:** in `app.py`, after asking one question and getting an answer,
typing a new question into the same input box and pressing Enter appeared to
do nothing — no new answer, no visible change at all.

**Root cause:** the input was `st.text_input`, which does not clear itself
after the app re-renders. Streamlit only re-runs the script when a widget's
*value* changes, not merely when Enter is pressed. So the box still held the
first question's text; unless you manually selected-all and retyped a
genuinely different string, Streamlit saw "no change" and never re-ran
`generate_answer()` at all. This wasn't a backend/retrieval bug — the whole
Q&A pipeline was working fine, the UI just never asked it to run again.

**Fix:** switched to `st.chat_input`, which is designed for exactly this —
it auto-clears after every submission and reliably triggers a rerun each
time, regardless of whether the new text happens to differ from the old.
Also added a `st.session_state`-backed message history so previous
questions/answers/sources stay visible as a running conversation instead of
being overwritten each turn, which is the more natural mental model for
"keep asking follow-ups."

**Future enhancement:** the current history keeps growing unbounded for the
lifetime of the browser session (cleared only on page refresh) — fine for a
personal study tool, but if this were shared or long-running, a "clear
conversation" button or a cap on stored turns would be a reasonable
addition. Also worth considering: passing recent conversation turns into
`generate_answer()`'s prompt so follow-up questions like "what about the
second one?" can resolve pronouns/references against prior turns, since
right now each question is answered in isolation with no memory of the
conversation.

---

## 9. Switching embedding models revealed a second, hidden top_k default

**Problem:** after swapping `EMBEDDING_MODEL_NAME` from `all-MiniLM-L6-v2` to
`all-mpnet-base-v2` (the future enhancement flagged in #7) and re-ingesting,
`scripts/eval_retrieval.py` still reported 77% (10/13) — no visible
improvement, even though the stronger model should have helped.

**Root cause:** two separate things, stacked:
1. mpnet did genuinely fix one failure (the "predictor variables" inference
   case) but *introduced* a different near-miss ("linearity assumption"
   dropped to rank 6, just outside `top_k=5`) — so the headline score didn't
   move even though the underlying ranking quality had changed.
2. `run_eval()` in `scripts/eval_retrieval.py` had its own hardcoded
   `top_k: int = 5` default, completely independent of whatever default
   `retrieve()` itself used. Bumping `retrieve()`'s default earlier (in
   problem area around structure-aware chunking) had no effect on eval
   results at all, because the eval script never picked it up — it was
   silently testing against a stale `top_k` the whole time.

**Fix:** raised `retrieve()`'s and `generate_answer()`'s default `top_k` from
5 to 8 (mpnet's near-misses were landing around rank 6, so 8 gives margin),
and fixed `run_eval()`'s default to match. Recall moved to 85% (11/13) —
our second-best result overall, one point of failure being the same
persistent math-notation chunk from #7.

**Future enhancement:** having the same conceptual parameter (`top_k`)
duplicated as independent defaults in three places (`retrieve()`,
`generate_answer()`, `run_eval()`) is a footgun — it's easy to change one
and assume the others followed. If this keeps needing to move, it should
live in `src/config.py` as `DEFAULT_TOP_K` and be imported everywhere,
rather than being three separately-maintained magic numbers.

---

## 10. git broke machine-wide — not a project bug, but blocked every push

**Problem:** mid-session, every `git` command (even `git status`) started
failing with `Error loading required libraries ... unable to locate
xcodebuild`, blocking commits and pushes entirely.

**Root cause:** on macOS, `/usr/bin/git` isn't a real standalone binary — it
resolves through Apple's active developer tools path (`xcode-select`) before
running. This machine has both a full Xcode.app and the standalone Command
Line Tools installed, and `xcode-select` was pointed at the Xcode.app copy,
which had a corrupted library (`libxcodebuildLoader.dylib` failing to load —
likely from an interrupted Xcode update). Since `git` needs that path to
resolve successfully before it can do anything, every git command failed
before reaching the actual command.

**Fix:** `sudo xcode-select --switch /Library/Developer/CommandLineTools` —
pointed the active developer directory at the lighter, standalone CLT
install instead of the broken Xcode.app, without touching or deleting
anything inside Xcode.app itself (considered and rejected deleting the
corrupted framework file directly — too destructive/uncertain a fix for a
large interlinked app bundle, and might not have even resolved the
`xcodebuild` lookup failure itself).

**Future enhancement:** none needed for this project specifically — this
was environment/OS-level, not something the codebase can guard against.
Worth remembering as a general lesson: if `git` ever fails with an
Xcode-flavored error on macOS, check `xcode-select -p` before assuming the
repo or git config is the problem.

---

## 11. Folder categorization crashed the app: `X | None` on Python 3.9

**Problem:** after adding `resolve_to_files()`'s return type annotation
`list[tuple[dict, str | None]]` (part of the category/folder-import
feature), the whole Streamlit app failed to even load — `TypeError:
unsupported operand type(s) for |: 'type' and 'NoneType'`, raised at import
time, before any app code ran.

**Root cause:** the `X | Y` union type syntax (PEP 604) is only evaluated
lazily (and thus safe) in Python 3.10+, or on 3.9 with `from __future__
import annotations` at the top of the file. This project's `.venv` runs
Python 3.9 (already flagged as past end-of-life when we installed the
Google API client libraries) without that future-import, so the `|`
operator tried to actually execute as a runtime expression between two
type objects — which `type` doesn't support — and crashed on the `def`
line itself.

**Fix:** used `typing.Optional[str]` instead of `str | None` in
`src/drive.py`. Grepped the rest of `src/*.py` and `app.py` for the same
pattern first to confirm it was the only occurrence.

**Future enhancement:** this class of bug is easy to reintroduce since `X |
None` is the more common style to type today and looks correct in an
editor with a modern Python configured. Either add `from __future__ import
annotations` to every module (cheapest, makes the modern syntax safe
everywhere), or — the more durable fix — upgrade the project off Python
3.9, which is already past its official end-of-life per the warnings
`google-auth` prints on every import.

---

## 12. Adding more study docs quietly dropped eval recall from 85% to 69%

**Problem:** after the user added 7 more real study documents (probability
distributions, Bayes' theorem, MLE, etc.) on top of the original 4
linear-regression docs, `scripts/eval_retrieval.py` dropped from 85%
(11/13) to 69% (9/13) — with no chunking or retrieval code changed at all.

**Root cause:** not a bug — the new docs are topically adjacent to the
original eval questions. Queries like "what does it mean for errors to be
normally distributed" now have to compete against `distributions_reference_
guide.docx` and `random_variables_to_ml.docx`, which are *legitimately*
about normal distributions (just not in the linear-regression-assumptions
sense the eval question intends), and rank highly on shared vocabulary
("normal", "distribution"). `top_k=8` isn't enough headroom once the corpus
covers more overlapping ground.

**Fix:** none applied — this is expected behavior as the corpus grows, not
a defect in the categorization feature that was actually being built when
this was noticed. Flagged rather than "fixed" by further inflating `top_k`,
since that's treating a symptom (this eval set is narrow and now
undersized relative to the corpus) rather than a cause.

**Future enhancement:** this is the clearest sign yet (see also problem
#2's future note) that the 13-case eval set needs to grow alongside the
document corpus — it should include questions that specifically probe
whether topically-similar-but-distinct documents are being disambiguated
correctly (e.g. "linear regression's normality assumption" vs "the normal
distribution in general"), not just whether *a* relevant chunk shows up
somewhere in the top-k.
