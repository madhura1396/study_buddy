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
