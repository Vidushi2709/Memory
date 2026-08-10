# Memory System on LongMemEval — Evaluation Report

**Date:** 2026-08-10
**Benchmark:** LongMemEval (oracle setting), arXiv 2410.10813
**Model:** Mistral Small 3.2 24B via OpenRouter
**Sample:** 30 questions, stratified 5 per question type

The memory chatbot scored **19 / 30 (63%)** on a stratified sample of LongMemEval's oracle setting. The profile is sharply uneven: preference and user-fact questions are near ceiling, while two categories fail for structural reasons — the system never stores what the *assistant* said, and it cannot aggregate multiple memories into one answer. Both point to concrete, buildable fixes rather than tuning.

## System under test

A Python memory layer over ChromaDB and a 24B open-weights model, built up over three tiers:

- **Write path (Mem0-style):** an LLM extracts atomic facts (with importance 1-10, resolved dates, keywords, and a context sentence), then decides per fact whether to ADD, UPDATE, SUPERSEDE, or NOOP against its ten nearest neighbors.
- **Retrieval:** vector similarity + BM25 keyword search + Personalized PageRank over a memory-link graph, fused by reciprocal rank, then re-ranked by Ebbinghaus retention (`e^(−days/strength)`, strength grows per recall) and importance.
- **Always-in-prompt core profile** (MemGPT-style), rewritten by the LLM as facts change.
- **Consolidation:** session-end dedup, importance-triggered reflection insights, A-Mem-style memory evolution of linked neighbors' context lines.
- **Session layer:** every memory tagged with its session; raw transcripts archived to an experience bank (`transcripts/`).

Old memories are soft-deleted (`is_current=0`), never removed, so history stays queryable.

## Method

LongMemEval embeds questions in multi-session chat histories and tests five memory abilities. The **oracle setting** supplies only the evidence sessions — no distractor haystack — so these scores measure whether the pipeline can extract, update, and reason over what it saw. They are an upper bound: the harder `_s` setting (~115k-token haystacks) additionally stresses needle-finding.

Each question's sessions were ingested through the real write path with their historical dates (the extractor resolves "last month" against the session's date, not today's), then the question was asked as of its question date and judged by an LLM against the gold answer. Abstention variants (`*_abs`) count as correct only when the system admits it lacks the information. Runner: `eval/run_longmemeval.py`; full answers in `eval/results/longmemeval-01-baseline.json` (see `eval/results/README.md` for the index of all runs).

## Results

| Question type | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Single-session preference | 5 | 5 | 100% |
| Single-session user | 4 | 5 | 80% |
| Knowledge update | 3 | 5 | 60% |
| Temporal reasoning | 3 | 5 | 60% |
| Multi-session | 2 | 5 | 40% |
| Single-session assistant | 2 | 5 | 40% |
| Abstention subset | 1 | 2 | 50% |
| **Total** | **19** | **30** | **63%** |

## Failure analysis

### 1. The system doesn't remember its own words

Worst structural gap (single-session-assistant, 40%). Extraction stores facts about the *user*; recommendations, lists, and answers the assistant produced are never memorized, so any "what did you tell me?" question fails.

> **Q:** What was my Bishop's second move in our chess game?
> **A:** "I don't have the record of that specific move."

The raw material for the fix already exists: every assistant turn is archived verbatim in the transcript experience bank — it just isn't searchable yet.

### 2. No aggregation across memories

Multi-session questions (40%) require collecting *all* matching memories and counting or comparing them. Top-5 conversational retrieval isn't built for enumeration, so answers came back partial, or a needed fact was missing entirely.

> **Q:** How much more money did I raise than my initial goal in the charity cycling event?
> **A:** "You raised $250 in total, but I don't have information about your initial goal."

### 3. Date arithmetic done by the model, not by code

Temporal questions (60%) usually retrieved the right dated memories, then miscomputed the duration between dates. The metadata holds exact timestamps; the subtraction should be deterministic instead of delegated to the LLM.

> **Q:** How long have I been collecting vintage cameras?
> **A:** "…about three and a half months" (gold: a much longer span)

### 4. Abstention collapses under similar-memory pressure

With a near-matching memory available, the model answered from the wrong memory instead of admitting the asked-about fact was never stated (1 of 2 abstention questions). The same failure mode is documented across all six systems in the MemConflict benchmark.

> **Q:** How many engineers do I lead in my new role as *Software Engineer Manager*? (never discussed)
> **A:** Answered with the team size from a *different, similar* role memory.

## Reading the numbers fairly

Published LongMemEval scores in the 80–95% range (Emergence 86%, vendor-reported Zep/Mastra and others higher) are not directly comparable: they run the harder haystack setting but on GPT-4o-class frontier models, typically with cross-encoder rerankers. This run pairs the easier oracle setting with a 24B open model. The per-category *shape* — strong on single-session user facts and preferences, weak on assistant recall, aggregation, and temporal arithmetic — matches the failure modes the benchmark's own paper reports for memory systems generally, which suggests the harness is measuring the right things.

For internal context: the project's own 13-question scenario harness scores 100%, which is why this benchmark was brought in — the internal set no longer discriminates. Two measured experiments preceded this run: compose-on-read (query-tailored memory digest) showed no accuracy gain, added latency, and blurred fact attribution once — disabled; PageRank over the link graph showed no measurable difference yet but costs no LLM calls — kept.

## Recommendations, ranked

1. **Search the transcript archive.** Make raw turns retrievable alongside distilled memories (turn-match, session-retrieve granularity). Directly targets the 40% single-session-assistant category and parts of multi-session.
2. **Aggregation-aware answering.** For enumeration/comparison questions, sweep by time-range or topic instead of top-5, and add an explicit "select relevant memories, then answer" stage (the selection behavior RL-trained managers learned in Memory-R1).
3. **Deterministic date math.** Compute durations and orderings in code from stored timestamps and hand the model the result — same philosophy as deterministic freshness resolution ("don't ask the LLM to track versions").
4. **Deterministic freshness for knowledge updates.** Same-subject conflicts resolved by timestamp in code, with the LLM only extracting candidates.
5. **Guarded abstention.** Prompt-side: answer only if a retrieved memory actually states the asked-about fact; near-matches should be named as such, not silently adopted.

## Addendum (2026-08-11): loss-attribution audit

To locate *where* failures happen, the runner gained two audit stages per question: a **store audit** (is the gold answer derivable from everything the write path stored?) and a **retrieval audit** (is it derivable from what retrieval actually surfaced?). Re-running only the 10 previously failed non-abstention questions:

| Outcome on re-run | Count |
|---|---:|
| Passed this time (run-to-run variance) | 4 |
| Write-path loss (answer never stored) | 4 |
| Retrieval loss (stored, not surfaced) | 1 |
| Reasoning loss (surfaced, answered wrong) | 1 |

Two findings changed the plan:

1. **The write path is the largest systematic loss — but it splits in two.** Two of the four write-path losses are the by-design gap (assistant content is never extracted); two are genuine extraction misses where the small model dropped a stated detail ("initial goal", "with my friend").
2. **40% of the original failures were nondeterminism**, not systematic error — all LLM calls ran at temperature 0.7, so per-fact judgments and answers were stochastic.

**Changes made in response** (this codebase, same day):

- **Temperature 0.0** on all memory operations and answering — removes the variance term.
- **Transcript-turn retrieval** — raw exchanges (the experience bank) are BM25-searched alongside memories, so assistant answers are recallable; targets the SSA 40% category.
- **Thin write path + sleep-time consolidation** — the hot path is now a single extraction call storing facts append-only; all judgment (supersede/link reconciliation in one large-context call, dedup, evolution, reflection, core-profile rewrite) moved to a session-end sleep pass with exclusive write authority (the Letta sleep-time pattern). This cuts per-fact hot-path LLM judgments from ~7 to ~2 and gives reconciliation full-session context.

Store-coverage ("answer present in store") is the tracked metric for the write path: 6/10 before the restructure, on the failed-question subset.

### Re-benchmark after the restructure (same 30 questions)

| Question type | Before | After |
|---|---:|---:|
| Single-session preference | 100% | 80% |
| Single-session user | 80% | 100% |
| Knowledge update | 60% | 80% |
| Temporal reasoning | 60% | 80% |
| Multi-session | 40% | 100% |
| Single-session assistant | 40% | 40% |
| Abstention subset | 50% | 50% |
| **Total** | **63%** | **80%** |

Store coverage rose from 6/10 (failure subset) to **26/28 (93%)** across all non-abstention questions — the write path now retains nearly everything, confirming the audit's diagnosis. Loss attribution after: 2 write-path, 2 retrieval, 1 reasoning.

Remaining weaknesses: single-session-assistant failures moved from *write-path* losses to *retrieval/truncation* losses — the assistant's words are now stored (transcripts) but turn search surfaces the wrong exchange or the 1200-character excerpt cap cuts the needed detail (e.g. item 7 of a long list). Abstention under similar-memory pressure is still unsolved. Next fixes: longer/smarter transcript excerpts and guarded abstention.

## Addendum (2026-08-11): retrieval and robustness fixes

Follow-up work on the two weaknesses the 80% run left open, plus three bugs the
haystack setup exposed. All verified; none committed.

**Turn search returned nothing on short histories (fixed).** Single-session-assistant
questions failed with the answer present in the transcript archive but never retrieved.
Root cause was not tuning: BM25's IDF term goes negative when a word appears in most
of a small corpus, so the `score > 0` filter discarded every candidate. `search_turns`
now ranks by BM25 but gates on non-stopword overlap. Targeted re-run: single-session
assistant **0/3 → 3/3**, with exact gold answers ("28. Kg3", "4 mummies",
"Transcriptionist"); 6/7 overall on the smoke set, 5/5 store coverage, 5/5 surviving
retrieval.

**Transcript excerpts were being ignored by the model (fixed).** Excerpts shared the
`retrieved_memories` field and were treated as vague recollections — the model answered
"I don't have the list" while the list sat in its context. They now occupy a separate
`past_conversations` field framed as a verbatim record to quote from, and excerpt caps
rose to 4000 characters so long lists survive.

**Abstention (still unsolved).** Asked about a job title never mentioned, the model
answers from a similar stored title. A `supporting_evidence` output field that forces
quoting the source before answering did not fix it. Prompt-level approaches look
exhausted; this likely needs a deterministic check that the specific entity asked
about appears in a retrieved source. For context, the MemConflict benchmark found the
best conflict-recognition score across six production memory systems was 0.25.

**Three robustness bugs found and fixed:**

| Bug | Symptom | Fix |
|---|---|---|
| Pre-1970 dates crash writes | `datetime.timestamp()` raises `[Errno 22]` on Windows for dates before 1970 — one childhood memory killed the whole session's write | `to_epoch()` subtracts from the epoch instead |
| ChromaDB telemetry races | Concurrent ingestion dropped ~14% of sessions with an opaque event-key error | `anonymized_telemetry=False`; writes also take a thread lock |
| `sleep_pass` skipped sessions | Running it periodically left earlier sessions unreconciled — no supersede, link, or core update | Accepts a batch of session ids |

The first two would corrupt real user memory, not just benchmark runs. A concurrency
stress test (60 concurrent writes plus links and supersedes) now passes with no lost
writes and no errors.

**Open finding, not yet addressed:** importance currently outweighs relevance in
ranking. Reciprocal rank fusion normalizes by rank position, so an unrelated memory
lands within ~3% of a perfect match on the relevance term while importance can differ
by 0.8 — in one test an unrelated memory outranked a near-exact match. Suspect first
if haystack retrieval underperforms.

**Haystack pilot: set up, not yet run.** `--haystack` (the `_s` file, median 48
sessions and 491 turns per question), `--sleep-every N`, and batched concurrent
ingestion (~6x faster) are in place and running cleanly; the run was stopped for time.

## Reproduce

```bash
python eval/run_longmemeval.py --per-type 5 --out my-run.json   # oracle, 30 questions
python eval/run_longmemeval.py --types temporal-reasoning,knowledge-update
python eval/run_longmemeval.py --haystack --per-type 1 --sleep-every 5   # ~48 sessions/question
python eval/run_eval.py                                          # internal 13-question harness
```

Results land in `eval/results/`; datasets live in `eval/data/` (gitignored — download
from the `xiaowu0162/longmemeval-cleaned` HuggingFace repo).

---

*Dataset: `longmemeval_oracle.json` (500 questions; stratified first-5 per type by question id, deterministic). Judge: same model as the system under test; abstention-aware rubric. All memory writes ran against a throwaway database in a temp directory — no production memories were touched.*
