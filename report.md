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

## Addendum (2026-08-11, later): haystack results, performance, model choice

### Haystack pilot — 5/6 (83%)

First run against the `_s` setting (median 48 sessions, ~490 turns per question).
Retrieval surfaced the needed evidence in **6/6** questions under real distractor
noise, which is the meaningful result: the pipeline had never faced competing
sessions before. Not comparable to Zep (71.2%) or Emergence (86%) — six questions
carries an enormous error bar, and those systems use frontier models on all 500.

The single failure was a counting question, and the cause was **not** aggregation:

> *How many doctor's appointments did I go to in March?* — gold **2**, answered **3**.

The haystack mixes attended, scheduled, and merely-considered appointments
("I'm *considering* scheduling with Dr. Patel", "I'm *scheduled* for an EMG on
April 1st"). Extraction flattened intentions into completed facts, so the count
was wrong before any counting happened. A retrieval-side aggregation fix — which
was the queued remedy — would not have helped.

**Fix: a `status` field** (`happened` / `planned` / `considered` / `ongoing`) on
every extracted memory, asserted in the fact text too ("User is considering X"),
persisted to ChromaDB, and surfaced at answer time as `[PLANNED, did not happen]`.
Verified on the failing question: now answers **"two doctor's appointments —
Dr. Smith on March 3rd and Dr. Thompson on March 20th"**. Beyond benchmarks, the
old behaviour told users they attended appointments they had only thought about.

### Performance: 18.4 min → 9.4 min per question

Profiling one full question (51 sessions) showed it is **~100% LLM latency** —
all local work (embeddings, ChromaDB, the O(n²) dedup comparison) totals ~75s of
1107s. The O(n²) loop specifically was **2.3 seconds**.

| Stage | Calls | LLM seconds |
|---|---:|---:|
| Extraction | 51 | 1225 (69%) |
| Core profile refresh | 11 | 400 (23%) |
| Reconcile | 11 | 118 |
| Reflect / evolve / dedup | 33 | 112 |

Two changes, both pure scheduling: **core refresh runs once** (it rebuilt an
80-word profile eleven times and discarded ten), and the **independent sleep-pass
stages run concurrently**. Measured result: **9.4 min/question, a 49% cut.**

A third change — trimming the extraction schema — caused a silent quality
regression the internal suite caught: "pilot with Indigo" became "pilot". The
terseness instruction had bled from the search fields into the fact text. Fixed
by scoping terseness explicitly and requiring names/employers/numbers to survive.
**Timing measurements would never have caught this; the regression suite did.**

### Provider latency is not stable

A full 30-question run was launched and abandoned at 4/30 (3 pass, 1 fail).
Elapsed time implied **~34 min/question**, versus 9.4 min measured hours earlier
on identically-sized questions (46–52 sessions). A single extraction call
re-timed at **38.3s against an 18.2s baseline** — same code, same input, ~2x
slower upstream. Mistral Small 3.2 on OpenRouter's shared pool degrades under
sustained load; it returned a 429 "temporarily rate-limited upstream" earlier.
**Any timing claim here is only valid against a same-session baseline.**

### Model selection

Cost is not the binding constraint: a full 30-question haystack run is ~$0.94.
Throughput is. Benchmarked on an identical extraction call:

| Model | Time | Facts | Verdict |
|---|---:|---:|---|
| mistral-small-3.2 (current) | 18.2s | 12 | baseline |
| gpt-oss-120b | 44.7s | 8 | 2.5x slower despite 0.53x price |
| gpt-oss-20b | — | — | failed to produce parseable output |

The gpt-oss models are reasoning models: they spend the token budget on
chain-of-thought before emitting structured output, hit `max_tokens`, and get
truncated. OpenRouter bills reasoning tokens as completion tokens, so the cheaper
sticker price is partly illusory. **Cheaper per token ≠ cheaper for this
workload.** Untested non-reasoning candidates: qwen3-30b-a3b-instruct (0.64x),
gemma-3-12b (0.57x), nova-micro (0.46x).

### Abstention: still unsolved after three attempts

`031748ae_abs` has failed in every run today. Asked "how many engineers do I lead
in my new role as **Software Engineer Manager**" when memory only records
**Senior Software Engineer**, the model answers from the near-match. Attempts:
a guarded-recall prompt clause; a `supporting_evidence` field forcing the model
to quote its source; status tagging (unrelated but adjacent). All failed because
they ask the model to verify a match — and that verification is the broken
faculty. The question also *presupposes* the role, and models accept user framing.

**What would work:** extract the distinctive noun phrase from the question, check
programmatically that it appears in a retrieved source, and force abstention when
it does not. Deterministic comparison, not instruction. For context, MemConflict
found the best conflict-recognition score across six production memory systems
was 0.25 — this is hard industry-wide.

## Addendum: correctness audit (before the next benchmark run)

A full read of the memory pipeline, with each finding verified by a probe script
rather than by inspection alone. Nine bugs were confirmed; one hypothesis was
tested and **disproved** (BM25 dropping negative scores in `memory_store` — the
`rank_bm25` library floors negative IDF internally, so that code was fine).

### Retrieval was ranking on the wrong signal

The re-ranking formula summed relevance, retention and importance with equal
weight. Measured spans: relevance could move a score by **0.71**, importance by
**0.90**, and retention sat at **~1.0 for every memory** because they had all just
been written. So an important-but-unrelated memory outranked an exact match, and
a superseded fact tied with the fact that replaced it. Relevance now leads, with
importance and retention as tiebreakers (0.15 / 0.10) and a 0.30 penalty for
superseded records.

### The relevance floor never filtered anything

The code read ChromaDB's cosine distance as if it spanned 0–2 in angular terms
and computed `1 − distance/2`. ChromaDB's cosine distance is `1 − similarity`, so
a completely unrelated memory (true similarity 0.0) scored **0.5** and sailed past
the 0.3 floor. The effective cutoff was a similarity of −0.4 — nothing was ever
excluded.

Fixing the formula made the floor bite, so the threshold was **calibrated rather
than guessed**, against real question/memory pairs:

| Floor | Relevant kept | Irrelevant admitted |
|---|---|---|
| 0.00 | 12/12 | 47/72 |
| **0.10** | **12/12** | **9/72** |
| 0.25 | 9/12 | 1/72 |
| 0.30 | 5/12 | 1/72 |

The originally intended 0.30 would have discarded **7 of 12** genuinely relevant
memories. `0.10` removes 81% of the noise at no recall cost. A question and the
memory answering it can sit as low as 0.15 ("what am I allergic to?" against
"user cannot eat shellfish") — lexically disjoint, so BM25 would not have saved it.

### Deduplication silently rewrote facts

The merged record was built without `status`, `keywords`, `context`, or `links`.
Because `status` defaults to `happened`, **merging a planned appointment into a
duplicate turned an intention into an event** — undoing the exact guarantee the
status field was added to provide. This is a plausible cause of the unexplained
doctor's-appointment *counting* failure, which fails with the evidence both stored
and retrieved: an over-count is what you would expect if plans were being counted
as visits. Dedup now clusters within `(kind, status)`, so an intention can never
merge with an event, and the merged record inherits keywords, context and links.

### Reconciliation could not see the fact it needed

It compared each session against the **40 most recent** existing memories. On a
haystack of hundreds, the older fact a new one contradicts was never in that
window, so nothing beyond the last 40 could ever be superseded. It now selects
the 40 most *semantically relevant* memories instead.

### Sessions were being ingested out of chronological order

**211 of 500 haystack questions (42%)** list their sessions out of date order, with
a median displacement of 15 positions. Since the system treats later-ingested as
newer, reconciliation was superseding **newer facts with older ones** — directly
attacking the knowledge-update and temporal-reasoning categories. Fixed on both
sides: the harness now ingests oldest-first, and reconciliation refuses any
supersession where the incoming memory is dated *earlier* than the one it would
replace (it links them instead). The second half matters for real users too, who
mention past facts in present conversations.

### The abstention guard was flagging phrases that were present

`_normalise` turned punctuation into spaces without collapsing runs, so a source
saying "Dr. Lee" became `dr__lee` while the question yielded `dr lee` — no match.
The guard reported the phrase as unverified and the assistant abstained on
information it actually held. Now normalised consistently, and padded so a phrase
matches whole words only ("New York" no longer satisfied by "New Yorker").

### Smaller confirmed defects

| Bug | Effect |
|---|---|
| PageRank ranked *every* node, including unreachable ones scoring ~1e-6 | Unrelated memories received rank-fusion credit |
| `add_links` released its lock between read and write | Concurrent link additions silently lost one |
| `max(session_ids)` used string ordering | `lme-0-9` treated as newer than `lme-0-11` |
| Prompt built *inside* reconcile's `try` | A coding error would have been swallowed as "reconcile call failed", disabling reconciliation with no sign |
| Vector query fetched a fixed 30 candidates before filtering | A store full of superseded memories returned almost nothing |

### Performance

| Change | Effect |
|---|---|
| Vectorised dedup clustering (one matmul, not a Python loop over every pair) | **~490x** on that stage; it cost ~2s per sleep pass at 600 memories |
| Embeddings no longer fetched by the three stages that never used them | 384 floats per record per stage saved |
| Write path batched into one embedding call and one upsert | Was one model round trip *per extracted fact* |

### Extraction was silently truncated on rich sessions

Found in the logs of the first post-fix haystack run, which emitted
`LM response was truncated due to exceeding max_tokens=2048` twice while
ingesting 47 sessions. DSPy detects this (`finish_reason == "length"`) but only
*warns* — so the write path stored a partially extracted session as though it
were complete, losing every fact after the cut.

The cap was set from measurement rather than raised by feel: one extracted fact
serialises to **~88 output tokens**, and benchmark sessions run to a median of
**14k characters** (28k at the tail), so a fact-dense session yields 30+ facts and
needs **~2,600–3,500 tokens**. The 2048 ceiling was below what a normal session
requires. It is now 4096, and extraction *reads the truncation signal itself*
and retries at 8192 rather than accepting a partial result — a fixed ceiling can
always be exceeded, so the durable fix is detection, not a bigger number.

Verified end to end: forced to truncate at a 48-token cap, extraction escalated
and recovered all 13 facts from a dense session; at the configured budget it does
not truncate at all.

### The ChromaDB telemetry race was never actually fixed

It resurfaced mid-run as `sleep pass error: '<uuid>CollectionGetEvent0'`, failing
an entire consolidation pass. The earlier fix — `anonymized_telemetry=False` —
addressed the wrong layer: that setting only sets `posthog.disabled = True`,
which suppresses the **network send**. The unsafe code still runs on every
`get()`:

```python
# chromadb/telemetry/product/posthog.py — no lock anywhere
batched_event = self.batched_events[batch_key].batch(event)
self.batched_events[batch_key] = batched_event
if batched_event.batch_size >= batched_event.max_batch_size:
    self._direct_capture(batched_event)
    del self.batched_events[batch_key]      # two threads both reach this
```

`CollectionGetEvent` batches up to 300 events, so under concurrent reads two
threads pass the size check together and the second `del` raises `KeyError`.

Reproduced deterministically: **17 failures in 1280 concurrent `get()` calls
(1.3%)** with the setting already applied, and **zero** once the capture path is
neutralised. Verified again through the real `memory_store` read paths under the
same concurrency shape `sleep_pass` uses: 60 batches, no errors.

The lesson is the one this codebase keeps re-learning — a config flag that
*sounds* like it disables a subsystem may only disable its output.

### Abstention: the guard is right, the model ignores it

`031748ae_abs` failed again. The deterministic guard **worked correctly** — checked
against the run's real retrieved sources, `unverified_terms` returned
`['Software Engineer Manager']`, exactly as designed. The model received that
warning, corrected the role to "Senior Software Engineer" in its answer, and then
**answered anyway** with a number instead of declining.

So detection is solved; compliance is not. The guard is advisory, and the model
treats a strong, directly-relevant memory ("leading a team of five engineers as a
Senior Software Engineer") as licence to answer. Making abstention reliable means
enforcing it in code rather than instructing it — with a real false-positive cost
that needs its own calibration before adopting. Note this question *passed* in an
earlier run with the same guard and model, so run-to-run store variation, not a
regression, separates the two outcomes.

### Re-told events were being superseded

Found by comparing the doctor's-appointment answer across runs: it degraded from
"1 appointment" to "0 — you have appointments *scheduled*" on the clean run.
The store explained it: the March 20 appointment — an event that genuinely
happened — was marked `is_current=0`, so the model saw it tagged
`[OLD/SUPERSEDED]` and discounted it.

The cause is a category error in reconciliation: supersession models *mutable
state* ("lives in Mumbai" → "moved to Bengaluru"), but a **completed event can
never become outdated**. When a later session re-mentions the same appointment,
the reconciler treated the re-mention as replacing the original, hiding a real
event behind an [OLD] tag. Reconciliation now refuses to supersede when both
memories are `happened` events on the same day — it links them and lets dedup
merge. Verified: the re-told appointment stays current; a genuine move between
cities still supersedes; an event still supersedes the plan it fulfils.

### Failure taxonomy across all runs (cross-run audit, no LLM calls)

Latest-run status of every question that ever failed, from
`eval/results/*.json`:

| Question | Verdict | Layer |
|---|---|---|
| `031748ae_abs` (abstention) | Guard detects correctly; model answers anyway | **Model compliance** — needs enforcement in code, has a false-positive cost |
| `00ca467f` (count appointments) | Mixed: [OLD] tag bug (fixed above) + model not counting "diagnosed by Dr. Smith" as an appointment | **Part memory (fixed), part model** |
| `0bc8ad93` (museum, "did I go alone?") | Extraction never stored the *absence* detail (went alone) | **Model extraction judgment**; last tested three runs ago |
| `09d032c9` (battery preference) | Preference not stored on one run, passed on another | **Flaky extraction**; last tested three runs ago |
| 8 other questions | Failed in early runs, pass in their latest run | Fixed by earlier work |

Loss-stage totals (latest run per question, non-abstention): **25 pass, 1
reasoning loss, 2 write-path loss** — both write-path losses are from the
old pipeline and untested since. Internal-suite blips (12/13 twice) never
reproduced; both were one-off judge/model variance.

### Open design tension: guarded recall vs. aggregation questions

With the event guard in place, the doctor's-appointment question failed a third
way: the model now says "I don't have a record of how many appointments you
attended" — over-abstaining rather than over-counting. The likely mechanism is
the guarded-recall rule itself: the prompt requires quoting exact source words
that *state* what was asked, and declining when nothing does. No memory literally
states a count; a count must be **derived** from several memories. The same
discipline that fixed the hallucination failures taxes aggregation questions.

Across three runs this question failed three different ways (over-count →
miscount via a false [OLD] tag → over-abstain). Each time the memory layer's
contribution shrank; the model's remained. Any fix means loosening quote-or-
decline for derivational questions without re-opening the abstention hole — a
deliberate, separately-tested change, not a quick edit.

### Caveat on the next benchmark

These changes alter retrieval ranking, so scores are **not** directly comparable to
the earlier 80% / 83% runs — that is a re-baseline, not a regression if it moves.
`top_k` remains at 5 and is untouched deliberately: it is a tuning knob rather
than a bug, and changing it in the same run would confound the comparison.

## Addendum: final batch-1 result with every fix active

`longmemeval-11-batch1-all-fixes.json`: **5/10 (50%)** — knowledge-update 2/5,
multi-session 3/5, abstention 0/1. Loss attribution: **store 9/9, retrieval
9/9, all four non-abstention failures are reasoning loss.** For the first time
no question was lost in the write path or retrieval — every failure happened
with the evidence in front of the model.

The failures are two enumeration questions (counting events; all pure
arithmetic questions passed), one store-baked hedge ("4 to 5 projects" — see
the merge-of-progressions finding), the abstention question (guard fired,
model answered anyway — 1-for-7 compliance across all runs), and one
world-knowledge override (model answered train-vs-taxi from real Tokyo prices
instead of the user's stored $50 delta).

Context for the 50%: this batch is deliberately the two hardest categories;
Zep+GPT-4o reports ~71% on the full mix with multi-session weakest. One
question (031748ae_abs) ingested through a ~1-minute OpenRouter outage that
dropped ~6 sessions; a store inspection confirmed the near-match evidence
survived, so its result stands.

Where the project goes from here — the gap analysis and prioritized roadmap —
lives in **`blackboard.md`**.

## Addendum: split-model runs — oracle-60 and haystack-30 (2026-08-13)

Configuration: Qwen3-30B-A3B on the memory pipeline, DeepSeek-3.2 on the answer
path (`MEMORY_CHAT_MODEL`), chosen after a five-mode reasoning gauntlet where
DeepSeek uniquely flagged a false premise in every hard-abstention trial.

**Oracle 60 (10/type): 44/60 (73%).** Four types at 80–90%; multi-session 60%;
temporal-reasoning 40% — six of ten temporal failures are date-arithmetic the
model declines to perform ("I don't have the exact dates recorded" with the
dates retrieved). Abstention subset 5/6.

**Haystack 30 (5/type): 23/30 (77%).** Haystack now out-scores oracle —
needle-finding at ~45 sessions/question is no longer the bottleneck. On the two
hard types the previous batch scored 5/10; this run scored 7/10. Abstention 2/2,
including a textbook near-match decline ("no record of a hamster; I know about
your cat Luna"). All seven failures are previously-diagnosed items; none new.

**The model-swap ledger.** Gained: abstention 7/8 across both runs (historically
~0), grounding on stored numbers. Lost: commitment on derived numbers — the
coffee-mug answer computed both candidates including the correct one and refused
to choose; three date-arithmetic questions declined outright; one verbatim-recall
regression (chess, both settings). Net: wrong-commitments became non-commitments.
Caveat: the audit judge is also DeepSeek now, and it labels derived-number
questions "not in store" more strictly — the write-path column is partly a
measurement change.

**Conclusion.** The aggregation path (counts and date-diffs computed in code,
handed to the model as committed numbers) now addresses the majority of all
remaining failures in both settings — promoted to the top of the roadmap in
`blackboard.md`.

## Addendum: the aggregation path, built and measured (2026-08-13)

`memory/aggregate.py` implements `design-aggregation.md`: count and time-span
questions are answered by a full-store scan (status/date filters in code), one
membership-only LLM call, and arithmetic in Python, delivered to the responder
as a computed basis it reports instead of derives. Internal suite 14/14,
including a new counting scenario with a booked-but-unattended trap.

Targeted retest of the 10 aggregation-shaped benchmark failures
(`longmemeval-14/-15`): **all 3 date-diffs with stored anchors flipped to
PASS** — the exact questions DeepSeek previously declined. A fourth (counting
pending obligations) exposed a modality gap — "need to pick up" counts
`planned` records, not events — fixed and verified by replaying the run's own
store (computes the gold count exactly). Where anchors were genuinely never
stored the aggregate correctly emitted nothing.

**The revealed bottleneck: extraction completeness.** Every remaining failure
traces to facts that were not stored on that particular run — the doctor
question failed both retests with *opposite* missing facts (one run stored
Dr. Thompson but not the bronchitis visit; the other stored Dr. Smith but not
Thompson). Extraction is nondeterministic run to run even at temperature 0.
With reasoning moved into code and retrieval at 9/9, P(every needed fact
survives extraction) is now the system's ceiling — promoted to the top of
`blackboard.md`, with the experience bank (re-extraction from archived raw
transcripts) as the designed lever.

## Addendum: the extraction gap check, built and verified offline (2026-08-13)

The completeness lever from `blackboard.md` item 5 is now in the write path.
After the first extraction stores its facts, `update_memory.py:
_completeness_pass` makes one more cheap-model call
(`extract_memory.py: MemoryGapCheck`) that sees the same transcript *plus the
list of facts just stored* and outputs only what is missing. The intuition:
extraction is nondeterministic — the same session yields different fact subsets
run to run — but a pass conditioned on the first pass's output is sampling from
a different distribution, one centred on the gaps.

Two guards keep it safe. Re-emitted paraphrases (the model restates stored
facts despite instructions) are dropped before writing by embedding similarity
at 0.9 — the same threshold the sleep pass uses to merge duplicates, so
anything the guard admits would have survived dedup anyway. And the pass is
best-effort: any failure logs a warning and leaves the first pass's writes
untouched.

**Verification, fully offline** (no benchmark spend): replayed the archived
experience-bank transcripts of the doctor question (`00ca467f` — failed both
aggregation retests with opposite missing facts) against a copy of that run's
store. The gap check ran on the three doctor-adjacent sessions, recovered 11
facts including the one that mattered — Dr. Thompson's March 20 follow-up —
and the aggregation path then computed the gold answer: count = 2, where the
live run had answered 1. The membership call also correctly collapsed a
near-duplicate Dr. Smith record the similarity guard had admitted, so the
layered defenses (insert-time guard, sleep-pass dedup, membership dedup) each
caught what the previous layer let through.

Internal suite with the pass active on every session: **14/14**
(`internal-14-gap-check.json`). Cost: one extra Qwen-30B call per session,
roughly doubling write-path spend — still pennies per benchmark question.

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
