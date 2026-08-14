# Building a Memory Layer, and Measuring It on LongMemEval

**Last updated:** 2026-08-14
**Benchmark:** LongMemEval (arXiv 2410.10813), both the oracle and haystack settings
**Models:** Qwen3-30B-A3B for all memory operations, DeepSeek-v3.2 for answering and judging
**Final scores:** 26/30 (87%) on oracle, 26/30 (87%) on haystack

---

## 1. The short version

We built a long-term memory layer for a chatbot — the component that decides what to
remember from a conversation, how to store it, and what to pull back when a later
question needs it. Then we measured it on LongMemEval, a public benchmark that hides
questions inside long multi-session chat histories.

The system started at 63% and finished at 87% on the same 30-question oracle sample,
and reached the same 87% on the harder haystack setting where each question is buried
in roughly 45 sessions of unrelated conversation.

The useful part of that number is not the number. It is what the intermediate
measurements revealed about *where* a memory system loses information. We instrumented
every run to answer two questions per benchmark item: was the answer ever written into
the store, and did retrieval actually surface it? That splits every failure into one of
three buckets — the write path lost it, retrieval buried it, or the model had it and
still got the answer wrong. Almost every improvement in this project came from reading
that split rather than from tuning a score.

Three findings shaped the final system:

1. **Most early losses were in the write path, not retrieval.** The intuition that a
   memory system lives or dies on search quality was wrong here. Facts were being
   dropped at extraction time, silently truncated mid-response, or corrupted by
   consolidation logic that turned plans into events.
2. **Reasoning that can be done in code should not be handed to the model.** Counting
   and date arithmetic failed persistently across every model we tried, in two opposite
   ways (one model committed to wrong numbers, the other refused to commit at all).
   Moving the arithmetic into Python fixed a whole category outright.
3. **Some problems are model-selection problems wearing an engineering costume.**
   Abstention — declining to answer when the memory genuinely is not there — resisted
   seven rounds of prompt engineering and one deterministic guard, then largely resolved
   itself when we changed which model writes the final answer.

---

## 2. What we tested

### 2.1 The system under test

A Python memory layer on top of ChromaDB (a local vector database — it stores text
alongside numeric embeddings so you can search by meaning rather than by exact words).
The pipeline has four parts.

**The write path.** After a conversation session, one model call reads the transcript
and extracts atomic facts. Each fact carries structured metadata: an importance score
from 1 to 10, a date resolved to an absolute value (the extractor turns "last month"
into a real date using the session's date, not today's), search keywords, a one-sentence
context, a `status` field, and an `about_user` flag. Facts are written append-only —
the hot path makes no judgment about whether a fact contradicts an existing one.

A second cheap call then runs a **completeness pass**: it sees the same transcript
*plus the list of facts just stored* and outputs only what the first pass missed. The
reasoning behind it is in §5.1.

**Consolidation, run at session end rather than inline.** This is the "sleep pass"
(the pattern comes from Letta): all the expensive judgment happens here, with the whole
session in view, and it holds exclusive authority to modify existing records. It does
reconciliation (deciding whether a new fact supersedes an old one), deduplication,
memory evolution (updating the context sentences of linked neighbours, from the A-Mem
design), reflection (deriving higher-level insights when enough important facts
accumulate), and a rewrite of the always-in-prompt core profile (the MemGPT pattern —
a short summary of the user that is included in every prompt regardless of retrieval).

**Retrieval.** Three independent searches run in parallel and their rankings are fused:
vector similarity, BM25 keyword matching, and Personalized PageRank over a graph of
links between related memories. The fused ranking is then re-scored, and the weights
matter enough that they are worth stating explicitly (`memory/memory_store.py:20-30`):
relevance leads, with importance weighted 0.15, Ebbinghaus retention weighted 0.10, and
a 0.30 penalty applied to superseded records. Retention here means a forgetting curve —
a memory's strength decays as `e^(−days/strength)` and the strength grows each time the
memory is recalled. There is a minimum true cosine similarity of 0.10 to enter the
ranking at all. The top 5 memories go to the answer prompt.

Alongside distilled memories, raw conversational turns are archived verbatim to an
**experience bank** and searched separately, so questions about what the assistant
itself said in a past conversation can be answered from the record.

**The aggregation path.** Questions asking for a number that no single memory states —
"how many appointments in March", "how long have I been doing X" — bypass similarity
search entirely and are answered by scanning the store and computing in Python. The
design is in §5.4.

Old memories are soft-deleted (a flag, `is_current=0`), never physically removed, so
the history of a changing fact stays queryable.

### 2.2 The benchmark

LongMemEval embeds a question inside a synthetic chat history spread over many sessions
and tests five distinct memory abilities. We ran both of its settings:

- **Oracle** supplies only the sessions that actually contain the evidence. This
  measures whether the pipeline can extract, update, and reason over what it saw, with
  no needle-in-haystack difficulty. Median around 2 to 5 sessions per question.
- **Haystack** (`_s`) supplies the full distractor set — a median of around 45 sessions
  and hundreds of turns per question, roughly 115k tokens of history. This adds the
  problem of finding the evidence among competing, superficially similar sessions.

The six question types, and what each one actually probes:

| Type | What it tests |
|---|---|
| single-session-user | A fact the user stated once. The baseline case. |
| single-session-assistant | Something the *assistant* said. Fails entirely unless assistant turns are stored. |
| single-session-preference | An implicit preference that should shape a later answer, never asked about directly. |
| multi-session | Combining or counting facts across several sessions. |
| knowledge-update | A fact that changed. Requires knowing which version is current. |
| temporal-reasoning | Dates, durations, and ordering. |

A subset of questions are **abstention** variants (their ids end in `_abs`). These ask
about something that was never discussed, often while presupposing it is true. They
count as correct only when the system admits it does not know.

We sampled 5 questions per type, deterministically (sorted by question id, first N per
type), giving 30 questions per run. One run took a larger sample of 10 per type (60
questions) to check that the smaller sample was not misleading us.

### 2.3 The internal suite

Alongside the benchmark we keep a hand-written suite of 14 scenarios
(`eval/scenarios.py`) covering single-hop recall, multi-hop reasoning, knowledge
updates, temporal questions, exact-token recall, abstention, and counting. It costs
pennies to run and catches regressions in minutes rather than hours. It exists because
benchmark runs are slow and expensive, and because a benchmark score tells you *that*
something regressed but not *what*. Its limitation is that it saturated at full marks
early — which is precisely why LongMemEval was brought in.

---

## 3. How we tested

### 3.1 The run

`eval/run_longmemeval.py` runs each question end to end through the real production
code path — no test doubles, no shortcuts around the memory layer:

1. **Ingest.** Every session in the question's history is written through the real
   write path, carrying its historical date so relative dates resolve correctly.
   Sessions are sorted oldest-first before ingestion (see §5.2 — this was a bug fix,
   not an incidental detail). Sessions ingest concurrently in batches because
   extraction is append-only and order-independent; one sleep pass then reconciles
   each batch, with the expensive core-profile rewrite deferred to the final batch.
2. **Archive.** Each user/assistant exchange is written to the experience bank so
   turn-level search can find it later.
3. **Ask.** The question is asked as of its question date, through the same retrieval
   and response path the chatbot uses.
4. **Judge.** An LLM judge compares the answer to the gold answer, using an
   abstention-aware rubric so that "I don't know" is scored correctly for `_abs`
   questions and incorrectly everywhere else.

Every question runs against a throwaway database in a temporary directory under its own
user id, so runs cannot contaminate each other or any real memories. Results are written
after every question, because these runs take hours and get interrupted.

### 3.2 The instrument that mattered: loss attribution

Scoring alone tells you nothing actionable. So each non-abstention question also runs
two audits, each a separate LLM call asking "is the gold answer derivable from this
material?":

- **Store audit** — given *everything* the write path stored (all memory records, the
  core profile, and the full transcript archive), is the answer derivable?
- **Retrieval audit** — given only what retrieval actually surfaced for this question
  (top-5 memories, matched transcript turns, core profile), is the answer derivable?

From the two answers plus correctness, every failure gets a stage
(`eval/run_longmemeval.py:245-254`):

| Stage | Meaning | Whose problem |
|---|---|---|
| write-path loss | The answer was never stored | Extraction or consolidation |
| retrieval loss | Stored, but not surfaced | Ranking, filtering, search |
| reasoning loss | Surfaced, and still answered wrong | The answering model |

This is the single most valuable thing we built for this project. It converts "the
score went down" into "extraction dropped three facts", which is a fixable statement.
Its limits are documented honestly in §6.1 — it is a noisy instrument at haystack
scale, and we can prove it.

---

## 4. Results

### 4.1 Final runs — everything active

Both runs used the full stack for the first time together: split models, the
aggregation path, the extraction completeness pass, and every correctness fix in §5.

| Question type | Oracle 30 | Haystack 30 |
|---|---:|---:|
| single-session-user | 5/5 | 4/5 |
| single-session-assistant | 4/5 | 5/5 |
| single-session-preference | 4/5 | 4/5 |
| multi-session | 3/5 | 3/5 |
| knowledge-update | 5/5 | 5/5 |
| temporal-reasoning | 5/5 | 5/5 |
| *(abstention subset)* | *2/2* | *2/2* |
| **Total** | **26/30 (87%)** | **26/30 (87%)** |

Files: `eval/results/longmemeval-16-oracle30-gap-check.json` and
`longmemeval-17-haystack30-gap-check-MERGED.json`.

Two results here are worth more than the headline:

**Haystack matched oracle.** Finding the needle among 45 sessions of distractors is no
longer what limits this system. Earlier in the project, haystack scored well below
oracle; the retrieval fixes in §5.3 closed that gap. Whatever is still failing fails
for reasons unrelated to search difficulty.

**Temporal reasoning went from the worst category to perfect.** On the 60-question
oracle run it scored 4/10, and six of those failures were the model declining to do
date arithmetic it had the dates for. The aggregation path erased that category of
failure completely — 5/5 in both final runs.

The aggregation path fired on 10 of 30 oracle questions and 7 of 30 haystack questions.
It emits nothing when it cannot compute an answer, so the remaining questions took the
normal path unchanged.

### 4.2 Trajectory

| Run | Setting | Score | What changed |
|---|---|---:|---|
| `longmemeval-01` | Oracle 30 | 19/30 (63%) | Baseline: per-fact write path, temperature 0.7 |
| `longmemeval-03` | Oracle 30 | 24/30 (80%) | Thin write path, sleep pass, transcript recall, temperature 0 |
| `longmemeval-08` | Haystack 6 | 5/6 | First haystack pilot (tiny sample) |
| `longmemeval-11` | Haystack 10 | 5/10 (50%) | Two hardest types only, after the correctness audit |
| `longmemeval-12` | Oracle 60 | 44/60 (73%) | Split models (Qwen memory, DeepSeek answers) |
| `longmemeval-13` | Haystack 30 | 23/30 (77%) | Same configuration, haystack |
| `longmemeval-16` | Oracle 30 | **26/30 (87%)** | Aggregation path + completeness pass |
| `longmemeval-17` | Haystack 30 | **26/30 (87%)** | Same |

The dip at 73% is not a regression — it is a 60-question sample of a harder mix, and the
50% run deliberately tested only the two weakest categories. Scores are only comparable
within the same sample and setting. The full run-by-run index, including the internal
suite, lives in `eval/results/README.md`.

Abstention deserves its own line because it moved the most: historically near zero, then
7 out of 8 across the split-model runs, then 4 out of 4 in the two final runs.

### 4.3 The remaining failures, all four of them

Eight failing instances across the two runs, covering six distinct questions. Every one
is diagnosed; no new failure mode appeared in either final run.

| Question | Setting | Stage | What happened |
|---|---|---|---|
| Doctor's appointments in March (`00ca467f`) | Both | write-path | Gold is 2. Oracle answered 1 (had Dr. Thompson, missed the Dr. Smith visit); haystack answered 1 (had Smith, missed Thompson). *Opposite* facts missing on the two runs — this is extraction variance, §6.2. |
| Battery-life preference (`09d032c9`) | Both | write-path | The user had earlier mentioned buying a power bank; the correct behaviour is to build on that. It was never stored on either run. Same variance. |
| Coffee mug price (`0100672e`) | Haystack | write-path | Gold is $12. The system had the $60 total but not the count of mugs, so it correctly declined to divide. |
| Leadership percentage (`099778bb`) | Oracle | retrieval | The store held "20 leadership positions"; the model needed the total to compute a percentage and asked for it. Stored but not surfaced. |
| Chess move (`1568498a`) | Oracle | reasoning | Gold is "28. Kg3". Answered "28. Kg3 Be6" — correct move plus an extra ply, judged wrong. Evidence fully delivered. |
| Vintage cameras (`15745da0`) | Haystack | reasoning | Gold is three months. Answered "about 16 days" by anchoring on the most recent camera rather than the first. Evidence fully delivered. |

The pattern: two questions fail because of extraction variance, one because a needed
component genuinely was not stated, one because retrieval buried a fact, and two because
the model mishandled evidence it had in front of it. Notably, the same two questions
(doctor and battery) fail in *both* settings, which tells us the difficulty is in
reading the source conversation, not in the size of the haystack.

---

## 5. What we understood, and what we changed

This section is organised by cause rather than by date, because the chronology is
misleading — several fixes were made before we understood why they worked.

### 5.1 The write path was losing facts, in five distinct ways

This was the largest and most surprising source of loss. We had assumed retrieval would
be the bottleneck.

**Extraction was being silently truncated.** Found in the logs of a haystack run, which
emitted `LM response was truncated due to exceeding max_tokens=2048` twice while
ingesting 47 sessions. DSPy detects this condition (the API reports `finish_reason ==
"length"`) but only *warns* — so the write path stored a partially extracted session as
though it were complete, losing every fact after the cut.

The cap was then set from measurement rather than raised by feel: one extracted fact
serialises to roughly 88 output tokens, and benchmark sessions run to a median of 14,000
characters (28,000 at the tail), so a fact-dense session yields 30 or more facts and
needs somewhere between 2,600 and 3,500 tokens. The 2048 ceiling was below what a normal
session requires. It is now 4096, and — more importantly — extraction *reads the
truncation signal itself* and retries at 8192 rather than accepting a partial result
(`memory/llm.py:45-56`). A fixed ceiling can always be exceeded, so the durable fix is
detection, not a bigger number. Verified end to end: forced to truncate at a 48-token
cap, extraction escalated and recovered all 13 facts from a dense session.

**Pre-1970 dates crashed the whole session's write.** `datetime.timestamp()` raises
`[Errno 22]` on Windows for dates before 1970, so a single childhood memory killed
every write in that session. `to_epoch()` now subtracts from the epoch instead.

**A ChromaDB telemetry race dropped sessions — and our first fix was wrong.** Concurrent
ingestion was losing around 14% of sessions to an opaque event-key error. We set
`anonymized_telemetry=False` and considered it fixed. It resurfaced mid-run and failed
an entire consolidation pass. The setting only sets `posthog.disabled = True`, which
suppresses the network *send*; the unsafe code still runs on every read:

```python
# chromadb/telemetry/product/posthog.py — no lock anywhere
batched_event = self.batched_events[batch_key].batch(event)
self.batched_events[batch_key] = batched_event
if batched_event.batch_size >= batched_event.max_batch_size:
    self._direct_capture(batched_event)
    del self.batched_events[batch_key]      # two threads both reach this
```

Under concurrent reads two threads pass the size check together and the second delete
raises `KeyError`. Reproduced deterministically: 17 failures in 1280 concurrent `get()`
calls (1.3%) with the setting already applied, and zero once the capture path is
neutralised. Verified again through the real read paths under the same concurrency shape
the sleep pass uses. The lesson generalises: **a config flag that sounds like it disables
a subsystem may only disable its output.**

**Assistant turns were never stored at all.** In the baseline the extractor produced
facts about the *user* only, so any question about what the assistant had said failed by
construction — the worst category at 40%. Raw exchanges are now archived to the
experience bank and searched with BM25 alongside memories. That surfaced a second bug:
BM25's inverse-document-frequency term goes negative when a word appears in most of a
small corpus, so the `score > 0` filter discarded every candidate on short histories.
Turn search now ranks by BM25 but gates on non-stopword overlap. A third problem sat on
top of both: excerpts shared the same prompt field as memories, so the model treated
verbatim transcript as a vague recollection and answered "I don't have the list" while
the list sat in its context. They now occupy a separate `past_conversations` field
framed as a record to quote from. Single-session-assistant went from 0/3 to 3/3 on the
targeted retest, and 5/5 on the final haystack run.

**Extraction is nondeterministic even at temperature 0 — and that is now the ceiling.**
This was the last thing we understood and the most important. The doctor question failed
two separate retests with *opposite* facts missing: one run stored Dr. Thompson but not
the bronchitis visit, the next stored Dr. Smith but not Thompson. Same code, same input,
same temperature.

The fix is a **completeness pass** (`memory/update_memory.py: _completeness_pass`,
calling `extract_memory.py: MemoryGapCheck`). After the first extraction stores its
facts, one more cheap call sees the same transcript plus the list of facts just stored,
and outputs only what is missing. The intuition: a second independent sample would just
re-roll the same lottery, but a pass *conditioned on the first pass's output* samples
from a different distribution — one centred on the gaps.

Two guards keep it safe. Re-emitted paraphrases (the model restates a stored fact
despite instructions) are dropped before writing by embedding similarity at 0.9, which
is the same threshold the sleep pass uses to merge duplicates — so anything the guard
admits would have survived deduplication anyway. And the pass is best-effort: any
failure logs a warning and leaves the first pass's writes untouched.

Verified fully offline, with no benchmark spend, by replaying the archived transcripts
of the failing doctor question against a copy of that run's store. The gap check ran on
the three doctor-adjacent sessions, recovered 11 facts including the one that mattered
(Dr. Thompson's March 20 follow-up), and the aggregation path then computed the gold
answer of 2, where the live run had answered 1. Cost: one extra cheap-model call per
session, roughly doubling write-path spend, which is still pennies per question.

It narrows the hole. It does not close it — the doctor and battery questions still fail,
and the probability that every needed fact survives extraction remains a probability
rather than a guarantee.

### 5.2 The store was quietly lying about what happened

A separate class of bug: the facts were stored, but consolidation corrupted their
meaning. These are worse than losses, because the system answers confidently from
corrupted state.

**Intentions were being recorded as events.** A haystack question asked how many
doctor's appointments the user attended in March; the answer was 3 against a gold of 2.
The cause was not counting — the history mixes attended, scheduled, and merely-considered
appointments ("I'm *considering* scheduling with Dr. Patel", "I'm *scheduled* for an EMG
on April 1st"), and extraction flattened all of them into completed facts. The count was
wrong before any counting happened.

The fix is a `status` field on every memory (`happened`, `planned`, `considered`,
`ongoing`), asserted in the fact text as well ("User is considering X"), persisted, and
surfaced at answer time as a tag like `[PLANNED, did not happen]`. Beyond benchmarks,
the old behaviour told users they had attended appointments they had only thought about.

**Deduplication then undid that guarantee.** The merged record was built without
`status`, `keywords`, `context`, or `links`. Because `status` defaults to `happened`,
merging a planned appointment into a duplicate turned an intention back into an event.
Deduplication now clusters within `(kind, status)`, so an intention can never merge with
an event, and the merged record inherits the missing fields.

**Completed events were being superseded by re-tellings.** Supersession is designed for
*mutable state* — "lives in Mumbai" is replaced by "moved to Bengaluru". But a completed
event can never become outdated. When a later session re-mentioned the March 20
appointment, the reconciler treated the re-mention as replacing the original, so a real
event was hidden behind an `[OLD/SUPERSEDED]` tag and the model discounted it.
Reconciliation now refuses to supersede when both memories are `happened` events on the
same day; it links them and lets deduplication merge. Verified three ways: the re-told
appointment stays current, a genuine move between cities still supersedes, and an event
still supersedes the plan it fulfils.

**Sessions were being ingested out of chronological order.** 211 of the 500 haystack
questions (42%) list their sessions out of date order, with a median displacement of 15
positions. Since the system treats later-ingested as newer, reconciliation was
superseding *newer* facts with older ones — attacking exactly the knowledge-update and
temporal-reasoning categories. Fixed on both sides: the harness ingests oldest-first,
and reconciliation refuses any supersession where the incoming memory is dated earlier
than the one it would replace, linking them instead. The second half matters for real
users too, who mention past facts in present conversations.

**A growing quantity was being merged into a hedge.** The store literally contained
"User has completed **4 to 5** painting projects" — deduplication had clustered
"completed 4 projects" (older) with "completed 5 projects" (newer) and the merge
faithfully preserved both. A quantity that grows over time is a knowledge update, not a
duplicate. Conflicting numbers in a cluster now resolve to the newest statement with the
history linked, with no LLM call; equal-number and numberless duplicates still merge.

**General knowledge was polluting the store.** The appointment question's top retrieved
memories included Wudhu rules, George Washington's birth year, and hiking-gear advice —
assistant answers stored as if they were facts about the user, eating retrieval slots the
real evidence needed. Extraction now emits an `about_user` flag per fact and the write
path filters on it. Verified directly: the Washington trivia and gear advice are dropped,
the user's own appointment and trip are kept.

**Reconciliation could not see the fact it needed to.** It compared each session against
the 40 *most recent* existing memories. On a haystack of hundreds, the older fact that a
new one contradicts was never in that window, so nothing beyond the last 40 could ever
be superseded. It now selects the 40 most semantically *relevant* memories instead.

### 5.3 Retrieval was ranking on the wrong signal, and its filter never filtered

**The relevance floor was inert.** The code read ChromaDB's cosine distance as if it
spanned 0 to 2 in angular terms and computed `1 − distance/2`. ChromaDB's cosine
distance is `1 − similarity`, so a completely unrelated memory (true similarity 0.0)
scored 0.5 and sailed past the 0.3 floor. The effective cutoff was a similarity of −0.4:
nothing was ever excluded.

Fixing the formula made the floor bite, so the threshold was **calibrated rather than
guessed**, against real question and memory pairs:

| Floor | Relevant kept | Irrelevant admitted |
|---|---|---|
| 0.00 | 12/12 | 47/72 |
| **0.10** | **12/12** | **9/72** |
| 0.25 | 9/12 | 1/72 |
| 0.30 | 5/12 | 1/72 |

The originally intended 0.30 would have discarded 7 of 12 genuinely relevant memories.
0.10 removes 81% of the noise at no cost to recall. A question and the memory that
answers it can sit as low as 0.15 — "what am I allergic to?" against "user cannot eat
shellfish" — which are lexically disjoint, so BM25 would not have rescued that one
either.

**Importance was outranking relevance.** The re-ranking formula summed relevance,
retention, and importance with equal weight. We measured the span each term could
actually move a score: relevance 0.71, importance 0.90, and retention sat at
approximately 1.0 for *every* memory, because they had all just been written. So an
important-but-unrelated memory could outrank an exact match, and a superseded fact could
tie with the fact that replaced it. Relevance now leads, with importance and retention
demoted to tiebreakers at 0.15 and 0.10, and a 0.30 penalty for superseded records.

**Smaller confirmed defects, each verified by a probe rather than by inspection:**

| Bug | Effect |
|---|---|
| PageRank ranked *every* node, including unreachable ones scoring ~1e-6 | Unrelated memories received rank-fusion credit |
| `add_links` released its lock between read and write | Concurrent link additions silently lost one |
| `max(session_ids)` used string ordering | `lme-0-9` was treated as newer than `lme-0-11` |
| The prompt was built *inside* reconcile's `try` block | A coding error would have been swallowed as "reconcile call failed", disabling reconciliation with no visible sign |
| The vector query fetched a fixed 30 candidates before filtering | A store full of superseded memories returned almost nothing |

One hypothesis in this area was tested and **disproved**: we suspected BM25 was dropping
negative scores inside `memory_store` as well, but the `rank_bm25` library floors
negative inverse-document-frequency internally, so that code was fine. Worth recording —
a probe that kills a hypothesis is as useful as one that confirms it.

### 5.4 Arithmetic belongs in code, not in the model

The most valuable single change, and the one that generalises furthest.

**The problem, stated precisely.** Questions asking for a number that no memory states
failed through two different models in two characteristic ways. Qwen committed to wrong
numbers (counting a plan as a visit). DeepSeek refused to commit — "I don't have the
exact dates recorded", with the dates sitting in its context, and on one question it
computed both candidate answers including the correct one and then declined to choose.

The root cause is structural rather than model-specific. Two jobs were being fused into
a single generation: *deciding which memories belong to the category*, and *doing
arithmetic over them*. Similarity search compounds it, because top-k retrieval cannot
promise it surfaced **all** members of a category — and a model that suspects its list
is incomplete is right to hedge.

**The design in one sentence:** detect number-deriving questions cheaply, collect
candidates by scanning the whole store instead of searching it, let one small LLM call
do membership judgment only, do the arithmetic in Python, and hand the responder a
computed answer that it reports rather than derives. Decomposition is the whole trick —
classify-then-count replaces classify-and-count, and each piece goes to the component
that is reliable at it. Implemented in `memory/aggregate.py`.

The four stages:

1. **Trigger — deterministic and free.** A small family of regular expressions on the
   question: "how many *noun*" means COUNT; "how long", "days between", "did it take"
   mean DATE-DIFF. False positives are harmless by construction, because if the later
   stages find nothing the aggregate is empty and the system behaves exactly as before.
   The trigger only decides whether to *try*.
2. **Collect — a scan, not a search.** Fetch all current records for the user (a few
   hundred even at haystack scale, all local, no API cost), then filter deterministically
   in code: by `status`, so "did I go to" counts only `happened` records and the April 1
   EMG can never be a candidate; and by date window, so "in March" becomes a range filter
   on the stored ISO date. Year ambiguity resolves by a fixed rule, and if ambiguity
   survives the path emits nothing rather than guessing a window.
3. **Select — one small call, membership only.** The surviving candidates (typically 40
   or fewer) go to a single call asking which of them describe the category in question,
   answered by index. This is where "diagnosed with bronchitis by Dr. Smith at a clinic
   visit" gets recognised as a doctor's appointment. Per-item classification is something
   small models do reliably; it was the *simultaneous* classify-and-count they failed.
   For date questions the same call labels the start and end anchors instead, and code
   reads their stored dates — which are already absolute, because extraction resolves
   relative dates at write time.
4. **Compute and deliver.** Python does the arithmetic and the responder receives a
   separate `computed_aggregate` field with the members listed, the exclusions named, and
   the answer basis stated, plus one instruction: this was produced by scanning all
   memories and computing in code, so state its number. The model no longer chooses a
   number; it reports one. The member list keeps the result explainable and auditable.

Scope was kept deliberately narrow: COUNT and DATE-DIFF only. Those two cover nine of
the observed failures. General amount arithmetic (the coffee-mug division, which needs a
total and a divisor labelled from different memories) was left out as a separate failure
family with exactly one observed instance. We explicitly did not build a general query
engine over memories or a learned question classifier.

**Measured result.** All three date-difference questions with stored anchors flipped from
fail to pass — the exact questions DeepSeek had previously declined. A fourth question
exposed a modality gap ("need to pick up" counts `planned` records, not events), which
was fixed and verified by replaying the run's own store, where it computes the gold count
exactly. Temporal reasoning went from 4/10 on the oracle-60 run to 5/5 on both final
runs. Where anchors were genuinely never stored, the aggregate correctly emitted nothing.

### 5.5 Abstention: an engineering problem that turned out to be a model problem

One question runs through this entire project. Asked "how many engineers do I lead in my
new role as **Software Engineer Manager**" when memory only records **Senior Software
Engineer**, the system should decline. The question also *presupposes* the role, and
models tend to accept user framing.

What we tried, in order, and what each attempt taught us:

1. **A guarded-recall prompt clause** — answer only from what a memory actually states.
   Failed.
2. **A `supporting_evidence` output field** forcing the model to quote its source before
   answering. Failed: the model quoted a near-match and answered anyway.
3. **A deterministic guard** (`memory/grounding.py`) that extracts distinctive noun
   phrases from the question and checks programmatically whether they appear in any
   retrieved source. This *worked as designed* — it returned
   `unverified_terms = ['Software Engineer Manager']`, exactly right. The model received
   the warning, corrected the role in its answer, and then **answered anyway** with a
   number. Detection was solved; compliance was not. Final tally across all runs: the
   guard detected the fabricated premise 7 times out of 7, and the model obeyed it 1 time
   out of 7.

   Building the guard did surface a real bug worth keeping: its text normaliser turned
   punctuation into spaces without collapsing runs, so a source saying "Dr. Lee" became
   `dr__lee` while the question yielded `dr lee`, and no match was found. The system
   abstained on information it actually held. Normalisation is now consistent, and
   phrases are padded so they match whole words only — "New York" is no longer satisfied
   by "New Yorker".

4. **Changing the answering model.** DeepSeek-v3.2 on the answer path, chosen after a
   five-mode reasoning gauntlet in which it uniquely flagged the false premise in every
   hard-abstention trial. Abstention went from historically near-zero to 7/8 across the
   two split-model runs and 4/4 in the final pair, including a textbook near-match
   decline: "no record of a hamster; I know about your cat Luna."

The conclusion we drew: three rounds of instruction could not make a model verify a match
when *verification itself* was the broken faculty. Code-enforced abstention (declining in
the response layer without asking the model) remains designed but unbuilt, shelved
because it carries a real false-positive cost that would need its own calibration and
the model swap made it unnecessary. For industry context, the MemConflict benchmark
found the best conflict-recognition score across six production memory systems was 0.25.

### 5.6 The model-swap ledger, and what it cost

Splitting the models — Qwen3-30B-A3B for the hundreds of memory operations, DeepSeek-v3.2
for the single answer call per question — is cheap by construction, because the answer
path is one call against hundreds of ingestion calls.

It was not free of regressions, and the ledger is worth stating honestly:

- **Gained:** abstention (near-zero to 7/8), and much better grounding on numbers that
  are actually stored.
- **Lost:** commitment on *derived* numbers. The coffee-mug answer computed both
  candidates including the correct one and refused to choose; three date-arithmetic
  questions were declined outright. In effect, wrong commitments became non-commitments.
- **Then recovered:** the aggregation path in §5.4 was built specifically to close that
  new gap, and did.

### 5.7 Things we tried that did not work

Recorded so they are not retried:

- **Compose-on-read** (generating a query-tailored digest of memories at read time): no
  accuracy gain, added latency, and blurred fact attribution in one case. Disabled.
- **Personalized PageRank**: no measurable accuracy difference, but it costs no LLM
  calls, so it stayed on.
- **gpt-oss-120b and gpt-oss-20b** as cheaper extraction models: 120b was 2.5 times
  slower than Qwen despite roughly half the sticker price, and 20b failed to produce
  parseable output at all. Both are reasoning models — they spend the token budget on
  chain-of-thought before emitting structured output, hit the token cap, and get
  truncated. OpenRouter bills reasoning tokens as completion tokens, so the cheaper price
  is partly illusory. **Cheaper per token is not cheaper for this workload.**
- **Trimming the extraction schema** for speed: caused a silent quality regression the
  internal suite caught — "pilot with Indigo" became "pilot", because the terseness
  instruction had bled from the search fields into the fact text. Fixed by scoping
  terseness explicitly and requiring names, employers, and numbers to survive. Timing
  measurements would never have caught this; the regression suite did.

### 5.8 Performance work

Profiling one full haystack question (51 sessions) showed it was essentially **100% LLM
latency** — all local work, including embeddings, ChromaDB, and an O(n²) deduplication
comparison, totalled about 75 seconds out of 1107. The O(n²) loop specifically was 2.3
seconds. The lesson is the standard one, learned concretely: optimise what you measured,
not what looks expensive.

| Stage | Calls | LLM seconds |
|---|---:|---:|
| Extraction | 51 | 1225 (69%) |
| Core profile refresh | 11 | 400 (23%) |
| Reconcile | 11 | 118 |
| Reflect / evolve / dedup | 33 | 112 |

Two pure-scheduling changes cut per-question time from 18.4 minutes to 9.4 (a 49%
reduction): the core profile is refreshed **once** at the end rather than after every
batch (it was rebuilding an 80-word profile eleven times and discarding ten), and the
independent sleep-pass stages now run concurrently. Separately, deduplication clustering
was vectorised into a single matrix multiplication (roughly 490 times faster on that
stage), the write path was batched into one embedding call and one upsert instead of one
round trip per fact, and the three consolidation stages that never used embeddings
stopped fetching them.

**A caveat on all timing claims here.** A full run was launched and abandoned at 4 of 30
questions when elapsed time implied about 34 minutes per question, against 9.4 measured
hours earlier on identically-sized questions. A single extraction call re-timed at 38.3
seconds against an 18.2-second baseline — same code, same input, roughly twice as slow
upstream. The shared provider pool degrades under sustained load, and it returned a rate
limit notice earlier the same day. **Any timing number in this report is only valid
against a baseline measured in the same session.**

---

## 6. What the measurements can and cannot tell us

Being straight about the instrument matters more than the headline score.

### 6.1 The store audit is unreliable at haystack scale, and we can prove it

The store audit works by dumping everything the write path stored into one LLM call and
asking whether the gold answer is derivable. At oracle scale that is a manageable
context. At haystack scale it is hundreds of records, and the audit starts missing
things that are there.

The proof is internal to the data. Retrieval surfaces a strict subset of what the store
holds, so it is logically impossible for the retrieval audit to find an answer the store
audit could not. In the final haystack run that happened on **6 of 28 questions**. In the
oracle run it happened once.

The consequence is concrete: the store coverage figures below understate the haystack
write path, and haystack failures labelled "write-path loss" are partly mislabelled
retrieval or judgment calls. We report the numbers with the caveat attached rather than
quietly dropping them.

| Run | Answer present in store | Survived retrieval |
|---|---|---|
| Oracle 30 | 22/28 | 20/28 |
| Haystack 30 | 15/28 *(understated — see above)* | 20/28 |

### 6.2 Run-to-run variance is real and was once 40% of our failure rate

Early in the project all LLM calls ran at temperature 0.7. A re-run of the 10 failures
from the baseline showed **4 of them passed on the second attempt** with no code change.
Forty percent of what looked like systematic error was noise. Everything now runs at
temperature 0.0 (`memory/llm.py:15`).

That removed the variance we could remove. What remains, documented in §5.1, is that
extraction still produces different fact subsets run to run at temperature 0. Any
single-run comparison of two configurations on 30 questions carries an error bar of at
least a couple of questions, and we treat differences of one or two as noise.

### 6.3 These scores are not comparable to published leaderboard numbers

Published LongMemEval results in the 80 to 95% range — full-context GPT-4o around 60%,
Zep with GPT-4o around 71%, Emergence around 86% — differ from ours in ways that all cut
the same direction:

- They run all 500 questions; we run a stratified 30. Our error bar is large.
- They use frontier models, often with cross-encoder rerankers; we use a 30B open-weights
  model for memory work.
- Some vendor-reported figures use their own harness and judge.

The honest reading of our 87% is *not* "better than Zep". It is that the per-category
shape is sensible, the failure modes are diagnosed rather than mysterious, and the score
moved for reasons we can point at.

### 6.4 The judge shares a model family with the answerer

DeepSeek writes the answers and also judges them, and it runs the derivability audits.
This is a known weakness in the setup. We saw it bite in one specific way: the audit
judge labels derived-number questions "not in store" more strictly than the previous
judge did, so part of the write-path column moving is a *measurement* change rather than
a system change. An independent judge model is the obvious next improvement to the
harness.

---

## 7. What is still open

Ranked by how much they would move the result.

1. **Extraction completeness is now the system's ceiling.** With reasoning moved into
   code and retrieval performing well, the probability that every needed fact survives
   extraction is what limits the score. The completeness pass narrows this but does not
   close it — the doctor and battery questions still fail in both settings. The designed
   next lever is re-extraction from the archived raw transcripts in the experience bank,
   targeted at questions the system could not answer.
2. **Guarded recall taxes derivational questions.** The rule that fixed the hallucination
   failures — quote an exact source or decline — makes aggregation questions harder,
   because no memory literally *states* a count. We watched the doctor question fail
   three different ways across three runs (over-counting, then miscounting because of a
   false `[OLD]` tag, then over-abstaining) with the memory layer's contribution
   shrinking each time. Loosening quote-or-decline for derived questions without
   reopening the abstention hole is a deliberate, separately-tested change.
3. **Amount arithmetic** (the coffee-mug case) is the one aggregation family deliberately
   left unbuilt. It needs the selection call to label roles — which number is the total,
   which is the divisor — rather than just membership.
4. **An independent judge model**, per §6.4.
5. **Code-enforced abstention** stays designed and shelved. It should only be revived if
   the model swap's gains regress, and it needs false-positive calibration on
   non-abstention questions first.

Explicitly **not** worth chasing: more retrieval tuning (it is not the failing stage);
prompt-level abstention fixes (seven runs of evidence say no); and the benchmark headline
itself, since the remaining gap is concentrated in extraction variance and model
judgment, neither of which responds to polish.

---

## 8. Reproducing this

```bash
# LongMemEval, oracle setting, 30 questions (5 per type)
python eval/run_longmemeval.py --per-type 5 --out my-run.json

# A single category, or specific question ids
python eval/run_longmemeval.py --types temporal-reasoning,knowledge-update
python eval/run_longmemeval.py --ids 00ca467f,09d032c9

# Haystack setting (~45 sessions/question); --sleep-every batches consolidation
python eval/run_longmemeval.py --haystack --per-type 5 --sleep-every 5

# The internal 14-scenario suite — pennies, minutes, catches regressions
python eval/run_eval.py
```

Results are written to `eval/results/` after every question, so an interrupted run keeps
what it finished. The run-by-run index with pipeline state and scores is
`eval/results/README.md`.

Datasets live in `eval/data/` and are gitignored — download `longmemeval_oracle.json`
and `longmemeval_s_cleaned.json` from the `xiaowu0162/longmemeval-cleaned` repository on
HuggingFace.

Models are set in `.env`: `MEMORY_MODEL` for memory operations and `MEMORY_CHAT_MODEL`
for the answer path. Both currently point at OpenRouter.

**Cost.** The two final 30-question runs together cost approximately $2.00. The internal
suite is a few cents. The working rule that emerged: verify offline against archived
stores first, then run the internal suite, and spend on a haystack run only to confirm a
specific failure mode is fixed.

---

*All memory writes during evaluation run against a throwaway database in a temporary
directory, under a per-question user id. No production or personal memories are touched
by any run in this report.*
