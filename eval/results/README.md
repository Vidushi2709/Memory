# Eval results

Raw per-question output from each run. Numbers are chronological — each file
measures a different state of the pipeline, so they are only comparable within
a series (internal vs internal, longmemeval vs longmemeval).

## Internal scenario harness (`run_eval.py`, 13 questions)

| File | Pipeline state | Score |
|---|---|---|
| `internal-01-baseline.json` | Tier 3 complete, compose-on-read off, PPR on | 13/13 |
| `internal-02-compose-on-read.json` | Same + compose-on-read enabled (`--compose`) | 13/13 |
| `internal-03-no-ppr.json` | Same + PageRank disabled (`--no-ppr`) | 12/13 |
| `internal-04-after-restructure.json` | Thin write path + sleep pass + transcript recall | 13/13 |
| `internal-05-after-perf-work.json` | Core-refresh-once + parallel sleep stages | 12/13 |
| `internal-06-detail-fix.json` | Extraction terseness scoped to keywords/context only | 13/13 |
| `internal-07-gemini.json` | Model trial: Gemini (rejected — see report) | 12/13 |
| `internal-08-qwen.json` | Model switched to Qwen3-30B-A3B | 13/13 |
| `internal-09-bugfixes.json` | Full correctness audit applied (ranking, floor, dedup, grounding) | 13/13 |

The harness stopped discriminating at 13/13, which is why LongMemEval was brought in.
Experiment verdicts: compose-on-read gained nothing and cost a call per message (left off);
PPR showed no measurable difference but costs no LLM calls (left on — the `internal-03`
failure was generation noise, not a retrieval miss).

## LongMemEval, oracle setting (`run_longmemeval.py`, 30 questions, 5 per type)

| File | Pipeline state | Score |
|---|---|---|
| `longmemeval-01-baseline.json` | Tier 3 complete, per-fact write path, temperature 0.7 | 19/30 (63%) |
| `longmemeval-02-failure-audit.json` | Same code, re-ran only the 10 failures with store/retrieval audits | 4 passed on re-run (variance), 4 write-path, 1 retrieval, 1 reasoning |
| `longmemeval-03-after-restructure.json` | Thin write path + sleep pass + transcript recall + temperature 0 | 24/30 (80%) |
| `longmemeval-04-excerpt-abstention-fixes.json` | Longer transcript excerpts + guarded-abstention prompt; re-ran 7 targeted questions | 3/7 — content now retrieved but the model refused to read it |
| `longmemeval-05-verbatim-recall.json` | Transcript excerpts moved to their own verbatim-framed prompt field, turn search top-5 | see file |

Store coverage (is the gold answer present in what the write path stored?) went
from 6/10 on the failure subset to 26/28 across the restructured run.

Runs 04 and 05 are targeted subsets (`--ids`), not full samples — their totals
are not comparable to the 30-question runs; read them per question.

## LongMemEval, haystack setting (`--haystack`, ~45 sessions/question)

| File | Pipeline state | Score |
|---|---|---|
| `longmemeval-07-smoke-after-fixes.json` | Oracle smoke of the 7 targeted questions after fixes | see file |
| `longmemeval-08-haystack-pilot.json` | First haystack pilot, 1 per type | 5/6 (83%) |
| `longmemeval-09-status-fix.json` | Status field added; targeted re-run | see file |
| `longmemeval-10-batch1-SUPERSEDED-telemetry-race.json` | Batch 1 partial (5/10 scored) — killed after a sleep pass failed on the ChromaDB telemetry race; kept for its diagnostics | 3/5, invalid |
| `longmemeval-11-batch1-all-fixes.json` | Batch 1 full, every fix active (ranking, floor, ordering, event guard, truncation, telemetry) | 5/10 (50%); store 9/9, retrieval 9/9, all failures reasoning loss |

Batch 1 is the two hardest categories (knowledge-update + multi-session) — its
totals are not comparable to per-type-sampled runs. The 50% run's loss
attribution is the meaningful number: no write-path or retrieval losses remain.
Analysis and roadmap: `blackboard.md` at the repo root.

## Split-model runs (Qwen3-30B memory + DeepSeek-3.2 answers)

| File | Setting | Score |
|---|---|---|
| `internal-10-progression-gate.json` | Internal, after progression + user-centricity fixes | 12/13 (flaky snack judgment; preferences demonstrably retrieved) |
| `internal-11-chat-deepseek.json` | Internal, DeepSeek answer path, first gate | 12/13 — exposed the gate wording dropping friend-facts |
| `internal-12-chat-deepseek-gate2.json` | Internal, after gate-wording fix | 13/13 |
| `longmemeval-12-oracle60-deepseek-chat.json` | Oracle, 10/type | 44/60 (73%); temporal 40% is date-arithmetic declines |
| `longmemeval-13-haystack30-deepseek-chat.json` + `-13b-...-remainder.json` (merged: `-MERGED.json`) | Haystack, 5/type (run split by an external stop) | **23/30 (77%)**; abstention 2/2 |
| `internal-13-aggregation.json` | Internal (now 14 questions), aggregation path live | 14/14 |
| `longmemeval-14-oracle-agg-retest.json` | Oracle, the 8 count/date failures re-run with aggregation | 3/8 flipped; all passes used the computed aggregate; remaining fails are per-run extraction misses |
| `longmemeval-15-haystack-agg-retest.json` | Haystack, doctor + museum re-run | 0/2 — both blocked by extraction variance (opposite facts missing vs the oracle run) |
| `internal-14-gap-check.json` | Internal, extraction gap check live (second "what did I miss?" pass per session) | 14/14, twice in a row (first run wasn't saved) — offline replay of the doctor question separately verified the pass recovers the missing appointment |

## Fields

Every record carries `question`, `expected`, `answer`, `correct`. LongMemEval
records add `question_type`, `is_abstention`, and — where the audit ran —
`store_has_answer`, `retrieval_has_answer`, and `loss_stage`
(`correct` / `write-path loss` / `retrieval loss` / `reasoning loss`).
