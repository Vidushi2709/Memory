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

## Fields

Every record carries `question`, `expected`, `answer`, `correct`. LongMemEval
records add `question_type`, `is_abstention`, and — where the audit ran —
`store_has_answer`, `retrieval_has_answer`, and `loss_stage`
(`correct` / `write-path loss` / `retrieval loss` / `reasoning loss`).
