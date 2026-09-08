# (o_O) mem - a memory for your whole laptop

Type `mem` and tell it things. It keeps every note, every Claude Code session and every
git commit in one SQLite file, turns them into facts overnight, and answers "what did I
say about X?" from any terminal or from inside Claude Code. Local, one process, no server.

---

## \o/ Use it

```powershell
mem                                  # profile, upcoming dates, recent notes, then a prompt
mem moved my desk to the window      # note: free, instant, timestamped
mem ? desk                           # search notes, commits and facts (keyword, ~2 s, no LLM)
mem on yesterday                     # everything logged that day (or an ISO date)
mem which desk did I move?           # ends with "?": answer from memory (one LLM call)
mem week                             # digest of the last seven days (one LLM call)
mem upcoming                         # facts dated after today
mem ping                             # once a day: message box with what is due today or tomorrow
mem all                              # every current fact with its id
mem forget 3f2a                      # delete one fact by id prefix
mem sleep                            # what the nightly task runs (see below)
```

Open it from anywhere with **Ctrl+Alt+M**, or Win+R → `mem`. Inside the window a note is
saved as typed; a line ending in `?` or starting with `ask` gets an answer. Speak instead
of typing with [Handy](https://github.com/cjpais/Handy) (open source, offline): hold
Ctrl+Space, talk, release.

**Inside Claude Code:** `/mem <question>?` hands Claude the matching notes and facts to
answer from (no extra model call); `/mem <note>` saves one. A `SessionStart` hook puts your
profile and last ten notes into every session.

**Every night at 23:00** a Windows scheduled task runs `mem sleep`, which

1. logs new Claude Code exchanges from `~/.claude/projects` and your own git commits from
   every repo under `~/Desktop` (`MEM_GIT_ROOT`), read-only;
2. extracts facts from anything not yet processed, at most `MEM_SLEEP_MAX_CHUNKS` (30)
   chunks a night, roughly ₹0.1 each; processed chunks are listed in `transcripts/extracted.txt`;
3. reconciles, dedups, links, and rewrites the profile.

A second task, "mem ping", checks every 30 minutes while the laptop is awake and, once per
day, shows a Windows message box listing plans and deadlines due today or tomorrow. Only forward-dated facts count, so "remind me in two weeks"
fires once, on the day, and the day before.

Everything written to the log passes through `redact()` in `memory/transcripts.py`, which
replaces API-key, token and password shapes with `[REDACTED]`.

### Setup

```bash
pip install -r requirements.txt
```

Put `GEMINI_API_KEY=...` in `.env`. The default model is Gemini 2.5 Flash-Lite,
without the key it uses the same model through OpenRouter (`OPEN_ROUTER_KEY`).

**Swap the LLM for whatever you like.** Two env vars in `.env`:

```
MEMORY_MODEL=openrouter/anthropic/claude-sonnet-5      # extraction + nightly consolidation
MEMORY_CHAT_MODEL=openrouter/openai/gpt-5-mini          # mem ask / mem week answers (defaults to MEMORY_MODEL)
```

Extraction and consolidation go through dspy, so any name litellm accepts works: `gemini/...`,
`openrouter/...`, `openai/...`, `anthropic/...`, or a local `ollama/...`. `mem ask` and `mem week`
make one plain HTTPS call and support two routes: a `gemini/` model with `GEMINI_API_KEY`, or
anything on OpenRouter. Set `MEMORY_API_KEY_ENV` if your OpenRouter key lives under another name.
Costs in this README assume Flash-Lite; a bigger model changes them, not the code.
The `mem` command is `mem.cmd` in `%LOCALAPPDATA%\Microsoft\WindowsApps`, a Start Menu
shortcut with the hotkey, and the "mem sleep" task in Task Scheduler.

---

## [=] Layout

```
mem.py                  ← the command
memory.db               ← the whole store, one SQLite file
transcripts/            ← raw log, one JSONL per user; extracted.txt marks processed chunks
memory/
  transcripts.py        ← append-only log, BM25 over it, redaction
  extract_memory.py     ← LLM turns a session into facts (life and work lenses)
  update_memory.py      ← thin write path: extract, store append-only
  consolidate.py        ← sleep pass: reconcile, dedup, evolve, reflect, profile
  memory_store.py       ← SQLite read/write, hybrid + graph search
  embedding_generation.py, answer.py, aggregate.py, llm.py
eval/                   ← LongMemEval runner, internal harness, re-judge; results/ has the runs
tests/                  ← pytest
```

---

## (>_<) How it works

- **Write path is thin.** A note costs nothing: it is appended to the log. Facts are
  extracted later, one LLM call per chunk, and stored append-only with the date they became
  true, an importance score, keywords, a context sentence, and links to related facts.
- **Sleep pass has the write authority.** One large-context call reconciles new facts
  against the store (supersede or link), then dedup-merge, context evolution, reflection
  (insight memories once enough importance accumulates), and the profile rewrite. Old facts
  are never deleted, only marked `is_current=0`, so "where did I live before?" still works.
- **Retrieval fuses three rankings.** Vector similarity (384-d, local ONNX), BM25 keywords,
  and Personalized PageRank over the link graph, combined with reciprocal rank fusion and
  re-ranked by Ebbinghaus retention and importance. Raw log lines are BM25-searched
  alongside, so what an assistant said stays recallable. `mem ?` uses keywords only, which
  measured within noise of the full stack and skips the model load.
- **Core profile.** A short per-user profile goes into every prompt so the basics never
  depend on retrieval.

### What was measured

On LongMemEval, a public benchmark that hides questions inside long multi-session chat
histories, scored with an LLM judge (full write-up in `report.md`, raw runs in `eval/results/`):

| Setting | Baseline | Final |
|---|---:|---:|
| Oracle, 30 questions (the right sessions are given) | 19/30 (63%) | **26/30 (87%)** |
| Haystack, 30 questions (buried in ~45 unrelated sessions) | | **26/30 (87%)** |

Per type in the final runs, oracle / haystack: single-session-user 5/5 / 4/5,
single-session-assistant 4/5 / 5/5, single-session-preference 4/5 / 4/5, multi-session 3/5 /
3/5, knowledge-update 5/5 / 5/5, temporal-reasoning 5/5 / 5/5, abstention 2/2 / 2/2.

What moved the score, in order: the write path, not retrieval (facts were dropped at
extraction, truncated mid-response, or turned from plans into events by consolidation);
doing counts and date arithmetic in code instead of the model (temporal reasoning went from
4/10 to 5/5); and changing which model writes the final answer, which fixed abstention after
seven rounds of prompting had not. Haystack matching oracle means search difficulty is no
longer the limit. Compose-on-read (an LLM digest of retrieved memories) was tried and
dropped: 13/13 vs 13/13 on the internal suite, extra latency, one blurred attribution.
Removing PageRank showed no measurable difference either way, so it stays on since it costs
no LLM calls. These scores are not comparable to published leaderboard numbers; section 6
of the report says why.

---

## (._.) Configuration

| Setting | Where | Default |
|---|---|---|
| Model | `MEMORY_MODEL` / `MEMORY_CHAT_MODEL` | `gemini-2.5-flash-lite` |
| Data dir | `MEMORY_DIR` | repo root (`memory.db`, `transcripts/`); model cache in `MEMORY_MODEL_CACHE` |
| User | `MEM_USER` | `1` |
| Nightly cap | `MEM_SLEEP_MAX_CHUNKS` | `30` chunks |
| Git root | `MEM_GIT_ROOT` | `~/Desktop`, two levels deep |
| Colours | `mem.py` → `DATE_STYLE`, `TEXT_STYLE`, `ANSWER_STYLES` | pink dates, purple text |
| Request throttle | `MEMORY_RPM` | `0` (off; set `10` on a free Gemini key) |
| Admission floor | `MEMORY_MIN_IMPORTANCE` | `0` (store everything; measured best) |
| Ranking knobs | `memory_store.py` → `RELEVANCE_FLOOR`, `STRENGTH_*`, `LINK_EXPANSION_CAP`, `PPR_SEEDS`, `USE_VECTOR/BM25/PPR` | see file |
| Sleep knobs | `consolidate.py` → `DEDUP_SIMILARITY`, `REFLECTION_THRESHOLD`, `EVOLVE_*` | `0.9`, `40`, `3/2` |

---

## <(^^)> Dependencies

`dspy` (structured LLM calls for extraction and consolidation), `fastembed` (local ONNX
embeddings, no torch), `rank-bm25`, `networkx` (PageRank), `rich` (tables and colour),
`requests` (the `mem ask` call, which skips dspy's 12 s import), `python-dotenv`, `pydantic`.
SQLite is the stdlib.

## \(^o^)/ Contribution?

like it? take it, break it. have fun :)