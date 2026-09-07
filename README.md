# 🧠 Memory — Persistent AI Memory System

A lightweight, **LLM-powered memory layer** that lets an AI chatbot remember things about you across sessions. Memories are stored as vector embeddings in a single SQLite file and retrieved semantically, so the AI finds the right memories even when you don't use exact wording.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Persistent memory** | All memories in one SQLite file (`memory.db`) — no server process, backups are a file copy |
| **Hybrid retrieval** | Vector similarity + BM25 keyword search fused with reciprocal rank fusion (Zep-style), then ranked by relevance + retention + importance |
| **Ebbinghaus forgetting** | Retention `e^(−t/S)` demotes stale memories in ranking — never deletes them; strength `S` grows each time a memory is recalled |
| **Thin write path** | One extraction call per exchange stores facts append-only; all judgment-heavy reconciliation happens off the hot path |
| **Sleep-time consolidation** | At session end, one large-context pass reconciles the session (supersede/link), dedups, evolves contexts, reflects, and rewrites the core profile (Letta pattern: only the sleep pass has write authority over consolidated memory) |
| **Transcript recall** | Retrieval also BM25-searches raw archived exchanges, so what the *assistant* said stays recallable — fact extraction alone loses it |
| **Core memory** | A short always-in-prompt user profile (MemGPT-style), rewritten by the LLM as facts change |
| **Consolidation** | Session-end dedup merges near-duplicate memories; importance-triggered reflection distills recent facts into higher-level insight memories |
| **Session layer** | Every memory is tagged with the session it came from; `/sessions` browses past conversations via their summaries |
| **Linked notes** | A-Mem-style: memories carry keywords + a context sentence and link to related memories; retrieval pulls linked memories along (multi-hop) |
| **Memory evolution** | New memories can trigger LLM rewrites of linked neighbors' context lines at session end |
| **Graph retrieval** | Personalized PageRank over the link graph joins vector + BM25 as a third ranking signal |
| **Experience bank** | Raw session transcripts archived to `./transcripts/` (LatentMem-style), so history can be re-extracted when the pipeline improves |
| **Eval harness** | `eval/run_eval.py` scores the whole pipeline per category (update, temporal, multi-hop, abstention…) with an LLM judge |
| **Memory versioning** | Old memories are **never deleted** — marked as superseded so history is preserved |
| **Historical queries** | Ask *"where did I live before?"* and the AI looks up your past memories |
| **Timestamps** | Every memory records when the fact became true (`date`, relative dates resolved by the LLM), when it was saved (`saved_at`), and when it was superseded |
| **Background writes** | Memory updates run in the background — no delay in chat responses |
| **Proactive recall** | Shows your most recent memories at the start of every session |
| **Session summaries** | Summarises each conversation and stores it as a memory on exit |
| **Multi-user** | Isolated memory per user ID |
| **Slash commands** | `/memories`, `/categories`, `/forget`, `/help`, `/quit` |

---

## 🗂️ Project Structure

```
Memory/
├── chatbot.py              ← Main chatbot (run this)
├── .env                    ← API keys (never commit)
├── memory.db               ← The whole store: one SQLite file (auto-created)
├── transcripts/            ← Raw session logs, one JSONL per user (auto-created)
├── eval/                   ← Eval harnesses
│   ├── scenarios.py              ← scripted conversations + expected answers
│   ├── run_eval.py               ← internal harness: ingest → ask → LLM-judge → table
│   ├── run_longmemeval.py        ← LongMemEval runner (oracle + haystack settings)
│   ├── data/                     ← benchmark datasets (gitignored, downloaded)
│   └── results/                  ← raw per-question output, indexed in its README
└── memory/                 ← Core memory package
    ├── __init__.py
    ├── embedding_generation.py   ← Embeds text → float vectors
    ├── extract_memory.py         ← LLM extracts structured memories (+ keywords, context)
    ├── memory_store.py           ← SQLite read/write + hybrid/graph search
    ├── consolidate.py            ← dedup merge, reflection insights, memory evolution
    ├── transcripts.py            ← raw experience bank (JSONL per user)
    └── update_memory.py          ← per-fact ADD/UPDATE/SUPERSEDE/NOOP + linking
```

---

## ⚡ Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

> The project uses **Gemini 2.5 Flash-Lite**, which is free on a [Google AI Studio key](https://aistudio.google.com/apikey) — measured the fastest of every candidate on both memory workloads. Without `GEMINI_API_KEY` it falls back to the same model through [OpenRouter](https://openrouter.ai) (about $1/month at personal volume).

### 3. Run the chatbot

```bash
python chatbot.py
```

---

## 💬 Chatbot Usage

```
You: Hi! My name is Vidushi and I love hiking.

╭──────────────────────── AI ────────────────────────╮
│ Nice to meet you, Vidushi! I'll remember that you  │
│ love hiking. Do you have a favourite trail?        │
╰────────────────────────────────────────────────────╯
  ✦ Memory updated in background: Added new memory.
```

### Slash Commands

| Command | What it does |
|---|---|
| `/memories` | Show a full table of all memories (current **and** old), with Status + timestamp |
| `/sessions` | List past sessions with their summaries and memory counts |
| `/categories` | List memory categories (e.g. hobbies, food, location) |
| `/forget` | Delete ALL your memories (asks for confirmation) |
| `/help` | Show command reference |
| `/quit` | Save a session summary, consolidate duplicate memories, and exit |

The `/memories` table now includes a **Status** column:

```
┌───┬────────────────────────────────┬───────────────┬─────────┬─────────────────────┐
│ # │ Memory                         │ Categories    │ Status  │ Saved At            │
├───┼────────────────────────────────┼───────────────┼─────────┼─────────────────────┤
│ 1 │ User's name is Vidushi         │ name          │ Current │ 2026-02-27 18:30:00 │
│ 2 │ User used to live in Delhi     │ location      │ Old     │ 2026-02-20 10:12:00 │
│ 3 │ User now lives in Bangalore    │ location      │ Current │ 2026-02-27 18:31:00 │
└───┴────────────────────────────────┴───────────────┴─────────┴─────────────────────┘
  2 current  |  1 old/superseded  |  3 total.
```

### Multiple Users

At startup, enter any number as your User ID. Each user has completely separate memories:

```
User ID (default 1): 42
Welcome back! I have 7 memories stored for you.
```

---

## 🏗️ How It Works

```
User message
     │
     ▼
[Embed message]  →  [Hybrid search: vector + BM25 + PageRank fusion]
                    over current AND old memories
                    + BM25 search over raw transcript exchanges
                              │
                              ▼
                    [LLM generates response]
                    using core profile + memories + transcript excerpts
                    (old memories tagged [OLD/SUPERSEDED])
                              │
                              ▼
                  [Background: THIN write path — 1 LLM call]
                  extract facts → store append-only (no judgment)
                  + raw exchange archived to transcripts/
                              │
                              ▼ (at session end)
                  [SLEEP PASS — consolidation with write authority]
                  reconcile session vs store (supersede / link, one call)
                  → dedup-merge → evolve contexts → reflect → core profile
                              │
                              ▼
                    [SQLite persists to disk]
                    (old memories kept with is_current=0)
```

### Memory lifecycle

1. **Extraction (the only hot-path LLM call)** — `extract_memory.py` pulls structured facts from conversation turns (text, category, importance 1-10, keywords, context sentence, and the ISO date the fact became true — relative dates are resolved against the conversation's date). Facts are stored **append-only**; no judgment happens while the user waits.
2. **Embedding** — `embedding_generation.py` converts memory text to a 384-dim vector using `all-MiniLM-L6-v2` through fastembed (ONNX, no torch: 197 MB resident and a 13 s cold start, against 1.4 GB and 54 s before).
3. **Storage** — `memory_store.py` inserts into SQLite with `user_id`, `saved_at`, `importance`, `last_accessed`, and `is_current=1` columns. Raw exchanges are appended to `transcripts/` in parallel.
4. **Retrieval** — On every message, vector-similarity and BM25 keyword rankings are fused with reciprocal rank fusion, then re-ranked by `relevance + retention + importance` where retention is `e^(−days_since_recall / strength)` (strength grows on every recall). Old memories are always searchable, tagged `[OLD/SUPERSEDED]`, and raw transcript exchanges are BM25-searched alongside (tagged `[PAST CONVERSATION]`).
5. **Sleep pass (session end)** — `consolidate.py: sleep_pass` holds all write authority over consolidated memory: one large-context call reconciles the session's new facts against the store (supersede changed facts / link related ones), then dedup-merge, context evolution, reflection, and the core-profile rewrite run. Fewer, bigger LLM calls instead of many per-fact ones — judgment errors no longer compound across 7+ decisions per fact on the hot path.
6. **Core memory** — a short user profile stored per user and injected into *every* prompt, so key facts (name, location, work) never depend on retrieval. Rewritten only by the sleep pass.
7. **Consolidation** — `consolidate.py` runs three passes: reflection after memory updates (once fresh facts accumulate ~40 summed importance, the LLM distills 2-3 insight memories from them), dedup at session end (memories above 0.9 cosine similarity are LLM-merged into one; originals soft-deleted, never removed), and memory evolution (new memories can trigger context rewrites of their linked neighbors). Derived memories (summaries, insights) never merge with the raw facts they came from.
8. **Sessions** — each run gets a `session_id` stamped on every memory it writes, so conversations can be browsed later; retrieval stays global across all sessions. Raw transcripts are appended to `./transcripts/user_<id>.jsonl` as an experience bank for future re-extraction.
9. **Linked notes** — the write path asks the action LLM which existing memories each new fact genuinely relates to and stores bidirectional links. At retrieval, up to 3 linked memories ride along with the search hits (tagged `[LINKED]`), and Personalized PageRank over the link graph contributes a third ranking to the fusion.
10. **Evaluation** — `python eval/run_eval.py` ingests scripted conversations through the real pipeline into a temp database and scores answers per category with an LLM judge. `--compose` toggles the compose-on-read experiment.

### Compose-on-read (experiment — switched off)

Inspired by LatentMem's memory composer: instead of injecting the raw retrieved memory strings into the chat prompt, an extra LLM call writes a 1-3 sentence digest tailored to the current question, and the responder sees only that.

We built it, measured it with the eval harness, and **turned it off** (`COMPOSE_ON_READ = False` in `chatbot.py`):

- Accuracy was identical to the baseline (13/13 vs 13/13) — no measurable benefit.
- It costs one extra LLM call on *every* message, adding latency.
- Compression can blur fact attribution — in one eval answer the digest reassigned a friend's new job to the user.

The code stays behind the flag so it can be re-tested (`python eval/run_eval.py --compose`) if retrieval ever returns enough noise that filtering earns its keep.

### PageRank ablation (experiment — kept on)

The other measured experiment: the same eval with the PageRank ranking removed (`python eval/run_eval.py --no-ppr`, `USE_PPR` in `memory_store.py`).

Result: 12/13 without PPR vs 13/13 with — but the one failure wasn't a retrieval miss (the answer proved the memory was retrieved; the model just phrased unsafe advice), so the honest reading is **no measurable retrieval difference on this scenario set**: link expansion alone covers the current multi-hop cases. PPR stays **enabled** anyway, because unlike compose-on-read it costs no LLM calls — just milliseconds of local graph math — and its value should grow as the link graph densifies.

---

## 🕰️ Memory Versioning & History

Memories are **never hard-deleted** when updated. Instead, the old version is marked `is_current=0` (superseded), and a new current memory is added. This means:

- The full **history of changes** is preserved in the database.
- The AI can answer questions like **"where did I live before?"** or **"what was my old job?"** by looking up superseded memories.
- The `/memories` table shows both `Current` and `Old` entries with their saved timestamps.

### How historical queries work

Old memories are always included in search results, tagged `[OLD/SUPERSEDED]`, so no keyword detection is needed — the LLM sees both states and decides which to use:

```
You: Where do I live?          → retrieves current "Bangalore" + old "Delhi" [OLD/SUPERSEDED]
                                → "Bangalore"
You: Where did I live before?  → same retrieval
                                → "You used to live in Delhi, and now live in Bangalore."
```

### Columns stored per memory

| Field | Description |
|---|---|
| `memory_text` | The memory content |
| `categories` | Comma-separated category tags |
| `date` | ISO timestamp of *when the fact occurred / was first noted* |
| `saved_at` | ISO timestamp of *when it was written to the DB* |
| `timestamp` | Unix epoch of `date` (for range queries) |
| `is_current` | `1` = active, `0` = superseded/old |
| `superseded_at` | ISO timestamp of when the memory was marked old (if applicable) |
| `importance` | LLM-rated 1-10 at extraction time (used in retrieval ranking) |
| `last_accessed` | Unix epoch of the memory's last retrieval hit (used for retention) |
| `strength` | Ebbinghaus strength — starts at 5, +1 per recall; higher = slower forgetting |
| `session_id` | ID of the chat session that wrote this memory |
| `kind` | `fact` (extracted), `summary` (session summary), or `insight` (from reflection) |
| `keywords` | LLM-extracted salient tokens, indexed by BM25 alongside the memory text |
| `context` | One-sentence LLM interpretation of the fact; rewritable by memory evolution |
| `links` | Comma-separated ids of related memories (bidirectional, A-Mem style) |
| `type` | `core` marks the per-user profile record (excluded from search/listing) |

---

## 🔧 Configuration

| Setting | Location | Default |
|---|---|---|
| LLM model | `chatbot.py` → `_lm` | `openrouter/mistralai/mistral-small-3.2-24b-instruct` |
| Embedding model | `memory/embedding_generation.py` → `MEMORY_EMBED_MODEL` | `all-MiniLM-L6-v2` (384-dim) via fastembed/ONNX |
| Data dir | `MEMORY_DIR` env (default: repo root) | holds `memory.db`, `transcripts/` and `models/`; the evals point it at a temp dir |
| LLM | `memory/llm.py` → `MEMORY_MODEL` / `MEMORY_CHAT_MODEL` | `gemini/gemini-2.5-flash-lite` when `GEMINI_API_KEY` is set (free tier), else the same model via OpenRouter |
| Request throttle | `memory/llm.py` → `MEMORY_RPM` | `0` (off). Set `10` on a **free** Gemini key — that tier allows 10 requests/min and only 20/day per model |
| Relevance floor | `memory/memory_store.py` → `RELEVANCE_FLOOR` | `0.10` true cosine (vector ranking only — BM25 hits bypass it). Calibrated: a question and the memory answering it can sit at 0.15 |
| Retrieval ranking | `memory/memory_store.py` → `search_memories` | `RRF(vector, BM25, PageRank) + 0.10·e^(−days/strength) + 0.15·importance/10 − 0.30 if superseded` |
| Ebbinghaus strength | `memory/memory_store.py` → `STRENGTH_INIT` / `STRENGTH_PER_RECALL` | `5.0` / `1.0` |
| Admission floor | `memory/update_memory.py` → `MEMORY_MIN_IMPORTANCE` | `0` (store everything). Above 0 the write path refuses facts scored below it — see `eval/sweep_admission.py` |
| Dedup similarity | `memory/consolidate.py` → `DEDUP_SIMILARITY` | `0.9` |
| Reflection trigger | `memory/consolidate.py` → `REFLECTION_THRESHOLD` | `40` (summed importance of fresh facts) |
| Link expansion cap | `memory/memory_store.py` → `LINK_EXPANSION_CAP` | `3` linked memories per query |
| PageRank seeds | `memory/memory_store.py` → `PPR_SEEDS` | `8` top fused hits |
| Evolution caps | `memory/consolidate.py` → `EVOLVE_NEW_CAP` / `EVOLVE_LINK_CAP` | `3` new memories / `2` neighbors each |
| Compose-on-read | `chatbot.py` → `COMPOSE_ON_READ` | `False` — tested, no accuracy gain for the extra latency (see "Compose-on-read" section) |
| Memories retrieved per query | `memory/memory_store.py` → `top_k` | `5` (chat) / `10` (update pipeline neighbors) |
| Include old memories | `memory/memory_store.py` → `include_old` | `False` (chatbot always passes `True`) |
| Transcript window | `chatbot.py` → `chat_loop` | last 10 messages |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `dspy` | LLM orchestration, structured outputs |
| `sqlite3` | The store itself — stdlib, no server, one `memory.db` file |
| `fastembed` | Text → embedding (local ONNX, no API cost, no torch) |
| `rank-bm25` | BM25 keyword ranking for hybrid retrieval |
| `networkx` | Personalized PageRank over the memory link graph |
| `pydantic` | Data validation for memory models |
| `rich` | Terminal UI |
| `python-dotenv` | Loads `.env` API keys |

## Contribution?

like it? take it, break it. have fun :)