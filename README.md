# 🧠 Memory — Persistent AI Memory System

A lightweight, **LLM-powered memory layer** that lets an AI chatbot remember things about you across sessions. Memories are stored as vector embeddings in ChromaDB and retrieved semantically, so the AI finds the right memories even when you don't use exact wording.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Persistent memory** | All memories saved to disk (`./chroma_db`) — survive restarts |
| **Three-factor retrieval** | Ranks memories by relevance + recency + importance (Generative Agents formula), not just vector similarity |
| **Mem0-style write path** | Extracts atomic facts, then per-fact decides ADD / UPDATE / SUPERSEDE / NOOP against its nearest neighbors |
| **Core memory** | A short always-in-prompt user profile (MemGPT-style), rewritten by the LLM as facts change |
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
├── chroma_db/              ← Persistent vector store (auto-created)
└── memory/                 ← Core memory package
    ├── __init__.py
    ├── embedding_generation.py   ← Embeds text → float vectors
    ├── extract_memory.py         ← LLM extracts structured memories from conversation
    ├── memory_store.py           ← ChromaDB read/write operations
    └── update_memory.py          ← ReAct agent: add/update/delete memories
```

---

## ⚡ Quickstart

### 1. Install dependencies

```bash
pip install dspy chromadb sentence-transformers pydantic python-dotenv rich
```

### 2. Set your API key

Create a `.env` file in the project root:

```
OPEN_ROUTER_KEY=your_key_here
```

> The project uses `mistralai/mistral-small-3.2-24b-instruct` via [OpenRouter](https://openrouter.ai) by default. Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

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
| `/categories` | List memory categories (e.g. hobbies, food, location) |
| `/forget` | Delete ALL your memories (asks for confirmation) |
| `/help` | Show command reference |
| `/quit` | Save a session summary to memory and exit |

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
[Embed message]  →  [Search ChromaDB — current AND old memories]
                    ranked by relevance + recency + importance
                              │
                              ▼
                    [LLM generates response]
                    using core memory profile + retrieved memories
                    (old memories tagged [OLD/SUPERSEDED])
                              │
                              ▼
                  [Background: Mem0-style write path]
                  extract facts → per fact: retrieve top-10 neighbors
                  → ADD / UPDATE (soft-delete old + add new)
                  / SUPERSEDE (mark old only) / NOOP
                  → refresh core memory profile
                              │
                              ▼
                    [ChromaDB persists to disk]
                    (old memories kept with is_current=0)
```

### Memory lifecycle

1. **Extraction** — `extract_memory.py` uses an LLM to pull structured facts from conversation turns (text, category, importance 1-10, and the ISO date the fact became true — relative dates like "next month" are resolved against today).
2. **Embedding** — `embedding_generation.py` converts memory text to a 384-dim vector using `all-MiniLM-L6-v2`.
3. **Storage** — `memory_store.py` upserts into ChromaDB with `user_id`, `saved_at`, `importance`, `last_accessed`, and `is_current=1` metadata.
4. **Retrieval** — On every message, candidate memories are fetched by vector similarity, then re-ranked by `relevance + recency + importance` where recency is `0.995^(hours since last recall)` and importance is the LLM's 1-10 rating scaled to [0,1]. Old memories are always searchable, tagged `[OLD/SUPERSEDED]`.
5. **Reconciliation** — `update_memory.py` retrieves each extracted fact's top-10 similar memories and has the LLM pick one action per fact: add, update (soft-delete old + add new), supersede, or noop. Afterwards it rewrites the core memory profile if the new facts change it.
6. **Core memory** — a short user profile stored per user and injected into *every* prompt, so key facts (name, location, work) never depend on retrieval.

---

## 🕰️ Memory Versioning & History

Memories are **never hard-deleted** when updated. Instead, the old version is marked `is_current=0` (superseded), and a new current memory is added. This means:

- The full **history of changes** is preserved in ChromaDB.
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

### Memory metadata stored in ChromaDB

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
| `last_accessed` | Unix epoch of the memory's last retrieval hit (used for recency) |
| `type` | `core` marks the per-user profile record (excluded from search/listing) |

---

## 🔧 Configuration

| Setting | Location | Default |
|---|---|---|
| LLM model | `chatbot.py` → `_lm` | `openrouter/mistralai/mistral-small-3.2-24b-instruct` |
| Embedding model | `memory/embedding_generation.py` | `all-MiniLM-L6-v2` (384-dim) |
| DB path | `memory/memory_store.py` | `./chroma_db` |
| Relevance floor | `memory/memory_store.py` → `search_memories` | `0.3` (candidates below this cosine similarity are dropped before re-ranking) |
| Retrieval ranking | `memory/memory_store.py` → `search_memories` | `relevance + 0.995^hours_since_recall + importance/10` |
| Memories retrieved per query | `memory/memory_store.py` → `top_k` | `5` (chat) / `10` (update pipeline neighbors) |
| Include old memories | `memory/memory_store.py` → `include_old` | `False` (chatbot always passes `True`) |
| Transcript window | `chatbot.py` → `chat_loop` | last 10 messages |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `dspy` | LLM orchestration, ReAct agents, structured outputs |
| `chromadb` | Local vector database |
| `sentence-transformers` | Text → embedding (local, no API cost) |
| `pydantic` | Data validation for memory models |
| `rich` | Terminal UI |
| `python-dotenv` | Loads `.env` API keys |

## Contribution?

like it? take it, break it. have fun :)