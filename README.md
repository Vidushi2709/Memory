# 🧠 Memory — Persistent AI Memory System

A lightweight, **LLM-powered memory layer** that lets an AI chatbot remember things about you across sessions. Memories are stored as vector embeddings in ChromaDB and retrieved semantically, so the AI finds the right memories even when you don't use exact wording.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Persistent memory** | All memories saved to disk (`./chroma_db`) — survive restarts |
| **Semantic search** | Retrieves relevant memories using vector similarity, not keywords |
| **LLM memory agent** | Automatically decides to ADD / UPDATE / SUPERSEDE / NOOP memories |
| **Memory versioning** | Old memories are **never deleted** — marked as superseded so history is preserved |
| **Historical queries** | Ask *"where did I live before?"* and the AI looks up your past memories |
| **Timestamps** | Every memory records when it was saved (`saved_at`) and when it was superseded |
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
    ├── update_memory.py          ← ReAct agent: add/update/delete memories
    └── responser.py              ← Alternative CLI chatbot (simpler, no rich UI)
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
MISTRAL_API_KEY=your_key_here
```

> The project uses `mistral/mistral-small-latest` by default. Get a free key at [console.mistral.ai](https://console.mistral.ai).

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
[Embed message]  →  [Search ChromaDB for similar memories]
                    (current only, OR include old for historical queries)
                              │
                              ▼
                    [LLM generates response]
                    using retrieved memories
                    (old memories tagged [OLD/SUPERSEDED])
                              │
                              ▼
                  [Background: Update memory agent]
                  ADD new  /  UPDATE (soft-delete old + add new)
                  SUPERSEDE (mark old only)  /  NOOP
                              │
                              ▼
                    [ChromaDB persists to disk]
                    (old memories kept with is_current=0)
```

### Memory lifecycle

1. **Extraction** — `extract_memory.py` uses an LLM to pull structured facts from conversation turns (text, category, sentiment).
2. **Embedding** — `embedding_generation.py` converts memory text to a 384-dim vector using `all-MiniLM-L6-v2`.
3. **Storage** — `memory_store.py` upserts into ChromaDB with `user_id`, `saved_at` timestamp, and `is_current=1` metadata.
4. **Retrieval** — On every message, the most semantically similar *current* memories are fetched and injected into the LLM prompt. Historical queries (containing words like *before*, *previously*, *used to*) automatically fetch old memories too.
5. **Update agent** — `update_memory.py` runs a DSPy `ReAct` agent that can add, update (soft-delete old + add new), or supersede memories.

---

## 🕰️ Memory Versioning & History

Memories are **never hard-deleted** when updated. Instead, the old version is marked `is_current=0` (superseded), and a new current memory is added. This means:

- The full **history of changes** is preserved in ChromaDB.
- The AI can answer questions like **"where did I live before?"** or **"what was my old job?"** by looking up superseded memories.
- The `/memories` table shows both `Current` and `Old` entries with their saved timestamps.

### How historical queries work

```
You: Where do I live?          → searches is_current=1 only  → "Bangalore"
You: Where did I live before?  → detects historical keywords
                                → searches is_current=0 too
                                → "You used to live in Delhi (old memory
                                   from 2026-02-20), and now live in Bangalore."
```

Keywords that trigger a historical search: `before`, `previously`, `used to`, `old`, `past`, `prior`, `earlier`, `last time`, `back then`, `formerly`, `history`, `what was`, `where did I`, `who did I`, `when did I`, `what did I`.

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

---

## 🔧 Configuration

| Setting | Location | Default |
|---|---|---|
| LLM model | `chatbot.py` → `_lm` | `mistral/mistral-small-latest` |
| Embedding model | `memory/embedding_generation.py` | `all-MiniLM-L6-v2` (384-dim) |
| DB path | `memory/memory_store.py` | `./chroma_db` |
| Similarity threshold | `memory/memory_store.py` → `search_memories` | `0.5` |
| Memories retrieved per query | `memory/memory_store.py` → `top_k` | `5` |
| Include old memories | `memory/memory_store.py` → `include_old` | `False` (auto `True` for historical queries) |
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