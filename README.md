# 🧠 Memory — Persistent AI Memory System

A lightweight, **LLM-powered memory layer** that lets an AI chatbot remember things about you across sessions. Memories are stored as vector embeddings in ChromaDB and retrieved semantically, so the AI finds the right memories even when you don't use exact wording.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Persistent memory** | All memories saved to disk (`./chroma_db`) — survive restarts |
| **Semantic search** | Retrieves relevant memories using vector similarity, not keywords |
| **LLM memory agent** | Automatically decides to ADD / UPDATE / DELETE / NOOP memories |
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
| `/memories` | Show a full table of all stored memories |
| `/categories` | List memory categories (e.g. hobbies, food, location) |
| `/forget` | Delete ALL your memories (asks for confirmation) |
| `/help` | Show command reference |
| `/quit` | Save a session summary to memory and exit |

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
                              │
                              ▼
                    [LLM generates response]
                    using retrieved memories
                              │
                              ▼
                  [Background: Update memory agent]
                  Decides to ADD / UPDATE / DELETE / NOOP
                              │
                              ▼
                    [ChromaDB persists to disk]
```

### Memory lifecycle

1. **Extraction** — `extract_memory.py` uses an LLM to pull structured facts from conversation turns (text, category, sentiment).
2. **Embedding** — `embedding_generation.py` converts memory text to a 384-dim vector using `all-MiniLM-L6-v2`.
3. **Storage** — `memory_store.py` upserts into ChromaDB with `user_id` metadata for isolation.
4. **Retrieval** — On every message, the most semantically similar memories are fetched and injected into the LLM prompt.
5. **Update agent** — `update_memory.py` runs a DSPy `ReAct` agent that can add, update, or delete memories based on new context.

---

## 🔧 Configuration

| Setting | Location | Default |
|---|---|---|
| LLM model | `chatbot.py` → `_lm` | `mistral/mistral-small-latest` |
| Embedding model | `memory/embedding_generation.py` | `all-MiniLM-L6-v2` (384-dim) |
| DB path | `memory/memory_store.py` | `./chroma_db` |
| Similarity threshold | `memory/memory_store.py` → `search_memories` | `0.5` |
| Memories retrieved per query | `memory/memory_store.py` → `top_k` | `5` |
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