"""Memory store: one SQLite file, no server process.

Replaced ChromaDB, which could not filter `is_current` or `categories`
server-side and so fetched every record the user owned on every single query —
310 ms at 10k memories, growing linearly. Here that filter is a WHERE clause.
Backups are a file copy and the resident footprint is whatever SQLite caches.

ponytail: vector search is a numpy brute-force dot product over the user's
eligible rows. On this laptop that measured ~60x faster than sqlite-vec at 50k
memories (13 ms vs 856 ms) and needs no loadable extension, at the cost of
being O(n) per query. Past ~100k memories for one user, add sqlite-vec or an
ANN index; the query path is the only thing that would change.

ponytail: one connection guarded by one lock. Writes already run in worker
threads, and at personal volume the lock is never contended. Per-connection
pooling only matters if this ever serves several users at once.
"""
import logging
import math
import os
import re
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import numpy as np
import networkx as nx
import asyncio
from memory import MEMORY_DIR
from memory.transcripts import delete_transcripts

log = logging.getLogger(__name__)

DB_PATH = os.path.join(MEMORY_DIR, "memory.db")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

STRENGTH_INIT = 5.0        # initial Ebbinghaus strength, in days of decay scale
STRENGTH_PER_RECALL = 1.0  # strength gained each time a memory is retrieved
RRF_K = 60                 # reciprocal rank fusion constant
RELEVANCE_FLOOR = 0.10     # min TRUE cosine similarity to enter the vector ranking.
                           # Calibrated, not guessed: a question and the memory
                           # answering it can sit as low as 0.15 ("allergic to?" vs
                           # "cannot eat shellfish"), so 0.25+ costs real recall.
LINK_EXPANSION_CAP = 3     # linked memories pulled in alongside search hits
PPR_SEEDS = 8              # top fused hits used to seed Personalized PageRank
PPR_MIN_SHARE = 0.01       # share of the top PageRank score a node must reach to count
USE_PPR = True             # PageRank ranking in the fusion (eval --no-ppr disables)
USE_VECTOR = True          # cosine ranking in the fusion (eval --retrieval bm25 disables)
USE_BM25 = True            # keyword ranking in the fusion (eval --retrieval vector disables)
IMPORTANCE_WEIGHT = 0.15   # importance and retention only break ties — relevance leads
RETENTION_WEIGHT = 0.10
STALE_PENALTY = 0.30       # a superseded memory must lose to a current one of equal relevance

_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None

# Search rebuilt the whole corpus on every query — fetching rows, building
# dicts, normalising vectors, tokenising and indexing BM25. Profiled at 10k
# memories that was ~370 ms of a ~700 ms query, and none of it changes between
# writes. Cache it against a write counter instead.
#
# ponytail: single-process cache. If this ever runs as a multi-process server,
# the counter has to live in the database rather than in this module.
_write_version = 0
_corpus_cache: dict = {}
CORPUS_CACHE_SIZE = 4  # (user_id, include_old) pairs held at once


def _bump():
    """Invalidate cached corpora. Callers hold _lock."""
    global _write_version
    _write_version += 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT    PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    memory_text   TEXT    NOT NULL,
    categories    TEXT    NOT NULL DEFAULT '',
    date          TEXT    NOT NULL DEFAULT '',
    timestamp     REAL    NOT NULL DEFAULT 0,
    saved_at      TEXT    NOT NULL DEFAULT '',
    is_current    INTEGER NOT NULL DEFAULT 1,
    superseded_at TEXT    NOT NULL DEFAULT '',
    importance    INTEGER NOT NULL DEFAULT 5,
    last_accessed REAL    NOT NULL DEFAULT 0,
    strength      REAL    NOT NULL DEFAULT 5.0,
    session_id    TEXT    NOT NULL DEFAULT '',
    kind          TEXT    NOT NULL DEFAULT 'fact',
    keywords      TEXT    NOT NULL DEFAULT '',
    context       TEXT    NOT NULL DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'happened',
    links         TEXT    NOT NULL DEFAULT '',
    reconciled    INTEGER NOT NULL DEFAULT 0,
    type          TEXT    NOT NULL DEFAULT '',
    embedding     BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, is_current);
"""

# every column except the embedding blob — this is the "metadata" dict that
# consolidation and retrieval read
_META_COLS = [
    "user_id", "memory_text", "categories", "date", "timestamp", "saved_at",
    "is_current", "superseded_at", "importance", "last_accessed", "strength",
    "session_id", "kind", "keywords", "context", "status", "links",
    "reconciled", "type",
]
_META_SQL = ", ".join(_META_COLS)


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")   # survives a crash mid-write
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def _to_blob(embedding) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _from_blob(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class EmbeddedMemory(BaseModel):
    user_id: int
    memory_text: str
    categories: List[str]
    embedding: List[float]
    date: str
    is_current: int = 1  # 1 = current/active, 0 = superseded/old
    importance: int = 5  # 1 = mundane, 10 = life-changing
    strength: float = STRENGTH_INIT
    session_id: str = ""
    kind: str = "fact"  # fact | summary | insight
    keywords: List[str] = []
    context: str = ""
    status: str = "happened"  # happened | planned | considered | ongoing
    links: List[str] = []     # point ids this memory is related to


class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float
    is_current: int = 1  # 1 = current/active, 0 = superseded/old
    importance: int = 5
    session_id: str = ""
    kind: str = "fact"
    context: str = ""
    status: str = "happened"
    links: list[str] = []
    linked: bool = False  # True if pulled in via link expansion, not ranked search


# collection setup

async def create_collection():
    """Ensure the database and schema exist."""
    await asyncio.to_thread(_db)
    print(f"Memory store ready at {DB_PATH}")


# write operations

async def add_memory(embedded_memories: List[EmbeddedMemory]) -> List[str]:
    def _add():
        now = datetime.now()
        ids = [uuid4().hex for _ in embedded_memories]
        rows = [
            (
                id_,
                m.user_id,
                m.memory_text,
                ",".join(m.categories),
                m.date,
                to_epoch(m.date),
                now.isoformat(),          # wall-clock time memory was written
                m.is_current,             # 1=active, 0=superseded
                "",                       # superseded_at
                m.importance,
                now.timestamp(),          # epoch of last retrieval hit
                m.strength,
                m.session_id,
                m.kind,
                ",".join(m.keywords),
                m.context,
                m.status,                 # happened/planned/considered/ongoing
                ",".join(m.links),        # comma-separated point ids
                0,                        # reconciled
                "",                       # type ('core' marks the profile record)
                _to_blob(m.embedding),
            )
            for id_, m in zip(ids, embedded_memories)
        ]
        with _lock:
            db = _db()
            db.executemany(
                f"INSERT INTO memories (id, {_META_SQL}, embedding) "
                f"VALUES ({','.join('?' * (len(_META_COLS) + 2))})",
                rows,
            )
            db.commit()
            _bump()
        return ids
    return await asyncio.to_thread(_add)


async def delete_user_records(user_id: int):
    """Forget a user completely: memories, core profile AND raw transcripts —
    the transcripts are searched on every query, so leaving them is not forgetting."""
    def _delete():
        with _lock:
            db = _db()
            db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            db.commit()
            _bump()
        delete_transcripts(user_id)
    await asyncio.to_thread(_delete)


async def delete_records(point_ids: List[str]):
    def _delete():
        if not point_ids:
            return
        with _lock:
            db = _db()
            db.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' * len(point_ids))})",
                point_ids,
            )
            db.commit()
            _bump()
    await asyncio.to_thread(_delete)


def _update_meta_sync(point_id: str, updates: dict):
    with _lock:
        db = _db()
        cols = ", ".join(f"{k} = ?" for k in updates)
        db.execute(f"UPDATE memories SET {cols} WHERE id = ?",
                   [*updates.values(), point_id])
        db.commit()
        _bump()


async def mark_reconciled(point_ids: List[str]):
    """Stamp facts the sleep pass has reconciled. Sessions whose facts lack the
    stamp (process killed before the pass ran) are found by
    consolidate.unreconciled_sessions and repaired on the next pass."""
    def _mark():
        if not point_ids:
            return
        with _lock:
            db = _db()
            db.execute(
                f"UPDATE memories SET reconciled = 1 "
                f"WHERE id IN ({','.join('?' * len(point_ids))})",
                point_ids,
            )
            db.commit()
            _bump()
    await asyncio.to_thread(_mark)


async def mark_memory_old(point_id: str):
    """
    Mark an existing memory as superseded (is_current=0) without deleting it.
    This preserves history so questions like "where did I live before?" can
    still be answered by searching with include_old=True.
    """
    await asyncio.to_thread(
        _update_meta_sync, point_id,
        {"is_current": 0, "superseded_at": datetime.now().isoformat()},
    )


async def add_links(point_id: str, linked_ids: List[str]):
    """Record bidirectional links between a memory and related memories (A-Mem)."""
    def _link():
        for a, b in [(point_id, lid) for lid in linked_ids if lid != point_id]:
            for src, dst in ((a, b), (b, a)):
                # read and write under ONE lock: releasing between them let a
                # concurrent link add overwrite the list this one just read
                with _lock:
                    db = _db()
                    row = db.execute("SELECT links FROM memories WHERE id = ?",
                                     (src,)).fetchone()
                    if row is None:
                        continue
                    existing = set(filter(None, row["links"].split(",")))
                    if dst in existing:
                        continue
                    existing.add(dst)
                    db.execute("UPDATE memories SET links = ? WHERE id = ?",
                               (",".join(sorted(existing)), src))
                    db.commit()
                    _bump()
    await asyncio.to_thread(_link)


async def set_context(point_id: str, context: str):
    """Rewrite a memory's context description (A-Mem memory evolution)."""
    await asyncio.to_thread(_update_meta_sync, point_id, {"context": context})


# read operations

_EPOCH = datetime(1970, 1, 1)


def to_epoch(iso: str) -> float:
    """ISO date -> unix seconds. datetime.timestamp() raises OSError on Windows
    for pre-1970 dates (childhood events, birth years), so subtract instead."""
    try:
        return (datetime.fromisoformat(iso) - _EPOCH).total_seconds()
    except (ValueError, TypeError):
        return (datetime.now() - _EPOCH).total_seconds()


def kind_of(meta) -> str:
    """Memory kind, with backfill for records written before the field existed."""
    if meta.get("kind"):
        return meta["kind"]
    return "summary" if "session_summary" in meta.get("categories", "") else "fact"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_retrieved(id_, metadata, score, linked=False) -> RetrievedMemory:
    return RetrievedMemory(
        point_id=id_,
        user_id=metadata["user_id"],
        memory_text=metadata["memory_text"],
        categories=metadata["categories"].split(","),
        date=metadata["date"],
        score=score,
        is_current=int(metadata.get("is_current", 1)),
        importance=int(metadata.get("importance", 5)),
        session_id=metadata.get("session_id", ""),
        kind=kind_of(metadata),
        context=metadata.get("context", ""),
        status=metadata.get("status", "happened"),
        links=list(filter(None, metadata.get("links", "").split(","))),
        linked=linked,
    )


def _corpus(user_id: int, include_old: bool, categories) -> dict:
    """Everything a search needs over a user's memories: ids, metadata, the
    normalised vector matrix, the BM25 index and the link graph.

    Cached until the next write. A `categories` filter bypasses the cache
    (nothing in the codebase passes one, so it is not worth a second key)."""
    key = (user_id, include_old)
    with _lock:
        cached = _corpus_cache.get(key)
        if cached is not None and cached["version"] == _write_version and not categories:
            return cached
        sql = (f"SELECT id, {_META_SQL}, embedding FROM memories "
               f"WHERE user_id = ? AND type != 'core'")
        if not include_old:
            sql += " AND is_current = 1"
        rows = _db().execute(sql, (user_id,)).fetchall()
        version = _write_version

    ids, eligible, blobs = [], {}, []
    for row in rows:
        meta = {k: row[k] for k in _META_COLS}
        if categories:
            stored_cats = [c.strip() for c in meta["categories"].split(",")]
            if not any(c in stored_cats for c in categories):
                continue
        ids.append(row["id"])
        eligible[row["id"]] = (meta, meta["memory_text"])
        blobs.append(row["embedding"])

    matrix, bm25 = None, None
    if ids:
        matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(ids), -1)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        # BM25 indexes content + keywords + context, not just the memory text
        enriched = [
            eligible[i][1] + " "
            + eligible[i][0]["keywords"].replace(",", " ") + " "
            + eligible[i][0]["context"]
            for i in ids
        ]
        bm25 = BM25Okapi([_tokenize(t) for t in enriched])

    graph = nx.Graph()
    for id_ in ids:
        for target in filter(None, eligible[id_][0]["links"].split(",")):
            if target in eligible:
                graph.add_edge(id_, target)

    corpus = {"version": version, "ids": ids, "eligible": eligible,
              "matrix": matrix, "bm25": bm25, "graph": graph}
    if not categories:
        with _lock:
            _corpus_cache[key] = corpus
            # Keep only the few most recent. A chat session touches one or two
            # corpora, but the eval harness runs dozens of users through this
            # process and would otherwise hold every one of them in memory.
            while len(_corpus_cache) > CORPUS_CACHE_SIZE:
                _corpus_cache.pop(next(iter(_corpus_cache)))
    return corpus


async def search_memories(
    search_vector: List[float],
    user_id: int,
    query_text: Optional[str] = None,
    categories: Optional[List[str]] = None,
    top_k: int = 5,
    include_old: bool = False,
    touch: bool = True,
) -> List[RetrievedMemory]:
    """
    Hybrid search over a user's memories.

    Vector similarity and BM25 keyword rankings are fused with reciprocal
    rank fusion (Zep-style), then re-ranked by:

        score = fused relevance                     (0-1, dominant)
              + 0.10 * retention
              + 0.15 * importance                   (1-10 scaled to [0,1])
              - 0.30 if superseded

    Relevance leads deliberately: giving retention and importance full weight
    let an important-but-unrelated memory outrank an exact match, and left a
    superseded fact tied with the fact that replaced it.

    Retention follows Ebbinghaus (MemoryBank): e^(-days_since_recall / strength),
    where strength grows on every retrieval hit — stale memories are demoted in
    ranking, never deleted. BM25 catches exact tokens (names, places) that
    embeddings miss; the cosine floor applies to the vector ranking only.

    Args:
        query_text: Raw query text for the BM25 ranking. If None, vector-only.
        include_old: If True, also search superseded (old) memories.
        touch: If True (default), hits are "rehearsed" — last_accessed resets
            and strength grows, which changes future rankings. Pass False for
            read-only lookups (browsing, evals, diagnostics).
    """
    def _search():
        # Candidate pool: the core profile is injected into every prompt rather
        # than searched, and superseded memories are excluded unless asked for.
        corpus = _corpus(user_id, include_old, categories)
        ids, eligible = corpus["ids"], corpus["eligible"]
        if not ids:
            return []

        fetch_k = max(top_k * 6, 30)

        # Vector ranking: cosine similarity, brute force over the eligible rows
        vec_rank = []
        if USE_VECTOR:
            try:
                q = np.asarray(search_vector, dtype=np.float32)
                sims = corpus["matrix"] @ (q / (np.linalg.norm(q) + 1e-9))
                order = np.argsort(-sims)[:fetch_k]
                vec_rank = [ids[i] for i in order if sims[i] >= RELEVANCE_FLOOR]
            except Exception as e:
                log.warning("vector search failed, falling back to keyword ranking: %s", e)

        # BM25 keyword ranking over enriched note text (content + keywords + context)
        bm25_rank = []
        if query_text and USE_BM25 and corpus["bm25"] is not None:
            scores = corpus["bm25"].get_scores(_tokenize(query_text))
            ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
            bm25_rank = [i for i, s in ranked if s > 0][:fetch_k]

        # Personalized PageRank over the link graph, seeded by the direct hits —
        # surfaces memories connected to what matched, even if they didn't match
        ppr_rank = []
        seeds = list(dict.fromkeys(vec_rank + bm25_rank))[:PPR_SEEDS] if USE_PPR else []
        graph = corpus["graph"]
        if seeds and graph.number_of_edges() > 0:
            personalization = {n: (1.0 if n in seeds else 0.0) for n in graph.nodes}
            if any(personalization.values()):
                try:
                    pr = nx.pagerank(graph, personalization=personalization)
                    # nodes the seeds cannot reach still carry a numerical crumb
                    # (~1e-6); ranking them handed unrelated memories RRF credit
                    cutoff = max(pr.values()) * PPR_MIN_SHARE
                    ppr_rank = [n for n, s in sorted(pr.items(), key=lambda x: x[1], reverse=True)
                                if s >= cutoff][:fetch_k]
                except Exception as e:
                    log.warning("pagerank failed, ranking without it: %s", e)

        # Reciprocal rank fusion across the three rankings
        rrf: dict = {}
        for rank_list in (vec_rank, bm25_rank, ppr_rank):
            for pos, id_ in enumerate(rank_list):
                rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (RRF_K + pos + 1)
        if not rrf:
            return []
        max_rrf = max(rrf.values())

        now = datetime.now().timestamp()
        scored = []
        for id_, fused in rrf.items():
            meta = eligible[id_][0]
            last_accessed = float(meta.get("last_accessed") or meta.get("timestamp") or now)
            strength = float(meta.get("strength") or STRENGTH_INIT)
            retention = math.exp(-((now - last_accessed) / 86400.0) / strength)
            importance = int(meta.get("importance", 5)) / 10.0
            stale = STALE_PENALTY if int(meta.get("is_current", 1)) == 0 else 0.0
            scored.append((
                fused / max_rrf
                + RETENTION_WEIGHT * retention
                + IMPORTANCE_WEIGHT * importance
                - stale,
                id_, meta,
            ))

        scored.sort(key=lambda s: s[0], reverse=True)
        top = scored[:top_k]

        # Touch winners: recency resets and strength grows (Ebbinghaus rehearsal)
        if top and touch:
            rehearsed = [
                (now, float(meta.get("strength") or STRENGTH_INIT) + STRENGTH_PER_RECALL, id_, meta)
                for _, id_, meta in top
            ]
            with _lock:
                db = _db()
                db.executemany(
                    "UPDATE memories SET last_accessed = ?, strength = ? WHERE id = ?",
                    [(a, s, i) for a, s, i, _ in rehearsed],
                )
                db.commit()
                # Rehearsal touches only these two fields, so update the cached
                # metadata in place rather than throwing the whole corpus away —
                # otherwise every search would invalidate the cache it just built.
                for accessed, strength_, _, meta in rehearsed:
                    meta["last_accessed"] = accessed
                    meta["strength"] = strength_

        out = [_build_retrieved(id_, meta, score) for score, id_, meta in top]

        # 1-hop link expansion: bring along memories linked to the winners
        top_ids = {id_ for _, id_, _ in top}
        expansion = []
        for _, _, meta in top:
            for lid in filter(None, meta.get("links", "").split(",")):
                if lid in eligible and lid not in top_ids and lid not in expansion:
                    expansion.append(lid)
        for lid in expansion[:LINK_EXPANSION_CAP]:
            out.append(_build_retrieved(lid, eligible[lid][0], 0.0, linked=True))

        return out

    return await asyncio.to_thread(_search)


async def fetch_user_records_raw(user_id: int, include_embeddings: bool = True) -> dict:
    """Raw dump for a user — used by consolidation. Only dedup needs the
    embeddings; every other stage was paying 384 floats per record for nothing."""
    def _fetch():
        cols = f"id, {_META_SQL}" + (", embedding" if include_embeddings else "")
        with _lock:
            rows = _db().execute(
                f"SELECT {cols} FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {
            "ids": [r["id"] for r in rows],
            "metadatas": [{k: r[k] for k in _META_COLS} for r in rows],
            "documents": [r["memory_text"] for r in rows],
            "embeddings": [_from_blob(r["embedding"]) for r in rows] if include_embeddings else None,
        }
    return await asyncio.to_thread(_fetch)


# core memory (always-in-prompt user profile)

async def get_core_memory(user_id: int) -> str:
    def _get():
        with _lock:
            row = _db().execute(
                "SELECT memory_text FROM memories WHERE id = ?", (f"core_{user_id}",)
            ).fetchone()
        return row["memory_text"] if row else ""
    return await asyncio.to_thread(_get)


async def set_core_memory(user_id: int, text: str, embedding: List[float]):
    def _set():
        now = datetime.now()
        with _lock:
            db = _db()
            db.execute(
                "INSERT INTO memories (id, user_id, memory_text, categories, date, "
                "timestamp, saved_at, is_current, type, embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET memory_text=excluded.memory_text, "
                "date=excluded.date, timestamp=excluded.timestamp, "
                "saved_at=excluded.saved_at, embedding=excluded.embedding",
                (f"core_{user_id}", user_id, text, "core", now.isoformat(),
                 now.timestamp(), now.isoformat(), 1, "core", _to_blob(embedding)),
            )
            db.commit()
            _bump()
    await asyncio.to_thread(_set)


async def fetch_all_user_records(user_id: int) -> List[RetrievedMemory]:
    def _fetch():
        with _lock:
            rows = _db().execute(
                f"SELECT id, {_META_SQL} FROM memories "
                f"WHERE user_id = ? AND type != 'core'", (user_id,)
            ).fetchall()
        return [_build_retrieved(r["id"], {k: r[k] for k in _META_COLS}, 0.0) for r in rows]
    return await asyncio.to_thread(_fetch)


async def get_all_categories(user_id: int) -> List[str]:
    def _fetch():
        with _lock:
            rows = _db().execute(
                "SELECT categories FROM memories WHERE user_id = ? AND type != 'core'",
                (user_id,),
            ).fetchall()
        seen = set()
        for row in rows:
            for cat in row["categories"].split(","):
                if cat.strip():
                    seen.add(cat.strip())
        return sorted(seen)
    return await asyncio.to_thread(_fetch)


# display helper

def stringify_retrieved_point(retrieved_memory: RetrievedMemory) -> str:
    status_tag = "" if retrieved_memory.is_current else " [OLD/SUPERSEDED]"
    linked_tag = " [LINKED]" if retrieved_memory.linked else ""
    # an intention must never read like something that happened
    state = retrieved_memory.status
    state_tag = f" [{state.upper()}, did not happen]" if state in ("planned", "considered") else ""
    saved = retrieved_memory.date[:19].replace("T", " ") if retrieved_memory.date else "unknown"
    context = f" Context: {retrieved_memory.context}" if retrieved_memory.context else ""
    return (
        f"{retrieved_memory.memory_text}{status_tag}{linked_tag}{state_tag} "
        f"(Categories: {retrieved_memory.categories}) "
        f"[Saved: {saved}] "
        f"Score: {retrieved_memory.score:.2f}{context}"
    )


if __name__ == "__main__":
    asyncio.run(create_collection())
