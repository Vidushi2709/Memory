import math
import re
from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import chromadb
import asyncio

COLLECTION_NAME = "memories_bring_back_memories"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

STRENGTH_INIT = 5.0        # initial Ebbinghaus strength, in days of decay scale
STRENGTH_PER_RECALL = 1.0  # strength gained each time a memory is retrieved
RRF_K = 60                 # reciprocal rank fusion constant
RELEVANCE_FLOOR = 0.3      # min cosine similarity to enter the vector ranking

_chroma = chromadb.PersistentClient(path="./chroma_db")  # persists to disk across sessions


def _get_collection():
    return _chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance
    )


class EmbeddedMemory(BaseModel):
    id: str
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


# collection setup 

async def create_collection():
    """Ensures the collection exists (ChromaDB creates it on first use)."""
    await asyncio.to_thread(_get_collection)
    print(f"Collection '{COLLECTION_NAME}' ready.")


# write operations 

async def add_memory(embedded_memories: List[EmbeddedMemory]):
    def _add():
        col = _get_collection()
        now = datetime.now()
        col.upsert(
            ids=[uuid4().hex for _ in embedded_memories],
            embeddings=[m.embedding for m in embedded_memories],
            metadatas=[
                {
                    "user_id":     m.user_id,
                    "memory_text": m.memory_text,
                    "categories":  ",".join(m.categories),  # ChromaDB metadata values must be str/int/float
                    "date":        m.date,
                    "timestamp":   datetime.fromisoformat(m.date).timestamp(),
                    "saved_at":    now.isoformat(),           # wall-clock time memory was written
                    "is_current":  m.is_current,             # 1=active, 0=superseded
                    "importance":  m.importance,
                    "last_accessed": now.timestamp(),        # epoch of last retrieval hit
                    "strength":    m.strength,
                    "session_id":  m.session_id,
                    "kind":        m.kind,
                }
                for m in embedded_memories
            ],
            documents=[m.memory_text for m in embedded_memories],
        )
    await asyncio.to_thread(_add)


async def delete_user_records(user_id: int):
    def _delete():
        col = _get_collection()
        col.delete(where={"user_id": {"$eq": user_id}})
    await asyncio.to_thread(_delete)


async def delete_records(point_ids: List[str]):
    def _delete():
        col = _get_collection()
        col.delete(ids=point_ids)
    await asyncio.to_thread(_delete)


async def mark_memory_old(point_id: str):
    """
    Mark an existing memory as superseded (is_current=0) without deleting it.
    This preserves history so questions like "where did I live before?" can
    still be answered by searching with include_old=True.
    """
    def _mark():
        col = _get_collection()
        # Fetch the existing record so we can re-upsert with updated metadata
        result = col.get(ids=[point_id], include=["metadatas", "embeddings", "documents"])
        if not result["ids"]:
            return  # already gone
        meta = result["metadatas"][0]
        embedding = result["embeddings"][0]
        document = result["documents"][0]
        meta["is_current"] = 0
        meta["superseded_at"] = datetime.now().isoformat()
        col.upsert(
            ids=[point_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[document],
        )
    await asyncio.to_thread(_mark)


# read operations 

def kind_of(meta) -> str:
    """Memory kind, with backfill for records written before the field existed."""
    if "kind" in meta:
        return meta["kind"]
    return "summary" if "session_summary" in meta.get("categories", "") else "fact"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_retrieved(id_, metadata, score) -> RetrievedMemory:
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
    )


async def search_memories(
    search_vector: List[float],
    user_id: int,
    query_text: Optional[str] = None,
    categories: Optional[List[str]] = None,
    top_k: int = 5,
    include_old: bool = False,
) -> List[RetrievedMemory]:
    """
    Hybrid search over a user's memories.

    Vector similarity and BM25 keyword rankings are fused with reciprocal
    rank fusion (Zep-style), then re-ranked by:

        score = fused relevance + retention + importance (1-10 scaled to [0,1])

    Retention follows Ebbinghaus (MemoryBank): e^(-days_since_recall / strength),
    where strength grows on every retrieval hit — stale memories are demoted in
    ranking, never deleted. BM25 catches exact tokens (names, places) that
    embeddings miss; the cosine floor applies to the vector ranking only.

    Args:
        query_text: Raw query text for the BM25 ranking. If None, vector-only.
        include_old: If True, also search superseded (old) memories.

    NOTE: ChromaDB 1.4.x metadata filters only support
    $eq / $ne / $gt / $gte / $lt / $lte / $in / $nin.
    `$contains` is NOT supported for metadata fields.

    We therefore filter by `user_id` (supported) and apply
    the optional `categories` / `is_current` checks client-side.
    """
    def _search():
        col = _get_collection()
        where: dict = {"user_id": {"$eq": user_id}}

        # Full candidate pool for this user (client-side filters)
        all_recs = col.get(where=where, include=["metadatas", "documents"])
        eligible: dict = {}
        for id_, meta, doc in zip(all_recs["ids"], all_recs["metadatas"], all_recs["documents"]):
            # Core profile record is injected into every prompt, not searched
            if meta.get("type") == "core":
                continue
            if not include_old and int(meta.get("is_current", 1)) == 0:
                continue
            if categories:
                stored_cats = [c.strip() for c in meta["categories"].split(",")]
                if not any(c in stored_cats for c in categories):
                    continue
            eligible[id_] = (meta, doc)
        if not eligible:
            return []

        fetch_k = max(top_k * 6, 30)

        # Vector ranking
        vec_rank = []
        try:
            results = col.query(
                query_embeddings=[search_vector],
                n_results=min(fetch_k, len(all_recs["ids"])),
                where=where,
                include=["distances"],
            )
            for id_, dist in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                if id_ in eligible and 1.0 - (dist / 2.0) >= RELEVANCE_FLOOR:
                    vec_rank.append(id_)
        except Exception:
            pass

        # BM25 keyword ranking
        bm25_rank = []
        if query_text:
            ids = list(eligible)
            bm25 = BM25Okapi([_tokenize(eligible[i][1]) for i in ids])
            scores = bm25.get_scores(_tokenize(query_text))
            ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
            bm25_rank = [i for i, s in ranked if s > 0][:fetch_k]

        # Reciprocal rank fusion across the two rankings
        rrf: dict = {}
        for rank_list in (vec_rank, bm25_rank):
            for pos, id_ in enumerate(rank_list):
                rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (RRF_K + pos + 1)
        if not rrf:
            return []
        max_rrf = max(rrf.values())

        now = datetime.now().timestamp()
        scored = []
        for id_, fused in rrf.items():
            meta = eligible[id_][0]
            last_accessed = float(meta.get("last_accessed", meta.get("timestamp", now)))
            strength = float(meta.get("strength", STRENGTH_INIT))
            retention = math.exp(-((now - last_accessed) / 86400.0) / strength)
            importance = int(meta.get("importance", 5)) / 10.0
            scored.append((fused / max_rrf + retention + importance, id_, meta))

        scored.sort(key=lambda s: s[0], reverse=True)
        top = scored[:top_k]

        # Touch winners: recency resets and strength grows (Ebbinghaus rehearsal)
        if top:
            hit = col.get(ids=[id_ for _, id_, _ in top],
                          include=["metadatas", "embeddings", "documents"])
            for meta in hit["metadatas"]:
                meta["last_accessed"] = now
                meta["strength"] = float(meta.get("strength", STRENGTH_INIT)) + STRENGTH_PER_RECALL
            col.upsert(
                ids=hit["ids"],
                embeddings=hit["embeddings"],
                metadatas=hit["metadatas"],
                documents=hit["documents"],
            )

        return [_build_retrieved(id_, meta, score) for score, id_, meta in top]

    return await asyncio.to_thread(_search)


async def fetch_user_records_raw(user_id: int) -> dict:
    """Raw ChromaDB dump for a user, embeddings included — used by consolidation."""
    def _fetch():
        col = _get_collection()
        return col.get(
            where={"user_id": {"$eq": user_id}},
            include=["metadatas", "embeddings", "documents"],
        )
    return await asyncio.to_thread(_fetch)


# core memory (always-in-prompt user profile)

async def get_core_memory(user_id: int) -> str:
    def _get():
        col = _get_collection()
        result = col.get(ids=[f"core_{user_id}"], include=["documents"])
        return result["documents"][0] if result["ids"] else ""
    return await asyncio.to_thread(_get)


async def set_core_memory(user_id: int, text: str, embedding: List[float]):
    def _set():
        col = _get_collection()
        now = datetime.now()
        col.upsert(
            ids=[f"core_{user_id}"],
            embeddings=[embedding],
            metadatas=[{
                "user_id":     user_id,
                "type":        "core",
                "memory_text": text,
                "categories":  "core",
                "date":        now.isoformat(),
                "timestamp":   now.timestamp(),
                "saved_at":    now.isoformat(),
                "is_current":  1,
            }],
            documents=[text],
        )
    await asyncio.to_thread(_set)


async def fetch_all_user_records(user_id: int) -> List[RetrievedMemory]:
    def _fetch():
        col = _get_collection()
        results = col.get(
            where={"user_id": {"$eq": user_id}},
            include=["metadatas"],
        )
        return [
            _build_retrieved(id_, meta, 0.0)
            for id_, meta in zip(results["ids"], results["metadatas"])
            if meta.get("type") != "core"
        ]
    return await asyncio.to_thread(_fetch)


async def get_all_categories(user_id: int) -> List[str]:
    def _fetch():
        col = _get_collection()
        results = col.get(
            where={"user_id": {"$eq": user_id}},
            include=["metadatas"],
        )
        seen = set()
        for meta in results["metadatas"]:
            if meta.get("type") == "core":
                continue
            for cat in meta["categories"].split(","):
                seen.add(cat.strip())
        return sorted(seen)
    return await asyncio.to_thread(_fetch)


# display helper 

def stringify_retrieved_point(retrieved_memory: RetrievedMemory) -> str:
    status_tag = "" if retrieved_memory.is_current else " [OLD/SUPERSEDED]"
    saved = retrieved_memory.date[:19].replace("T", " ") if retrieved_memory.date else "unknown"
    return (
        f"{retrieved_memory.memory_text}{status_tag} "
        f"(Categories: {retrieved_memory.categories}) "
        f"[Saved: {saved}] "
        f"Score: {retrieved_memory.score:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(create_collection())