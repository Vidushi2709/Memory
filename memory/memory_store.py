import math
import re
import threading
from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import networkx as nx
import asyncio

COLLECTION_NAME = "memories_bring_back_memories"
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
IMPORTANCE_WEIGHT = 0.15   # importance and retention only break ties — relevance leads
RETENTION_WEIGHT = 0.10
STALE_PENALTY = 0.30       # a superseded memory must lose to a current one of equal relevance

# ChromaDB batches telemetry events in a plain dict with no lock, so two threads
# can race to delete the same key and raise KeyError in the middle of a query —
# which surfaces here as a whole sleep pass failing. anonymized_telemetry=False
# only suppresses the network send; the unsafe batching still runs on every
# get(). Neutralise the code path itself. Reproduced at ~1.3% of concurrent
# reads before this, zero after.
try:
    from chromadb.telemetry.product.posthog import Posthog

    Posthog.capture = lambda self, event: None
except Exception:  # library layout changed — the setting below still applies
    pass

_chroma = chromadb.PersistentClient(
    path="./chroma_db",  # persists to disk across sessions
    settings=Settings(anonymized_telemetry=False),
)
_write_lock = threading.Lock()  # writes run in worker threads and can overlap


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
    """Ensures the collection exists (ChromaDB creates it on first use)."""
    await asyncio.to_thread(_get_collection)
    print(f"Collection '{COLLECTION_NAME}' ready.")


# write operations 

async def add_memory(embedded_memories: List[EmbeddedMemory]) -> List[str]:
    def _add():
        now = datetime.now()
        ids = [uuid4().hex for _ in embedded_memories]
        metadatas = [
            {
                "user_id":     m.user_id,
                "memory_text": m.memory_text,
                "categories":  ",".join(m.categories),  # ChromaDB metadata values must be str/int/float
                "date":        m.date,
                "timestamp":   to_epoch(m.date),
                "saved_at":    now.isoformat(),           # wall-clock time memory was written
                "is_current":  m.is_current,             # 1=active, 0=superseded
                "importance":  m.importance,
                "last_accessed": now.timestamp(),        # epoch of last retrieval hit
                "strength":    m.strength,
                "session_id":  m.session_id,
                "kind":        m.kind,
                "keywords":    ",".join(m.keywords),
                "context":     m.context,
                "status":      m.status,                 # happened/planned/considered/ongoing
                "links":       ",".join(m.links),        # comma-separated point ids
            }
            for m in embedded_memories
        ]
        with _write_lock:
            _get_collection().upsert(
                ids=ids,
                embeddings=[m.embedding for m in embedded_memories],
                metadatas=metadatas,
                documents=[m.memory_text for m in embedded_memories],
            )
        return ids
    return await asyncio.to_thread(_add)


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


def _update_meta_locked(point_id: str, updates: dict):
    """Re-upsert a record with modified metadata (ChromaDB has no partial update).
    Caller must already hold _write_lock."""
    col = _get_collection()
    result = col.get(ids=[point_id], include=["metadatas", "embeddings", "documents"])
    if not result["ids"]:
        return  # already gone
    meta = result["metadatas"][0]
    meta.update(updates)
    col.upsert(
        ids=[point_id],
        embeddings=[result["embeddings"][0]],
        metadatas=[meta],
        documents=[result["documents"][0]],
    )


def _update_meta_sync(point_id: str, updates: dict):
    with _write_lock:
        _update_meta_locked(point_id, updates)


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
        col = _get_collection()
        for a, b in [(point_id, lid) for lid in linked_ids if lid != point_id]:
            for src, dst in ((a, b), (b, a)):
                # read and write under ONE lock: releasing between them let a
                # concurrent link add overwrite the list this one just read
                with _write_lock:
                    result = col.get(ids=[src], include=["metadatas"])
                    if not result["ids"]:
                        continue
                    existing = set(filter(None, result["metadatas"][0].get("links", "").split(",")))
                    if dst in existing:
                        continue
                    existing.add(dst)
                    _update_meta_locked(src, {"links": ",".join(sorted(existing))})
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
    if "kind" in meta:
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
            # ask for the ineligible ones on top: they are filtered out below,
            # and without the headroom a store full of superseded memories
            # returns almost nothing
            ineligible = len(all_recs["ids"]) - len(eligible)
            results = col.query(
                query_embeddings=[search_vector],
                n_results=min(fetch_k + ineligible, len(all_recs["ids"])),
                where=where,
                include=["distances"],
            )
            for id_, dist in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB cosine distance is 1 - similarity: 0 identical,
                # 1 unrelated, 2 opposite. Halving it rated unrelated memories
                # 0.5 and let everything through the floor.
                if id_ in eligible and 1.0 - dist >= RELEVANCE_FLOOR:
                    vec_rank.append(id_)
        except Exception:
            pass

        # BM25 keyword ranking over enriched note text (content + keywords + context)
        bm25_rank = []
        if query_text:
            ids = list(eligible)
            enriched = [
                eligible[i][1] + " "
                + eligible[i][0].get("keywords", "").replace(",", " ") + " "
                + eligible[i][0].get("context", "")
                for i in ids
            ]
            bm25 = BM25Okapi([_tokenize(t) for t in enriched])
            scores = bm25.get_scores(_tokenize(query_text))
            ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
            bm25_rank = [i for i, s in ranked if s > 0][:fetch_k]

        # Personalized PageRank over the link graph, seeded by the direct hits —
        # surfaces memories connected to what matched, even if they didn't match
        ppr_rank = []
        seeds = list(dict.fromkeys(vec_rank + bm25_rank))[:PPR_SEEDS] if USE_PPR else []
        graph = nx.Graph()
        for id_, (meta, _) in eligible.items():
            for target in filter(None, meta.get("links", "").split(",")):
                if target in eligible:
                    graph.add_edge(id_, target)
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
                except Exception:
                    pass

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
            last_accessed = float(meta.get("last_accessed", meta.get("timestamp", now)))
            strength = float(meta.get("strength", STRENGTH_INIT))
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
        if top:
            with _write_lock:
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
    """Raw ChromaDB dump for a user — used by consolidation. Only dedup needs the
    embeddings; every other stage was paying 384 floats per record for nothing."""
    def _fetch():
        col = _get_collection()
        include = ["metadatas", "documents"]
        if include_embeddings:
            include.append("embeddings")
        return col.get(where={"user_id": {"$eq": user_id}}, include=include)
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