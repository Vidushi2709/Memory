from datetime import datetime
from typing import Optional, List
from uuid import uuid4
from pydantic import BaseModel
import chromadb
import asyncio

COLLECTION_NAME = "memories_bring_back_memories"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

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


class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float
    is_current: int = 1  # 1 = current/active, 0 = superseded/old


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

def _build_retrieved(id_, metadata, score) -> RetrievedMemory:
    return RetrievedMemory(
        point_id=id_,
        user_id=metadata["user_id"],
        memory_text=metadata["memory_text"],
        categories=metadata["categories"].split(","),
        date=metadata["date"],
        score=score,
        is_current=int(metadata.get("is_current", 1)),
    )


async def search_memories(
    search_vector: List[float],
    user_id: int,
    categories: Optional[List[str]] = None,
    top_k: int = 5,
    include_old: bool = False,
) -> List[RetrievedMemory]:
    """
    Semantic search over a user's memories.

    Args:
        include_old: If True, also search superseded (old) memories. Use this
                     when the user asks historical questions like
                     "where did I live before?".

    NOTE: ChromaDB 1.4.x metadata filters only support
    $eq / $ne / $gt / $gte / $lt / $lte / $in / $nin.
    `$contains` is NOT supported for metadata fields.

    We therefore filter by `user_id` (supported) and apply
    the optional `categories` / `is_current` checks client-side.
    """
    def _search():
        col = _get_collection()

        # Only filter by user_id — supported by all ChromaDB versions
        where: dict = {"user_id": {"$eq": user_id}}

        # Retrieve more results than we need so the client-side
        # category / is_current filters still have enough candidates.
        fetch_k = max(top_k * 6, 30) if (categories or not include_old) else max(top_k * 4, 20)

        try:
            results = col.query(
                query_embeddings=[search_vector],
                n_results=fetch_k,
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception:
            # Collection might be empty — return empty list gracefully
            return []

        out = []
        if not results["ids"] or not results["ids"][0]:
            return out

        for id_, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score in [0, 1]
            score = 1.0 - (dist / 2.0)
            if score < 0.5:
                continue

            # Skip old/superseded memories unless explicitly requested
            if not include_old and int(meta.get("is_current", 1)) == 0:
                continue

            # Client-side category filter
            if categories:
                stored_cats = [c.strip() for c in meta["categories"].split(",")]
                if not any(c in stored_cats for c in categories):
                    continue

            out.append(_build_retrieved(id_, meta, score))
            if len(out) >= top_k:
                break

        return out

    return await asyncio.to_thread(_search)


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
        f"Relevance: {retrieved_memory.score:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(create_collection())