from datetime import datetime
import numpy as np
import dspy
from memory.embedding_generation import generate_embeddings
from memory.memory_store import (
    STRENGTH_INIT,
    EmbeddedMemory,
    add_memory,
    fetch_user_records_raw,
    kind_of,
    mark_memory_old,
)
import os
from dotenv import load_dotenv

load_dotenv()

DEDUP_SIMILARITY = 0.9     # cosine similarity above which memories count as duplicates
REFLECTION_THRESHOLD = 40  # summed importance of fresh facts that triggers reflection

_lm = dspy.LM(
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
    api_key=os.getenv("OPEN_ROUTER_KEY"),
)


class MergeMemoriesSignature(dspy.Signature):
    """
    Merge near-duplicate memories about a user into a single memory.
    Keep every distinct detail; drop only repetition. Use ONLY information
    present in the inputs — never invent details. One or two sentences.
    """

    memories: list[str] = dspy.InputField()
    merged: str = dspy.OutputField()


class ReflectionSignature(dspy.Signature):
    """
    You are given recent memories about a user. State 2-3 higher-level insights
    that emerge from combining them (patterns, goals, traits). Only state
    insights strongly supported by the memories — never guess or invent.
    Return an empty list if nothing meaningful emerges.
    """

    recent_memories: list[str] = dspy.InputField()
    insights: list[str] = dspy.OutputField()


_merger = dspy.Predict(MergeMemoriesSignature)
_reflector = dspy.Predict(ReflectionSignature)


def _cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _clusters(items) -> list[list[int]]:
    """Union-find over pairs above DEDUP_SIMILARITY; returns groups of 2+ indices."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _cosine(items[i][3], items[j][3]) >= DEDUP_SIMILARITY:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


async def dedup_memories(user_id: int, session_id: str = "") -> str:
    recs = await fetch_user_records_raw(user_id)

    by_kind: dict = {}
    for id_, meta, doc, emb in zip(
        recs["ids"], recs["metadatas"], recs["documents"], recs["embeddings"]
    ):
        if meta.get("type") == "core" or int(meta.get("is_current", 1)) == 0:
            continue
        # derived memories never merge with the raw facts they came from
        by_kind.setdefault(kind_of(meta), []).append((id_, meta, doc, emb))

    merged_count = 0
    for kind, items in by_kind.items():
        for group in _clusters(items):
            members = [items[i] for i in group]
            try:
                with dspy.context(lm=_lm):
                    out = await _merger.acall(memories=[m[2] for m in members])
                text = out.merged.strip()
            except Exception:
                continue
            if not text:
                continue

            categories = sorted({c.strip() for m in members for c in m[1]["categories"].split(",")})
            embedding = (await generate_embeddings([text]))[0]
            await add_memory([
                EmbeddedMemory(
                    id="",
                    user_id=user_id,
                    memory_text=text,
                    categories=categories,
                    embedding=embedding,
                    date=min(m[1]["date"] for m in members),
                    importance=max(int(m[1].get("importance", 5)) for m in members),
                    # merged memory inherits the group's accumulated recall strength
                    strength=sum(float(m[1].get("strength", STRENGTH_INIT)) for m in members),
                    session_id=session_id,
                    kind=kind,
                )
            ])
            for m in members:
                await mark_memory_old(m[0])
            merged_count += 1

    return f"merged {merged_count} duplicate group(s)" if merged_count else "no duplicates found"


async def maybe_reflect(user_id: int, session_id: str = "") -> list[str]:
    """
    Generative Agents-style reflection: once enough important facts have
    accumulated since the last reflection, distill them into insight memories.
    """
    recs = await fetch_user_records_raw(user_id)

    last_reflection = ""
    facts = []
    for meta in recs["metadatas"]:
        if meta.get("type") == "core":
            continue
        kind = kind_of(meta)
        if kind == "insight":
            last_reflection = max(last_reflection, meta.get("saved_at", ""))
        elif kind == "fact" and int(meta.get("is_current", 1)) == 1:
            facts.append(meta)

    fresh = [m for m in facts if m.get("saved_at", "") > last_reflection]
    if sum(int(m.get("importance", 5)) for m in fresh) < REFLECTION_THRESHOLD:
        return []

    fresh.sort(key=lambda m: m.get("saved_at", ""))
    try:
        with dspy.context(lm=_lm):
            out = await _reflector.acall(recent_memories=[m["memory_text"] for m in fresh[-30:]])
        insights = [s.strip() for s in out.insights if s.strip()][:3]
    except Exception:
        return []

    for text in insights:
        embedding = (await generate_embeddings([text]))[0]
        await add_memory([
            EmbeddedMemory(
                id="",
                user_id=user_id,
                memory_text=text,
                categories=["insight"],
                embedding=embedding,
                date=datetime.now().isoformat(),
                importance=7,
                session_id=session_id,
                kind="insight",
            )
        ])
    return insights