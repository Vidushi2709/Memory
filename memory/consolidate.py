import re
from datetime import datetime
import numpy as np
import dspy
from memory.embedding_generation import generate_embeddings
from memory.memory_store import (
    STRENGTH_INIT,
    EmbeddedMemory,
    add_links,
    add_memory,
    fetch_user_records_raw,
    get_core_memory,
    kind_of,
    mark_memory_old,
    set_context,
    set_core_memory,
)
import os
from dotenv import load_dotenv

load_dotenv()

DEDUP_SIMILARITY = 0.9     # cosine similarity above which memories count as duplicates
REFLECTION_THRESHOLD = 40  # summed importance of fresh facts that triggers reflection
EVOLVE_NEW_CAP = 3         # newest session memories considered for evolution
EVOLVE_LINK_CAP = 2        # linked neighbors re-examined per new memory

_lm = dspy.LM(
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
    api_key=os.getenv("OPEN_ROUTER_KEY"),
    temperature=0.0,
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


class EvolveContextSignature(dspy.Signature):
    """
    Given a NEW memory about a user and one EXISTING related memory, decide
    whether the existing memory's one-sentence context should be reinterpreted
    in light of the new memory (e.g. "bought hiking boots" becomes trip
    preparation once a trek is planned). Output the updated one-sentence
    context, or an empty string if no change is needed. Use ONLY stated
    information — never invent details.
    """

    new_memory: str = dspy.InputField()
    existing_memory: str = dspy.InputField()
    existing_context: str = dspy.InputField()
    updated_context: str = dspy.OutputField()


class ReconcilePlanSignature(dspy.Signature):
    """
    Reconcile a session's NEW memories with the user's EXISTING memories in
    one plan. Memories are labelled N0, N1, ... (new) and E0, E1, ... (existing).

    - supersede_pairs: pairs "Nx:Ey" where the new memory makes the existing one
      outdated (the fact changed — moved cities, changed jobs). The existing
      memory will be marked old but preserved in history.
    - link_pairs: pairs "Nx:Ey" that are related and BOTH still true (same
      topic, person, or activity). Prefer linking over superseding when the new
      memory adds detail rather than replacing the fact.

    Only include pairs you are confident about. Output empty strings if none.
    """

    new_memories: list[str] = dspy.InputField()
    existing_memories: list[str] = dspy.InputField()
    supersede_pairs: str = dspy.OutputField(desc='Comma-separated "Nx:Ey" pairs, e.g. "N0:E2,N1:E5". Empty if none.')
    link_pairs: str = dspy.OutputField(desc='Comma-separated "Nx:Ey" pairs. Empty if none.')


class CoreMemorySignature(dspy.Signature):
    """
    Maintain a short always-visible profile of the user: name, location, work,
    and their most important preferences. Plain sentences, under 80 words.
    Use ONLY information present in current_core or new_facts — NEVER invent,
    guess, or embellish details that were not stated.
    Fold the new facts into current_core, dropping nothing that is still true.
    If the new facts change nothing, return current_core unchanged.
    """

    current_core: str = dspy.InputField()
    new_facts: list[str] = dspy.InputField()
    updated_core: str = dspy.OutputField()


_merger = dspy.Predict(MergeMemoriesSignature)
_reflector = dspy.Predict(ReflectionSignature)
_evolver = dspy.Predict(EvolveContextSignature)
_reconciler = dspy.Predict(ReconcilePlanSignature)
_core_updater = dspy.Predict(CoreMemorySignature)


def _parse_pairs(raw: str, n_new: int, n_existing: int) -> list[tuple[int, int]]:
    pairs = []
    for chunk in (raw or "").replace(" ", "").split(","):
        m = re.fullmatch(r"[Nn](\d+):[Ee](\d+)", chunk)
        if m and int(m.group(1)) < n_new and int(m.group(2)) < n_existing:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def _as_set(session_ids) -> set:
    return {session_ids} if isinstance(session_ids, str) else set(session_ids)


async def reconcile_session(user_id: int, session_ids) -> str:
    """
    One large-context call reconciles the given sessions' new memories against
    the existing store: supersedes changed facts, links related ones.
    """
    sids = _as_set(session_ids)
    recs = await fetch_user_records_raw(user_id)
    new, existing = [], []
    for id_, meta in zip(recs["ids"], recs["metadatas"]):
        if meta.get("type") == "core" or int(meta.get("is_current", 1)) == 0:
            continue
        if kind_of(meta) != "fact":
            continue
        if meta.get("session_id") in sids:
            new.append((id_, meta))
        else:
            existing.append((id_, meta))
    if not new or not existing:
        return "nothing to reconcile"

    existing.sort(key=lambda x: x[1].get("saved_at", ""), reverse=True)
    existing = existing[:40]

    fmt = lambda tag, i, meta: f"{tag}{i}: {meta['memory_text']} (date: {meta.get('date', '')[:10]})"
    try:
        with dspy.context(lm=_lm):
            out = await _reconciler.acall(
                new_memories=[fmt("N", i, m) for i, (_, m) in enumerate(new)],
                existing_memories=[fmt("E", i, m) for i, (_, m) in enumerate(existing)],
            )
    except Exception:
        return "reconcile call failed"

    superseded = 0
    for ni, ei in _parse_pairs(out.supersede_pairs, len(new), len(existing)):
        await mark_memory_old(existing[ei][0])
        await add_links(new[ni][0], [existing[ei][0]])  # keep the history reachable
        superseded += 1
    linked = 0
    for ni, ei in _parse_pairs(out.link_pairs, len(new), len(existing)):
        await add_links(new[ni][0], [existing[ei][0]])
        linked += 1
    return f"{superseded} superseded, {linked} linked"


async def refresh_core_memory(user_id: int, session_ids):
    sids = _as_set(session_ids)
    recs = await fetch_user_records_raw(user_id)
    facts = [
        meta["memory_text"]
        for meta in recs["metadatas"]
        if meta.get("session_id") in sids and kind_of(meta) == "fact"
        and meta.get("type") != "core" and int(meta.get("is_current", 1)) == 1
    ]
    if not facts:
        return
    current_core = await get_core_memory(user_id)
    try:
        with dspy.context(lm=_lm):
            out = await _core_updater.acall(current_core=current_core, new_facts=facts)
        new_core = out.updated_core.strip()
    except Exception:
        return
    if new_core and new_core != current_core:
        embedding = (await generate_embeddings([new_core]))[0]
        await set_core_memory(user_id, new_core, embedding)


async def sleep_pass(user_id: int, session_ids) -> str:
    """
    Sleep-time consolidation (Letta pattern): the online path only appends;
    this pass holds all write authority over reconciliation and derived memory.

    Accepts one session id or a batch of them — a batch lets the pass run
    periodically over long histories without leaving sessions unreconciled.
    """
    sids = _as_set(session_ids)
    latest = max(sids)
    parts = [await reconcile_session(user_id, sids)]
    parts.append(await dedup_memories(user_id, latest))
    evolved = await evolve_memories(user_id, sids)
    if evolved:
        parts.append(f"{evolved} context(s) evolved")
    insights = await maybe_reflect(user_id, latest)
    if insights:
        parts.append(f"{len(insights)} insight(s)")
    await refresh_core_memory(user_id, sids)
    return ", ".join(p for p in parts if p)


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


async def evolve_memories(user_id: int, session_ids) -> int:
    """
    A-Mem memory evolution: new memories may change how their linked neighbors
    should be read — rewrite those neighbors' context lines.
    """
    sids = _as_set(session_ids)
    recs = await fetch_user_records_raw(user_id)
    by_id = dict(zip(recs["ids"], recs["metadatas"]))

    new_metas = [
        m for m in recs["metadatas"]
        if m.get("session_id") in sids and kind_of(m) == "fact"
        and int(m.get("is_current", 1)) == 1 and m.get("links")
    ]
    new_metas.sort(key=lambda m: m.get("saved_at", ""), reverse=True)

    changed = 0
    for new_meta in new_metas[:EVOLVE_NEW_CAP]:
        for lid in list(filter(None, new_meta.get("links", "").split(",")))[:EVOLVE_LINK_CAP]:
            neighbor = by_id.get(lid)
            if neighbor is None or int(neighbor.get("is_current", 1)) == 0:
                continue
            try:
                with dspy.context(lm=_lm):
                    out = await _evolver.acall(
                        new_memory=new_meta["memory_text"],
                        existing_memory=neighbor["memory_text"],
                        existing_context=neighbor.get("context", ""),
                    )
                context = out.updated_context.strip()
            except Exception:
                continue
            if context and context != neighbor.get("context", ""):
                await set_context(lid, context)
                changed += 1
    return changed


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