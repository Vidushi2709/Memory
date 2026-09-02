import asyncio
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
    mark_reconciled,
    set_context,
    set_core_memory,
)
from memory.llm import get_lm, was_truncated

DEDUP_SIMILARITY = 0.9     # cosine similarity above which memories count as duplicates
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set:
    return set(_NUM_RE.findall(text))
REFLECTION_THRESHOLD = 40  # summed importance of fresh facts that triggers reflection
EVOLVE_NEW_CAP = 3         # newest session memories considered for evolution
EVOLVE_LINK_CAP = 2        # linked neighbors re-examined per new memory


async def _call(predictor, **kwargs):
    """Run a consolidation predictor and refuse a truncated response.

    Every output here is short by design (a merged sentence, a pair list, an
    80-word profile), so a response that fills the token budget is runaway
    repetition — parsing it would store garbage like a half-sentence merged
    memory. Raise instead; every call site already treats a raise as "skip
    this stage". Fresh LM per call so the truncation check reads the right
    response under concurrency."""
    lm = get_lm()
    with dspy.context(lm=lm):
        out = await predictor.acall(**kwargs)
    if was_truncated(lm):
        raise RuntimeError("consolidation response truncated (runaway repetition)")
    return out


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
    You are given recent memories about a user and the insights previously
    drawn about them. Output the COMPLETE updated set of 2-3 higher-level
    insights (patterns, goals, traits): keep previous insights that still
    hold, revise ones the recent memories change, drop ones they contradict.
    Only state insights strongly supported by the memories — never guess or
    invent. Return an empty list if nothing meaningful emerges.
    """

    recent_memories: list[str] = dspy.InputField()
    previous_insights: list[str] = dspy.InputField()
    insights: list[str] = dspy.OutputField(desc="The full current set; replaces previous_insights.")


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

    A completed occurrence NEVER supersedes another completed occurrence.
    Attending a concert, seeing a doctor, taking a trip, finishing a project:
    each one is its own event, and events accumulate rather than replace each
    other. Two concerts on different dates are two concerts — link them, never
    supersede. Superseding is only for a mutually exclusive STATE, where the
    new fact makes the old one false: where the user lives, who they work for,
    what they own, anything that can hold only one value at a time.

    The test to apply to every candidate pair: could both still be true of the
    user's life? If yes, it is a link, not a supersede.

    Only include pairs you are confident about. Output empty strings if none.
    """

    new_memories: list[str] = dspy.InputField()
    existing_memories: list[str] = dspy.InputField()
    supersede_pairs: str = dspy.OutputField(desc='Comma-separated "Nx:Ey" pairs, e.g. "N0:E2,N1:E5". Empty if none.')
    link_pairs: str = dspy.OutputField(desc='Comma-separated "Nx:Ey" pairs. Empty if none.')


class CoreMemorySignature(dspy.Signature):
    """
    Write a short always-visible profile of the user: name, location, work,
    and their most important preferences. Plain sentences, under 80 words.
    Use ONLY information present in facts — NEVER invent, guess, or embellish
    details that were not stated. The facts are the user's CURRENT state:
    anything not in them is not in the profile.

    Other people's names are not the user's attributes. If a fact mentions
    "Dr. Thompson" or "my friend Zubin", those names belong to them, not to the
    user — never turn one into the user's own name, job, or trait. State the
    user's name only if a fact says explicitly that it is theirs.
    """

    facts: list[str] = dspy.InputField(desc="The user's current facts, most important first.")
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


def _natural_key(sid: str) -> list:
    """Sort "lme-0-11" after "lme-0-9" — plain string order puts 9 last."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", sid)]


RECONCILE_EXISTING_CAP = 40  # existing memories one reconcile call can weigh


def _most_relevant(new, existing, cap):
    """The existing memories closest to this session's new ones.

    Selecting the most RECENT ones instead meant that on a long history the
    reconciler never saw the old fact a new one contradicts — so on a haystack
    nothing beyond the last 40 memories could ever be superseded.
    """
    if len(existing) <= cap:
        return existing
    unit = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    new_vecs = unit(np.asarray([e for _, _, e in new], dtype=np.float32))
    old_vecs = unit(np.asarray([e for _, _, e in existing], dtype=np.float32))
    closeness = (old_vecs @ new_vecs.T).max(axis=1)
    keep = sorted(np.argsort(closeness)[::-1][:cap])
    return [existing[i] for i in keep]


async def reconcile_session(user_id: int, session_ids) -> str:
    """
    One large-context call reconciles the given sessions' new memories against
    the existing store: supersedes changed facts, links related ones.
    """
    sids = _as_set(session_ids)
    recs = await fetch_user_records_raw(user_id)
    new, existing = [], []
    for id_, meta, emb in zip(recs["ids"], recs["metadatas"], recs["embeddings"]):
        if meta.get("type") == "core" or int(meta.get("is_current", 1)) == 0:
            continue
        if kind_of(meta) != "fact":
            continue
        if meta.get("session_id") in sids:
            new.append((id_, meta, emb))
        else:
            existing.append((id_, meta, emb))
    if not new or not existing:
        return "nothing to reconcile"

    existing = _most_relevant(new, existing, RECONCILE_EXISTING_CAP)

    fmt = lambda tag, i, meta: f"{tag}{i}: {meta['memory_text']} (date: {meta.get('date', '')[:10]})"
    # build the prompt outside the try: only the call itself may fail silently,
    # or a coding error here disables reconciliation with no sign it happened
    new_texts = [fmt("N", i, rec[1]) for i, rec in enumerate(new)]
    existing_texts = [fmt("E", i, rec[1]) for i, rec in enumerate(existing)]
    try:
        out = await _call(_reconciler,
                          new_memories=new_texts, existing_memories=existing_texts)
    except Exception:
        return "reconcile call failed"

    superseded, refused = 0, 0
    for ni, ei in _parse_pairs(out.supersede_pairs, len(new), len(existing)):
        ex_meta, new_meta = existing[ei][1], new[ni][1]
        # An older statement must never supersede a newer one. Sessions do not
        # always arrive in chronological order — a user can mention a past fact
        # today — and ingestion order alone would mark the current fact stale.
        stale_wins = ex_meta.get("date", "") > new_meta.get("date", "")
        # A completed event cannot become outdated — only mutable state can.
        # When a later session re-tells the same event (same day, both
        # happened), superseding hides a real event behind an [OLD] tag; link
        # them and let dedup merge. Different-day events (moved cities) still
        # supersede normally.
        same_event = (
            ex_meta.get("status", "happened") == "happened"
            and new_meta.get("status", "happened") == "happened"
            and ex_meta.get("date", "")[:10] == new_meta.get("date", "")[:10]
        )
        if stale_wins or same_event:
            await add_links(new[ni][0], [existing[ei][0]])
            refused += 1
            continue
        await mark_memory_old(existing[ei][0])
        await add_links(new[ni][0], [existing[ei][0]])  # keep the history reachable
        superseded += 1
    linked = 0
    for ni, ei in _parse_pairs(out.link_pairs, len(new), len(existing)):
        await add_links(new[ni][0], [existing[ei][0]])
        linked += 1
    summary = f"{superseded} superseded, {linked + refused} linked"
    return summary + (f" ({refused} supersede refused: stale or a re-told event)"
                      if refused else "")


CORE_FACT_CAP = 30  # most important facts the profile is rebuilt from


async def refresh_core_memory(user_id: int):
    """Rebuild the always-in-prompt profile from the user's most important
    current facts — the profile describes the whole person, not one session.

    Rebuilt from scratch, not edited: feeding the old profile back in meant a
    superseded fact ("lives in Delhi") survived every rewrite, because nothing
    told the model which sentence had become false."""
    recs = await fetch_user_records_raw(user_id, include_embeddings=False)
    current = [
        meta for meta in recs["metadatas"]
        if kind_of(meta) == "fact" and meta.get("type") != "core"
        and int(meta.get("is_current", 1)) == 1
    ]
    current.sort(key=lambda m: (int(m.get("importance", 5)), m.get("saved_at", "")), reverse=True)
    facts = [m["memory_text"] for m in current[:CORE_FACT_CAP]]
    if not facts:
        return
    current_core = await get_core_memory(user_id)
    try:
        out = await _call(_core_updater, facts=facts)
        new_core = out.updated_core.strip()
    except Exception:
        return
    if new_core and new_core != current_core:
        embedding = (await generate_embeddings([new_core]))[0]
        await set_core_memory(user_id, new_core, embedding)


async def sleep_pass(user_id: int, session_ids, refresh_core: bool = True) -> str:
    """
    Sleep-time consolidation (Letta pattern): the online path only appends;
    this pass holds all write authority over reconciliation and derived memory.

    Accepts one session id or a batch of them — a batch lets the pass run
    periodically over long histories without leaving sessions unreconciled.

    refresh_core=False skips the profile rewrite. On long ingests it is the
    most expensive stage and the intermediate profiles are thrown away anyway,
    so callers doing many passes should refresh only on the last one.
    """
    sids = _as_set(session_ids)
    latest = max(sids, key=_natural_key)

    # reconcile first: it supersedes and links, which the later stages read
    parts = [await reconcile_session(user_id, sids)]

    # dedup->evolve must stay ordered (evolve reads links and skips merged-away
    # memories); reflection only appends insights, so it can run alongside
    async def _dedup_then_evolve():
        merged = await dedup_memories(user_id, latest)
        evolved = await evolve_memories(user_id, sids)
        return merged, evolved

    (merged, evolved), insights = await asyncio.gather(
        _dedup_then_evolve(),
        maybe_reflect(user_id, latest),
    )
    parts.append(merged)
    if evolved:
        parts.append(f"{evolved} context(s) evolved")
    if insights:
        parts.append(f"{len(insights)} insight(s)")

    if refresh_core:
        await refresh_core_memory(user_id)

    # stamp the sessions this pass covered, so a pass that never ran (process
    # killed mid-session) is visible to unreconciled_sessions and repaired later
    recs = await fetch_user_records_raw(user_id, include_embeddings=False)
    await mark_reconciled([
        id_ for id_, meta in zip(recs["ids"], recs["metadatas"])
        if meta.get("session_id") in sids and kind_of(meta) == "fact"
    ])
    return ", ".join(p for p in parts if p)


async def unreconciled_sessions(user_id: int) -> list[str]:
    """Session ids with current facts no sleep pass has reconciled. Callers run
    `sleep_pass(user_id, unreconciled_sessions(user_id))` at startup."""
    recs = await fetch_user_records_raw(user_id, include_embeddings=False)
    pending = {
        meta["session_id"] for meta in recs["metadatas"]
        if meta.get("session_id") and kind_of(meta) == "fact"
        and int(meta.get("is_current", 1)) == 1 and not meta.get("reconciled")
    }
    return sorted(pending, key=_natural_key)


def _clusters(items) -> list[list[int]]:
    """Union-find over pairs above DEDUP_SIMILARITY; returns groups of 2+ indices.

    One matrix multiply instead of a Python loop over every pair — at a few
    hundred memories that loop cost seconds on every single sleep pass.
    """
    n = len(items)
    if n < 2:
        return []
    embs = np.asarray([it[3] for it in items], dtype=np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    similar = np.triu(embs @ embs.T >= DEDUP_SIMILARITY, k=1)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(similar)):
        parent[find(int(i))] = find(int(j))

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


async def dedup_memories(user_id: int, session_id: str = "") -> str:
    recs = await fetch_user_records_raw(user_id)

    by_group: dict = {}
    for id_, meta, doc, emb in zip(
        recs["ids"], recs["metadatas"], recs["documents"], recs["embeddings"]
    ):
        if meta.get("type") == "core" or int(meta.get("is_current", 1)) == 0:
            continue
        # derived memories never merge with the raw facts they came from, and an
        # intention never merges with an event — "planned" and "happened" are
        # different facts however similar they read
        by_group.setdefault((kind_of(meta), meta.get("status", "happened")), []).append(
            (id_, meta, doc, emb)
        )

    merged_count = 0
    resolved_count = 0
    for (kind, status), items in by_group.items():
        for group in _clusters(items):
            members = [items[i] for i in group]

            # Members that disagree on a number are not duplicates: either a
            # progression ("completed 4 projects" then "5") or distinct events
            # ("ran 5 km on the 3rd", "ran 8 km on the 10th"). Merging bakes the
            # conflict into one memory; hiding the older one erased real events.
            # Keep all current and linked — retrieval shows both with dates, and
            # the reconcile call (which sees dates) is the one allowed to supersede.
            nums = [_numbers(m[2]) for m in members]
            if any(n != nums[0] for n in nums):
                members.sort(key=lambda m: m[1].get("saved_at", ""))
                await add_links(members[-1][0], [m[0] for m in members[:-1]])
                resolved_count += 1
                continue

            try:
                out = await _call(_merger, memories=[m[2] for m in members])
                text = out.merged.strip()
            except Exception:
                continue
            if not text:
                continue

            categories = sorted({c.strip() for m in members for c in m[1]["categories"].split(",")})
            keywords = sorted({k.strip() for m in members
                               for k in m[1].get("keywords", "").split(",") if k.strip()})
            context = next((m[1].get("context", "") for m in members if m[1].get("context")), "")
            embedding = (await generate_embeddings([text]))[0]
            new_ids = await add_memory([
                EmbeddedMemory(
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
                    # without these the merge quietly discarded the search aids and
                    # relabelled a planned/considered memory as something that happened
                    keywords=keywords,
                    context=context,
                    status=status,
                )
            ])
            # inherit the group's links too, or the graph loses those edges entirely
            inherited = {l for m in members
                         for l in filter(None, m[1].get("links", "").split(","))}
            inherited -= {m[0] for m in members}
            if inherited:
                await add_links(new_ids[0], sorted(inherited))
            for m in members:
                await mark_memory_old(m[0])
            merged_count += 1

    parts = []
    if merged_count:
        parts.append(f"merged {merged_count} duplicate group(s)")
    if resolved_count:
        parts.append(f"linked {resolved_count} numeric-variant group(s)")
    return ", ".join(parts) if parts else "no duplicates found"


async def evolve_memories(user_id: int, session_ids) -> int:
    """
    A-Mem memory evolution: new memories may change how their linked neighbors
    should be read — rewrite those neighbors' context lines.
    """
    sids = _as_set(session_ids)
    recs = await fetch_user_records_raw(user_id, include_embeddings=False)
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
                out = await _call(_evolver,
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
    recs = await fetch_user_records_raw(user_id, include_embeddings=False)

    last_reflection = ""
    facts, previous = [], []
    for id_, meta in zip(recs["ids"], recs["metadatas"]):
        if meta.get("type") == "core":
            continue
        kind = kind_of(meta)
        if kind == "insight":
            last_reflection = max(last_reflection, meta.get("saved_at", ""))
            if int(meta.get("is_current", 1)) == 1:
                previous.append((id_, meta["memory_text"]))
        elif kind == "fact" and int(meta.get("is_current", 1)) == 1:
            facts.append(meta)

    fresh = [m for m in facts if m.get("saved_at", "") > last_reflection]
    if sum(int(m.get("importance", 5)) for m in fresh) < REFLECTION_THRESHOLD:
        return []

    fresh.sort(key=lambda m: m.get("saved_at", ""))
    try:
        out = await _call(_reflector,
                          recent_memories=[m["memory_text"] for m in fresh[-30:]],
                          previous_insights=[t for _, t in previous])
        insights = [s.strip() for s in out.insights if s.strip()][:3]
    except Exception:
        return []

    # the new set replaces the old: insights are re-derived, not accumulated,
    # so a generalisation the facts no longer support does not live forever
    for text in insights:
        embedding = (await generate_embeddings([text]))[0]
        await add_memory([
            EmbeddedMemory(
                user_id=user_id,
                memory_text=text,
                categories=["insight"],
                embedding=embedding,
                date=datetime.now().isoformat(),
                # default importance: a derived generalisation must not outrank
                # the specific facts it was derived from
                session_id=session_id,
                kind="insight",
            )
        ])
    for pid, _ in previous:
        await mark_memory_old(pid)
    return insights