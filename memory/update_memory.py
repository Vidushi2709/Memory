import logging
from datetime import datetime
import dspy
import numpy as np
from memory.embedding_generation import generate_embeddings
from memory.extract_memory import extract_memory, extract_missed, Memory
from memory.memory_store import (
    EmbeddedMemory,
    add_memory,
    get_all_categories,
)
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Same threshold the sleep pass uses to merge duplicates (consolidate.py):
# a "missed" fact this similar to a stored one would only be merged back later,
# so don't store it at all.
GAP_DUP_SIMILARITY = 0.9

load_dotenv()

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

def _safe_date(raw: str, fallback: str = "") -> str:
    for candidate in (raw, fallback):
        try:
            return datetime.fromisoformat(candidate).isoformat()
        except (ValueError, TypeError):
            continue
    return datetime.now().isoformat()


async def _store_all(user_id: int, facts: list[Memory], dates: list[str], session_id: str = "") -> list[str]:
    """One embedding call and one write for the whole session's facts — the
    per-fact version paid a model round trip and a DB upsert each time."""
    embeddings = await generate_embeddings([f.information for f in facts])
    return await add_memory(
        embedded_memories=[
            EmbeddedMemory(
                user_id=user_id,
                memory_text=fact.information,
                categories=fact.predicted_category,
                embedding=embedding,
                date=date,
                importance=fact.importance,
                session_id=session_id,
                keywords=fact.keywords,
                context=fact.context,
                status=fact.status,
            )
            for fact, embedding, date in zip(facts, embeddings, dates)
        ]
    )


async def _completeness_pass(
    user_id: int,
    messages: list[dict],
    stored: list[Memory],
    categories: list[str],
    session_id: str,
    current_date: str,
) -> int:
    """Recover facts the first extraction dropped. Best-effort: any failure here
    leaves the first pass's writes intact and returns 0.

    The same session extracts different fact subsets run to run even at
    temperature 0, and with reasoning done in code and retrieval at 9/9,
    P(all needed facts stored) is the accuracy ceiling. A second call that sees
    the first pass's output is differently conditioned, so it surfaces the
    dropped facts instead of re-deriving the same subset.
    """
    try:
        out = None
        for _ in range(2):  # structured-output parsing fails intermittently
            try:
                out = await extract_missed(
                    messages, [f.information for f in stored], categories,
                    current_date or None,
                )
                break
            except Exception:
                continue
        if out is None or out.nothing_missed or not out.missed_memories:
            return 0

        fresh = [f for f in out.missed_memories if f.about_user]
        if not fresh:
            return 0

        # The gap pass re-emits paraphrases of stored facts despite being told
        # not to — drop anything near-duplicate to a first-pass fact or to a
        # gap fact already kept, at the sleep pass's own merge threshold.
        embeddings = await generate_embeddings(
            [f.information for f in fresh] + [f.information for f in stored]
        )
        arr = np.asarray(embeddings, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        new_rows, seen = arr[: len(fresh)], arr[len(fresh):]

        keep: list[tuple[Memory, list[float]]] = []
        for fact, row, raw in zip(fresh, new_rows, embeddings):
            if float((seen @ row).max()) >= GAP_DUP_SIMILARITY:
                continue
            keep.append((fact, raw))
            seen = np.vstack([seen, row[None, :]])
        if not keep:
            return 0

        await add_memory([
            EmbeddedMemory(
                user_id=user_id,
                memory_text=fact.information,
                categories=fact.predicted_category,
                embedding=raw,
                date=_safe_date(fact.date, current_date),
                importance=fact.importance,
                session_id=session_id,
                keywords=fact.keywords,
                context=fact.context,
                status=fact.status,
            )
            for fact, raw in keep
        ])
        return len(keep)
    except Exception as e:
        log.warning("completeness pass failed (first-pass facts unaffected): %s", e)
        return 0


EXTRACT_ATTEMPTS = 3


async def _extract_with_retry(messages, categories, current_date):
    """Structured-output parsing fails intermittently on every model we have
    tried. A dropped extraction silently loses a whole session, so retry."""
    last = None
    for attempt in range(EXTRACT_ATTEMPTS):
        try:
            return await extract_memory(messages, categories, current_date or None)
        except Exception as e:
            last = e
    raise last


async def update_memories(user_id: int, messages: list[dict], session_id: str = "", current_date: str = ""):
    """
    Thin write path: extract facts and store them append-only, one LLM call total.

    All judgment-heavy work — reconciling contradictions, superseding, linking,
    dedup, core-profile rewrites — runs in the sleep-time pass at session end
    (memory/consolidate.py: sleep_pass), where one large-context call sees the
    whole session at once instead of many small calls compounding errors.
    """
    categories = await get_all_categories(user_id=user_id)
    extracted = await _extract_with_retry(messages, categories, current_date)

    if extracted.no_info or not extracted.new_memories:
        return "No new facts."

    # general knowledge and assistant-produced content are not memories about
    # the user — stored, they steal retrieval slots from the facts that are
    facts = [f for f in extracted.new_memories if f.about_user]
    dropped = len(extracted.new_memories) - len(facts)
    if not facts:
        return "No new facts."
    dates = [_safe_date(f.date, current_date) for f in facts]
    await _store_all(user_id, facts, dates, session_id)

    recovered = await _completeness_pass(
        user_id, messages, facts, categories, session_id, current_date
    )

    n = len(facts)
    msg = f"{n} fact(s) noted"
    if dropped:
        msg += f" ({dropped} general-knowledge dropped)"
    if recovered:
        msg += f" (+{recovered} recovered by gap check)"
    return msg


async def test():
    messages = [{"role": "user", "content": "I want to go to Tokyo"}]
    response = await update_memories(user_id=1, messages=messages)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())