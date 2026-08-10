from collections import Counter
from typing import Literal
import dspy
from pydantic import BaseModel
from datetime import datetime
from memory.embedding_generation import generate_embeddings
from memory.extract_memory import extract_memory, Memory
from memory.memory_store import (
    EmbeddedMemory,
    get_all_categories,
    get_core_memory,
    mark_memory_old,
    add_memory,
    search_memories,
    set_core_memory,
)
import os
from dotenv import load_dotenv

load_dotenv()

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

_lm = dspy.LM(
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
    api_key=os.getenv("OPEN_ROUTER_KEY"),
)


class MemoryWithIds(BaseModel):
    memory_id: int
    memory_text: str
    memory_categories: list[str]


class MemoryActionSignature(dspy.Signature):
    """
    Decide how a single new fact should change the memory store, given its most
    similar existing memories.

    Actions meaning:
    - ADD: no equivalent memory exists — store the fact as a new memory
    - UPDATE: an existing memory covers this topic — mark it old and store richer
              combined text. The old memory is preserved in history (not deleted),
              so the user can still ask questions like "where did I live before?".
    - SUPERSEDE: the fact makes an existing memory outdated with no replacement
                 (e.g. the information is simply no longer relevant)
    - NOOP: the fact adds nothing beyond what is already stored
    """

    fact: str = dspy.InputField()
    similar_memories: list[MemoryWithIds] = dspy.InputField()
    action: Literal["ADD", "UPDATE", "SUPERSEDE", "NOOP"] = dspy.OutputField()
    target_memory_id: str = dspy.OutputField(
        desc="Index of the memory to update/supersede. -1 or empty for ADD/NOOP."
    )
    memory_text: str = dspy.OutputField(
        desc="Final text to store for ADD/UPDATE. Empty string otherwise."
    )


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


_decide_action = dspy.Predict(MemoryActionSignature)
_core_updater = dspy.Predict(CoreMemorySignature)


def _safe_date(raw: str) -> str:
    try:
        return datetime.fromisoformat(raw).isoformat()
    except (ValueError, TypeError):
        return datetime.now().isoformat()


async def _store(user_id: int, text: str, categories: list[str], importance: int, date: str):
    embeddings = await generate_embeddings([text])
    await add_memory(
        embedded_memories=[
            EmbeddedMemory(
                id="",
                user_id=user_id,
                memory_text=text,
                categories=categories,
                embedding=embeddings[0],
                date=date,
                importance=importance,
            )
        ]
    )


async def _apply_fact(user_id: int, fact: Memory) -> str:
    embedding = (await generate_embeddings([fact.information]))[0]
    neighbors = await search_memories(search_vector=embedding, user_id=user_id, top_k=10)

    similar = [
        MemoryWithIds(
            memory_id=i,
            memory_text=m.memory_text,
            memory_categories=m.categories,
        )
        for i, m in enumerate(neighbors)
    ]

    try:
        with dspy.context(lm=_lm):
            out = await _decide_action.acall(fact=fact.information, similar_memories=similar)
    except Exception:
        # LLM/parse failure — storing the fact as-is beats losing it
        await _store(user_id, fact.information, fact.predicted_category, fact.importance, _safe_date(fact.date))
        return "added"

    if out.action == "NOOP":
        return "noop"

    text = (out.memory_text or "").strip() or fact.information
    date = _safe_date(fact.date)

    if out.action == "ADD":
        await _store(user_id, text, fact.predicted_category, fact.importance, date)
        return "added"

    try:
        target = int(out.target_memory_id)
    except (ValueError, TypeError):
        target = -1

    if not (0 <= target < len(neighbors)):
        # LLM pointed at a nonexistent memory — fall back to a plain add
        await _store(user_id, text, fact.predicted_category, fact.importance, date)
        return "added"

    await mark_memory_old(neighbors[target].point_id)

    if out.action == "UPDATE":
        await _store(user_id, text, fact.predicted_category, fact.importance, date)
        return "updated"

    return "superseded"


async def _refresh_core_memory(user_id: int, facts: list[str]):
    current_core = await get_core_memory(user_id)
    with dspy.context(lm=_lm):
        out = await _core_updater.acall(current_core=current_core, new_facts=facts)

    new_core = out.updated_core.strip()
    if new_core and new_core != current_core:
        embedding = (await generate_embeddings([new_core]))[0]
        await set_core_memory(user_id, new_core, embedding)


async def update_memories(user_id: int, messages: list[dict]):
    categories = await get_all_categories(user_id=user_id)
    extracted = await extract_memory(messages, categories)

    if extracted.no_info or not extracted.new_memories:
        return "No new facts."

    results = [await _apply_fact(user_id, fact) for fact in extracted.new_memories]
    await _refresh_core_memory(user_id, [f.information for f in extracted.new_memories])

    counts = Counter(results)
    return ", ".join(f"{n} {action}" for action, n in counts.items())


async def test():
    messages = [{"role": "user", "content": "I want to go to Tokyo"}]
    response = await update_memories(user_id=1, messages=messages)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())