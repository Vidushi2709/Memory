from datetime import datetime
import dspy
from memory.embedding_generation import generate_embeddings
from memory.extract_memory import extract_memory, Memory
from memory.memory_store import (
    EmbeddedMemory,
    add_memory,
    get_all_categories,
)
from dotenv import load_dotenv

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


async def _store(user_id: int, fact: Memory, date: str, session_id: str = "") -> str:
    embeddings = await generate_embeddings([fact.information])
    ids = await add_memory(
        embedded_memories=[
            EmbeddedMemory(
                id="",
                user_id=user_id,
                memory_text=fact.information,
                categories=fact.predicted_category,
                embedding=embeddings[0],
                date=date,
                importance=fact.importance,
                session_id=session_id,
                keywords=fact.keywords,
                context=fact.context,
            )
        ]
    )
    return ids[0]


async def update_memories(user_id: int, messages: list[dict], session_id: str = "", current_date: str = ""):
    """
    Thin write path: extract facts and store them append-only, one LLM call total.

    All judgment-heavy work — reconciling contradictions, superseding, linking,
    dedup, core-profile rewrites — runs in the sleep-time pass at session end
    (memory/consolidate.py: sleep_pass), where one large-context call sees the
    whole session at once instead of many small calls compounding errors.
    """
    categories = await get_all_categories(user_id=user_id)
    extracted = await extract_memory(messages, categories, current_date or None)

    if extracted.no_info or not extracted.new_memories:
        return "No new facts."

    for fact in extracted.new_memories:
        await _store(user_id, fact, _safe_date(fact.date, current_date), session_id)

    n = len(extracted.new_memories)
    return f"{n} fact(s) noted"


async def test():
    messages = [{"role": "user", "content": "I want to go to Tokyo"}]
    response = await update_memories(user_id=1, messages=messages)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())