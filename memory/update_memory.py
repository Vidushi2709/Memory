import dspy
from pydantic import BaseModel
from datetime import datetime
from memory.embedding_generation import generate_embeddings
from memory.memory_store import (
    EmbeddedMemory,
    RetrievedMemory,
    mark_memory_old,
    fetch_all_user_records,
    add_memory,
    search_memories,
)
import os
from dotenv import load_dotenv

load_dotenv()

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

class MemoryWithIds(BaseModel):
    memory_id: int      
    memory_text: str
    memory_categories: list[str]


class UpdateMemorySignature(dspy.Signature):
    """
    You will be given the conversation between user and assistant and some similar memories
    from the database. Your goal is to decide how to combine the new memories into the
    database with the existing memories.

    Actions meaning:
    - ADD: add new memories into the database as a new memory
    - UPDATE: mark an existing memory as old, then add a new richer memory to replace it.
              The old memory is preserved in history (not deleted), so the user can still
              ask questions like "where did I live before?".
    - SUPERSEDE: mark a memory as old/outdated without adding a replacement (e.g. the
                 information is simply no longer relevant).
    - NOOP: No need to take any action

    If no action is required you can finish.

    Think less and do actions.
    """

    messages: list[dict] = dspy.InputField()
    existing_memories: list[MemoryWithIds] = dspy.InputField()
    summary: str = dspy.OutputField(
        description="Summarize what you did. Very short (less than 10 words)"
    )


async def update_memory_agent(
    user_id: int,
    message: list[dict],
    existing_memories: list[RetrievedMemory],   
):
    def get_point_id(memory_id: int) -> str:
        return existing_memories[memory_id].point_id

    async def add_new_memory(memory_text: str, categories: list[str]) -> str:
        """Add a brand-new memory to the database."""
        embeddings = await generate_embeddings([memory_text])
        await add_memory(
            embedded_memories=[
                EmbeddedMemory(
                    id="",
                    user_id=user_id,
                    memory_text=memory_text,
                    categories=categories,
                    embedding=embeddings[0],
                    date=datetime.now().isoformat(),
                )
            ]
        )
        return f"Memory added: {memory_text}"

    async def update_existing_memory(memory_id: int, update_memory_text: str, categories: list[str]) -> str:
        """Replace an existing memory (identified by its list index) with richer text.
        The old memory is marked as superseded (preserved in history) and a new one is added."""
        point_id = get_point_id(memory_id)
        # Mark the old memory as superseded (soft-delete) — NOT hard-deleted
        await mark_memory_old(point_id)

        embeddings = await generate_embeddings([update_memory_text])
        await add_memory(
            embedded_memories=[
                EmbeddedMemory(
                    id="",
                    user_id=user_id,
                    memory_text=update_memory_text,
                    categories=categories,
                    embedding=embeddings[0],
                    date=datetime.now().isoformat(),
                    is_current=1,
                )
            ]
        )
        return f"Memory updated: {update_memory_text}"

    async def supersede_memory(memory_id: int) -> str:
        """Mark an existing memory as old/superseded without adding a replacement.
        The record is preserved in history (not deleted) so historical questions
        like 'where did I live before?' can still be answered."""

        point_id = get_point_id(memory_id)
        await mark_memory_old(point_id)
        return f"Memory superseded (marked old): {memory_id}"

    async def noop() -> str:
        """No operation needed — nothing to add, update, or delete."""
        return "No operation needed"

    # Build the MemoryWithIds list that the LLM will reason about
    existing_memories_with_ids = [
        MemoryWithIds(
            memory_id=i,
            memory_text=mem.memory_text,
            memory_categories=mem.categories,
        )
        for i, mem in enumerate(existing_memories)
    ]

    memory_update = dspy.ReAct(
        signature=UpdateMemorySignature,
        tools=[add_new_memory, update_existing_memory, supersede_memory, noop],
    )
    with dspy.context(
        lm=dspy.LM(
            model="mistral/mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
    ):
        out = await memory_update.acall(
            messages=message,
            existing_memories=existing_memories_with_ids,
        )

    return out.summary


async def update_memories(user_id: int, messages: list[dict]):
    latest_user_message = [x["content"] for x in messages if x["role"] == "user"][-1]
    embedding = (await generate_embeddings([latest_user_message]))[0]

    retrieved_memories = await search_memories(search_vector=embedding, user_id=user_id)

    response = await update_memory_agent(
        user_id=user_id,
        existing_memories=retrieved_memories,   
        message=messages,
    )
    return response


async def test():
    messages = [{"role": "user", "content": "I want to go to Tokyo"}]
    response = await update_memories(user_id=1, messages=messages)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())