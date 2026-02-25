import dspy
from pydantic import BaseModel
from datetime import datetime
from memory.embedding_generation import generate_embeddings
from memory.memory_store import (
    EmbeddedMemory,
    RetrievedMemory,
    delete_records,
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
    - UPDATE: update an existing memory with richer information.
    - DELETE: remove memory items from the database that aren't required anymore due to new information
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
        print("adding memory:", memory_text)
        print("categories:", categories)

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
        """Replace an existing memory (identified by its list index) with richer text."""
        print("updating memory:", update_memory_text)
        print("categories:", categories)

        point_id = get_point_id(memory_id)
        await delete_records([point_id])

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
                )
            ]
        )
        return f"Memory updated: {update_memory_text}"

    async def delete_memory(memory_id: int) -> str:
        """Delete an existing memory by its list index."""
        print("deleting memory:", memory_id)

        point_id = get_point_id(memory_id)
        await delete_records([point_id])
        return f"Memory deleted: {memory_id}"

    async def noop() -> str:
        """No operation needed — nothing to add, update, or delete."""
        print("no operation needed")
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
        tools=[add_new_memory, update_existing_memory, delete_memory, noop],
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