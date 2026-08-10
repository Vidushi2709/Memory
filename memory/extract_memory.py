from typing import List
import dspy
from pydantic import BaseModel, field_validator
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()


class Memory(BaseModel):
    information: str
    predicted_category: List[str]
    importance: int = 5  # 1 = mundane, 10 = life-changing
    date: str = ""  # ISO date the fact became true
    keywords: List[str] = []  # salient tokens for keyword search
    context: str = ""  # one-sentence LLM interpretation of what this fact means

    @field_validator("importance", mode="before")
    @classmethod
    def _coerce_importance(cls, v):
        # LLMs sometimes emit "" or junk —> fall back instead of failing the whole batch
        try:
            return min(10, max(1, int(v)))
        except (ValueError, TypeError):
            return 5

    @field_validator("date", mode="before")
    @classmethod
    def _coerce_date(cls, v):
        return v if isinstance(v, str) else ""

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, v):
        return v if isinstance(v, list) else []

    @field_validator("context", mode="before")
    @classmethod
    def _coerce_context(cls, v):
        return v if isinstance(v, str) else ""


class MemoryExtractor(dspy.Signature):
    """
    Extract relevant information from the conversation.
    Create memory entries that you should remember while speaking to the user later.
    Each memory is one atomic unit of information that can be stored and retrieved later.
    Extract EVERY distinct fact the user shares — a single message may yield several
    memories (e.g. name, location, and job are three separate entries).

    Rate each memory's importance from 1 (mundane detail) to 10 (core fact about the
    user's identity or life). Resolve relative dates ("last week", "next month")
    against current_date and set each memory's date to the ISO date the fact became
    true; use current_date if unknown.

    For each memory also produce: keywords — the 2-5 salient tokens someone would
    search by (names, places, specific things); and context — one sentence saying
    what this fact reveals about the user, grounded only in what was said.

    You will be given a list of existing memory categories that have already been stored
    for this user. You can decide whether to create a new category or to pick from an
    existing category. If the information is too personal (e.g. name, age, location),
    you should create a new category for it. If the information is more general
    (e.g. preferences, interests), you can choose to store it under an existing category
    or create a new one if it doesn't fit well with existing categories.
    If the transcript contains information that is not relevant or important to remember,
    you can choose not to create a memory entry for it. In that case, set no_info to True
    and new_memories to an empty list.
    """

    transcript: str = dspy.InputField(desc="The transcript of the conversation so far.")
    current_date: str = dspy.InputField(desc="Today's ISO date, for resolving relative dates.")
    existing_categories: List[str] = dspy.InputField(
        desc="A list of existing memory categories that have already been stored for this user."
    )
    no_info: bool = dspy.OutputField(
        desc="If there is no relevant information to extract, set it True. Otherwise, set it False."
    )
    new_memories: List[Memory] = dspy.OutputField(
        desc="A list of new memory entries to add to the user's memory. Each entry should have an 'information', 'predicted_category', 'importance', and 'date' field."
    )


memory_extractor = dspy.Predict(MemoryExtractor)


async def extract_memory(messages, categories=None):
    if categories is None:
        categories = []

    transcript = json.dumps(messages)

    with dspy.context(
        lm=dspy.LM(
            model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
            api_key=os.getenv("OPEN_ROUTER_KEY"),
        )
    ):
        out = await memory_extractor.acall(
            transcript=transcript,
            current_date=datetime.now().date().isoformat(),
            existing_categories=categories,
        )

    return out


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hi, my name is Vin and I love hiking."},
        {
            "role": "assistant",
            "content": "Nice to meet you, Vin! Hiking is a great way to stay active. Do you have any favorite trails?",
        },
        {"role": "user", "content": "Yes, I really enjoy hiking in the mountains near my hometown."},
    ]
    existing_categories = ["name", "hobbies"]
    result = asyncio.run(extract_memory(messages, existing_categories))
    print("Memories:", result)