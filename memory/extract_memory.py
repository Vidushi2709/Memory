from typing import Dict, List, Literal
import dspy
from pydantic import BaseModel 
import json                    
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()


class Memory(BaseModel):
    information: str
    predicted_category: List[str]
    sentiment: Literal["happy", "sad", "neutral"]


class MemoryExtractor(dspy.Signature):
    """
    Extract relevant information from the conversation.
    Create memory entries that you should remember while speaking to the user later.
    Each memory is one atomic unit of information that can be stored and retrieved later.

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
    exisiting_categories: List[str] = dspy.InputField(
        desc="A list of existing memory categories that have already been stored for this user."
    )
    no_info: bool = dspy.OutputField(
        desc="If there is no relevant information to extract, set it True. Otherwise, set it False."
    )
    new_memories: List[Memory] = dspy.OutputField(
        desc="A list of new memory entries to add to the user's memory. Each entry should have an 'information', 'predicted_category', and 'sentiment' field."
    )


memory_extractor = dspy.Predict(MemoryExtractor)


async def extract_memory(messages, categories=None):
    if categories is None:
        categories = []

    transcript = json.dumps(messages)

    with dspy.context(
        lm=dspy.LM(
            model="mistral/mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
    ):
        out = await memory_extractor.acall(
            transcript=transcript,
            exisiting_categories=categories,
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