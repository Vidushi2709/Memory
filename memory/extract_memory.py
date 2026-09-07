from typing import List
import dspy
from pydantic import BaseModel, field_validator
import json
import asyncio
import logging
from datetime import datetime
from memory.llm import (
    MEMORY_MAX_TOKENS,
    MEMORY_MAX_TOKENS_RETRY,
    get_lm,
    was_truncated,
)

log = logging.getLogger(__name__)

STATUSES = ("happened", "planned", "considered", "ongoing")


class Memory(BaseModel):
    information: str
    predicted_category: List[str]
    importance: int = 5  # 1 = mundane, 10 = life-changing
    date: str = ""  # ISO date the fact became true
    keywords: List[str] = []  # salient tokens for keyword search
    context: str = ""  # one-sentence LLM interpretation of what this fact means
    status: str = "happened"  # happened | planned | considered | ongoing
    about_user: bool = True  # False = general knowledge, not a fact about the user

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

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        v = str(v).strip().lower()
        return v if v in STATUSES else "happened"

    @field_validator("about_user", mode="before")
    @classmethod
    def _coerce_about_user(cls, v):
        # default True: a parse hiccup must never silently drop a real fact
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() != "false"


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

    Each memory's information must keep every specific detail that was stated —
    names, employers, places, numbers, brands. Never generalise them away
    ("pilot with Indigo", not "pilot").

    Never turn an intention into a completed fact. Set status to:
      happened   — the user says it took place ("I had an appointment with Dr. Lee")
      planned    — scheduled or committed but not yet done ("I'm seeing Dr. Lee on the 4th")
      considered — only being weighed ("I'm thinking of seeing Dr. Lee")
      ongoing    — a continuing state or routine ("I see Dr. Lee every month")
    Say it in the information text too — write "User is considering X", not "User X".
    The date is when the event happens, which for planned events is in the future.

    For each memory also produce: keywords — the 2-5 salient tokens someone would
    search by (names, places, specific things); and context — at most 10 words on
    what this fact reveals, grounded only in what was said. Keep only these two
    fields terse; they are search aids, not prose.

    Set about_user to False ONLY for impersonal content: general knowledge,
    history, trivia, religious or legal rulings, and advice, recommendations,
    or explanations the assistant produced. Facts about the user's friends,
    family, colleagues, and pets ARE facts about the user's life — "My friend
    Zubin is a pilot with Indigo" is a memory (about_user True). Advice stays
    False even when it concerns the user's own plans: for "I'm hiking Shiretoko
    — what gear?", the user's hike is a memory; the assistant's gear list is
    not. When unsure whether something is the user's fact or the assistant's
    content, ask who stated it: the user's statements about their life are kept.

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
        desc="A list of new memory entries. Each has 'information', 'predicted_category', 'importance', 'date', 'keywords', 'context', 'status' (happened/planned/considered/ongoing), and 'about_user' (False for general knowledge or assistant-produced content)."
    )


memory_extractor = dspy.Predict(MemoryExtractor)


class MemoryGapCheck(dspy.Signature):
    """A first extraction pass over this transcript already produced the facts
    listed in already_extracted. Re-read the transcript and output ONLY facts
    about the user's life that are MISSING from that list.

    A fact is missing only if no entry in already_extracted covers the same
    information — in any wording, at any level of detail. Never restate,
    rephrase, correct, or refine an existing entry. If the list already covers
    everything, set nothing_missed to True and output no memories.

    First passes most often drop: events mentioned in passing while discussing
    something else ("when I was at the clinic for my cough last month...");
    individual members of a list where only the list was captured; dates and
    quantities attached to an event; and additional instances of a kind already
    extracted once (a second doctor's visit, a third concert — each distinct
    instance is its own memory).

    Each missing fact follows the same rules as the first pass. One atomic unit
    of information, keeping every stated specific — names, employers, places,
    numbers, brands. Importance 1 (mundane) to 10 (life-changing). Resolve
    relative dates against current_date and set date to the ISO date the fact
    became true (future for planned events). Set status to: happened (took
    place), planned (committed but not done), considered (only being weighed),
    or ongoing (a continuing state or routine) — and say it in the information
    text ("User is considering X"). keywords: the 2-5 salient search tokens;
    context: at most 10 words on what the fact reveals.

    Set about_user to False ONLY for impersonal content: general knowledge,
    trivia, rulings, and advice, recommendations, or explanations the assistant
    produced. Facts about the user's friends, family, colleagues, and pets ARE
    facts about the user's life. Advice stays False even when it concerns the
    user's own plans.
    """

    transcript: str = dspy.InputField(desc="The transcript of the conversation.")
    current_date: str = dspy.InputField(desc="Today's ISO date, for resolving relative dates.")
    existing_categories: List[str] = dspy.InputField(
        desc="Memory categories already stored for this user."
    )
    already_extracted: List[str] = dspy.InputField(
        desc="The fact texts the first extraction pass produced for this transcript."
    )
    nothing_missed: bool = dspy.OutputField(
        desc="True if already_extracted covers every fact about the user in the transcript."
    )
    missed_memories: List[Memory] = dspy.OutputField(
        desc="Facts present in the transcript but absent from already_extracted. Same fields as a first-pass memory."
    )


gap_checker = dspy.Predict(MemoryGapCheck)


async def extract_memory(messages, categories=None, current_date=None):
    if categories is None:
        categories = []

    transcript = json.dumps(messages)
    date = current_date or datetime.now().date().isoformat()

    # A cut-off response loses every fact after the cut, so escalate the budget
    # rather than store a partial session as though it were the whole one.
    for max_tokens in (MEMORY_MAX_TOKENS, MEMORY_MAX_TOKENS_RETRY):
        lm = get_lm(max_tokens=max_tokens)
        with dspy.context(lm=lm):
            out = await memory_extractor.acall(
                transcript=transcript,
                current_date=date,
                existing_categories=categories,
            )
        if not was_truncated(lm):
            return out
        log.warning("extraction hit the %d-token cap; retrying with a larger budget",
                    max_tokens)

    log.error("extraction still truncated at %d tokens — facts may be missing "
              "from this session", MEMORY_MAX_TOKENS_RETRY)
    return out


async def extract_missed(messages, already_extracted, categories=None, current_date=None):
    """Second 'what did I miss?' pass: same transcript, conditioned on the first
    pass's output. Extraction is nondeterministic — the same session yields
    different fact subsets run to run — so a pass that sees what was already
    caught recovers facts the first pass dropped."""
    transcript = json.dumps(messages)
    date = current_date or datetime.now().date().isoformat()

    for max_tokens in (MEMORY_MAX_TOKENS, MEMORY_MAX_TOKENS_RETRY):
        lm = get_lm(max_tokens=max_tokens)
        with dspy.context(lm=lm):
            out = await gap_checker.acall(
                transcript=transcript,
                current_date=date,
                existing_categories=categories or [],
                already_extracted=already_extracted,
            )
        if not was_truncated(lm):
            return out
        log.warning("gap check hit the %d-token cap; retrying with a larger budget",
                    max_tokens)

    log.error("gap check still truncated at %d tokens — treating as no gaps found",
              MEMORY_MAX_TOKENS_RETRY)
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

class WorkLogExtractor(dspy.Signature):
    """Extract only DURABLE facts about the user from a working session — a
    transcript of them building something with an AI coding assistant.

    Keep what will still be true and worth knowing weeks from now:
      - standing preferences and working style ("always branch before starting
        work", "PRs need a description, type of change and testing section")
      - rules and corrections the user gave the assistant, especially ones
        phrased as "always", "never", "from now on", or given as a rebuke
      - what they are building, and the decisions and constraints behind it:
        an architecture settled on, a library chosen or rejected and why
      - tools, accounts, models, hardware and keys they do or do not have
      - their role, expertise, and the projects they own

    Discard everything transient, which is most of a working session. A single
    task instruction ("run the tests", "create a branch", "fix this error"),
    the state of one debugging session, what a command printed, a number that
    was true only that afternoon, and anything already superseded by the end of
    the same conversation are NOT memories. When a session is pure task
    execution with nothing durable in it, set no_info to True and return an
    empty list — that is the common and correct outcome, not a failure.

    Write each fact to stand alone weeks later, naming the project it belongs
    to. Not "User wants to add a pipecat path", which means nothing out of
    context, but "User is building an ETL agent in the NovaEval project that
    maps pipecat traces, mirroring an existing livekit path."

    Rate importance 1-10: a standing rule about how they want to work is high,
    a passing detail about one file is low. Resolve relative dates against
    current_date. Set status to happened, planned, considered or ongoing —
    a standing preference is ongoing. Give 2-5 salient keywords and a context
    line of at most 10 words. Set about_user False for general programming
    knowledge and for the assistant's own explanations and suggestions.
    """

    transcript: str = dspy.InputField(desc="The working session transcript.")
    current_date: str = dspy.InputField(desc="Today's ISO date, for resolving relative dates.")
    existing_categories: List[str] = dspy.InputField(
        desc="Memory categories already stored for this user."
    )
    no_info: bool = dspy.OutputField(
        desc="True if the session holds no durable fact about the user."
    )
    new_memories: List[Memory] = dspy.OutputField(
        desc="Durable facts only. Same fields as a first-pass memory."
    )


worklog_extractor = dspy.Predict(WorkLogExtractor)


async def extract_worklog(messages, categories=None, current_date=None):
    """Extraction lens for coding-assistant sessions, where most of the
    transcript is transient task execution rather than anything to remember."""
    transcript = json.dumps(messages)
    date = current_date or datetime.now().date().isoformat()
    for max_tokens in (MEMORY_MAX_TOKENS, MEMORY_MAX_TOKENS_RETRY):
        lm = get_lm(max_tokens=max_tokens)
        with dspy.context(lm=lm):
            out = await worklog_extractor.acall(
                transcript=transcript, current_date=date,
                existing_categories=categories or [],
            )
        if not was_truncated(lm):
            return out
        log.warning("worklog extraction hit the %d-token cap; retrying larger", max_tokens)
    return out
