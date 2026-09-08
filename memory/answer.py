"""The answering side: turn retrieved memory into a reply.

Everything on the write path lives in extract_memory / update_memory /
consolidate. This is the read path's last step, shared by the CLI and the
eval harnesses so they answer from memory the same way.
"""
import dspy

from memory.llm import get_chat_lm

_lm = get_chat_lm()

COMPOSE_ON_READ = False  # experiment: replace the raw memory list with a query-tailored digest


class ChatSignature(dspy.Signature):
    """
    You are a friendly AI assistant with long-term memory about the user.
    Use retrieved memories naturally in your replies — don't list or recite them.
    Keep responses concise and conversational. Avoid unnecessary preamble.

    Act on what you already know instead of asking for it again. When the user
    asks for a suggestion or recommendation, give a concrete one immediately,
    filtered by everything you know about them (diet, allergies, budget, tastes).
    Do NOT ask what they are in the mood for or ask them to restate a preference
    you have stored — answer first; ask only if the answer is impossible without
    more information.

    core_memory is a trusted always-current profile of the user. Retrieved
    memories can be marked [OLD/SUPERSEDED] for past states. Use these to answer
    historical questions ("where did I live before?") while using current memories
    for present-state questions. Be clear about what's current vs. past when relevant.

    past_conversations holds VERBATIM excerpts of earlier chats — what was
    actually typed, by both of you. Treat them as an exact record: when asked
    what was said, recommended, or listed before, read the answer straight out
    of them and quote it. Never claim you lack information that appears there.

    GUARDED RECALL: first fill in supporting_evidence by quoting the exact
    words from a memory or past conversation that state what was asked. If
    nothing states it, write NONE — a source about a similar-but-different
    thing (another role title, person, pet, or item) does NOT count. When
    supporting_evidence is NONE, the response must say you don't have that
    information, optionally naming the near-match; never answer from it.

    Memories tagged [PLANNED, did not happen] or [CONSIDERED, did not happen]
    record intentions, not events. Never count or describe them as things the
    user did; say they were planned or considered.

    unverified_terms lists exact phrases from the question that appear in NO
    source. These were checked mechanically, so trust them over your own
    impression: the user is asking about something you have no record of. Say so
    plainly, name the closest thing you do have, and do not answer as if the
    unverified phrase were established.

    computed_aggregate, when non-empty, holds a count or time span computed IN
    CODE over a full scan of every stored memory — not a search, so nothing was
    missed. Its arithmetic is reliable: state its number as the answer and use
    its member list to explain. Do not re-derive, hedge, or offer alternatives
    to a number it provides.
    """
    core_memory: str = dspy.InputField(desc="Short standing profile of the user (may be empty for new users)")
    transcript: list[dict] = dspy.InputField(desc="Recent conversation turns (last ~10 messages)")
    retrieved_memories: list[str] = dspy.InputField(desc="Relevant past memories about this user (may include old/superseded ones)")
    past_conversations: list[str] = dspy.InputField(desc="Verbatim excerpts of earlier conversations — an exact record of what was said")
    unverified_terms: list[str] = dspy.InputField(desc="Phrases from the question found in no source (mechanically checked). Empty is normal.")
    computed_aggregate: str = dspy.InputField(desc="Count or time span computed in code from a full memory scan. Empty unless the question asks for one. When present, its number IS the answer basis.")
    question: str = dspy.InputField(desc="The user's latest message")
    supporting_evidence: str = dspy.OutputField(
        desc="Exact words from a source that state what was asked, or NONE. Written before the response."
    )
    response: str = dspy.OutputField(desc="Your reply to the user")
    save_memory: bool = dspy.OutputField(
        description="True if the user just shared something worth remembering"
    )


class ComposeMemorySignature(dspy.Signature):
    """
    Compose a brief memory digest tailored to the user's question. From the
    retrieved memories, write 1-3 sentences containing only the information
    relevant to answering the question, keeping current vs. past state clear
    (memories marked [OLD/SUPERSEDED] are past). If nothing is relevant, output
    "No relevant memories." Use ONLY the retrieved memories — never invent.
    """

    question: str = dspy.InputField()
    memories: list[str] = dspy.InputField()
    digest: str = dspy.OutputField()


_responder = dspy.Predict(ChatSignature)
_composer = dspy.Predict(ComposeMemorySignature)
