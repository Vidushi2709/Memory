"""Single place the LLM is configured — swap models here or via env."""
import os
import dspy
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MEMORY_MODEL", "openrouter/mistralai/mistral-small-3.2-24b-instruct")
# The answer path is one call per question while ingestion is hundreds, so a
# stronger (pricier) model here costs almost nothing extra. Defaults to MODEL.
CHAT_MODEL = os.getenv("MEMORY_CHAT_MODEL", MODEL)
API_KEY_ENV = os.getenv("MEMORY_API_KEY_ENV", "OPEN_ROUTER_KEY")

# memory work is judgment, not creativity — variance was 40% of eval failures
MEMORY_TEMPERATURE = 0.0
# Measured, not guessed: one extracted fact serialises to ~88 tokens, and a
# session in the benchmark runs to 14k characters (28k at the tail). A rich
# session yields 30+ facts, so the old 2048 cap cut the response mid-list and
# every fact after the cut was lost with only a warning.
MEMORY_MAX_TOKENS = 4096
MEMORY_MAX_TOKENS_RETRY = 8192  # escalation when a response still comes back cut
CHAT_TEMPERATURE = 0.0
CHAT_MAX_TOKENS = 1024


def get_lm(temperature: float = MEMORY_TEMPERATURE, max_tokens: int = MEMORY_MAX_TOKENS) -> dspy.LM:
    return dspy.LM(
        model=MODEL,
        api_key=os.getenv(API_KEY_ENV),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_chat_lm() -> dspy.LM:
    """LM for the answer/reasoning path — the chatbot response, not memory work."""
    return dspy.LM(
        model=CHAT_MODEL,
        api_key=os.getenv(API_KEY_ENV),
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_TOKENS,
    )


def was_truncated(lm: dspy.LM) -> bool:
    """True if the model stopped because it hit the token cap rather than
    because it was finished. DSPy only logs a warning for this, which on the
    write path means a partially extracted session is stored as if complete."""
    try:
        response = lm.history[-1].get("response")
    except (AttributeError, IndexError, KeyError):
        return False
    return any(
        getattr(choice, "finish_reason", None) == "length"
        for choice in (getattr(response, "choices", None) or [])
    )
