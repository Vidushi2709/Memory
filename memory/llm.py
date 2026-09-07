"""Single place the LLM is configured — swap models here or via env."""
import asyncio
import os
import threading
import time
from collections import deque

import dspy
from dotenv import load_dotenv

load_dotenv()

# Gemini 2.5 Flash-Lite: measured fastest of every candidate on both memory
# workloads (3.1 s to extract a 5-turn session, 1.1 s to first token, most
# facts found)
_DIRECT = "gemini/gemini-2.5-flash-lite"
_VIA_OPENROUTER = "openrouter/google/gemini-2.5-flash-lite"
_DEFAULT = _DIRECT if os.getenv("GEMINI_API_KEY") else _VIA_OPENROUTER

MODEL = os.getenv("MEMORY_MODEL", _DEFAULT)
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

RPM = int(os.getenv("MEMORY_RPM", "0"))
NUM_RETRIES = int(os.getenv("MEMORY_NUM_RETRIES", "5"))

_sent = deque()          # monotonic timestamps of calls in the last 60 s
_rate_lock = threading.Lock()


def _throttle():
    """Block until sending one more request keeps us under RPM."""
    if RPM <= 0:
        return
    while True:
        with _rate_lock:
            now = time.monotonic()
            while _sent and now - _sent[0] >= 60.0:
                _sent.popleft()
            if len(_sent) < RPM:
                _sent.append(now)
                return
            wait = 60.0 - (now - _sent[0]) + 0.05
        time.sleep(wait)


class ThrottledLM(dspy.LM):
    """dspy.LM that waits its turn instead of hitting a 429."""

    def forward(self, *args, **kwargs):
        _throttle()
        return super().forward(*args, **kwargs)

    async def aforward(self, *args, **kwargs):
        await asyncio.to_thread(_throttle)   # never block the event loop
        return await super().aforward(*args, **kwargs)


def _api_key(model: str) -> str:
    """Each route reads its own key: Google direct wants GEMINI_API_KEY,
    everything else the OpenRouter key."""
    return os.getenv("GEMINI_API_KEY" if model.startswith("gemini/") else API_KEY_ENV)


def get_lm(temperature: float = MEMORY_TEMPERATURE, max_tokens: int = MEMORY_MAX_TOKENS) -> dspy.LM:
    return ThrottledLM(
        model=MODEL,
        api_key=_api_key(MODEL),
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=NUM_RETRIES,
    )


def get_chat_lm() -> dspy.LM:
    """LM for the answer/reasoning path — the chatbot response, not memory work."""
    return ThrottledLM(
        model=CHAT_MODEL,
        api_key=_api_key(CHAT_MODEL),
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_TOKENS,
        num_retries=NUM_RETRIES,
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
