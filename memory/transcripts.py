import json
import os
import re
from datetime import datetime

from rank_bm25 import BM25Okapi

from memory import MEMORY_DIR

TRANSCRIPT_DIR = os.path.join(MEMORY_DIR, "transcripts")
TURN_SEARCH_TOP_K = 5


def delete_transcripts(user_id: int):
    path = os.path.join(TRANSCRIPT_DIR, f"user_{user_id}.jsonl")
    if os.path.exists(path):
        os.remove(path)


# Secrets never enter the log: every source (chat, jots, Claude Code transcripts,
# git) writes through archive_exchange, so this is the one place to scrub.
_SECRET_RES = [
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{16,}"),                 # OpenAI/Stripe-style
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),                           # Google API key
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),  # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),                     # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                 # AWS access key id
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-.=+/]{16,}"),
    re.compile(r"(?i)\b([A-Z_]*(?:api[_-]?key|secret|token|password|passwd)[A-Z_]*)(\s*[=:]\s*['\"]?)[A-Za-z0-9_\-.=+/]{12,}"),
]


def redact(text: str) -> str:
    if not text:
        return text
    for rx in _SECRET_RES[:-1]:
        text = rx.sub("[REDACTED]", text)
    return _SECRET_RES[-1].sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)


def archive_exchange(user_id: int, session_id: str, user_msg: str, assistant_msg: str, ts: str = ""):
    """Append one exchange to the user's raw transcript log (experience bank).

    Raw transcripts are kept separately from distilled memories so history can
    be re-extracted later when the memory pipeline improves — and searched
    directly, so what the ASSISTANT said stays recallable.
    """
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    line = {
        "session_id": session_id,
        "ts": ts or datetime.now().isoformat(),
        "user": redact(user_msg),
        "assistant": redact(assistant_msg),
    }
    with open(os.path.join(TRANSCRIPT_DIR, f"user_{user_id}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_transcripts(user_id: int, session_id: str = None) -> list[dict]:
    path = os.path.join(TRANSCRIPT_DIR, f"user_{user_id}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            if session_id is None or line.get("session_id") == session_id:
                out.append(line)
    return out


STOPWORDS = {
    "a", "an", "the", "i", "you", "we", "my", "me", "your", "it", "is", "was", "were",
    "am", "are", "be", "been", "do", "did", "does", "have", "has", "had", "of", "to",
    "in", "on", "at", "for", "with", "about", "from", "and", "or", "but", "that",
    "this", "what", "which", "who", "how", "when", "where", "can", "could", "would",
    "will", "s", "t", "if", "so", "as", "by", "our", "us", "any", "some", "just",
    "remind", "tell", "know", "think", "previous", "earlier", "again",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def search_turns(user_id: int, query_text: str, top_k: int = TURN_SEARCH_TOP_K,
                 session_prefix="") -> list[dict]:
    """BM25 search over raw exchanges — recalls what was actually said,
    including assistant answers that fact extraction never stores.

    BM25's IDF goes negative when a term appears in most of a small corpus, so
    a score threshold silently drops everything on short histories. Rank by
    BM25, but gate on real (non-stopword) word overlap instead.
    """
    lines = [l for l in load_transcripts(user_id) if l["session_id"].startswith(session_prefix)]
    if not lines or not query_text:
        return []
    docs = [_tokenize(l["user"] + " " + l["assistant"]) for l in lines]
    query = _tokenize(query_text)
    content_words = {w for w in query if w not in STOPWORDS}
    if not content_words:
        return []

    scores = BM25Okapi(docs).get_scores(query)
    candidates = []
    for i, doc in enumerate(docs):
        overlap = len(content_words & set(doc))
        if overlap:
            candidates.append((scores[i], overlap, i))
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return [lines[i] for _, _, i in candidates[:top_k]]


def stringify_turn(line: dict) -> str:
    # generous caps: truncating a long assistant answer (a list, a game record)
    # cuts exactly the detail these excerpts exist to preserve
    date = (line.get("ts") or "")[:10]
    return (
        f"[PAST CONVERSATION {date}] "
        f"User: {line['user'][:600]} | Assistant: {line['assistant'][:4000]}"
    )