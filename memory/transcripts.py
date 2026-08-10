import json
import os
from datetime import datetime

TRANSCRIPT_DIR = "./transcripts"


def archive_exchange(user_id: int, session_id: str, user_msg: str, assistant_msg: str):
    """Append one exchange to the user's raw transcript log (experience bank).

    Raw transcripts are kept separately from distilled memories so history can
    be re-extracted later when the memory pipeline improves.
    """
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    line = {
        "session_id": session_id,
        "ts": datetime.now().isoformat(),
        "user": user_msg,
        "assistant": assistant_msg,
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
