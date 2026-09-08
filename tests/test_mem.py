"""One check for the two branches in mem.py that can silently go wrong:
the per-day transcript filter and the forget-by-prefix guard."""
import mem


def test_day_filter_and_forget_prefix(monkeypatch):
    lines = [
        {"session_id": "mem-2026-09-05", "ts": "2026-09-05T09:00:00", "user": "a", "assistant": ""},
        {"session_id": "mem-2026-09-06", "ts": "2026-09-06T09:00:00", "user": "b", "assistant": ""},
    ]
    monkeypatch.setattr("memory.transcripts.load_transcripts", lambda uid: lines)
    assert [l["user"] for l in mem._day("2026-09-05")] == ["a"]

    class R:
        def __init__(self, pid):
            self.point_id = pid
            self.memory_text = pid

    recs = [R("abc123"), R("abd456")]

    async def fake_fetch(uid):
        return recs

    deleted = []

    async def fake_delete(ids):
        deleted.extend(ids)

    monkeypatch.setattr("memory.memory_store.fetch_all_user_records", fake_fetch)
    monkeypatch.setattr("memory.memory_store.delete_records", fake_delete)
    mem.forget("ab")  # ambiguous: two matches, must not delete
    assert deleted == []
    mem.forget("abd")
    assert deleted == ["abd456"]


def test_redact_scrubs_key_shapes_and_keeps_prose():
    from memory.transcripts import redact
    s = ("set OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234 and GEMINI_API_KEY: AIzaSyA-1234567890abcdefghijklmnopqrstuv "
         "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789 Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abcdefghijklmnop "
         "password: hunter2hunter2hunter2 ; my desk is by the window, commit 6638776")
    out = redact(s)
    for leaked in ("sk-abc", "AIzaSy", "ghp_", "eyJ", "hunter2"):
        assert leaked not in out, leaked
    assert "my desk is by the window, commit 6638776" in out
    assert out.count("[REDACTED]") == 5
