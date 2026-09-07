"""Code-only checks: no LLM calls, no embedding model. Run: pytest tests -q

Vectors here are hand-made, so these run in about a second and test the ranking
and storage logic rather than the quality of any model.
"""
import os
import tempfile

# isolate the store before any memory module imports
os.environ["MEMORY_DIR"] = tempfile.mkdtemp(prefix="memory_test_")
os.environ["MEMORY_PERSONAL_MODE"] = "0"   # these assert the benchmark-mode behaviour

import asyncio  # noqa: E402

import numpy as np  # noqa: E402

from memory.aggregate import _detect  # noqa: E402
from memory.consolidate import _clusters, _parse_pairs, unreconciled_sessions  # noqa: E402
from memory.grounding import question_phrases, unverified_terms  # noqa: E402
from memory.memory_store import (  # noqa: E402
    EMBEDDING_DIM, EmbeddedMemory, add_memory, delete_user_records,
    fetch_all_user_records, fetch_user_records_raw, get_all_categories,
    get_core_memory, mark_memory_old, mark_reconciled, search_memories,
    set_core_memory, to_epoch,
)
from memory.transcripts import TRANSCRIPT_DIR, archive_exchange  # noqa: E402

run = asyncio.run


def vec(*leading):
    """A unit-ish vector whose first components are given, rest zero."""
    v = [0.0] * EMBEDDING_DIM
    v[:len(leading)] = leading
    return v


def mem(user_id, text, session_id="s1", embedding=None, **kw):
    return EmbeddedMemory(
        user_id=user_id, memory_text=text, categories=kw.pop("categories", ["t"]),
        embedding=embedding or vec(1.0), date=kw.pop("date", "2026-01-01"),
        session_id=session_id, **kw,
    )


# --- pure functions -------------------------------------------------------

def test_aggregate_only_triggers_on_questions_about_the_user():
    assert _detect("How many concerts have I been to?") == "count"
    assert _detect("How long have I been at Zomato?") == "diff"
    assert _detect("How many calories are in an egg?") is None
    assert _detect("How many days until Diwali?") is None


def test_parse_pairs_drops_junk_and_out_of_range():
    assert _parse_pairs("N0:E2, n1:e5,N9:E0,foo,N0:E99", n_new=3, n_existing=6) == [(0, 2), (1, 5)]
    assert _parse_pairs("", 3, 6) == []


def test_to_epoch_handles_pre_1970_and_garbage():
    assert to_epoch("1965-06-01") < 0          # Windows timestamp() would raise here
    assert to_epoch("2000-01-01T00:00:00") == 946684800.0
    assert to_epoch("not a date") > 0          # falls back to now instead of raising


def test_clusters_groups_only_near_duplicates():
    a = np.array([1.0, 0.0, 0.0]); b = np.array([0.99, 0.1, 0.0]); c = np.array([0.0, 1.0, 0.0])
    items = [(i, {}, "", v) for i, v in enumerate([a, b, c])]
    assert _clusters(items) == [[0, 1]]
    assert _clusters(items[:1]) == []


def test_grounding_flags_multiword_proper_phrases_only():
    assert question_phrases("Where do I live?") == []
    assert question_phrases("Did I see Dr. Lee on Monday?") == ["Dr Lee"]  # punctuation stripped
    assert unverified_terms("Did I see Dr. Lee?", ["User saw Dr Lee in March"]) == []
    assert unverified_terms("Should I visit New York?", ["User lives in Bangalore"]) == ["New York"]
    assert unverified_terms("Is New York nice?", ["User is a New Yorker"]) == ["New York"]


# --- storage --------------------------------------------------------------

def test_forget_user_removes_memories_and_transcripts():
    async def go():
        await add_memory([mem(77, "User likes tea")])
        archive_exchange(77, "s1", "I like tea", "Noted")
        path = os.path.join(TRANSCRIPT_DIR, "user_77.jsonl")
        assert os.path.exists(path) and await fetch_all_user_records(77)
        await delete_user_records(77)
        assert not os.path.exists(path) and not await fetch_all_user_records(77)
    run(go())


def test_unreconciled_sessions_until_marked():
    async def go():
        ids = await add_memory([mem(78, "a", "s1"), mem(78, "b", "s2")])
        assert await unreconciled_sessions(78) == ["s1", "s2"]
        await mark_reconciled(ids[:1])
        assert await unreconciled_sessions(78) == ["s2"]
    run(go())


def test_core_profile_round_trips_and_stays_out_of_search():
    async def go():
        await add_memory([mem(79, "User bikes to work")])
        await set_core_memory(79, "Rides a bike.", vec(1.0))
        assert await get_core_memory(79) == "Rides a bike."
        await set_core_memory(79, "Rides a bike. Lives in Pune.", vec(1.0))
        assert await get_core_memory(79) == "Rides a bike. Lives in Pune."   # upsert, not duplicate
        # the profile is injected into every prompt, so it must never be a search hit
        texts = [m.memory_text for m in await search_memories(vec(1.0), 79, top_k=10)]
        assert texts == ["User bikes to work"]
        assert [r.memory_text for r in await fetch_all_user_records(79)] == ["User bikes to work"]
        assert await get_all_categories(79) == ["t"]
    run(go())


# --- retrieval ------------------------------------------------------------

def test_relevance_floor_excludes_unrelated_vectors():
    async def go():
        await add_memory([
            mem(80, "about hiking", embedding=vec(1.0, 0.0)),
            mem(80, "about taxes", embedding=vec(0.0, 1.0)),
        ])
        hits = await search_memories(vec(1.0, 0.0), 80, top_k=5, touch=False)
        # the orthogonal memory sits at cosine 0.0, under RELEVANCE_FLOOR
        assert [h.memory_text for h in hits] == ["about hiking"]
        both = await search_memories(vec(0.9, 0.4), 80, top_k=5, touch=False)
        assert [h.memory_text for h in both] == ["about hiking", "about taxes"]
    run(go())


def test_superseded_memories_hidden_unless_asked_for():
    async def go():
        ids = await add_memory([
            mem(81, "User lives in Delhi", embedding=vec(1.0)),
            mem(81, "User lives in Bangalore", embedding=vec(1.0, 0.05)),
        ])
        await mark_memory_old(ids[0])
        current = await search_memories(vec(1.0), 81, top_k=5, touch=False)
        assert [h.memory_text for h in current] == ["User lives in Bangalore"]
        with_old = await search_memories(vec(1.0), 81, top_k=5, include_old=True, touch=False)
        assert len(with_old) == 2
        # the superseded one carries the penalty, so it must not outrank the current fact
        assert with_old[0].memory_text == "User lives in Bangalore"
        assert with_old[1].is_current == 0
    run(go())


def test_touch_rehearses_only_when_asked():
    async def go():
        await add_memory([mem(82, "User plays chess", embedding=vec(1.0))])
        strength = lambda recs: float(recs["metadatas"][0]["strength"])
        before = strength(await fetch_user_records_raw(82, include_embeddings=False))
        await search_memories(vec(1.0), 82, top_k=5, touch=False)
        assert strength(await fetch_user_records_raw(82, include_embeddings=False)) == before
        await search_memories(vec(1.0), 82, top_k=5, touch=True)
        assert strength(await fetch_user_records_raw(82, include_embeddings=False)) > before
    run(go())


def test_admission_floor_drops_low_importance_facts():
    """The write path stores everything at threshold 0 and only facts at or above
    the floor once it is set — the knob the admission sweep turns."""
    import memory.update_memory as um

    class Fact:
        def __init__(self, text, importance):
            self.information, self.importance = text, importance
            self.about_user, self.date, self.status = True, "", "happened"
            self.predicted_category, self.keywords, self.context = ["t"], [], ""

    class Extracted:
        no_info = False
        new_memories = [Fact("trivial", 2), Fact("worth keeping", 8)]

    stored: list = []

    async def fake_extract(*a, **kw):
        return Extracted()

    async def fake_store(user_id, facts, dates, session_id=""):
        stored.append([f.information for f in facts])
        return []

    async def fake_categories(**kw):
        return []

    orig = (um._extract_with_retry, um._store_all, um.get_all_categories,
            um._completeness_pass, os.environ.get("MEMORY_MIN_IMPORTANCE"))
    um._extract_with_retry, um._store_all = fake_extract, fake_store
    um.get_all_categories = fake_categories
    um._completeness_pass = lambda *a, **kw: _zero()
    try:
        os.environ["MEMORY_MIN_IMPORTANCE"] = "0"
        run(um.update_memories(1, [{"role": "user", "content": "hi"}]))
        assert stored[-1] == ["trivial", "worth keeping"]

        os.environ["MEMORY_MIN_IMPORTANCE"] = "5"
        run(um.update_memories(1, [{"role": "user", "content": "hi"}]))
        assert stored[-1] == ["worth keeping"]
    finally:
        (um._extract_with_retry, um._store_all, um.get_all_categories,
         um._completeness_pass) = orig[:4]
        if orig[4] is None:
            os.environ.pop("MEMORY_MIN_IMPORTANCE", None)
        else:
            os.environ["MEMORY_MIN_IMPORTANCE"] = orig[4]


async def _zero():
    return 0
