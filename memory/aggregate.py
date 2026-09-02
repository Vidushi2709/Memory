"""Aggregation path: answer "how many / how long" by scanning, not searching.

Counting questions fail through every model tried: deciding which memories
belong to a category and doing arithmetic over them is one fused generation,
and top-k search cannot promise the model it saw every member. So decompose:
collect candidates from a FULL store scan with status/date filters in code,
let one small LLM call judge membership only, do the arithmetic in code, and
hand the responder a computed number it reports instead of derives.

Empty string on any miss (no trigger, no members, unresolved anchors) — the
normal retrieval path is untouched.
"""
import re
from datetime import datetime

import dspy

from memory.llm import get_lm, was_truncated
from memory.memory_store import fetch_all_user_records

CANDIDATE_CAP = 60  # membership call size limit; lexical prefilter beyond this

# "how many days/weeks/..." is a time-span question, not a count of things
_UNITS = r"(?:days?|weeks?|months?|years?|hours?)"
_DIFF_RE = re.compile(
    rf"\bhow many {_UNITS}\b|\bhow long\b|\b{_UNITS} (?:had |have )?passed\b"
    rf"|\bdays? between\b|\bdid it take\b",
    re.I,
)
_COUNT_RE = re.compile(r"\bhow many\b", re.I)
# "how many do I NEED TO pick up" counts open obligations, which live under
# status=planned — the exact records the past-tense filter must exclude
_PENDING_RE = re.compile(
    r"\b(?:do|does|will) i (?:need|have) to\b|\bneed to (?:pick up|return|collect|get|buy|do)\b"
    r"|\bstill (?:need|have|owe)\b|\bpending\b|\bleft to\b",
    re.I,
)

_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

_STOP = {
    "a", "an", "the", "i", "my", "me", "we", "you", "your", "of", "to", "in",
    "on", "at", "for", "with", "from", "and", "or", "did", "do", "does",
    "have", "has", "had", "was", "were", "is", "are", "it", "how", "many",
    "much", "long", "since", "between", "when", "passed", "take", "went",
    "go", "been", "days", "weeks", "months", "years", "last", "ago", "that",
    "this", "what", "which", "there",
}


class SelectMembers(dspy.Signature):
    """The question asks how many of something. From the candidate memories,
    select the ones that each describe ONE distinct instance of the kind given
    in instance_kind. Instances may be described in different words — a clinic
    visit where a doctor examined the user IS a doctor's appointment. Do not
    select general observations, habits, or summaries — only concrete
    instances. If two memories describe the same instance, select only one.

    Memories sharing a date are the prime suspects for this. Extraction often
    writes the same event twice, once generically and once with the detail
    ("attended a concert yesterday" alongside "attended the indie band's show
    at Fandom yesterday"). Same date, same kind of event, and the generic one
    adds no instance the specific one does not already cover: that is ONE
    instance. Keep the specific memory and drop the generic one.

    Output the 0-based indices, comma-separated; empty if none.
    """

    question: str = dspy.InputField()
    instance_kind: str = dspy.InputField(desc="what counts as one instance for this question")
    candidates: list[str] = dspy.InputField()
    member_indices: str = dspy.OutputField(desc='e.g. "0,3,4"; empty string if none')


class SelectAnchors(dspy.Signature):
    """The question asks for the time between two events, or how long ago one
    event happened. Pick the memory describing the START event and the memory
    describing the END event. If the end point is "now" (the question asks how
    long has passed up to today), set end_index to -1. If either anchor is not
    present among the candidates, set that index to -2.
    """

    question: str = dspy.InputField()
    candidates: list[str] = dspy.InputField()
    start_index: int = dspy.OutputField()
    end_index: int = dspy.OutputField()


_members = dspy.Predict(SelectMembers)
_anchors = dspy.Predict(SelectAnchors)


_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'?ve|i'?m|me|my|we|our)\b", re.I)


def _detect(question: str):
    # "how many calories in an egg" is general knowledge: only a question about
    # the user's own record earns a full store scan plus an LLM selection call
    if not _FIRST_PERSON_RE.search(question):
        return None
    if _DIFF_RE.search(question):
        return "diff"
    if _COUNT_RE.search(question):
        return "count"
    return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_words(question: str) -> set:
    return {w for w in _tokens(question) if w not in _STOP}


def _month_window(question: str, dates: list[str], today: datetime):
    """'in March' -> (2023-03-01, 2023-03-31), year resolved from the data:
    the unique year with that month among candidates, else nearest to today.
    None when no month is named or the year stays ambiguous."""
    m = re.search(r"\bin (" + "|".join(_MONTHS) + r")\b(?: (\d{4}))?", question, re.I)
    if not m:
        return None
    month = _MONTHS[m.group(1).lower()]
    if m.group(2):
        year = int(m.group(2))
    else:
        years = sorted({int(d[:4]) for d in dates if len(d) >= 7 and int(d[5:7]) == month})
        if len(years) == 1:
            year = years[0]
        elif years:
            year = min(years, key=lambda y: abs(y - today.year))
        else:
            return None
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return start, end


_NOUN_SKIP = _STOP | {"new", "more", "additional", "other", "different", "total", "own"}


def _counted_noun_stem(question: str) -> str:
    """The thing being counted: first content word after 'how many', singular.
    Qualifier adjectives ("new postcards") are skipped or the guard below
    searches for the wrong word."""
    m = re.search(r"\bhow many\b(.*)", question, re.I)
    if not m:
        return ""
    for w in _tokens(m.group(1)):
        if w not in _NOUN_SKIP:
            return w.rstrip("s")
    return ""


def _parse_date(raw: str):
    try:
        return datetime.fromisoformat(raw[:19])
    except (ValueError, TypeError):
        return None


def _fmt(i: int, r) -> str:
    return f"{i}. [{r.status}] {r.date[:10]} — {r.memory_text[:160]}"


async def _select(predictor, **kwargs):
    lm = get_lm()
    with dspy.context(lm=lm):
        out = await predictor.acall(**kwargs)
    if was_truncated(lm):
        raise RuntimeError("selection truncated")
    return out


def _parse_indices(raw: str, n: int) -> list[int]:
    picked = []
    for chunk in re.findall(r"-?\d+", raw or ""):
        i = int(chunk)
        if 0 <= i < n and i not in picked:
            picked.append(i)
    return picked


async def maybe_aggregate(user_id: int, question: str, current_date: str = "") -> str:
    kind = _detect(question)
    if kind is None:
        return ""
    today = _parse_date(current_date) or datetime.now()

    try:
        records = await fetch_all_user_records(user_id)
    except Exception:
        return ""
    current = [r for r in records if r.is_current and r.kind == "fact"]
    if not current:
        return ""

    if kind == "count":
        return await _aggregate_count(question, current, today)
    return await _aggregate_diff(question, current, today)


async def _aggregate_count(question: str, current, today) -> str:
    # past-tense questions count events; "need to" questions count open plans
    pending = bool(_PENDING_RE.search(question))
    statuses = ("planned", "ongoing") if pending else ("happened", "ongoing")
    kind_desc = (
        "a pending task or obligation the user has not completed yet"
        if pending else
        "an actual event or thing that happened or exists (a plan or intention is NOT one)"
    )
    candidates = [r for r in current if r.status in statuses]
    if not pending:
        # You cannot already have attended a concert that happens next January.
        # Extraction labels "booked tickets for the January show" as happened —
        # the booking did happen — while correctly dating the event in the
        # future, so status alone does not keep it out of a past-tense count.
        candidates = [r for r in candidates
                      if not ((d := _parse_date(r.date)) and d.date() > today.date())]

    window = _month_window(question, [r.date for r in candidates], today)
    if window:
        start, end = window
        candidates = [r for r in candidates
                      if (d := _parse_date(r.date)) and start <= d < end]
    elif len(candidates) > CANDIDATE_CAP:
        # no window to narrow by — keep the lexically closest, best-first
        words = _content_words(question)
        scored = [(len(words & set(_tokens(r.memory_text))), i, r)
                  for i, r in enumerate(candidates)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))
        candidates = [r for _, _, r in scored[:CANDIDATE_CAP]]
    if not candidates:
        return ""

    try:
        out = await _select(_members, question=question, instance_kind=kind_desc,
                            candidates=[_fmt(i, r) for i, r in enumerate(candidates)])
    except Exception:
        return ""
    members = [candidates[i] for i in _parse_indices(out.member_indices, len(candidates))]
    if not members:
        return ""

    # Counting RECORDS is wrong when the records carry their own totals
    # ("acquired 8 postcards"): those questions are answered by the stated
    # number, which the normal path already reports. Step aside.
    stem = _counted_noun_stem(question)
    if stem and any(
        re.search(rf"\b\d+\s+(?:\w+\s+){{0,2}}{re.escape(stem)}", r.memory_text, re.I)
        for r in members
    ):
        return ""

    excluded = ""
    if window and not pending:
        planned = [r for r in current
                   if r.status in ("planned", "considered")
                   and (d := _parse_date(r.date)) and window[0] <= d < window[1]]
        if planned:
            excluded = "\nExcluded (intentions, not events): " + "; ".join(
                r.memory_text[:80] for r in planned[:3])

    lines = "\n".join(f"  {i + 1}. {r.date[:10]} — {r.memory_text[:120]}"
                      for i, r in enumerate(members))
    span = f", window {window[0].date()}..{(window[1]).date()}" if window else ""
    scope = "pending items" if pending else "events only"
    return (
        "COMPUTED (full scan of stored memories, arithmetic done in code — "
        "not a search):\n"
        f"COUNT of instances matching the question{span}, {scope}\n"
        f"Members ({len(members)}):\n{lines}{excluded}\n"
        f"ANSWER BASIS: count = {len(members)}"
    )


async def _aggregate_diff(question: str, current, today) -> str:
    words = _content_words(question)
    scored = [(len(words & set(_tokens(r.memory_text))), i, r)
              for i, r in enumerate(current)]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    candidates = [r for _, _, r in scored[:CANDIDATE_CAP]]
    if not candidates:
        return ""

    try:
        out = await _select(_anchors, question=question,
                            candidates=[_fmt(i, r) for i, r in enumerate(candidates)])
        si, ei = int(out.start_index), int(out.end_index)
    except Exception:
        return ""
    if si < 0 or si >= len(candidates) or ei == -2 or ei >= len(candidates):
        return ""

    start_rec = candidates[si]
    start = _parse_date(start_rec.date)
    if ei == -1:
        end, end_desc = today, f"now ({today.date()})"
    else:
        end_rec = candidates[ei]
        end = _parse_date(end_rec.date)
        end_desc = f"{end_rec.date[:10]} — {end_rec.memory_text[:90]}"
    if not start or not end or end < start:
        return ""

    days = (end - start).days
    weeks = round(days / 7)
    months = round(days / 30.44)
    return (
        "COMPUTED (full scan of stored memories, arithmetic done in code — "
        "not a search):\n"
        "TIME SPAN between stored events\n"
        f"  start: {start_rec.date[:10]} — {start_rec.memory_text[:90]}\n"
        f"  end:   {end_desc}\n"
        f"ANSWER BASIS: {days} day(s) ≈ {weeks} week(s) ≈ {months} month(s) — "
        "use the unit the question asks for"
    )
