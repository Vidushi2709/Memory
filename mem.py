"""mem: jot things down, get them back.

    mem                        profile, recent jots, then a prompt (blank line quits)
    mem I moved my desk        jot it (free, instant, timestamped)
    mem ? desk                 search your jots and known facts (free; `find` works too)
    mem on 2026-09-05          everything said that day; also: today, yesterday
    mem what changed?          ends with "?": answer from memory (one LLM call)
                               (`mem ask ...` does the same without the "?")
    mem week                   digest of the last seven days (one LLM call)
    mem upcoming               facts dated after today (also shown when `mem` opens)
    mem ping                   message box with what is due today/tomorrow (9 am task)
    mem sleep                  log new Claude Code sessions + git commits, extract,
                               consolidate, rebuild profile (nightly task does this)
    mem all                    every current fact, with ids
    mem profile                the standing profile
    mem forget 3f2a            delete one fact by id prefix

Jots go to the raw transcript log and cost nothing; `sleep` is where the LLM
turns them into facts. One store for everything: memory.db + transcripts/
next to this file.
"""
import asyncio
import os
import sys
from datetime import date, timedelta

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO, ".env"))  # llm.py's bare load_dotenv() only looks in the cwd

USER = int(os.getenv("MEM_USER", "1"))
run = asyncio.run
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252

# colours: any rich style name or hex, edit freely
DATE_STYLE, TEXT_STYLE, TAG_STYLE = "hot_pink", "medium_purple", "grey50"
ANSWER_STYLES = ["hot_pink", "medium_purple", "dark_orange", "orchid", "light_salmon1", "plum1"]
def _session(d=None) -> str:
    return f"mem-{(d or date.today()).isoformat()}"


def _day(when: str) -> list:
    """Transcript lines from one day. Accepts an ISO date, 'today' or 'yesterday'."""
    from memory.transcripts import load_transcripts
    d = {"today": date.today(), "yesterday": date.today() - timedelta(days=1)}.get(when)
    d = d or date.fromisoformat(when)
    return [l for l in load_transcripts(USER) if (l.get("ts") or "").startswith(d.isoformat())]


def _match(prefix: str) -> list:
    from memory.memory_store import fetch_all_user_records
    return [r for r in run(fetch_all_user_records(USER)) if r.point_id.startswith(prefix)]


def _table(rows, title: str = ""):
    """rows of (date, text[, tag]) as a wrapped, coloured table."""
    from rich.console import Console
    from rich.table import Table
    t = Table(title=title or None, show_header=False, box=None, padding=(0, 1), title_justify="left")
    t.add_column(style=DATE_STYLE, no_wrap=True)
    t.add_column(style=TEXT_STYLE)
    t.add_column(style=TAG_STYLE, no_wrap=True)
    for r in rows:
        t.add_row(*r, *([""] * (3 - len(r))))
    Console().print(t)


def _when(l: dict, chars: int = 16) -> str:
    return (l.get("ts") or "")[:chars].replace("T", " ")


def jot(text: str):
    from memory.transcripts import archive_exchange
    archive_exchange(USER, _session(), text, "")
    print("noted.")


def on(when: str):
    lines = _day(when)
    _table([(_when(l), l["user"], "") for l in lines])
    print(f"{len(lines)} line(s) on {when}")


def _words(text: str) -> set:
    import re
    from memory.transcripts import STOPWORDS
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def search(q: str):
    """Jots and facts, one dated line each. Raw Claude Code turns are for
    `mem ask`, where the model reads them; printed raw they bury the answer."""
    from memory import memory_store
    from memory.transcripts import search_turns
    jots = search_turns(USER, q, top_k=5, session_prefix=("mem-", "git-"))
    if jots:
        _table([(_when(l, 10), l["user"]) for l in jots], "you said")
    # keyword-only: loading the embedding model costs ~3 s per process and the
    # retrieval sweep put BM25 within noise of vectors. `ask` and `claude` keep vectors.
    memory_store.USE_VECTOR = False
    facts = run(memory_store.search_memories(search_vector=[], user_id=USER, query_text=q,
                                             include_old=True, top_k=5, touch=False))
    # ponytail: nearest-neighbour search always returns something; keep only facts
    # that share a real word with the question so filler never prints. Costs
    # synonym matches (desk/table); relax if that bites.
    words = _words(q)
    facts = [m for m in facts if words & _words(m.memory_text)][:3]
    if facts:
        _table([(m.date[:10], m.memory_text, "" if m.is_current else "old") for m in facts], "known")
    if not jots and not facts:
        print("nothing.")


def _complete(prompt: str) -> str:
    """One chat completion over plain HTTPS. dspy/litellm cost 12 s to import;
    this costs 0.4 s. Same routing as memory/llm.py: Gemini direct if
    GEMINI_API_KEY is set, else the same model through OpenRouter."""
    import requests
    model = os.getenv("MEMORY_CHAT_MODEL") or os.getenv("MEMORY_MODEL") or "gemini-2.5-flash-lite"
    if os.getenv("GEMINI_API_KEY") and model.split("/")[0].startswith("gemini"):
        model = model.split("/")[-1]
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0, "maxOutputTokens": 1024}},
            timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    # OpenRouter names are provider/model; a bare or gemini/-prefixed name is Google's
    model = model.removeprefix("openrouter/")
    if "/" not in model or model.startswith("gemini/"):
        model = "google/" + model.split("/")[-1]
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ[os.getenv("MEMORY_API_KEY_ENV", "OPEN_ROUTER_KEY")]},
        json={"model": model.removeprefix("openrouter/"), "temperature": 0, "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def ask(q: str):
    from memory.embedding_generation import generate_embeddings
    from memory.memory_store import get_core_memory, search_memories, stringify_retrieved_point
    from memory.transcripts import search_turns, stringify_turn
    vec = run(generate_embeddings([q]))[0]
    facts = [stringify_retrieved_point(m) for m in run(search_memories(
        search_vector=vec, user_id=USER, query_text=q, include_old=True, touch=False))]
    turns = [stringify_turn(t) for t in search_turns(USER, q)]
    prompt = f"""You are the user's memory. Today is {date.today().isoformat()}.
Answer the question from the material below only. Quote the date something was
said when it matters. If the material does not contain the answer, say you have
no record of it; never guess. Facts marked OLD/SUPERSEDED were true once and
are no longer current. Be brief.

PROFILE:
{run(get_core_memory(USER)) or "(none)"}

FACTS:
{chr(10).join(facts) or "(none)"}

THINGS SAID (verbatim, dated):
{chr(10).join(turns) or "(none)"}

QUESTION: {q}"""
    import random
    from rich.console import Console
    from rich.markdown import Markdown
    Console().print(Markdown(_complete(prompt)), style=random.choice(ANSWER_STYLES))


CLAUDE_PROJECTS = os.path.expanduser("~/.claude/projects")
CHUNK = 20                       # exchanges per extraction call for Claude Code sessions
MAX_CHUNKS = int(os.getenv("MEM_SLEEP_MAX_CHUNKS", "30"))  # ponytail: nightly spend cap (~₹0.1/chunk)


def _ingest_claude() -> int:
    """Append new human<->assistant exchanges from every Claude Code session to the
    transcript log as cc-<project>-<uuid8>-<chunk> sessions. Idempotent: an
    exchange already logged (same session, same prompt) is skipped."""
    import glob
    import json
    from memory.transcripts import archive_exchange, load_transcripts
    seen, chunks = set(), {}
    for l in load_transcripts(USER):
        sid = l["session_id"]
        if sid.startswith("cc-"):
            base, _, n = sid.rpartition("-")
            seen.add((base, l["user"][:200]))
            chunks[base] = max(chunks.get(base, 0), int(n) + 1 if n.isdigit() else 0)
    added = 0
    for f in sorted(glob.glob(os.path.join(CLAUDE_PROJECTS, "*", "*.jsonl"))):
        proj = os.path.basename(os.path.dirname(f)).replace("C--Users-VIDUSHI-Desktop-", "")
        base = f"cc-{proj}-{os.path.basename(f)[:8]}"
        pairs, cur = [], None
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("isSidechain"):
                continue
            c = (d.get("message") or {}).get("content")
            human = (d.get("origin") or {}).get("kind", "human") == "human"
            if d.get("type") == "user" and isinstance(c, str) and not c.lstrip().startswith("<") and human:
                cur = {"ts": d.get("timestamp", "")[:19], "user": c.strip(), "assistant": []}
                pairs.append(cur)
            elif d.get("type") == "assistant" and cur is not None and isinstance(c, list):
                cur["assistant"] += [b.get("text", "") for b in c if b.get("type") == "text"]
        fresh = [p for p in pairs if (base, p["user"][:200]) not in seen]
        # ponytail: a session still in progress gets its tail chunk numbered later;
        # chunks are extraction units, not exact 20-blocks. Fine.
        n = chunks.get(base, 0)
        for i, p in enumerate(fresh):
            archive_exchange(USER, f"{base}-{n + i // CHUNK}", p["user"], "\n".join(p["assistant"]).strip(), ts=p["ts"])
            added += 1
    return added


GIT_ROOT = os.getenv("MEM_GIT_ROOT", os.path.expanduser("~/Desktop"))


def _ingest_git() -> int:
    """Log your own commits from every repo under GIT_ROOT (two levels deep) as
    git-<repo> lines dated by commit time. Read-only `git log`; no extraction,
    a commit subject already is the fact."""
    import glob
    import subprocess
    from memory.transcripts import archive_exchange, load_transcripts
    # keyed on time + subject, not repo: two clones of one repo are one history
    seen = {(l["ts"][:19], l["user"].split("] ", 1)[-1]) for l in load_transcripts(USER) if l["session_id"].startswith("git-")}
    me = subprocess.run(["git", "config", "--global", "user.name"], capture_output=True, text=True).stdout.strip()
    added = 0
    for g in glob.glob(os.path.join(GIT_ROOT, "*", ".git")) + glob.glob(os.path.join(GIT_ROOT, "*", "*", ".git")):
        repo = os.path.dirname(g)
        sid = f"git-{os.path.basename(repo)}"
        out = subprocess.run(["git", "-C", repo, "log", "--all", f"--author={me}", "--format=%cI%x09%s"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        for line in out.splitlines():
            ts, _, subject = line.partition("\t")
            if (ts[:19], subject.strip()) in seen:
                continue
            seen.add((ts[:19], subject.strip()))
            archive_exchange(USER, sid, f"[{os.path.basename(repo)}] {subject.strip()}", "", ts=ts[:19])
            added += 1
    return added


def sleep():
    from memory.consolidate import sleep_pass, unreconciled_sessions
    from memory.memory_store import fetch_all_user_records
    from memory.transcripts import load_transcripts
    from memory.update_memory import update_memories
    print(f"claude code: {_ingest_claude()} new exchange(s) logged; git: {_ingest_git()} new commit(s) logged")
    # a chunk counts as done if it produced facts OR is listed in the marker file
    # (zero-fact chunks would otherwise be re-sent every night and hog the cap)
    done_file = os.path.join(REPO, "transcripts", "extracted.txt")
    extracted = {r.session_id for r in run(fetch_all_user_records(USER))}
    if os.path.exists(done_file):
        extracted |= set(open(done_file, encoding="utf-8").read().split())
    todo = {}
    for l in load_transcripts(USER):
        s = l["session_id"]
        if (s.startswith("mem-") or s.startswith("cc-")) and s not in extracted:
            todo.setdefault(s, []).append({"role": "user", "content": l["user"][:1500]})
            if l.get("assistant"):
                todo[s].append({"role": "assistant", "content": l["assistant"][:1500]})
    pending = sorted(todo)
    if len(pending) > MAX_CHUNKS:
        print(f"{len(pending)} chunk(s) waiting; doing {MAX_CHUNKS} tonight (MEM_SLEEP_MAX_CHUNKS)")
    for s in pending[:MAX_CHUNKS]:
        day = s[4:14] if s.startswith("mem-") else (load_transcripts(USER, s)[0].get("ts") or "")[:10]
        lens = "life" if s.startswith("mem-") else "work"
        print(f"{s}: {run(update_memories(USER, todo[s], session_id=s, current_date=day, lens=lens))}")
        with open(done_file, "a", encoding="utf-8") as f:
            f.write(s + "\n")
    pending = run(unreconciled_sessions(USER))
    if pending:
        print(run(sleep_pass(USER, pending)))
    else:
        print("nothing new to consolidate.")
    profile()


def show_all():
    from memory.memory_store import fetch_all_user_records
    recs = sorted((r for r in run(fetch_all_user_records(USER)) if r.is_current),
                  key=lambda r: r.date)
    _table([(r.date[:10], r.memory_text, r.point_id[:8]) for r in recs])
    print(f"{len(recs)} current fact(s)")


def profile():
    from memory.memory_store import get_core_memory
    print(run(get_core_memory(USER)) or "(no profile yet; `mem sleep` builds it)")


def forget(prefix: str):
    from memory.memory_store import delete_records
    hits = _match(prefix)
    if len(hits) != 1:
        print(f"{len(hits)} fact(s) start with {prefix!r}; use a longer prefix from `mem all`")
        return
    run(delete_records([hits[0].point_id]))
    print(f"forgot: {hits[0].memory_text}")


def upcoming():
    """Current facts dated after today: deadlines, plans, appointments."""
    from memory.memory_store import fetch_all_user_records
    today = date.today().isoformat()
    ahead = sorted((r for r in run(fetch_all_user_records(USER)) if r.is_current and r.date[:10] > today),
                   key=lambda r: r.date)
    if ahead:
        _table([(r.date[:10], r.memory_text) for r in ahead], "upcoming")
    return len(ahead)


def ping(days: int = 1):
    """Facts due today or within `days`: show them in a Windows message box.
    Runs on every unlock and at logon, but pops at most once a day (stamp file).
    Stdlib only, no model call."""
    import ctypes
    from memory.memory_store import fetch_user_records_raw
    lo, hi = date.today().isoformat(), (date.today() + timedelta(days=days)).isoformat()
    stamp = os.path.join(REPO, "transcripts", "pinged.txt")
    if os.path.exists(stamp) and open(stamp, encoding="utf-8").read().strip() == lo:
        return print("already pinged today.")
    # forward-dated when saved = a plan or deadline; same-day = just something said today
    metas = run(fetch_user_records_raw(USER, include_embeddings=False))["metadatas"]
    due = sorted((m for m in metas if int(m.get("is_current", 1)) and m.get("type") != "core"
                  and lo <= str(m.get("date", ""))[:10] <= hi
                  and str(m.get("date", ""))[:10] > str(m.get("saved_at", ""))[:10]),
                 key=lambda m: m["date"])
    with open(stamp, "w", encoding="utf-8") as f:
        f.write(lo)
    if not due:
        return print("nothing due.")
    text = "\n\n".join(f"{m['date'][:10]}  {m.get('memory_text') or m.get('text', '')}" for m in due)
    print(text)
    if os.name == "nt":
        # MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST
        ctypes.windll.user32.MessageBoxW(None, text, "mem: coming up", 0x40 | 0x10000 | 0x40000)


def week():
    """One LLM call over the last seven days of notes, commits and Claude Code
    sessions: a standup-ready digest."""
    from memory.transcripts import load_transcripts
    since = (date.today() - timedelta(days=7)).isoformat()
    lines = [l for l in load_transcripts(USER) if (l.get("ts") or "")[:10] >= since]
    if not lines:
        return print("nothing in the last seven days.")
    # ponytail: flat 60k-char cap (~15k tokens, ~₹0.15). Sample instead of cut if weeks get bigger.
    body, kept = [], 0
    for l in lines:
        row = f"{_when(l)} [{l['session_id'].split('-')[0]}] {l['user'][:400]}"
        if l.get("assistant"):
            row += f"\n    -> {l['assistant'][:300]}"
        if kept + len(row) > 60000:
            break
        body.append(row)
        kept += len(row)
    prompt = f"""Today is {date.today().isoformat()}. Below is one week of the user's own notes
([mem]), git commits ([git]) and coding-assistant sessions ([cc]), oldest first.
Write a short digest for the user: what they worked on, grouped by project; what
they decided or changed; anything still open or due. Plain language, markdown
bullets, dates where they matter. Do not invent anything not in the log.

{chr(10).join(body)}"""
    import random
    from rich.console import Console
    from rich.markdown import Markdown
    print(f"{len(body)} line(s) since {since}")
    Console().print(Markdown(_complete(prompt)), style=random.choice(ANSWER_STYLES))


def home():
    from memory.transcripts import load_transcripts
    profile()
    upcoming()
    lines = load_transcripts(USER)
    recent = [l for l in lines if l["session_id"].startswith("mem-")][-10:]
    if recent:
        _table([(_when(l), l["user"]) for l in recent], "recent")
    print(f"\n{len(lines)} line(s) in the log. Speak or type a note to jot it; end with ? or start with ask to get an answer; blank line quits.")
    while True:
        try:
            t = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t:
            break
        if t.startswith("?"):
            search(t[1:].strip())
        elif t.endswith("?") or t.lower().startswith("ask "):
            ask(t[4:] if t.lower().startswith("ask ") else t)
        else:
            jot(t)


def hook():
    """SessionStart hook: hand Claude Code the profile and recent jots as JSON."""
    import json
    from memory.memory_store import get_core_memory
    from memory.transcripts import load_transcripts
    jots = [l for l in load_transcripts(USER) if l["session_id"].startswith("mem-")][-10:]
    ctx = "The user's laptop-wide memory (`mem`). Profile:\n" + (run(get_core_memory(USER)) or "(none yet)")
    if jots:
        ctx += "\n\nRecent notes the user jotted:\n" + "\n".join(f"- {_when(l)}  {l['user']}" for l in jots)
    ctx += "\n\nUse /mem <question>? to look something up in it, /mem <note> to save one."
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))


def claude(text: str):
    """Slash command body: a question prints raw matches for Claude to read (no
    Gemini call); anything else is a jot."""
    if not text.endswith("?"):
        return jot(text)
    from memory.embedding_generation import generate_embeddings
    from memory.memory_store import search_memories, stringify_retrieved_point
    from memory.transcripts import search_turns, stringify_turn
    q = text.rstrip("?")
    for l in search_turns(USER, q, top_k=5, session_prefix="mem-"):
        print(f"[NOTE {_when(l)}] {l['user']}")
    vec = run(generate_embeddings([q]))[0]
    for m in run(search_memories(search_vector=vec, user_id=USER, query_text=q, include_old=True, top_k=8, touch=False)):
        print(stringify_retrieved_point(m))
    for t in search_turns(USER, q, top_k=5):
        print(stringify_turn(t)[:1200])


def main(argv: list):
    if not argv:
        return home()
    cmd, rest = argv[0], " ".join(argv[1:])
    if cmd in ("?", "find"):
        return search(rest)
    if cmd == "on":
        return on(rest or "today")
    if cmd == "ask":
        return ask(rest)
    if cmd == "sleep":
        return sleep()
    if cmd == "all":
        return show_all()
    if cmd == "profile":
        return profile()
    if cmd == "forget":
        return forget(rest)
    if cmd == "hook":
        return hook()
    if cmd == "week":
        return week()
    if cmd == "upcoming":
        return print(f"{upcoming()} upcoming")
    if cmd == "ping":
        return ping()
    if cmd == "claude":
        return claude(rest)
    text = " ".join(argv)
    # ponytail: a trailing "?" means a question. A jot that happens to end in "?"
    # ("buy the desk?") becomes an ask; drop the "?" or it stays a question.
    return ask(text) if text.endswith("?") else jot(text)


if __name__ == "__main__":
    main(sys.argv[1:])
