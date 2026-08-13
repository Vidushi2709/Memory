"""
LongMemEval runner (oracle setting).

Ingests each question's evidence sessions through the real memory pipeline
(with historical session dates), asks the question as of its question_date,
and scores with an LLM judge. Stratified sample per question type.

Oracle caveat: only evidence sessions are ingested (no haystack noise), so
scores measure extraction/update/temporal ability, not needle-finding.

Settings:
    oracle    — only the evidence sessions (~3). Measures extraction/update/reasoning.
    haystack  — the _s file, ~40 sessions per question (~115k tokens). Adds the
                needle-finding problem, and costs ~10x more to ingest.

Usage (from the repo root):
    python eval/run_longmemeval.py                        # oracle, 5 per type (30 total)
    python eval/run_longmemeval.py --per-type 3
    python eval/run_longmemeval.py --types temporal-reasoning,knowledge-update
    python eval/run_longmemeval.py --ids 0862e8bf,118b2229
    python eval/run_longmemeval.py --haystack --per-type 1 --sleep-every 5
    python eval/run_longmemeval.py --out my-run.json     # → eval/results/my-run.json
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ORACLE_PATH = os.path.join(DATA_DIR, "longmemeval_oracle.json")
HAYSTACK_PATH = os.path.join(DATA_DIR, "longmemeval_s_cleaned.json")

SLEEP_EVERY = 1  # run the sleep pass after every Nth ingested session

from dotenv import dotenv_values

os.environ.setdefault("OPEN_ROUTER_KEY", dotenv_values(os.path.join(REPO, ".env")).get("OPEN_ROUTER_KEY") or "")

# Isolate ./chroma_db and ./transcripts before the memory modules import
_workdir = tempfile.mkdtemp(prefix="lme_eval_")
os.chdir(_workdir)
sys.path.insert(0, REPO)

import dspy
from rich.console import Console
from rich.table import Table
from rich import box

import chatbot
from memory.aggregate import maybe_aggregate
from memory.consolidate import sleep_pass
from memory.embedding_generation import generate_embeddings
from memory.memory_store import (
    fetch_all_user_records,
    get_core_memory,
    search_memories,
    stringify_retrieved_point,
)
from memory.grounding import unverified_terms
from memory.transcripts import archive_exchange, load_transcripts, search_turns, stringify_turn
from memory.update_memory import update_memories

console = Console()


class LMEJudgeSignature(dspy.Signature):
    """
    Judge whether the assistant's answer to a question about the user's chat
    history is correct, given the gold expected answer.

    If is_abstention is True, the information was never in the history:
    correct means the assistant indicated it doesn't have that information.
    Otherwise: paraphrase is fine, extra correct detail is fine; a wrong or
    contradicting value is incorrect.
    """

    question: str = dspy.InputField()
    expected: str = dspy.InputField()
    answer: str = dspy.InputField()
    is_abstention: bool = dspy.InputField()
    correct: bool = dspy.OutputField()


class DerivableSignature(dspy.Signature):
    """
    Decide whether the expected answer to the question could be derived from
    the given memory records alone. Judge information PRESENCE, not phrasing:
    derivable is True only if the records contain the facts needed to produce
    the expected answer (dates count as facts). Ignore how well-written the
    records are.
    """

    question: str = dspy.InputField()
    expected: str = dspy.InputField()
    records: list[str] = dspy.InputField()
    derivable: bool = dspy.OutputField()


_judge = dspy.Predict(LMEJudgeSignature)
_derivable = dspy.Predict(DerivableSignature)


AUDIT_CHUNK_CHARS = 40_000  # one judge call's worth of records


async def audit_derivable(question: str, expected: str, records: list[str]) -> bool:
    """True if the expected answer is derivable from these records.

    Haystack stores hold hundreds of records (~500k chars), far past a single
    call's context, so audit in chunks and take the OR — one chunk containing
    the evidence is enough. Without chunking every large store audits as False.
    """
    if not records:
        return False

    chunks, current, size = [], [], 0
    for rec in records:
        if current and size + len(rec) > AUDIT_CHUNK_CHARS:
            chunks.append(current)
            current, size = [], 0
        current.append(rec)
        size += len(rec)
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            with dspy.context(lm=chatbot._lm):
                out = _derivable(question=question, expected=expected, records=chunk)
            if bool(out.derivable):
                return True
        except Exception as e:
            console.print(f"  [dim red]audit chunk failed: {str(e)[:60]}[/dim red]")
    return False


def lme_date(raw: str) -> str:
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10'"""
    return raw.split(" ")[0].replace("/", "-")


async def run_question(idx: int, item: dict) -> dict:
    user_id = 5000 + idx
    sessions = list(zip(item["haystack_sessions"], item["haystack_dates"]))
    # oldest first: the dataset lists sessions out of chronological order for 42%
    # of haystack questions, and a real deployment sees conversations in time
    # order. Ingesting as-listed made newer facts look like the ones to supersede.
    sessions.sort(key=lambda session_and_date: session_and_date[1])
    if len(sessions) > 10:
        console.print(f"  [dim]ingesting {len(sessions)} sessions…[/dim]")

    async def ingest_one(j: int, session, sdate: str):
        messages = [{"role": t["role"], "content": t["content"]} for t in session]
        session_id = f"lme-{idx}-{j}"
        # archive raw exchanges (experience bank) so turn search can recall them
        pending_user = None
        for t in messages:
            if t["role"] == "user":
                pending_user = t["content"]
            elif t["role"] == "assistant" and pending_user is not None:
                archive_exchange(user_id, session_id, pending_user, t["content"], ts=lme_date(sdate))
                pending_user = None
        try:
            await update_memories(user_id, messages, session_id=session_id, current_date=lme_date(sdate))
        except Exception as e:
            console.print(f"  [dim red]ingest error session {j}: {e}[/dim red]")
        return session_id

    # ingest in batches: extraction runs concurrently (it is append-only, so order
    # does not matter), then one sleep pass reconciles the whole batch
    for start in range(0, len(sessions), SLEEP_EVERY):
        batch = sessions[start:start + SLEEP_EVERY]
        session_ids = await asyncio.gather(
            *(ingest_one(start + k, s, d) for k, (s, d) in enumerate(batch))
        )
        if len(sessions) > 10:
            console.print(f"  [dim]  …{min(start + len(batch), len(sessions))}/{len(sessions)} sessions[/dim]")
        try:
            # the profile rewrite is the most expensive stage and intermediate
            # versions are discarded — build it once, after the last batch
            is_last = start + len(batch) >= len(sessions)
            await sleep_pass(user_id, list(session_ids), refresh_core=is_last)
        except Exception as e:
            console.print(f"  [dim red]sleep pass error: {e}[/dim red]")

    question = item["question"]
    expected = str(item["answer"])
    is_abs = item["question_id"].endswith("_abs")

    # retrieval, exactly as the chatbot does it (memories + raw transcript turns)
    vec = (await generate_embeddings([question]))[0]
    retrieved = await search_memories(
        search_vector=vec, user_id=user_id, query_text=question, include_old=True
    )
    retrieved_strings = [stringify_retrieved_point(m) for m in retrieved]
    past_turns = [stringify_turn(t) for t in search_turns(user_id, question)]
    core = await get_core_memory(user_id)

    # stage audits (skip for abstention questions — the gold answer is absence)
    store_has, retrieval_has = None, None
    if not is_abs:
        all_recs = await fetch_all_user_records(user_id)
        store_strings = ([f"CORE PROFILE: {core}"] if core else []) + [
            ("[OLD] " if not r.is_current else "")
            + r.memory_text
            + (f" (context: {r.context})" if r.context else "")
            + f" [date: {r.date[:10]}]"
            for r in all_recs
        ] + [stringify_turn(l) for l in load_transcripts(user_id)]
        store_has = await audit_derivable(question, expected, store_strings)
        retrieval_has = await audit_derivable(
            question, expected,
            ([f"CORE PROFILE: {core}"] if core else []) + retrieved_strings + past_turns,
        )

    transcript = [
        {"role": "user", "content": f"(For reference, today's date is {lme_date(item['question_date'])}.)"},
        {"role": "assistant", "content": "Noted."},
    ]
    aggregate = await maybe_aggregate(user_id, question, current_date=lme_date(item["question_date"]))
    with dspy.context(lm=chatbot._lm):
        out = chatbot._responder(
            core_memory=core, transcript=transcript,
            retrieved_memories=retrieved_strings, past_conversations=past_turns,
            unverified_terms=unverified_terms(question, [core] + retrieved_strings + past_turns),
            computed_aggregate=aggregate,
            question=question,
        )
    answer = out.response

    try:
        with dspy.context(lm=chatbot._lm):
            verdict = _judge(question=question, expected=expected,
                             answer=answer, is_abstention=is_abs)
        correct = bool(verdict.correct)
    except Exception:
        correct = False

    if is_abs:
        stage = "n/a (abstention)"
    elif correct:
        stage = "correct"
    elif not store_has:
        stage = "write-path loss"
    elif not retrieval_has:
        stage = "retrieval loss"
    else:
        stage = "reasoning loss"

    mark = "[green]PASS[/green]" if correct else "[red]FAIL[/red]"
    console.print(f"  {mark} [{item['question_type']}{'/abs' if is_abs else ''}] "
                  f"store={store_has} retr={retrieval_has} | {question[:56]} | [dim]{answer[:56]}[/dim]")
    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "is_abstention": is_abs,
        "question": question,
        "expected": expected,
        "answer": answer,
        "correct": correct,
        "store_has_answer": store_has,
        "retrieval_has_answer": retrieval_has,
        "loss_stage": stage,
        "aggregate_used": bool(aggregate),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--types", default="", help="comma-separated question types to include")
    parser.add_argument("--ids", default="", help="comma-separated question_ids to run (overrides sampling)")
    parser.add_argument("--haystack", action="store_true",
                        help="use the _s haystack file (~40 sessions/question) instead of oracle")
    parser.add_argument("--sleep-every", type=int, default=1,
                        help="run the sleep pass every Nth session (cost control on haystacks)")
    parser.add_argument("--data", default="")
    parser.add_argument("--out", default="longmemeval.json",
                        help="filename (written to eval/results/) or an absolute path")
    args = parser.parse_args()

    global SLEEP_EVERY
    SLEEP_EVERY = max(1, args.sleep_every)

    data_path = args.data or (HAYSTACK_PATH if args.haystack else ORACLE_PATH)
    if not os.path.exists(data_path):
        console.print(f"[red]Dataset not found: {data_path}[/red]")
        console.print("[dim]Download with:  curl -sL -o eval/data/<file> "
                      "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/<file>[/dim]")
        return
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.ids:
        ids = set(filter(None, args.ids.split(",")))
        sample = [x for x in sorted(data, key=lambda x: x["question_id"]) if x["question_id"] in ids]
    else:
        wanted = set(filter(None, args.types.split(",")))
        by_type: dict = {}
        for item in sorted(data, key=lambda x: x["question_id"]):
            if wanted and item["question_type"] not in wanted:
                continue
            by_type.setdefault(item["question_type"], []).append(item)

        sample = []
        for qtype in sorted(by_type):
            sample.extend(by_type[qtype][:args.per_type])

    setting = "haystack (_s)" if args.haystack or "_s" in os.path.basename(data_path) else "oracle"
    console.print(f"[bold]LongMemEval {setting} — {len(sample)} questions[/bold] "
                  f"(sleep pass every {SLEEP_EVERY} session(s), workdir: {_workdir})\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(RESULTS_DIR, args.out)

    results = []
    for idx, item in enumerate(sample):
        console.print(f"[bold cyan]{idx + 1}/{len(sample)} {item['question_id']}[/bold cyan]")
        results.append(await run_question(idx, item))
        # save after every question — these runs are long and get interrupted
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Question type")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    groups: dict = {}
    for r in results:
        groups.setdefault(r["question_type"], []).append(r["correct"])
    abst = [r["correct"] for r in results if r["is_abstention"]]
    for qtype in sorted(groups):
        marks = groups[qtype]
        table.add_row(qtype, str(sum(marks)), str(len(marks)), f"{100 * sum(marks) / len(marks):.0f}%")
    if abst:
        table.add_row("(abstention subset)", str(sum(abst)), str(len(abst)),
                      f"{100 * sum(abst) / len(abst):.0f}%")
    total = [r["correct"] for r in results]
    if total:
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{sum(total)}[/bold]", f"[bold]{len(total)}[/bold]",
                      f"[bold]{100 * sum(total) / len(total):.0f}%[/bold]")
    console.print(table)

    # loss attribution: at which stage did each non-abstention question fail?
    audit = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
                  title="Loss attribution (non-abstention)")
    audit.add_column("Stage")
    audit.add_column("Questions", justify="right")
    non_abs = [r for r in results if not r["is_abstention"]]
    stages: dict = {}
    for r in non_abs:
        stages[r["loss_stage"]] = stages.get(r["loss_stage"], 0) + 1
    for stage in ("correct", "write-path loss", "retrieval loss", "reasoning loss"):
        if stage in stages:
            audit.add_row(stage, str(stages[stage]))
    console.print(audit)
    in_store = [r["store_has_answer"] for r in non_abs]
    in_retr = [r["retrieval_has_answer"] for r in non_abs]
    console.print(
        f"[dim]Answer present in store: {sum(in_store)}/{len(in_store)}  |  "
        f"survived retrieval: {sum(in_retr)}/{len(in_retr)}[/dim]"
    )

    console.print(f"[dim]Raw results written to {out_path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
