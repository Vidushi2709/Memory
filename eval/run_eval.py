"""
Memory eval harness.

Ingests scripted conversations through the real memory pipeline, asks the
scenario questions, and scores answers with an LLM judge. Runs against a
throwaway database in a temp dir — never touches ./chroma_db in the repo.

Usage (from the repo root):
    python eval/run_eval.py                 # full run
    python eval/run_eval.py --filter update # scenarios whose name contains "update"
    python eval/run_eval.py --compose       # with compose-on-read enabled
    python eval/run_eval.py --out results.json
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)

from dotenv import dotenv_values

os.environ.setdefault("OPEN_ROUTER_KEY", dotenv_values(os.path.join(REPO, ".env")).get("OPEN_ROUTER_KEY") or "")

# Isolate ./chroma_db and ./transcripts before the memory modules import
_workdir = tempfile.mkdtemp(prefix="memory_eval_")
os.chdir(_workdir)
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPT_DIR)

import dspy
from rich.console import Console
from rich.table import Table
from rich import box

import chatbot
from scenarios import SCENARIOS
from memory.aggregate import maybe_aggregate
from memory.consolidate import sleep_pass
from memory.embedding_generation import generate_embeddings
from memory import memory_store
from memory.memory_store import get_core_memory, search_memories, stringify_retrieved_point
from memory.grounding import unverified_terms
from memory.transcripts import search_turns, stringify_turn
from memory.update_memory import update_memories

console = Console()


class JudgeSignature(dspy.Signature):
    """
    Judge whether the assistant's answer is correct given the expected answer.

    If expected is "ABSTAIN": correct means the assistant admitted it doesn't
    know or has no such information, rather than guessing.
    Otherwise: correct means the answer contains the expected information —
    paraphrasing is fine, extra correct detail is fine, contradiction is not.
    """

    question: str = dspy.InputField()
    expected: str = dspy.InputField()
    answer: str = dspy.InputField()
    correct: bool = dspy.OutputField()


_judge = dspy.Predict(JudgeSignature)


async def ask(user_id: int, question: str) -> str:
    vec = (await generate_embeddings([question]))[0]
    retrieved = await search_memories(
        search_vector=vec, user_id=user_id, query_text=question, include_old=True
    )
    strings = [stringify_retrieved_point(m) for m in retrieved]
    past_turns = [stringify_turn(t) for t in search_turns(user_id, question)]
    core = await get_core_memory(user_id)
    aggregate = await maybe_aggregate(user_id, question)
    if chatbot.COMPOSE_ON_READ and strings:
        with dspy.context(lm=chatbot._lm):
            strings = [chatbot._composer(question=question, memories=strings).digest]
    with dspy.context(lm=chatbot._lm):
        out = chatbot._responder(
            core_memory=core, transcript=[], retrieved_memories=strings,
            past_conversations=past_turns,
            unverified_terms=unverified_terms(question, [core] + strings + past_turns),
            computed_aggregate=aggregate,
            question=question,
        )
    return out.response


async def run_scenario(idx: int, scenario: dict) -> list[dict]:
    user_id = 1000 + idx
    for j, session in enumerate(scenario["sessions"]):
        session_id = f"eval-{idx}-{j}"
        await update_memories(user_id, session, session_id=session_id)
        await sleep_pass(user_id, session_id)

    results = []
    for item in scenario["questions"]:
        answer = await ask(user_id, item["q"])
        try:
            with dspy.context(lm=chatbot._lm):
                verdict = _judge(question=item["q"], expected=item["expect"], answer=answer)
            correct = bool(verdict.correct)
        except Exception:
            correct = False
        results.append({
            "scenario": scenario["name"],
            "category": item["category"],
            "question": item["q"],
            "expected": item["expect"],
            "answer": answer,
            "correct": correct,
        })
        mark = "[green]PASS[/green]" if correct else "[red]FAIL[/red]"
        console.print(f"  {mark} [{item['category']}] {item['q']} | [dim]{answer[:90]}[/dim]")
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default="", help="only scenarios whose name contains this")
    parser.add_argument("--compose", action="store_true", help="enable compose-on-read")
    parser.add_argument("--no-ppr", action="store_true", help="disable PageRank ranking")
    parser.add_argument("--out", default="", help="filename (written to eval/results/) or an absolute path")
    args = parser.parse_args()

    chatbot.COMPOSE_ON_READ = args.compose
    memory_store.USE_PPR = not args.no_ppr

    scenarios = [s for s in SCENARIOS if args.filter in s["name"]]
    console.print(
        f"[bold]Running {len(scenarios)} scenario(s)[/bold] "
        f"(compose-on-read: {'ON' if args.compose else 'off'}, "
        f"PPR: {'off' if args.no_ppr else 'ON'}, workdir: {_workdir})\n"
    )

    all_results = []
    for idx, scenario in enumerate(scenarios):
        console.print(f"[bold cyan]{scenario['name']}[/bold cyan]")
        all_results.extend(await run_scenario(idx, scenario))
        console.print()

    by_category: dict = {}
    for r in all_results:
        by_category.setdefault(r["category"], []).append(r["correct"])

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Category")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    for cat in sorted(by_category):
        marks = by_category[cat]
        table.add_row(cat, str(sum(marks)), str(len(marks)), f"{100 * sum(marks) / len(marks):.0f}%")
    total = [r["correct"] for r in all_results]
    if total:
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{sum(total)}[/bold]",
                      f"[bold]{len(total)}[/bold]",
                      f"[bold]{100 * sum(total) / len(total):.0f}%[/bold]")
    console.print(table)

    if args.out:
        results_dir = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        out_path = args.out if os.path.isabs(args.out) else os.path.join(results_dir, args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]Raw results written to {out_path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
