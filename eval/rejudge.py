"""Re-score finished eval runs with the current judge, without re-ingesting.

The judge that produced the runs in eval/results/ counted a refusal as a correct
answer: it was told a contradicting value is wrong, but never that an answer
omitting the expected value is wrong, so "I don't have any information about
that" passed against an expected answer of "$12". That inflates every arm, and
inflates the weakest arms most, because a system with less stored refuses more.

Every run already saves the question, the gold answer and what the assistant
replied, so re-scoring needs no ingest and no retrieval — just the judge. Loss
stages are recomputed too, since a question that flips to wrong has to be
attributed to a stage.

    python eval/rejudge.py                       # every sweep-*.json, in place
    python eval/rejudge.py --glob "sweep-min0*"  # a subset
    python eval/rejudge.py --dry-run             # report the deltas, write nothing

Originals are copied to <name>.prejudge.json before anything is overwritten.
"""
import argparse
import asyncio
import glob
import json
import os
import shutil
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

from dotenv import dotenv_values

os.environ.setdefault("OPEN_ROUTER_KEY",
                      dotenv_values(os.path.join(REPO, ".env")).get("OPEN_ROUTER_KEY") or "")
os.environ["MEMORY_DIR"] = tempfile.mkdtemp(prefix="rejudge_")
os.environ["MEMORY_PERSONAL_MODE"] = "0"
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPT_DIR)

import dspy
from rich import box
from rich.console import Console
from rich.table import Table

from memory import answer
from run_longmemeval import _judge

console = Console()


def restage(r: dict) -> str:
    """Same attribution the runner uses, recomputed for the new verdict."""
    if r["is_abstention"]:
        return "n/a (abstention)"
    if r["correct"]:
        return "correct"
    if not r.get("store_has_answer"):
        return "write-path loss"
    if not r.get("retrieval_has_answer"):
        return "retrieval loss"
    return "reasoning loss"


async def rejudge_file(path: str, dry_run: bool) -> dict:
    rows = json.load(open(path, encoding="utf-8"))

    async def one(r):
        try:
            with dspy.context(lm=answer._lm):
                return bool(await asyncio.to_thread(
                    lambda: _judge(question=r["question"], expected=r["expected"],
                                   answer=r["answer"], is_abstention=r["is_abstention"]).correct
                ))
        except Exception:
            return r["correct"]          # keep the old verdict rather than invent one

    verdicts = await asyncio.gather(*(one(r) for r in rows))

    flipped_down = flipped_up = 0
    for r, v in zip(rows, verdicts):
        if r["correct"] and not v:
            flipped_down += 1
        elif v and not r["correct"]:
            flipped_up += 1
        r["correct"] = v
        r["loss_stage"] = restage(r)

    after = sum(verdicts)
    if not dry_run:
        shutil.copyfile(path, path.replace(".json", ".prejudge.json"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    return {"n": len(rows), "after": after, "down": flipped_down, "up": flipped_up}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="sweep-*.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    paths = [f for f in sorted(glob.glob(os.path.join(RESULTS_DIR, args.glob)))
             if ".prejudge." not in f]
    if not paths:
        console.print("[red]No result files matched.[/red]")
        return

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
                  title="Re-scored with the corrected judge")
    for col, just in (("Run", "left"), ("Was", "right"), ("Now", "right"),
                      ("Pass to fail", "right"), ("Fail to pass", "right")):
        table.add_column(col, justify=just)

    for path in paths:
        rows = json.load(open(path, encoding="utf-8"))
        was = sum(r["correct"] for r in rows)
        out = await rejudge_file(path, args.dry_run)
        name = os.path.basename(path).replace(".json", "")
        table.add_row(name, f"{100 * was / out['n']:.0f}%",
                      f"{100 * out['after'] / out['n']:.0f}%",
                      str(out["down"]), str(out["up"]))
        console.print(f"[dim]{name}: {out['n']} questions judged[/dim]")

    console.print(table)
    if args.dry_run:
        console.print("[dim]--dry-run: nothing written[/dim]")
    else:
        console.print("[dim]Originals kept as *.prejudge.json[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
