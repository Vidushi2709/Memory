"""Does a smaller memory store answer better than a bigger one?

Every memory system I looked at stores whatever arrives and tries to be clever
at retrieval time. None of them decides at write time that a fact is not worth
keeping. The reason is not that a write-time gate is hard to build — it is that
nobody can measure whether it helps: LongMemEval and LoCoMo fix the corpus as
part of the benchmark, so you are not allowed to vary store size and still be
running the benchmark.

This varies it. Each pass re-ingests the same questions with a different
admission threshold (the write path refuses facts the extractor scored below
it), and run_longmemeval's loss attribution says which stage each failure came
from. The two failure types move in opposite directions:

    raising the threshold  -> more WRITE-PATH loss  (the answer was never stored)
                           -> less RETRIEVAL loss   (fewer junk facts competing
                                                     for the five search slots)

If total accuracy peaks at a threshold above 0, "store everything" is the wrong
default and there is an optimum below it. If accuracy is highest at 0, the
field's default is right and the idea is dead — which is also a result.

    python eval/sweep_admission.py --thresholds 0,4,6,8 --per-type 3
    python eval/sweep_admission.py --thresholds 0,6 --types knowledge-update

Each threshold is a full eval run, so a sweep costs N times one run. Results are
cached per threshold in eval/results/; re-run with --force to redo them.
"""
import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RUNNER = os.path.join(SCRIPT_DIR, "run_longmemeval.py")

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def summarise(rows: list[dict]) -> dict:
    """One threshold's run -> the numbers that go in a row of the table."""
    non_abs = [r for r in rows if not r["is_abstention"]]
    # .get: results from before store_size was recorded still summarise
    sizes = [r["store_size"] for r in non_abs if r.get("store_size") is not None]
    stages = {}
    for r in non_abs:
        stage = r.get("loss_stage", "unattributed")
        stages[stage] = stages.get(stage, 0) + 1
    correct = [r["correct"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": 100 * sum(correct) / len(correct) if correct else 0.0,
        "mean_store": sum(sizes) / len(sizes) if sizes else 0.0,
        "write_path_loss": stages.get("write-path loss", 0),
        "retrieval_loss": stages.get("retrieval loss", 0),
        "reasoning_loss": stages.get("reasoning loss", 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thresholds", default="0,4,6,8",
                   help="comma-separated importance floors to compare (0 = store everything)")
    p.add_argument("--per-type", type=int, default=3)
    p.add_argument("--types", default="")
    p.add_argument("--haystack", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run thresholds already on disk")
    args = p.parse_args()

    thresholds = [int(t) for t in args.thresholds.split(",") if t.strip()]
    console.print(f"[bold]Admission sweep[/bold] over thresholds {thresholds} "
                  f"({len(thresholds)} full eval run(s))\n")

    # the cache key has to carry the sample too, or a --per-type 5 run silently
    # reuses a 12-question file for one threshold and compares it against 30
    tag = f"p{args.per_type}"
    if args.haystack:
        tag += "-haystack"
    if args.types:
        tag += "-" + re.sub(r"[^a-z0-9]+", "", args.types.lower())[:14]

    summaries = {}
    for t in thresholds:
        out_name = f"sweep-min{t}-{tag}.json"
        out_path = os.path.join(RESULTS_DIR, out_name)
        if args.force or not os.path.exists(out_path):
            cmd = [sys.executable, RUNNER, "--min-importance", str(t),
                   "--per-type", str(args.per_type), "--out", out_name]
            if args.types:
                cmd += ["--types", args.types]
            if args.haystack:
                cmd.append("--haystack")
            console.print(f"[cyan]threshold {t}:[/cyan] {' '.join(cmd[1:])}")
            if subprocess.run(cmd).returncode != 0:
                console.print(f"[red]threshold {t} failed, skipping[/red]")
                continue
        else:
            console.print(f"[dim]threshold {t}: reusing {out_name} (--force to redo)[/dim]")
        with open(out_path, encoding="utf-8") as f:
            summaries[t] = summarise(json.load(f))

    if not summaries:
        return

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan",
                  title="Accuracy vs store size")
    table.add_column("Min importance")
    table.add_column("Facts stored", justify="right")
    table.add_column("Write-path loss", justify="right")
    table.add_column("Retrieval loss", justify="right")
    table.add_column("Reasoning loss", justify="right")
    table.add_column("Accuracy", justify="right")
    # ties go to the higher threshold: at equal accuracy the smaller store is
    # strictly better, since it costs less to hold and less to search
    best = max(summaries, key=lambda t: (summaries[t]["accuracy"], t))
    for t in sorted(summaries):
        s = summaries[t]
        acc = f"{s['accuracy']:.0f}%"
        table.add_row(str(t), f"{s['mean_store']:.1f}", str(s["write_path_loss"]),
                      str(s["retrieval_loss"]), str(s["reasoning_loss"]),
                      f"[bold green]{acc}[/bold green]" if t == best else acc)
    console.print(table)

    base = min(summaries)
    top_acc = summaries[best]["accuracy"]
    if best == base:
        console.print("[dim]Best accuracy at the lowest threshold: on this sample "
                      "storing everything wins and admission control does not pay.[/dim]")
    elif abs(top_acc - summaries[base]["accuracy"]) < 1e-9:
        console.print(f"[bold]Tie at {top_acc:.0f}%: threshold {best} stores "
                      f"{summaries[best]['mean_store']:.1f} facts against "
                      f"{summaries[base]['mean_store']:.1f} for the same accuracy.[/bold] "
                      "Not evidence of a peak, but evidence the store can shrink for free — "
                      "check whether retrieval loss fell as write-path loss rose.")
    else:
        console.print(f"[bold]Best accuracy at threshold {best}, storing "
                      f"{summaries[best]['mean_store']:.1f} facts vs "
                      f"{summaries[base]['mean_store']:.1f} at threshold {base}.[/bold] "
                      "Accuracy is not monotonic in store size.")


if __name__ == "__main__":
    main()
