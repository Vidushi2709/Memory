"""
chatbot.py — Real-time memory-enabled chatbot.

Features:
  - Persistent memory via ChromaDB (./chroma_db)
  - Background memory writes  → no delay after responses
  - Proactive recall at session start → feels magical
  - Session summary on /quit  → conversations are never lost

Run:
    python chatbot.py
"""

import asyncio
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import dspy
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from memory.aggregate import maybe_aggregate
from memory.consolidate import sleep_pass
from memory.embedding_generation import generate_embeddings
from memory.grounding import unverified_terms
from memory.llm import get_chat_lm
from memory.transcripts import archive_exchange, search_turns, stringify_turn
from memory.memory_store import (
    add_memory,
    create_collection,
    delete_user_records,
    fetch_all_user_records,
    get_all_categories,
    get_core_memory,
    search_memories,
    stringify_retrieved_point,
    EmbeddedMemory,
)
from memory.update_memory import update_memories

# LLM setup 

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

_lm = get_chat_lm()


# DSPy Signatures 

class ChatSignature(dspy.Signature):
    """
    You are a friendly AI assistant with long-term memory about the user.
    Use retrieved memories naturally in your replies — don't list or recite them.
    Keep responses concise and conversational. Avoid unnecessary preamble.

    Act on what you already know instead of asking for it again. When the user
    asks for a suggestion or recommendation, give a concrete one immediately,
    filtered by everything you know about them (diet, allergies, budget, tastes).
    Do NOT ask what they are in the mood for or ask them to restate a preference
    you have stored — answer first; ask only if the answer is impossible without
    more information.

    core_memory is a trusted always-current profile of the user. Retrieved
    memories can be marked [OLD/SUPERSEDED] for past states. Use these to answer
    historical questions ("where did I live before?") while using current memories
    for present-state questions. Be clear about what's current vs. past when relevant.

    past_conversations holds VERBATIM excerpts of earlier chats — what was
    actually typed, by both of you. Treat them as an exact record: when asked
    what was said, recommended, or listed before, read the answer straight out
    of them and quote it. Never claim you lack information that appears there.

    GUARDED RECALL: first fill in supporting_evidence by quoting the exact
    words from a memory or past conversation that state what was asked. If
    nothing states it, write NONE — a source about a similar-but-different
    thing (another role title, person, pet, or item) does NOT count. When
    supporting_evidence is NONE, the response must say you don't have that
    information, optionally naming the near-match; never answer from it.

    Memories tagged [PLANNED, did not happen] or [CONSIDERED, did not happen]
    record intentions, not events. Never count or describe them as things the
    user did; say they were planned or considered.

    unverified_terms lists exact phrases from the question that appear in NO
    source. These were checked mechanically, so trust them over your own
    impression: the user is asking about something you have no record of. Say so
    plainly, name the closest thing you do have, and do not answer as if the
    unverified phrase were established.

    computed_aggregate, when non-empty, holds a count or time span computed IN
    CODE over a full scan of every stored memory — not a search, so nothing was
    missed. Its arithmetic is reliable: state its number as the answer and use
    its member list to explain. Do not re-derive, hedge, or offer alternatives
    to a number it provides.
    """
    core_memory: str = dspy.InputField(desc="Short standing profile of the user (may be empty for new users)")
    transcript: list[dict] = dspy.InputField(desc="Recent conversation turns (last ~10 messages)")
    retrieved_memories: list[str] = dspy.InputField(desc="Relevant past memories about this user (may include old/superseded ones)")
    past_conversations: list[str] = dspy.InputField(desc="Verbatim excerpts of earlier conversations — an exact record of what was said")
    unverified_terms: list[str] = dspy.InputField(desc="Phrases from the question found in no source (mechanically checked). Empty is normal.")
    computed_aggregate: str = dspy.InputField(desc="Count or time span computed in code from a full memory scan. Empty unless the question asks for one. When present, its number IS the answer basis.")
    question: str = dspy.InputField(desc="The user's latest message")
    supporting_evidence: str = dspy.OutputField(
        desc="Exact words from a source that state what was asked, or NONE. Written before the response."
    )
    response: str = dspy.OutputField(desc="Your reply to the user")
    save_memory: bool = dspy.OutputField(
        description="True if the user just shared something worth remembering"
    )


class SessionSummarySignature(dspy.Signature):
    """
    You are given a full conversation transcript from a single chat session.
    Write a concise 1-3 sentence summary of what was discussed or learned
    about the user during this session. Focus only on facts about the USER,
    not the AI's responses. This will be stored as a memory.
    If nothing meaningful was shared, output an empty string.
    """
    transcript: list[dict] = dspy.InputField(desc="Full conversation transcript for this session")
    summary: str = dspy.OutputField(
        desc="1-3 sentence summary of what was learned about the user. Empty string if nothing notable."
    )


COMPOSE_ON_READ = False  # experiment: replace the raw memory list with a query-tailored digest


class ComposeMemorySignature(dspy.Signature):
    """
    Compose a brief memory digest tailored to the user's question. From the
    retrieved memories, write 1-3 sentences containing only the information
    relevant to answering the question, keeping current vs. past state clear
    (memories marked [OLD/SUPERSEDED] are past). If nothing is relevant, output
    "No relevant memories." Use ONLY the retrieved memories — never invent.
    """

    question: str = dspy.InputField()
    memories: list[str] = dspy.InputField()
    digest: str = dspy.OutputField()


_responder = dspy.Predict(ChatSignature)
_summariser = dspy.Predict(SessionSummarySignature)
_composer = dspy.Predict(ComposeMemorySignature)

# Console

console = Console()

BANNER = """
███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝ 
██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝  
██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║   
╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝  
"""

HELP_TEXT = (
    "[bold cyan]/memories[/bold cyan]    — show all stored memories\n"
    "[bold cyan]/sessions[/bold cyan]    — list past sessions with their summaries\n"
    "[bold cyan]/categories[/bold cyan]  — list all memory categories\n"
    "[bold cyan]/forget[/bold cyan]      — delete ALL your memories (irreversible)\n"
    "[bold cyan]/help[/bold cyan]        — show this help\n"
    "[bold cyan]/quit[/bold cyan]        — save summary, consolidate memories & exit\n"
)


def print_banner():
    console.print(BANNER, style="bold magenta", highlight=False)
    console.print(
        Panel(
            "[bold white]Your personal AI with persistent memory[/bold white]\n"
            "[dim]Memories survive across sessions — I remember you![/dim]",
            border_style="magenta",
            padding=(0, 2),
        )
    )
    console.print()


def show_help():
    console.print(Panel(HELP_TEXT, title="[bold]Commands[/bold]", border_style="cyan", padding=(0, 2)))


async def show_memories(user_id: int):
    core = await get_core_memory(user_id)
    if core:
        console.print(Panel(core, title="[bold]Core Memory[/bold]", border_style="yellow", padding=(0, 2)))
    records = await fetch_all_user_records(user_id=user_id)
    if not records:
        console.print("[dim]No memories stored yet.[/dim]")
        return
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Memory", style="white")
    table.add_column("Categories", style="magenta")
    table.add_column("Status", width=10)
    table.add_column("Saved At", style="dim", width=20)
    current_count = 0
    for i, r in enumerate(records, 1):
        if r.is_current:
            status = "[bold green]Current[/bold green]"
            current_count += 1
        else:
            status = "[dim]Old[/dim]"
        table.add_row(
            str(i),
            r.memory_text,
            ", ".join(r.categories),
            status,
            r.date[:19].replace("T", " "),
        )
    console.print(table)
    old_count = len(records) - current_count
    console.print(
        f"[dim]  {current_count} current  |  {old_count} old/superseded  |  {len(records)} total.[/dim]"
    )


async def show_sessions(user_id: int):
    records = await fetch_all_user_records(user_id=user_id)
    if not records:
        console.print("[dim]No memories stored yet.[/dim]")
        return
    sessions: dict[str, list] = {}
    for r in records:
        sessions.setdefault(r.session_id or "(untagged)", []).append(r)
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("Session", style="magenta", width=18)
    table.add_column("Started", style="dim", width=12)
    table.add_column("Memories", style="dim", width=8)
    table.add_column("Summary", style="white")
    for sid, group in sorted(sessions.items(), reverse=True):
        summary = next((r.memory_text for r in group if r.kind == "summary"), "")
        if not summary:
            summary = group[0].memory_text
        if len(summary) > 70:
            summary = summary[:70] + "…"
        table.add_row(sid, min(r.date for r in group)[:10], str(len(group)), summary)
    console.print(table)


async def show_categories(user_id: int):
    cats = await get_all_categories(user_id=user_id)
    if not cats:
        console.print("[dim]No categories yet.[/dim]")
        return
    console.print(
        "[bold cyan]Categories:[/bold cyan] "
        + "  ".join(f"[magenta]{c}[/magenta]" for c in cats)
    )


async def handle_forget(user_id: int):
    confirm = console.input(
        "[bold red]Are you sure you want to delete ALL your memories? "
        "Type [bold white]yes[/bold white] to confirm: [/bold red]"
    ).strip().lower()
    if confirm == "yes":
        await delete_user_records(user_id=user_id)
        console.print("[bold red]All memories deleted.[/bold red]")
    else:
        console.print("[dim]Cancelled.[/dim]")

_pending_memory_tasks: set[asyncio.Task] = set()


def fire_and_forget_memory(user_id: int, messages: list[dict], session_id: str):
    """
    Schedule update_memories as a background task so the chat loop
    returns the AI response immediately without waiting for the
    (potentially slow) LLM memory-update agent.
    """
    async def _run():
        try:
            summary = await update_memories(user_id=user_id, messages=messages, session_id=session_id)
            console.print(f"\n[dim]  ✦ Memory: {summary}[/dim]")
        except Exception as e:
            console.print(f"\n[dim red]  Memory update failed: {e}[/dim red]")

    task = asyncio.create_task(_run())
    _pending_memory_tasks.add(task)
    task.add_done_callback(_pending_memory_tasks.discard)

async def proactive_recall(user_id: int):
    """
    Surface the core profile and most recent memories as a warm greeting so
    the user immediately feels that the AI remembers them.
    """
    core = await get_core_memory(user_id)
    records = await fetch_all_user_records(user_id=user_id)
    if not core and not records:
        return  # new user — nothing to recall

    # Sort by date descending and take the 5 most recent
    sorted_records = sorted(records, key=lambda r: r.date, reverse=True)
    recent = sorted_records[:5]

    parts = []
    if core:
        parts.append(f"[white]{core}[/white]")
    if recent:
        bullets = "\n".join(f"  • {r.memory_text}" for r in recent)
        parts.append(f"[dim]Recent memories:[/dim]\n[white]{bullets}[/white]")
    console.print(
        Panel(
            "[dim]Here's what I remember about you:[/dim]\n\n" + "\n\n".join(parts),
            title="[bold yellow]✦ From Memory[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()


async def save_session_summary(user_id: int, past_messages: list[dict], session_id: str):
    """
    Ask the LLM to summarise the session, then store it as a memory
    so the AI knows what was discussed even across restarts.
    """
    if len(past_messages) < 2:
        return  # nothing was said

    with console.status("[dim]Saving session summary…[/dim]", spinner="dots"):
        try:
            with dspy.context(lm=_lm):
                out = _summariser(transcript=past_messages)

            summary_text: str = out.summary.strip()
            if not summary_text:
                console.print("[dim]  Nothing notable to summarise this session.[/dim]")
                return

            # Embed and store the summary as a regular memory
            embeddings = await generate_embeddings([summary_text])
            await add_memory(
                embedded_memories=[
                    EmbeddedMemory(
                        id="",
                        user_id=user_id,
                        memory_text=f"[Session {datetime.now().strftime('%Y-%m-%d')}] {summary_text}",
                        categories=["session_summary"],
                        embedding=embeddings[0],
                        date=datetime.now().isoformat(),
                        session_id=session_id,
                        kind="summary",
                    )
                ]
            )
            console.print(
                Panel(
                    f"[white]{summary_text}[/white]",
                    title="[bold yellow]✦ Session Saved to Memory[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )
        except Exception as e:
            console.print(f"[dim red]  Could not save session summary: {e}[/dim red]")


async def finish_session(user_id: int, past_messages: list[dict], session_id: str):
    """
    End-of-session housekeeping: wait for in-flight memory writes, save the
    session summary, then run the dedup/consolidation pass over memories.
    """
    if _pending_memory_tasks:
        console.print("[dim]  Waiting for background memory writes to finish…[/dim]")
        await asyncio.gather(*_pending_memory_tasks, return_exceptions=True)

    await save_session_summary(user_id, past_messages, session_id)

    with console.status("[dim]Consolidating memories (sleep pass)…[/dim]", spinner="dots"):
        try:
            result = await sleep_pass(user_id, session_id)
            console.print(f"[dim]  ✦ Sleep pass: {result}.[/dim]")
        except Exception as e:
            console.print(f"[dim red]  Sleep pass failed: {e}[/dim red]")


# Core chat loop 

async def chat_loop(user_id: int):
    past_messages: list[dict] = []
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    console.print(Rule(style="magenta"))
    console.print(
        f"[bold green]Session started for user [bold white]{user_id}[/bold white].[/bold green]  "
        "[dim]Type [bold]/help[/bold] for commands or just start chatting![/dim]"
    )
    console.print(Rule(style="magenta"))
    console.print()

    await proactive_recall(user_id=user_id)

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Interrupted.[/dim]")
            # Still save the summary and consolidate on Ctrl+C
            await finish_session(user_id, past_messages, session_id)
            break

        if not user_input:
            continue

        # slash commands
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            await finish_session(user_id, past_messages, session_id)
            console.print("[dim]Goodbye! Your memories are saved.[/dim]")
            break

        if cmd == "/help":
            show_help()
            continue
        if cmd == "/memories":
            await show_memories(user_id)
            continue
        if cmd == "/sessions":
            await show_sessions(user_id)
            continue
        if cmd == "/categories":
            await show_categories(user_id)
            continue
        if cmd == "/forget":
            await handle_forget(user_id)
            continue

        # retrieve relevant memories — old ones included, tagged [OLD/SUPERSEDED]
        with console.status("[dim]Thinking…[/dim]", spinner="dots"):
            search_vec = (await generate_embeddings([user_input]))[0]
            retrieved = await search_memories(
                search_vector=search_vec,
                user_id=user_id,
                query_text=user_input,
                include_old=True,
            )
            retrieved_strings = [stringify_retrieved_point(m) for m in retrieved]
            # raw-transcript recall: what was actually said, incl. assistant answers
            past_turns = [stringify_turn(t) for t in search_turns(user_id, user_input)]
            core = await get_core_memory(user_id)
            aggregate = await maybe_aggregate(user_id, user_input)

            if COMPOSE_ON_READ and retrieved_strings:
                with dspy.context(lm=_lm):
                    composed = _composer(question=user_input, memories=retrieved_strings)
                retrieved_strings = [composed.digest]

            # generate response
            with dspy.context(lm=_lm):
                out = _responder(
                    core_memory=core,
                    transcript=past_messages[-10:],
                    retrieved_memories=retrieved_strings,
                    past_conversations=past_turns,
                    unverified_terms=unverified_terms(user_input, [core] + retrieved_strings + past_turns),
                    computed_aggregate=aggregate,
                    question=user_input,
                )

        response: str = out.response
        save: bool = out.save_memory

        # update conversation history
        past_messages.extend([
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": response},
        ])
        # raw experience bank — enables re-extraction when the pipeline improves
        archive_exchange(user_id, session_id, user_input, response)

        if save:
            fire_and_forget_memory(user_id=user_id, messages=list(past_messages[-6:]), session_id=session_id)

        # print response
        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold green]AI[/bold green]",
            border_style="green",
            padding=(0, 2),
        ))

        console.print()


# Session setup 

async def pick_user() -> int:
    """Ask for (or create) a user profile."""
    console.print(Panel(
        "[bold white]Who are you?[/bold white]\n\n"
        "Enter a numeric [bold cyan]user ID[/bold cyan] to load your profile, "
        "or just press [bold]Enter[/bold] to use the default (ID = 1).",
        border_style="cyan",
        padding=(0, 2),
    ))
    raw = console.input("[bold cyan]User ID (default 1):[/bold cyan] ").strip()
    try:
        uid = int(raw) if raw else 1
    except ValueError:
        console.print("[yellow]Invalid ID — defaulting to 1.[/yellow]")
        uid = 1

    records = await fetch_all_user_records(user_id=uid)
    if records:
        console.print(
            f"[bold green]Welcome back![/bold green] "
            f"[dim]I have [bold]{len(records)}[/bold] memory/memories stored for you.[/dim]"
        )
    else:
        console.print(
            "[bold green]Welcome![/bold green] "
            "[dim]No memories yet — I'll start learning about you as we chat![/dim]"
        )
    return uid


async def main():
    print_banner()
    await create_collection()
    user_id = await pick_user()
    console.print()
    await chat_loop(user_id=user_id)


if __name__ == "__main__":
    asyncio.run(main())