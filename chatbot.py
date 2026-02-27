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

from memory.embedding_generation import generate_embeddings
from memory.extract_memory import extract_memory
from memory.memory_store import (
    add_memory,
    create_collection,
    delete_user_records,
    fetch_all_user_records,
    get_all_categories,
    search_memories,
    stringify_retrieved_point,
    EmbeddedMemory,
)
from memory.update_memory import update_memories

# LLM setup 

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

_lm = dspy.LM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.7,
    max_tokens=1024,
)


# DSPy Signatures 

class ChatSignature(dspy.Signature):
    """
    You are a helpful, friendly AI assistant with long-term memory.
    You are given the recent conversation transcript and relevant memories
    retrieved from the user's personal memory store.

    Use the retrieved memories naturally — don't just recite them.
    If the memories are empty or irrelevant, just respond normally.
    Be warm, concise, and conversational.

    IMPORTANT — Memory versioning:
    Memories can be marked [OLD/SUPERSEDED] when they were true in the past
    but have since been replaced by newer information. When answering a
    question like "where did I live before?" or "what was my old job?",
    look for [OLD/SUPERSEDED] memories to answer the historical part, and
    use the most recent (non-old) memory for the current state.
    Always make it clear to the user which information is current vs. past.
    """
    transcript: list[dict] = dspy.InputField(desc="Recent conversation turns (last ~10 messages)")
    retrieved_memories: list[str] = dspy.InputField(desc="Relevant past memories about this user (may include old/superseded ones)")
    question: str = dspy.InputField(desc="The user's latest message")
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


_responder = dspy.Predict(ChatSignature)
_summariser = dspy.Predict(SessionSummarySignature)

# Historical-query detection keywords
_HISTORICAL_KEYWORDS = [
    "before", "previously", "used to", "old", "past", "prior", "earlier",
    "last time", "back then", "formerly", "previous", "history", "what was",
    "where did i", "who did i", "when did i", "what did i",
]


def _is_historical_query(text: str) -> bool:
    """Return True if the user's message looks like a question about their past."""
    lower = text.lower()
    return any(kw in lower for kw in _HISTORICAL_KEYWORDS)

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
    "[bold cyan]/categories[/bold cyan]  — list all memory categories\n"
    "[bold cyan]/forget[/bold cyan]      — delete ALL your memories (irreversible)\n"
    "[bold cyan]/help[/bold cyan]        — show this help\n"
    "[bold cyan]/quit[/bold cyan]        — save session summary & exit\n"
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


def fire_and_forget_memory(user_id: int, messages: list[dict]):
    """
    Schedule update_memories as a background task so the chat loop
    returns the AI response immediately without waiting for the
    (potentially slow) LLM memory-update agent.
    """
    async def _run():
        try:
            summary = await update_memories(user_id=user_id, messages=messages)
            # Print softly so it doesn't interrupt the current prompt
            console.print(f"\n[dim]  ✦ Memory updated in background: {summary}[/dim]")
        except Exception as e:
            console.print(f"\n[dim red]  Memory update failed: {e}[/dim red]")

    task = asyncio.create_task(_run())
    _pending_memory_tasks.add(task)
    task.add_done_callback(_pending_memory_tasks.discard)

async def proactive_recall(user_id: int):
    """
    Fetch the most recent memories and surface them as a warm greeting so
    the user immediately feels that the AI remembers them.
    """
    records = await fetch_all_user_records(user_id=user_id)
    if not records:
        return  # new user — nothing to recall

    # Sort by date descending and take the 5 most recent
    sorted_records = sorted(records, key=lambda r: r.date, reverse=True)
    recent = sorted_records[:5]

    bullets = "\n".join(f"  • {r.memory_text}" for r in recent)
    console.print(
        Panel(
            f"[dim]Here's what I remember about you:[/dim]\n\n[white]{bullets}[/white]",
            title="[bold yellow]✦ From Memory[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()


async def save_session_summary(user_id: int, past_messages: list[dict]):
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


# Core chat loop 

async def chat_loop(user_id: int):
    past_messages: list[dict] = []

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
            # Still save the session summary on Ctrl+C
            await save_session_summary(user_id, past_messages)
            break

        if not user_input:
            continue

        # slash commands
        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            await save_session_summary(user_id, past_messages)
            # Wait for any in-flight background memory tasks to finish
            if _pending_memory_tasks:
                console.print("[dim]  Waiting for background memory writes to finish…[/dim]")
                await asyncio.gather(*_pending_memory_tasks, return_exceptions=True)
            console.print("[dim]Goodbye! Your memories are saved.[/dim]")
            break

        if cmd == "/help":
            show_help()
            continue
        if cmd == "/memories":
            await show_memories(user_id)
            continue
        if cmd == "/categories":
            await show_categories(user_id)
            continue
        if cmd == "/forget":
            await handle_forget(user_id)
            continue

        # retrieve relevant memories
        # For historical questions, also include old/superseded memories
        historical = _is_historical_query(user_input)
        with console.status("[dim]Thinking…[/dim]", spinner="dots"):
            search_vec = (await generate_embeddings([user_input]))[0]
            retrieved = await search_memories(
                search_vector=search_vec,
                user_id=user_id,
                include_old=historical,
            )
            retrieved_strings = [stringify_retrieved_point(m) for m in retrieved]

            # generate response
            with dspy.context(lm=_lm):
                out = _responder(
                    transcript=past_messages[-10:],
                    retrieved_memories=retrieved_strings,
                    question=user_input,
                )

        response: str = out.response
        save: bool = out.save_memory

        # update conversation history
        past_messages.extend([
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": response},
        ])

        if save:
            fire_and_forget_memory(user_id=user_id, messages=list(past_messages[-6:]))

        # print response
        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold green]AI[/bold green]",
            border_style="green",
            padding=(0, 2),
        ))

        if retrieved_strings:
            console.print(
                "[dim]  Recalled: "
                + " | ".join(r.split("(")[0].strip() for r in retrieved_strings[:3])
                + "[/dim]"
            )

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