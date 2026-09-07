"""Memory package. MEMORY_DIR is where chroma_db/ and transcripts/ live —
anchored to the repo, not the working directory, and overridable via env
(the evals point it at a temp dir)."""
import os

MEMORY_DIR = os.getenv("MEMORY_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Personal mode: this is one person's own memory, not a benchmark run.
#
# The benchmark defaults are tuned for LongMemEval, which rewards refusing to
# answer when the evidence is not literally present. That behaviour is wrong
# for a personal assistant: it makes ordinary questions ("should I visit New
# York?") come back as "I have no record of New York". Turning this on relaxes
# the abstention machinery and stops the store filling with LLM generalisations.
#
# Set MEMORY_PERSONAL_MODE=0 for eval runs (the harnesses do this themselves).
PERSONAL_MODE = os.getenv("MEMORY_PERSONAL_MODE", "1") != "0"
