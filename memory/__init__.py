"""Memory package. MEMORY_DIR is where chroma_db/ and transcripts/ live —
anchored to the repo, not the working directory, and overridable via env
(the evals point it at a temp dir)."""
import os

MEMORY_DIR = os.getenv("MEMORY_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
