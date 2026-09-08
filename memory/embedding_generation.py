"""Text -> 384-d vectors via ONNX, no torch.

Measured against the sentence-transformers/torch path this replaced: identical
vectors (cosine 1.0000 on both probes, so nothing in the store needs
re-embedding), 197 MB resident instead of 1381 MB, 12.8 s cold start instead of
54 s. That drop is what lets the whole layer run on a 1 GB free VM.
"""
import asyncio
import os
from typing import List

from fastembed import TextEmbedding

MODEL_NAME = os.getenv("MEMORY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# The model cache is deliberately NOT under MEMORY_DIR. The evals point that at
# a fresh temp dir, so caching there re-downloaded 83 MB per run, and on Windows
# a partial download leaves a snapshot missing config.json that every later run
# then fails to load. Keep it beside the code, where it is written once.
MODEL_CACHE = os.getenv("MEMORY_MODEL_CACHE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EMBEDDING_DIM = 384

_model = None


def _get_model() -> TextEmbedding:
    """Loaded on first use, not on import — listing or deleting memories should
    not pay for the model."""
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME, cache_dir=MODEL_CACHE)
    return _model


async def generate_embeddings(strings: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns plain Python float lists."""
    def _embed():
        return [v.tolist() for v in _get_model().embed(strings)]
    return await asyncio.to_thread(_embed)


if __name__ == "__main__":
    strings = ["Hello world", "How are you?", "This is a test."]
    embeddings = asyncio.run(generate_embeddings(strings))
    print(len(embeddings), "x", len(embeddings[0]))
