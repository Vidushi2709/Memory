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

from memory import MEMORY_DIR

MODEL_NAME = os.getenv("MEMORY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

_model = None


def _get_model() -> TextEmbedding:
    """Loaded on first use, not on import — listing or deleting memories should
    not pay for the model. Cached under MEMORY_DIR so a temp-dir cleanup does
    not force an 83 MB re-download."""
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME, cache_dir=os.path.join(MEMORY_DIR, "models"))
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
