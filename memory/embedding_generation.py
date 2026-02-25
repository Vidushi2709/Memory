from sentence_transformers import SentenceTransformer
import asyncio
from typing import List

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def generate_embeddings(strings: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of strings.
    
    Returns a list of plain Python float lists (ChromaDB-compatible).
    """
    print(strings)
    embeddings = await asyncio.to_thread(_model.encode, strings)
    print(embeddings.shape)
    return [emb.tolist() for emb in embeddings]


if __name__ == "__main__":
    strings = ["Hello world", "How are you?", "This is a test."]
    embeddings = asyncio.run(generate_embeddings(strings))
    print(embeddings)