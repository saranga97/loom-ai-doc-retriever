"""
=====================================================================
Loom AI - Document Retriever Service | Embedding Module
=====================================================================
Handles text-to-vector conversion using OpenAI's Embeddings API.
The embedding model converts text into high-dimensional vectors that
capture semantic meaning, enabling similarity search in Weaviate.

Current model: text-embedding-3-small
  - Dimensions: 1536
  - Good balance of quality vs cost
  - Suitable for RAG retrieval tasks
=====================================================================
"""

from openai import OpenAI
from app.config import settings

# ---------------------------------------------------------------------------
# Module-level OpenAI client (lazy singleton).
# Initialized on first use to avoid importing OpenAI before env vars load.
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """
    Return the OpenAI client, creating it on first call.
    Uses the API key from settings (loaded from .env file).
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of text strings into vectors using OpenAI.

    Args:
        texts: List of text strings to embed (e.g., document chunks)

    Returns:
        List of embedding vectors (each is a list of floats)

    Note: OpenAI's API supports batch embedding, which is more efficient
    than embedding one text at a time. The order of results matches
    the order of inputs.
    """
    if not texts:
        return []

    # Call OpenAI embeddings API with the configured model
    response = _get_client().embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )

    # Extract the embedding vector from each response item
    # response.data is ordered to match the input texts
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string into a vector.
    This is used during search to convert the user's query
    into the same vector space as the stored document chunks.

    Args:
        query: The user's search query text

    Returns:
        A single embedding vector (list of floats)
    """
    return embed_texts([query])[0]
