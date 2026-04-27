"""
=====================================================================
Loom AI - Document Retriever Service | Text Chunking Module
=====================================================================
Splits large documents into smaller, fixed-size token chunks with
overlap. This is essential for RAG because:

  1. Embedding models have input token limits
  2. Smaller chunks produce more focused/precise embeddings
  3. Overlap ensures context isn't lost at chunk boundaries
  4. Search returns the most relevant *portion* of a document

Chunking strategy: Fixed-size token windows with overlap
  - Uses tiktoken (OpenAI's tokenizer) for accurate token counting
  - Default: 500 tokens per chunk, 50 tokens overlap
  - cl100k_base encoding (used by GPT-4 and text-embedding-3-small)
=====================================================================
"""

import tiktoken

# ---------------------------------------------------------------------------
# Initialize the tokenizer with cl100k_base encoding.
# This is the encoding used by GPT-4, GPT-3.5-turbo, and
# text-embedding-3-small/large models. It ensures our token count
# matches what OpenAI's embedding API expects.
# ---------------------------------------------------------------------------
_encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size token chunks with sliding window overlap.

    Algorithm:
      1. Encode the full text into tokens
      2. If it fits in one chunk, return as-is
      3. Otherwise, create chunks using a sliding window:
         - Each window is `chunk_size` tokens wide
         - Windows overlap by `overlap` tokens
         - The overlap preserves context between adjacent chunks

    Args:
        text: The full document text to chunk
        chunk_size: Maximum tokens per chunk (default: 500)
        overlap: Number of tokens to overlap between chunks (default: 50)

    Returns:
        List of text chunks (decoded back to strings)

    Example with chunk_size=500, overlap=50:
        Chunk 1: tokens[0:500]
        Chunk 2: tokens[450:950]    (overlaps 50 tokens with chunk 1)
        Chunk 3: tokens[900:1400]   (overlaps 50 tokens with chunk 2)
        ...and so on until all tokens are covered
    """
    # Encode the text into token IDs
    tokens = _encoding.encode(text)

    # If the text fits in a single chunk, return it as-is (no splitting needed)
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        # Define the end of the current chunk window
        end = start + chunk_size

        # Extract tokens for this chunk
        chunk_tokens = tokens[start:end]

        # Decode tokens back to text and add to chunks list
        chunks.append(_encoding.decode(chunk_tokens))

        # If we've reached the end of the text, stop
        if end >= len(tokens):
            break

        # Move the window forward, keeping `overlap` tokens from the end
        # of the current chunk for context continuity
        start = end - overlap

    return chunks
