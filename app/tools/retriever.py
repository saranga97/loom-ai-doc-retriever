"""
=====================================================================
Loom AI - Document Retriever Service | LangChain Tool Interface
=====================================================================
This module provides the document retrieval function that will be
used as a LangChain tool by the Chat Agent Framework. It wraps the
document search service in a simple function interface.

When the agent framework calls this tool, it passes:
  - tenant_name: Which tenant's documents to search
  - query: The user's question or search text
  - top_k: How many results to return (configurable per user)
  - min_confidence: Minimum confidence threshold

This module acts as the bridge between the Agent Framework and
the Document Retriever Service.
=====================================================================
"""

from app.services.document import search


def retrieve_documents(
    tenant_name: str,
    query: str,
    top_k: int = 5,
    min_confidence: float = 0.0,
    tags: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve relevant document chunks for a given query.

    This function is designed to be registered as a LangChain tool
    in the agent framework. The agent will call this when it needs
    to look up information from the tenant's knowledge base.

    Args:
        tenant_name: The tenant (chatbot app) to search within
        query: The user's question or search text
        top_k: Number of document chunks to retrieve (configurable per user)
        min_confidence: Minimum confidence score to include a result (0.0-1.0)
        tags: Optional tag filter to narrow search to specific categories

    Returns:
        List of dicts, each containing:
          - doc_id: Document identifier
          - context: The relevant text chunk
          - tags: Tags on this document
          - confidence: Similarity score (0.0-1.0)
          - chunk_index: Position of this chunk in the original document
    """
    # Call the search service and return just the results list
    # (stripping the wrapper metadata like tenant_name and query)
    result = search(
        tenant_name=tenant_name,
        query=query,
        top_k=top_k,
        min_confidence=min_confidence,
        tags=tags,
    )
    return result["results"]
