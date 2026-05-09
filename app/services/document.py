"""
=====================================================================
Loom AI - Document Retriever Service | Document Service Module
=====================================================================
Core business logic for document management and semantic search.
This module handles:
  - CSV parsing and document ingestion (upload)
  - Document CRUD operations (get, update, delete)
  - Semantic vector search with tag filtering and confidence scores
  - Chunk management (splitting, re-embedding on update)

All operations are scoped to a tenant (multi-tenant isolation).
Each tenant has its own Weaviate collection.
=====================================================================
"""

import csv
import io
from fastapi import UploadFile
from weaviate.classes.query import MetadataQuery, Filter

from app.config import settings
from app.services import weaviate_client, embedding, chunking


# ===========================================================================
# CSV Upload / Document Ingestion
# ===========================================================================

async def upload_csv(
    tenant_name: str,
    file: UploadFile,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """
    Parse a CSV file, chunk each document, generate embeddings,
    and store everything in Weaviate.

    CSV format expected:
      id,context,tags
      1,"Full document text here","tag1,tag2,tag3"
      2,"Another document...","faq,billing"

    Pipeline for each row:
      1. Parse id, context, and tags from CSV
      2. Split context into token-based chunks (chunking.chunk_text)
      3. Generate embedding vectors for all chunks (embedding.embed_texts)
      4. Insert each chunk with its vector into Weaviate

    Args:
        tenant_name: Identifier for the tenant (creates/uses LoomAI_{tenant_name} collection)
        file: Uploaded CSV file
        chunk_size: Override default chunk size (tokens per chunk)
        chunk_overlap: Override default chunk overlap (tokens)

    Returns:
        Dict with tenant_name, documents_processed count, and total_chunks count
    """
    # Use settings defaults if no override provided
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Read and decode the uploaded CSV file content
    content = await file.read()
    text = content.decode("utf-8")

    # Parse CSV using DictReader (expects header row: id, context, tags)
    reader = csv.DictReader(io.StringIO(text))

    # Get or create the Weaviate collection for this tenant
    collection = weaviate_client.get_collection(tenant_name)

    documents_processed = 0
    total_chunks = 0

    # Process each row (document) in the CSV
    for row in reader:
        # Extract document ID (numeric, ascending order as per CSV)
        doc_id = int(row["doc_id"])

        # Extract the document text content
        context = row["context"]

        # Parse comma-separated tags into a list, stripping whitespace
        tags = [t.strip() for t in row.get("tags", "").split(",") if t.strip()]

        # Step 1: Split the document text into manageable chunks
        chunks = chunking.chunk_text(context, chunk_size, chunk_overlap)

        # Step 2: Generate embedding vectors for all chunks in one batch
        # (batch embedding is more efficient than one-by-one)
        vectors = embedding.embed_texts(chunks)

        # Step 3: Insert each chunk with its metadata and vector into Weaviate
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            collection.data.insert(
                properties={
                    "doc_id": doc_id,           # Parent document ID
                    "context": chunk_text,       # Chunk text content
                    "tags": tags,                # Tags from CSV (same for all chunks of a doc)
                    "chunk_index": i,            # Chunk position within the document
                    "total_chunks": len(chunks), # Total chunks for this document
                },
                vector=vector,  # The embedding vector for this chunk
            )

        documents_processed += 1
        total_chunks += len(chunks)
        print(f"[Upload] doc_id={doc_id} -> {len(chunks)} chunks")

    return {
        "tenant_name": tenant_name,
        "documents_processed": documents_processed,
        "total_chunks": total_chunks,
    }


# ===========================================================================
# Document Retrieval (GET operations)
# ===========================================================================

def get_documents(tenant_name: str, limit: int = 100, offset: int = 0) -> dict:
    """
    List all distinct documents for a tenant (paginated).
    Since documents are stored as individual chunks, we need to
    de-duplicate by doc_id to return unique document entries.

    Args:
        tenant_name: The tenant to query
        limit: Maximum number of documents to return
        offset: Number of documents to skip (for pagination)

    Returns:
        Dict with tenant_name, list of documents, and total count

    Raises:
        ValueError: If the tenant collection does not exist
    """
    if not weaviate_client.collection_exists(tenant_name):
        raise ValueError(f"Tenant '{tenant_name}' not found")
    collection = weaviate_client.get_collection(tenant_name)

    # Fetch objects from Weaviate (we fetch extra to account for multiple chunks per doc)
    results = collection.query.fetch_objects(limit=limit + offset)

    # De-duplicate chunks into unique documents by doc_id
    seen: dict[int, dict] = {}
    for obj in results.objects:
        props = obj.properties
        doc_id = props["doc_id"]
        # Only keep the first occurrence of each doc_id
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "context": props["context"],       # Shows first chunk as preview
                "tags": props["tags"],
                "total_chunks": props["total_chunks"],
                "chunks": [],                       # Chunks not loaded in list view
            }

    # Apply pagination to the de-duplicated document list
    docs = list(seen.values())
    paginated = docs[offset: offset + limit]

    return {
        "tenant_name": tenant_name,
        "documents": paginated,
        "total": len(docs),
    }


def get_document(tenant_name: str, doc_id: int) -> dict | None:
    """
    Get a single document with all its chunks reassembled.
    Fetches all chunks for a doc_id, sorts them by chunk_index,
    and joins them back into the full document text.

    Args:
        tenant_name: The tenant to query
        doc_id: The document ID to retrieve

    Returns:
        Dict with full document data including chunks, or None if not found

    Raises:
        ValueError: If the tenant collection does not exist
    """
    if not weaviate_client.collection_exists(tenant_name):
        raise ValueError(f"Tenant '{tenant_name}' not found")
    collection = weaviate_client.get_collection(tenant_name)

    # Filter by doc_id to get all chunks belonging to this document
    results = collection.query.fetch_objects(
        filters=_doc_id_filter(doc_id),
        limit=1000,  # Upper bound; a single doc unlikely to have >1000 chunks
    )

    # Return None if no chunks found (document doesn't exist)
    if not results.objects:
        return None

    # Build the list of chunk details
    chunks = []
    for obj in results.objects:
        props = obj.properties
        chunks.append({
            "doc_id": props["doc_id"],
            "context": props["context"],
            "tags": props["tags"],
            "chunk_index": props["chunk_index"],
            "total_chunks": props["total_chunks"],
        })

    # Sort chunks by their index to reconstruct the original document order
    chunks.sort(key=lambda c: c["chunk_index"])

    # Join all chunk texts to reconstruct the full document content
    full_context = " ".join(c["context"] for c in chunks)

    return {
        "doc_id": doc_id,
        "context": full_context,
        "tags": chunks[0]["tags"],
        "total_chunks": chunks[0]["total_chunks"],
        "chunks": chunks,
    }


# ===========================================================================
# Document Update
# ===========================================================================

def update_document(
    tenant_name: str,
    doc_id: int,
    context: str,
    tags: list[str],
) -> dict | None:
    """
    Update a document by replacing all its chunks with new content.

    Process:
      1. Delete all existing chunks for this doc_id
      2. Re-chunk the new content
      3. Re-embed all new chunks
      4. Insert new chunks into Weaviate

    Args:
        tenant_name: The tenant that owns this document
        doc_id: The document ID to update
        context: New full text content
        tags: New tags list

    Returns:
        Updated document dict with new chunks, or None if doc_id not found
    """
    # First, delete all existing chunks for this document
    deleted = _delete_doc_chunks(tenant_name, doc_id)
    if not deleted:
        return None  # Document didn't exist

    # Get the collection reference
    collection = weaviate_client.get_collection(tenant_name)

    # Re-chunk the new content using default settings
    chunks = chunking.chunk_text(context, settings.chunk_size, settings.chunk_overlap)

    # Generate new embeddings for all chunks
    vectors = embedding.embed_texts(chunks)

    # Insert the new chunks with their vectors
    for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
        collection.data.insert(
            properties={
                "doc_id": doc_id,
                "context": chunk_text,
                "tags": tags,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
            vector=vector,
        )

    print(f"[Update] doc_id={doc_id} -> {len(chunks)} new chunks")

    # Return the updated document (fetched fresh to confirm)
    return get_document(tenant_name, doc_id)


# ===========================================================================
# Document Deletion
# ===========================================================================

def delete_document(tenant_name: str, doc_id: int) -> bool:
    """
    Delete a document and all its chunks from Weaviate.

    Args:
        tenant_name: The tenant that owns this document
        doc_id: The document ID to delete

    Returns:
        True if the document existed and was deleted, False otherwise
    """
    return _delete_doc_chunks(tenant_name, doc_id)


# ===========================================================================
# Semantic Search
# ===========================================================================

def search(
    tenant_name: str,
    query: str,
    top_k: int = 5,
    min_confidence: float = 0.0,
    tags: list[str] | None = None,
) -> dict:
    """
    Perform semantic vector search across a tenant's documents.

    How it works:
      1. Convert the query text into an embedding vector
      2. Search Weaviate for the nearest vectors (most similar chunks)
      3. Optionally filter by tags to narrow results
      4. Convert Weaviate's distance metric to a confidence score
      5. Filter out results below the minimum confidence threshold

    Args:
        tenant_name: The tenant to search within
        query: The user's natural language search query
        top_k: Maximum number of results to return (configurable per user)
        min_confidence: Minimum confidence score threshold (0.0 to 1.0)
        tags: Optional list of tags to filter by (results must have ANY of these tags)

    Returns:
        Dict with tenant_name, query, ranked results list, and total count
    """
    if not weaviate_client.collection_exists(tenant_name):
        raise ValueError(f"Tenant '{tenant_name}' not found")
    collection = weaviate_client.get_collection(tenant_name)

    # Step 1: Convert the search query into an embedding vector
    # This puts the query in the same vector space as the stored chunks
    query_vector = embedding.embed_query(query)

    # Step 2: Build optional tag filter
    # If tags are provided, filter to only return chunks that have at least one matching tag
    filters = None
    if tags:
        # Use ContainsAny to match chunks that have ANY of the specified tags
        filters = Filter.by_property("tags").contains_any(tags)

    # Step 3: Perform vector similarity search in Weaviate
    # near_vector finds the closest vectors to our query vector
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        filters=filters,  # None if no tag filter, otherwise applies tag constraint
        return_metadata=MetadataQuery(distance=True),  # Include distance for confidence calc
    )

    # Step 4: Process results and calculate confidence scores
    search_results = []
    for obj in results.objects:
        # Weaviate returns cosine distance (0 = identical, 2 = opposite)
        # Convert to confidence: confidence = 1 - distance
        # This gives us a 0-1 score where 1 = perfect match
        distance = obj.metadata.distance or 0.0
        confidence = 1.0 - distance

        # Step 5: Filter out results below the confidence threshold
        if confidence < min_confidence:
            continue

        props = obj.properties
        search_results.append({
            "doc_id": props["doc_id"],
            "context": props["context"],
            "tags": props["tags"],
            "confidence": round(confidence, 4),  # Round for clean output
            "chunk_index": props["chunk_index"],
        })

    return {
        "tenant_name": tenant_name,
        "query": query,
        "results": search_results,
        "total": len(search_results),
    }


# ===========================================================================
# Internal Helper Functions
# ===========================================================================

def _doc_id_filter(doc_id: int):
    """
    Create a Weaviate filter expression to match a specific doc_id.
    Used internally to find all chunks belonging to a document.
    """
    return Filter.by_property("doc_id").equal(doc_id)


def _delete_doc_chunks(tenant_name: str, doc_id: int) -> bool:
    """
    Delete all chunks belonging to a specific doc_id from Weaviate.
    This is used by both delete_document() and update_document().

    Args:
        tenant_name: The tenant collection to delete from
        doc_id: The document ID whose chunks should be deleted

    Returns:
        True if chunks were found and deleted, False if no chunks existed

    Raises:
        ValueError: If the tenant collection does not exist
    """
    if not weaviate_client.collection_exists(tenant_name):
        raise ValueError(f"Tenant '{tenant_name}' not found")
    collection = weaviate_client.get_collection(tenant_name)

    # Find all chunks for this doc_id
    results = collection.query.fetch_objects(
        filters=_doc_id_filter(doc_id),
        limit=1000,
    )

    # If no chunks found, the document doesn't exist
    if not results.objects:
        return False

    # Delete each chunk by its Weaviate UUID
    for obj in results.objects:
        collection.data.delete_by_id(obj.uuid)

    print(f"[Delete] Removed {len(results.objects)} chunks for doc_id={doc_id}")
    return True
