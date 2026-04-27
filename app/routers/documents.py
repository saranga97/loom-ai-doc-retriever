"""
=====================================================================
Loom AI - Document Retriever Service | API Router
=====================================================================
Defines all FastAPI endpoints for document management and search.
All endpoints are prefixed with /api/v1/documents/ and grouped under
the "documents" tag in Swagger UI.

Endpoints:
  POST   /upload/{tenant_name}       - Upload CSV documents
  GET    /{tenant_name}              - List all documents (paginated)
  GET    /{tenant_name}/{doc_id}     - Get a single document with chunks
  PUT    /{tenant_name}/{doc_id}     - Update a document
  DELETE /{tenant_name}/{doc_id}     - Delete a document
  POST   /search/{tenant_name}       - Semantic search with filters
  DELETE /tenant/{tenant_name}       - Delete entire tenant collection
=====================================================================
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.models.schemas import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    DocumentUpdateRequest,
    SearchRequest,
    SearchResponse,
    TenantDeleteResponse,
)
from app.services import document, weaviate_client

# ---------------------------------------------------------------------------
# Create the router with a common prefix and tag for all document endpoints.
# The prefix means all routes here are relative to /api/v1/documents/
# The tag groups these endpoints together in the Swagger UI.
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# ===========================================================================
# CSV Upload Endpoint
# ===========================================================================

@router.post("/upload/{tenant_name}", response_model=DocumentUploadResponse)
async def upload_csv(
    tenant_name: str,
    file: UploadFile = File(..., description="CSV file with columns: id, context, tags"),
    chunk_size: int | None = Query(
        None, ge=50, le=2000,
        description="Override default chunk size (tokens per chunk). Default: 500"
    ),
    chunk_overlap: int | None = Query(
        None, ge=0, le=500,
        description="Override default chunk overlap (tokens). Default: 50"
    ),
):
    """
    Upload a CSV file to ingest documents for a specific tenant.

    The CSV must have these columns:
    - **id**: Numeric document identifier (ascending order)
    - **context**: The text content of the document
    - **tags**: Comma-separated tags for categorization and filtering

    Each document is split into chunks, embedded via OpenAI, and stored
    in a tenant-specific Weaviate collection (LoomAI_{tenant_name}).
    """
    # Validate that the uploaded file is a CSV
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV (.csv)")

    # Process the CSV through the document service pipeline
    result = await document.upload_csv(tenant_name, file, chunk_size, chunk_overlap)
    return DocumentUploadResponse(**result)


# ===========================================================================
# Tenant Management Endpoints
# ===========================================================================
# IMPORTANT: These routes use /tenant/{tenant_name} and must be registered
# BEFORE the dynamic /{tenant_name}/{doc_id} routes, otherwise FastAPI
# would match "tenant" as a tenant_name and "app_cinnamon" as a doc_id.
# ===========================================================================

@router.delete("/tenant/{tenant_name}", response_model=TenantDeleteResponse)
def delete_tenant(tenant_name: str):
    """
    Delete an entire tenant's collection from Weaviate.

    WARNING: This permanently removes ALL documents, chunks, and vectors
    for this tenant. This action cannot be undone.

    Use this when a user deletes their chatbot and all associated data
    should be cleaned up.
    """
    deleted = weaviate_client.delete_collection(tenant_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{tenant_name}' collection not found"
        )
    return TenantDeleteResponse(tenant_name=tenant_name)


# ===========================================================================
# Document List Endpoint
# ===========================================================================

@router.get("/{tenant_name}", response_model=DocumentListResponse)
def list_documents(
    tenant_name: str,
    limit: int = Query(100, ge=1, le=1000, description="Max documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
):
    """
    List all documents for a tenant with pagination.

    Returns de-duplicated documents (since each document may have
    multiple chunks stored in Weaviate). The context field shows
    the first chunk as a preview.
    """
    return document.get_documents(tenant_name, limit, offset)


# ===========================================================================
# Single Document Endpoint
# ===========================================================================

@router.get("/{tenant_name}/{doc_id}", response_model=DocumentResponse)
def get_document(tenant_name: str, doc_id: int):
    """
    Get a single document with all its chunks.

    Returns the full reconstructed document text (all chunks joined)
    along with individual chunk details including their text content,
    tags, and position within the document.
    """
    result = document.get_document(tenant_name, doc_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return result


# ===========================================================================
# Document Update Endpoint
# ===========================================================================

@router.put("/{tenant_name}/{doc_id}", response_model=DocumentResponse)
def update_document(
    tenant_name: str,
    doc_id: int,
    body: DocumentUpdateRequest,
):
    """
    Update a document's content and tags.

    This will:
    1. Delete all existing chunks for this document
    2. Re-chunk the new content
    3. Re-embed all new chunks via OpenAI
    4. Store the new chunks in Weaviate

    Use this when a document's content has changed and needs
    fresh embeddings for accurate search results.
    """
    result = document.update_document(tenant_name, doc_id, body.context, body.tags)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return result


# ===========================================================================
# Document Delete Endpoint
# ===========================================================================

@router.delete("/{tenant_name}/{doc_id}")
def delete_document(tenant_name: str, doc_id: int):
    """
    Delete a document and all its chunks from the tenant's collection.

    This permanently removes all chunks and their vectors from Weaviate.
    The document ID can be reused after deletion.
    """
    deleted = document.delete_document(tenant_name, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"message": "Document deleted successfully", "doc_id": doc_id}


# ===========================================================================
# Semantic Search Endpoint
# ===========================================================================

@router.post("/search/{tenant_name}", response_model=SearchResponse)
def search_documents(tenant_name: str, body: SearchRequest):
    """
    Perform semantic search across a tenant's documents.

    The search converts the query into an embedding vector and finds
    the most similar document chunks in Weaviate using cosine similarity.

    Request body parameters:
    - **query**: Natural language search text
    - **top_k**: Number of results to return (1-100, default: 5).
      Customizable per user to control how many documents are retrieved.
    - **min_confidence**: Minimum confidence threshold (0.0-1.0, default: 0.0).
      Results below this confidence are filtered out.
    - **tags**: Optional list of tags to filter by. Only chunks with
      at least one matching tag will be returned.

    Each result includes a **confidence** score (0.0 to 1.0) indicating
    how semantically similar the chunk is to the query.
    """
    result = document.search(
        tenant_name=tenant_name,
        query=body.query,
        top_k=body.top_k,
        min_confidence=body.min_confidence,
        # Filter out empty strings from tags list (e.g. [""] -> None)
        tags=[t for t in body.tags if t.strip()] or None,
    )
    # Return only the documents list, no wrapper metadata
    return SearchResponse(documents=result["results"])
