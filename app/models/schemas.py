"""
=====================================================================
Loom AI - Document Retriever Service | Pydantic Schemas
=====================================================================
Defines all request/response models used by the API endpoints.
Pydantic models provide automatic validation, serialization,
and Swagger documentation generation.
=====================================================================
"""

from pydantic import BaseModel, Field


# ===========================================================================
# Upload Responses
# ===========================================================================

class DocumentUploadResponse(BaseModel):
    """Response returned after a successful CSV document upload."""
    tenant_name: str              # The tenant (app) that owns these documents
    documents_processed: int      # Number of rows/documents processed from CSV
    total_chunks: int             # Total number of chunks created across all documents
    message: str = "Upload successful"


# ===========================================================================
# Document Chunk Models
# ===========================================================================

class ChunkResponse(BaseModel):
    """Represents a single chunk of a document stored in Weaviate."""
    doc_id: int          # Parent document ID (from CSV "id" column)
    context: str         # The text content of this chunk
    tags: list[str]      # Tags associated with this document (from CSV "tags" column)
    chunk_index: int     # Position of this chunk within the document (0-based)
    total_chunks: int    # Total number of chunks the document was split into


class DocumentResponse(BaseModel):
    """Full document with all its chunks reassembled."""
    doc_id: int                           # Document ID
    context: str                          # Full reconstructed text (all chunks joined)
    tags: list[str]                       # Document tags
    total_chunks: int                     # Number of chunks this document has
    chunks: list[ChunkResponse] = []      # Individual chunks (optional detail)


class DocumentListResponse(BaseModel):
    """Paginated list of documents for a tenant."""
    tenant_name: str                      # The tenant name
    documents: list[DocumentResponse]     # List of documents
    total: int                            # Total number of distinct documents


# ===========================================================================
# Update Request
# ===========================================================================

class DocumentUpdateRequest(BaseModel):
    """Request body for updating a single document's content and tags."""
    context: str                                   # New text content for the document
    tags: list[str] = Field(default_factory=list)  # New tags (replaces existing tags)


# ===========================================================================
# Search Models
# ===========================================================================

class SearchRequest(BaseModel):
    """Request body for semantic search across a tenant's documents."""
    query: str                                                      # User's search query text
    top_k: int = Field(default=5, ge=1, le=100)                     # Max number of results to return
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)      # Minimum confidence threshold (0-1)
    tags: list[str] = Field(default_factory=list)                   # Optional: filter results by these tags


class SearchResult(BaseModel):
    """A single search result with confidence score."""
    doc_id: int          # Document ID that this chunk belongs to
    context: str         # The matched chunk text content
    tags: list[str]      # Tags on this document
    confidence: float    # Confidence score (0.0 to 1.0, higher = more relevant)
    chunk_index: int     # Which chunk of the document this result is from


class SearchResponse(BaseModel):
    """Search response - returns only the list of matching document chunks."""
    documents: list[SearchResult]     # Ranked list of matching chunks


# ===========================================================================
# Tenant Management Models
# ===========================================================================

class TenantDeleteResponse(BaseModel):
    """Response after deleting a tenant's entire collection."""
    tenant_name: str
    message: str = "Tenant collection deleted successfully"
