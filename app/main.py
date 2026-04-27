"""
=====================================================================
Loom AI - Document Retriever Service | FastAPI Application Entry Point
=====================================================================
This is the main entry point for the Document Retriever microservice.
It sets up the FastAPI application with:
  - Weaviate connection lifecycle (connect on startup, disconnect on shutdown)
  - CORS middleware for cross-origin requests from frontend clients
  - API router registration for document management endpoints
  - Health check endpoint for monitoring
=====================================================================
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services import weaviate_client
from app.routers import documents


# ---------------------------------------------------------------------------
# Lifespan context manager - handles startup and shutdown events.
# This replaces the deprecated @app.on_event("startup") pattern.
# On startup: connect to Weaviate vector database
# On shutdown: gracefully close the Weaviate connection
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: establish Weaviate connection ---
    weaviate_client.connect()
    print(f"[Loom AI] Connected to Weaviate at {settings.weaviate_url}")
    print(f"[Loom AI] Environment: {settings.environment}")
    print(f"[Loom AI] Embedding model: {settings.embedding_model}")

    yield  # Application runs here

    # --- Shutdown: close Weaviate connection ---
    weaviate_client.disconnect()
    print("[Loom AI] Disconnected from Weaviate")


# ---------------------------------------------------------------------------
# FastAPI application instance with metadata for Swagger UI documentation.
# Visit http://localhost:8001/docs to see the interactive API docs.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Loom AI - Document Retriever",
    description=(
        "RAG (Retrieval-Augmented Generation) document ingestion and "
        "semantic search service. Supports multi-tenant document management, "
        "CSV upload, chunking, embedding, and vector search via Weaviate."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware - allows the frontend (React/Vite) to call this API
# from a different origin (e.g., localhost:5173 -> localhost:8001).
# In production, restrict allow_origins to specific domains.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # TODO: restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],       # Allow all headers (Authorization, Content-Type, etc.)
)

# ---------------------------------------------------------------------------
# Register the documents router - all document CRUD and search endpoints.
# This adds routes under /api/v1/documents/
# ---------------------------------------------------------------------------
app.include_router(documents.router)


# ---------------------------------------------------------------------------
# Health check endpoint - used by Docker, load balancers, and monitoring
# tools to verify the service is running and responsive.
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health_check():
    """Return service health status and current environment."""
    return {
        "status": "healthy",
        "service": "document-retriever",
        "environment": settings.environment,
    }


# ---------------------------------------------------------------------------
# Direct execution entry point - run with: python -m app.main
# In dev mode, hot-reload is enabled for faster development.
# In live mode, reload is disabled for stability.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.doc_retriever_port,
        reload=settings.environment == "dev",  # Auto-reload only in dev mode
    )
