"""
=====================================================================
Loom AI - Document Retriever Service | Weaviate Client Module
=====================================================================
Manages the connection to the Weaviate vector database and handles
collection (schema) creation for each tenant. Each tenant gets its
own isolated collection with the naming pattern: LoomAI_{tenant_name}

Weaviate is used as the vector store because:
  - It supports hybrid search (vector + keyword)
  - Self-hosted with no cloud dependency
  - Efficient for multi-tenant setups with separate collections

Connection modes:
  - Docker (default): Connects to Weaviate running in a Docker container
  - Embedded: Runs Weaviate as a subprocess (no Docker required)
    Set WEAVIATE_EMBEDDED=true in your .env file to use this mode
=====================================================================
"""

import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.embedded import EmbeddedOptions
from app.config import settings


# ---------------------------------------------------------------------------
# Module-level Weaviate client instance (singleton pattern).
# This is initialized on app startup via connect() and cleaned up
# on shutdown via disconnect().
# ---------------------------------------------------------------------------
client: weaviate.WeaviateClient | None = None
_connected: bool = False


def is_connected() -> bool:
    """Check if the Weaviate client is connected and ready."""
    return _connected and client is not None


def get_client() -> weaviate.WeaviateClient:
    """
    Return the active Weaviate client instance.
    Raises RuntimeError if connect() hasn't been called yet or connection failed.
    """
    global client
    if client is None or not _connected:
        raise RuntimeError(
            "Weaviate client not connected. "
            "Ensure Weaviate is running and reachable, then restart the service."
        )
    return client


def connect() -> None:
    """
    Establish connection to Weaviate.

    Supports two modes:
      - Embedded (WEAVIATE_EMBEDDED=true): Runs Weaviate as a subprocess,
        no Docker required. Data persists in ./weaviate_data/
      - Docker/External (default): Connects to Weaviate at WEAVIATE_URL
        (typically a Docker container on localhost:8080)

    Raises WeaviateConnectionError if connection fails.
    """
    global client, _connected

    if settings.weaviate_embedded:
        # --- Embedded mode: no Docker required ---
        client = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(
                persistence_data_path="./weaviate_data",
            )
        )
        client.connect()
    else:
        # --- External mode: connect to Docker/remote Weaviate ---
        host = settings.weaviate_url.replace("http://", "").split(":")[0]
        port = int(settings.weaviate_url.split(":")[-1])

        client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=settings.weaviate_grpc_port,
        )

    _connected = True


def disconnect() -> None:
    """
    Gracefully close the Weaviate connection.
    Called during application shutdown to release resources.
    """
    global client, _connected
    if client is not None:
        client.close()
        client = None
    _connected = False


# ---------------------------------------------------------------------------
# Collection (Schema) Management
# ---------------------------------------------------------------------------

def _collection_name(tenant_name: str) -> str:
    """
    Generate the Weaviate collection name for a tenant.
    Pattern: LoomAI_{tenant_name}
    Example: tenant "acme_corp" -> collection "LoomAI_acme_corp"
    """
    return f"LoomAI_{tenant_name}"


def ensure_collection(tenant_name: str) -> None:
    """
    Create the Weaviate collection for a tenant if it doesn't exist.

    Collection schema:
      - doc_id (INT): Unique document identifier from the CSV "id" column
      - context (TEXT): The text content of a chunk (indexed for keyword search)
      - tags (TEXT_ARRAY): Tags from CSV for categorization and filtering
      - chunk_index (INT): Position of this chunk within the parent document
      - total_chunks (INT): Total number of chunks the parent document was split into

    Vectorizer is set to "none" because we provide our own vectors
    via OpenAI's embedding API (external embedding).
    """
    c = get_client()
    name = _collection_name(tenant_name)

    # Skip creation if the collection already exists
    if c.collections.exists(name):
        return

    # Create collection with explicit property definitions and no auto-vectorizer
    c.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),  # We supply our own vectors
        properties=[
            Property(name="doc_id", data_type=DataType.INT),
            Property(name="context", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="total_chunks", data_type=DataType.INT),
        ],
    )
    print(f"[Weaviate] Created collection: {name}")


def get_collection(tenant_name: str):
    """
    Get a reference to a tenant's Weaviate collection.
    Automatically creates the collection if it doesn't exist yet.
    Returns a Weaviate Collection object for CRUD and query operations.
    """
    ensure_collection(tenant_name)
    return get_client().collections.get(_collection_name(tenant_name))


def delete_collection(tenant_name: str) -> bool:
    """
    Delete an entire tenant's collection from Weaviate.
    This removes all documents, chunks, and vectors for the tenant.
    Returns True if the collection existed and was deleted, False otherwise.
    """
    c = get_client()
    name = _collection_name(tenant_name)
    if c.collections.exists(name):
        c.collections.delete(name)
        print(f"[Weaviate] Deleted collection: {name}")
        return True
    return False


def collection_exists(tenant_name: str) -> bool:
    """Check if a collection exists for the given tenant name."""
    c = get_client()
    return c.collections.exists(_collection_name(tenant_name))


def list_tenant_collections() -> list[str]:
    """
    List all tenant names that have a Weaviate collection.
    Returns tenant names (without the LoomAI_ prefix).
    """
    c = get_client()
    prefix = "LoomAI_"
    tenant_names = []
    for name in c.collections.list_all().keys():
        if name.startswith(prefix):
            tenant_names.append(name[len(prefix):])
    return tenant_names
