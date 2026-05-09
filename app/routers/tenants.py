"""
=====================================================================
Loom AI - Document Retriever Service | Tenants Router
=====================================================================
Manages tenant collections in Weaviate. Provides endpoints to
create, list, and delete tenant collections without auto-creating
collections on read operations.

Endpoints:
  POST   /api/v1/tenants/                   - Create a tenant collection
  GET    /api/v1/tenants/                   - List all tenant collections
  GET    /api/v1/tenants/{tenant_name}      - Check if a tenant exists
  DELETE /api/v1/tenants/{tenant_name}      - Delete a tenant collection
=====================================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import weaviate_client


router = APIRouter(prefix="/api/v1/tenants", tags=["Tenants"])


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TenantCreateRequest(BaseModel):
    tenant_name: str


class TenantResponse(BaseModel):
    tenant_name: str
    exists: bool


class TenantListResponse(BaseModel):
    tenants: list[str]
    total: int


class TenantDeleteResponse(BaseModel):
    tenant_name: str
    message: str = "Tenant collection deleted successfully"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_model=TenantResponse, status_code=201)
def create_tenant(body: TenantCreateRequest):
    """
    Create a Weaviate collection for a tenant.
    Returns 409 if the collection already exists.
    """
    if weaviate_client.collection_exists(body.tenant_name):
        raise HTTPException(
            status_code=409,
            detail=f"Tenant '{body.tenant_name}' already exists",
        )
    weaviate_client.ensure_collection(body.tenant_name)
    return TenantResponse(tenant_name=body.tenant_name, exists=True)


@router.get("/", response_model=TenantListResponse)
def list_tenants():
    """
    List all tenant names that have a Weaviate collection.
    """
    tenants = weaviate_client.list_tenant_collections()
    return TenantListResponse(tenants=tenants, total=len(tenants))


@router.get("/{tenant_name}", response_model=TenantResponse)
def get_tenant(tenant_name: str):
    """
    Check if a tenant collection exists.
    Returns 404 if the tenant does not exist.
    """
    if not weaviate_client.collection_exists(tenant_name):
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{tenant_name}' not found",
        )
    return TenantResponse(tenant_name=tenant_name, exists=True)


@router.delete("/{tenant_name}", response_model=TenantDeleteResponse)
def delete_tenant(tenant_name: str):
    """
    Delete an entire tenant's collection from Weaviate.
    This permanently removes ALL documents, chunks, and vectors.
    """
    deleted = weaviate_client.delete_collection(tenant_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{tenant_name}' collection not found",
        )
    return TenantDeleteResponse(tenant_name=tenant_name)
