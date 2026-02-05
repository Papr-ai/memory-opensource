"""
OMO (Open Memory Object) Routes.

Endpoints for exporting and importing memories in OMO standard format.
This enables memory portability across OMO-compliant platforms.

See: https://github.com/papr-ai/open-memory-object
"""

from fastapi import APIRouter, Request, Response, Depends, Query, Body, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json

from memory.memory_graph import MemoryGraph
from services.auth_utils import get_user_from_token_optimized
from services.utils import get_memory_graph
from services.logger_singleton import LoggerSingleton
from models.omo import (
    OpenMemoryObject,
    memory_to_omo,
    from_omo,
    export_omo_json,
    import_omo_json,
)
from models.memory_models import AddMemoryRequest, MemoryMetadata
from models.parse_server import AddMemoryResponse
from routers.v1.memory_routes_v1 import api_key_header, bearer_auth, session_token_header
from fastapi import Security

logger = LoggerSingleton.get_logger(__name__)

# Create OMO-specific router
router = APIRouter(prefix="/omo", tags=["omo"])


# =============================================================================
# Request/Response Models
# =============================================================================

class OMOExportRequest(BaseModel):
    """Request model for exporting memories to OMO format."""
    memory_ids: List[str] = Field(
        ...,
        description="List of memory IDs to export"
    )


class OMOExportResponse(BaseModel):
    """Response model for OMO export."""
    code: int = 200
    status: str = "success"
    count: int = Field(description="Number of memories exported")
    memories: List[Dict[str, Any]] = Field(description="Memories in OMO v1 format")
    error: Optional[str] = None


class OMOImportRequest(BaseModel):
    """Request model for importing memories from OMO format."""
    memories: List[Dict[str, Any]] = Field(
        ...,
        description="List of memories in OMO v1 format"
    )
    skip_duplicates: bool = Field(
        default=True,
        description="Skip memories with IDs that already exist"
    )


class OMOImportResponse(BaseModel):
    """Response model for OMO import."""
    code: int = 200
    status: str = "success"
    imported: int = Field(description="Number of memories successfully imported")
    skipped: int = Field(default=0, description="Number of memories skipped (duplicates)")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Import errors")
    memory_ids: List[str] = Field(default_factory=list, description="IDs of imported memories")


# =============================================================================
# Export Endpoint
# =============================================================================

@router.post(
    "/export",
    response_model=OMOExportResponse,
    summary="Export memories to OMO format",
    description="""
    Export memories in Open Memory Object (OMO) standard format.

    This enables memory portability to other OMO-compliant platforms.
    The exported format follows the OMO v1 schema.

    **OMO Standard:** https://github.com/papr-ai/open-memory-object
    """
)
async def export_memories_omo(
    request: Request,
    export_request: OMOExportRequest,
    response: Response,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> OMOExportResponse:
    """Export specified memories to OMO format."""
    try:
        # Authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            auth_header = f'APIKey {api_key}'

        if not auth_header and not api_key and not session_token:
            response.status_code = 401
            return OMOExportResponse(
                code=401,
                status="error",
                count=0,
                memories=[],
                error="Missing authentication"
            )

        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Authenticate
        try:
            import httpx
            async with httpx.AsyncClient() as httpx_client:
                auth_response = await get_user_from_token_optimized(
                    auth_header, client_type, memory_graph,
                    search_request=None, memory_request=None,
                    httpx_client=httpx_client
                )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            response.status_code = 401
            return OMOExportResponse(
                code=401,
                status="error",
                count=0,
                memories=[],
                error="Authentication failed"
            )

        # Fetch memories by ID
        omo_memories = []
        for memory_id in export_request.memory_ids:
            try:
                # Fetch memory from Parse Server
                memory = await memory_graph.get_memory_by_id(memory_id)
                if memory:
                    # Convert to OMO format
                    content = memory.get('content', '')
                    memory_type = memory.get('type', 'text')

                    # Build metadata from memory fields
                    metadata = MemoryMetadata(
                        createdAt=memory.get('createdAt'),
                        consent=memory.get('consent', 'implicit'),
                        risk=memory.get('risk', 'none'),
                        topics=memory.get('topics'),
                        sourceUrl=memory.get('sourceUrl'),
                        external_user_id=memory.get('external_user_id'),
                        user_id=memory.get('user_id'),
                        workspace_id=memory.get('workspace_id'),
                        organization_id=memory.get('organization_id'),
                        namespace_id=memory.get('namespace_id'),
                    )

                    omo_obj = memory_to_omo(
                        memory_id=memory_id,
                        content=content,
                        memory_type=memory_type,
                        metadata=metadata
                    )
                    omo_memories.append(omo_obj.model_dump(mode='json'))
            except Exception as e:
                logger.warning(f"Failed to export memory {memory_id}: {e}")

        return OMOExportResponse(
            code=200,
            status="success",
            count=len(omo_memories),
            memories=omo_memories
        )

    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        response.status_code = 500
        return OMOExportResponse(
            code=500,
            status="error",
            count=0,
            memories=[],
            error=str(e)
        )


# =============================================================================
# Import Endpoint
# =============================================================================

@router.post(
    "/import",
    response_model=OMOImportResponse,
    summary="Import memories from OMO format",
    description="""
    Import memories from Open Memory Object (OMO) standard format.

    This enables importing memories from other OMO-compliant platforms.

    **OMO Standard:** https://github.com/papr-ai/open-memory-object
    """
)
async def import_memories_omo(
    request: Request,
    import_request: OMOImportRequest,
    response: Response,
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
) -> OMOImportResponse:
    """Import memories from OMO format."""
    try:
        # Authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header and api_key:
            auth_header = f'APIKey {api_key}'

        if not auth_header and not api_key and not session_token:
            response.status_code = 401
            return OMOImportResponse(
                code=401,
                status="error",
                imported=0,
                error="Missing authentication"
            )

        client_type = request.headers.get('X-Client-Type', 'papr_plugin')

        # Authenticate
        try:
            import httpx
            async with httpx.AsyncClient() as httpx_client:
                auth_response = await get_user_from_token_optimized(
                    auth_header, client_type, memory_graph,
                    search_request=None, memory_request=None,
                    httpx_client=httpx_client
                )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            response.status_code = 401
            return OMOImportResponse(
                code=401,
                status="error",
                imported=0,
                error="Authentication failed"
            )

        # Import memories
        imported_count = 0
        skipped_count = 0
        errors = []
        memory_ids = []

        for idx, omo_dict in enumerate(import_request.memories):
            try:
                # Parse OMO object
                omo_obj = OpenMemoryObject(**omo_dict)

                # Convert to Papr format
                papr_data = from_omo(omo_obj)

                # Create AddMemoryRequest
                metadata_dict = papr_data.get('metadata', {})
                metadata = MemoryMetadata(**metadata_dict) if metadata_dict else None

                add_request = AddMemoryRequest(
                    content=papr_data['content'],
                    type=papr_data.get('type', 'text'),
                    metadata=metadata
                )

                # Import via memory service
                from services.memory_service import handle_incoming_memory
                result = await handle_incoming_memory(
                    memory_graph=memory_graph,
                    memory_request=add_request,
                    auth_response=auth_response,
                    api_key=api_key,
                    background_tasks=None,
                    request=request
                )

                if result and hasattr(result, 'data') and result.data:
                    imported_count += 1
                    memory_ids.append(result.data[0].memoryId)
                else:
                    errors.append({
                        "index": idx,
                        "omo_id": omo_obj.id,
                        "error": "Import failed"
                    })

            except Exception as e:
                logger.warning(f"Failed to import memory at index {idx}: {e}")
                errors.append({
                    "index": idx,
                    "omo_id": omo_dict.get('id', 'unknown'),
                    "error": str(e)
                })

        return OMOImportResponse(
            code=200,
            status="success",
            imported=imported_count,
            skipped=skipped_count,
            errors=errors,
            memory_ids=memory_ids
        )

    except Exception as e:
        logger.error(f"Import error: {e}", exc_info=True)
        response.status_code = 500
        return OMOImportResponse(
            code=500,
            status="error",
            imported=0,
            error=str(e)
        )


# =============================================================================
# File Export/Import Endpoints
# =============================================================================

@router.get(
    "/export.json",
    summary="Export memories as .omo.json file",
    description="Export memories in OMO JSON file format for download."
)
async def export_omo_file(
    request: Request,
    response: Response,
    memory_ids: str = Query(..., description="Comma-separated list of memory IDs"),
    api_key: Optional[str] = Security(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth),
    session_token: Optional[str] = Security(session_token_header),
    memory_graph: MemoryGraph = Depends(get_memory_graph)
):
    """Export memories as downloadable .omo.json file."""
    # Parse memory IDs
    ids = [id.strip() for id in memory_ids.split(',') if id.strip()]

    # Use the export endpoint
    export_request = OMOExportRequest(memory_ids=ids)
    result = await export_memories_omo(
        request=request,
        export_request=export_request,
        response=response,
        api_key=api_key,
        bearer_token=bearer_token,
        session_token=session_token,
        memory_graph=memory_graph
    )

    if result.status == "error":
        return JSONResponse(
            status_code=result.code,
            content={"error": result.error}
        )

    # Return as downloadable JSON file
    return Response(
        content=json.dumps(result.memories, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=memories.omo.json"
        }
    )
