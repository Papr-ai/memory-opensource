"""
WebSocket routes for real-time document processing status updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import Optional
import json
import logging
from datetime import datetime

from services.auth_utils import get_user_from_token_optimized
from services.multi_tenant_utils import extract_multi_tenant_context
from core.document_processing.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/document-status")
async def websocket_document_status(
    websocket: WebSocket,
    token: str = Query(...),
    namespace: Optional[str] = Query(None)
):
    """WebSocket endpoint for real-time document processing status"""

    try:
        # Authenticate WebSocket connection
        auth_response = await get_user_from_token_optimized(f"Bearer {token}", "websocket")
        auth_context = extract_multi_tenant_context(auth_response)

        organization_id = auth_context.get('organization_id', 'default')
        namespace_id = namespace or auth_context.get('namespace_id')

        # Handle WebSocket connection
        ws_manager = get_websocket_manager()
        await ws_manager.handle_websocket(websocket, organization_id, namespace_id)

    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        await websocket.close(code=1008, reason="Authentication failed")


@router.websocket("/document-status/{upload_id}")
async def websocket_specific_document(
    websocket: WebSocket,
    upload_id: str,
    token: str = Query(...),
    namespace: Optional[str] = Query(None)
):
    """WebSocket endpoint for specific document upload status"""

    try:
        # Authenticate WebSocket connection
        auth_response = await get_user_from_token_optimized(f"Bearer {token}", "websocket")
        auth_context = extract_multi_tenant_context(auth_response)

        organization_id = auth_context.get('organization_id', 'default')
        namespace_id = namespace or auth_context.get('namespace_id')

        await websocket.accept()

        # Send initial subscription confirmation
        await websocket.send_text(json.dumps({
            "type": "subscription_confirmed",
            "upload_id": upload_id,
            "organization_id": organization_id,
            "namespace_id": namespace_id,
            "timestamp": datetime.now().isoformat()
        }))

        # Subscribe to updates for this specific upload
        ws_manager = get_websocket_manager()

        try:
            while True:
                # Keep connection alive and handle client messages
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "upload_id": upload_id,
                        "timestamp": datetime.now().isoformat()
                    }))

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for upload_id: {upload_id}")

    except Exception as e:
        logger.error(f"WebSocket error for upload {upload_id}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal error")
        except:
            pass