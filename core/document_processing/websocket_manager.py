"""
WebSocket management for real-time document processing status updates
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
import json
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        # Connection pools organized by organization and namespace
        self.connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    def _get_key(self, organization_id: str, namespace_id: Optional[str] = None) -> str:
        """Generate connection key"""
        return f"{organization_id}:{namespace_id or 'default'}"

    async def connect(
        self,
        websocket: WebSocket,
        organization_id: str,
        namespace_id: Optional[str] = None
    ):
        """Connect a WebSocket for an organization/namespace"""
        await websocket.accept()

        async with self._lock:
            key = self._get_key(organization_id, namespace_id)
            if key not in self.connections:
                self.connections[key] = []

            self.connections[key].append(websocket)

        logger.info(f"WebSocket connected for {organization_id}:{namespace_id}")

    async def disconnect(
        self,
        websocket: WebSocket,
        organization_id: str,
        namespace_id: Optional[str] = None
    ):
        """Disconnect a WebSocket"""
        async with self._lock:
            key = self._get_key(organization_id, namespace_id)
            if key in self.connections:
                try:
                    self.connections[key].remove(websocket)
                    if not self.connections[key]:
                        del self.connections[key]
                except ValueError:
                    pass  # Connection already removed

        logger.info(f"WebSocket disconnected for {organization_id}:{namespace_id}")

    async def broadcast_to_organization(
        self,
        message: Dict[str, Any],
        organization_id: str,
        namespace_id: Optional[str] = None
    ):
        """Broadcast message to all connections for an organization/namespace"""
        key = self._get_key(organization_id, namespace_id)

        if key not in self.connections:
            return

        connections = self.connections[key].copy()
        message_str = json.dumps(message, default=str)

        disconnected = []
        for connection in connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)

        # Remove dead connections
        if disconnected:
            async with self._lock:
                for connection in disconnected:
                    try:
                        self.connections[key].remove(connection)
                    except ValueError:
                        pass


class WebSocketManager:
    """High-level WebSocket management for document processing"""

    def __init__(self):
        self.connection_manager = ConnectionManager()

    async def handle_websocket(
        self,
        websocket: WebSocket,
        organization_id: str,
        namespace_id: Optional[str] = None
    ):
        """Handle WebSocket connection lifecycle"""
        await self.connection_manager.connect(websocket, organization_id, namespace_id)

        try:
            while True:
                # Keep connection alive and handle client messages
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get("type") == "subscribe":
                    # Handle subscription to specific upload IDs
                    upload_id = message.get("upload_id")
                    if upload_id:
                        await self._handle_subscription(websocket, upload_id, organization_id, namespace_id)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self.connection_manager.disconnect(websocket, organization_id, namespace_id)

    async def broadcast_status_update(
        self,
        update: Dict[str, Any],
        organization_id: str,
        namespace_id: Optional[str] = None
    ):
        """Broadcast document processing status update"""
        message = {
            "type": "document_status_update",
            "data": update,
            "timestamp": datetime.now().isoformat()
        }

        await self.connection_manager.broadcast_to_organization(
            message,
            organization_id,
            namespace_id
        )

    async def _handle_subscription(
        self,
        websocket: WebSocket,
        upload_id: str,
        organization_id: str,
        namespace_id: Optional[str]
    ):
        """Handle subscription to specific upload ID"""
        # Send current status if available
        # This would query the current status from storage/cache
        try:
            await websocket.send_text(json.dumps({
                "type": "subscription_confirmed",
                "upload_id": upload_id,
                "timestamp": datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error(f"Failed to confirm subscription: {e}")


# Global WebSocket manager instance
_websocket_manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager instance"""
    return _websocket_manager