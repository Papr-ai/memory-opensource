import asyncio
import httpx
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import os

# Optional Azure Service Bus imports
try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage
    from azure.servicebus.exceptions import ServiceBusError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Azure Service Bus not available. Install with: pip install azure-servicebus")

logger = logging.getLogger(__name__)

class WebhookService:
    """Service for sending webhook notifications with support for multiple backends"""
    
    def __init__(self):
        self.azure_connection_string = os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
        self.azure_queue_name = os.getenv("AZURE_SERVICE_BUS_QUEUE_NAME", "batch-notifications")
        self.max_retries = int(os.getenv("WEBHOOK_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("WEBHOOK_RETRY_DELAY", "1.0"))
        
    async def send_batch_completion_webhook(
        self,
        webhook_url: str,
        webhook_secret: Optional[str],
        batch_data: Dict[str, Any],
        use_azure: bool = False
    ) -> bool:
        """
        Send batch completion webhook notification
        
        Args:
            webhook_url: The webhook URL to send to
            webhook_secret: Optional secret for authentication
            batch_data: The batch completion data
            use_azure: Whether to use Azure Service Bus for reliable delivery
            
        Returns:
            bool: True if webhook was sent successfully
        """
        if use_azure and self.azure_connection_string and AZURE_AVAILABLE:
            return await self._send_via_azure(webhook_url, webhook_secret, batch_data)
        elif use_azure and not AZURE_AVAILABLE:
            logger.warning("Azure Service Bus requested but not available. Falling back to HTTP webhook.")
            return await self._send_via_http(webhook_url, webhook_secret, batch_data)
        else:
            return await self._send_via_http(webhook_url, webhook_secret, batch_data)
    
    async def _send_via_azure(
        self,
        webhook_url: str,
        webhook_secret: Optional[str],
        batch_data: Dict[str, Any]
    ) -> bool:
        """Send webhook via Azure Service Bus for reliable delivery"""
        try:
            # Create message with webhook details
            message_data = {
                "webhook_url": webhook_url,
                "webhook_secret": webhook_secret,
                "payload": batch_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0
            }
            
            # Send to Azure Service Bus
            async with ServiceBusClient.from_connection_string(
                self.azure_connection_string
            ) as client:
                async with client.get_queue_sender(self.azure_queue_name) as sender:
                    message = ServiceBusMessage(
                        json.dumps(message_data),
                        content_type="application/json"
                    )
                    await sender.send_messages(message)
                    
            logger.info(f"Webhook queued to Azure Service Bus: {webhook_url}")
            return True
            
        except ServiceBusError as e:
            logger.error(f"Failed to send webhook to Azure Service Bus: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook to Azure: {e}")
            return False
    
    async def _send_via_http(
        self,
        webhook_url: str,
        webhook_secret: Optional[str],
        batch_data: Dict[str, Any]
    ) -> bool:
        """Send webhook via direct HTTP request"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PaprMemory-BatchWebhook/1.0"
        }
        
        if webhook_secret:
            headers["X-Webhook-Secret"] = webhook_secret
        
        # Add standard webhook headers
        headers.update({
            "X-Webhook-Event": "batch.completed",
            "X-Webhook-Timestamp": str(int(datetime.now(timezone.utc).timestamp())),
            "X-Webhook-Signature": self._generate_signature(batch_data, webhook_secret) if webhook_secret else ""
        })
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        webhook_url,
                        json=batch_data,
                        headers=headers
                    )
                    
                    if response.status_code in [200, 201, 202]:
                        logger.info(f"Webhook sent successfully to {webhook_url}")
                        return True
                    else:
                        logger.warning(
                            f"Webhook returned status {response.status_code}: {response.text}"
                        )
                        
            except httpx.TimeoutException:
                logger.warning(f"Webhook timeout (attempt {attempt + 1}/{self.max_retries})")
            except httpx.RequestError as e:
                logger.warning(f"Webhook request error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.error(f"Unexpected webhook error: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"Failed to send webhook after {self.max_retries} attempts: {webhook_url}")
        return False
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook authentication"""
        import hmac
        import hashlib
        
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def create_batch_webhook_payload(
        self,
        batch_id: str,
        user_id: str,
        status: str,
        total_memories: int,
        successful_memories: int,
        failed_memories: int,
        errors: List[Dict[str, Any]],
        memory_ids: List[str],
        processing_time_ms: int
    ) -> Dict[str, Any]:
        """Create standardized webhook payload for batch completion"""
        return {
            "batch_id": batch_id,
            "user_id": user_id,
            "status": status,  # "completed", "failed", "partial"
            "total_memories": total_memories,
            "successful_memories": successful_memories,
            "failed_memories": failed_memories,
            "errors": errors,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": processing_time_ms,
            "memory_ids": memory_ids,
            "webhook_version": "1.0"
        }

# Global webhook service instance
webhook_service = WebhookService() 