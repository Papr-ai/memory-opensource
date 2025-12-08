import asyncio
import json
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import os

# Optional Azure Service Bus imports
try:
    from azure.servicebus import ServiceBusClient, ServiceBusReceiver
    from azure.servicebus.exceptions import ServiceBusError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Azure Service Bus not available. Install with: pip install azure-servicebus")

logger = logging.getLogger(__name__)

class AzureWebhookConsumer:
    """Consumer for processing webhook messages from Azure Service Bus"""
    
    def __init__(self):
        self.connection_string = os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
        self.queue_name = os.getenv("AZURE_SERVICE_BUS_QUEUE_NAME", "batch-notifications")
        self.max_retries = int(os.getenv("WEBHOOK_MAX_RETRIES", "5"))
        self.retry_delay = float(os.getenv("WEBHOOK_RETRY_DELAY", "1.0"))
        self.max_processing_time = int(os.getenv("WEBHOOK_MAX_PROCESSING_TIME", "300"))  # 5 minutes
        
    async def start_consumer(self):
        """Start the webhook consumer"""
        if not AZURE_AVAILABLE:
            logger.error("Azure Service Bus not available. Install with: pip install azure-servicebus")
            return
            
        if not self.connection_string:
            logger.error("Azure Service Bus connection string not configured")
            return
            
        logger.info("Starting Azure webhook consumer...")
        
        async with ServiceBusClient.from_connection_string(self.connection_string) as client:
            async with client.get_queue_receiver(self.queue_name) as receiver:
                async for message in receiver:
                    try:
                        await self._process_webhook_message(message)
                        await receiver.complete_message(message)
                    except Exception as e:
                        logger.error(f"Error processing webhook message: {e}")
                        await receiver.dead_letter_message(message, reason=str(e))
    
    async def _process_webhook_message(self, message):
        """Process a single webhook message"""
        try:
            # Parse message data
            message_data = json.loads(str(message))
            webhook_url = message_data["webhook_url"]
            webhook_secret = message_data.get("webhook_secret")
            payload = message_data["payload"]
            retry_count = message_data.get("retry_count", 0)
            
            logger.info(f"Processing webhook message (attempt {retry_count + 1}): {webhook_url}")
            
            # Send webhook
            success = await self._send_webhook_with_retry(
                webhook_url, webhook_secret, payload, retry_count
            )
            
            if success:
                logger.info(f"Webhook sent successfully: {webhook_url}")
            else:
                logger.error(f"Failed to send webhook after retries: {webhook_url}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid message format: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing webhook message: {e}")
            raise
    
    async def _send_webhook_with_retry(
        self,
        webhook_url: str,
        webhook_secret: Optional[str],
        payload: Dict[str, Any],
        current_retry: int
    ) -> bool:
        """Send webhook with retry logic"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PaprMemory-AzureWebhook/1.0"
        }
        
        if webhook_secret:
            headers["X-Webhook-Secret"] = webhook_secret
        
        # Add standard webhook headers
        headers.update({
            "X-Webhook-Event": "batch.completed",
            "X-Webhook-Timestamp": str(int(datetime.now(timezone.utc).timestamp())),
            "X-Webhook-Retry-Count": str(current_retry + 1)
        })
        
        # Calculate remaining retries
        remaining_retries = self.max_retries - current_retry
        
        for attempt in range(remaining_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code in [200, 201, 202]:
                        return True
                    else:
                        logger.warning(
                            f"Webhook returned status {response.status_code}: {response.text}"
                        )
                        
            except httpx.TimeoutException:
                logger.warning(f"Webhook timeout (attempt {attempt + 1}/{remaining_retries})")
            except httpx.RequestError as e:
                logger.warning(f"Webhook request error (attempt {attempt + 1}/{remaining_retries}): {e}")
            except Exception as e:
                logger.error(f"Unexpected webhook error: {e}")
            
            if attempt < remaining_retries - 1:
                # Exponential backoff with jitter
                delay = self.retry_delay * (2 ** attempt) * (0.5 + 0.5 * asyncio.get_event_loop().time() % 1)
                await asyncio.sleep(delay)
        
        return False

async def main():
    """Main function to run the webhook consumer"""
    consumer = AzureWebhookConsumer()
    await consumer.start_consumer()

if __name__ == "__main__":
    asyncio.run(main()) 