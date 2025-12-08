#!/usr/bin/env python3
"""
Example script demonstrating webhook usage with the batch memory endpoint.

This script shows how to:
1. Send a batch request with webhook configuration
2. Handle webhook notifications
3. Use Azure Service Bus for reliable delivery
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, Any

# Example webhook server (you would implement this in your application)
class WebhookServer:
    """Simple webhook server to receive batch completion notifications"""
    
    def __init__(self):
        self.received_webhooks = []
    
    async def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook"""
        print(f"📨 Received webhook at {datetime.now()}")
        print(f"📋 Webhook payload: {json.dumps(payload, indent=2)}")
        
        # Store the webhook for later analysis
        self.received_webhooks.append({
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        })
        
        # Return success response
        return {"status": "received", "message": "Webhook processed successfully"}

# Example batch request with webhook
EXAMPLE_BATCH_REQUEST = {
    "memories": [
        {
            "content": "Meeting notes from the product planning session",
            "type": "text",
            "metadata": {
                "topics": "product, planning",
                "hierarchical_structures": "Business/Planning/Product",
                "createdAt": "2024-03-21T10:00:00Z",
                "emoji_tags": "📊,💡,📝",
                "emotion_tags": "focused, productive"
            }
        },
        {
            "content": "Follow-up tasks from the planning meeting",
            "type": "text",
            "metadata": {
                "topics": "tasks, planning",
                "hierarchical_structures": "Business/Tasks/Planning",
                "createdAt": "2024-03-21T11:00:00Z",
                "emoji_tags": "✅,📋",
                "emotion_tags": "organized"
            }
        }
    ],
    "batch_size": 10,
    "webhook_url": "https://your-webhook-endpoint.com/webhook",
    "webhook_secret": "your-webhook-secret-key"
}

async def send_batch_with_webhook():
    """Send a batch request with webhook configuration"""
    
    # Your API endpoint
    api_url = "https://your-api-domain.com/v1/memory/batch"
    
    # Your authentication headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer your-auth-token",
        "X-Client-Type": "papr_plugin"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                json=EXAMPLE_BATCH_REQUEST,
                headers=headers
            )
            
            if response.status_code in [200, 207]:
                print("✅ Batch request sent successfully")
                print(f"📊 Response: {response.json()}")
                
                # The webhook will be sent automatically when processing completes
                print("🔄 Webhook will be sent when batch processing completes")
            else:
                print(f"❌ Batch request failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error sending batch request: {e}")

async def start_webhook_server():
    """Start a simple webhook server to receive notifications"""
    from fastapi import FastAPI, Request
    import uvicorn
    
    app = FastAPI()
    webhook_server = WebhookServer()
    
    @app.post("/webhook")
    async def webhook_endpoint(request: Request):
        """Webhook endpoint to receive batch completion notifications"""
        payload = await request.json()
        return await webhook_server.handle_webhook(payload)
    
    @app.get("/webhooks")
    async def get_webhooks():
        """Get all received webhooks"""
        return {"webhooks": webhook_server.received_webhooks}
    
    # Start the server
    config = uvicorn.Config(app, host="0.0.0.0", port=8001)
    server = uvicorn.Server(config)
    await server.serve()

def main():
    """Main function demonstrating webhook usage"""
    print("🚀 Webhook Example for Papr Memory Batch API")
    print("=" * 50)
    
    print("\n📋 Example batch request with webhook:")
    print(json.dumps(EXAMPLE_BATCH_REQUEST, indent=2))
    
    print("\n🔧 Environment Variables needed:")
    print("AZURE_SERVICE_BUS_CONNECTION_STRING=your-azure-connection-string")
    print("AZURE_SERVICE_BUS_QUEUE_NAME=batch-notifications")
    print("WEBHOOK_MAX_RETRIES=3")
    print("WEBHOOK_RETRY_DELAY=1.0")
    
    print("\n📨 Webhook payload structure:")
    webhook_payload_example = {
        "batch_id": "uuid-string",
        "user_id": "user-123",
        "status": "completed",  # or "failed", "partial"
        "total_memories": 5,
        "successful_memories": 5,
        "failed_memories": 0,
        "errors": [],
        "completed_at": "2024-03-21T15:30:45.123Z",
        "processing_time_ms": 1250,
        "memory_ids": ["mem_001", "mem_002", "mem_003"],
        "webhook_version": "1.0"
    }
    print(json.dumps(webhook_payload_example, indent=2))
    
    print("\n🔐 Webhook Authentication:")
    print("- X-Webhook-Secret: Your secret key for authentication")
    print("- X-Webhook-Signature: HMAC-SHA256 signature of the payload")
    print("- X-Webhook-Event: 'batch.completed'")
    print("- X-Webhook-Timestamp: Unix timestamp")
    
    print("\n⚙️ Azure Service Bus Integration:")
    print("- Set AZURE_SERVICE_BUS_CONNECTION_STRING to enable Azure delivery")
    print("- Webhooks are queued in Azure Service Bus for reliable delivery")
    print("- Automatic retry with exponential backoff")
    print("- Dead letter queue for failed webhooks")

if __name__ == "__main__":
    main() 