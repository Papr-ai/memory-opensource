# Webhook Guide for Batch Memory Processing

This guide explains how to use webhooks with the Papr Memory batch processing endpoint to receive notifications when batch operations complete.

## Overview

Webhooks allow you to receive real-time notifications when batch memory processing completes, eliminating the need to poll for status updates. The system supports both direct HTTP webhooks and Azure Service Bus for enterprise-grade reliability.

## Features

- **Real-time notifications** when batch processing completes
- **Multiple delivery methods**: Direct HTTP or Azure Service Bus
- **Authentication** with webhook secrets and HMAC signatures
- **Retry logic** with exponential backoff
- **Comprehensive payload** with detailed batch results
- **Enterprise reliability** with Azure Service Bus integration

## Quick Start

### 1. Basic Webhook Setup

Add webhook configuration to your batch request:

```json
{
  "memories": [
    {
      "content": "Meeting notes from the product planning session",
      "type": "text",
      "metadata": {
        "topics": "product, planning"
      }
    }
  ],
  "webhook_url": "https://your-endpoint.com/webhook",
  "webhook_secret": "your-secret-key"
}
```

### 2. Handle Webhook Notifications

Your webhook endpoint will receive POST requests with this structure:

```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "status": "completed",
  "total_memories": 5,
  "successful_memories": 5,
  "failed_memories": 0,
  "errors": [],
  "completed_at": "2024-03-21T15:30:45.123Z",
  "processing_time_ms": 1250,
  "memory_ids": ["mem_001", "mem_002", "mem_003"],
  "webhook_version": "1.0"
}
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_SERVICE_BUS_CONNECTION_STRING` | Azure Service Bus connection string | None |
| `AZURE_SERVICE_BUS_QUEUE_NAME` | Azure Service Bus queue name | `batch-notifications` |
| `WEBHOOK_MAX_RETRIES` | Maximum retry attempts | `3` |
| `WEBHOOK_RETRY_DELAY` | Base retry delay in seconds | `1.0` |

### Webhook Headers

The system sends these headers with each webhook:

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` |
| `X-Webhook-Secret` | Your webhook secret (if provided) |
| `X-Webhook-Signature` | HMAC-SHA256 signature of payload |
| `X-Webhook-Event` | `batch.completed` |
| `X-Webhook-Timestamp` | Unix timestamp |
| `User-Agent` | `PaprMemory-BatchWebhook/1.0` |

## Delivery Methods

### 1. Direct HTTP Webhooks

**Pros:**
- Simple setup
- No external dependencies
- Immediate delivery

**Cons:**
- No delivery guarantees
- Limited retry logic
- Can be unreliable for critical notifications

**Best for:** Simple integrations, development/testing

### 2. Azure Service Bus (Recommended for Production)

**Pros:**
- Guaranteed delivery
- Automatic retry with exponential backoff
- Dead letter queue for failed webhooks
- Enterprise-grade reliability
- Built-in monitoring and analytics

**Cons:**
- Requires Azure subscription
- More complex setup

**Best for:** Production applications, enterprise use cases

## Webhook Payload Reference

### Success Response

```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "status": "completed",
  "total_memories": 5,
  "successful_memories": 5,
  "failed_memories": 0,
  "errors": [],
  "completed_at": "2024-03-21T15:30:45.123Z",
  "processing_time_ms": 1250,
  "memory_ids": ["mem_001", "mem_002", "mem_003"],
  "webhook_version": "1.0"
}
```

### Partial Success Response

```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "status": "partial",
  "total_memories": 5,
  "successful_memories": 3,
  "failed_memories": 2,
  "errors": [
    {
      "index": 1,
      "error": "Content size exceeds limit"
    },
    {
      "index": 3,
      "error": "Invalid memory type"
    }
  ],
  "completed_at": "2024-03-21T15:30:45.123Z",
  "processing_time_ms": 1250,
  "memory_ids": ["mem_001", "mem_002", "mem_003"],
  "webhook_version": "1.0"
}
```

### Failure Response

```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "status": "failed",
  "total_memories": 5,
  "successful_memories": 0,
  "failed_memories": 5,
  "errors": [
    {
      "index": -1,
      "error": "All batch items failed"
    }
  ],
  "completed_at": "2024-03-21T15:30:45.123Z",
  "processing_time_ms": 1250,
  "memory_ids": [],
  "webhook_version": "1.0"
}
```

## Authentication

### Webhook Secret

If you provide a `webhook_secret`, the system will:

1. Include it in the `X-Webhook-Secret` header
2. Generate an HMAC-SHA256 signature of the payload
3. Include the signature in the `X-Webhook-Signature` header

### Verifying Signatures

```python
import hmac
import hashlib
import json

def verify_webhook_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify webhook signature"""
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)
```

## Azure Service Bus Setup

### 1. Create Azure Service Bus Namespace

```bash
az servicebus namespace create \
  --name your-namespace \
  --resource-group your-resource-group \
  --location eastus \
  --sku Standard
```

### 2. Create Queue

```bash
az servicebus queue create \
  --namespace-name your-namespace \
  --resource-group your-resource-group \
  --name batch-notifications
```

### 3. Get Connection String

```bash
az servicebus namespace authorization-rule keys list \
  --namespace-name your-namespace \
  --resource-group your-resource-group \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString \
  --output tsv
```

### 4. Set Environment Variables

```bash
export AZURE_SERVICE_BUS_CONNECTION_STRING="your-connection-string"
export AZURE_SERVICE_BUS_QUEUE_NAME="batch-notifications"
```

## Error Handling

### HTTP Status Codes

Your webhook endpoint should return:

- `200`, `201`, or `202` for success
- Any other status code will trigger retries

### Retry Logic

- **Direct HTTP**: Up to 3 retries with exponential backoff
- **Azure Service Bus**: Up to 5 retries with exponential backoff and jitter
- Failed webhooks are moved to dead letter queue (Azure only)

### Dead Letter Queue

When using Azure Service Bus, failed webhooks are automatically moved to a dead letter queue where you can:

1. Inspect failed messages
2. Reprocess them manually
3. Analyze failure patterns

## Best Practices

### 1. Idempotency

Make your webhook handlers idempotent - they should be safe to call multiple times with the same data.

### 2. Quick Response

Respond quickly to webhook requests (within 5 seconds) to avoid timeouts.

### 3. Logging

Log all webhook events for debugging and monitoring.

### 4. Error Handling

Handle webhook errors gracefully and provide meaningful error responses.

### 5. Security

- Use HTTPS for webhook URLs
- Validate webhook signatures
- Use strong webhook secrets
- Rate limit webhook endpoints

## Monitoring

### Azure Service Bus Metrics

Monitor these Azure Service Bus metrics:

- **Message Count**: Number of messages in queue
- **Dead Letter Message Count**: Failed webhooks
- **Server Errors**: Service Bus errors
- **Throttled Requests**: Rate limiting events

### Application Logs

Monitor these log entries:

- `Webhook sent successfully for batch {batch_id}`
- `Failed to send webhook for batch {batch_id}`
- `Error sending webhook notification: {error}`

## Troubleshooting

### Common Issues

1. **Webhook not received**
   - Check webhook URL is accessible
   - Verify firewall/network settings
   - Check webhook endpoint logs

2. **Authentication failures**
   - Verify webhook secret matches
   - Check signature verification logic
   - Ensure proper encoding

3. **Azure Service Bus issues**
   - Verify connection string
   - Check queue exists and is accessible
   - Monitor Azure Service Bus metrics

### Debug Mode

Enable debug logging by setting:

```bash
export LOG_LEVEL=DEBUG
```

This will log detailed webhook delivery information.

## Examples

See `examples/webhook_example.py` for complete working examples.

## Support

For webhook-related issues:

1. Check the application logs
2. Verify webhook endpoint is accessible
3. Test with a simple webhook endpoint first
4. Contact support with batch ID and error details 