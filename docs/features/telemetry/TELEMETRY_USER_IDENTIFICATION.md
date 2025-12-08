# Telemetry User Identification Guide

## Overview

The telemetry system tracks events differently based on **edition** and distinguishes between **developers** (API key owners) and **end users** (the developer's customers).

---

## Key Concepts

### 1. Developer vs End User

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Cloud Service                    │
│                                                              │
│  Developer A (API Key: abc123)                              │
│  ├── End User 1 (Alice)   → Creates memories               │
│  ├── End User 2 (Bob)     → Creates memories               │
│  └── End User 3 (Charlie) → Creates memories               │
│                                                              │
│  Developer B (API Key: xyz789)                              │
│  ├── End User 4 (Diana)   → Creates memories               │
│  └── End User 5 (Eve)     → Creates memories               │
└─────────────────────────────────────────────────────────────┘
```

**Developer** = The person/company who:
- Owns the API key
- Owns the workspace
- Pays for the subscription
- Has multiple end users

**End User** = The developer's customer who:
- Uses the developer's application
- Creates memories through the developer's API key
- Doesn't directly interact with your service

---

## How User IDs Flow Through the System

### Request Flow

```
1. Request comes in with authentication
   ├── API Key: "abc123" (identifies Developer A)
   └── End User ID: "user_1234" (from developer's app)

2. Auth system extracts both IDs:
   ├── developer_user_id = "dev_abc123" (from API key lookup)
   └── end_user_id = "user_1234" (from request metadata)

3. Telemetry receives both:
   await telemetry.track(
       "memory_created",
       properties={...},
       user_id=end_user_id,        # "user_1234"
       developer_id=developer_user_id  # "dev_abc123"
   )

4. TelemetryService decides what to track:
   
   Cloud Mode (Amplitude):
   ├── Primary ID: developer_id ("dev_abc123")
   └── Properties: {end_user_id: "user_1234", ...}
   
   OSS Mode (PostHog):
   ├── Primary ID: hash(user_id or developer_id)
   └── Properties: {<anonymized>}
```

---

## Code Examples

### Example 1: Memory Creation (Cloud)

```python
# In memory_service.py
async def handle_incoming_memory(
    memory_request: AddMemoryRequest,
    end_user_id: str,           # "user_1234" (developer's customer)
    developer_user_id: str,     # "dev_abc123" (API key owner)
    sessionToken: str,
    ...
):
    # Track memory creation
    telemetry = get_telemetry()
    await telemetry.track(
        "memory_created", 
        {
            "type": "text",
            "has_metadata": True,
        }, 
        user_id=end_user_id,        # The developer's end user
        developer_id=developer_user_id  # The API key owner
    )
    
    # Result in Amplitude (Cloud):
    # User ID: "dev_abc123"
    # Properties: {
    #   "end_user_id": "user_1234",
    #   "developer_id": "dev_abc123",
    #   "type": "text",
    #   "has_metadata": true,
    #   "email": "developer@company.com",
    #   "country": "US"
    # }
```

### Example 2: Analytics Query (Cloud)

In Amplitude, you can now ask:

```
"Show me all events for Developer A"
→ Filter by user_id = "dev_abc123"
→ See all their end users' activities

"How many unique end users did Developer A have?"
→ Count distinct end_user_id where user_id = "dev_abc123"

"Which developer has the most active end users?"
→ Group by developer_id, count distinct end_user_id
```

---

## Where IDs Come From

### 1. API Key Authentication

```python
# In auth_utils.py (simplified)
async def get_user_from_token_optimized(...):
    if api_key:
        # Look up who owns this API key
        api_key_doc = await mongo.api_keys.find_one({"key": api_key})
        developer_user_id = api_key_doc["userId"]  # "dev_abc123"
        
        # Get end user from request metadata
        end_user_id = memory_request.metadata.external_user_id  # "user_1234"
        
        return OptimizedAuthResponse(
            developer_id=developer_user_id,
            end_user_id=end_user_id,
            ...
        )
```

### 2. Session Token Authentication

```python
# In auth_utils.py (simplified)
async def get_user_from_token_optimized(...):
    if session_token:
        # Parse Server session token
        user_info = await verify_session_token(session_token)
        user_id = user_info["objectId"]  # "user_1234"
        
        # In this case, user IS the developer
        return OptimizedAuthResponse(
            developer_id=user_id,
            end_user_id=user_id,
            ...
        )
```

### 3. External User ID (Developer's System)

```python
# Developer's request:
POST /v1/memories
Headers:
  X-API-Key: abc123
Body:
  {
    "content": "Meeting notes",
    "metadata": {
      "external_user_id": "alice_from_dev_system"  # Developer's user ID
    }
  }

# You resolve this to your internal ID:
async def resolve_external_user_ids_to_internal(...):
    # Look up in your database
    internal_user = await mongo.users.find_one({
        "developer_id": developer_id,
        "external_id": "alice_from_dev_system"
    })
    
    return internal_user["objectId"]  # "user_1234"
```

---

## Telemetry Logic Flow

### Cloud Mode (Amplitude)

```python
# In core/services/telemetry.py
async def track(self, event_name, properties, user_id, developer_id):
    edition = os.getenv("PAPR_EDITION", "opensource")
    
    if edition == "cloud" and self.provider == TelemetryProvider.AMPLITUDE:
        # Track by developer, include end user in properties
        tracking_user_id = developer_id  # "dev_abc123"
        safe_properties = properties or {}
        
        if user_id and developer_id:
            safe_properties['end_user_id'] = user_id  # "user_1234"
        
        if developer_id:
            safe_properties['developer_id'] = developer_id
        
        # Send to Amplitude
        await amplitude_client.track(
            user_id=tracking_user_id,  # Primary: developer
            event_type=event_name,
            event_properties=safe_properties
        )
```

**Result in Amplitude**:
```json
{
  "user_id": "dev_abc123",
  "event_type": "memory_created",
  "event_properties": {
    "end_user_id": "user_1234",
    "developer_id": "dev_abc123",
    "type": "text",
    "email": "developer@company.com",
    "country": "US",
    "client_type": "api"
  }
}
```

### OSS Mode (PostHog)

```python
# In core/services/telemetry.py
async def track(self, event_name, properties, user_id, developer_id):
    edition = os.getenv("PAPR_EDITION", "opensource")
    
    if edition == "opensource":
        # Anonymize everything
        safe_properties = self._anonymize_properties(properties)
        
        # Hash the user ID for privacy
        tracking_user_id = self._hash_user_id(
            user_id or developer_id
        ) if (user_id or developer_id) else self.anonymous_id
        
        # Send to PostHog
        await posthog_client.capture(
            distinct_id=tracking_user_id,  # Hashed
            event=event_name,
            properties=safe_properties  # Anonymized
        )
```

**Result in PostHog**:
```json
{
  "distinct_id": "sha256:a1b2c3d4...",  // Hashed
  "event": "memory_created",
  "properties": {
    "type": "text",
    "has_metadata": true,
    "edition": "opensource",
    "os": "Linux",
    "version": "1.0.0"
    // No PII, no user IDs, no emails
  }
}
```

---

## Summary Table

| Scenario | Primary Tracking ID | Additional Context | Use Case |
|----------|--------------------|--------------------|----------|
| **Cloud - API Key** | `developer_id` (API key owner) | `end_user_id` in properties | Track developer's customers |
| **Cloud - Session Token** | `developer_id` = `user_id` | Same ID for both | Developer using their own account |
| **Cloud - External User** | `developer_id` | Resolved `end_user_id` | Developer's system → Your system |
| **OSS - Any Auth** | `hash(user_id or developer_id)` | Anonymized properties | Privacy-first tracking |

---

## Analytics Queries You Can Run

### Cloud Mode (Amplitude)

```sql
-- How many memories did Developer A create (across all their users)?
SELECT COUNT(*) 
WHERE user_id = 'dev_abc123'
AND event_type = 'memory_created'

-- Which developer has the most active end users?
SELECT developer_id, COUNT(DISTINCT end_user_id) as unique_users
GROUP BY developer_id
ORDER BY unique_users DESC

-- How many memories per end user for Developer A?
SELECT end_user_id, COUNT(*) as memory_count
WHERE developer_id = 'dev_abc123'
AND event_type = 'memory_created'
GROUP BY end_user_id

-- Developer retention over time
SELECT developer_id, DATE(timestamp) as date, COUNT(DISTINCT end_user_id)
GROUP BY developer_id, date
ORDER BY date
```

### OSS Mode (PostHog)

```sql
-- Total usage (anonymous)
SELECT event, COUNT(*) as count
GROUP BY event
ORDER BY count DESC

-- Edition distribution
SELECT properties.edition, COUNT(*) as count
GROUP BY properties.edition

-- OS distribution
SELECT properties.os, COUNT(*) as count
GROUP BY properties.os

-- NO user-level queries (privacy!)
```

---

## Best Practices

### ✅ DO

1. **Always pass both IDs** when available:
   ```python
   await telemetry.track("event", {...}, 
                        user_id=end_user_id,
                        developer_id=developer_user_id)
   ```

2. **Use developer_id for cloud analytics**:
   - Track developer behavior
   - Measure product adoption
   - Identify power users

3. **Include end_user_id in properties** (cloud only):
   - Multi-user tracking
   - End user activity patterns
   - Developer's customer insights

4. **Trust the anonymization** (OSS):
   - TelemetryService handles it
   - No manual filtering needed
   - Respects user privacy

### ❌ DON'T

1. **Don't swap the IDs**:
   ```python
   # WRONG
   await telemetry.track("event", {...},
                        user_id=developer_user_id,  # ❌
                        developer_id=end_user_id)   # ❌
   ```

2. **Don't include PII in custom properties** (OSS):
   ```python
   # WRONG for OSS
   properties = {
       "user_email": "alice@example.com",  # ❌ Filtered anyway
       "user_name": "Alice"                # ❌ Filtered anyway
   }
   ```

3. **Don't rely on telemetry for auth**:
   - Telemetry is fire-and-forget
   - Failures are silent
   - Only for analytics, not security

---

## Debugging

### Check What's Being Tracked

```python
# Add to telemetry.py for debugging
async def track(self, event_name, properties, user_id, developer_id):
    edition = os.getenv("PAPR_EDITION", "opensource")
    
    # DEBUG: Log what's being tracked
    logger.debug(f"Telemetry: {edition} mode")
    logger.debug(f"  Event: {event_name}")
    logger.debug(f"  User ID: {user_id}")
    logger.debug(f"  Developer ID: {developer_id}")
    logger.debug(f"  Properties: {properties}")
    
    # ... rest of the method
```

### Verify in Amplitude/PostHog

**Cloud (Amplitude)**:
1. Go to Amplitude dashboard
2. Click "User Lookup"
3. Search for `developer_id`
4. See all events for that developer
5. Check `end_user_id` in properties

**OSS (PostHog)**:
1. Go to PostHog dashboard
2. Click "Events"
3. See aggregated anonymous events
4. NO user-level data visible

---

## Migration Checklist

When migrating from old Amplitude code:

- [x] Replace `amplitude_client.track(user_id=X)` with `telemetry.track(..., user_id=X, developer_id=Y)`
- [x] Extract `developer_id` from `user_info` or `extra_properties`
- [x] Pass both IDs to all `telemetry.track()` calls
- [x] Set `PAPR_EDITION=cloud` for production
- [x] Set `PAPR_EDITION=opensource` for OSS distribution
- [x] Test both modes work correctly
- [x] Verify Amplitude shows developer-level analytics
- [x] Verify PostHog shows anonymous analytics

---

**Key Takeaway**: 
- **Cloud**: Track by `developer_id`, include `end_user_id` for multi-user insights
- **OSS**: Everything anonymized automatically for privacy
- **No code changes needed** when switching editions - TelemetryService handles it!

