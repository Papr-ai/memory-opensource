# Automatic Schema Registration

## Problem

Previously, the AgentLearning schema needed to be **manually registered** per organization:

```bash
# Manual registration (old way)
python scripts/register_agent_learning_schema.py --api-key YOUR_API_KEY
```

**Issues:**
- ❌ Users forgot to register the schema
- ❌ Batch processing silently skipped graph generation if schema missing
- ❌ No learnings or structured nodes created without manual setup
- ❌ Extra setup step for each organization

---

## Solution: Auto-Registration on First Use

The AgentLearning schema is now **automatically created** when first needed.

### How It Works

```python
# services/default_schema_initializer.py
async def ensure_agent_learning_schema(user_id, organization_id):
    """Auto-register schema if it doesn't exist"""
    
    # 1. Check if schema exists
    existing_schemas = await schema_service.get_active_schemas(
        user_id=user_id,
        organization_id=organization_id
    )
    
    for schema in existing_schemas:
        if schema.name == "AgentLearning":
            return schema.id  # ✅ Found existing
    
    # 2. Doesn't exist - create it automatically
    logger.info("Auto-registering AgentLearning schema...")
    
    created_schema = await schema_service.create_schema(
        schema_data=AGENT_LEARNING_SCHEMA,
        user_id=user_id,
        organization_id=organization_id
    )
    
    return created_schema.id  # ✅ Newly created
```

### Triggered During Batch Processing

```python
# services/message_batch_analysis.py
async def process_batch_analysis_results(...):
    # When first learning is detected
    if agent_learning_schema_id is None:
        agent_learning_schema_id = await get_agent_learning_schema_id(
            user_id=user_id,
            organization_id=organization_id
        )
        # ↑ This will auto-create if missing!
    
    if agent_learning_schema_id:
        # Create learning nodes with schema
        await add_memory(
            memory_request,
            graph_generation=GraphGeneration(
                mode=AUTO,
                schema_id=agent_learning_schema_id  # ✅ Uses auto-created schema
            )
        )
```

---

## Flow Diagram

### First Message Batch (Schema Doesn't Exist):

```
User sends 15 messages
  ↓
Batch analysis triggered
  ↓
Learning detected → Need AgentLearning schema
  ↓
get_agent_learning_schema_id() called
  ↓
ensure_agent_learning_schema() checks:
  - Schema exists? NO
  ↓
📝 Auto-registers AgentLearning schema
  ↓
Returns schema_id
  ↓
Creates Learning nodes with schema
  ↓
✅ Graph generated successfully!
```

### Subsequent Message Batches:

```
User sends 15 more messages
  ↓
Batch analysis triggered
  ↓
Learning detected → Need AgentLearning schema
  ↓
get_agent_learning_schema_id() called
  ↓
ensure_agent_learning_schema() checks:
  - Schema exists? YES ✅
  ↓
Returns existing schema_id (no creation)
  ↓
Creates Learning nodes with schema
  ↓
✅ Fast lookup, no overhead!
```

---

## Benefits

### ✅ Zero Setup Required

```javascript
// Paprwork - Just send messages!
await paprManager.sendMessage({
  content: "Can you help me build React auth?",
  role: "user",
  sessionId: chatId
});

// After 15 messages → Schema auto-created if needed
// Learnings automatically captured in graph
// No manual registration required!
```

### ✅ Per-Organization Isolation

Each organization gets its own schema:

```cypher
// Organization A
(:AgentLearning {organization_id: "org_a"})

// Organization B  
(:AgentLearning {organization_id: "org_b"})

// Completely isolated!
```

### ✅ Graceful Fallback

If schema creation fails (e.g., permissions), batch processing continues:

```python
schema_id = await get_agent_learning_schema_id(...)

if not schema_id:
    logger.warning("Schema not available, skipping graph generation")
    # Still creates summaries in Parse Server
    # Just no Neo4j nodes
```

### ✅ One-Time Cost

Schema creation only happens **once per organization**:

```
Batch 1 (15 msgs): Auto-create schema (~200ms)
Batch 2 (30 msgs): Use existing schema (~5ms lookup)
Batch 3 (45 msgs): Use existing schema (~5ms lookup)
...
```

---

## Schema Definition

The auto-registered schema includes:

### 9 Node Types:
1. **Learning** - Captured insights
2. **User** - System users
3. **Project** - Software projects
4. **Goal** - Objectives
5. **MessageSession** - Conversations (with title!)
6. **Technology** - React, TypeScript, etc.
7. **Task** - Work items
8. **Person** - Team members
9. **Agent** - AI agents

### 9 Relationship Types:
- `LEARNED_FROM` - Learning → User
- `IN_PROJECT` - Entity → Project
- `USES_TECHNOLOGY` - Project → Technology
- `WORKING_ON` - Person → Task
- `INVOLVES` - Session → Person
- `LED_BY` - Session → Agent
- etc.

**Full definition**: `services/default_schema_initializer.py`

---

## Migration

### For Existing Organizations:

If you **already registered** AgentLearning manually:
- ✅ No action needed
- ✅ Existing schema will be found and reused
- ✅ Auto-registration won't create duplicates

If you **never registered** AgentLearning:
- ✅ No action needed
- ✅ Schema auto-created on first batch (message 15)
- ✅ All future batches use the created schema

---

## Manual Registration (Still Supported)

If you prefer to manually register (e.g., for testing):

```bash
cd /Users/amirkabbara/Documents/GitHub/memory

# Option 1: Using script (old way)
python scripts/register_agent_learning_schema.py --api-key YOUR_API_KEY

# Option 2: Let auto-registration handle it (new way)
# Just send messages - schema created automatically!
```

---

## Verification

### Check if schema was auto-created:

**Via Parse Server Dashboard:**
1. Open Parse Dashboard
2. Go to `UserGraphSchema` class
3. Look for schema with `name: "AgentLearning"`
4. Check `tags` includes `"auto_registered"`

**Via Neo4j:**
```cypher
// Check for Learning nodes
MATCH (l:Learning)
RETURN l.role, l.content, l.learning_type
LIMIT 5

// Check for MessageSession nodes
MATCH (s:MessageSession)
RETURN s.title, s.sessionId, s.tech_stack
LIMIT 5

// Check relationships
MATCH (l:Learning)-[r]->(n)
RETURN type(r), labels(n), count(*) as count
```

**Via Logs:**
```
✅ AgentLearning schema not found for organization org_123, auto-registering...
✅ Auto-registered AgentLearning schema: schema_abc123
✅ AgentLearning schema ready: schema_abc123
```

---

## Error Handling

### Scenario 1: Schema creation fails

```python
# Auto-registration fails (permissions, network, etc.)
logger.error("Failed to auto-register AgentLearning schema")

# Batch processing continues without graph generation:
# - Summaries still created in Parse Server ✅
# - Learning detection still happens ✅
# - Only Neo4j graph generation skipped
```

### Scenario 2: Concurrent batch processing

```python
# Two batches process simultaneously for same org
# Both try to create schema

# Race condition handled by Parse Server:
# - First creation succeeds
# - Second creation gets "already exists" error
# - Both batches use the existing schema ✅
```

---

## Summary

### Before (Manual):
```bash
# Step 1: Register schema manually
python scripts/register_agent_learning_schema.py --api-key KEY

# Step 2: Send messages
await paprManager.sendMessage({...})

# Step 3: Check if schema was used
# (Graph generation only works if Step 1 was done!)
```

### After (Automatic):
```javascript
// Step 1: Send messages
await paprManager.sendMessage({...})

// That's it! Schema auto-created on first batch.
```

**Result**: Zero setup, automatic graph generation, seamless experience! 🎉

---

## For Paprwork Users

You don't need to do anything! Just:

1. ✅ Send messages from Paprwork
2. ✅ After 15 messages, schema auto-created
3. ✅ Learnings captured in graph automatically
4. ✅ Project context detected
5. ✅ Tech stack mapped
6. ✅ Everything works out of the box!

No manual registration, no scripts, no setup. Just works! 🚀
