# Papr Memory API Design Principles

> "Steve Jobs meets Elon Musk for API design" - Simple, intuitive, powerful with control.

## Table of Contents

1. [Core Design Philosophy](#core-design-philosophy)
2. [The Three-Layer Architecture](#the-three-layer-architecture)
3. [Field Placement Decision Framework](#field-placement-decision-framework)
4. [Schema + memory_policy + Metadata Interaction](#schema--memory_policy--metadata-interaction)
5. [Unified Experience Across Endpoints](#unified-experience-across-endpoints)
6. [Common Use Cases & Patterns](#common-use-cases--patterns)
7. [Pain Points & Solutions](#pain-points--solutions)

---

## Core Design Philosophy

### The Five Questions

When deciding where a field belongs, ask:

| Question | Answer | Placement |
|----------|--------|-----------|
| **WHO** created this? | User identity | Request-level (`external_user_id`) |
| **WHERE** does it belong? | Tenant scoping | Request-level (`organization_id`, `namespace_id`) |
| **HOW** should it be processed? | Processing rules | Request-level (`memory_policy`) |
| **WHO** can access it? | Security/ACL | Request-level (`memory_policy.acl`, `.consent`, `.risk`) |
| **WHAT** is it about? | Content description | Metadata (`role`, `category`, `topics`, `customMetadata`) |

### Design Principles

1. **Single Source of Truth**: Each concept has ONE canonical location
2. **Separation of Concerns**: Security at request-level, content in metadata
3. **Sensible Defaults**: Works without configuration, customizable when needed
4. **Backwards Compatibility**: Deprecated fields accepted with warnings
5. **Predictable Precedence**: Request > Schema > System defaults

---

## The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: SCHEMA                               │
│  "What types of things can exist in my knowledge graph?"                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  UserGraphSchema:                                                       │
│  - node_types: {Project, Task, Person, ...}  ← Define structure        │
│  - relationship_types: {ASSIGNED_TO, ...}    ← Define connections      │
│  - memory_policy: {...}                      ← Default processing rules│
│                                                                         │
│  Scope: Organization/Namespace/Workspace level                          │
│  Persistence: Long-lived, reusable across many memories                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                          Schema ID referenced
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: REQUEST                                  │
│  "How should THIS specific memory/document/message be handled?"         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  AddMemoryRequest / UploadDocumentRequest / MessageRequest:             │
│                                                                         │
│  # Identity - WHO                                                        │
│  external_user_id: "user_alice_123"                                     │
│                                                                         │
│  # Scoping - WHERE                                                       │
│  organization_id: "org_acme"                                            │
│  namespace_id: "ns_production"                                          │
│                                                                         │
│  # Processing - HOW (overrides schema defaults)                         │
│  memory_policy:                                                          │
│    mode: "auto" | "manual" | "hybrid"                                   │
│    schema_id: "schema_123"        ← Reference schema                    │
│    node_constraints: [...]        ← Override schema constraints         │
│    consent: "explicit"            ← OMO safety                          │
│    risk: "none"                   ← OMO safety                          │
│    acl: {read: [...], write: [...]}  ← OMO safety                       │
│                                                                         │
│  # Content Description - WHAT                                            │
│  metadata: MemoryMetadata                                               │
│                                                                         │
│  Scope: Per-request                                                     │
│  Persistence: Applied once at ingestion time                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: METADATA                                │
│  "What descriptive information about the content should be stored?"     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MemoryMetadata:                                                        │
│                                                                         │
│  # Content Metadata - WHAT the memory contains                          │
│  role: "user" | "assistant"       ← Message role                        │
│  category: "preference" | "task"  ← Memory type                         │
│  topics: ["ai", "memory"]         ← Subject matter                      │
│  sourceUrl: "https://..."         ← Content origin                      │
│                                                                         │
│  # Developer Extensions                                                  │
│  customMetadata:                  ← Your filterable fields              │
│    project_id: "proj_123"                                               │
│    customer_tier: "enterprise"                                          │
│    session_id: "sess_456"                                               │
│                                                                         │
│  Scope: Per-memory, stored with content                                 │
│  Persistence: Stored in vector DB, used for filtering                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Field Placement Decision Framework

### Request-Level Fields

| Field | Purpose | When to Use |
|-------|---------|-------------|
| `external_user_id` | Identify your app's user | Every request (primary user identity) |
| `organization_id` | Multi-tenant org scoping | Multi-tenant apps |
| `namespace_id` | Environment scoping | Separate dev/staging/prod |
| `memory_policy` | Processing & safety rules | Custom graph generation or ACL |
| `memory_policy.mode` | Graph generation mode | Control LLM extraction |
| `memory_policy.schema_id` | Reference a schema | Use predefined node types |
| `memory_policy.acl` | Access control | Share memories between users |
| `memory_policy.consent` | Data consent level | Compliance requirements |
| `memory_policy.risk` | Safety assessment | Flag sensitive content |

### Metadata Fields (MemoryMetadata)

| Field | Purpose | When to Use |
|-------|---------|-------------|
| `role` | User vs assistant message | Conversation memories |
| `category` | Memory classification | Organize by type (preference, task, etc.) |
| `topics` | Subject tags | Topic-based filtering |
| `sourceUrl` | Content origin | Track where content came from |
| `conversationId` | Group related memories | Conversation threading |
| `customMetadata.*` | Your filterable fields | App-specific filtering needs |

### Internal Fields (Auto-populated, not for API use)

| Field | Purpose | Notes |
|-------|---------|-------|
| `user_read_access`, `user_write_access` | Vector store filtering | Auto-populated from `acl` |
| `workspace_read_access`, etc. | Vector store filtering | Internal use only |
| `relatedGoals`, `sessionId` | Internal tracking | System-managed |

---

## Schema + memory_policy + Metadata Interaction

### Precedence Rules

```
Request memory_policy  →  Schema memory_policy  →  System Defaults
     (highest)                (middle)                (lowest)
```

### Example: Project Management App

**Step 1: Define Schema (once)**

```python
# Create schema for your app
schema = UserGraphSchema(
    name="project_management_v1",
    node_types={
        "Project": UserNodeType(
            name="Project",
            label="Project",
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "status": PropertyDefinition(type="string", enum_values=["active", "completed", "archived"]),
                "priority": PropertyDefinition(type="string", enum_values=["low", "medium", "high"])
            },
            unique_identifiers=["name"]  # Merge by name
        ),
        "Task": UserNodeType(
            name="Task",
            label="Task",
            properties={
                "title": PropertyDefinition(type="string", required=True),
                "status": PropertyDefinition(type="string")
            }
        )
    },
    memory_policy={
        "mode": "hybrid",
        "node_constraints": [
            {
                "node_type": "Project",
                "create": "auto",
                "search": {"mode": "semantic", "threshold": 0.85}
            }
        ]
    }
)
```

**Step 2: Add Memories (many times)**

```python
# Memory about a specific project
await add_memory(
    content="Project Alpha milestone 2 is now completed. Great work team!",
    external_user_id="user_alice",
    organization_id="org_acme",

    # Processing rules - references schema, overrides one constraint
    memory_policy=MemoryPolicy(
        schema_id="project_management_v1",
        node_constraints=[
            # Override: force this memory's Project to have high priority
            {"node_type": "Project", "force": {"priority": "high"}}
        ]
    ),

    # Content metadata - for filtering
    metadata=MemoryMetadata(
        topics=["milestone", "completion"],
        customMetadata={
            "project_id": "proj_alpha_123",  # For fast filtering
            "team": "backend"
        }
    )
)
```

### When to Use Graph Nodes vs customMetadata

| Use Case | Graph Node | customMetadata | Why |
|----------|------------|----------------|-----|
| **Semantic relationships** | ✅ | ❌ | "Project has Tasks" needs graph |
| **Entity disambiguation** | ✅ | ❌ | Merge duplicate "Project Alpha" mentions |
| **Fast filtering** | ❌ | ✅ | "All memories for project_id=X" |
| **Analytics grouping** | ❌ | ✅ | "Count by customer_tier" |
| **Both semantic + filtering** | ✅ | ✅ | Create node AND add to customMetadata |

**Key Insight**: Graph nodes and customMetadata serve different purposes:
- **Graph nodes** = Semantic knowledge structure, entity relationships
- **customMetadata** = Fast vector search filtering, analytics dimensions

They can (and often should) contain the same identifiers for different purposes.

---

## Unified Experience Across Endpoints

### Memory Endpoint

```python
POST /v1/memory
{
    "content": "User prefers dark mode",
    "external_user_id": "user_123",
    "organization_id": "org_acme",
    "memory_policy": {
        "mode": "auto",
        "consent": "explicit"
    },
    "metadata": {
        "role": "user",
        "category": "preference",
        "customMetadata": {"source": "settings_page"}
    }
}
```

### Document Endpoint

```python
POST /v1/document
Content-Type: multipart/form-data

file: <PDF>
external_user_id: "user_123"
organization_id: "org_acme"
memory_policy: {"mode": "auto", "schema_id": "docs_schema"}
metadata: {"sourceType": "pdf", "customMetadata": {"department": "engineering"}}
```

### Message Endpoint

```python
POST /v1/message
{
    "messages": [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "It's sunny, 72°F"}
    ],
    "external_user_id": "user_123",
    "organization_id": "org_acme",
    "memory_policy": {"mode": "auto"},
    "metadata": {
        "conversationId": "conv_456",
        "customMetadata": {"channel": "mobile_app"}
    }
}
```

### Consistency Guarantee

All three endpoints support:
- `external_user_id` at request level
- `organization_id` / `namespace_id` at request level
- `memory_policy` at request level (with all OMO safety fields)
- `metadata` with `customMetadata` for filtering

---

## Common Use Cases & Patterns

### Use Case 1: Joe.coffee (100% Structured Data)

**Scenario**: POS transaction data from Square/Toast, no LLM extraction needed.

```python
# Schema defines exact structure
schema = UserGraphSchema(
    name="joe_coffee_orders",
    node_types={"Transaction": ..., "Product": ..., "Store": ...}
)

# Memory with exact nodes (no LLM)
await add_memory(
    content="Transaction: Alice bought Latte for $5.50",
    external_user_id="customer_alice",
    memory_policy=MemoryPolicy(
        mode="manual",  # No LLM extraction
        nodes=[
            NodeSpec(id="txn_123", type="Transaction", properties={"amount": 5.50}),
            NodeSpec(id="prod_latte", type="Product", properties={"name": "Latte"})
        ],
        relationships=[
            RelationshipSpec(source="txn_123", target="prod_latte", type="PURCHASED")
        ]
    ),
    metadata=MemoryMetadata(
        customMetadata={"store_id": "store_sf_01", "order_date": "2026-01-22"}
    )
)
```

### Use Case 2: DeepTrust (Compliance + Transcripts)

**Scenario**: Load compliance rules (manual), analyze call transcripts (auto).

```python
# 1. Load compliance rules (manual mode)
await add_memory(
    content="Security behavior: Agent must verify caller identity",
    memory_policy=MemoryPolicy(
        mode="manual",
        nodes=[NodeSpec(id="secbe_001", type="SecurityBehavior", properties={...})]
    ),
    metadata=MemoryMetadata(customMetadata={"label": "security_behavior"})
)

# 2. Analyze call transcript (auto mode with schema)
await add_memory(
    content="Call transcript: Caller asked about account balance...",
    memory_policy=MemoryPolicy(
        mode="auto",
        schema_id="call_analysis_schema",
        node_constraints=[
            {"node_type": "Action", "force": {"call_id": "call_789", "agent_id": "agent_42"}}
        ]
    ),
    metadata=MemoryMetadata(
        customMetadata={"call_id": "call_789", "agent_id": "agent_42"}
    )
)
```

### Use Case 3: Meeting Notes with Project Linking

**Scenario**: Extract entities from meeting notes, link to existing projects.

```python
await add_memory(
    content="Meeting: Discussed Project Alpha. John will complete API review by Friday.",
    external_user_id="user_alice",
    memory_policy=MemoryPolicy(
        mode="hybrid",
        schema_id="project_management_v1",
        node_constraints=[
            # Only link to existing projects (controlled vocabulary)
            {"node_type": "Project", "create": "never", "search": {"mode": "semantic", "threshold": 0.9}},
            # Allow creating new tasks
            {"node_type": "Task", "create": "auto", "merge": ["status"]}
        ]
    ),
    metadata=MemoryMetadata(
        topics=["meeting", "api-review"],
        customMetadata={
            "project_id": "proj_alpha_123",  # Fast filtering
            "meeting_type": "weekly_sync"
        }
    )
)
```

---

## Pain Points & Solutions

### Pain Point 1: Duplicate Fields

**Problem**: `external_user_id` appeared at both request-level and in metadata.

**Solution**: Single source of truth at request-level. Metadata fields deprecated with warnings.

```python
# ✅ CORRECT - Request level
await add_memory(
    content="...",
    external_user_id="user_123",  # Primary location
    metadata=MemoryMetadata(topics=["ai"])
)

# ⚠️ DEPRECATED - Still works but logs warning
await add_memory(
    content="...",
    metadata=MemoryMetadata(external_user_id="user_123")  # Deprecated
)
```

### Pain Point 2: ACL Complexity

**Problem**: 12 granular ACL fields vs simplified `acl` dict.

**Solution**:
- Developer-facing: Use `memory_policy.acl` (simple dict with prefixed entities)
- Internal: Granular fields auto-populated for efficient vector store filtering

#### ACL Entity Prefixes

| Prefix | Description | Validation | Example |
|--------|-------------|------------|---------|
| `external_user:` | Your app's user ID | **Not validated** (your responsibility) | `external_user:alice_123` |
| `user:` | Internal Papr user ID | Validated against Parse users | `user:mhnkVbAdgG` |
| `organization:` | Organization ID | Validated against your orgs | `organization:org_acme` |
| `namespace:` | Namespace ID | Validated against your namespaces | `namespace:ns_prod` |
| `workspace:` | Workspace ID | Validated against your workspaces | `workspace:ws_123` |
| `role:` | Parse role ID | Validated against your roles | `role:admin` |

**Validation Rules:**
- **Internal entities** (`user:`, `organization:`, `namespace:`, `workspace:`, `role:`) are validated against your Papr data. Invalid IDs return an error.
- **External entities** (`external_user:`) are NOT validated. Your app is responsible for ensuring these are valid.
- **Unprefixed values** default to `external_user:` for backwards compatibility.

```python
# ✅ Recommended: Use explicit prefixes
memory_policy=MemoryPolicy(
    acl={
        "read": [
            "external_user:alice_123",      # Your app's user (not validated)
            "organization:org_acme",        # Entire org can read (validated)
            "namespace:ns_production"       # Namespace-scoped access (validated)
        ],
        "write": ["external_user:alice_123"]
    }
)

# ⚠️ Backwards compatible: Unprefixed defaults to external_user:
memory_policy=MemoryPolicy(
    acl={"read": ["alice_123"], "write": ["alice_123"]}
    # Internally becomes: external_user:alice_123
)

# System internally expands to granular fields for filtering:
# external_user_read_access=["alice_123"]
# organization_read_access=["org_acme"]
# namespace_read_access=["ns_production"]
# external_user_write_access=["alice_123"]
```

#### ACL Use Cases

| Scenario | ACL Configuration |
|----------|-------------------|
| **Private to user** | `{"read": ["external_user:alice"], "write": ["external_user:alice"]}` |
| **Share with team** | `{"read": ["external_user:alice", "external_user:bob", "external_user:carol"], "write": ["external_user:alice"]}` |
| **Organization-wide** | `{"read": ["organization:org_acme"], "write": ["external_user:admin"]}` |
| **Environment-scoped** | `{"read": ["namespace:ns_prod"], "write": ["namespace:ns_prod"]}` |
| **Role-based** | `{"read": ["role:viewer", "role:editor"], "write": ["role:editor"]}` |

### Pain Point 3: Schema vs Memory-Level Confusion

**Problem**: Where do I set processing rules - schema or request?

**Solution**: Clear precedence and use case guidance.

| Scenario | Where to Set | Why |
|----------|--------------|-----|
| "All my app's memories should use these node types" | Schema | Reusable default |
| "This specific memory has special handling" | Request `memory_policy` | Override for one memory |
| "Most memories use schema defaults" | Reference `schema_id` only | Inherit schema settings |

### Pain Point 4: Nodes vs customMetadata

**Problem**: Should `project_id` go in graph nodes or customMetadata?

**Solution**: They serve different purposes - use both when needed.

```python
# Graph node: For semantic relationships and entity merging
memory_policy=MemoryPolicy(
    mode="hybrid",
    nodes=[NodeSpec(id="proj_123", type="Project", properties={"name": "Alpha"})]
)

# customMetadata: For fast filtering in search
metadata=MemoryMetadata(
    customMetadata={"project_id": "proj_123"}  # Same ID, different purpose
)
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAPR MEMORY API QUICK REF                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  REQUEST LEVEL (per-request control)                            │
│  ├── external_user_id     WHO created this                      │
│  ├── organization_id      WHERE (org scope)                     │
│  ├── namespace_id         WHERE (environment)                   │
│  └── memory_policy        HOW to process                        │
│      ├── mode             auto | manual | hybrid                │
│      ├── schema_id        Reference schema                      │
│      ├── nodes            Exact nodes (manual mode)             │
│      ├── node_constraints Rules for LLM extraction              │
│      ├── consent          explicit | implicit | terms | none    │
│      ├── risk             none | sensitive | flagged            │
│      └── acl              {read: [...], write: [...]}           │
│                                                                 │
│  METADATA (content description)                                 │
│  ├── role                 user | assistant                      │
│  ├── category             preference | task | goal | fact       │
│  ├── topics               ["topic1", "topic2"]                  │
│  ├── sourceUrl            Content origin                        │
│  └── customMetadata       {your_field: "value"}                 │
│                                                                 │
│  PRECEDENCE                                                     │
│  Request memory_policy > Schema memory_policy > System defaults │
│                                                                 │
│  OMO SAFETY (conforms to Open Memory Object standard)           │
│  https://github.com/anthropics/open-memory-object               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Migration Guide

See [MIGRATION_GUIDE.md](../features/memory_oriented_policies/MIGRATION_GUIDE.md) for:
- Deprecated field mapping
- Code migration examples
- Backwards compatibility details
