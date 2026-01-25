# Memory Policy API Documentation

This document describes the unified Memory Policy API for Papr Memory, providing developers with intuitive control over how memories are processed, stored, and protected.

## Table of Contents

1. [Overview](#overview)
2. [User Identification](#user-identification)
3. [Memory Policy](#memory-policy)
4. [Policy Modes](#policy-modes)
5. [Node Constraints](#node-constraints)
6. [OMO Safety Standards](#omo-safety-standards)
7. [API Reference](#api-reference)
8. [Examples](#examples)

---

## Overview

The Memory Policy API provides a unified way to:
- **Identify users** consistently across all endpoints
- **Control graph generation** from unstructured to fully structured data
- **Apply constraints** to LLM-extracted entities
- **Enforce safety standards** with consent and risk management

### Design Principles

1. **Simplicity**: Single, obvious way to do things
2. **Control**: Power users can access advanced features
3. **Safety**: Prevent common errors with validation
4. **Backwards Compatibility**: Old APIs continue working

---

## User Identification

### Primary Field: `external_user_id`

Use `external_user_id` for all user identification. This is your application's user identifier.

```python
# Recommended
response = await client.add_memory(
    content="User prefers dark mode",
    external_user_id="user_alice_123"  # Your app's user ID
)
```

### Field Comparison

| Field | Purpose | When to Use |
|-------|---------|-------------|
| `external_user_id` | Your application's user identifier | **Always** - Primary method |
| `user_id` | Papr internal Parse user ID | Rarely - Advanced use only |
| `end_user_id` | Alias for `external_user_id` | Backwards compatibility |

### Validation

The API validates user IDs to prevent common errors:

```json
// Error when external ID is used in user_id field
{
  "code": 400,
  "error": "Invalid user_id format",
  "details": {
    "field": "user_id",
    "provided_value": "user_alice_123",
    "reason": "This looks like an external user identifier. Did you mean to use 'external_user_id' instead?",
    "suggestion": "Use 'external_user_id' for your application's user identifiers."
  }
}
```

---

## Memory Policy

The `memory_policy` field controls how memories are processed and what graph nodes are created. It is defined in `SchemaSpecificationMixin`, making it available on all memory request types.

### Structure

```python
class MemoryPolicy(BaseModel):
    """
    Unified memory processing policy.
    Available via SchemaSpecificationMixin on AddMemoryRequest and other request types.
    """

    # ========== GRAPH GENERATION ==========
    mode: PolicyMode = "auto"                    # auto, manual
    nodes: Optional[List[NodeSpec]] = None       # For manual mode
    relationships: Optional[List[RelationshipSpec]] = None  # Supports $this and $previous
    node_constraints: Optional[List[NodeConstraint]] = None # For auto mode (applied when present)
    schema_id: Optional[str] = None              # Reference schema constraints

    # ========== OMO SAFETY STANDARDS ==========
    # Aligned with Open Memory Object v1 schema:
    # https://github.com/papr-ai/open-memory-object/blob/main/schema/omo-v1.schema.json
    consent: ConsentLevel = "implicit"           # explicit, implicit, terms, none
    risk: RiskLevel = "none"                     # none, sensitive, flagged
    omo_acl: Optional[Dict[str, List[str]]] = None  # {read: [...], write: [...]}

# Memory Linking via relationships
# Use special placeholders:
#   $this - the Memory node being created
#   $previous - the user's most recent memory
#
# Example:
#   relationships=[
#       RelationshipSpec(source="$this", target="$previous", type="FOLLOWS")
#   ]
```

**Note**: `structured` is a deprecated alias for `manual`. `hybrid` is a deprecated alias for `auto`.

### SchemaSpecificationMixin

All memory request types inherit from `SchemaSpecificationMixin`, which provides the `memory_policy` field:

```python
class SchemaSpecificationMixin(BaseModel):
    """
    Mixin for consistent memory processing policy across all memory request types.
    """

    # PRIMARY: Unified memory policy (RECOMMENDED)
    memory_policy: Optional[MemoryPolicy] = None

    # DEPRECATED: Legacy graph generation (backwards compatible)
    graph_generation: Optional[GraphGeneration] = None

# Usage: AddMemoryRequest inherits memory_policy
class AddMemoryRequest(SchemaSpecificationMixin):
    content: str
    # ... other fields
    # memory_policy is available via mixin!
```

---

## Policy Modes

### Mode: `auto` (Default)

LLM extracts entities from memory content. If `node_constraints` are provided, they are automatically applied.

```python
# Simple auto - LLM extracts freely
response = await client.add_memory(
    content="Met with John from Acme Corp to discuss Q4 roadmap",
    external_user_id="user_alice_123"
    # mode defaults to "auto" - LLM extracts Person (John), Company (Acme Corp), etc.
)
```

**Best for**: Unstructured data like notes, conversations, documents.

### Mode: `auto` with Constraints

When you provide `node_constraints`, the LLM still extracts entities but your rules are applied.

```python
response = await client.add_memory(
    content="Meeting: Discussed Project Alpha. John will complete the API review by Friday.",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="auto",  # Constraints are automatically applied when present
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                create="never",  # Only link to existing tasks
                search=SearchConfig(mode="semantic", threshold=0.85)
            ),
            NodeConstraint(
                node_type="Person",
                create="never"  # Controlled vocabulary - existing team members only
            )
        ]
    )
)
```

**Best for**: Unstructured data with business rules (meetings, emails).

### Mode: `manual`

Developer provides exact nodes and relationships. No LLM extraction.

```python
response = await client.add_memory(
    content="Transaction: Alice bought Latte for $5.50",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="manual",
        nodes=[
            NodeSpec(
                id="txn_12345",
                type="Transaction",
                properties={"amount": 5.50, "product": "Latte"}
            ),
            NodeSpec(
                id="product_latte",
                type="Product",
                properties={"name": "Latte", "category": "Coffee"}
            )
        ],
        relationships=[
            RelationshipSpec(
                source="txn_12345",
                target="product_latte",
                type="PURCHASED"
            )
        ]
    )
)
```

**Best for**: Structured data from databases, APIs, CRM systems.

### Deprecated Mode Aliases

| Deprecated | Maps To | Notes |
|------------|---------|-------|
| `structured` | `manual` | Same behavior |
| `hybrid` | `auto` | Constraints are now auto-applied in `auto` mode |

---

## Node Constraints

Node constraints define policies for how specific node types are handled. They can be defined at **two levels**:

1. **Schema level**: Inside `UserNodeType.constraint` - defines defaults for all nodes of that type
2. **Memory level**: In `memory_policy.node_constraints[]` - overrides schema defaults per memory

> **Full Reference**: See [NODE_CONSTRAINTS_API.md](./NODE_CONSTRAINTS_API.md) for complete documentation.

### Full Structure

```python
class NodeConstraint(BaseModel):
    # WHAT - Node type to constrain
    node_type: Optional[str]  # Required at memory level, implicit at schema level

    # WHEN - Conditional with logical operators (_and, _or, _not)
    when: Optional[Dict[str, Any]] = None  # Supports: {"_and": [...], "_or": [...], "_not": {...}}

    # CREATION POLICY
    create: Literal["auto", "never"] = "auto"  # "auto": create if not found, "never": link only

    # NODE SELECTION (property-based matching with PropertyMatch)
    search: Optional[SearchConfig] = None  # Uses PropertyMatch list

    # PROPERTY VALUES (unified)
    set: Optional[Dict[str, SetValue]] = None  # Exact values or auto-extract
```

### Constraint Examples

#### Schema Level - Controlled Vocabulary (Link Only)

```python
# In UserNodeType definition
UserNodeType(
    name="Person",
    constraint=NodeConstraint(
        # node_type is implicit from parent
        create="never",  # Don't create new Person nodes
        search=SearchConfig(
            properties=[
                PropertyMatch(name="email", mode="exact"),
                PropertyMatch(name="name", mode="semantic", threshold=0.90)
            ]
        )
    )
)
# Result: Only links to existing team members, ignores unknown names
```

#### Memory Level - Select Specific Node

```python
# Use PropertyMatch with value instead of old node_id
NodeConstraint(
    node_type="Project",
    search=SearchConfig(
        properties=[
            PropertyMatch(name="id", mode="exact", value="proj_alpha_123")
        ]
    ),
    set={
        "status": {"mode": "auto"},  # Auto-extract from content
        "milestone": {"mode": "auto"}
    }
)
```

#### Conditional Constraints with Logical Operators

```python
NodeConstraint(
    node_type="Task",
    when={
        "_and": [
            {"priority": "high"},
            {"_not": {"status": "completed"}}
        ]
    },
    set={"urgent": True, "notify_team": True}
)
```

#### Auto-Extract on Existing

```python
NodeConstraint(
    node_type="Task",
    create="never",
    set={
        "status": {"mode": "auto"},
        "assignee": {"mode": "auto"}
    }
)
```

### Search Configuration (Property-Based Matching)

```python
class SearchConfig(BaseModel):
    # Property-based matching (replaces old node_id + properties as List[str])
    properties: Optional[List[PropertyMatch]] = None  # PropertyMatch list

    # Default settings when property doesn't specify
    mode: Literal["semantic", "exact", "fuzzy"] = "semantic"
    threshold: float = 0.85  # 0.0-1.0 for semantic search

class PropertyMatch(BaseModel):
    name: str                    # Property name (e.g., "id", "email", "title")
    mode: Literal["exact", "semantic", "fuzzy"] = "exact"
    threshold: float = 0.85      # For semantic/fuzzy only
    value: Optional[Any] = None  # Runtime value override (replaces old node_id)
```

**Key Change:** `node_id` was removed. Use `PropertyMatch` with `value` instead:
```python
# Old approach (removed)
SearchConfig(node_id="TASK-123")

# New approach
SearchConfig(properties=[PropertyMatch(name="id", mode="exact", value="TASK-123")])
```

### SetValue Types

```python
# SetValue can be:
# 1. Exact value: str, int, float, bool, list, dict
# 2. Auto-extract config: PropertyValue

class PropertyValue(BaseModel):
    mode: Literal["auto"] = "auto"     # LLM extracts from content
    text_mode: Literal["replace", "append", "merge"] = "replace"
```

---

## OMO Safety Standards

Open Memory Object (OMO) safety standards are integrated directly into `MemoryPolicy`, aligned with the [OMO v1 Schema](https://github.com/papr-ai/open-memory-object/blob/main/schema/omo-v1.schema.json).

### Consent Levels

From OMO v1 schema: `"consent": {"type": "string", "enum": ["explicit", "implicit", "terms", "none"]}`

| Value | Description |
|-------|-------------|
| `explicit` | User explicitly agreed to store this memory |
| `implicit` | Inferred from usage context (default) |
| `terms` | Covered by Terms of Service |
| `none` | No consent recorded - **graph extraction will be skipped** |

### Risk Levels

From OMO v1 schema: `"risk": {"type": "string", "enum": ["none", "sensitive", "flagged"]}`

| Value | Description |
|-------|-------------|
| `none` | Safe content (default) |
| `sensitive` | Contains PII, financial, health info |
| `flagged` | Requires human review - **ACL restricted to owner only** |

### ACL Configuration

From OMO v1 schema: `"acl": {"type": "object", "properties": {"read": [...], "write": [...]}}`

```python
omo_acl={"read": ["user_alice_123", "admin_bob"], "write": ["user_alice_123"]}
```

### Using OMO in MemoryPolicy (RECOMMENDED)

OMO fields are now part of `MemoryPolicy`:

```python
response = await client.add_memory(
    content="Customer SSN: 123-45-6789",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="auto",
        consent="explicit",           # OMO consent
        risk="sensitive",             # OMO risk
        omo_acl={                     # OMO ACL
            "read": ["user_alice_123", "admin_bob"],
            "write": ["user_alice_123"]
        }
    )
)
```

### Legacy: Using OMO in Metadata

For backwards compatibility, OMO fields also exist in `MemoryMetadata`:

```python
response = await client.add_memory(
    content="Customer SSN: 123-45-6789",
    external_user_id="user_alice_123",
    metadata=MemoryMetadata(
        consent="explicit",
        risk="sensitive",
        omo_acl={
            "read": ["user_alice_123", "admin_bob"],
            "write": ["user_alice_123"]
        }
    )
)
```

### OMO Extension Architecture

Memory policies are stored as Papr-specific extensions in the OMO `ext` namespace:

```json
{
  "id": "mem_123",
  "createdAt": "2026-01-21T10:30:00Z",
  "type": "text",
  "content": "Meeting notes with John...",
  "consent": "explicit",
  "risk": "none",
  "acl": {"read": ["user_alice"], "write": ["user_alice"]},
  "ext": {
    "papr:memory_policy": {
      "mode": "auto",
      "node_constraints": [
        {"node_type": "Task", "create": "never", "set": {"status": {"mode": "auto"}}}
      ]
    },
    "papr:schema_id": "project_management",
    "papr:relationships": [...]
  }
}
```

This architecture:
- Keeps OMO core fields portable across platforms
- Memory policies are Papr extensions via `ext.papr:memory_policy`
- Other Papr features use `ext.papr:*` namespace

See the [Open Memory Object standard](https://github.com/papr-ai/open-memory-object) for more details.

### Safety Pipeline

1. **Consent Check**: Memories with `consent=NONE` skip graph extraction
2. **Risk Assessment**: `FLAGGED` content restricts ACL to owner only
3. **ACL Propagation**: Memory ACL propagates to extracted nodes
4. **Audit Trail**: All operations are logged for compliance

---

## API Reference

### AddMemoryRequest

Inherits from `SchemaSpecificationMixin`, which provides `memory_policy`:

```python
class AddMemoryRequest(SchemaSpecificationMixin):
    # Required
    content: str

    # User identification
    external_user_id: Optional[str] = None  # Primary - your app's user ID

    # Type
    type: MemoryType = MemoryType.TEXT

    # Metadata
    metadata: Optional[MemoryMetadata] = None

    # Multi-tenant scoping
    organization_id: Optional[str] = None
    namespace_id: Optional[str] = None

    # --- From SchemaSpecificationMixin ---
    # memory_policy: Optional[MemoryPolicy] = None  # Unified policy (RECOMMENDED)
    # graph_generation: Optional[GraphGeneration] = None  # DEPRECATED

    # Deprecated (backwards compatible)
    user_id: Optional[str] = None  # Use external_user_id instead
```

### Schema-Level Memory Policy

You can define a default `memory_policy` on a `UserGraphSchema`. When memories reference this schema, the schema's policy is used as defaults (memory-level policy takes precedence):

```python
# Define schema with default policy
schema = UserGraphSchema(
    name="project_management",
    description="Schema for project management memories",
    memory_policy={
        "mode": "auto",
        "consent": "terms",
        "node_constraints": [
            {"node_type": "Task", "create": "never"},
            {"node_type": "Person", "create": "never"}
        ]
    }
)

# Use schema - inherits memory_policy defaults
response = await client.add_memory(
    content="Meeting notes with John...",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        schema_id="project_management",  # Inherit defaults
        node_constraints=[               # Override: allow Person creation
            {"node_type": "Person", "create": "auto"}
        ]
    )
)
```

### SearchRequest

```python
class SearchRequest(BaseModel):
    query: str
    external_user_id: Optional[str] = None  # Filter by user
    max_memories: int = 20
    max_nodes: int = 15
    # ... other fields
```

---

## Examples

### Example 1: Simple Memory

```python
# Most common case - just add a memory
response = await client.add_memory(
    content="User prefers dark mode and large fonts",
    external_user_id="user_alice_123"
)
```

### Example 2: Structured Transaction Data

```python
# From database/API - developer controls exact graph structure
response = await client.add_memory(
    content="Transaction: Alice bought Latte for $5.50",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="manual",
        nodes=[
            NodeSpec(id="txn_001", type="Transaction", properties={"amount": 5.50}),
            NodeSpec(id="prod_latte", type="Product", properties={"name": "Latte"})
        ],
        relationships=[
            RelationshipSpec(source="txn_001", target="prod_latte", type="PURCHASED")
        ]
    )
)
```

### Example 3: Meeting Notes with Task Integration

```python
# Unstructured meeting notes, but tasks come from Linear
response = await client.add_memory(
    content="Sprint planning: John will complete the API review by Friday. Sarah handles testing.",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="auto",  # Constraints are automatically applied
        schema_id="project_management",
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                create="never",  # Only link to existing Linear tasks
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="id", mode="exact"),
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                set={
                    "status": {"mode": "auto"},
                    "assignee": {"mode": "auto"}
                }
            ),
            NodeConstraint(
                node_type="Person",
                create="never",  # Only team members from HR system
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="email", mode="exact"),
                        PropertyMatch(name="name", mode="semantic", threshold=0.90)
                    ]
                )
            )
        ]
    )
)
```

### Example 4: Sensitive Data with OMO

```python
# Customer data with explicit consent and restricted access
response = await client.add_memory(
    content="Customer feedback: John Smith prefers email contact. Account #12345.",
    external_user_id="user_alice_123",
    metadata=MemoryMetadata(
        consent=ConsentLevel.EXPLICIT,
        risk=RiskLevel.SENSITIVE,
        acl=ACLConfig(
            read=["user_alice_123", "support_team"],
            write=["user_alice_123"]
        )
    )
)
```

### Example 5: Document Upload

```python
# Document with consistent user identification
response = await client.upload_document(
    file=document_file,
    external_user_id="user_alice_123",  # Same field as memory routes
    preferred_provider="reducto",
    hierarchical_enabled=True
)
```

---

## Backwards Compatibility

All existing APIs continue to work. Deprecation warnings are logged but no errors are raised.

| Old API | New API | Status |
|---------|---------|--------|
| `user_id` | `external_user_id` | Deprecated, works |
| `end_user_id` | `external_user_id` | Alias, works |
| `graph_generation` | `memory_policy` | Deprecated, auto-converted |
| `graph_generation.manual` | `memory_policy.mode="manual"` | Auto-converted |
| `mode="structured"` | `mode="manual"` | Deprecated alias, auto-converted |
| `mode="hybrid"` | `mode="auto"` | Deprecated alias (constraints now auto-applied) |
| `metadata.consent` | `memory_policy.consent` | Both work, `memory_policy` preferred |
| `metadata.risk` | `memory_policy.risk` | Both work, `memory_policy` preferred |
| `metadata.omo_acl` | `memory_policy.omo_acl` | Both work, `memory_policy` preferred |
| `search.node_id` | `PropertyMatch.value` | Use `PropertyMatch(name="id", value="...")` |
| `search.properties` as `List[str]` | `List[PropertyMatch]` | Use `PropertyMatch` with per-property config |
| `unique_identifiers` in `UserNodeType` | `constraint.search.properties` | Define in `UserNodeType.constraint` |

### Migration Priority

1. **`graph_generation` → `memory_policy`**: Highest priority, provides unified experience
2. **OMO fields in `metadata` → `memory_policy`**: Recommended for consistency
3. **`user_id` → `external_user_id`**: Important for clarity

---

## Error Handling

### Common Errors

```json
// Invalid user_id format
{
  "code": 400,
  "error": "Invalid user_id format",
  "details": {
    "field": "user_id",
    "suggestion": "Use 'external_user_id' for your application's user identifiers."
  }
}

// Invalid property value reference
{
  "code": 404,
  "error": "Node not found",
  "details": {
    "property": "id",
    "value": "proj_unknown",
    "suggestion": "Verify the node exists or use semantic matching instead of exact mode."
  }
}

// Invalid constraint configuration
{
  "code": 400,
  "error": "Invalid NodeConstraint",
  "details": {
    "issue": "Cannot set both exact value and auto-extract mode for the same property",
    "properties": ["status"]
  }
}
```

---

## Migration Guide

See [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) for detailed migration instructions.
