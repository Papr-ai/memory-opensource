# Memory-Oriented Policies

This directory contains documentation for the Memory Policy API - a unified approach to controlling how memories are processed, stored, and protected in Papr Memory.

## Overview

The Memory Policy API addresses several key challenges:
1. **User ID confusion** - Standardizes on `external_user_id` across all endpoints
2. **Graph control** - Unified `memory_policy` for structured and unstructured data
3. **Node constraints** - Fine-grained control over LLM-extracted entities
4. **Safety standards** - OMO (Open Memory Object) compliance with consent and risk tracking

## Documentation

| Document | Description |
|----------|-------------|
| [API_SIMPLIFICATION_PLAN.md](./API_SIMPLIFICATION_PLAN.md) | Full implementation plan with phases, tasks, and timeline |
| [MEMORY_POLICY_API.md](./MEMORY_POLICY_API.md) | Complete API reference with examples |
| [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) | Step-by-step migration guide for existing users |

## Quick Start

### Simple Memory (Most Common)

```python
response = await client.add_memory(
    content="User prefers dark mode",
    external_user_id="user_alice_123"
)
```

### Structured Data (Graph Override)

```python
response = await client.add_memory(
    content="Transaction: Alice bought Latte",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="manual",
        nodes=[NodeSpec(id="txn_1", type="Transaction", properties={...})],
        relationships=[RelationshipSpec(source="txn_1", target="prod_1", type="PURCHASED")]
    )
)
```

### Auto with Constraints (LLM + Rules)

```python
response = await client.add_memory(
    content="Meeting notes with task assignments...",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="auto",  # Constraints are automatically applied when provided
        node_constraints=[
            NodeConstraint(node_type="Task", create="never", merge=["status"])
        ]
    )
)
```

## Key Concepts

### Policy Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `auto` | LLM extracts entities. If `node_constraints` provided, they are applied. | Unstructured data (notes, conversations), with optional business rules |
| `manual` | Developer provides exact nodes (no LLM extraction) | Structured data (database, API) |

**Note**: `structured` is a deprecated alias for `manual`. `hybrid` is a deprecated alias for `auto` (constraints are now auto-applied).

### Node Constraints

Control how specific node types are handled:

```python
NodeConstraint(
    node_type="Task",          # What type
    when={"priority": "high"}, # When to apply
    create="never",            # Don't create new
    force={"urgent": True},    # Always set
    merge=["status"]           # Update on existing
)
```

### OMO Safety Standards

OMO safety fields are now integrated directly into `MemoryPolicy`, aligned with the [Open Memory Object (OMO) v1 Schema](https://github.com/papr-ai/open-memory-object/blob/main/schema/omo-v1.schema.json):

```python
response = await client.add_memory(
    content="Customer data with explicit consent",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="auto",
        consent="explicit",    # OMO standard: explicit, implicit, terms, none
        risk="sensitive",      # OMO standard: none, sensitive, flagged
        omo_acl={"read": ["user_alice"], "write": ["user_alice"]}
    )
)
```

## Implementation Status

- [x] API Design Complete
- [x] Documentation Written
- [x] Phase 1: Validation Layer
- [x] Phase 2: external_user_id Standardization
- [x] Phase 3: MemoryPolicy Model
- [x] Phase 4: OMO Integration
- [x] Phase 5: Testing & SDK Updates

### Key Implementation Details

1. **Unified `MemoryPolicy`**: All graph generation and OMO safety settings in one model
2. **`SchemaSpecificationMixin`**: `memory_policy` available on all request types (AddMemoryRequest, etc.)
3. **Schema-Level Defaults**: Define `memory_policy` on UserGraphSchema for automatic application
4. **Backwards Compatible**: `graph_generation` still works but is deprecated

## Related Documentation

- [Architecture: OMO Integration](../../architecture/OMO_NODE_CONSTRAINTS_INTEGRATION.md)
- [Guides: Parse Server Format](../../guides/PARSE_SERVER_FORMAT.md)
- [Features: ACL](../acl/)
- [Open Memory Object (OMO) Repository](https://github.com/papr-ai/open-memory-object) - Open standard for AI memory

## OMO Standard Architecture

Papr Memory implements the [Open Memory Object (OMO)](https://github.com/papr-ai/open-memory-object) standard with extensions:

### Core OMO Fields (from omo-v1.schema.json)

| OMO Field | Description | Values |
|-----------|-------------|--------|
| `consent` | How data owner allowed storage | explicit, implicit, terms, none |
| `risk` | Post-ingest safety assessment | none, sensitive, flagged |
| `acl` | Access control list | {read: [...], write: [...]} |

### Papr Extension (`ext.papr:*`)

Memory policies are implemented as Papr-specific extensions to the OMO standard:

```json
{
  "id": "mem_123",
  "content": "Meeting notes...",
  "consent": "explicit",
  "risk": "none",
  "acl": {"read": ["user_alice"], "write": ["user_alice"]},
  "ext": {
    "papr:memory_policy": {
      "mode": "auto",
      "node_constraints": [
        {"node_type": "Task", "create": "never"}
      ]
    },
    "papr:schema_id": "project_management"
  }
}
```

This keeps the OMO core portable while allowing Papr-specific features.

See the [OMO specification](https://github.com/papr-ai/open-memory-object/blob/main/docs/draft-v0.1.md) for the full standard.
