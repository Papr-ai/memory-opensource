# Migration Guide: Memory Policy API

This guide helps existing Papr Memory users migrate to the new unified Memory Policy API. All changes are backwards compatible - your existing code will continue to work.

## Table of Contents

1. [Quick Start](#quick-start)
2. [User ID Migration](#user-id-migration)
3. [Graph Generation Migration](#graph-generation-migration)
4. [Document Upload Migration](#document-upload-migration)
5. [OMO Safety Features](#omo-safety-features)
6. [Timeline](#timeline)

---

## Quick Start

### Minimum Required Change: None

Your existing code continues to work. However, we recommend updating to the new API for:
- Clearer code
- Better error messages
- Access to new features

### Recommended Changes

```python
# BEFORE
response = await client.add_memory(
    content="User preference",
    user_id="user_alice_123"  # Ambiguous - internal or external?
)

# AFTER (recommended)
response = await client.add_memory(
    content="User preference",
    external_user_id="user_alice_123"  # Clear - your app's user ID
)
```

---

## User ID Migration

### The Problem

Previously, there was confusion between:
- `user_id` - Could mean Parse internal ID or your app's ID
- `external_user_id` - Your app's user identifier
- `end_user_id` - Same as external_user_id (documents only)

This led to bugs where memories became unsearchable.

### The Solution

Use `external_user_id` everywhere.

### Migration Steps

#### Step 1: Replace `user_id` with `external_user_id`

```python
# BEFORE - memory routes
await client.add_memory(
    content="...",
    user_id="user_alice_123"
)

# AFTER
await client.add_memory(
    content="...",
    external_user_id="user_alice_123"
)
```

#### Step 2: Update document uploads

```python
# BEFORE - document routes
await client.upload_document(
    file=...,
    end_user_id="user_alice_123"  # Different field name!
)

# AFTER (both work, but this is clearer)
await client.upload_document(
    file=...,
    external_user_id="user_alice_123"  # Same as memory routes
)
```

#### Step 3: Update search requests

```python
# BEFORE
await client.search(
    query="...",
    user_id="user_alice_123",  # Was this working?
    external_user_id="user_alice_123"  # Redundant
)

# AFTER
await client.search(
    query="...",
    external_user_id="user_alice_123"  # Single source of truth
)
```

### What Happens to Old Code?

| Old Code | Behavior | Recommendation |
|----------|----------|----------------|
| `user_id="user_123"` | Works, deprecation warning logged | Update to `external_user_id` |
| `end_user_id="user_123"` | Works, alias supported | Update to `external_user_id` |
| `metadata.user_id="user_123"` | Works, precedence rules apply | Prefer request-level field |

---

## Graph Generation Migration

### The Problem

`graph_generation` was separate from node constraints, making it hard to control LLM extraction behavior.

### The Solution

Use `memory_policy` which unifies:
- Graph generation mode (auto, manual)
- Node constraints (automatically applied in auto mode when present)
- Structured data specification
- **OMO safety standards** (consent, risk, ACL)
- **Schema-level defaults** via `schema_id`

### Key Change: `memory_policy` in SchemaSpecificationMixin

`memory_policy` is now available on ALL request types that inherit from `SchemaSpecificationMixin`:

```python
# All these request types have memory_policy:
AddMemoryRequest(SchemaSpecificationMixin):
    memory_policy: Optional[MemoryPolicy] = None  # Unified policy
    graph_generation: Optional[GraphGeneration] = None  # DEPRECATED
```

### Migration by Use Case

#### Use Case 1: Default Graph Generation

```python
# BEFORE
await client.add_memory(
    content="Meeting notes...",
    graph_generation=None  # Default
)

# AFTER (unchanged - auto mode is default)
await client.add_memory(
    content="Meeting notes..."
)
```

#### Use Case 2: Manual Graph Override

```python
# BEFORE
await client.add_memory(
    content="...",
    graph_generation=GraphGeneration(
        enabled=False,
        manual=True,
        nodes=[...],
        relationships=[...]
    )
)

# AFTER
await client.add_memory(
    content="...",
    memory_policy=MemoryPolicy(
        mode="structured",
        nodes=[
            NodeSpec(id="...", type="...", properties={...})
        ],
        relationships=[
            RelationshipSpec(source="...", target="...", type="...")
        ]
    )
)
```

#### Use Case 3: Graph Generation with Constraints

```python
# BEFORE - not possible in single request
await client.add_memory(
    content="...",
    graph_generation=GraphGeneration(enabled=True)
)
# Then separately manage constraints...

# AFTER - unified (constraints auto-applied in auto mode)
await client.add_memory(
    content="...",
    memory_policy=MemoryPolicy(
        mode="auto",  # Constraints are automatically applied when present
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                create="never",
                force={"workspace_id": "ws_123"}
            )
        ]
    )
)
```

### Migration Table

| Old `graph_generation` | New `memory_policy` |
|------------------------|---------------------|
| `enabled=True, manual=False` | `mode="auto"` (default) |
| `enabled=False, manual=True, nodes=[...]` | `mode="manual", nodes=[...]` |
| `enabled=True` + separate constraints | `mode="auto", node_constraints=[...]` |

**Note**: `mode="structured"` is a deprecated alias for `mode="manual"`. `mode="hybrid"` is a deprecated alias for `mode="auto"` (constraints are now auto-applied).

### Full Feature Comparison

No functionality is lost. `MemoryPolicy` is a superset of `GraphGeneration`:

| GraphGeneration Feature | MemoryPolicy Equivalent | Notes |
|------------------------|------------------------|-------|
| `mode: auto` | `mode: "auto"` | Same (constraints auto-applied when present) |
| `mode: manual` | `mode: "manual"` | Same |
| `auto.schema_id` | `schema_id` (top-level) | Moved up for easier access |
| `auto.property_overrides` | `node_constraints[].force` + `when` | NodeConstraint is superset |
| `manual.nodes[].label` | `nodes[].type` | Renamed for consistency |
| `manual.nodes[].id/properties` | `nodes[].id/properties` | Same |
| `manual.relationships` | `relationships` | Same (field names normalized) |

**Removed**: `simple_schema_mode` has been removed as it was unused in the codebase.

### PropertyOverrideRule → NodeConstraint

NodeConstraint is a superset of PropertyOverrideRule with more features:

```python
# BEFORE: PropertyOverrideRule
property_overrides=[
    PropertyOverrideRule(
        nodeLabel="Task",
        match={"priority": "high"},
        set={"urgent": True}
    )
]

# AFTER: NodeConstraint (same + more features)
memory_policy=MemoryPolicy(
    mode="auto",  # Constraints are automatically applied when present
    node_constraints=[
        NodeConstraint(
            node_type="Task",           # nodeLabel → node_type
            when={"priority": "high"},  # match → when (alias supported)
            force={"urgent": True},     # set → force (alias supported)
            # NEW features:
            create="never",             # Control node creation
            merge=["status"],           # Update existing nodes
            search=SearchConfig(...)    # How to find existing nodes
        )
    ]
)
```

---

## Schema-Level Memory Policies (NEW)

### The Feature

You can now define a default `memory_policy` at the schema level. When memories reference this schema, they automatically inherit the schema's policy as defaults.

### How It Works

```python
# Step 1: Define schema with default memory_policy
schema = await client.create_schema(
    name="project_management",
    description="Schema for project management memories",
    memory_policy={
        "mode": "auto",  # Constraints are automatically applied
        "consent": "terms",
        "node_constraints": [
            {"node_type": "Task", "create": "never"},
            {"node_type": "Person", "create": "never"}
        ]
    }
)

# Step 2: Add memory using schema - inherits defaults automatically
response = await client.add_memory(
    content="Meeting notes about Project Alpha...",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        schema_id="project_management"  # Inherit schema defaults
    )
)
```

### Precedence Rules

Memory-level policy overrides schema-level policy:

```
Precedence (highest to lowest):
1. memory_policy values in request  ← Always wins
2. schema memory_policy defaults    ← Used if not in request
3. System defaults                  ← Fallback

Node constraints are MERGED:
- Schema constraints are base
- Memory constraints override same node_type
- Memory constraints for new node_types are added
```

### Example: Override Schema Defaults

```python
# Schema has: Task create="never", Person create="never"
response = await client.add_memory(
    content="New team member John joined...",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        schema_id="project_management",
        node_constraints=[
            # Override: Allow Person creation for this memory
            {"node_type": "Person", "create": "auto"}
        ]
    )
)
# Result: Task create="never" (from schema), Person create="auto" (override)
```

---

## Document Upload Migration

### The Problem

Document routes used `end_user_id` while memory routes used `external_user_id`.

### The Solution

Use `external_user_id` everywhere. `end_user_id` is kept as an alias.

### Migration

```python
# BEFORE
await client.upload_document(
    file=document,
    user_id="parse_internal_id",  # Often misused
    end_user_id="user_alice_123"
)

# AFTER
await client.upload_document(
    file=document,
    external_user_id="user_alice_123"
)
```

---

## OMO Safety Features

### New Feature: Consent Tracking

```python
# NEW - track data consent
await client.add_memory(
    content="Customer data...",
    external_user_id="user_alice_123",
    metadata=MemoryMetadata(
        consent=ConsentLevel.EXPLICIT  # User explicitly agreed
    )
)
```

### New Feature: Risk Levels

```python
# NEW - mark sensitive data
await client.add_memory(
    content="SSN: 123-45-6789",
    external_user_id="user_alice_123",
    metadata=MemoryMetadata(
        risk=RiskLevel.SENSITIVE
    )
)
```

### New Feature: Fine-grained ACL

```python
# NEW - explicit access control
await client.add_memory(
    content="...",
    external_user_id="user_alice_123",
    metadata=MemoryMetadata(
        acl=ACLConfig(
            read=["user_alice_123", "admin_team"],
            write=["user_alice_123"]
        )
    )
)
```

---

## Property Overrides Migration

### The Problem

`PropertyOverrideRule` used `set` for forced values, which conflicted with node constraint naming.

### The Solution

Node constraints use clearer names:
- `set` → `force` (always set these values)
- `update` → `merge` (update on existing nodes only)
- `match` → `when` (conditional application)

### Migration

```python
# BEFORE - PropertyOverrideRule
property_overrides=[
    PropertyOverrideRule(
        nodeLabel="Task",
        match={"priority": "high"},
        set={"urgent": True}
    )
]

# AFTER - NodeConstraint (both syntaxes work)
memory_policy=MemoryPolicy(
    mode="hybrid",
    node_constraints=[
        NodeConstraint(
            node_type="Task",
            when={"priority": "high"},  # or still use "match"
            force={"urgent": True}      # or still use "set"
        )
    ]
)
```

### Alias Support

The old names (`set`, `match`, `update`) are supported as aliases:

```python
# This still works
NodeConstraint(
    node_type="Task",
    match={"priority": "high"},  # Alias for "when"
    set={"urgent": True}         # Alias for "force"
)
```

---

## Timeline

### Phase 1: Complete ✅
- All new features available
- `memory_policy` in `SchemaSpecificationMixin`
- OMO safety fields integrated
- Schema-level memory policies
- Old APIs work with deprecation warnings (logged only)
- No breaking changes

### Phase 2: 3 Months
- SDK documentation updated
- TypeScript/Python SDK types updated
- Deprecation warnings in API responses

### Phase 3: 6 Months
- Old field names still work
- Warnings more prominent
- New documentation primary

### Phase 4: 12 Months (Future)
- Evaluate removing deprecated fields
- Community feedback period
- No removal without advance notice

---

## Checklist

Use this checklist to track your migration:

- [ ] Replace `user_id` with `external_user_id` in memory requests
- [ ] Replace `end_user_id` with `external_user_id` in document uploads
- [ ] Update `graph_generation` to `memory_policy` (if using manual mode)
- [ ] Convert `PropertyOverrideRule` to `NodeConstraint` (if using)
- [ ] Consider using schema-level `memory_policy` for default settings
- [ ] Consider adding `consent` tracking for compliance (now in `memory_policy`)
- [ ] Consider adding `risk` levels for sensitive data (now in `memory_policy`)
- [ ] Test with deprecation warnings enabled

---

## Getting Help

- **Documentation**: [MEMORY_POLICY_API.md](./MEMORY_POLICY_API.md)
- **Examples**: See the examples section in the API documentation
- **Support**: Contact support@papr.ai or open an issue

---

## FAQ

### Q: Will my existing code break?

**A**: No. All existing APIs continue to work. Deprecation warnings are logged but don't affect functionality.

### Q: Do I need to migrate immediately?

**A**: No, but we recommend migrating to get better error messages and access to new features.

### Q: What if I'm using both `user_id` and `external_user_id`?

**A**: The precedence is: `external_user_id` at request level > `external_user_id` in metadata > `user_id` in metadata.

### Q: Can I use the old `graph_generation` with new `node_constraints`?

**A**: Yes, they're converted internally. But we recommend using `memory_policy` for clarity.

### Q: Do I lose any functionality by switching to `memory_policy`?

**A**: No. `MemoryPolicy` is a superset of `GraphGeneration`. All features are preserved:
- `mode: manual` → `mode: manual` (same)
- `mode: auto` → `mode: auto` (constraints now auto-applied when present)
- `auto.schema_id` → `schema_id` (top-level)
- `auto.property_overrides` → `node_constraints` (with more features)
- Plus: OMO safety (`consent`, `risk`, `omo_acl`), schema-level defaults

**Removed**: `simple_schema_mode` has been removed as it was unused.

### Q: What about schema-level memory policies?

**A**: You can now define `memory_policy` on a `UserGraphSchema`. When memories reference this schema via `schema_id`, they automatically inherit the schema's policy as defaults. Memory-level policy overrides schema-level.

### Q: How do I know if I'm using deprecated fields?

**A**: Check your logs for deprecation warnings. They include the field name and suggested replacement.
