# Schema Constraints Quick Reference

*January 2026*

---

## Overview

This guide shows how to define schema-level constraints using the Python SDK, from simple to advanced. All examples are type-safe with full IDE support.

---

## Quick Summary Table

| Feature | Purpose | Decorator | Field | Example |
|---------|---------|-----------|-------|---------|
| **Creation Policy** | Control if nodes can be created | `@link_only` / `@auto_create` | `create="never"` / `create="auto"` | Reference data vs dynamic entities |
| **Search/Matching** | Define unique identifiers | `prop(search=...)` | `search.properties` | Match by email, semantic title |
| **Conditional Logic** | Apply constraints conditionally | `@constraint(when=...)` | `when={...}` | Only for priority="high" |
| **Property Values** | Set/override properties | `@constraint(set=...)` | `set={...}` | Auto-flag critical items |

---

## Feature Comparison

### Creation Policy: `@link_only` vs `@auto_create`

| Decorator | Equivalent | Behavior | Use Case |
|-----------|------------|----------|----------|
| `@link_only` | `create="never"` | Only link to existing nodes | Pre-populated reference data (MITRE tactics, users from IdP) |
| `@auto_create` | `create="auto"` | Create new if not found | Dynamic entities (conversations, actions, events) |
| `@controlled_vocabulary` | `create="never"` | Alias for `@link_only` | Same as link_only, more descriptive name |

### Search Modes

| Mode | Syntax | Use Case | Example |
|------|--------|----------|---------|
| `exact()` | `prop(search=exact())` | IDs, emails, codes | `id: str = prop(search=exact())` |
| `semantic(threshold)` | `prop(search=semantic(0.85))` | Names, titles, descriptions | `title: str = prop(search=semantic(0.85))` |
| `fuzzy(threshold)` | `prop(search=fuzzy(0.80))` | Typo-tolerant matching | `name: str = prop(search=fuzzy(0.80))` |

### Constraint Fields

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `when` | `Dict` | Conditional application | `{"priority": "high"}` |
| `set` | `Dict` | Property values to apply | `{"flagged": True}` |
| `search` | `SearchConfig` | How to find existing nodes | `properties=[exact("id")]` |
| `create` | `"auto"` \| `"never"` | Creation policy | `"never"` for controlled vocab |
| `link_only` | `bool` | Shorthand for `create="never"` | `True` |

---

## Progressive Examples

### Level 1: Simple Schema (Just Properties)

Minimal schema - no constraints, system uses defaults.

```python
from papr.sdk import schema, node, prop
from typing import Optional

@schema("simple_project")
class SimpleSchema:

    @node
    class Task:
        title: str = prop(required=True)
        status: Optional[str] = prop()
        priority: Optional[str] = prop()
```

**Behavior:** Creates all nodes, no deduplication.

---

### Level 2: Add Search/Matching (Unique Identifiers)

Define how to find existing nodes to avoid duplicates.

```python
from papr.sdk import schema, node, prop, exact, semantic

@schema("project_with_search")
class ProjectSchema:

    @node
    class Task:
        # Search strategies (in priority order)
        id: str = prop(search=exact())              # Try exact ID first
        title: str = prop(required=True, search=semantic(0.85))  # Then semantic title
        status: Optional[str] = prop()
        priority: Optional[str] = prop()

    @node
    class Person:
        email: str = prop(search=exact())           # Exact email match
        name: str = prop(required=True, search=semantic(0.90))  # High-threshold name
        role: Optional[str] = prop()
```

**Behavior:**
- Task: Search by exact ID, then semantic title. Create if not found.
- Person: Search by exact email, then semantic name. Create if not found.

---

### Level 3: Add Creation Policy (Controlled Vocabulary)

Control which node types can create new entries.

```python
from papr.sdk import schema, node, prop, exact, semantic
from papr.sdk import link_only, auto_create  # Creation policy decorators

@schema("project_with_policies")
class ProjectSchema:

    @node
    @auto_create  # Can create new tasks
    class Task:
        id: str = prop(search=exact())
        title: str = prop(required=True, search=semantic(0.85))
        status: Optional[str] = prop()
        priority: Optional[str] = prop()

    @node
    @link_only  # NEVER create - only link to existing people
    class Person:
        email: str = prop(search=exact())
        name: str = prop(required=True, search=semantic(0.90))
        role: Optional[str] = prop()

    @node
    @link_only  # Pre-loaded categories
    class Category:
        id: str = prop(search=exact())
        name: str = prop(search=semantic(0.85))
```

**Behavior:**
- Task: Search, create if not found (`@auto_create`)
- Person: Search, skip if not found (`@link_only`)
- Category: Search, skip if not found (`@link_only`)

---

### Level 4: Add Conditional Logic (`when`)

Apply constraints only when conditions match.

```python
from papr.sdk import schema, node, prop, exact, semantic, constraint
from papr.sdk import link_only, auto_create
from papr.sdk import And, Or, Not  # Logical operators

@schema("project_with_conditions")
class ProjectSchema:

    @node
    @auto_create
    @constraint(
        when={"priority": "critical"}  # Only apply to critical tasks
    )
    class Task:
        id: str = prop(search=exact())
        title: str = prop(required=True, search=semantic(0.85))
        status: Optional[str] = prop()
        priority: Optional[str] = prop()
        flagged: Optional[bool] = prop()
        sla_hours: Optional[int] = prop()

    @node
    @link_only
    @constraint(
        when={
            "_and": [
                {"role": "engineer"},
                {"_not": {"status": "inactive"}}
            ]
        }
    )
    class Person:
        email: str = prop(search=exact())
        name: str = prop(required=True, search=semantic(0.90))
        role: Optional[str] = prop()
        status: Optional[str] = prop()
```

**Behavior:**
- Task constraint only applies when `priority="critical"`
- Person constraint only applies when `role="engineer"` AND `status != "inactive"`

---

### Level 5: Add Property Values (`set`)

Automatically set property values when constraints apply.

```python
from papr.sdk import schema, node, prop, exact, semantic, constraint
from papr.sdk import link_only, auto_create
from papr.sdk import Auto  # For LLM extraction

@schema("project_full_policies")
class ProjectSchema:

    @node
    @auto_create
    @constraint(
        when={"priority": "critical"},
        set={
            "flagged": True,           # Exact value
            "sla_hours": 4,            # Exact value
            "reviewed_by": Auto()      # LLM extracts from content
        }
    )
    class Task:
        id: str = prop(search=exact())
        title: str = prop(required=True, search=semantic(0.85))
        status: Optional[str] = prop()
        priority: Optional[str] = prop()
        flagged: Optional[bool] = prop()
        sla_hours: Optional[int] = prop()
        reviewed_by: Optional[str] = prop()

    @node
    @link_only
    @constraint(
        when={"role": "engineer"},
        set={
            "last_mentioned": Auto(),  # LLM extracts timestamp/context
            "mention_count": Auto()    # LLM counts mentions
        }
    )
    class Person:
        email: str = prop(search=exact())
        name: str = prop(required=True, search=semantic(0.90))
        role: Optional[str] = prop()
        last_mentioned: Optional[str] = prop()
        mention_count: Optional[int] = prop()
```

**Behavior:**
- Critical tasks: Auto-flagged, 4-hour SLA, reviewer extracted by LLM
- Engineers: Last mention and count tracked

---

## Memory-Level Overrides

Schema defines defaults. Memory requests can override per-call.

### Override Example

```python
# Schema defines Person as link_only (never create)
# But for this specific memory, we want to create if not found

await client.add_memory(
    content="New contractor Alex joined the team",
    external_user_id="manager",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Person",
                create="auto",  # Override: allow creation for this memory
                search=SearchConfig(properties=[
                    PropertyMatch(name="name", mode="semantic", threshold=0.95)
                ]),
                set={"role": "contractor", "onboarded": True}
            )
        ]
    )
)
```

### Override with `link_to` Shorthand

```python
# Simple override using link_to shorthand
await client.add_memory(
    content="Critical bug in authentication module",
    external_user_id="developer",
    link_to={
        "Task:title": {
            "set": {"priority": "critical", "flagged": True},
            "when": {"status": "open"}
        }
    }
)
```

### Override Precedence

| Level | Precedence | Scope |
|-------|------------|-------|
| Memory-level `node_constraints` | Highest | This memory only |
| Memory-level `link_to` | High | This memory only |
| Schema-level `constraint` | Default | All memories using schema |

---

## Complete Real-World Example

### Security Monitoring Schema

```python
from papr.sdk import schema, node, edge, prop, exact, semantic, constraint
from papr.sdk import link_only, auto_create, Auto
from typing import Optional
from datetime import datetime

@schema("security_monitoring")
class SecuritySchema:
    """Security call analysis with controlled vocabularies."""

    # ═══════════════════════════════════════════════════════════════
    # CONTROLLED VOCABULARY (Pre-loaded, never create)
    # ═══════════════════════════════════════════════════════════════

    @node
    @link_only
    class TacticDef:
        """MITRE ATT&CK tactics - pre-loaded reference data."""
        id: str = prop(search=exact())           # TA0001, TA0043
        name: str = prop(search=semantic(0.90))  # "Defense Evasion"
        severity: Optional[str] = prop()

    @node
    @link_only
    @constraint(
        when={"category": "access_control"},
        set={"priority": "high"}
    )
    class SecurityBehavior:
        """Security policies - pre-loaded rules."""
        id: str = prop(search=exact())           # SB001, SB080
        name: str = prop(required=True, search=semantic(0.85))
        category: Optional[str] = prop()         # access_control, data_protection
        priority: Optional[str] = prop()

    # ═══════════════════════════════════════════════════════════════
    # DYNAMIC ENTITIES (Created per call)
    # ═══════════════════════════════════════════════════════════════

    @node
    @auto_create
    class Conversation:
        """Call session - created per transcript."""
        call_id: str = prop(required=True, search=exact())
        timestamp: Optional[datetime] = prop()
        risk_level: Optional[str] = prop()

    @node
    @auto_create
    @constraint(
        when={"severity": "critical"},
        set={
            "flagged": True,
            "requires_review": True,
            "escalation_reason": Auto()  # LLM explains why
        }
    )
    class DetectedTactic:
        """Tactics detected in call - always created."""
        tactic_name: str = prop(search=semantic(0.85))
        severity: Optional[str] = prop()
        context: Optional[str] = prop()
        flagged: Optional[bool] = prop()
        requires_review: Optional[bool] = prop()
        escalation_reason: Optional[str] = prop()

    # ═══════════════════════════════════════════════════════════════
    # EDGES
    # ═══════════════════════════════════════════════════════════════

    # DetectedTactic maps to a TacticDef
    maps_to = edge(
        DetectedTactic >> TacticDef,
        search=(TacticDef.id.exact(), TacticDef.name.semantic(0.90)),
        create="never"  # Only link to existing tactics
    )

    # SecurityBehavior mitigates TacticDef
    mitigates = edge(
        SecurityBehavior >> TacticDef,
        search=(TacticDef.id.exact(), TacticDef.name.semantic(0.90)),
        create="never"
    )
```

### Using the Schema

```python
# Register schema
await client.register_schema(SecuritySchema)

# Add memory - schema handles everything
await client.add_memory(
    content="""
    Call transcript: Caller claimed they lost their phone and needed
    immediate access to their account. Agent verified identity using
    security questions before granting access.
    """,
    external_user_id="call_analyzer"
    # No policy needed! Schema defines:
    # - TacticDef: link_only (matches "Defense Evasion")
    # - SecurityBehavior: link_only (matches "Verify Identity")
    # - Conversation: auto_create (new call session)
    # - DetectedTactic: auto_create with flagging if critical
)
```

---

## Summary: When to Use What

| Scenario | Solution |
|----------|----------|
| Pre-populated reference data (users, categories, tactics) | `@link_only` + `search=exact()` |
| Dynamic entities created from content | `@auto_create` |
| Deduplication by unique ID | `prop(search=exact())` |
| Deduplication by similar name/title | `prop(search=semantic(0.85))` |
| Different behavior for specific values | `@constraint(when={...})` |
| Auto-set properties when conditions met | `@constraint(set={...})` |
| Override schema defaults for one memory | `memory_policy.node_constraints` or `link_to` |

---

## Decorator Quick Reference

```python
# Creation policy
@link_only              # Never create (controlled vocabulary)
@auto_create            # Create if not found (default)
@controlled_vocabulary  # Alias for @link_only

# Constraints with conditions and values
@constraint(
    when={"field": "value"},           # Simple condition
    when={"_and": [{...}, {...}]},     # AND logic
    when={"_or": [{...}, {...}]},      # OR logic
    when={"_not": {...}},              # Negation
    set={"field": "exact_value"},      # Exact value
    set={"field": Auto()}              # LLM extraction
)

# Property search modes
prop(search=exact())           # Exact match (IDs, emails)
prop(search=semantic(0.85))    # Semantic similarity
prop(search=fuzzy(0.80))       # Fuzzy string match
```

---

## Related Documentation

- [NODE_CONSTRAINTS_API.md](./NODE_CONSTRAINTS_API.md) - Full API reference
- [DEEPTRUST_EXAMPLE.md](./DEEPTRUST_EXAMPLE.md) - Complete security example
- [DX_IMPROVEMENTS_PROPOSAL.md](./DX_IMPROVEMENTS_PROPOSAL.md) - Design rationale
