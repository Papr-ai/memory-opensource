# Node Constraints API Reference

*Updated: January 2026*

## Overview

Node constraints control how AI-extracted entities become nodes in your knowledge graph. They can be defined at two levels:

1. **Schema level**: Inside `UserNodeType.constraint` - defines defaults for all nodes of that type
2. **Memory level**: In `memory_policy.node_constraints[]` - overrides schema defaults per memory

---

## Quick Start - Shorthand Helpers

For common use cases, use the shorthand helpers to reduce boilerplate:

### PropertyMatch Shortcuts

```python
# Instead of verbose:
PropertyMatch(name="id", mode="exact", value="TASK-123")

# Use shorthand:
PropertyMatch.exact("id", "TASK-123")
PropertyMatch.semantic("title", threshold=0.9)
PropertyMatch.fuzzy("name", 0.8)
```

### SearchConfig String Shorthand

```python
# Simple case - strings become exact matches:
SearchConfig(properties=["id", "email"])

# Equivalent to:
SearchConfig(properties=[
    PropertyMatch.exact("id"),
    PropertyMatch.exact("email")
])

# Mix strings with PropertyMatch for flexibility:
SearchConfig(properties=[
    "id",                                    # String -> exact match
    PropertyMatch.semantic("title", 0.85)    # Full control
])
```

### NodeConstraint Shorthand Constructors

```python
# Controlled vocabulary (never create, only link to existing)
NodeConstraint.for_controlled_vocabulary(
    "Person",
    ["email", PropertyMatch.semantic("name", 0.9)]
)

# Or use link_only shorthand directly
NodeConstraint(
    node_type="Person",
    link_only=True,  # Equivalent to create="never"
    search=SearchConfig(properties=["email", PropertyMatch.semantic("name", 0.9)])
)

# Update a specific node by ID
NodeConstraint.for_update(
    "Task",
    "TASK-123",
    {"status": {"mode": "auto"}}
)

# Semantic search and update
NodeConstraint.for_semantic_search(
    "Task",
    "title",
    "authentication bug",
    threshold=0.85,
    set_properties={"status": {"mode": "auto"}}
)
```

---

## Model Structure

### NodeConstraint

```python
class NodeConstraint(BaseModel):
    # === WHAT ===
    node_type: Optional[str]  # Required at memory level, implicit at schema level

    # === WHEN (with logical operators) ===
    when: Optional[Dict[str, Any]]  # Conditional with _and, _or, _not support

    # === CREATION ===
    create: Literal["auto", "never"] = "auto"
    link_only: bool = False  # Shorthand for create="never" (controlled vocabulary)

    # === SELECTION (property-based matching) ===
    search: Optional[SearchConfig]  # Uses PropertyMatch list or string shorthand

    # === VALUES ===
    set: Optional[Dict[str, SetValue]]
```

> **Note:** When `link_only=True`, the `create` field is automatically set to `"never"`.
> This is equivalent to the `@link_only` decorator in schema definitions.

### SearchConfig (Property-Based Matching)

```python
class SearchConfig(BaseModel):
    # Properties to match on (in priority order - first match wins)
    # Accepts strings (-> exact match) or PropertyMatch objects
    properties: Optional[List[Union[str, PropertyMatch]]] = None

    # Default settings when property doesn't specify
    mode: Literal["semantic", "exact", "fuzzy"] = "semantic"
    threshold: float = 0.85
```

### PropertyMatch

```python
class PropertyMatch(BaseModel):
    name: str                    # Property name (e.g., "id", "email", "title")
    mode: Literal["exact", "semantic", "fuzzy"] = "exact"
    threshold: float = 0.85      # For semantic/fuzzy only
    value: Optional[Any] = None  # Runtime value override

    # Shorthand class methods:
    @classmethod
    def exact(cls, name: str, value: Any = None) -> PropertyMatch
    @classmethod
    def semantic(cls, name: str, threshold: float = 0.85, value: Any = None) -> PropertyMatch
    @classmethod
    def fuzzy(cls, name: str, threshold: float = 0.85, value: Any = None) -> PropertyMatch
```

**Key Design Decision:** `node_id` was removed from SearchConfig. Use `PropertyMatch.exact()` with value instead:

```python
# Old approach (removed)
"search": {"node_id": "TASK-123"}

# New approach - shorthand
SearchConfig(properties=[PropertyMatch.exact("id", "TASK-123")])

# Or full form
"search": {"properties": [{"name": "id", "mode": "exact", "value": "TASK-123"}]}
```

### SetValue (Property Value Types)

```python
# SetValue can be:
# 1. Exact value: str, int, float, bool, list, dict
# 2. Auto-extract config: PropertyValue

class PropertyValue(BaseModel):
    mode: Literal["auto"] = "auto"
    text_mode: Literal["replace", "append", "merge"] = "replace"
```

---

## Schema vs Memory Level Usage

### Schema Level (Inline in UserNodeType)

At schema level, `node_type` is **implicit** - it's taken from the parent `UserNodeType.name`:

```python
UserNodeType(
    name="Task",
    label="Task",
    properties={
        "id": PropertyDefinition(type="string"),
        "title": PropertyDefinition(type="string", required=True),
        "status": PropertyDefinition(type="string")
    },
    constraint=NodeConstraint(
        # node_type not needed - implicit from parent "Task"
        search=SearchConfig(
            properties=[
                PropertyMatch(name="id", mode="exact"),
                PropertyMatch(name="title", mode="semantic", threshold=0.85)
            ]
        ),
        create="auto"
    )
)
```

**What this tells the developer:**
- "id and title are unique identifiers for Tasks"
- "Try exact id match first, then semantic title match"
- "Create new Tasks if not found"

### Memory Level (Override for Specific Memory)

At memory level, `node_type` is **required**:

```python
MemoryPolicy(
    node_constraints=[
        NodeConstraint(
            node_type="Task",  # Required at memory level
            search=SearchConfig(
                properties=[
                    # I know the exact task ID
                    PropertyMatch(name="id", mode="exact", value="TASK-123")
                ]
            ),
            set={"status": {"mode": "auto"}}
        )
    ]
)
```

---

## Field Reference

### `node_type`

**Type:** `Optional[str]`
**At Schema Level:** Implicit (from parent `UserNodeType.name`)
**At Memory Level:** Required

```python
# Schema level - implicit
UserNodeType(name="Task", constraint=NodeConstraint(...))

# Memory level - required
NodeConstraint(node_type="Task", ...)
```

---

### `when` (Conditional with Logical Operators)

**Type:** `Dict[str, Any]`
**Default:** None (applies to all nodes)
**Description:** Conditions with full logical operator support

**Logical Operators:**
- `_and`: All conditions must match
- `_or`: At least one must match
- `_not`: Negation

**Examples:**

```python
# Simple match
"when": {"priority": "high"}

# AND: All must match
"when": {
    "_and": [
        {"priority": "high"},
        {"status": "active"}
    ]
}

# OR: Any must match
"when": {
    "_or": [
        {"status": "active"},
        {"status": "pending"}
    ]
}

# NOT: Negation
"when": {
    "_not": {"status": "completed"}
}

# Complex: priority=high AND (status=active OR urgent=true)
"when": {
    "_and": [
        {"priority": "high"},
        {
            "_or": [
                {"status": "active"},
                {"urgent": True}
            ]
        }
    ]
}
```

---

### `create`

**Type:** `Literal["auto", "never"]`
**Default:** `"auto"`

| Value | Description |
|-------|-------------|
| `"auto"` | Create new node if no match found (default) |
| `"never"` | Only link to existing nodes (controlled vocabulary) |

---

### `link_only` (Shorthand for Controlled Vocabulary)

**Type:** `bool`
**Default:** `False`
**Description:** Shorthand for `create="never"`. When `True`, only links to existing nodes (controlled vocabulary).

This is equivalent to the `@link_only` decorator in schema definitions.

```python
# These are equivalent:
NodeConstraint(link_only=True)
NodeConstraint(create="never")

# At schema level with decorator:
@node
@link_only
class TacticDef:
    id: str = prop(search=exact())
    name: str = prop(search=semantic(0.90))
```

**When to use `link_only`:**
- Pre-populated reference data (MITRE tactics, categories, status codes)
- External systems of record (users from IdP, products from catalog)
- Controlled vocabularies where you never want to create new entries

**Example:**
```python
# Using link_only shorthand
UserNodeType(
    name="TacticDef",
    label="Tactic Definition",
    properties={...},
    link_only=True,  # Shorthand - creates constraint with create="never"
    constraint=NodeConstraint(
        search=SearchConfig(properties=[
            PropertyMatch(name="id", mode="exact"),
            PropertyMatch(name="name", mode="semantic", threshold=0.90)
        ])
    )
)

# Or directly on constraint
NodeConstraint(
    node_type="Person",
    link_only=True,  # Equivalent to create="never"
    search=SearchConfig(properties=[
        PropertyMatch(name="email", mode="exact")
    ])
)
```

---

### `search` (Property-Based Matching)

**Type:** `SearchConfig`
**Description:** Defines unique identifiers AND how to match them

**Key Concept:** The `properties` list defines:
1. Which properties are unique identifiers
2. How to match on each (exact, semantic, fuzzy)
3. Priority order (first match wins)

**Examples:**

```python
# Schema level - define matching strategy
"search": {
    "properties": [
        {"name": "id", "mode": "exact"},
        {"name": "title", "mode": "semantic", "threshold": 0.85}
    ]
}

# Memory level - with specific value (replaces old node_id)
"search": {
    "properties": [
        {"name": "id", "mode": "exact", "value": "TASK-123"}
    ]
}

# Memory level - semantic search for specific text
"search": {
    "properties": [
        {"name": "title", "mode": "semantic", "value": "authentication bug"}
    ]
}
```

---

### `set`

**Type:** `Dict[str, SetValue]`
**Description:** Property values to set (exact or auto-extract)

```python
# Exact values
"set": {"workspace_id": "ws_123", "team": "backend"}

# Auto-extract
"set": {"status": {"mode": "auto"}, "priority": {"mode": "auto"}}

# Mixed
"set": {
    "workspace_id": "ws_123",      # Exact
    "status": {"mode": "auto"},    # AI extracts
    "summary": {"mode": "auto", "text_mode": "merge"}  # AI merges
}
```

---

## Complete Examples

### Example 1: Schema Level - Task with Multiple Match Strategies

```python
UserNodeType(
    name="Task",
    label="Task",
    properties={
        "id": PropertyDefinition(type="string"),
        "title": PropertyDefinition(type="string", required=True),
        "status": PropertyDefinition(type="string")
    },
    constraint=NodeConstraint(
        search=SearchConfig(
            properties=[
                PropertyMatch(name="id", mode="exact"),
                PropertyMatch(name="title", mode="semantic", threshold=0.85)
            ]
        ),
        create="auto"
    )
)
```

**Developer reads:** "id and title are unique. Try exact id first, then semantic title. Create if not found."

---

### Example 2: Schema Level - Controlled Vocabulary (Never Create)

```python
# Option 1: Using create="never"
UserNodeType(
    name="Person",
    label="Person",
    properties={
        "email": PropertyDefinition(type="string"),
        "name": PropertyDefinition(type="string", required=True)
    },
    constraint=NodeConstraint(
        search=SearchConfig(
            properties=[
                PropertyMatch(name="email", mode="exact"),
                PropertyMatch(name="name", mode="semantic", threshold=0.90)
            ]
        ),
        create="never"  # Controlled vocabulary
    )
)

# Option 2: Using link_only shorthand (equivalent to above)
UserNodeType(
    name="Person",
    label="Person",
    properties={
        "email": PropertyDefinition(type="string"),
        "name": PropertyDefinition(type="string", required=True)
    },
    link_only=True,  # Shorthand - auto-creates constraint with create="never"
    constraint=NodeConstraint(
        search=SearchConfig(
            properties=[
                PropertyMatch(name="email", mode="exact"),
                PropertyMatch(name="name", mode="semantic", threshold=0.90)
            ]
        )
    )
)
```

**Developer reads:** "email and name are unique. Try exact email first, then semantic name. Never create new."

---

### Example 3: Memory Level - Select Specific Node by ID

```python
MemoryPolicy(
    node_constraints=[
        NodeConstraint(
            node_type="Task",
            search=SearchConfig(
                properties=[
                    PropertyMatch(name="id", mode="exact", value="TASK-123")
                ]
            ),
            set={"status": {"mode": "auto"}}
        )
    ]
)
```

---

### Example 4: Memory Level - Semantic Search with Value

```python
MemoryPolicy(
    node_constraints=[
        NodeConstraint(
            node_type="Task",
            search=SearchConfig(
                properties=[
                    PropertyMatch(name="title", mode="semantic", value="authentication bug")
                ]
            )
        )
    ]
)
```

---

### Example 5: Conditional with Logical Operators

```python
NodeConstraint(
    node_type="Task",
    when={
        "_and": [
            {"priority": "high"},
            {"_not": {"status": "completed"}}
        ]
    },
    create="never",
    set={"urgent": True}
)
```

---

### Example 6: Complex Real-World Example

```python
# In schema:
UserNodeType(
    name="Project",
    label="Project",
    properties={
        "name": PropertyDefinition(type="string", required=True),
        "status": PropertyDefinition(type="string"),
        "priority": PropertyDefinition(type="string")
    },
    constraint=NodeConstraint(
        search=SearchConfig(
            properties=[
                PropertyMatch(name="name", mode="semantic", threshold=0.90)
            ]
        ),
        create="auto"
    )
)

# In memory request (override with specific value):
MemoryPolicy(
    schema_id="project_management",
    node_constraints=[
        NodeConstraint(
            node_type="Project",
            search=SearchConfig(
                properties=[
                    PropertyMatch(name="name", mode="semantic", value="Project Alpha")
                ]
            ),
            set={
                "status": {"mode": "auto"},
                "priority": {"mode": "auto"},
                "summary": {"mode": "auto", "text_mode": "merge"}
            }
        )
    ]
)
```

---

## How Matching Works

**Scenario:** Standup - "John is working on the authentication bug"

**Schema defines:**
```python
constraint=NodeConstraint(
    search=SearchConfig(
        properties=[
            PropertyMatch(name="id", mode="exact"),
            PropertyMatch(name="title", mode="semantic", threshold=0.85)
        ]
    )
)
```

**Flow:**
1. **LLM extracts:** `{type: "Task", properties: {title: "authentication bug"}}`
2. **System checks strategies in order:**
   - `id` exact match? No id extracted → **skip**
   - `title` semantic match? Search "authentication bug" → finds "Fix authentication bug" (0.91) → **MATCH**
3. **Result:** Link to existing task

**If ID was mentioned:** "TASK-123 is complete"
1. **LLM extracts:** `{type: "Task", properties: {id: "TASK-123", status: "complete"}}`
2. **System checks:**
   - `id` exact match? Yes → **FOUND immediately**
3. **Result:** No semantic search needed, update existing task

---

## Migration from Old API

### Removed: `search.node_id`

```python
# Old
"search": {"node_id": "TASK-123"}

# New - use PropertyMatch with value
"search": {"properties": [{"name": "id", "mode": "exact", "value": "TASK-123"}]}
```

### Removed: `search.properties` as `List[str]`

```python
# Old
"search": {"mode": "exact", "properties": ["name", "email"]}

# New - PropertyMatch with per-property config
"search": {
    "properties": [
        {"name": "name", "mode": "exact"},
        {"name": "email", "mode": "exact"}
    ]
}
```

### Removed: `unique_identifiers` in UserNodeType

```python
# Old
UserNodeType(
    name="Task",
    unique_identifiers=["id", "title"],
    ...
)

# New - use constraint.search.properties
UserNodeType(
    name="Task",
    constraint=NodeConstraint(
        search=SearchConfig(
            properties=[
                PropertyMatch(name="id", mode="exact"),
                PropertyMatch(name="title", mode="semantic")
            ]
        )
    ),
    ...
)
```

---

## Best Practices

### 1. Define Constraints at Schema Level

Put matching strategies in `UserNodeType.constraint` for reuse:

```python
# Good: Define once, use everywhere
UserNodeType(
    name="Task",
    constraint=NodeConstraint(
        search=SearchConfig(properties=[...]),
        create="auto"
    )
)
```

### 2. Use Memory Level for Overrides Only

Only use `memory_policy.node_constraints` when you need to override:

```python
# Good: Override with specific value
MemoryPolicy(
    node_constraints=[
        NodeConstraint(
            node_type="Task",
            search=SearchConfig(properties=[
                PropertyMatch(name="id", mode="exact", value="TASK-123")
            ])
        )
    ]
)
```

### 3. Order PropertyMatch by Specificity

More specific matchers should come first:

```python
# Good: Exact ID first, then semantic title
properties=[
    PropertyMatch(name="id", mode="exact"),
    PropertyMatch(name="title", mode="semantic")
]
```

### 4. Use Appropriate Thresholds

| Use Case | Threshold |
|----------|-----------|
| Exact identifiers (id, email) | Not needed (exact mode) |
| Names | 0.85-0.95 |
| Descriptions | 0.75-0.85 |

---

## Related Documentation

- [Memory Policy API](./MEMORY_POLICY_API.md) - Full memory policy reference
- [Migration Guide](./MIGRATION_GUIDE.md) - Migrating from old API
- [API Design Principles](../../architecture/API_DESIGN_PRINCIPLES.md) - Design decisions
