# DX Improvements Proposal: Radical Simplification

*Proposal Date: January 2026*
*Goal: 10x improvement in Time-to-First-Success*

---

## Executive Summary

This proposal clarifies the mental model and introduces the `link_to` shorthand that preserves the power of `memory_policy.node_constraints` while making common operations trivial.

### Key Insight: Two Different Use Cases

Developers have **two distinct needs**:

1. **Auto-extraction with policies** (80% case) - LLM extracts entities from content, schema policies control behavior
2. **Explicit linking with overrides** (20% case) - Developer knows what entity to link to and may need custom constraints

### The `link_to` Shorthand: Three Forms

| Form | Use Case | Example |
|------|----------|---------|
| **String** | Single entity, just link | `link_to="Task:title"` |
| **List** | Multiple entities, just link | `link_to=["Task:title", "Person:email"]` |
| **Dict** | Per-entity constraints (set, when, create) | `link_to={"Task:title": {"set": {...}}}` |

### Mapping: `link_to` → `memory_policy.node_constraints`

| `link_to` | Expands To |
|-----------|------------|
| `"Task:title"` | `node_constraints=[NodeConstraint(node_type="Task", search=SearchConfig(properties=[PropertyMatch(name="title", mode="semantic")]))]` |
| `"Task:id=TASK-123"` | `...PropertyMatch(name="id", mode="exact", value="TASK-123")...` |
| `"Task:title~auth bug"` | `...PropertyMatch(name="title", mode="semantic", value="auth bug")...` |
| `{"Task:title": {"set": {...}}}` | `NodeConstraint(..., set={...})` |
| `{"Task:title": {"when": {...}}}` | `NodeConstraint(..., when={...})` |
| `{"Task:title": {"create": "never"}}` | `NodeConstraint(..., create="never")` |
| `{"Task:title": {"link_only": true}}` | `NodeConstraint(..., link_only=True)` ≡ `create="never"` |

### Shorthand: `link_only` Field

The `link_only` field is a shorthand for `create="never"`:

```python
# These are equivalent:
{"Task:title": {"create": "never"}}
{"Task:title": {"link_only": true}}

# At schema level (dict-based):
UserNodeType(name="Person", link_only=True, ...)  # Shorthand field on UserNodeType

# At schema level (class-based decorators):
@node
@link_only  # Decorator equivalent
class Person: ...

@node
@controlled_vocabulary  # Alias for @link_only
class Person: ...
```

**Target Outcomes:**
- Time-to-first-success: 30 minutes → 5 minutes
- 80% case: 0 lines of policy code (schema handles it)
- Policy overrides: 1 line instead of 15

---

## Part 1: Understanding the Mental Model

### The 80% Case: Auto-Extraction (Schema Handles Everything)

**The whole point of auto mode:** LLM extracts entities based on your schema. You don't specify what to extract.

```python
# Schema defines what entities exist and their policies
schema = UserGraphSchema(
    name="security_monitoring",
    node_types={
        "SecurityPolicy": UserNodeType(
            name="SecurityPolicy",
            properties={"name": PropertyDefinition(type="string")},
            link_only=True,  # Shorthand for create="never" - controlled vocabulary
            constraint=NodeConstraint(
                search=SearchConfig(properties=[
                    PropertyMatch.semantic("name", 0.85)
                ])
            )
        ),
        "User": UserNodeType(
            name="User",
            properties={"username": PropertyDefinition(type="string")},
            constraint=NodeConstraint(create="auto")  # Can create new users
        )
    }
)

# 80% Case: Just add memory - LLM extracts, schema policies apply
await client.add_memory(
    content="User john_doe attempted SQL injection, blocked by WAF policy",
    external_user_id="alice"
    # That's it! No need to specify:
    # - What entities to extract (LLM figures it out from schema)
    # - How to match SecurityPolicy (schema says semantic on name)
    # - Whether to create SecurityPolicy (schema says never)
)
```

**What happens:**
1. LLM reads content, sees schema has SecurityPolicy and User
2. LLM extracts: `SecurityPolicy(name="WAF policy")`, `User(username="john_doe")`
3. System applies schema constraints:
   - SecurityPolicy: `create="never"` → search for existing, link if found, skip if not
   - User: `create="auto"` → search for existing, create if not found
4. Memory is linked to matched/created entities

**Developer doesn't need to specify anything about extraction or policies!**

### The 15% Case: Override Policies for This Memory

Sometimes you need different behavior than schema defaults:

```python
# Override: For this memory, also update status on matched tasks
await client.add_memory(
    content="Sprint complete: auth bug fixed, API review done",
    external_user_id="alice",
    link_to={
        "Task:title": {"set": {"status": "completed"}}  # Override: update status
    }
)
```

### The 5% Case: Explicit Linking

When you KNOW exactly what entity to link to (e.g., user selected from dropdown):

```python
# Explicit: Link to specific task I know about
await client.add_memory(
    content="Notes about this task...",
    external_user_id="alice",
    link_to="Task:id=TASK-123"  # I know the exact ID
)
```

---

## Part 2: The Full API (What We're Simplifying)

### Schema Level: `UserNodeType.constraint`

Define default behavior for each node type:

```python
schema = UserGraphSchema(
    name="project_management",
    node_types={
        "Task": UserNodeType(
            name="Task",
            properties={
                "id": PropertyDefinition(type="string"),
                "title": PropertyDefinition(type="string", required=True),
                "status": PropertyDefinition(type="string")
            },
            constraint=NodeConstraint(
                # node_type implicit from parent
                search=SearchConfig(properties=[
                    PropertyMatch(name="id", mode="exact"),
                    PropertyMatch(name="title", mode="semantic", threshold=0.85)
                ]),
                create="auto"
            )
        ),
        "Person": UserNodeType(
            name="Person",
            properties={
                "email": PropertyDefinition(type="string"),
                "name": PropertyDefinition(type="string", required=True)
            },
            link_only=True,  # Shorthand for create="never" - controlled vocabulary
            constraint=NodeConstraint(
                search=SearchConfig(properties=[
                    PropertyMatch(name="email", mode="exact"),
                    PropertyMatch(name="name", mode="semantic", threshold=0.90)
                ])
            )
        )
    }
)
```

### Memory Level: `memory_policy.node_constraints`

Override schema defaults per-memory:

```python
await client.add_memory(
    content="Sprint complete: auth bug fixed by John",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",  # Required at memory level
                search=SearchConfig(
                    properties=[PropertyMatch(name="title", mode="semantic")]
                ),
                set={"status": "completed"},
                when={"priority": "high"}
            ),
            NodeConstraint(
                node_type="Person",
                link_only=True,  # Shorthand for create="never"
                search=SearchConfig(
                    properties=[PropertyMatch(name="name", mode="semantic")]
                ),
                create="never"
            )
        ]
    )
)
```

**Problem:** This is 20+ lines for a common operation!

---

## Part 3: The `link_to` Shorthand

### Design Principle

`link_to` is a **shorthand for `memory_policy.node_constraints`** - same power, fewer tokens.

### Three Forms of `link_to`

#### Form 1: String (Single Entity, Just Link)

```python
# Shorthand
link_to="Task:title"

# Expands to
memory_policy=MemoryPolicy(
    node_constraints=[
        NodeConstraint(
            node_type="Task",
            search=SearchConfig(properties=[PropertyMatch(name="title", mode="semantic")])
        )
    ]
)
```

#### Form 2: List (Multiple Entities, Just Link)

```python
# Shorthand
link_to=["Task:title", "Person:email"]

# Expands to
memory_policy=MemoryPolicy(
    node_constraints=[
        NodeConstraint(node_type="Task", search=SearchConfig(properties=[PropertyMatch(name="title", mode="semantic")])),
        NodeConstraint(node_type="Person", search=SearchConfig(properties=[PropertyMatch(name="email", mode="exact")]))
    ]
)
```

#### Form 3: Dict (Per-Entity Constraints)

```python
# Shorthand
link_to={
    "Task:title": {"set": {"status": "completed"}, "when": {"priority": "high"}},
    "Person:email": {"create": "never"}
}

# Expands to
memory_policy=MemoryPolicy(
    node_constraints=[
        NodeConstraint(
            node_type="Task",
            search=SearchConfig(properties=[PropertyMatch(name="title", mode="semantic")]),
            set={"status": "completed"},
            when={"priority": "high"}
        ),
        NodeConstraint(
            node_type="Person",
            search=SearchConfig(properties=[PropertyMatch(name="email", mode="exact")]),
            create="never"
        )
    ]
)
```

---

### `link_to` Key Syntax (DSL)

The dictionary key encodes `Type:property` with optional operator and value:

| Key Syntax | Meaning | Equivalent |
|------------|---------|------------|
| `Task:title` | Semantic match on title | `PropertyMatch(name="title", mode="semantic")` |
| `Task:id=TASK-123` | Exact match with value | `PropertyMatch(name="id", mode="exact", value="TASK-123")` |
| `Task:title~auth bug` | Semantic match with value | `PropertyMatch(name="title", mode="semantic", value="auth bug")` |
| `Person:email=john@x.com` | Exact match with value | `PropertyMatch(name="email", mode="exact", value="john@x.com")` |

### Special References

| Reference | Meaning |
|-----------|---------|
| `$this` | The memory being created |
| `$previous` | User's most recent memory |
| `$context:N` | Last N memories in conversation |

```python
# Link to previous memory (conversation flow)
link_to={"$previous": {}}

# Link to last 3 messages for context
link_to={"$context:3": {}}

# Combine entity linking with context
link_to={
    "Task:title": {"set": {"status": "done"}},
    "$previous": {}  # Also link to previous memory
}
```

### `link_to` Value Options

When using dict form, each value can include:

| Field | Type | Description |
|-------|------|-------------|
| `set` | `Dict[str, Any]` | Properties to set (exact value or `{"mode": "auto"}`) |
| `when` | `Dict[str, Any]` | Condition for this constraint to apply |
| `create` | `"auto" \| "never"` | Override creation policy |

---

### Complete Examples

#### Example 1: Simple Link (80% Case)

```python
# Just add memory - schema handles extraction
await client.add_memory(
    content="John fixed the authentication bug",
    external_user_id="alice"
)
```

#### Example 2: Link to Specific Entity (Exact Match)

```python
await client.add_memory(
    content="Notes about this task",
    external_user_id="alice",
    link_to="Task:id=TASK-123"
)
```

#### Example 3: Link with Semantic Search

```python
await client.add_memory(
    content="Additional context about the authentication bug",
    external_user_id="alice",
    link_to="Task:title~authentication bug"  # Explicit semantic search
)
```

#### Example 4: Link to Person by Email (Exact Match)

```python
await client.add_memory(
    content="Feedback for John",
    external_user_id="alice",
    link_to="Person:email=john@acme.com"
)
```

#### Example 5: Link and Update Properties

```python
await client.add_memory(
    content="Sprint complete: auth bug fixed",
    external_user_id="alice",
    link_to={"Task:title": {"set": {"status": "completed"}}}
)
```

#### Example 6: Multiple Entities with Different Constraints

```python
await client.add_memory(
    content="John completed the API review task",
    external_user_id="alice",
    link_to={
        "Task:title": {
            "set": {"status": "completed"},
            "when": {"priority": "high"}
        },
        "Person:name": {"create": "never"}
    }
)
```

#### Example 7: With Context Linking

```python
# Link to previous memory
await client.add_memory(
    content="Continuing our discussion...",
    external_user_id="alice",
    link_to={"$previous": {}}
)

# Link to last 3 messages for context
await client.add_memory(
    content="Follow-up on security incident",
    external_user_id="alice",
    link_to={
        "SecurityPolicy:name": {"create": "never"},
        "$context:3": {}  # Include conversation context
    }
)
```

#### Example 8: Mixed - Some from Content, Some Explicit

```python
await client.add_memory(
    content="Sprint planning: John will complete the API review",
    external_user_id="alice",
    link_to={
        "Task:title": {},                        # LLM extracts from content
        "Person:email=john@acme.com": {},        # Explicit exact match
        "Project:id=PROJ-123": {}                # Explicit exact match
    }
)
```

---

### `when` Clause Examples

```python
# Simple condition
link_to={"Task:title": {"when": {"priority": "high"}}}

# AND (all must match)
link_to={"Task:title": {"when": {"_and": [{"priority": "high"}, {"status": "active"}]}}}

# OR (any must match)
link_to={"Task:title": {"when": {"_or": [{"status": "active"}, {"status": "pending"}]}}}

# NOT
link_to={"Task:title": {"when": {"_not": {"status": "completed"}}}}

# Complex: priority=high AND NOT completed
link_to={"Task:title": {"when": {
    "_and": [
        {"priority": "high"},
        {"_not": {"status": "completed"}}
    ]
}}}
```

---

### `set` Property Examples

```python
# Exact values
link_to={"Task:title": {"set": {"status": "completed", "priority": "low"}}}

# Auto-extract from content
link_to={"Task:title": {"set": {"status": {"mode": "auto"}}}}

# Mixed: exact + auto-extract
link_to={"Task:title": {"set": {
    "workspace_id": "ws_123",           # Exact value
    "status": {"mode": "auto"},         # LLM extracts from content
    "summary": {"mode": "auto", "text_mode": "merge"}  # LLM merges with existing
}}}
```

---

## Part 4: DeepSeek Example (Real-World Use Case)

### Scenario: Security Monitoring

**Requirements:**
1. Store security events as memories
2. Link to previous 3 messages for context
3. Auto-extract entities (User, SecurityPolicy, Violation)
4. SecurityPolicy is controlled vocabulary (never create new)
5. Update severity on Violations based on AI analysis

### Schema Definition

```python
schema = UserGraphSchema(
    name="security_monitoring",
    node_types={
        "SecurityPolicy": UserNodeType(
            name="SecurityPolicy",
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "category": PropertyDefinition(type="string")
            },
            constraint=NodeConstraint(
                create="never",  # Controlled vocabulary
                search=SearchConfig(properties=[
                    PropertyMatch.semantic("name", 0.85)
                ])
            )
        ),
        "User": UserNodeType(
            name="User",
            properties={
                "username": PropertyDefinition(type="string", required=True),
                "role": PropertyDefinition(type="string")
            },
            constraint=NodeConstraint(
                create="auto",
                search=SearchConfig(properties=[
                    PropertyMatch.exact("username")
                ])
            )
        ),
        "Violation": UserNodeType(
            name="Violation",
            properties={
                "type": PropertyDefinition(type="string"),
                "severity": PropertyDefinition(type="string"),
                "timestamp": PropertyDefinition(type="datetime")
            },
            constraint=NodeConstraint(create="auto")
        )
    }
)
```

### Memory Ingestion

```python
# The simple case - schema handles everything
await client.add_memory(
    content="User john_doe attempted SQL injection attack, blocked by WAF policy",
    external_user_id="security_system",
    link_to={"$context:3": {}}  # Link to last 3 messages for context
)

# LLM extracts:
# - User(username="john_doe")
# - SecurityPolicy(name="WAF policy")
# - Violation(type="SQL injection")

# Schema applies:
# - User: create="auto" → creates if not exists
# - SecurityPolicy: create="never" → searches, links if found, skips if not
# - Violation: create="auto" → creates new violation record
```

### With Policy Override (Update Severity)

```python
await client.add_memory(
    content="CRITICAL: User john_doe attempted SQL injection attack, blocked by WAF policy",
    external_user_id="security_system",
    link_to={
        "$context:3": {},
        "Violation:type": {
            "set": {
                "severity": {"mode": "auto"},  # LLM extracts "CRITICAL"
                "timestamp": datetime.now().isoformat()  # Exact value
            }
        }
    }
)
```

### With Conditional Policy

```python
await client.add_memory(
    content="Multiple failed login attempts by admin_user",
    external_user_id="security_system",
    link_to={
        "User:username": {
            "when": {"role": "admin"},
            "set": {"requires_review": True}  # Flag admin violations for review
        },
        "Violation:type": {
            "when": {"_or": [{"type": "login_failure"}, {"type": "brute_force"}]},
            "set": {"alert_security_team": True}
        }
    }
)
```

---

## Part 5: How Auto-Extraction Actually Works

### Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUT                                                             │
│                                                                   │
│ content: "User john_doe attempted SQL injection, blocked by WAF" │
│ link_to: {                                                        │
│   "$context:3": {},                                               │
│   "SecurityPolicy:name": {"create": "never"}                     │
│ }                                                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Schema                                               │
│                                                                   │
│ Schema defines: User, SecurityPolicy, Violation                   │
│ Each has: properties, constraint (create, search, set)           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: Expand link_to to node_constraints                        │
│                                                                   │
│ link_to={"SecurityPolicy:name": {"create": "never"}}             │
│ Expands to: NodeConstraint(node_type="SecurityPolicy",            │
│   search=SearchConfig(properties=[PropertyMatch("name", "semantic")]),
│   create="never")                                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: Merge with Schema Constraints                             │
│                                                                   │
│ Schema: SecurityPolicy.create="never"                             │
│ Request: SecurityPolicy.create="never" (confirms)                │
│ Result: SecurityPolicy.create="never"                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: LLM Extraction                                            │
│                                                                   │
│ Prompt: "Given schema {User, SecurityPolicy, Violation},          │
│          extract entities from this content..."                   │
│                                                                   │
│ LLM Output:                                                       │
│   - User(username="john_doe")                                    │
│   - SecurityPolicy(name="WAF")                                   │
│   - Violation(type="SQL injection")                              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: Apply Policies Per Entity                                 │
│                                                                   │
│ User:                                                             │
│   - Policy: create="auto", search.exact("username")              │
│   - Action: Search for username="john_doe"                        │
│   - Result: Not found → CREATE new User node                      │
│                                                                   │
│ SecurityPolicy:                                                   │
│   - Policy: create="never", search.semantic("name", 0.85)        │
│   - Action: Semantic search for "WAF"                             │
│   - Result: Found "WAF Policy" (0.91) → LINK to existing         │
│                                                                   │
│ Violation:                                                        │
│   - Policy: create="auto"                                        │
│   - Action: CREATE new Violation node                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 6: Link Context                                              │
│                                                                   │
│ $context:3 in link_to                                             │
│ Action: Create FOLLOWS relationships to last 3 memories           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ RESULT: Memory Linked to Graph                                    │
│                                                                   │
│ Memory → MENTIONS → User(john_doe)                               │
│ Memory → MENTIONS → SecurityPolicy(WAF Policy)                   │
│ Memory → MENTIONS → Violation(SQL injection)                     │
│ Memory → FOLLOWS → [last 3 memories]                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Implementation

### Request Model Changes

```python
class AddMemoryRequest(BaseModel):
    content: str
    external_user_id: str

    # === EXISTING (full power) ===
    memory_policy: Optional[MemoryPolicy] = None

    # === SHORTHAND (expands to memory_policy.node_constraints) ===
    link_to: Optional[Union[str, List[str], Dict[str, Dict]]] = Field(
        default=None,
        description="Shorthand for node_constraints. Supports three forms:\n"
                   "- String: 'Type:property' or 'Type:property=value'\n"
                   "- List: ['Task:title', 'Person:email']\n"
                   "- Dict: {'Task:title': {'set': {...}, 'when': {...}, 'create': 'never'}}"
    )

    @model_validator(mode='after')
    def expand_link_to_to_memory_policy(self):
        """Expand link_to shorthand to full memory_policy.node_constraints."""
        if self.link_to:
            self.memory_policy = self._expand_link_to(self.link_to)
        return self

    def _expand_link_to(self, link_to):
        """
        Expand link_to to memory_policy.

        String: "Task:title" → NodeConstraint(node_type="Task", search=...)
        List: ["Task:title", "Person:email"] → [NodeConstraint(...), NodeConstraint(...)]
        Dict: {"Task:title": {"set": {...}}} → NodeConstraint(..., set={...})
        """
        # Implementation details...
        pass
```

### String DSL Grammar

```
# Single entity (string form)
link_to  := entity_ref | special_ref

# Multiple entities (list form)
link_to  := "[" (entity_ref | special_ref) ("," (entity_ref | special_ref))* "]"

# With constraints (dict form)
link_to  := "{" key ":" constraint_obj ("," key ":" constraint_obj)* "}"

entity_ref   := Type ":" Property [Operator Value]
special_ref  := "$this" | "$previous" | "$context:" Number
key          := entity_ref | special_ref
constraint_obj := "{" ["set:" Dict ","] ["when:" Dict ","] ["create:" ("auto"|"never")] "}"

Type     := [A-Z][a-zA-Z]*           # e.g., Task, Person, Project
Property := [a-z][a-zA-Z_]*          # e.g., id, title, email
Operator := "=" | "~"                # exact | semantic
Value    := literal_string
Number   := [0-9]+

# Default behavior when operator+value omitted:
# - Defaults to semantic match
# - Value extracted from content by LLM
"Task:title"  ≡  "Task:title~$extract_from_content"
```

**DSL Examples:**

| `link_to` Value | Meaning |
|-----------------|---------|
| `"Task:title"` | Semantic match Task.title using memory content (LLM extracts) |
| `"Task:id=TASK-123"` | Exact match Task.id with value "TASK-123" |
| `"Task:title~auth bug"` | Semantic match Task.title with explicit value |
| `["Task:title", "Person:name"]` | Link to Task AND Person using content |
| `["Task:id=T-1", "Person:email=j@x.com"]` | Link to specific Task and Person |
| `{"Task:title": {"set": {"status": "done"}}}` | Link to Task and update status |

---

## Part 7: Type Safety and Schema Introspection

### The Problem: Invalid References

Without validation, developers can make mistakes that fail silently or produce confusing errors:

```python
# What if Task doesn't exist in schema?
link_to="Task:title"  # Silent failure or cryptic error

# What if Task exists but has no "title" property?
link_to="Task:title"  # Searches on non-existent property

# What if developer misspells?
link_to="Taks:title"  # Typo goes unnoticed
```

### Solution: Schema Validation with Clear Errors

When processing `link_to`, the system validates against the registered schema:

```python
# Step 1: Parse the DSL
link_to = "Task:title"
parsed = {
    "node_type": "Task",
    "property": "title",
    "mode": "semantic",
    "value": None  # Extract from content
}

# Step 2: Validate against schema
schema = get_schema(schema_id)

# Check 1: Does node type exist?
if parsed.node_type not in schema.node_types:
    raise ValidationError(
        f"Unknown entity type: '{parsed.node_type}'. "
        f"Available types: {list(schema.node_types.keys())}"
    )

# Check 2: Does property exist on that type?
node_type = schema.node_types[parsed.node_type]
if parsed.property not in node_type.properties:
    raise ValidationError(
        f"Property '{parsed.property}' not found on type '{parsed.node_type}'. "
        f"Available properties: {list(node_type.properties.keys())}"
    )

# Check 3: Is semantic search allowed on this property?
# (Optional: warn if searching semantically on non-text property)
prop_def = node_type.properties[parsed.property]
if parsed.mode == "semantic" and prop_def.type not in ["string", "text"]:
    warn(
        f"Semantic search on '{parsed.property}' ({prop_def.type}) may not work well. "
        f"Consider using exact match: '{parsed.node_type}:{parsed.property}=value'"
    )
```

### Error Messages: Developer-Friendly

**Bad error (current):**
```
Error: node_type validation failed
```

**Good error (proposed):**
```
ValidationError: Unknown entity type 'Taks' in link_to="Taks:title"

Did you mean 'Task'?

Available entity types in schema 'project_management':
  - Task (properties: id, title, status, priority, assignee)
  - Person (properties: id, email, name, role)
  - Project (properties: id, name, description)

Example: link_to="Task:title" or link_to="Person:email=john@acme.com"
```

### Implementation: Validation Function

```python
def validate_link_to(
    link_to: Union[str, List[str], Dict[str, Dict]],
    schema: UserGraphSchema
) -> List[ParsedLinkTo]:
    """
    Validate and parse link_to DSL against schema.

    Returns parsed structures or raises ValidationError with helpful message.
    """
    results = []

    # Normalize to dict form
    if isinstance(link_to, str):
        link_to = {link_to: {}}
    elif isinstance(link_to, list):
        link_to = {k: {} for k in link_to}

    for key, constraints in link_to.items():
        # Handle special references
        if key.startswith("$"):
            results.append(ParsedLinkTo(special_ref=key, constraints=constraints))
            continue

        parsed = parse_link_dsl(key)  # Parse "Type:property=value"

        # Validate node type exists
        if parsed.node_type not in schema.node_types:
            available = list(schema.node_types.keys())
            suggestion = find_closest_match(parsed.node_type, available)
            raise ValidationError(
                message=f"Unknown entity type '{parsed.node_type}'",
                suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
                available_types=available,
                example=f'link_to="{available[0]}:{get_first_searchable_prop(schema.node_types[available[0]])}"'
            )

        node_type_def = schema.node_types[parsed.node_type]

        # Validate property exists
        if parsed.property not in node_type_def.properties:
            available_props = list(node_type_def.properties.keys())
            suggestion = find_closest_match(parsed.property, available_props)
            raise ValidationError(
                message=f"Property '{parsed.property}' not found on '{parsed.node_type}'",
                suggestion=f"Did you mean '{suggestion}'?" if suggestion else None,
                available_properties=available_props,
                example=f'link_to="{parsed.node_type}:{available_props[0]}"'
            )

        results.append(ParsedLinkTo(
            node_type=parsed.node_type,
            property=parsed.property,
            mode=parsed.mode,
            value=parsed.value,
            constraints=constraints
        ))

    return results
```

### Schema-Aware Autocomplete (SDK Feature)

For TypeScript/Python SDKs with type support:

```typescript
// TypeScript SDK with schema types
import { createMemoryClient, InferSchema } from '@papr/memory';

// Schema generates types
const schema = {
  node_types: {
    Task: {
      properties: {
        id: { type: 'string' },
        title: { type: 'string' },
        status: { type: 'string' }
      }
    },
    Person: {
      properties: {
        email: { type: 'string' },
        name: { type: 'string' }
      }
    }
  }
} as const;

type MySchema = InferSchema<typeof schema>;
const client = createMemoryClient<MySchema>({ apiKey: '...' });

// Now link_to is type-safe!
await client.addMemory({
  content: "...",
  link_to: "Task:title"      // ✅ Valid
  // link_to: "Task:foo"     // ❌ TypeScript error: 'foo' not in Task properties
  // link_to: "Foo:bar"      // ❌ TypeScript error: 'Foo' not in schema
});
```

```python
# Python SDK with runtime validation
from papr_memory import MemoryClient
from papr_memory.schemas import load_schema

# Load schema for validation
schema = load_schema("project_management")
client = MemoryClient(api_key="...", schema=schema)

# Runtime validation with helpful errors
await client.add_memory(
    content="...",
    link_to="Task:title"     # Validated against schema
)
```

### Validation Timing

| When | What's Validated | Error Type |
|------|-----------------|------------|
| **SDK (client-side)** | TypeScript: compile-time type checking | Compile error |
| **SDK (client-side)** | Python: runtime if schema provided | `ValidationError` |
| **API (server-side)** | Always validates against registered schema | HTTP 400 with details |

### Default Behavior Without Schema

If no schema is registered or provided, the API still works but with relaxed validation:

```python
# No schema - create dynamically
await client.add_memory(
    content="John fixed the bug",
    link_to="Task:title"  # Creates Task if not exists, uses default semantic search
)
```

This allows quick prototyping while encouraging schema definition for production.

---

### SDK Type-Safe Builders (Dual-Layer Approach)

The string DSL (`"Task:title"`) is token-efficient but lacks IDE support. SDKs solve this by providing **type-safe builders** that compile to the same DSL.

#### The Dual-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: SDK (Type-Safe)                      │
│  For: Developers AND Agents using Python/TypeScript SDKs        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  from papr.schemas.my_schema import Task, Person                │
│                                                                  │
│  await client.add_memory(                                        │
│      content="...",                                              │
│      link=[Task.title, Person.email]  # IDE autocomplete!       │
│  )                                                               │
│                                                                  │
│  Benefits:                                                       │
│  ✓ IDE autocomplete for node types and properties               │
│  ✓ Compile-time type checking (catches typos)                   │
│  ✓ Fewer errors = fewer tokens (no retries)                     │
│  ✓ Works for developers AND AI agents using SDKs                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Compiles down to
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: REST API (String DSL)                │
│  For: Direct REST calls, other languages, raw HTTP              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  POST /v1/memory                                                │
│  {                                                               │
│      "content": "...",                                           │
│      "link_to": ["Task:title", "Person:email"]                  │
│  }                                                               │
│                                                                  │
│  Benefits:                                                       │
│  ✓ Token-efficient for tool definitions                         │
│  ✓ Language-agnostic                                            │
│  ✓ Simple string format                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Python SDK: Generated Type-Safe Builders

```python
# Step 1: Generate types from your schema
# $ papr codegen my_schema --output ./papr_types

# Step 2: Import generated types
from papr_types import Task, Person, Project

# Step 3: Use with full IDE support
await client.add_memory(
    content="John fixed the auth bug",
    external_user_id="alice",
    link=[
        Task.title,                    # IDE shows: title, id, status, priority
        Person.email.exact(),          # IDE shows: exact(), semantic(), fuzzy()
        Project.name.semantic(0.9)     # Custom threshold with autocomplete
    ]
)

# What the SDK generates under the hood:
# link_to=["Task:title", "Person:email", "Project:name"]
```

**Generated Class Structure:**

```python
# Auto-generated from schema (papr_types/task.py)
class Task:
    """Task node type from schema 'project_management'"""

    @classmethod
    @property
    def title(cls) -> PropertyRef:
        """string - Task title (semantic match by default)"""
        return PropertyRef("Task", "title", mode="semantic")

    @classmethod
    @property
    def id(cls) -> PropertyRef:
        """string - Task ID (exact match by default)"""
        return PropertyRef("Task", "id", mode="exact")

    @classmethod
    @property
    def status(cls) -> PropertyRef:
        """string - Task status"""
        return PropertyRef("Task", "status", mode="semantic")

class PropertyRef:
    """Reference to a property with matching configuration."""

    def __init__(self, node_type: str, property: str, mode: str = "semantic"):
        self.node_type = node_type
        self.property = property
        self._mode = mode
        self._value = None
        self._threshold = 0.85

    def exact(self, value: Optional[str] = None) -> "PropertyRef":
        """Use exact matching."""
        self._mode = "exact"
        self._value = value
        return self

    def semantic(self, threshold: float = 0.85, value: Optional[str] = None) -> "PropertyRef":
        """Use semantic matching with optional threshold."""
        self._mode = "semantic"
        self._threshold = threshold
        self._value = value
        return self

    def __str__(self) -> str:
        """Convert to DSL string."""
        if self._value:
            op = "=" if self._mode == "exact" else "~"
            return f"{self.node_type}:{self.property}{op}{self._value}"
        return f"{self.node_type}:{self.property}"
```

#### TypeScript SDK: Generated Types

```typescript
// Auto-generated from schema
import { createMemoryClient, SchemaTypes } from '@papr/memory';
import type { ProjectManagementSchema } from './papr-types';

const client = createMemoryClient<ProjectManagementSchema>({ apiKey: '...' });

// Full autocomplete!
await client.addMemory({
    content: "John fixed the auth bug",
    externalUserId: "alice",
    link: [
        client.schema.Task.title,                    // Autocomplete shows Task properties
        client.schema.Person.email.exact(),          // Methods available
        client.schema.Project.name.semantic(0.9)     // Threshold parameter
    ]
});

// TypeScript catches errors at compile time:
await client.addMemory({
    link: [
        client.schema.Task.titl,     // ❌ Error: Property 'titl' does not exist
        client.schema.Taks.title,    // ❌ Error: Property 'Taks' does not exist
    ]
});
```

#### Benefits for AI Agents

AI agents using the SDK get the same benefits:

```python
# Agent tool definition using SDK
@tool
async def remember_with_context(content: str, task_title: str = None, person_email: str = None):
    """Store information and link to related entities."""
    links = []
    if task_title:
        links.append(Task.title.semantic(value=task_title))
    if person_email:
        links.append(Person.email.exact(value=person_email))

    await client.add_memory(
        content=content,
        external_user_id=current_user,
        link=links
    )

# Agent uses typed parameters → fewer errors → fewer retries → fewer tokens
```

**Token Savings for Agents:**

| Approach | Tool Definition Tokens | Error Rate | Effective Tokens |
|----------|----------------------|------------|------------------|
| Full `memory_policy` JSON | ~500 | High (complex nesting) | ~700+ with retries |
| String DSL `link_to` | ~50 | Medium (string typos) | ~80 with retries |
| SDK Type-Safe | ~50 | Low (compile-time checks) | ~55 |

#### Code Generation Commands

```bash
# Python: Generate types from registered schema
papr codegen my_schema_id --lang python --output ./papr_types

# TypeScript: Generate types from registered schema
papr codegen my_schema_id --lang typescript --output ./src/papr-types

# From local schema file
papr codegen ./schemas/project_management.json --lang python --output ./papr_types
```

#### Alternative: Class-Based Schema Definition (Full IDE Support)

Instead of dict-based schema definition, use Python decorators for full IDE introspection:

```python
from papr.sdk import schema, node, prop, edge, constraint
from papr.sdk import exact, semantic, controlled_vocabulary, auto_create
from typing import Optional

@schema("project_management")
class ProjectManagementSchema:
    """Project management schema with task tracking."""

    @node
    @auto_create
    class Task:
        """Task entity - can create new."""
        id: str = prop(search=exact())
        title: str = prop(required=True, search=semantic(0.85))
        status: str = prop(enum=["open", "in_progress", "completed"])
        priority: str = prop(enum=["low", "medium", "high"])
        assignee: Optional[str] = prop()

        # Relationships
        belongs_to = edge(to="Project")

    @node
    @controlled_vocabulary  # create="never" (or use @link_only - they're equivalent)
    class Person:
        """Person entity - controlled vocabulary."""
        email: str = prop(search=exact())
        name: str = prop(required=True, search=semantic(0.90))
        role: Optional[str] = prop()

    @node
    @auto_create
    class Project:
        """Project entity - can create new."""
        id: str = prop(search=exact())
        name: str = prop(required=True, search=semantic(0.85))
        status: Optional[str] = prop()


# Register schema
await client.register_schema(ProjectManagementSchema)

# Generated types available for memory operations:
from papr.schemas.project_management import Task, Person, Project
from papr import Auto

await client.add_memory(
    content="John completed the auth bug",
    link={
        Task.title.semantic(0.85).set({Task.status: Auto()}),
        Person.name.controlled_vocabulary()
    }
)
```

**Benefits of Class-Based Schema:**
- Full IDE autocomplete for properties and types
- Type checking catches errors at development time
- Docstrings visible in IDE
- Refactoring support (rename propagates everywhere)
- More Pythonic and readable
- Relationships defined inline with `edge()`

#### SDK vs String DSL Comparison

| Feature | String DSL | SDK Type-Safe |
|---------|------------|---------------|
| Token efficiency | Excellent | Excellent (same output) |
| IDE autocomplete | No | Yes |
| Compile-time checks | No | Yes (TypeScript) |
| Runtime validation | Server-side | Client + Server |
| Error discovery | At API call | At development time |
| Works for agents | Yes | Yes (better) |
| Language support | Any | Python, TypeScript |

**Recommendation:** Use SDK type-safe builders when possible (Python/TypeScript). Fall back to string DSL for other languages or raw REST calls.

---

### Python SDK Complete Workflow: Schema to Memory Override

This section shows the full developer workflow using the Python SDK - from defining schemas with constraints to overriding at memory level.

#### Step 1: Define Schema with Node Constraints (Schema Level)

```python
from papr import (
    Schema, NodeType, Property,
    Constraint, Search, Match
)

# Define schema using type-safe builders
schema = Schema(
    name="project_management",
    node_types=[
        # Task: Can create new, search by id (exact) or title (semantic)
        NodeType(
            name="Task",
            properties=[
                Property("id", type="string"),
                Property("title", type="string", required=True),
                Property("status", type="string", enum=["open", "in_progress", "completed"]),
                Property("priority", type="string", enum=["low", "medium", "high"]),
                Property("assignee", type="string"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([
                    Match.exact("id"),              # Try exact ID first
                    Match.semantic("title", 0.85)   # Then semantic title
                ])
            )
        ),

        # Person: Controlled vocabulary - never create new
        NodeType(
            name="Person",
            properties=[
                Property("email", type="string"),
                Property("name", type="string", required=True),
                Property("role", type="string"),
            ],
            constraint=Constraint(
                create="never",  # Controlled vocabulary
                search=Search([
                    Match.exact("email"),           # Try exact email first
                    Match.semantic("name", 0.90)    # Then semantic name (high threshold)
                ])
            )
        ),

        # Project: Can create, search by name
        NodeType(
            name="Project",
            properties=[
                Property("id", type="string"),
                Property("name", type="string", required=True),
                Property("status", type="string"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([Match.semantic("name", 0.85)])
            )
        ),
    ]
)

# Register schema
await client.register_schema(schema)
```

#### Step 2: Basic Memory (Schema Handles Everything)

```python
# 80% case: Just add memory - schema constraints apply automatically
await client.add_memory(
    content="John completed the authentication bug fix",
    external_user_id="alice"
)

# What happens:
# 1. LLM extracts: Task(title="authentication bug fix"), Person(name="John")
# 2. Task: search by title (semantic 0.85) → found → link
# 3. Person: search by name (semantic 0.90) → found → link (create="never" prevents new)
```

#### Step 3: Memory-Level Override with `link` (Type-Safe)

After schema registration, import generated types:

```python
from papr.schemas.project_management import Task, Person, Project

# Override: Update task status when linking
await client.add_memory(
    content="Sprint complete: auth bug is done",
    external_user_id="alice",
    link={
        Task.title: {
            "set": {"status": "completed"}  # Override: update status
        }
    }
)

# Override: Update multiple properties
await client.add_memory(
    content="High priority auth bug assigned to John, now in progress",
    external_user_id="alice",
    link={
        Task.title: {
            "set": {
                "status": "in_progress",
                "priority": "high",
                "assignee": {"mode": "auto"}  # LLM extracts from content
            }
        }
    }
)
```

#### Step 4: Conditional Override with `when`

```python
# Only update status for high-priority tasks
await client.add_memory(
    content="Sprint retrospective: completed the API review task",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {"priority": "high"},
            "set": {"status": "completed"}
        }
    }
)

# Complex condition: high priority AND not already completed
await client.add_memory(
    content="Urgent fix deployed for auth bug",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {
                "_and": [
                    {"priority": "high"},
                    {"_not": {"status": "completed"}}
                ]
            },
            "set": {"status": "completed"}
        }
    }
)
```

#### Real-World Example: DeepTrust Security Monitoring

```python
from papr import Schema, NodeType, Property, Constraint, Search, Match

# Schema Definition
security_schema = Schema(
    name="security_monitoring",
    node_types=[
        # SecurityBehavior: Controlled vocabulary (pre-defined compliance rules)
        NodeType(
            name="SecurityBehavior",
            properties=[
                Property("id", type="string"),
                Property("name", type="string", required=True),
                Property("description", type="string"),
                Property("category", type="string", enum=["access_control", "data_protection", "audit"]),
                Property("severity", type="string", enum=["low", "medium", "high", "critical"]),
            ],
            constraint=Constraint(
                create="never",  # NEVER create - these are pre-defined compliance rules
                search=Search([
                    Match.exact("id"),
                    Match.semantic("name", 0.85),
                    Match.semantic("description", 0.80)
                ])
            )
        ),

        # User: Can create (agents, customers discovered in calls)
        NodeType(
            name="User",
            properties=[
                Property("username", type="string", required=True),
                Property("role", type="string"),
                Property("department", type="string"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([Match.exact("username")])
            )
        ),

        # Violation: Always create new (each incident is unique)
        NodeType(
            name="Violation",
            properties=[
                Property("type", type="string"),
                Property("severity", type="string"),
                Property("timestamp", type="datetime"),
                Property("resolved", type="boolean"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([])  # No search - always create new
            )
        ),

        # CallAction: Actions taken during calls
        NodeType(
            name="CallAction",
            properties=[
                Property("action_type", type="string"),
                Property("call_id", type="string"),
                Property("agent_id", type="string"),
                Property("timestamp", type="datetime"),
            ],
            constraint=Constraint(create="auto")
        ),
    ]
)

await client.register_schema(security_schema)
```

**Using the Schema - Basic (Schema Handles It):**

```python
# Analyze call transcript - schema handles SecurityBehavior matching
await client.add_memory(
    content="Agent verified caller identity before discussing account details. "
            "Followed data protection protocol.",
    external_user_id="call_analyzer",
    metadata={"call_id": "call_789", "agent_id": "agent_42"}
)

# What happens:
# 1. LLM extracts: SecurityBehavior(name="data protection protocol"), User(...)
# 2. SecurityBehavior: create="never" → searches existing rules → links if found
# 3. User: create="auto" → creates if new
```

**Memory-Level Override - Force Values:**

```python
from papr.schemas.security_monitoring import SecurityBehavior, Violation, CallAction

# Override: Force call_id and agent_id on all extracted actions
await client.add_memory(
    content="Agent transferred call without verifying recipient. "
            "Potential security violation.",
    external_user_id="call_analyzer",
    link={
        CallAction.action_type: {
            "set": {
                "call_id": "call_789",      # Force exact value
                "agent_id": "agent_42",     # Force exact value
                "timestamp": {"mode": "auto"}  # LLM extracts
            }
        },
        Violation.type: {
            "set": {
                "severity": {"mode": "auto"},  # LLM determines from content
                "resolved": False
            }
        },
        SecurityBehavior.name: {
            "create": "never"  # Confirm: only link to existing rules
        }
    }
)
```

**Memory-Level Override - Conditional Severity:**

```python
# Different handling based on violation severity
await client.add_memory(
    content="CRITICAL: Agent shared customer SSN over unencrypted channel",
    external_user_id="call_analyzer",
    link={
        Violation.type: {
            "when": {"severity": "critical"},
            "set": {
                "requires_immediate_review": True,
                "escalated": True
            }
        },
        SecurityBehavior.name: {}  # Use schema defaults
    }
)
```

#### Real-World Example: Project Management Meeting Notes

```python
from papr.schemas.project_management import Task, Person, Project

# Meeting notes: Update multiple tasks mentioned
await client.add_memory(
    content="""
    Weekly sync meeting notes:
    - Auth bug fix completed by John
    - API review in progress, assigned to Sarah
    - Database migration blocked, waiting on DevOps
    """,
    external_user_id="meeting_bot",
    link={
        Task.title: {
            "set": {"status": {"mode": "auto"}}  # LLM extracts status per task
        },
        Person.name: {
            "create": "never"  # Only link to known team members
        }
    }
)

# Specific task update with exact ID
await client.add_memory(
    content="Closing TASK-123: Authentication bug verified fixed in production",
    external_user_id="alice",
    link={
        Task.id.exact("TASK-123"): {
            "set": {
                "status": "completed",
                "verified_in_prod": True
            }
        }
    }
)

# Conditional: Only update high-priority tasks from this meeting
await client.add_memory(
    content="Emergency standup: All high-priority items must be resolved by EOD",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {"priority": "high"},
            "set": {
                "deadline": "2026-01-25T17:00:00Z",
                "escalated": True
            }
        }
    }
)
```

#### Pattern Summary: Schema vs Memory-Level

| Level | What You Define | When to Use |
|-------|-----------------|-------------|
| **Schema** | Default behavior for ALL memories | One-time setup, organization-wide rules |
| **Memory** | Override for THIS specific memory | Per-request customization |

| Capability | Schema Level | Memory Level Override |
|------------|--------------|----------------------|
| Search strategy | `search=Search([Match...])` | `link={Task.title: {...}}` |
| Create policy | `create="auto"` or `"never"` | `{"create": "never"}` |
| Set properties | Not available | `{"set": {"status": "done"}}` |
| Conditional | Not available | `{"when": {"priority": "high"}}` |

**Key Insight:** Schema defines the guardrails, memory-level `link` provides runtime flexibility within those guardrails.

---

### Python SDK: Complete API Reference with Examples

This section provides comprehensive Python SDK examples for ALL controls available in `NodeConstraint`, `SearchConfig`, `PropertyMatch`, and `SetValue`. Each control is shown at both schema level and memory-level override.

---

#### 1. NodeConstraint Fields

##### 1.1 `node_type` - Entity Type Specification

```python
from papr import Schema, NodeType, Constraint

# SCHEMA LEVEL: node_type is IMPLICIT (taken from NodeType.name)
schema = Schema(
    name="my_schema",
    node_types=[
        NodeType(
            name="Task",  # ← This becomes the implicit node_type
            constraint=Constraint(
                # node_type NOT needed here - implicit from parent
                create="auto"
            )
        )
    ]
)

# MEMORY LEVEL: node_type is REQUIRED
from papr.schemas.my_schema import Task

await client.add_memory(
    content="...",
    link={
        Task.title: {...}  # SDK handles node_type="Task" automatically
    }
)

# Or with raw dict (node_type required):
await client.add_memory(
    content="...",
    memory_policy={
        "node_constraints": [{
            "node_type": "Task",  # ← REQUIRED at memory level
            "set": {"status": "done"}
        }]
    }
)
```

##### 1.2 `create` - Creation Policy

```python
from papr import Constraint

# SCHEMA LEVEL
NodeType(
    name="Task",
    constraint=Constraint(
        create="auto"  # Create new node if no match found (default)
    )
)

NodeType(
    name="Person",
    constraint=Constraint(
        create="never"  # Only link to existing - controlled vocabulary
    )
)

# MEMORY LEVEL OVERRIDE
from papr.schemas.my_schema import Task, Person

# Override: Don't create Tasks for this memory (even though schema says "auto")
await client.add_memory(
    content="Discussing potential new feature",
    link={
        Task.title: {"create": "never"}  # Override schema default
    }
)

# Override: Allow creating Person (if schema allows, or confirm schema setting)
await client.add_memory(
    content="Meeting with new contractor Bob",
    link={
        Person.name: {"create": "auto"}  # Would fail if schema says "never"
    }
)
```

##### 1.3 `when` - Conditional Constraints

All logical operators supported: simple match, `_and`, `_or`, `_not`, and complex combinations.

```python
from papr import Constraint

# SCHEMA LEVEL: Apply constraint only when condition matches

# Simple condition
NodeType(
    name="Task",
    constraint=Constraint(
        when={"priority": "high"},  # Only apply to high-priority tasks
        create="never"
    )
)

# AND: All conditions must match
NodeType(
    name="Task",
    constraint=Constraint(
        when={
            "_and": [
                {"priority": "high"},
                {"status": "active"}
            ]
        },
        create="never"
    )
)

# OR: At least one must match
NodeType(
    name="Task",
    constraint=Constraint(
        when={
            "_or": [
                {"status": "active"},
                {"status": "pending"}
            ]
        },
        create="auto"
    )
)

# NOT: Negation
NodeType(
    name="Task",
    constraint=Constraint(
        when={
            "_not": {"status": "completed"}
        },
        create="auto"
    )
)

# COMPLEX: priority=high AND (status=active OR urgent=true)
NodeType(
    name="Task",
    constraint=Constraint(
        when={
            "_and": [
                {"priority": "high"},
                {
                    "_or": [
                        {"status": "active"},
                        {"urgent": True}
                    ]
                }
            ]
        },
        create="never"
    )
)

# MEMORY LEVEL OVERRIDE with when
from papr.schemas.my_schema import Task

await client.add_memory(
    content="Sprint complete: all done",
    link={
        Task.title: {
            "when": {"priority": "high"},
            "set": {"status": "completed"}
        }
    }
)

# Complex when at memory level
await client.add_memory(
    content="Emergency escalation",
    link={
        Task.title: {
            "when": {
                "_and": [
                    {"priority": "high"},
                    {"_not": {"status": "completed"}}
                ]
            },
            "set": {"escalated": True, "urgent": True}
        }
    }
)
```

##### 1.4 `set` - Property Values

SetValue supports: exact values (str, int, float, bool, list, dict), auto-extract, and text modes.

```python
from papr import Constraint

# SCHEMA LEVEL: Not typically used (set is for runtime values)
# But can define default forced values
NodeType(
    name="Task",
    constraint=Constraint(
        set={"source": "api"}  # Always set source="api" for all Tasks
    )
)

# MEMORY LEVEL: Full set capabilities

# Exact values (str, int, float, bool)
await client.add_memory(
    content="Task completed",
    link={
        Task.title: {
            "set": {
                "status": "completed",     # string
                "priority_score": 5,       # int
                "completion_rate": 0.95,   # float
                "verified": True           # bool
            }
        }
    }
)

# Exact values (list, dict)
await client.add_memory(
    content="Task with metadata",
    link={
        Task.title: {
            "set": {
                "tags": ["urgent", "frontend", "bug"],  # list
                "metadata": {                            # dict
                    "sprint": "2026-Q1",
                    "reviewer": "alice"
                }
            }
        }
    }
)

# Auto-extract: LLM extracts from content
await client.add_memory(
    content="High priority auth bug is now in progress",
    link={
        Task.title: {
            "set": {
                "status": {"mode": "auto"},    # LLM extracts "in progress"
                "priority": {"mode": "auto"}   # LLM extracts "high"
            }
        }
    }
)

# Text modes for auto-extract: replace, append, merge
await client.add_memory(
    content="Additional notes: needs security review before deploy",
    link={
        Task.title: {
            "set": {
                # Replace: Overwrites existing value (default)
                "status": {"mode": "auto", "text_mode": "replace"},

                # Append: Adds to existing text
                "notes": {"mode": "auto", "text_mode": "append"},

                # Merge: Intelligently combines with existing
                "summary": {"mode": "auto", "text_mode": "merge"}
            }
        }
    }
)

# Mixed: Exact values + auto-extract
await client.add_memory(
    content="John finished the auth bug fix",
    link={
        Task.title: {
            "set": {
                "workspace_id": "ws_123",           # Exact value
                "completed_by": "john@acme.com",   # Exact value
                "status": {"mode": "auto"},         # LLM extracts
                "completion_notes": {"mode": "auto", "text_mode": "append"}
            }
        }
    }
)
```

---

#### 2. SearchConfig - Property-Based Matching

##### 2.1 Basic SearchConfig with PropertyMatch

```python
from papr import Constraint, Search, Match

# SCHEMA LEVEL: Define matching strategy

# Single property - exact match
NodeType(
    name="User",
    constraint=Constraint(
        search=Search([
            Match.exact("email")
        ])
    )
)

# Single property - semantic match with threshold
NodeType(
    name="Task",
    constraint=Constraint(
        search=Search([
            Match.semantic("title", threshold=0.85)
        ])
    )
)

# Multiple properties - priority order (first match wins)
NodeType(
    name="Task",
    constraint=Constraint(
        search=Search([
            Match.exact("id"),              # Try exact ID first
            Match.semantic("title", 0.85)   # Then semantic title
        ])
    )
)

# Fuzzy matching (for typo tolerance)
NodeType(
    name="Person",
    constraint=Constraint(
        search=Search([
            Match.exact("email"),           # Exact email first
            Match.fuzzy("name", 0.80)       # Fuzzy name (handles typos)
        ])
    )
)
```

##### 2.2 PropertyMatch Modes: exact, semantic, fuzzy

```python
from papr import Match

# EXACT: String equality (case-sensitive)
Match.exact("id")                    # Match id exactly
Match.exact("email")                 # Match email exactly
Match.exact("id", value="TASK-123") # With specific value

# SEMANTIC: Embedding similarity with threshold
Match.semantic("title")                      # Default threshold (0.85)
Match.semantic("title", threshold=0.90)      # High threshold (strict)
Match.semantic("title", threshold=0.75)      # Low threshold (loose)
Match.semantic("title", value="auth bug")    # With specific search value

# FUZZY: String similarity (Levenshtein-like)
Match.fuzzy("name")                          # Default threshold (0.85)
Match.fuzzy("name", threshold=0.80)          # Custom threshold
Match.fuzzy("name", value="Jon")             # Matches "John", "Jon", etc.
```

##### 2.3 PropertyMatch with Runtime Value Override

```python
from papr import Match

# SCHEMA LEVEL: Define matching mode (no value)
NodeType(
    name="Task",
    constraint=Constraint(
        search=Search([
            Match.exact("id"),           # No value - extracted from content
            Match.semantic("title")      # No value - extracted from content
        ])
    )
)

# MEMORY LEVEL: Override with specific value
from papr.schemas.my_schema import Task

# Exact match with specific ID
await client.add_memory(
    content="Update on task",
    link={
        Task.id.exact("TASK-123"): {  # Specific value override
            "set": {"status": "updated"}
        }
    }
)

# Semantic search with specific query
await client.add_memory(
    content="More details about the bug",
    link={
        Task.title.semantic("authentication bug", threshold=0.85): {
            "set": {"has_details": True}
        }
    }
)

# Fuzzy search with specific value
await client.add_memory(
    content="Feedback for team member",
    link={
        Person.name.fuzzy("Jon", threshold=0.80): {}  # Matches "John", "Jon"
    }
)
```

##### 2.4 SearchConfig String Shorthand

```python
from papr import Search, Match

# Strings become exact matches
Search(["id", "email"])
# Equivalent to:
Search([Match.exact("id"), Match.exact("email")])

# Mix strings with Match for flexibility
Search([
    "id",                              # String → exact match
    Match.semantic("title", 0.85)      # Full control
])

# With default mode and threshold
Search(
    properties=["title", "description"],
    mode="semantic",      # Default mode for properties without Match
    threshold=0.80        # Default threshold
)
```

---

#### 3. NodeConstraint Shorthand Constructors

```python
from papr import Constraint, Match

# 3.1 Controlled Vocabulary: Never create, only link to existing
constraint = Constraint.for_controlled_vocabulary(
    ["email", Match.semantic("name", 0.9)]
)
# Equivalent to:
constraint = Constraint(
    create="never",
    search=Search([
        Match.exact("email"),
        Match.semantic("name", 0.9)
    ])
)

# 3.2 Update Specific Node by ID
constraint = Constraint.for_update(
    id_value="TASK-123",
    set_properties={"status": {"mode": "auto"}}
)
# Equivalent to:
constraint = Constraint(
    search=Search([Match.exact("id", value="TASK-123")]),
    set={"status": {"mode": "auto"}}
)

# 3.3 Semantic Search and Update
constraint = Constraint.for_semantic_search(
    property="title",
    value="authentication bug",
    threshold=0.85,
    set_properties={"status": {"mode": "auto"}}
)
# Equivalent to:
constraint = Constraint(
    search=Search([Match.semantic("title", threshold=0.85, value="authentication bug")]),
    set={"status": {"mode": "auto"}}
)
```

---

#### 4. Complete Examples: Schema + Memory Level

##### Example A: Task Management System

```python
from papr import Schema, NodeType, Property, Constraint, Search, Match

# ═══════════════════════════════════════════════════════════════════
# SCHEMA DEFINITION
# ═══════════════════════════════════════════════════════════════════

schema = Schema(
    name="task_management",
    node_types=[
        # TASK: Can create, search by id (exact) or title (semantic)
        NodeType(
            name="Task",
            properties=[
                Property("id", type="string"),
                Property("title", type="string", required=True),
                Property("status", type="string", enum=["open", "in_progress", "completed", "blocked"]),
                Property("priority", type="string", enum=["low", "medium", "high", "critical"]),
                Property("assignee", type="string"),
                Property("due_date", type="datetime"),
                Property("tags", type="list"),
                Property("notes", type="string"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([
                    Match.exact("id"),
                    Match.semantic("title", 0.85)
                ])
            )
        ),

        # PERSON: Controlled vocabulary - only link to existing team members
        NodeType(
            name="Person",
            properties=[
                Property("email", type="string"),
                Property("name", type="string", required=True),
                Property("role", type="string"),
                Property("department", type="string"),
            ],
            constraint=Constraint(
                create="never",  # Controlled vocabulary
                search=Search([
                    Match.exact("email"),
                    Match.semantic("name", 0.90)
                ])
            )
        ),

        # PROJECT: Can create, search by name
        NodeType(
            name="Project",
            properties=[
                Property("id", type="string"),
                Property("name", type="string", required=True),
                Property("status", type="string"),
                Property("deadline", type="datetime"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([
                    Match.exact("id"),
                    Match.semantic("name", 0.85)
                ])
            )
        ),
    ]
)

await client.register_schema(schema)

# ═══════════════════════════════════════════════════════════════════
# MEMORY LEVEL: Various Override Patterns
# ═══════════════════════════════════════════════════════════════════

from papr.schemas.task_management import Task, Person, Project

# Pattern 1: Simple - Let schema handle everything
await client.add_memory(
    content="John is working on the authentication bug",
    external_user_id="alice"
)

# Pattern 2: Link to specific task by ID
await client.add_memory(
    content="TASK-123 has been verified in production",
    external_user_id="alice",
    link={
        Task.id.exact("TASK-123"): {
            "set": {"status": "completed", "verified": True}
        }
    }
)

# Pattern 3: Semantic search with status update
await client.add_memory(
    content="The login bug fix is now in progress",
    external_user_id="alice",
    link={
        Task.title.semantic("login bug"): {
            "set": {"status": "in_progress"}
        }
    }
)

# Pattern 4: Conditional update (only high priority)
await client.add_memory(
    content="Escalating all critical items for immediate attention",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {"priority": "critical"},
            "set": {
                "escalated": True,
                "urgent": True,
                "due_date": "2026-01-25T17:00:00Z"
            }
        }
    }
)

# Pattern 5: Complex condition with auto-extract
await client.add_memory(
    content="High priority blocked tasks need immediate review: API timeout issue",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {
                "_and": [
                    {"priority": "high"},
                    {"status": "blocked"}
                ]
            },
            "set": {
                "needs_review": True,
                "review_notes": {"mode": "auto", "text_mode": "append"}
            }
        }
    }
)

# Pattern 6: Multiple entity types with different constraints
await client.add_memory(
    content="Sprint planning: John will complete the API review by Friday",
    external_user_id="alice",
    link={
        Task.title: {
            "set": {
                "status": {"mode": "auto"},
                "due_date": {"mode": "auto"},
                "assignee": {"mode": "auto"}
            }
        },
        Person.name: {
            "create": "never"  # Only link to existing team members
        },
        Project.name: {}  # Use schema defaults
    }
)

# Pattern 7: Force specific values for traceability
await client.add_memory(
    content="Automated scan found security issue in auth module",
    external_user_id="security_scanner",
    link={
        Task.title: {
            "set": {
                "source": "security_scan",
                "priority": "critical",
                "tags": ["security", "automated"],
                "status": "open",
                "created_by": "security_scanner"
            }
        }
    }
)

# Pattern 8: Text mode - append notes to existing
await client.add_memory(
    content="Update: Customer confirmed the issue is reproducible on mobile",
    external_user_id="support_team",
    link={
        Task.title.semantic("mobile issue"): {
            "set": {
                "notes": {"mode": "auto", "text_mode": "append"},
                "affected_platforms": ["mobile"],
                "customer_confirmed": True
            }
        }
    }
)
```

##### Example B: Compliance Monitoring (DeepTrust-style)

```python
from papr import Schema, NodeType, Property, Constraint, Search, Match

# ═══════════════════════════════════════════════════════════════════
# SCHEMA DEFINITION
# ═══════════════════════════════════════════════════════════════════

compliance_schema = Schema(
    name="compliance_monitoring",
    node_types=[
        # SECURITY BEHAVIOR: Pre-defined rules - NEVER create new
        NodeType(
            name="SecurityBehavior",
            properties=[
                Property("id", type="string"),
                Property("name", type="string", required=True),
                Property("description", type="string"),
                Property("category", type="string", enum=["access_control", "data_protection", "audit", "encryption"]),
                Property("severity", type="string", enum=["info", "warning", "critical"]),
                Property("regulation", type="string"),  # GDPR, HIPAA, SOC2, etc.
            ],
            constraint=Constraint(
                create="never",  # CRITICAL: Never create - these are pre-defined
                search=Search([
                    Match.exact("id"),
                    Match.semantic("name", 0.85),
                    Match.semantic("description", 0.80)
                ])
            )
        ),

        # VIOLATION: Always create new (each incident is unique)
        NodeType(
            name="Violation",
            properties=[
                Property("type", type="string"),
                Property("severity", type="string", enum=["low", "medium", "high", "critical"]),
                Property("description", type="string"),
                Property("timestamp", type="datetime"),
                Property("resolved", type="boolean"),
                Property("resolution_notes", type="string"),
                Property("call_id", type="string"),
                Property("agent_id", type="string"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([])  # Empty = always create new
            )
        ),

        # AGENT: Track call center agents
        NodeType(
            name="Agent",
            properties=[
                Property("id", type="string"),
                Property("name", type="string"),
                Property("department", type="string"),
                Property("compliance_score", type="float"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([Match.exact("id")])
            )
        ),

        # CALL: Each call is unique
        NodeType(
            name="Call",
            properties=[
                Property("id", type="string"),
                Property("timestamp", type="datetime"),
                Property("duration", type="integer"),
                Property("caller_verified", type="boolean"),
                Property("compliance_status", type="string"),
            ],
            constraint=Constraint(
                create="auto",
                search=Search([Match.exact("id")])
            )
        ),
    ]
)

await client.register_schema(compliance_schema)

# ═══════════════════════════════════════════════════════════════════
# MEMORY LEVEL: Compliance Analysis Patterns
# ═══════════════════════════════════════════════════════════════════

from papr.schemas.compliance_monitoring import SecurityBehavior, Violation, Agent, Call

# Pattern 1: Analyze call transcript - schema handles SecurityBehavior linking
await client.add_memory(
    content="""
    Call transcript analysis:
    - Agent verified caller identity before discussing account details
    - Followed data protection protocol for sensitive information
    - Properly encrypted PII before storage
    """,
    external_user_id="compliance_analyzer",
    metadata={"call_id": "call_789", "agent_id": "agent_42"}
)

# Pattern 2: Force call_id and agent_id on all violations
await client.add_memory(
    content="Agent shared customer SSN without proper verification",
    external_user_id="compliance_analyzer",
    link={
        Violation.type: {
            "set": {
                "call_id": "call_789",           # Force exact value
                "agent_id": "agent_42",          # Force exact value
                "timestamp": "2026-01-25T10:30:00Z",
                "severity": {"mode": "auto"},    # LLM determines
                "resolved": False
            }
        },
        SecurityBehavior.name: {
            "create": "never"  # Confirm: only link to existing rules
        }
    }
)

# Pattern 3: Conditional severity handling
await client.add_memory(
    content="CRITICAL: Agent transferred call to unauthorized third party",
    external_user_id="compliance_analyzer",
    link={
        Violation.type: {
            "when": {"severity": "critical"},
            "set": {
                "requires_immediate_review": True,
                "escalated_to_legal": True,
                "auto_flagged": True
            }
        }
    }
)

# Pattern 4: Update agent compliance score
await client.add_memory(
    content="Agent completed all verification steps correctly on call_789",
    external_user_id="compliance_analyzer",
    link={
        Agent.id.exact("agent_42"): {
            "set": {
                "compliance_score": 0.95,  # High score
                "last_audit": "2026-01-25"
            }
        },
        Call.id.exact("call_789"): {
            "set": {
                "compliance_status": "passed",
                "caller_verified": True
            }
        }
    }
)

# Pattern 5: Complex condition - flag for training
await client.add_memory(
    content="Agent did not follow script for identity verification",
    external_user_id="compliance_analyzer",
    link={
        Agent.id.exact("agent_42"): {
            "when": {
                "_and": [
                    {"compliance_score": {"$lt": 0.8}},  # If score below 0.8
                    {"_not": {"department": "training"}}  # And not in training
                ]
            },
            "set": {
                "requires_training": True,
                "training_topics": ["identity_verification", "script_compliance"]
            }
        }
    }
)

# Pattern 6: Append to resolution notes (text_mode)
await client.add_memory(
    content="Violation VOL-456 reviewed: Agent received verbal warning, additional training scheduled",
    external_user_id="compliance_manager",
    link={
        Violation.type.semantic("unauthorized transfer"): {
            "set": {
                "resolved": True,
                "resolution_notes": {"mode": "auto", "text_mode": "append"},
                "resolved_by": "compliance_manager",
                "resolved_at": "2026-01-25T15:00:00Z"
            }
        }
    }
)
```

---

#### 5. Summary: All Controls at a Glance

| Control | Schema Level | Memory Level Override |
|---------|--------------|----------------------|
| **node_type** | Implicit from `NodeType.name` | `Task.title` (SDK) or `"node_type": "Task"` |
| **create** | `Constraint(create="auto"\|"never")` | `{"create": "never"}` |
| **search** | `Search([Match...])` | `Task.title.semantic(...)` or full config |
| **set** | `Constraint(set={...})` (rare) | `{"set": {"status": "done"}}` |
| **when** | `Constraint(when={...})` | `{"when": {"priority": "high"}}` |

| PropertyMatch Mode | SDK Shorthand | Use Case |
|--------------------|---------------|----------|
| **exact** | `Match.exact("id")` | IDs, emails, exact strings |
| **semantic** | `Match.semantic("title", 0.85)` | Natural language, descriptions |
| **fuzzy** | `Match.fuzzy("name", 0.80)` | Names with typo tolerance |

| SetValue Type | Example | Description |
|---------------|---------|-------------|
| **Exact string** | `"status": "done"` | Set exact value |
| **Exact int/float** | `"score": 95` | Set numeric value |
| **Exact bool** | `"verified": True` | Set boolean |
| **Exact list** | `"tags": ["a", "b"]` | Set list value |
| **Exact dict** | `"meta": {"k": "v"}` | Set dict value |
| **Auto-extract** | `{"mode": "auto"}` | LLM extracts from content |
| **Auto + replace** | `{"mode": "auto", "text_mode": "replace"}` | Replace existing (default) |
| **Auto + append** | `{"mode": "auto", "text_mode": "append"}` | Add to existing text |
| **Auto + merge** | `{"mode": "auto", "text_mode": "merge"}` | Intelligently combine |

| When Operator | Example | Description |
|---------------|---------|-------------|
| **Simple** | `{"priority": "high"}` | Property equals value |
| **_and** | `{"_and": [{...}, {...}]}` | All must match |
| **_or** | `{"_or": [{...}, {...}]}` | Any must match |
| **_not** | `{"_not": {...}}` | Negation |
| **Complex** | Nested combinations | Full boolean logic |

---

### Three Ways to Use the API: Comparison

This section compares the same operations using three approaches:
1. **Full API** - `memory_policy` with `node_constraints` (verbose, maximum control)
2. **String DSL** - `link_to` shorthand (token-efficient, REST-friendly)
3. **Python SDK** - Type-safe builders with IDE introspection (best DX)

---

#### Comparison 1: Basic Link to Entity

**Goal:** Link memory to a Task by semantic title match

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API (memory_policy / node_constraints)
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="John fixed the authentication bug",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                )
            )
        ]
    )
)
# Lines: 14 | Concepts: 5 (MemoryPolicy, NodeConstraint, SearchConfig, PropertyMatch, mode)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL (link_to shorthand)
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="John fixed the authentication bug",
    external_user_id="alice",
    link_to="Task:title"
)
# Lines: 5 | Concepts: 1 (link_to DSL syntax)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (IDE introspection)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task

await client.add_memory(
    content="John fixed the authentication bug",
    external_user_id="alice",
    link=[Task.title]  # IDE autocomplete: Task.id, Task.title, Task.status...
)
# Lines: 5 | Concepts: 1 | Bonus: IDE autocomplete + type checking
```

---

#### Comparison 2: Exact Match by ID

**Goal:** Link to specific task with known ID

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Status update on task",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="id", mode="exact", value="TASK-123")
                    ]
                )
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Status update on task",
    external_user_id="alice",
    link_to="Task:id=TASK-123"
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task

await client.add_memory(
    content="Status update on task",
    external_user_id="alice",
    link=[Task.id.exact("TASK-123")]  # IDE shows: .exact(), .semantic(), .fuzzy()
)
```

---

#### Comparison 3: Link and Update Properties (Auto-Extract vs Exact Value)

**Goal:** Link to Task, auto-extract status from content, set known workspace_id

**Real-world scenario:** Meeting notes processor knows the workspace_id, but wants LLM to extract status from the meeting content.

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Sprint complete: auth bug is now done and verified",
    external_user_id="meeting_bot",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                set={
                    "status": {"mode": "auto"},  # LLM extracts "done" from content
                    "workspace_id": "ws_123"     # We know this from meeting context
                }
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Sprint complete: auth bug is now done and verified",
    external_user_id="meeting_bot",
    link_to={
        "Task:title": {
            "set": {
                "status": {"mode": "auto"},  # LLM extracts from content
                "workspace_id": "ws_123"     # Exact value we know
            }
        }
    }
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (Type-Safe)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task
from papr import Auto  # Helper for auto-extract

await client.add_memory(
    content="Sprint complete: auth bug is now done and verified",
    external_user_id="meeting_bot",
    link={
        Task.title: {
            "set": {
                Task.status: Auto(),         # Type-safe: LLM extracts from content
                Task.workspace_id: "ws_123"  # Type-safe: exact value we know
            }
        }
    }
)

# Alternative Python SDK syntax with inline helpers:
await client.add_memory(
    content="Sprint complete: auth bug is now done and verified",
    external_user_id="meeting_bot",
    link={
        Task.title.set(
            status=Auto(),           # LLM extracts "done" → "completed"
            workspace_id="ws_123"    # We know this from meeting context
        )
    }
)
```

**Key distinction:**
- `status`: Use `Auto()` - LLM reads "done and verified" and extracts appropriate status
- `workspace_id`: Use exact value - we know this from the meeting/context, not from content

---

#### Comparison 4: Conditional Update with `when`

**Goal:** Only update status for high-priority tasks - LLM extracts status from content

**Real-world scenario:** Sprint retrospective - only mark high-priority items as done based on what's discussed.

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="All critical items resolved and deployed to production",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                when={"priority": "high"},
                set={"status": {"mode": "auto"}}  # LLM extracts "resolved" → "completed"
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="All critical items resolved and deployed to production",
    external_user_id="alice",
    link_to={
        "Task:title": {
            "when": {"priority": "high"},
            "set": {"status": {"mode": "auto"}}
        }
    }
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (Type-Safe)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task
from papr import Auto

await client.add_memory(
    content="All critical items resolved and deployed to production",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {Task.priority: "high"},  # Type-safe condition
            "set": {Task.status: Auto()}       # Type-safe auto-extract
        }
    }
)

# Alternative fluent syntax:
await client.add_memory(
    content="All critical items resolved and deployed to production",
    external_user_id="alice",
    link={
        Task.title.when(Task.priority == "high").set(
            status=Auto()  # LLM extracts from content
        )
    }
)
```

---

#### Comparison 5: Complex Condition with `_and`, `_not`

**Goal:** Update tasks that are high priority AND not yet completed

**Real-world scenario:** Emergency response - flag active high-priority items for immediate attention.

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Emergency escalation for active items - need immediate attention",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                when={
                    "_and": [
                        {"priority": "high"},
                        {"_not": {"status": "completed"}}
                    ]
                },
                set={
                    "escalated": True,           # Exact value - we're flagging
                    "urgent": True,              # Exact value - we're flagging
                    "escalation_reason": {"mode": "auto"}  # LLM extracts reason
                }
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Emergency escalation for active items - need immediate attention",
    external_user_id="alice",
    link_to={
        "Task:title": {
            "when": {
                "_and": [
                    {"priority": "high"},
                    {"_not": {"status": "completed"}}
                ]
            },
            "set": {
                "escalated": True,
                "urgent": True,
                "escalation_reason": {"mode": "auto"}
            }
        }
    }
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (Type-Safe)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task
from papr import Auto, And, Not  # Logical operator helpers

await client.add_memory(
    content="Emergency escalation for active items - need immediate attention",
    external_user_id="alice",
    link={
        Task.title: {
            "when": And(
                Task.priority == "high",
                Not(Task.status == "completed")
            ),
            "set": {
                Task.escalated: True,              # Exact - we're flagging it
                Task.urgent: True,                 # Exact - we're flagging it
                Task.escalation_reason: Auto()    # LLM extracts from content
            }
        }
    }
)

# Alternative dict syntax (also valid):
await client.add_memory(
    content="Emergency escalation for active items - need immediate attention",
    external_user_id="alice",
    link={
        Task.title: {
            "when": {
                "_and": [
                    {Task.priority: "high"},
                    {"_not": {Task.status: "completed"}}
                ]
            },
            "set": {
                Task.escalated: True,
                Task.urgent: True,
                Task.escalation_reason: Auto()
            }
        }
    }
)
```

---

#### Comparison 6: Multiple Entities with Different Constraints

**Goal:** Link to Task (auto-extract status) and Person (controlled vocabulary, link assignee)

**Real-world scenario:** Meeting notes - LLM extracts status from discussion, we link to known team members only.

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="John completed the API review task, now in testing phase",
    external_user_id="meeting_bot",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                set={
                    "status": {"mode": "auto"},      # LLM extracts "in testing"
                    "completed_by": {"mode": "auto"} # LLM extracts "John"
                }
            ),
            NodeConstraint(
                node_type="Person",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="name", mode="semantic", threshold=0.90)
                    ]
                ),
                create="never"  # Only link to existing team members
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="John completed the API review task, now in testing phase",
    external_user_id="meeting_bot",
    link_to={
        "Task:title": {
            "set": {
                "status": {"mode": "auto"},
                "completed_by": {"mode": "auto"}
            }
        },
        "Person:name": {"create": "never"}
    }
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (Type-Safe)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task, Person
from papr import Auto

await client.add_memory(
    content="John completed the API review task, now in testing phase",
    external_user_id="meeting_bot",
    link={
        Task.title: {
            "set": {
                Task.status: Auto(),        # LLM extracts "in testing"
                Task.completed_by: Auto()   # LLM extracts "John"
            }
        },
        Person.name: {"create": "never"}    # Only existing team members
    }
)

# Alternative with fluent syntax:
await client.add_memory(
    content="John completed the API review task, now in testing phase",
    external_user_id="meeting_bot",
    link={
        Task.title.set(status=Auto(), completed_by=Auto()),
        Person.name.controlled_vocabulary()  # Shorthand for create="never"
    }
)
```

---

#### Comparison 7: Auto-Extract with Text Mode (replace, append, merge)

**Goal:** Auto-extract status, append new notes to existing notes, set known project_id

**Real-world scenario:** Follow-up on a task - add new notes without overwriting existing ones.

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Task is now in progress. Update: needs security review before deploy",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                set={
                    "status": {"mode": "auto"},                        # LLM extracts "in progress"
                    "notes": {"mode": "auto", "text_mode": "append"},  # Add to existing notes
                    "project_id": "PROJ-456"                           # Exact value we know
                }
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="Task is now in progress. Update: needs security review before deploy",
    external_user_id="alice",
    link_to={
        "Task:title": {
            "set": {
                "status": {"mode": "auto"},
                "notes": {"mode": "auto", "text_mode": "append"},
                "project_id": "PROJ-456"
            }
        }
    }
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (Type-Safe)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task
from papr import Auto

await client.add_memory(
    content="Task is now in progress. Update: needs security review before deploy",
    external_user_id="alice",
    link={
        Task.title: {
            "set": {
                Task.status: Auto(),                      # LLM extracts "in progress"
                Task.notes: Auto(text_mode="append"),     # Add to existing (not overwrite)
                Task.project_id: "PROJ-456"               # Exact value we know
            }
        }
    }
)

# Alternative fluent syntax with text modes:
await client.add_memory(
    content="Task is now in progress. Update: needs security review before deploy",
    external_user_id="alice",
    link={
        Task.title.set(
            status=Auto(),                    # replace (default) - overwrite status
            notes=Auto.append(),              # append - add to existing text
            summary=Auto.merge(),             # merge - intelligently combine
            project_id="PROJ-456"             # exact value
        )
    }
)
```

**Text Mode Reference:**
| Mode | Behavior | Use Case |
|------|----------|----------|
| `replace` (default) | Overwrite existing value | Status, single values |
| `append` | Add to end of existing text | Notes, logs, history |
| `merge` | Intelligently combine with existing | Summaries, descriptions |

---

### Python SDK Complete Reference: All Options

This section provides a comprehensive reference for all Python SDK options with IDE introspection.

#### Imports

```python
from papr import MemoryClient, Schema, NodeType, Property, Constraint, Search, Match
from papr import Auto, And, Or, Not  # Helpers for set and when
from papr import This  # Reference to current memory context
from papr.schemas.my_schema import Task, Person, Project  # Generated from schema
```

---

#### Core Philosophy: LLM Extracts, You Provide Context

**The whole point of `link` is that LLM extracts entities from content automatically.**

| Pattern | Frequency | Example | When to Use |
|---------|-----------|---------|-------------|
| **LLM Extracts** | 80% | `Task.title` | Default - LLM reads content, finds task |
| **Context Reference** | 15% | `This.metadata.customMetadata.project_id` | You know context (workspace, project) |
| **Explicit Value** | 5% | `"TASK-123"` | User selected from UI, rare edge case |

**Anti-pattern:** Don't do `Task.title.semantic("authentication bug")` - if you're hardcoding the search term, why use LLM at all? Just use exact match or let LLM extract from content.

---

#### `link` - All Variations

The key insight: **LLM extracts from content automatically**. You rarely need to specify explicit values.

```python
from papr.schemas.my_schema import Task, Person, Project
from papr import This  # Reference to current memory

# ─────────────────────────────────────────────────────────────────
# DEFAULT (80% of cases) - LLM extracts from content
# ─────────────────────────────────────────────────────────────────

# Semantic match - LLM extracts task title from content
link=[Task.title]

# Exact match - LLM extracts ID from content (e.g., "TASK-123 is complete")
link=[Task.id.exact()]

# Multiple - LLM extracts all from content
link=[Task.title, Person.name, Project.name]

# ─────────────────────────────────────────────────────────────────
# CONTEXT REFERENCES (15% of cases) - Use metadata or context
# ─────────────────────────────────────────────────────────────────

# Use value from customMetadata (dev passes workspace_id in metadata)
link=[Task.workspace_id.exact(This.metadata.customMetadata.workspace_id)]

# Use value from customMetadata for semantic search
link=[Project.name.semantic(This.metadata.customMetadata.project_name)]

# Use the full content for semantic search (explicit, same as default)
link=[Task.title.semantic(This.content)]

# ─────────────────────────────────────────────────────────────────
# CUSTOM THRESHOLD (when you need stricter/looser matching)
# ─────────────────────────────────────────────────────────────────

# Strict semantic match (0.95 threshold)
link=[Task.title.semantic(threshold=0.95)]

# Loose semantic match (0.75 threshold)
link=[Task.title.semantic(threshold=0.75)]

# Fuzzy match for names (handles typos)
link=[Person.name.fuzzy(threshold=0.80)]

# ─────────────────────────────────────────────────────────────────
# ADVANCED (5% of cases) - Explicit value (rare, for edge cases)
# Only use when you have a specific known value to search for
# ─────────────────────────────────────────────────────────────────

# Exact ID you know (e.g., from user selection in UI)
link=[Task.id.exact("TASK-123")]

# Semantic search with explicit query (rare - usually LLM extracts)
link=[Task.title.semantic("authentication bug")]

# ─────────────────────────────────────────────────────────────────
# MULTIPLE ENTITIES
# ─────────────────────────────────────────────────────────────────

# Multiple entities - list form
link=[Task.title, Person.email, Project.name]

# Multiple with different match modes
link=[
    Task.id.exact("TASK-123"),           # Exact ID
    Task.title.semantic(),                # Semantic title
    Person.email.exact("john@acme.com"), # Exact email
    Person.name.fuzzy()                   # Fuzzy name
]

# ─────────────────────────────────────────────────────────────────
# WITH CONSTRAINTS (dict form)
# ─────────────────────────────────────────────────────────────────

# Single entity with constraints
link={
    Task.title: {
        "set": {...},
        "when": {...},
        "create": "never"
    }
}

# Multiple entities with different constraints
link={
    Task.title: {"set": {Task.status: Auto()}},
    Person.name: {"create": "never"},
    Project.name: {}  # Use schema defaults
}

# ─────────────────────────────────────────────────────────────────
# SPECIAL REFERENCES
# ─────────────────────────────────────────────────────────────────

from papr import This, Previous, Context

# Link to previous memory
link={Previous(): {}}

# Link to last N memories (conversation context)
link={Context(3): {}}  # Last 3 memories

# Combine entity linking with context
link={
    Task.title: {"set": {Task.status: Auto()}},
    Context(3): {}  # Include conversation context
}
```

---

#### `set` - All Variations

```python
from papr.schemas.my_schema import Task
from papr import Auto

# ─────────────────────────────────────────────────────────────────
# EXACT VALUES (We know the value)
# ─────────────────────────────────────────────────────────────────

# String
"set": {Task.status: "completed"}

# Integer
"set": {Task.priority_score: 5}

# Float
"set": {Task.completion_rate: 0.95}

# Boolean
"set": {Task.verified: True}

# List
"set": {Task.tags: ["urgent", "frontend", "bug"]}

# Dict
"set": {Task.metadata: {"sprint": "2026-Q1", "team": "backend"}}

# ─────────────────────────────────────────────────────────────────
# AUTO-EXTRACT (LLM extracts from content)
# ─────────────────────────────────────────────────────────────────

# Basic auto-extract (replace mode - default)
"set": {Task.status: Auto()}

# Auto-extract with text_mode: replace (overwrites existing)
"set": {Task.status: Auto(text_mode="replace")}

# Auto-extract with text_mode: append (adds to existing)
"set": {Task.notes: Auto(text_mode="append")}

# Auto-extract with text_mode: merge (intelligently combines)
"set": {Task.summary: Auto(text_mode="merge")}

# Shorthand for text modes
"set": {
    Task.status: Auto(),           # Same as Auto(text_mode="replace")
    Task.notes: Auto.append(),     # Same as Auto(text_mode="append")
    Task.summary: Auto.merge()     # Same as Auto(text_mode="merge")
}

# ─────────────────────────────────────────────────────────────────
# MIXED (Exact values + Auto-extract)
# ─────────────────────────────────────────────────────────────────

"set": {
    # Exact values we know
    Task.workspace_id: "ws_123",
    Task.project_id: "PROJ-456",
    Task.updated_by: "meeting_bot",
    Task.updated_at: "2026-01-25T10:30:00Z",

    # Auto-extract from content
    Task.status: Auto(),
    Task.priority: Auto(),
    Task.notes: Auto.append(),
    Task.summary: Auto.merge()
}

# ─────────────────────────────────────────────────────────────────
# FULL EXAMPLE
# ─────────────────────────────────────────────────────────────────

await client.add_memory(
    content="Sprint planning: Auth bug is now high priority and in progress",
    external_user_id="meeting_bot",
    link={
        Task.title: {
            "set": {
                # LLM extracts from content
                Task.status: Auto(),              # Extracts "in progress"
                Task.priority: Auto(),            # Extracts "high"
                Task.notes: Auto.append(),        # Adds to existing notes

                # Exact values we know
                Task.workspace_id: "ws_sprint_planning",
                Task.last_discussed: "2026-01-25"
            }
        }
    }
)
```

---

#### `when` - All Variations

```python
from papr.schemas.my_schema import Task
from papr import And, Or, Not

# ─────────────────────────────────────────────────────────────────
# SIMPLE CONDITION (single property match)
# ─────────────────────────────────────────────────────────────────

# Property equals value
"when": {Task.priority: "high"}

# Multiple simple conditions (implicit AND)
"when": {Task.priority: "high", Task.status: "active"}

# ─────────────────────────────────────────────────────────────────
# AND - All conditions must match
# ─────────────────────────────────────────────────────────────────

# Using And() helper
"when": And(
    Task.priority == "high",
    Task.status == "active"
)

# Using dict form
"when": {
    "_and": [
        {Task.priority: "high"},
        {Task.status: "active"}
    ]
}

# Three or more conditions
"when": And(
    Task.priority == "high",
    Task.status == "active",
    Task.assignee == "john"
)

# ─────────────────────────────────────────────────────────────────
# OR - At least one must match
# ─────────────────────────────────────────────────────────────────

# Using Or() helper
"when": Or(
    Task.status == "active",
    Task.status == "pending"
)

# Using dict form
"when": {
    "_or": [
        {Task.status: "active"},
        {Task.status: "pending"}
    ]
}

# Multiple OR conditions
"when": Or(
    Task.priority == "high",
    Task.priority == "critical",
    Task.urgent == True
)

# ─────────────────────────────────────────────────────────────────
# NOT - Negation
# ─────────────────────────────────────────────────────────────────

# Using Not() helper
"when": Not(Task.status == "completed")

# Using dict form
"when": {"_not": {Task.status: "completed"}}

# NOT with multiple properties (NOT all of these)
"when": Not(And(Task.status == "completed", Task.archived == True))

# ─────────────────────────────────────────────────────────────────
# COMPLEX COMBINATIONS
# ─────────────────────────────────────────────────────────────────

# priority=high AND NOT completed
"when": And(
    Task.priority == "high",
    Not(Task.status == "completed")
)

# priority=high AND (status=active OR urgent=true)
"when": And(
    Task.priority == "high",
    Or(
        Task.status == "active",
        Task.urgent == True
    )
)

# (priority=high OR priority=critical) AND NOT (completed OR archived)
"when": And(
    Or(Task.priority == "high", Task.priority == "critical"),
    Not(Or(Task.status == "completed", Task.archived == True))
)

# Complex nested: high priority active tasks OR any critical tasks
"when": Or(
    And(Task.priority == "high", Task.status == "active"),
    Task.priority == "critical"
)

# Dict form for complex conditions (also valid)
"when": {
    "_and": [
        {Task.priority: "high"},
        {
            "_or": [
                {Task.status: "active"},
                {Task.urgent: True}
            ]
        },
        {
            "_not": {Task.archived: True}
        }
    ]
}

# ─────────────────────────────────────────────────────────────────
# FULL EXAMPLES
# ─────────────────────────────────────────────────────────────────

# Example 1: Update only high-priority active tasks
await client.add_memory(
    content="Sprint complete: all items resolved",
    external_user_id="alice",
    link={
        Task.title: {
            "when": And(
                Task.priority == "high",
                Task.status == "active"
            ),
            "set": {Task.status: Auto()}
        }
    }
)

# Example 2: Escalate tasks that are high/critical AND not done
await client.add_memory(
    content="Emergency escalation required",
    external_user_id="alice",
    link={
        Task.title: {
            "when": And(
                Or(Task.priority == "high", Task.priority == "critical"),
                Not(Task.status == "completed")
            ),
            "set": {
                Task.escalated: True,
                Task.escalation_reason: Auto()
            }
        }
    }
)

# Example 3: Flag for review if in certain states
await client.add_memory(
    content="QA team flagged for review",
    external_user_id="qa_bot",
    link={
        Task.title: {
            "when": Or(
                Task.status == "blocked",
                Task.status == "needs_review",
                And(Task.priority == "critical", Task.status == "in_progress")
            ),
            "set": {
                Task.flagged_for_review: True,
                Task.review_notes: Auto.append()
            }
        }
    }
)
```

---

#### Fluent Syntax (Alternative)

The SDK also supports a fluent builder pattern for more readable code:

```python
from papr.schemas.my_schema import Task, Person
from papr import Auto

# ─────────────────────────────────────────────────────────────────
# FLUENT BUILDERS
# ─────────────────────────────────────────────────────────────────

# Basic fluent
await client.add_memory(
    content="...",
    link={
        Task.title.set(status=Auto(), notes=Auto.append())
    }
)

# With when condition (fluent)
await client.add_memory(
    content="...",
    link={
        Task.title
            .when(Task.priority == "high")
            .set(status=Auto(), escalated=True)
    }
)

# Controlled vocabulary (fluent)
await client.add_memory(
    content="...",
    link={
        Person.name.controlled_vocabulary()  # Same as {"create": "never"}
    }
)

# Multiple entities (fluent)
await client.add_memory(
    content="...",
    link={
        Task.title.set(status=Auto()),
        Person.name.controlled_vocabulary(),
        Project.name  # Use schema defaults
    }
)

# Complex example (fluent)
await client.add_memory(
    content="Sprint complete: John finished the auth bug, needs security review",
    external_user_id="meeting_bot",
    link={
        Task.title
            .when(Task.priority == "high")
            .set(
                status=Auto(),
                completed_by=Auto(),
                notes=Auto.append(),
                workspace_id="ws_123"
            ),
        Person.name.controlled_vocabulary()
    }
)
```

---

#### Quick Reference Table

| SDK Syntax | Meaning | Use Case |
|------------|---------|----------|
| **Link - Default (LLM extracts)** | | |
| `Task.title` | Semantic match, LLM extracts from content | 80% of cases |
| `Task.id.exact()` | Exact match, LLM extracts ID from content | When content has ID |
| `Person.name.fuzzy()` | Fuzzy match, LLM extracts (typo-tolerant) | Names with typos |
| **Link - Context Reference** | | |
| `Task.id.exact(This.metadata.customMetadata.task_id)` | Use value from metadata | Dev passes value in metadata |
| `Project.name.semantic(This.metadata.customMetadata.project)` | Semantic search using metadata | Search by known project |
| **Link - Custom Threshold** | | |
| `Task.title.semantic(threshold=0.95)` | Strict semantic match | High precision needed |
| `Task.title.semantic(threshold=0.75)` | Loose semantic match | Broad matching |
| **Link - Advanced (explicit value)** | | |
| `Task.id.exact("TASK-123")` | Exact match with hardcoded value | User selected from UI |
| **Set Variations** | |
| `Task.status: "done"` | Exact value |
| `Task.status: Auto()` | LLM extracts (replace mode) |
| `Task.notes: Auto.append()` | LLM extracts, append to existing |
| `Task.summary: Auto.merge()` | LLM extracts, merge with existing |
| **When Variations** | |
| `{Task.priority: "high"}` | Simple match |
| `And(a, b)` | All must match |
| `Or(a, b)` | Any must match |
| `Not(a)` | Negation |
| `And(a, Or(b, c))` | Complex nesting |
| **Create** | |
| `"create": "auto"` | Create if not found (default) |
| `"create": "never"` | Only link to existing |
| `.controlled_vocabulary()` | Shorthand for create="never" |

---

#### Comparison 8: Using Context Reference (Metadata Value)

**Goal:** Search for task using project_id from metadata (dev passes context)

**Real-world scenario:** Dev knows the project context and passes it in metadata. LLM should link to tasks in that project.

```python
# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Full API
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="More details about the issue in this project",
    external_user_id="alice",
    metadata={"customMetadata": {"project_id": "PROJ-456"}},
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", threshold=0.85)
                    ]
                ),
                set={"project_id": "$this.metadata.customMetadata.project_id"}
            )
        ]
    )
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: String DSL
# ═══════════════════════════════════════════════════════════════════
await client.add_memory(
    content="More details about the issue in this project",
    external_user_id="alice",
    metadata={"customMetadata": {"project_id": "PROJ-456"}},
    link_to={
        "Task:title": {
            "set": {"project_id": "$this.metadata.customMetadata.project_id"}
        }
    }
)

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: Python SDK (Type-Safe)
# ═══════════════════════════════════════════════════════════════════
from papr.schemas.my_schema import Task
from papr import This, Auto

await client.add_memory(
    content="More details about the issue in this project",
    external_user_id="alice",
    metadata={"customMetadata": {"project_id": "PROJ-456"}},
    link={
        Task.title: {  # LLM extracts task title from content
            "set": {
                Task.project_id: This.metadata.customMetadata.project_id,  # From context
                Task.notes: Auto.append()  # LLM extracts from content
            }
        }
    }
)
```

**Key pattern:**
- `Task.title` - LLM extracts from content (default)
- `This.metadata.customMetadata.project_id` - Use value dev passed in metadata

---

#### Summary: When to Use Each Approach

| Approach | Lines | IDE Support | Token Cost | Best For |
|----------|-------|-------------|------------|----------|
| **Full API** | 15-25 | No | High | Max control, edge cases |
| **String DSL** | 4-8 | No | Low | REST API, agents, quick scripts |
| **Python SDK** | 4-8 | Yes | Low | Developers, production code |

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick prototype / testing | String DSL |
| Production Python code | Python SDK |
| AI agent using REST | String DSL |
| AI agent using Python SDK | Python SDK |
| Complex edge case | Full API |
| Need IDE autocomplete | Python SDK |
| TypeScript/other language | String DSL (until SDK available) |

---

## Part 8: Shorthand Behavior by Mode

### Understanding Auto vs Manual Mode

The `MemoryPolicy.mode` field controls WHO generates the graph:

| Mode | Who | Description |
|------|-----|-------------|
| `auto` (default) | LLM | LLM extracts entities from content. `node_constraints` control matching/creation. |
| `manual` | Developer | Developer provides exact `nodes` and `relationships`. No LLM extraction. |

### `link_to` Only Works in Auto Mode

| Mode | `link_to` | Why |
|------|-----------|-----|
| `auto` (default) | ✅ Works | LLM extracts, `link_to` controls matching |
| `manual` | ❌ Error | Developer provides exact `nodes[]`, nothing to "link to" |

**Why `link_to` doesn't apply to manual mode:**

Manual mode means "I know exactly what I want - no LLM involvement." The developer provides:
- Exact node IDs, types, and properties in `nodes[]`
- Exact relationships in `relationships[]`

There's nothing to "link to" because you're not asking the LLM to extract anything.

### Auto Mode: `link_to` Examples

```python
# Auto mode (default) - LLM extracts, link_to works
await client.add_memory(
    content="John fixed the authentication bug",
    external_user_id="alice",
    link_to="Task:title"  # → node_constraints for Task with semantic search
)

# Equivalent full form
await client.add_memory(
    content="John fixed the authentication bug",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        mode="auto",  # Default
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[PropertyMatch(name="title", mode="semantic")]
                )
                # Value extracted from content by LLM
            )
        ]
    )
)
```

### Manual Mode: Full Control

```python
# Manual mode - no LLM extraction, no link_to
await client.add_memory(
    content="Order #12345: John purchased 2 lattes",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        mode="manual",
        nodes=[
            NodeSpec(id="txn_12345", type="Transaction", properties={
                "order_id": "12345",
                "amount": 11.00,
                "timestamp": "2026-01-24T10:30:00Z"
            }),
            NodeSpec(id="prod_latte", type="Product", properties={
                "name": "Latte",
                "price": 5.50
            }),
            NodeSpec(id="person_john", type="Person", properties={
                "name": "John"
            })
        ],
        relationships=[
            RelationshipSpec(source="txn_12345", target="prod_latte", type="CONTAINS"),
            RelationshipSpec(source="txn_12345", target="person_john", type="PURCHASED_BY"),
            RelationshipSpec(source="$this", target="txn_12345", type="DESCRIBES")
        ]
    )
)
```

**Why use manual mode?**

1. **Structured data ingestion** - POS transactions, API webhooks, database syncs
2. **Deterministic graphs** - You know exactly what nodes should exist
3. **Performance** - Skip LLM extraction, direct graph operations
4. **Compliance** - Audit trail requires exact control over what's stored

### Mode Detection

When `link_to` is used, mode is implicitly `auto`:

```python
# This is auto mode (link_to implies it)
await client.add_memory(
    content="...",
    link_to="Task:title"  # Implies mode="auto"
)
```

If you try to mix `link_to` with manual mode, you get an error:

```python
# ERROR: Can't mix link_to with manual mode
await client.add_memory(
    content="...",
    link_to="Task:title",  # Shorthand
    memory_policy=MemoryPolicy(
        mode="manual",  # Explicit manual
        nodes=[...]
    )
)
# ValidationError: "link_to" cannot be used with mode="manual".
# Manual mode requires exact "nodes" specification.
```

### Summary: When to Use What

| Scenario | Use |
|----------|-----|
| Natural language content, find related entities | `link_to` shorthand (auto mode) |
| Meeting notes, extract and link people/tasks | `link_to=["Task:title", "Person:name"]` (auto mode) |
| API webhook, structured data | `memory_policy.mode="manual"` with `nodes[]` |
| POS transaction, exact graph needed | `memory_policy.mode="manual"` with `nodes[]` |
| User preference, simple storage | No shorthand, just `content` (auto mode, free extraction) |

---

## Part 9: Before/After Comparison

### Scenario: Developer Links Memory to Task

**Before (Current API):**

```python
# 15 lines, 5 concepts to understand
await client.add_memory(
    content="John fixed the authentication bug today",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                create="never",
                search=SearchConfig(
                    properties=[
                        PropertyMatch(name="title", mode="semantic", value="authentication bug")
                    ]
                )
            )
        ]
    )
)
```

- **Lines of code:** 15
- **Concepts to learn:** 5 (MemoryPolicy, NodeConstraint, SearchConfig, PropertyMatch, mode)
- **Time to understand:** 20-30 minutes

**After (New Shorthand):**

```python
# 4 lines, 1 concept to understand
await client.add_memory(
    content="John fixed the authentication bug today",
    external_user_id="alice",
    link_to="Task:title~authentication bug"
)
```

- **Lines of code:** 4
- **Concepts to learn:** 1 (link_to DSL)
- **Time to understand:** 2 minutes

### Scenario: Multiple Entities with Constraints

**Before:**

```python
# 25+ lines
await client.add_memory(
    content="John completed the high-priority API review",
    external_user_id="alice",
    memory_policy=MemoryPolicy(
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                search=SearchConfig(
                    properties=[PropertyMatch(name="title", mode="semantic")]
                ),
                set={"status": "completed"},
                when={"priority": "high"}
            ),
            NodeConstraint(
                node_type="Person",
                search=SearchConfig(
                    properties=[PropertyMatch(name="name", mode="semantic")]
                ),
                create="never"
            )
        ]
    )
)
```

**After:**

```python
# 8 lines
await client.add_memory(
    content="John completed the high-priority API review",
    external_user_id="alice",
    link_to={
        "Task:title": {"set": {"status": "completed"}, "when": {"priority": "high"}},
        "Person:name": {"create": "never"}
    }
)
```

### Developer First Integration

**Before:**
1. Read API docs (10 min)
2. Understand memory_policy structure (10 min)
3. Learn NodeConstraint, SearchConfig, PropertyMatch (15 min)
4. Write first linked memory (5 min)
5. Debug validation errors (10 min)
- **Total: 50 minutes**

**After:**
1. Read quickstart (2 min)
2. Copy example with `link_to` (1 min)
3. Modify for their use case (2 min)
- **Total: 5 minutes**

---

## Part 10: API Surface Summary

### `link_to` Parameter

| Type | Description | Example |
|------|-------------|---------|
| `str` | Single entity, semantic match from content | `"Task:title"` |
| `str` | Single entity, exact match | `"Task:id=TASK-123"` |
| `str` | Single entity, semantic with explicit value | `"Task:title~auth bug"` |
| `List[str]` | Multiple entities | `["Task:title", "Person:email"]` |
| `Dict[str, Dict]` | Per-entity constraints | `{"Task:title": {"set": {...}}}` |

### DSL Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `:` | Type/property separator | `Task:title` |
| `=` | Exact match | `id=TASK-123` |
| `~` | Semantic match with explicit value | `title~auth bug` |
| (none after `:`) | Semantic match from content | `Task:title` |

### Special References

| Reference | Meaning |
|-----------|---------|
| `$this` | The memory being created |
| `$previous` | User's most recent memory |
| `$context:N` | Last N memories |

### Constraint Options (Dict Value)

| Field | Type | Description |
|-------|------|-------------|
| `set` | `Dict[str, Any]` | Properties to set |
| `when` | `Dict[str, Any]` | Condition for constraint |
| `create` | `"auto" \| "never"` | Creation policy |

---

## Part 11: Success Metrics

### Developer Experience

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Time-to-first-success | ~30 min | <5 min | User testing |
| Lines of code for link | 15 | 4 | Code samples |
| Concepts to learn | 5 | 1 | Documentation |
| Docs pages to read | 3 | 1 | Analytics |

---

## Part 12: Control Model Analysis

### The Key Question: Who Controls Policy?

Should developers give AI agents control over `memory_policy` and `node_constraints`, or should developers control this themselves?

### Two Mental Models

#### Model A: Developer Controls Policy, Agent Executes

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEVELOPER (Build Time)                       │
├─────────────────────────────────────────────────────────────────┤
│  Defines:                                                        │
│  • Schema (node types, properties, relationships)                │
│  • Constraints (matching rules, create policies)                 │
│  • Guardrails (what can/can't be created)                       │
│  • Tool definitions (what capabilities agent has access to)     │
│                                                                  │
│  UserNodeType("Task", constraint=NodeConstraint(                │
│      search=SearchConfig(properties=["id", "title"]),           │
│      create="auto"                                               │
│  ))                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI AGENT (Run Time)                          │
├─────────────────────────────────────────────────────────────────┤
│  Uses the memory API with developer-defined tool:                │
│  • add_memory(content="...", link_to="Task:title")              │
│  • search_memory(query="...")                                    │
│                                                                  │
│  Cannot:                                                         │
│  • Change matching thresholds (schema-defined)                   │
│  • Override create policies (schema-defined)                     │
│  • Define new node types (schema-defined)                        │
└─────────────────────────────────────────────────────────────────┘
```

**Analogy:** Like giving an intern a style guide. They work within it, don't redefine it.

#### Model B: Agent Has Full Control

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI AGENT (Run Time)                          │
├─────────────────────────────────────────────────────────────────┤
│  Full access to:                                                 │
│  • memory_policy (any mode, any constraint)                     │
│  • node_constraints (any matching, any create policy)           │
│  • PropertyMatch (any threshold, any mode)                      │
│                                                                  │
│  add_memory(content="...", memory_policy={...complex...})       │
└─────────────────────────────────────────────────────────────────┘
```

**Analogy:** Like giving an intern the keys to everything. Flexible but risky.

### What Developers Actually Want

Based on use case analysis:

| Use Case | Who Should Control Policy? | Why? |
|----------|---------------------------|------|
| **Joe.coffee** (POS data) | Developer | Schema is fixed (Transaction, Product, Store). Agent shouldn't invent new types. |
| **DeepTrust** (Compliance) | Developer | SecurityBehavior rules are regulatory. Agent can't modify them. |
| **Meeting Notes** | Developer | Project/Task/Person matching rules should be consistent. |
| **Personal Assistant** | Mixed | User might want to override "always create" for their use case. |

**Insight:** In 90% of cases, developers want to:
1. **Pre-define the guardrails** at schema level
2. **Let agents operate within those guardrails** using the standard API
3. **Choose what capabilities to expose** via tool definitions

### Recommended Control Model: "Schema-Defined, Agent-Executed"

```python
# DEVELOPER (at build time) - full control over schema
schema = UserGraphSchema(
    name="project_management",
    node_types={
        "Task": UserNodeType(
            name="Task",
            constraint=NodeConstraint(
                search=SearchConfig(properties=[
                    PropertyMatch.exact("id"),
                    PropertyMatch.semantic("title", 0.85)
                ]),
                create="auto"  # Agent CAN create Tasks
            )
        ),
        "Person": UserNodeType(
            name="Person",
            constraint=NodeConstraint(
                search=SearchConfig(properties=[
                    PropertyMatch.exact("email"),
                    PropertyMatch.semantic("name", 0.9)
                ]),
                create="never"  # Agent CANNOT create People (controlled vocab)
            )
        )
    }
)

# DEVELOPER (at build time) - defines tool for agent
tools = [
    {
        "name": "add_memory",
        "description": "Store information linked to tasks or people",
        "parameters": {
            "content": {"type": "string"},
            "link_to": {"type": "string", "description": "Task:title or Person:name"}
        }
    }
]

# AI AGENT (at run time) - uses the tool within guardrails
await client.add_memory(
    content="John completed the auth task",
    external_user_id="alice",
    link_to="Task:title"  # Uses schema-defined matching (0.85 threshold)
)

# Agent tries to create Person - blocked by schema
# "Person has create='never', cannot create new Person nodes"
```

### What Agents SHOULD Control (Runtime Values)

Agents should provide **values**, not **structural rules**:

| Agent Provides | Schema Defines |
|----------------|----------------|
| Memory content | Node types and properties |
| Which entity to link to (`link_to` value) | Matching thresholds |
| Property values to set | Create policy (auto/never) |
| Search queries | Unique identifiers |

### Developer Controls Tool Exposure

Developers decide what capabilities their agent has by defining tools:

```python
# MINIMAL: Agent can only store memories (no linking)
tools = [
    {
        "name": "remember",
        "parameters": {"content": {"type": "string"}}
    }
]

# STANDARD: Agent can store and link
tools = [
    {
        "name": "add_memory",
        "parameters": {
            "content": {"type": "string"},
            "link_to": {"type": "string"}  # Expose shorthand
        }
    }
]

# FULL CONTROL: Agent has access to full memory_policy (opt-in)
tools = [
    {
        "name": "add_memory_with_policy",
        "parameters": {
            "content": {"type": "string"},
            "memory_policy": {...}  # Full structure exposed
        }
    }
]
```

### Summary: The Right Model

| Concern | Controlled By | Rationale |
|---------|--------------|-----------|
| **Schema structure** | Developer | Business rules, compliance, data model |
| **Matching strategies** | Developer | Consistency, performance tuning |
| **Create policies** | Developer | Data integrity, controlled vocabularies |
| **Tool definitions** | Developer | What agent can do |
| **Memory content** | Agent | Content decisions |
| **Link targets** | Agent | Runtime context |
| **Property values** | Agent | Dynamic updates |

**Bottom Line:** Developers define the "HOW" (schema + tools), agents provide the "WHAT" (values).

---

## Conclusion

This proposal addresses the core weaknesses identified in the API evaluation:

1. **Time-to-First-Success** → `link_to` shorthand reduces 15 lines to 4
2. **Type Safety** → Schema validation with helpful error messages
3. **Flexibility** → Three forms (string/list/dict) cover simple to complex cases

The changes are **additive** - the full `memory_policy` API remains for power users while `link_to` makes common cases trivial.

**Recommended Implementation Order:**
1. `link_to` shorthand with string/list/dict support (highest impact)
2. Schema validation with developer-friendly errors
3. SDK type generation for TypeScript/Python

---

## Appendix: Full DSL Specification

```
# EBNF Grammar for link_to DSL

link_to     = string_form | list_form | dict_form ;

string_form = entity_ref | special_ref ;
list_form   = "[" (entity_ref | special_ref) { "," (entity_ref | special_ref) } "]" ;
dict_form   = "{" dict_entry { "," dict_entry } "}" ;

dict_entry  = (entity_ref | special_ref) ":" constraint_obj ;
constraint_obj = "{" [ set_clause ] [ when_clause ] [ create_clause ] "}" ;

entity_ref  = type_name ":" prop_name [ operator value ] ;
special_ref = "$this" | "$previous" | "$context:" number ;

set_clause    = '"set"' ":" "{" prop_value { "," prop_value } "}" ;
when_clause   = '"when"' ":" condition ;
create_clause = '"create"' ":" ( '"auto"' | '"never"' ) ;

prop_value  = prop_name ":" ( literal | auto_extract ) ;
auto_extract = "{" '"mode"' ":" '"auto"' [ "," '"text_mode"' ":" text_mode ] "}" ;
text_mode   = '"replace"' | '"append"' | '"merge"' ;

condition   = simple_cond | and_cond | or_cond | not_cond ;
simple_cond = "{" prop_name ":" value "}" ;
and_cond    = "{" '"_and"' ":" "[" condition { "," condition } "]" "}" ;
or_cond     = "{" '"_or"' ":" "[" condition { "," condition } "]" "}" ;
not_cond    = "{" '"_not"' ":" condition "}" ;

type_name   = uppercase { alphanumeric } ;
prop_name   = lowercase { alphanumeric | "_" } ;
operator    = "=" | "~" ;
value       = { any_char - ( "," | "}" | "]" ) } ;
number      = digit { digit } ;

# Examples:
# String form:
#   "Task:title"
#   "Task:id=TASK-123"
#   "Task:title~auth bug"
#   "$previous"
#   "$context:3"
#
# List form:
#   ["Task:title", "Person:email"]
#   ["Task:id=T-1", "$previous"]
#
# Dict form:
#   {"Task:title": {}}
#   {"Task:title": {"set": {"status": "done"}}}
#   {"Task:title": {"set": {"status": "done"}, "when": {"priority": "high"}}}
#   {"Task:title": {"create": "never"}}
#   {"$context:3": {}, "Task:title": {"set": {"status": "done"}}}
```
