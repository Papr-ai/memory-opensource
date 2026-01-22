# Papr Memory API Simplification & Memory-Oriented Policies Plan

## Executive Summary

Design a simpler, more intuitive developer experience for Papr Memory APIs that:
1. Eliminates user ID confusion (`user_id` vs `external_user_id` vs `end_user_id`)
2. Simplifies the metadata structure to avoid common errors
3. Integrates node constraints (memory-oriented policies) cleanly
4. Supports both structured data (graph override) and unstructured data (LLM extraction)
5. Maintains backwards compatibility

---

## Problem Statement

### Issue 1: Inconsistent User ID Naming
- **Document routes** use `end_user_id` (Form param)
- **Memory routes** use `external_user_id` (in MemoryMetadata)
- **Search routes** use both `user_id` and `external_user_id` at top level
- **Confusion**: Developers put their external user ID in `user_id` field (meant for Parse Server internal IDs), causing ACL issues where memories become unsearchable

### Issue 2: user_id vs external_user_id Confusion
- `user_id` = Parse Server internal ObjectId (e.g., `mkcNHhG5KP`)
- `external_user_id` = Developer-provided identifier (e.g., `user_alice_123`, email, UUID)
- **No validation**: If developer puts external ID in `user_id` field, we set ACL to a non-existent Parse user → memory becomes inaccessible

### Issue 3: Nested Metadata Complexity
- `user_id` and `external_user_id` appear both:
  - At request level (`AddMemoryRequest.user_id`)
  - Inside metadata (`AddMemoryRequest.metadata.user_id`)
- Precedence rules are unclear → developers make mistakes

### Issue 4: Node Constraints Not Integrated
- Graph generation settings separate from node constraints
- No unified way to specify policies for structured vs unstructured data
- Open Memory Object (OMO) standard not fully implemented

---

## Unified Architecture: OMO + Node Constraints + Graph Override

### The Three Concepts and How They Fit Together

| Concept | Purpose | When to Use |
|---------|---------|-------------|
| **Graph Override** (`mode="structured"`) | Developer provides exact nodes/relationships | Structured data (Postgres transactions, Linear tasks, CRM records) |
| **Node Constraints** | Policies for LLM-extracted entities | Unstructured data with rules (meetings, emails, documents) |
| **OMO (Open Memory Object)** | Safety/consent/ACL standards | ALL memories - wraps both structured and unstructured |

### Architectural Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    AddMemoryRequest                         │
├─────────────────────────────────────────────────────────────┤
│  content: str                                               │
│  external_user_id: str                                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              OMO Layer (safety)                      │   │
│  │  consent: ConsentLevel                               │   │
│  │  risk: RiskLevel                                     │   │
│  │  acl: ACLConfig                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MemoryPolicy (processing)               │   │
│  │  mode: "auto" | "structured" | "hybrid"              │   │
│  │                                                      │   │
│  │  IF mode="structured":                               │   │
│  │    → nodes: [NodeSpec, ...]      # Exact nodes      │   │
│  │    → relationships: [RelSpec, ...] # Exact edges    │   │
│  │                                                      │   │
│  │  IF mode="auto" or "hybrid":                         │   │
│  │    → node_constraints: [NodeConstraint, ...]         │   │
│  │                                                      │   │
│  │  schema_id: str  # Merge schema-level constraints    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Processing Flow by Mode

**Mode: `auto` (default)**
```
Memory content → LLM extracts entities → Create nodes → Apply OMO safety
```

**Mode: `structured` (graph override)**
```
Memory content → Use provided nodes/relationships directly → Apply OMO safety
(No LLM extraction - developer controls exactly what nodes are created)
```

**Mode: `hybrid` (LLM + constraints)**
```
Memory content → LLM extracts entities → Apply node_constraints → Create/link nodes → Apply OMO safety
```

### Key Design Decision: OMO Wraps Everything

OMO safety standards (consent, risk, ACL) apply to ALL memories regardless of mode:
- Structured data (Joe.coffee transactions) still gets consent tracking and ACL
- Unstructured data (meeting notes) gets the same safety pipeline
- This ensures compliance and audit trails across all data types

### Precedence Rules

```
Value Precedence (highest to lowest):
1. force (NodeConstraint)     ← Always wins
2. merge (NodeConstraint)     ← Only for existing nodes
3. AI extraction              ← LLM-determined values
4. Schema defaults            ← Fallback

Constraint Precedence:
1. Memory-level node_constraints  ← Override
2. Schema-level node_constraints  ← Base
(Memory-level merged on top of schema-level)
```

---

## Proposed Solution: Unified Developer Experience

### Design Principle: "Steve Jobs meets Elon Musk for Developer APIs"
- **Simplicity**: Single, obvious way to do things
- **Control**: Power users can access advanced features
- **Safety**: Prevent common errors with validation
- **Clarity**: Names match meaning

### 1. Standardize on `external_user_id`

**Change**: Make `external_user_id` the primary developer-facing field

```python
# BEFORE (confusing)
class AddMemoryRequest:
    user_id: Optional[str]        # Is this internal or external?
    metadata: MemoryMetadata      # Also has user_id and external_user_id!

class MemoryMetadata:
    user_id: Optional[str]        # Internal Parse ID
    external_user_id: Optional[str]  # Developer's ID

# AFTER (clear)
class AddMemoryRequest:
    external_user_id: Optional[str]  # Primary: Developer's user identifier
    _internal_user_id: Optional[str] # Hidden: Only for advanced use (Parse ID)
    metadata: MemoryMetadata

class MemoryMetadata:
    external_user_id: Optional[str]  # Developer's user identifier (alias: end_user_id)
    # Note: user_id kept for backwards compat but deprecated
```

### 2. Add Validation Layer

```python
# New validation in auth flow
async def validate_user_identification(request, auth_response):
    """
    Validate user IDs and prevent common errors.
    """
    # If user_id is provided, validate it's a real Parse user
    if request.user_id or (request.metadata and request.metadata.user_id):
        internal_id = request.user_id or request.metadata.user_id

        # Check if it looks like an external ID (UUID, email, custom format)
        if looks_like_external_id(internal_id):
            return ErrorResponse(
                code=400,
                error="Invalid user_id format",
                details={
                    "field": "user_id",
                    "provided_value": internal_id,
                    "reason": "This looks like an external user identifier. Did you mean to use 'external_user_id' instead?",
                    "suggestion": "Use 'external_user_id' for your application's user identifiers. "
                                 "'user_id' is reserved for Papr internal user IDs."
                }
            )

        # Validate it's a real Parse user
        parse_user = await memory_graph.fetch_parse_user(internal_id)
        if not parse_user:
            return ErrorResponse(
                code=400,
                error="Invalid user_id",
                details={
                    "field": "user_id",
                    "provided_value": internal_id,
                    "reason": "No Papr user found with this ID. Use 'external_user_id' for your app's user IDs."
                }
            )

    return None  # Validation passed
```

### 3. Unified Memory Object with Node Constraints

Integrate Open Memory Object (OMO) standard with node constraints:

```python
class AddMemoryRequest(BaseModel):
    """Simplified memory request with integrated policies."""

    # Required
    content: str

    # User identification (simplified)
    external_user_id: Optional[str] = Field(
        None,
        description="Your application's user identifier. This is the primary way to identify users."
    )

    # Type
    type: MemoryType = MemoryType.TEXT

    # Metadata (simplified)
    metadata: Optional[MemoryMetadata] = None

    # Memory Policies (NEW - unified node constraints)
    memory_policy: Optional[MemoryPolicy] = Field(
        None,
        description="Policies for how this memory should be processed and stored."
    )

    # Multi-tenant scoping
    organization_id: Optional[str] = None
    namespace_id: Optional[str] = None

    # Deprecated but kept for backwards compatibility
    user_id: Optional[str] = Field(
        None,
        deprecated=True,
        description="DEPRECATED: Use external_user_id instead. Internal Parse user ID."
    )
    graph_generation: Optional[GraphGeneration] = Field(
        None,
        deprecated=True,
        description="DEPRECATED: Use memory_policy.graph_rules instead."
    )


class MemoryPolicy(BaseModel):
    """
    Unified memory-oriented policies.
    Controls how the memory is processed, what graph nodes are created,
    and how they're constrained.
    """

    # Graph generation mode
    mode: PolicyMode = Field(
        PolicyMode.AUTO,
        description="How to generate graph from this memory. "
                   "'auto': LLM extracts entities. "
                   "'structured': You provide exact nodes. "
                   "'hybrid': LLM extracts with constraints."
    )

    # For STRUCTURED mode: Direct graph specification
    nodes: Optional[List[NodeSpec]] = Field(
        None,
        description="For structured data: Exact nodes to create (no LLM extraction)."
    )
    relationships: Optional[List[RelationshipSpec]] = Field(
        None,
        description="For structured data: Exact relationships between nodes."
    )

    # For AUTO/HYBRID mode: Node constraints (policies)
    node_constraints: Optional[List[NodeConstraint]] = Field(
        None,
        description="Rules for how LLM-extracted nodes should be created/updated."
    )

    # Schema reference
    schema_id: Optional[str] = Field(
        None,
        description="Custom schema ID. If set, node_constraints from schema are merged."
    )


class PolicyMode(str, Enum):
    AUTO = "auto"           # LLM extracts entities freely
    STRUCTURED = "structured"  # Developer provides exact nodes (graph_override)
    HYBRID = "hybrid"       # LLM extracts with constraints


class NodeConstraint(BaseModel):
    """
    Policy for how nodes of a specific type should be handled.
    Aligned with policy-oriented-node-constraints branch design.
    """
    # === WHAT ===
    node_type: str = Field(..., description="Node type this constraint applies to (e.g., 'Task', 'Project')")

    # === WHEN (conditional application) ===
    when: Optional[Dict[str, Any]] = Field(
        None,
        description="Condition for when this constraint applies. All conditions must match (AND logic).",
        alias="match"  # Backwards compatibility
    )

    # === CREATION POLICY ===
    create: Literal["auto", "never"] = Field(
        "auto",
        description="'auto': Create if not found via search. 'never': Only link to existing (controlled vocabulary)."
    )

    # === NODE SELECTION ===
    node_id: Optional[str] = Field(
        None,
        description="Direct: Skip search, use this exact node ID."
    )
    search: Optional[SearchConfig] = Field(
        None,
        description="How to find existing nodes for linking/updating."
    )

    # === VALUE POLICIES ===
    force: Optional[Dict[str, Any]] = Field(
        None,
        description="Force these property values on ALL matching nodes (new or existing). Always wins over AI extraction.",
        alias="set"  # Backwards compatibility
    )
    merge: Optional[List[str]] = Field(
        None,
        description="Update these properties from LLM extraction on EXISTING nodes only.",
        alias="update_on_match"  # Backwards compatibility (also accept 'update')
    )

    class Config:
        populate_by_name = True  # Accept both field name and alias


class SearchConfig(BaseModel):
    """Configuration for finding existing nodes."""
    mode: Literal["semantic", "exact", "fuzzy"] = Field(
        "semantic",
        description="Search mode: 'semantic' (vector similarity), 'exact' (property match), 'fuzzy' (partial match)"
    )
    threshold: float = Field(
        0.85,
        description="Similarity threshold for semantic search (0.0-1.0)"
    )
    properties: Optional[List[str]] = Field(
        None,
        description="For exact/fuzzy mode: which properties to match on"
    )
```

### 4. Simplified API Examples

#### Example 1: Simple Memory (No Policy Needed)
```python
# Most common case - just add a memory
response = await client.add_memory(
    content="User prefers dark mode and large fonts",
    external_user_id="user_alice_123"
)
# Graph is auto-generated, no constraints
```

#### Example 2: Structured Data (Joe.coffee Transaction)
```python
# Structured data from Postgres - developer knows exactly what nodes to create
response = await client.add_memory(
    content="Transaction: Alice bought Latte for $5.50",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="structured",
        nodes=[
            NodeSpec(
                id="txn_12345",
                type="Transaction",
                properties={"amount": 5.50, "product": "Latte", "timestamp": "2026-01-21T10:30:00Z"}
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

#### Example 3: Hybrid - Meeting Notes with Task Constraints
```python
# Unstructured meeting notes, but we want specific Task handling
response = await client.add_memory(
    content="Meeting: Discussed Project Alpha. John will complete the API review by Friday.",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="hybrid",
        schema_id="project_management_schema",
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                create="never",  # Only link to existing tasks (from Linear)
                search=SearchConfig(mode="semantic", threshold=0.85),
                merge=["status", "assignee"]  # Allow AI to update these on EXISTING nodes
            ),
            NodeConstraint(
                node_type="Project",
                force={"workspace_id": "ws_123"},  # Force workspace on ALL projects
                merge=["status"]  # Update status from AI on existing
            ),
            NodeConstraint(
                node_type="Person",
                create="never",  # Controlled vocabulary - only existing team members
                search=SearchConfig(mode="semantic", threshold=0.90)
            )
        ]
    )
)
```

#### Example 4: Conditional Constraints with `when`
```python
# Apply different policies based on node properties
response = await client.add_memory(
    content="High priority task: Fix authentication bug. Low priority: Update README.",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="hybrid",
        node_constraints=[
            NodeConstraint(
                node_type="Task",
                when={"priority": "high"},  # Only apply to high-priority tasks
                force={"urgent": True, "notify_team": True}
            ),
            NodeConstraint(
                node_type="Task",
                when={"priority": "low"},  # Different policy for low-priority
                force={"urgent": False}
            )
        ]
    )
)
```

#### Example 5: Direct Node Reference with `node_id`
```python
# Link to a specific existing node by ID
response = await client.add_memory(
    content="Update on Project Alpha: milestone 2 completed.",
    external_user_id="user_alice_123",
    memory_policy=MemoryPolicy(
        mode="hybrid",
        node_constraints=[
            NodeConstraint(
                node_type="Project",
                node_id="proj_alpha_123",  # Skip search, use this exact node
                merge=["status", "milestone"]  # Update these from AI
            )
        ]
    )
)
```

### 5. Document Upload Consistency

Fix `end_user_id` vs `external_user_id` in document routes:

```python
# BEFORE (document_routes_v2.py)
async def upload_document(
    user_id: Optional[str] = Form(None),      # Confusing name
    end_user_id: Optional[str] = Form(None),  # Different from memory routes!
)

# AFTER
async def upload_document(
    external_user_id: Optional[str] = Form(
        None,
        description="Your application's user identifier",
        alias="end_user_id"  # Accept old name for backwards compat
    ),
    _internal_user_id: Optional[str] = Form(
        None,
        alias="user_id",  # Accept old name but deprecated
        description="DEPRECATED: Internal Parse user ID. Use external_user_id instead."
    ),
)
```

---

## Implementation Plan

### Phase 1: Validation & Error Messages (Quick Win)
**Goal**: Prevent the most common error (external ID in user_id field)

1. Add `looks_like_external_id()` heuristic function
2. Add Parse user validation when `user_id` is provided
3. Return clear, actionable error messages
4. Update error responses with suggestions

**Files to modify**:
- `services/auth_utils.py` - Add validation
- `routers/v1/memory_routes_v1.py` - Call validation
- `routers/v1/document_routes_v2.py` - Call validation

### Phase 2: Standardize on external_user_id
**Goal**: Make API consistent across all endpoints

1. Add `external_user_id` as primary field at request level
2. Add `alias="end_user_id"` for backwards compatibility in documents
3. Deprecate `user_id` at request level (keep in metadata for advanced use)
4. Update SDK documentation

**Files to modify**:
- `models/memory_models.py` - AddMemoryRequest, SearchRequest
- `models/shared_types.py` - MemoryMetadata
- `routers/v1/document_routes_v2.py` - Form params
- `routers/v1/memory_routes_v1.py` - Handler logic
- `routers/v1/message_routes.py` - Parameter naming

### Phase 3: Unified MemoryPolicy Model
**Goal**: Integrate node constraints with simplified API

1. Create new `MemoryPolicy` model
2. Create `NodeConstraint`, `NodeSpec`, `RelationshipSpec` models
3. Add `PolicyMode` enum (auto, structured, hybrid)
4. Map old `graph_generation` to new `memory_policy` (backwards compat)
5. Implement constraint merging (schema-level + memory-level)

**Files to modify**:
- `models/memory_models.py` - New models
- `models/shared_types.py` - NodeConstraint
- `memory/memory_graph.py` - Apply constraints
- `services/graph_extraction.py` - Constraint application

### Phase 4: OMO Integration
**Goal**: Full Open Memory Object standard support

1. Add OMO `consent` and `risk` fields
2. Add `ext.papr:*` namespace support
3. Implement safety standards (consent enforcement, risk filtering)
4. Add audit trail for compliance

**Files to modify**:
- `models/memory_models.py` - OMO fields
- `models/parse_server.py` - Storage mapping
- New file: `services/omo_safety.py`

### Phase 5: Documentation & Migration
**Goal**: Help existing users migrate smoothly

1. Update API documentation
2. Add migration guide
3. Add deprecation warnings (not errors) for old field names
4. Update SDK with new types

---

## Backwards Compatibility Strategy

| Old API | New API | Migration |
|---------|---------|-----------|
| `user_id` in request | `external_user_id` | Accept both, log deprecation warning |
| `end_user_id` (documents) | `external_user_id` | Alias support, accept both |
| `graph_generation` | `memory_policy` | Auto-convert internally |
| `graph_generation.manual` | `memory_policy.mode="structured"` | Auto-convert |
| `ext.papr:node_constraints` | `memory_policy.node_constraints` | Support both |
| `set` (NodeConstraint) | `force` | Alias support via Pydantic |
| `update` / `update_on_match` | `merge` | Alias support via Pydantic |
| `match` (NodeConstraint) | `when` | Alias support via Pydantic |

All changes are additive - existing integrations continue working with deprecation warnings.

---

## Verification Plan

### Unit Tests
1. Test validation catches external ID in user_id field
2. Test external_user_id flows through correctly
3. Test NodeConstraint application (set, update, match)
4. Test backwards compatibility (old API still works)

### Integration Tests
1. Add memory with external_user_id → verify ACL correct
2. Add memory with invalid user_id → verify clear error
3. Add memory with memory_policy.mode="structured" → verify nodes created exactly
4. Add memory with node_constraints → verify LLM respects them
5. Document upload with external_user_id → verify consistency

### End-to-End Tests
1. Create user via API → add memories → search → verify isolation
2. Test Joe.coffee scenario: structured transaction data → graph override
3. Test meeting notes scenario: unstructured + task constraints

---

## User Preferences (Confirmed)

1. **Deprecation Strategy**: **Gentle** - Accept old names silently, log deprecation warnings. No breaking changes.
2. **Implementation Priority**: **Full Implementation** - All phases together for comprehensive solution.
3. **OMO Support**: **Full OMO** - Include consent, risk, ACL propagation, and audit trails.

---

## Full OMO Implementation Details

### OMO Fields to Add

```python
class MemoryMetadata(BaseModel):
    """Enhanced with OMO standard fields."""

    # Existing fields...
    external_user_id: Optional[str] = None

    # NEW: OMO Safety Standards
    consent: ConsentLevel = Field(
        ConsentLevel.IMPLICIT,
        description="How the data owner allowed this memory to be stored/used. "
                   "'explicit': User explicitly agreed. "
                   "'implicit': Inferred from context. "
                   "'terms': Covered by ToS. "
                   "'none': No consent recorded."
    )

    risk: RiskLevel = Field(
        RiskLevel.NONE,
        description="Post-ingest safety assessment. "
                   "'none': Safe content. "
                   "'sensitive': Contains PII or sensitive info. "
                   "'flagged': Requires review before retrieval."
    )

    # NEW: ACL (simplified from OMO)
    acl: Optional[ACLConfig] = Field(
        None,
        description="Access control list. If not provided, defaults to developer + external_user."
    )


class ConsentLevel(str, Enum):
    EXPLICIT = "explicit"   # User explicitly agreed to store
    IMPLICIT = "implicit"   # Inferred from usage context
    TERMS = "terms"         # Covered by Terms of Service
    NONE = "none"          # No consent recorded


class RiskLevel(str, Enum):
    NONE = "none"           # Safe content
    SENSITIVE = "sensitive" # Contains PII, financial, health info
    FLAGGED = "flagged"     # Requires human review


class ACLConfig(BaseModel):
    """Simplified ACL from OMO standard."""
    read: List[str] = Field(default_factory=list, description="User IDs that can read")
    write: List[str] = Field(default_factory=list, description="User IDs that can write")
```

### Safety Standards Implementation

```python
# New file: services/omo_safety.py

async def enforce_consent_standard(memory: AddMemoryRequest, nodes: List[Dict]) -> List[Dict]:
    """Skip or annotate based on consent level."""
    consent = memory.metadata.consent if memory.metadata else ConsentLevel.IMPLICIT

    if consent == ConsentLevel.NONE:
        logger.warning(f"Memory has no consent - skipping graph extraction")
        return []  # Don't extract nodes without consent

    # Annotate all nodes with consent provenance
    for node in nodes:
        node["properties"]["_omo_consent"] = consent.value
        node["properties"]["_omo_source_memory_id"] = memory.id

    return nodes


async def enforce_risk_standard(memory: AddMemoryRequest, nodes: List[Dict]) -> List[Dict]:
    """Apply stricter constraints for high-risk content."""
    risk = memory.metadata.risk if memory.metadata else RiskLevel.NONE

    if risk == RiskLevel.FLAGGED:
        for node in nodes:
            node["properties"]["_omo_risk"] = "flagged"
            node["properties"]["_omo_requires_review"] = True
            # Restrict ACL to only the memory owner
            node["acl"] = {"read": [memory.external_user_id], "write": [memory.external_user_id]}

    elif risk == RiskLevel.SENSITIVE:
        for node in nodes:
            node["properties"]["_omo_risk"] = "sensitive"

    return nodes


async def propagate_acl(memory: AddMemoryRequest, nodes: List[Dict]) -> List[Dict]:
    """Propagate ACL from memory to extracted nodes."""
    if memory.metadata and memory.metadata.acl:
        for node in nodes:
            node["acl"] = {
                "read": memory.metadata.acl.read,
                "write": memory.metadata.acl.write
            }
    return nodes


async def create_audit_trail(memory: AddMemoryRequest, nodes: List[Dict]) -> List[Dict]:
    """Add audit trail for compliance."""
    for node in nodes:
        node["properties"]["_omo_audit"] = {
            "source_memory_id": memory.id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "consent": memory.metadata.consent.value if memory.metadata else "implicit",
            "risk": memory.metadata.risk.value if memory.metadata else "none",
            "extraction_method": "llm" if not memory.memory_policy or memory.memory_policy.mode != PolicyMode.STRUCTURED else "manual"
        }
    return nodes
```

### Integration Points

The OMO safety pipeline integrates at the graph extraction stage:

```python
# In memory_graph.py or graph_extraction.py

async def process_memory_with_omo(
    memory: AddMemoryRequest,
    extracted_nodes: List[Dict]
) -> List[Dict]:
    """Apply OMO safety standards pipeline."""

    # 1. Consent check (may skip extraction)
    nodes = await enforce_consent_standard(memory, extracted_nodes)
    if not nodes:
        return []

    # 2. Risk assessment (may restrict ACL)
    nodes = await enforce_risk_standard(memory, nodes)

    # 3. ACL propagation (memory ACL → node ACL)
    nodes = await propagate_acl(memory, nodes)

    # 4. Audit trail (compliance tracking)
    nodes = await create_audit_trail(memory, nodes)

    return nodes
```

---

## Complete File Change List

### Core Models
| File | Changes |
|------|---------|
| `models/memory_models.py` | Add `external_user_id` to AddMemoryRequest/SearchRequest, add `MemoryPolicy`, `NodeConstraint`, `PolicyMode`, deprecate `graph_generation` |
| `models/shared_types.py` | Add `consent`, `risk`, `ACLConfig` to MemoryMetadata, add `NodeSpec`, `RelationshipSpec`, `SearchConfig` |
| `models/parse_server.py` | Add OMO fields to Memory storage model |

### Routes
| File | Changes |
|------|---------|
| `routers/v1/memory_routes_v1.py` | Call validation, handle `external_user_id`, route to OMO pipeline |
| `routers/v1/document_routes_v2.py` | Rename `end_user_id` → `external_user_id` with alias, add validation |
| `routers/v1/message_routes.py` | Standardize parameter naming |
| `routers/v1/schema_routes_v1.py` | Add `node_constraints` to schema endpoints |

### Services
| File | Changes |
|------|---------|
| `services/auth_utils.py` | Add `validate_user_identification()`, `looks_like_external_id()` |
| New: `services/omo_safety.py` | OMO safety standards implementation |
| New: `services/memory_policy_resolver.py` | Resolve schema + memory level constraints |
| `memory/memory_graph.py` | Integrate OMO pipeline, apply node constraints |

### Documentation
| File | Changes |
|------|---------|
| New: `docs/features/memory_oriented_policies/MEMORY_POLICY_API.md` | API documentation |
| New: `docs/features/memory_oriented_policies/MIGRATION_GUIDE.md` | Migration guide for existing users |
| Update: `docs/architecture/OMO_NODE_CONSTRAINTS_INTEGRATION.md` | Mark as implemented |

---

## Testing Strategy

### Unit Tests to Add
```python
# tests/test_user_id_validation.py
def test_external_id_in_user_id_field_returns_helpful_error()
def test_valid_parse_user_id_passes()
def test_external_user_id_flows_to_acl_correctly()
def test_backwards_compat_user_id_still_works()

# tests/test_memory_policy.py
def test_structured_mode_creates_exact_nodes()
def test_hybrid_mode_applies_constraints()
def test_node_constraint_force_overrides_ai()        # force > merge > AI
def test_node_constraint_merge_only_on_existing()    # merge only for existing nodes
def test_node_constraint_when_filters_correctly()    # conditional application
def test_node_constraint_node_id_skips_search()      # direct node reference
def test_node_constraint_create_never_links_only()   # controlled vocabulary
def test_schema_and_memory_constraints_merge()
def test_backwards_compat_set_alias_works()          # set → force
def test_backwards_compat_update_alias_works()       # update → merge
def test_backwards_compat_match_alias_works()        # match → when

# tests/test_omo_safety.py
def test_no_consent_skips_extraction()
def test_flagged_risk_restricts_acl()
def test_audit_trail_created()
def test_acl_propagates_to_nodes()
```

### Integration Tests
```bash
# Run full flow tests
pytest tests/integration/test_memory_flow.py -v
pytest tests/integration/test_document_flow.py -v
```

### Manual Verification
1. Create memory with external_user_id → verify ACL correct
2. Try invalid user_id → verify clear error message
3. Add structured memory (Joe.coffee style) → verify exact nodes
4. Add hybrid memory with constraints → verify AI respects them
5. Add flagged content → verify restricted ACL

---

## Implementation Order

1. **Week 1**: Validation layer + error messages
   - `looks_like_external_id()` heuristic
   - Parse user validation
   - Clear error responses

2. **Week 2**: Model changes + backwards compat
   - Add `external_user_id` to all models
   - Add aliases for old field names
   - Add deprecation logging

3. **Week 3**: MemoryPolicy implementation
   - New `MemoryPolicy` model
   - `NodeConstraint` application logic
   - Schema + memory constraint merging

4. **Week 4**: OMO safety standards
   - Consent/risk fields
   - ACL propagation
   - Audit trail
   - Safety pipeline integration

5. **Week 5**: Documentation + testing
   - API docs
   - Migration guide
   - Full test suite
   - SDK updates

---

## Success Metrics

1. **Error Reduction**: 90% decrease in "memories not searchable" support tickets
2. **Developer Experience**: Clear error messages lead to self-service resolution
3. **API Consistency**: Single `external_user_id` field across all endpoints
4. **Compliance Ready**: Full audit trail for enterprise customers

---

## Implementation Tasks

### Phase 1: Validation & Error Messages

- [ ] **Task 1.1**: Create `looks_like_external_id()` heuristic function in `services/auth_utils.py`
  - Detect UUIDs, emails, custom prefixes (user_, ext_, etc.)
  - Return True if value matches external ID patterns

- [ ] **Task 1.2**: Add `validate_user_identification()` async function in `services/auth_utils.py`
  - Check if user_id looks like external ID → return helpful error
  - Validate Parse user exists when user_id provided
  - Return clear, actionable error messages with suggestions

- [ ] **Task 1.3**: Integrate validation into `memory_routes_v1.py` add_memory endpoint
  - Call validation before processing
  - Return structured error response

- [ ] **Task 1.4**: Integrate validation into `document_routes_v2.py` upload_document endpoint
  - Same validation logic for document uploads

- [ ] **Task 1.5**: Write unit tests for validation functions
  - Test UUID detection
  - Test email detection
  - Test custom prefix detection
  - Test Parse user validation

### Phase 2: Standardize external_user_id

- [ ] **Task 2.1**: Add `external_user_id` field to `AddMemoryRequest` in `models/memory_models.py`
  - Primary field for developer user identification
  - Clear documentation string

- [ ] **Task 2.2**: Add `external_user_id` field to `SearchRequest` in `models/memory_models.py`
  - Same treatment for search operations

- [ ] **Task 2.3**: Mark `user_id` as deprecated in request models
  - Add deprecation notice in docstring
  - Keep for backwards compatibility

- [ ] **Task 2.4**: Update `document_routes_v2.py` Form params
  - Rename `end_user_id` → `external_user_id` with alias
  - Keep alias for backwards compatibility

- [ ] **Task 2.5**: Add deprecation logging
  - Log warning when deprecated fields are used
  - Include migration suggestion in log

- [ ] **Task 2.6**: Update handler logic to use new field names
  - Precedence: request-level external_user_id > metadata.external_user_id

### Phase 3: Unified MemoryPolicy Model

- [ ] **Task 3.1**: Create `PolicyMode` enum in `models/memory_models.py`
  - AUTO, STRUCTURED, HYBRID modes

- [ ] **Task 3.2**: Create `NodeSpec` model in `models/shared_types.py`
  - id, type, properties for structured data

- [ ] **Task 3.3**: Create `RelationshipSpec` model in `models/shared_types.py`
  - source, target, type for structured relationships

- [ ] **Task 3.4**: Create `SearchConfig` model in `models/shared_types.py`
  - mode (semantic/exact/fuzzy), threshold, properties

- [ ] **Task 3.5**: Create `NodeConstraint` model in `models/shared_types.py`
  - node_type, when, create, node_id, search, force, merge
  - With aliases for backwards compatibility (set→force, match→when)

- [ ] **Task 3.6**: Create `MemoryPolicy` model in `models/memory_models.py`
  - mode, nodes, relationships, node_constraints, schema_id

- [ ] **Task 3.7**: Add `memory_policy` field to `AddMemoryRequest`
  - Optional, defaults to None (auto mode)

- [ ] **Task 3.8**: Create `services/memory_policy_resolver.py`
  - Merge schema-level and memory-level constraints
  - Handle precedence rules

- [ ] **Task 3.9**: Implement constraint application in `memory/memory_graph.py`
  - Apply force values
  - Handle merge for existing nodes
  - Filter by when conditions

- [ ] **Task 3.10**: Add backwards compatibility for `graph_generation`
  - Auto-convert to memory_policy internally

### Phase 4: OMO Integration

- [ ] **Task 4.1**: Create `ConsentLevel` enum in `models/shared_types.py`
  - EXPLICIT, IMPLICIT, TERMS, NONE

- [ ] **Task 4.2**: Create `RiskLevel` enum in `models/shared_types.py`
  - NONE, SENSITIVE, FLAGGED

- [ ] **Task 4.3**: Create `ACLConfig` model in `models/shared_types.py`
  - read and write lists

- [ ] **Task 4.4**: Add OMO fields to `MemoryMetadata`
  - consent, risk, acl fields

- [ ] **Task 4.5**: Create `services/omo_safety.py`
  - enforce_consent_standard()
  - enforce_risk_standard()
  - propagate_acl()
  - create_audit_trail()

- [ ] **Task 4.6**: Create `process_memory_with_omo()` pipeline function
  - Chain all safety standards

- [ ] **Task 4.7**: Integrate OMO pipeline into `memory_graph.py`
  - Call after graph extraction

- [ ] **Task 4.8**: Add OMO fields to `parse_server.py` Memory model
  - Storage mapping for Parse Server

### Phase 5: Documentation & Migration

- [ ] **Task 5.1**: Create `docs/features/memory_oriented_policies/MEMORY_POLICY_API.md`
  - Full API documentation with examples

- [ ] **Task 5.2**: Create `docs/features/memory_oriented_policies/MIGRATION_GUIDE.md`
  - Step-by-step migration for existing users
  - Before/after code examples

- [ ] **Task 5.3**: Update `docs/architecture/OMO_NODE_CONSTRAINTS_INTEGRATION.md`
  - Mark as implemented, link to new docs

- [ ] **Task 5.4**: Write comprehensive unit tests
  - Validation tests
  - MemoryPolicy tests
  - OMO safety tests
  - Backwards compatibility tests

- [ ] **Task 5.5**: Write integration tests
  - Full flow tests for each mode
  - ACL verification tests

- [ ] **Task 5.6**: Update SDK documentation (if applicable)
  - TypeScript/Python SDK updates
