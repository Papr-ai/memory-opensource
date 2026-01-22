# Papr Memory API Cleanup Proposal

## Executive Summary

This document proposes a cleanup of the Papr Memory API to address:
1. Duplicate `memory_policy` field in `AddMemoryRequest`
2. Confusing field placement in `MemoryMetadata` vs request-level
3. Rename `PolicyMode.STRUCTURED` to `MANUAL` per design decision
4. Add `memory_policy` support to Schema creation

---

## Issue 1: Duplicate memory_policy in AddMemoryRequest

### Current State
```python
class SchemaSpecificationMixin(BaseModel):
    memory_policy: Optional[MemoryPolicy] = Field(...)  # Defined here
    graph_generation: Optional[GraphGeneration] = Field(...)  # Deprecated

class AddMemoryRequest(SchemaSpecificationMixin):
    # ... other fields ...
    memory_policy: Optional[MemoryPolicy] = Field(...)  # DUPLICATE!
```

### Problem
- `AddMemoryRequest` inherits `memory_policy` from `SchemaSpecificationMixin`
- Then redefines `memory_policy` again at class level
- This creates confusion and potential override issues

### Solution
Remove the duplicate `memory_policy` field from `AddMemoryRequest`. The mixin inheritance is sufficient.

```python
class AddMemoryRequest(SchemaSpecificationMixin):
    content: str = Field(...)
    type: MemoryType = Field(...)
    external_user_id: Optional[str] = Field(...)
    metadata: Optional[MemoryMetadata] = Field(...)
    # ...
    # memory_policy inherited from SchemaSpecificationMixin - DO NOT REDEFINE
```

---

## Issue 2: MemoryMetadata Field Placement Principles

### Current Problems

| Field | In MemoryMetadata | In Request Level | Problem |
|-------|-------------------|------------------|---------|
| `user_id` | Yes | Yes (`AddMemoryRequest`) | Duplicate |
| `external_user_id` | Yes | Yes (`AddMemoryRequest`) | Duplicate |
| `organization_id` | Yes | Yes (`AddMemoryRequest`) | Duplicate |
| `namespace_id` | Yes | Yes (`AddMemoryRequest`) | Duplicate |
| ACL fields | Yes (12+ fields) | No | Too complex |
| `consent`, `risk`, `omo_acl` | Yes | No | Part of OMO |
| `relatedGoals`, `sessionId`, etc. | Yes | No | Internal use |
| `customMetadata` | Yes (nested dict) | No | Developer extension |

### Design Principles for Field Placement

**Principle 1: Request-Level = Identity & Scoping**
Fields at request level control WHO is making the request and WHERE the data belongs:
- `external_user_id` - WHO created this memory (primary user identifier)
- `organization_id` - WHICH organization this belongs to (multi-tenancy)
- `namespace_id` - WHICH namespace within the organization (isolation)

**Principle 2: MemoryMetadata = Content Metadata**
Fields in MemoryMetadata describe the CONTENT of the memory:
- Content classification: `role`, `category`, `topics`, `emotion_tags`, `emoji_tags`
- Content source: `sourceUrl`, `sourceType`, `pageId`, `conversationId`
- Content structure: `hierarchical_structures`
- Content timing: `createdAt`

**Principle 3: Separate Concerns for ACL**
ACL (Access Control List) is a SECURITY concept that controls WHO can access the memory:
- Should be at request level, not buried in metadata
- OMO fields (`consent`, `risk`, `omo_acl`) are also security/safety concepts
- Propose: Move to `memory_policy.acl` or dedicated `acl` field at request level

**Principle 4: Internal Fields Should Be Hidden**
Fields used internally by Papr should not be exposed in API docs:
- `relatedGoals`, `relatedUseCases`, `relatedSteps` - internal classification
- `goalClassificationScores`, etc. - internal scoring
- `sessionId`, `post`, `userMessage`, `assistantMessage` - internal references
- `upload_id` - internal workflow tracking

**Principle 5: Developer Extensions via customMetadata**
Developers should use `customMetadata` for their own fields, not add to MemoryMetadata:
- `customMetadata` supports: string, number, boolean, list of strings
- No nested dicts (keeps it flat and searchable)

### Proposed Structure

```
AddMemoryRequest (Request Level)
├── content: str                    # Required - the memory content
├── type: MemoryType               # Memory type (text, etc.)
│
├── # === IDENTITY & SCOPING ===
├── external_user_id: str          # Primary: WHO created this (your app's user ID)
├── organization_id: str           # Multi-tenant: WHICH organization
├── namespace_id: str              # Multi-tenant: WHICH namespace
│
├── # === PROCESSING POLICY ===
├── memory_policy: MemoryPolicy    # Graph generation mode, constraints
│   ├── mode: "auto" | "manual" | "hybrid"
│   ├── schema_id: str
│   ├── nodes: [...]               # For manual mode
│   ├── relationships: [...]       # For manual mode
│   └── node_constraints: [...]    # For hybrid mode
│
├── # === ACCESS CONTROL (NEW - moved from metadata) ===
├── acl: ACLConfig                 # Simplified ACL
│   ├── read: [user_ids...]
│   └── write: [user_ids...]
│
├── # === OMO SAFETY (NEW - moved from metadata) ===
├── consent: ConsentLevel          # "explicit" | "implicit" | "terms" | "none"
├── risk: RiskLevel                # "none" | "sensitive" | "flagged"
│
├── # === CONTENT METADATA ===
└── metadata: MemoryMetadata       # Content-related metadata only
    ├── role: MessageRole          # "user" | "assistant"
    ├── category: Category         # Based on role
    ├── topics: [str]
    ├── emotion_tags: [str]
    ├── emoji_tags: [str]
    ├── hierarchical_structures: str
    ├── sourceUrl: str
    ├── sourceType: str
    ├── conversationId: str
    ├── location: str
    └── customMetadata: Dict       # Developer extension point
```

### Migration Strategy: Gentle Deprecation

**Phase 1: Accept Both, Prefer Request-Level (Current Release)**
```python
class MemoryMetadata(BaseModel):
    # DEPRECATED - use request-level fields instead
    user_id: Optional[str] = Field(
        None,
        deprecated=True,
        description="DEPRECATED: Use 'external_user_id' at request level. Will be removed in v2."
    )
    external_user_id: Optional[str] = Field(
        None,
        deprecated=True,
        description="DEPRECATED: Use 'external_user_id' at request level. Will be removed in v2."
    )
    organization_id: Optional[str] = Field(
        None,
        deprecated=True,
        description="DEPRECATED: Use 'organization_id' at request level. Will be removed in v2."
    )
    # ... same for other duplicated fields
```

**Internal Behavior:**
- Request-level fields take precedence over metadata fields
- If metadata field is provided but request-level is not, copy it up
- Log deprecation warning when metadata fields are used

**Phase 2: Remove Deprecated Fields (v2)**
- Remove deprecated fields from MemoryMetadata
- Only accept request-level fields

### Internal Fields Handling

For internal fields (`relatedGoals`, `sessionId`, etc.), we have two options:

**Option A: Hide from OpenAPI but keep in model**
```python
class MemoryMetadata(BaseModel):
    # Public fields...

    # Internal fields - excluded from OpenAPI schema
    relatedGoals: Optional[List[str]] = Field(
        default_factory=list,
        json_schema_extra={"hidden": True}  # Custom marker
    )
```

**Option B: Separate InternalMetadata model**
```python
class MemoryMetadata(BaseModel):
    """Public metadata fields for API consumers"""
    role: Optional[MessageRole] = None
    category: Optional[Category] = None
    # ... public fields only

class InternalMemoryMetadata(MemoryMetadata):
    """Extended metadata with internal fields - not exposed in API"""
    relatedGoals: Optional[List[str]] = Field(default_factory=list)
    sessionId: Optional[str] = None
    # ... internal fields
```

**Recommendation**: Option A for backwards compatibility with existing integrations.

---

## Issue 3: Rename PolicyMode.STRUCTURED to MANUAL

### Current State
```python
class PolicyMode(str, Enum):
    AUTO = "auto"
    STRUCTURED = "structured"  # Should be MANUAL
    HYBRID = "hybrid"
```

### Rationale for MANUAL
Based on customer use cases (DeepTrust, Joe.coffee):
1. **Customer terminology**: Both call it "manual graph generation"
2. **Legacy consistency**: `graph_generation.mode` already uses `manual` vs `auto`
3. **Action-oriented**: "Manual" = developer controls, "Structured" = data format (confusing)
4. **Clear intent**: Mode describes WHO controls the graph, not data format

### Solution
```python
class PolicyMode(str, Enum):
    """
    Memory processing mode - describes WHO controls graph generation.

    - AUTO: LLM extracts entities freely (default)
    - MANUAL: Developer provides exact nodes (no LLM extraction)
    - HYBRID: LLM extracts with developer constraints
    """
    AUTO = "auto"
    MANUAL = "manual"       # Renamed from STRUCTURED
    HYBRID = "hybrid"

    # Backwards compatibility alias
    STRUCTURED = "manual"   # Deprecated alias, maps to MANUAL
```

**Validation with alias:**
```python
@field_validator('mode', mode='before')
def normalize_mode(cls, v):
    """Accept 'structured' as alias for 'manual' for backwards compatibility."""
    if v == 'structured':
        logger.warning("mode='structured' is deprecated, use mode='manual' instead")
        return 'manual'
    return v
```

---

## Issue 4: Add memory_policy to Schema Creation

### Current State
Schema creation (`POST /schemas`) creates `UserGraphSchema` but doesn't support `memory_policy`.

### Why Schemas Need memory_policy
1. **Schema-level defaults**: Define default node_constraints for all memories using this schema
2. **OMO policy inheritance**: Set default consent/risk levels for schema
3. **Processing mode**: Define whether schema expects auto, manual, or hybrid mode

### Proposed UserGraphSchema Extension
```python
class UserGraphSchema(BaseModel):
    # Existing fields
    name: str
    description: Optional[str]
    node_types: Dict[str, NodeType]
    relationship_types: Dict[str, RelationshipType]

    # NEW: Schema-level memory policy
    default_memory_policy: Optional[MemoryPolicy] = Field(
        None,
        description="Default memory policy for all memories using this schema. "
                   "Memory-level policy overrides these defaults."
    )
```

### How It Works
```
Memory Request                    Schema
├── memory_policy                ├── default_memory_policy
│   ├── mode: "hybrid"          │   ├── mode: "auto"
│   ├── schema_id: "my_schema"  │   ├── node_constraints: [...]
│   └── node_constraints: [A]   │   └── consent: "implicit"
│
└── Final Policy = Merge(Schema defaults, Memory overrides)
    ├── mode: "hybrid"           # Memory wins
    ├── node_constraints: [A]    # Memory wins (can merge with schema)
    └── consent: "implicit"      # Schema default (not overridden)
```

---

## Implementation Plan

### Phase 1: Model Cleanup (This PR)

1. **Remove duplicate memory_policy from AddMemoryRequest**
   - File: `models/memory_models.py`
   - Just delete the duplicate field

2. **Rename PolicyMode.STRUCTURED to MANUAL**
   - File: `models/shared_types.py`
   - Add backwards compat alias
   - Update all references

3. **Deprecate metadata fields**
   - File: `models/shared_types.py`
   - Add deprecation markers to duplicated fields in MemoryMetadata
   - Add validation warnings

### Phase 2: Request-Level Fields (Follow-up PR)

1. **Add ACL to request level**
   - Add `acl: Optional[ACLConfig]` to AddMemoryRequest
   - Add `consent: Optional[ConsentLevel]` to AddMemoryRequest
   - Add `risk: Optional[RiskLevel]` to AddMemoryRequest

2. **Update processing to prefer request-level**
   - Modify handle_incoming_memory() to copy request-level to metadata
   - Log deprecation warnings

### Phase 3: Schema memory_policy (Follow-up PR)

1. **Add default_memory_policy to UserGraphSchema**
2. **Implement policy merging in handle_incoming_memory()**
3. **Update schema routes**

---

## Summary of Changes

| Issue | Current | Proposed | Breaking? |
|-------|---------|----------|-----------|
| Duplicate memory_policy | Both in mixin and class | Only in mixin | No |
| PolicyMode.STRUCTURED | "structured" | "manual" (accept both) | No |
| Metadata user_id, etc. | In metadata | Deprecated, use request-level | No |
| ACL in metadata | 12+ fields in metadata | Simplified ACL at request level | No |
| Schema memory_policy | Not supported | default_memory_policy field | No |

All changes maintain backwards compatibility through aliases and deprecation warnings.
