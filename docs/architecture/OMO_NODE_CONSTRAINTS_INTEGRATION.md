# OMO + Node Constraints Integration for Safety Standards

## 🎯 Overview

This document describes how **Open Memory Object (OMO)** integrates with **Node Constraints** to provide safety standards, data governance, and policy enforcement for memory storage and retrieval. This integration enables:

- ✅ **Safety Standards**: Enforce consent, risk assessment, and access control at the memory level
- ✅ **Data Governance**: Control how AI-extracted entities become nodes via constraints
- ✅ **Policy Enforcement**: Apply workspace-level policies to all memories and extracted nodes
- ✅ **Schema-Level Constraints**: Define constraints at schema level that apply to all memories using that schema
- ✅ **Memory-Level Constraints**: Override schema constraints for specific memories
- ✅ **Compliance**: Track consent, risk levels, and access patterns for audit trails

### Constraint Hierarchy

**Precedence (highest to lowest):**
1. **Memory-level constraints** (from `ext.papr:node_constraints`) - Override everything
2. **Schema-level constraints** (from `UserGraphSchema.node_constraints`) - Base rules
3. **System defaults** - Fallback behavior

**Use Cases:**
- **Schema-level**: "All Customer nodes must have workspace_id" (applies to all memories using this schema)
- **Memory-level**: "This specific memory: Customer nodes should also have priority='high'" (overrides/adds to schema constraints)

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Input (OMO Format)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OMO Object:                                                    │
│  {                                                              │
│    "id": "omo:memory:abc123",                                  │
│    "type": "text",                                             │
│    "content": "Project Alpha is completed",                     │
│    "consent": "explicit",                                      │
│    "risk": "none",                                             │
│    "acl": {"read": ["user:123"], "write": ["user:123"]},      │
│    "ext": {                                                    │
│      "papr:schema_id": "schema_123",  ← Schema reference      │
│      "papr:node_constraints": [...]    ← Memory-level (optional)│
│    }                                                            │
│  }                                                              │
│                                                                 │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   │ Fetch Schema + Resolve Constraints
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│         Schema-Level Constraints (Base Rules)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  UserGraphSchema:                                               │
│  {                                                              │
│    "id": "schema_123",                                          │
│    "node_constraints": [  ← Schema-level constraints          │
│      {"node_type": "Project", "force": {"workspace_id": "ws_123"}},
│      {"node_type": "Person", "create": "never"}                │
│    ]                                                            │
│  }                                                              │
│                                                                 │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   │ Merge: Schema + Memory Constraints
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│              Node Constraint Engine                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Fetch schema-level constraints (from schema_id)            │
│  2. Merge with memory-level constraints (memory-level wins)     │
│  3. Extract nodes from OMO content (LLM)                      │
│  4. Apply merged constraints to nodes                          │
│  5. Enforce safety standards (consent, risk, ACL)              │
│  6. Store in paprDB (SQLite) or cloud (Neo4j + Qdrant)       │
│                                                                 │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   │ Constrained Nodes
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│              Storage (paprDB / Cloud)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Nodes:                                                         │
│  - Project(name="Alpha", status="completed",                    │
│         workspace_id="ws_123",  ← from schema constraint       │
│         priority="high",         ← from memory constraint      │
│         consent="explicit",      ← from OMO.consent            │
│         risk="none")             ← from OMO.risk               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Integration Flow

### Phase 1: Memory Input (OMO Format)

#### Step 1.1: User Creates Memory with OMO

```python
# User creates memory following OMO standard
memory_omo = {
    "$schema": "https://open-memory.org/OMO-v1.schema.json",
    "id": "omo:memory:abc123",
    "createdAt": "2024-01-15T10:30:00Z",
    "type": "text",
    "content": "Project Alpha is now completed. Team backend worked on it.",
    "consent": "explicit",  # Safety: explicit user consent
    "risk": "none",          # Safety: no sensitive data
    "topics": ["project-management", "completion"],
    "sourceUrl": "https://example.com/meeting-notes",
    "acl": {
        "read": ["user:123", "workspace:ws_123"],
        "write": ["user:123"]
    },
    "ext": {
        # papr-specific extensions
        "papr:node_constraints": [
            {
                "node_type": "Project",
                "force": {"workspace_id": "ws_123", "team": "backend"},
                "merge": ["status"],
                "search": {"mode": "semantic", "threshold": 0.85}
            },
            {
                "node_type": "Person",
                "create": "never",  # Controlled vocabulary
                "search": {"mode": "semantic", "threshold": 0.90}
            }
        ],
        "papr:workspace_id": "ws_123",
        "papr:user_id": "user_123",
        "papr:metadata": {
            "conversationId": "conv_456",
            "sessionId": "session_789"
        }
    }
}
```

#### Step 1.2: Validate OMO Schema

```python
from jsonschema import validate
import json

def validate_omo(memory_omo: dict) -> bool:
    """Validate memory against OMO schema"""
    with open("OMO-v1.schema.json") as f:
        omo_schema = json.load(f)
    
    try:
        validate(instance=memory_omo, schema=omo_schema)
        return True
    except ValidationError as e:
        logger.error(f"OMO validation failed: {e}")
        return False
```

---

### Phase 2: Node Extraction with Safety Checks

#### Step 2.1: Extract Nodes from OMO Content

```python
async def extract_nodes_from_omo(memory_omo: dict) -> List[Dict]:
    """
    Extract nodes from OMO content, respecting safety standards.
    """
    content = memory_omo["content"]
    consent = memory_omo.get("consent", "none")
    risk = memory_omo.get("risk", "none")
    
    # Safety check: Skip extraction if no consent
    if consent == "none":
        logger.warning("Skipping extraction: no consent")
        return []
    
    # Safety check: Flag high-risk content
    if risk == "flagged":
        logger.warning("High-risk content detected - applying strict constraints")
        # Apply additional safety constraints
    
    # Extract nodes using LLM (existing logic)
    extracted_nodes = await llm_extract_nodes(
        content=content,
        schema_id=memory_omo.get("ext", {}).get("papr:schema_id")
    )
    
    # Annotate each node with OMO safety metadata
    for node in extracted_nodes:
        node["omo_metadata"] = {
            "consent": consent,
            "risk": risk,
            "source_memory_id": memory_omo["id"],
            "created_at": memory_omo["createdAt"]
        }
    
    return extracted_nodes
```

#### Step 2.2: Resolve Constraints (Schema-Level + Memory-Level)

```python
async def resolve_constraints(
    memory_omo: dict,
    schema_service: SchemaService
) -> List[Dict]:
    """
    Resolve constraints by merging schema-level and memory-level constraints.
    Memory-level constraints override schema-level constraints.
    """
    schema_id = memory_omo.get("ext", {}).get("papr:schema_id")
    memory_constraints = memory_omo.get("ext", {}).get("papr:node_constraints", [])
    
    # Start with schema-level constraints (if schema_id provided)
    schema_constraints = []
    if schema_id:
        schema = await schema_service.get_schema(schema_id)
        if schema and hasattr(schema, "node_constraints"):
            schema_constraints = schema.node_constraints or []
            logger.info(f"Loaded {len(schema_constraints)} schema-level constraints from schema {schema_id}")
    
    # Merge constraints: memory-level overrides schema-level
    merged_constraints = merge_constraints(schema_constraints, memory_constraints)
    
    return merged_constraints

def merge_constraints(
    schema_constraints: List[Dict],
    memory_constraints: List[Dict]
) -> List[Dict]:
    """
    Merge schema-level and memory-level constraints.
    Memory-level constraints override schema-level for same node_type.
    """
    # Start with schema constraints as base
    merged = {c["node_type"]: c.copy() for c in schema_constraints}
    
    # Apply memory constraints (override schema)
    for mem_constraint in memory_constraints:
        node_type = mem_constraint["node_type"]
        
        if node_type in merged:
            # Merge: memory-level overrides schema-level
            schema_const = merged[node_type]
            
            # Merge 'force': memory-level properties override schema-level
            if mem_constraint.get("force"):
                schema_force = schema_const.get("force", {})
                schema_force.update(mem_constraint["force"])  # Memory wins
                merged[node_type]["force"] = schema_force
            
            # Merge 'merge': combine lists (unique)
            if mem_constraint.get("merge"):
                schema_merge = set(schema_const.get("merge", []))
                schema_merge.update(mem_constraint["merge"])
                merged[node_type]["merge"] = list(schema_merge)
            
            # Override other fields (memory wins)
            for key in ["create", "node_id", "search", "when"]:
                if key in mem_constraint:
                    merged[node_type][key] = mem_constraint[key]
        else:
            # New constraint (not in schema) - add it
            merged[node_type] = mem_constraint.copy()
    
    return list(merged.values())
```

#### Step 2.3: Apply Node Constraints

```python
async def apply_constraints_to_nodes(
    nodes: List[Dict],
    node_constraints: List[Dict],
    omo_metadata: Dict
) -> List[Dict]:
    """
    Apply node constraints to extracted nodes, incorporating OMO safety standards.
    """
    constrained_nodes = []
    
    for node in nodes:
        node_type = node["type"]
        
        # Find applicable constraints
        applicable_constraints = [
            c for c in node_constraints
            if c["node_type"] == node_type
            and matches_when_condition(node, c.get("when"))
        ]
        
        # Apply constraints
        for constraint in applicable_constraints:
            # 1. Apply 'force' (override values)
            if constraint.get("force"):
                node["properties"].update(constraint["force"])
            
            # 2. Enforce OMO safety standards
            node["properties"].update({
                "consent": omo_metadata["consent"],
                "risk": omo_metadata["risk"],
                "source_memory_id": omo_metadata["source_memory_id"]
            })
            
            # 3. Apply ACL from OMO
            if "acl" in omo_metadata:
                node["acl"] = omo_metadata["acl"]
            
            # 4. Search for existing node (if needed)
            if not constraint.get("node_id"):
                existing = await search_existing_node(
                    node_type=node_type,
                    properties=node["properties"],
                    search_config=constraint.get("search", {})
                )
                
                if existing:
                    # Apply 'merge' (update existing)
                    if constraint.get("merge"):
                        for prop in constraint["merge"]:
                            if prop in node["properties"]:
                                existing["properties"][prop] = node["properties"][prop]
                    node = existing
                elif constraint.get("create") == "never":
                    # Skip creation (controlled vocabulary)
                    logger.info(f"Skipping {node_type} creation (create: never)")
                    continue
        
        constrained_nodes.append(node)
    
    return constrained_nodes
```

---

### Phase 3: Storage with Safety Metadata

#### Step 3.1: Store Nodes with OMO Metadata

```python
async def store_nodes_with_omo_metadata(
    nodes: List[Dict],
    memory_omo: dict,
    storage: Union[PaprDB, MemoryGraph]
):
    """
    Store nodes with OMO safety metadata preserved.
    """
    for node in nodes:
        # Store node with OMO metadata
        stored_node = await storage.create_node(
            node_type=node["type"],
            properties={
                **node["properties"],
                # OMO safety fields
                "omo_id": memory_omo["id"],
                "omo_consent": memory_omo.get("consent", "none"),
                "omo_risk": memory_omo.get("risk", "none"),
                "omo_created_at": memory_omo.get("createdAt"),
                "omo_source_url": memory_omo.get("sourceUrl"),
                # ACL from OMO
                "acl": memory_omo.get("acl", {})
            }
        )
        
        # Store relationship to source memory
        await storage.create_edge(
            source_id=memory_omo["id"],
            target_id=stored_node["id"],
            edge_type="EXTRACTED_FROM",
            properties={
                "extraction_method": "llm",
                "confidence": node.get("confidence", 0.85)
            }
        )
```

---

## 📋 Schema-Level Constraints

### UserGraphSchema Extension

**Update `UserGraphSchema` model to include `node_constraints`:**

```python
# models/user_schemas.py

class UserGraphSchema(BaseModel):
    """Complete user-defined graph schema"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    version: str = Field(default="1.0.0", pattern=r'^\d+\.\d+\.\d+$')
    
    # ... existing fields ...
    
    # Schema-level node constraints (NEW)
    node_constraints: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Node constraints that apply to all memories using this schema. Memory-level constraints override these."
    )
    
    # ... rest of fields ...
```

### Schema-Level Constraint Storage

**Parse Server Schema Update:**

```javascript
// Parse Class: UserGraphSchema
// Add new field:
"node_constraints": Array,  // Array of NodeConstraint objects
```

**SQLite Schema (paprDB):**

```sql
-- Update schemas table to include constraints
ALTER TABLE schemas ADD COLUMN node_constraints JSON;

-- Example schema with constraints
{
  "id": "schema_123",
  "name": "Customer Management",
  "node_types": {...},
  "relationship_types": {...},
  "node_constraints": [
    {
      "node_type": "Customer",
      "force": {"workspace_id": "ws_123"},
      "merge": ["status", "last_contacted"]
    },
    {
      "node_type": "SecurityPolicy",
      "create": "never",
      "search": {"mode": "semantic", "threshold": 0.90}
    }
  ]
}
```

### Constraint Resolution Examples

#### Example 1: Schema-Level Only

```python
# Schema definition
schema = {
    "id": "schema_123",
    "node_constraints": [
        {"node_type": "Project", "force": {"workspace_id": "ws_123"}}
    ]
}

# Memory (no memory-level constraints)
memory_omo = {
    "id": "omo:memory:abc123",
    "content": "Project Alpha is completed",
    "ext": {"papr:schema_id": "schema_123"}
}

# Result: Uses schema constraint
# Project node gets: workspace_id="ws_123"
```

#### Example 2: Schema + Memory (Memory Overrides)

```python
# Schema definition
schema = {
    "id": "schema_123",
    "node_constraints": [
        {"node_type": "Project", "force": {"workspace_id": "ws_123"}}
    ]
}

# Memory (with memory-level override)
memory_omo = {
    "id": "omo:memory:abc123",
    "content": "Project Alpha is completed",
    "ext": {
        "papr:schema_id": "schema_123",
        "papr:node_constraints": [
            {"node_type": "Project", "force": {"workspace_id": "ws_456", "priority": "high"}}
        ]
    }
}

# Result: Memory constraint overrides schema
# Project node gets: workspace_id="ws_456" (memory wins), priority="high" (memory adds)
```

#### Example 3: Schema + Memory (Merge Lists)

```python
# Schema definition
schema = {
    "id": "schema_123",
    "node_constraints": [
        {"node_type": "Project", "merge": ["status", "priority"]}
    ]
}

# Memory (adds to merge list)
memory_omo = {
    "id": "omo:memory:abc123",
    "content": "Project Alpha is completed",
    "ext": {
        "papr:schema_id": "schema_123",
        "papr:node_constraints": [
            {"node_type": "Project", "merge": ["status", "last_updated"]}
        ]
    }
}

# Result: Merged merge list
# Project node merges: ["status", "priority", "last_updated"]
```

---

## 📋 OMO Schema Extensions for Node Constraints

### Updated OMO Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://open-memory.org/OMO-v1.schema.json",
  "title": "Open Memory Object (OMO)",
  "type": "object",
  "required": ["id", "createdAt", "type", "content", "consent"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Global URI or UUID"
    },
    "createdAt": {
      "type": "string",
      "format": "date-time"
    },
    "type": {
      "type": "string",
      "enum": ["text", "image", "audio", "video", "file", "code"],
      "description": "Primary media type of the memory"
    },
    "content": {
      "type": "string",
      "description": "UTF-8 (or base64) body—may be truncated for binary blobs"
    },
    "consent": {
      "type": "string",
      "enum": ["explicit", "implicit", "terms", "none"],
      "description": "How the data owner allowed this memory to be stored/used"
    },
    "risk": {
      "type": "string",
      "enum": ["none", "sensitive", "flagged"],
      "default": "none",
      "description": "Post-ingest safety assessment: suppress or filter on retrieval"
    },
    "topics": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "sourceUrl": {
      "type": "string",
      "format": "uri"
    },
    "acl": {
      "type": "object",
      "properties": {
        "read": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "write": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      },
      "additionalProperties": false
    },
    "ext": {
      "type": "object",
      "description": "Namespaced extension fields; keys SHOULD use a vendor prefix (e.g. 'papr:relationships')",
      "properties": {
        "papr:schema_id": {
          "type": "string",
          "description": "Custom schema ID for node extraction. Schema-level constraints from this schema will be applied to all extracted nodes."
        },
        "papr:node_constraints": {
          "type": "array",
          "description": "Memory-level node constraints to apply when extracting entities. These override schema-level constraints from papr:schema_id. If no schema_id is provided, these are the only constraints applied.",
          "items": {
            "$ref": "#/definitions/NodeConstraint"
          }
        },
        "papr:workspace_id": {
          "type": "string",
          "description": "Workspace ID for multi-tenant isolation"
        },
        "papr:user_id": {
          "type": "string",
          "description": "User ID who created this memory"
        },
        "papr:metadata": {
          "type": "object",
          "description": "Additional papr-specific metadata",
          "additionalProperties": true
        }
      },
      "additionalProperties": true
    }
  },
  "definitions": {
    "NodeConstraint": {
      "type": "object",
      "properties": {
        "node_type": {
          "type": "string",
          "description": "Node type this constraint applies to"
        },
        "when": {
          "type": "object",
          "description": "Conditional: When to apply this constraint",
          "additionalProperties": true
        },
        "create": {
          "type": "string",
          "enum": ["auto", "never"],
          "default": "auto",
          "description": "Creation policy: can new nodes be created?"
        },
        "node_id": {
          "type": "string",
          "description": "Direct node selection: use this exact node ID"
        },
        "search": {
          "type": "object",
          "description": "Search configuration for finding existing nodes",
          "properties": {
            "mode": {
              "type": "string",
              "enum": ["semantic", "exact"],
              "default": "semantic"
            },
            "property": {
              "type": "string",
              "description": "Property to search by (defaults to unique_identifier)"
            },
            "threshold": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "default": 0.85,
              "description": "Minimum similarity score (0.0-1.0)"
            },
            "only_ids": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "description": "Restrict search to specific node IDs"
            }
          }
        },
        "force": {
          "type": "object",
          "description": "Force these values on ALL nodes (new or existing)",
          "additionalProperties": true
        },
        "merge": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Merge these properties from AI extraction into EXISTING nodes only"
        }
      },
      "required": ["node_type"]
    }
  },
  "additionalProperties": false
}
```

---

## 🔄 Migration: Memory → OMO Format

### Current Memory Class Structure

```python
# models/parse_server.py
class Memory(BaseModel):
    id: str
    content: str
    title: Optional[str] = None
    type: str
    metadata: Optional[Union[str, Dict[str, Any]]] = None
    customMetadata: Optional[Dict[str, Any]] = None
    # ... ACL fields, workspace_id, etc.
```

### Migration: Memory → OMO

```python
def memory_to_omo(memory: Memory) -> dict:
    """
    Convert existing Memory object to OMO format.
    """
    # Map Memory fields to OMO
    omo = {
        "$schema": "https://open-memory.org/OMO-v1.schema.json",
        "id": f"omo:memory:{memory.id}",
        "createdAt": memory.created_at.isoformat() if memory.created_at else None,
        "type": memory.type,  # text, image, etc.
        "content": memory.content,
        "consent": "explicit",  # Default - should be stored in Memory
        "risk": "none",  # Default - should be stored in Memory
        "topics": memory.topics or [],
        "sourceUrl": memory.source_url or "",
        "acl": {
            "read": _extract_acl_read(memory),
            "write": _extract_acl_write(memory)
        },
        "ext": {
            "papr:workspace_id": memory.workspace_id,
            "papr:user_id": memory.user_id,
            "papr:metadata": {
                "conversationId": memory.conversation_id,
                "sessionId": getattr(memory, "sessionId", None),
                "source_document_id": memory.source_document_id,
                "source_message_id": memory.source_message_id
            }
        }
    }
    
    # Add node constraints if present in customMetadata
    if memory.customMetadata and "node_constraints" in memory.customMetadata:
        omo["ext"]["papr:node_constraints"] = memory.customMetadata["node_constraints"]
    
    return omo

def omo_to_memory(omo: dict) -> Memory:
    """
    Convert OMO format to Memory object.
    """
    # Extract OMO fields
    memory_id = omo["id"].replace("omo:memory:", "")
    content = omo["content"]
    memory_type = omo["type"]
    
    # Build Memory object
    memory = Memory(
        id=memory_id,
        content=content,
        type=memory_type,
        title=omo.get("title"),
        source_url=omo.get("sourceUrl", ""),
        topics=omo.get("topics", []),
        created_at=datetime.fromisoformat(omo["createdAt"].replace("Z", "+00:00")),
        # ACL mapping
        acl=_convert_omo_acl_to_memory_acl(omo.get("acl", {})),
        user_id=omo.get("ext", {}).get("papr:user_id"),
        workspace_id=omo.get("ext", {}).get("papr:workspace_id"),
        # Custom metadata
        customMetadata={
            "consent": omo.get("consent", "none"),
            "risk": omo.get("risk", "none"),
            "node_constraints": omo.get("ext", {}).get("papr:node_constraints", []),
            **omo.get("ext", {}).get("papr:metadata", {})
        }
    )
    
    return memory
```

---

## 🔄 Migration: MemoryMetadata → OMO Extensions

### Current MemoryMetadata Structure

```python
# models/shared_types.py
class MemoryMetadata(BaseModel):
    hierarchical_structures: Optional[str] = None
    createdAt: Optional[str] = None
    location: Optional[str] = None
    topics: Optional[List[str]] = None
    conversationId: Optional[str] = None
    sourceUrl: Optional[str] = None
    workspace_id: Optional[str] = None
    # ... ACL fields, user_id, etc.
```

### Migration: MemoryMetadata → OMO Extensions

```python
def memory_metadata_to_omo_ext(metadata: MemoryMetadata) -> dict:
    """
    Convert MemoryMetadata to OMO ext field.
    """
    return {
        "papr:workspace_id": metadata.workspace_id,
        "papr:user_id": metadata.user_id,
        "papr:metadata": {
            "hierarchical_structures": metadata.hierarchical_structures,
            "location": metadata.location,
            "conversationId": metadata.conversationId,
            "sessionId": metadata.sessionId,
            "relatedGoals": metadata.relatedGoals or [],
            "relatedUseCases": metadata.relatedUseCases or [],
            "relatedSteps": metadata.relatedSteps or []
        }
    }

def omo_ext_to_memory_metadata(omo: dict) -> MemoryMetadata:
    """
    Convert OMO ext field to MemoryMetadata.
    """
    ext = omo.get("ext", {})
    papr_metadata = ext.get("papr:metadata", {})
    
    return MemoryMetadata(
        hierarchical_structures=papr_metadata.get("hierarchical_structures"),
        createdAt=omo.get("createdAt"),
        location=papr_metadata.get("location"),
        topics=omo.get("topics", []),
        conversationId=papr_metadata.get("conversationId"),
        sourceUrl=omo.get("sourceUrl"),
        workspace_id=ext.get("papr:workspace_id"),
        user_id=ext.get("papr:user_id"),
        sessionId=papr_metadata.get("sessionId"),
        relatedGoals=papr_metadata.get("relatedGoals", []),
        relatedUseCases=papr_metadata.get("relatedUseCases", []),
        relatedSteps=papr_metadata.get("relatedSteps", [])
    )
```

---

## 🛡️ Safety Standards Implementation

### Safety Standard 1: Consent Enforcement

```python
def enforce_consent_standard(memory_omo: dict, extracted_nodes: List[Dict]) -> List[Dict]:
    """
    Enforce consent standard: only extract/store nodes if consent is granted.
    """
    consent = memory_omo.get("consent", "none")
    
    if consent == "none":
        logger.warning(f"Memory {memory_omo['id']} has no consent - skipping extraction")
        return []
    
    # Annotate all nodes with consent level
    for node in extracted_nodes:
        node["properties"]["consent"] = consent
        node["properties"]["consent_source"] = memory_omo["id"]
    
    return extracted_nodes
```

### Safety Standard 2: Risk Assessment

```python
def enforce_risk_standard(memory_omo: dict, extracted_nodes: List[Dict]) -> List[Dict]:
    """
    Enforce risk standard: apply stricter constraints for high-risk content.
    """
    risk = memory_omo.get("risk", "none")
    
    if risk == "flagged":
        # High-risk: apply strict constraints
        for node in extracted_nodes:
            # Force additional safety properties
            node["properties"]["risk_level"] = "flagged"
            node["properties"]["requires_review"] = True
            
            # Restrict ACL
            if "acl" not in node:
                node["acl"] = {}
            node["acl"]["read"] = node["acl"].get("read", [])
            # Only allow original user to read flagged content
            node["acl"]["read"] = [memory_omo.get("ext", {}).get("papr:user_id")]
    
    elif risk == "sensitive":
        # Medium-risk: apply moderate constraints
        for node in extracted_nodes:
            node["properties"]["risk_level"] = "sensitive"
            node["properties"]["requires_review"] = False
    
    return extracted_nodes
```

### Safety Standard 3: ACL Propagation

```python
def enforce_acl_propagation(memory_omo: dict, extracted_nodes: List[Dict]) -> List[Dict]:
    """
    Propagate ACL from memory to extracted nodes.
    """
    memory_acl = memory_omo.get("acl", {})
    
    for node in extracted_nodes:
        # Inherit ACL from source memory
        node["acl"] = {
            "read": memory_acl.get("read", []),
            "write": memory_acl.get("write", [])
        }
    
    return extracted_nodes
```

### Safety Standard 4: Audit Trail

```python
def create_audit_trail(memory_omo: dict, extracted_nodes: List[Dict]) -> List[Dict]:
    """
    Create audit trail for compliance tracking.
    """
    for node in extracted_nodes:
        node["properties"]["audit_trail"] = {
            "source_memory_id": memory_omo["id"],
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "consent": memory_omo.get("consent", "none"),
            "risk": memory_omo.get("risk", "none"),
            "extraction_method": "llm",
            "constraints_applied": True
        }
    
    return extracted_nodes
```

---

## 🔄 Complete Integration Flow

### End-to-End: OMO → Constrained Nodes → Storage

```python
async def process_memory_with_omo_and_constraints(
    memory_omo: dict,
    storage: Union[PaprDB, MemoryGraph],
    schema_service: SchemaService
) -> Dict[str, Any]:
    """
    Complete flow: OMO memory → node extraction → constraints → storage.
    Supports both schema-level and memory-level constraints.
    """
    # 1. Validate OMO schema
    if not validate_omo(memory_omo):
        raise ValueError("Invalid OMO format")
    
    # 2. Resolve constraints (schema-level + memory-level)
    resolved_constraints = await resolve_constraints(memory_omo, schema_service)
    logger.info(f"Resolved {len(resolved_constraints)} constraints (schema + memory merged)")
    
    # 3. Extract nodes from content
    extracted_nodes = await extract_nodes_from_omo(memory_omo)
    
    # 4. Apply safety standards
    extracted_nodes = enforce_consent_standard(memory_omo, extracted_nodes)
    extracted_nodes = enforce_risk_standard(memory_omo, extracted_nodes)
    extracted_nodes = enforce_acl_propagation(memory_omo, extracted_nodes)
    extracted_nodes = create_audit_trail(memory_omo, extracted_nodes)
    
    # 5. Apply resolved constraints (schema + memory merged)
    constrained_nodes = await apply_constraints_to_nodes(
        nodes=extracted_nodes,
        node_constraints=resolved_constraints,
        omo_metadata={
            "consent": memory_omo.get("consent", "none"),
            "risk": memory_omo.get("risk", "none"),
            "source_memory_id": memory_omo["id"],
            "created_at": memory_omo.get("createdAt"),
            "acl": memory_omo.get("acl", {})
        }
    )
    
    # 6. Store nodes with OMO metadata
    stored_nodes = []
    for node in constrained_nodes:
        stored_node = await store_nodes_with_omo_metadata(
            nodes=[node],
            memory_omo=memory_omo,
            storage=storage
        )
        stored_nodes.append(stored_node)
    
    return {
        "memory_id": memory_omo["id"],
        "schema_id": memory_omo.get("ext", {}).get("papr:schema_id"),
        "nodes_created": len(stored_nodes),
        "nodes": stored_nodes,
        "constraints_applied": {
            "schema_level": len([c for c in resolved_constraints if c.get("_source") == "schema"]),
            "memory_level": len([c for c in resolved_constraints if c.get("_source") == "memory"])
        }
    }
```

---

## 📊 Example: Complete OMO + Constraints Workflow

### Input: OMO Memory with Constraints

```json
{
  "$schema": "https://open-memory.org/OMO-v1.schema.json",
  "id": "omo:memory:abc123",
  "createdAt": "2024-01-15T10:30:00Z",
  "type": "text",
  "content": "Project Alpha is now completed. Team backend worked on it. John Smith was the lead developer.",
  "consent": "explicit",
  "risk": "none",
  "topics": ["project-management"],
  "sourceUrl": "https://example.com/meeting-notes",
  "acl": {
    "read": ["user:123", "workspace:ws_123"],
    "write": ["user:123"]
  },
  "ext": {
    "papr:node_constraints": [
      {
        "node_type": "Project",
        "force": {"workspace_id": "ws_123", "team": "backend"},
        "merge": ["status"],
        "search": {"mode": "semantic", "threshold": 0.85}
      },
      {
        "node_type": "Person",
        "create": "never",
        "search": {"mode": "semantic", "threshold": 0.90}
      }
    ],
    "papr:workspace_id": "ws_123",
    "papr:user_id": "user_123"
  }
}
```

### Output: Constrained Nodes with Safety Metadata

```json
{
  "memory_id": "omo:memory:abc123",
  "nodes_created": 2,
  "nodes": [
    {
      "id": "project_alpha_123",
      "type": "Project",
      "properties": {
        "name": "Alpha",
        "status": "completed",
        "workspace_id": "ws_123",
        "team": "backend",
        "consent": "explicit",
        "risk": "none",
        "source_memory_id": "omo:memory:abc123",
        "audit_trail": {
          "source_memory_id": "omo:memory:abc123",
          "extracted_at": "2024-01-15T10:30:05Z",
          "consent": "explicit",
          "risk": "none",
          "extraction_method": "llm",
          "constraints_applied": true
        }
      },
      "acl": {
        "read": ["user:123", "workspace:ws_123"],
        "write": ["user:123"]
      }
    },
    {
      "id": "person_john_smith_456",
      "type": "Person",
      "properties": {
        "name": "John Smith",
        "role": "lead developer",
        "consent": "explicit",
        "risk": "none",
        "source_memory_id": "omo:memory:abc123"
      },
      "acl": {
        "read": ["user:123", "workspace:ws_123"],
        "write": ["user:123"]
      }
    }
  ]
}
```

---

## 🎯 Use Cases

### Use Case 1: Safety-Compliant Memory Storage

**Scenario**: User wants to store a memory with explicit consent and apply workspace-level constraints.

**Solution**:
```python
# Create OMO memory with consent
memory_omo = {
    "id": "omo:memory:xyz789",
    "type": "text",
    "content": "Customer feedback: Product X needs improvement",
    "consent": "explicit",
    "risk": "sensitive",
    "ext": {
        "papr:node_constraints": [
            {
                "node_type": "Customer",
                "force": {"workspace_id": "ws_123"},
                "create": "never"  # Only link to existing customers
            }
        ]
    }
}

# Process with safety standards
result = await process_memory_with_omo_and_constraints(memory_omo, storage)
# Result: Nodes created with consent="explicit", risk="sensitive", constraints applied
```

### Use Case 2: High-Risk Content Filtering

**Scenario**: Memory contains flagged content - need strict access control.

**Solution**:
```python
memory_omo = {
    "id": "omo:memory:flagged001",
    "type": "text",
    "content": "Confidential: Project Beta details",
    "consent": "explicit",
    "risk": "flagged",  # High risk
    "acl": {
        "read": ["user:admin_123"],  # Restricted access
        "write": ["user:admin_123"]
    }
}

# Process with risk standard enforcement
result = await process_memory_with_omo_and_constraints(memory_omo, storage)
# Result: Nodes created with risk="flagged", restricted ACL, requires_review=True
```

### Use Case 3: Schema-Level Controlled Vocabulary

**Scenario**: Define at schema level that SecurityPolicy nodes can never be created (applies to all memories using this schema).

**Solution**:
```python
# Schema definition (stored in UserGraphSchema)
schema = {
    "id": "schema_security",
    "name": "Security Management Schema",
    "node_constraints": [
        {
            "node_type": "SecurityPolicy",
            "create": "never",  # Schema-level: applies to ALL memories
            "search": {"mode": "semantic", "threshold": 0.90}
        }
    ]
}

# Memory (uses schema, no memory-level override needed)
memory_omo = {
    "id": "omo:memory:security001",
    "type": "text",
    "content": "Applied security policy: GDPR compliance",
    "consent": "explicit",
    "risk": "none",
    "ext": {
        "papr:schema_id": "schema_security"  # Uses schema constraints
    }
}

# Process with schema-level constraint enforcement
result = await process_memory_with_omo_and_constraints(memory_omo, storage, schema_service)
# Result: Only links to existing SecurityPolicy nodes (from schema constraint)
```

### Use Case 4: Memory-Level Override

**Scenario**: Schema says "all Projects get workspace_id", but this specific memory needs a different workspace_id.

**Solution**:
```python
# Schema definition
schema = {
    "id": "schema_projects",
    "node_constraints": [
        {"node_type": "Project", "force": {"workspace_id": "ws_123"}}
    ]
}

# Memory (overrides schema constraint)
memory_omo = {
    "id": "omo:memory:project001",
    "type": "text",
    "content": "Project Alpha is completed",
    "ext": {
        "papr:schema_id": "schema_projects",
        "papr:node_constraints": [
            {"node_type": "Project", "force": {"workspace_id": "ws_456"}}  # Override
        ]
    }
}

# Result: Project gets workspace_id="ws_456" (memory-level wins)
```

---

## 🔧 Implementation Checklist

### Phase 1: OMO Schema Support
- [ ] Add OMO schema validation
- [ ] Add `consent` and `risk` fields to Memory class
- [ ] Add `ext.papr:schema_id` parsing
- [ ] Add `ext.papr:node_constraints` parsing (memory-level)
- [ ] Create `memory_to_omo()` and `omo_to_memory()` converters

### Phase 2: Schema-Level Constraints
- [ ] Add `node_constraints` field to `UserGraphSchema` model
- [ ] Update Parse Server schema to include `node_constraints` field
- [ ] Update SQLite schemas table to store `node_constraints` JSON
- [ ] Create API endpoints to get/update schema constraints
- [ ] Implement schema constraint validation

### Phase 3: Constraint Resolution
- [ ] Implement `resolve_constraints()` (schema + memory merge)
- [ ] Implement `merge_constraints()` (memory overrides schema)
- [ ] Add constraint source tracking (`_source: "schema"` or `"memory"`)
- [ ] Test constraint precedence (memory > schema > default)

### Phase 4: Safety Standards
- [ ] Implement `enforce_consent_standard()`
- [ ] Implement `enforce_risk_standard()`
- [ ] Implement `enforce_acl_propagation()`
- [ ] Implement `create_audit_trail()`

### Phase 5: Node Constraint Integration
- [ ] Integrate schema-level constraints from schema_id
- [ ] Integrate memory-level constraints from OMO ext field
- [ ] Apply merged constraints during node extraction
- [ ] Store OMO metadata on extracted nodes
- [ ] Create audit trail for compliance

### Phase 6: Migration
- [ ] Migrate existing Memory objects to OMO format
- [ ] Migrate MemoryMetadata to OMO ext field
- [ ] Migrate existing schemas to include node_constraints field
- [ ] Update API endpoints to accept OMO format
- [ ] Update storage layer to preserve OMO metadata

---

## 📚 Related Documentation

- [Node Constraints API Reference](./NODE_CONSTRAINTS_API_REFERENCE.md) - Complete constraint API
- [paprDB Node Constraints Architecture](./PAPRDB_NODE_CONSTRAINTS_ARCHITECTURE.md) - Architecture details
- [OMO Specification](https://open-memory.org/) - Official OMO standard
- [Memory Migration Guide](./MEMORY_MIGRATION_GUIDE.md) - How to migrate existing memories

---

## 🎯 Summary

**OMO + Node Constraints Integration** provides:

1. ✅ **Safety Standards**: Consent, risk assessment, ACL enforcement
2. ✅ **Data Governance**: Node constraints control entity extraction
3. ✅ **Schema-Level Constraints**: Define constraints once at schema level, apply to all memories
4. ✅ **Memory-Level Constraints**: Override schema constraints for specific memories
5. ✅ **Policy Enforcement**: Workspace-level policies via constraints
6. ✅ **Compliance**: Audit trails and consent tracking
7. ✅ **Standards Compliance**: OMO format for interoperability

### Constraint Resolution Summary

**Hierarchy (Precedence):**
```
Memory-Level Constraints (ext.papr:node_constraints)
    ↓ (overrides)
Schema-Level Constraints (UserGraphSchema.node_constraints)
    ↓ (fallback)
System Defaults
```

**When to Use Each:**

| Level | Use Case | Example |
|-------|----------|---------|
| **Schema-Level** | Apply to ALL memories using this schema | "All Customer nodes must have workspace_id" |
| **Memory-Level** | Override for specific memory | "This memory: Customer gets priority='high'" |
| **Both** | Base rules in schema, overrides in memory | Schema: `force: {workspace_id}`, Memory: `force: {workspace_id, priority}` |

**Key Benefits**:
- **DRY Principle**: Define constraints once at schema level, reuse across memories
- **Flexibility**: Override schema constraints per-memory when needed
- **Unified Safety**: OMO safety standards (consent, risk) + constraint governance
- **Standards-Compliant**: OMO format for interoperability
- **Production-Ready**: Complete compliance and audit trail support

**Result**: Safe, governed, compliant memory storage with schema-level and memory-level policy enforcement! 🎯

