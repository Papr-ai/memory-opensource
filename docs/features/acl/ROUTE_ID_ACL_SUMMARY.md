# Route ID and ACL Field Usage Summary

This document summarizes how IDs and ACL fields are used across all memory-related routes, including storage and search patterns in Parse Server, Neo4j, and Qdrant.

## Table of Contents
- [1. Add Memory (POST /v1/memory)](#1-add-memory-post-v1memory)
- [2. Search Memory (POST /v1/memory/search)](#2-search-memory-post-v1memorysearch)
- [3. Update Memory (PUT /v1/memory/{memory_id})](#3-update-memory-put-v1memorymemory_id)
- [4. Get Memory (GET /v1/memory/{memory_id})](#4-get-memory-get-v1memorymemory_id)
- [5. Batch Add Memory (POST /v1/memory/batch)](#5-batch-add-memory-post-v1memorybatch)
- [6. Upload Document (POST /v1/documents)](#6-upload-document-post-v1documents)
- [7. Messages Route (POST /v1/messages)](#7-messages-route-post-v1messages)
- [Reference: Parse Server Storage](#reference-parse-server-storage)
- [Reference: Central ACL Processing Patterns](#reference-central-acl-processing-patterns)

---

## 1. Add Memory (POST /v1/memory)

**Route**: `POST /v1/memory`  
**Handler**: `add_memory_v1`  
**Location**: `routers/v1/memory_routes_v1.py:227`

### a. Request Structure

**Top-Level Fields in API Request**:
- `content` (required): str - Memory content
- `type` (optional): MemoryType - Defaults to "text"
- `metadata` (optional): MemoryMetadata - See metadata fields below
- `organization_id` (optional): str - **Can be BOTH top-level AND in metadata** (top-level takes precedence)
- `namespace_id` (optional): str - **Can be BOTH top-level AND in metadata** (top-level takes precedence)
- `context` (optional): List[ContextItem]
- `relationships_json` (optional): List[RelationshipItem] - **Top-level only, NOT in metadata**
- `graph_generation` (optional): GraphGeneration - **Inherited from SchemaSpecificationMixin**

**Fields in `metadata` Object (`MemoryMetadata`)**:
- **IDs**: `user_id`, `workspace_id`, `organization_id`, `namespace_id`, `external_user_id`
- **ACL Fields** (all arrays):
  - `user_read_access`, `user_write_access`
  - `workspace_read_access`, `workspace_write_access`
  - `role_read_access`, `role_write_access`
  - `namespace_read_access`, `namespace_write_access`
  - `organization_read_access`, `organization_write_access`
  - `external_user_read_access`, `external_user_write_access`
- **Content Metadata**: `role`, `category`, `topics`, `hierarchical_structures`, `location`, `emoji_tags`, `emotion_tags`, `conversationId`, `sourceUrl`, `sourceType`, `pageId`
- **Timestamps**: `createdAt` (ISO format)
- **Classification**: `relatedGoals`, `relatedUseCases`, `relatedSteps`, classification scores
- **Custom**: `customMetadata` (Dict[str, Any])

**Response Model (`AddMemoryResponse`)**:
- `code`: int
- `status`: str
- `data`: List[AddMemoryItem]
  - `memoryId`: str
  - `createdAt`: datetime
  - `objectId`: str
  - `memoryChunkIds`: List[str]

### b. IDs Used

**Important**: This table shows WHERE the ID itself is stored, NOT the ACL arrays. The relationship is:
- **ID Storage**: `organization_id` is stored as a Pointer + deprecated string field
- **ACL Arrays**: `organization_read_access` is a SEPARATE field containing IDs (see ACL table below)
- Example: `user_id` is stored as a Pointer. The pointer's `objectId` is THEN extracted and added to the `user_read_access` ACL array

| Field | Can Pass | Location in Request | Default Behavior | Used | Parse Server | Neo4j | Qdrant Qwen | Qdrant Property | Notes |
|-------|----------|---------------------|------------------|------|--------------|-------|-------------|-----------------|-------|
| `user_id` | ✅ | In `metadata` object | **Auto-filled** from `auth_response.end_user_id` if not provided | ✅ | **Pointer** (`user`) | String property | String in payload | String in payload | Parse: Pointer to _User class. The objectId is extracted and used in ACL arrays |
| `workspace_id` | ✅ | In `metadata` object | **Auto-filled** from `auth_response.workspace_id` if not provided | ✅ | **Pointer** (`workspace`) | String property | String in payload | String in payload | Parse: Pointer to WorkSpace class. The objectId is extracted and used in ACL arrays |
| `organization_id` | ⚠️ | **Top-level** OR in `metadata` | **Auth OVERRIDES** - `auth_context.organization_id` always wins if present | ✅ | **Pointer** (`organization`) + deprecated string field | String property | String in payload | String in payload | Parse: Pointer to Organization + deprecated string field for backward compatibility |
| `namespace_id` | ⚠️ | **Top-level** OR in `metadata` | **Auth OVERRIDES** - `auth_context.namespace_id` always wins if present | ✅ | **Pointer** (`namespace`) + deprecated string field | String property | String in payload | String in payload | Parse: Pointer to Namespace + deprecated string field for backward compatibility |
| `external_user_id` | ✅ | In `metadata` object | Optional - for external user tracking | ✅ | String inside `developerUser.external_id` | ❌ NOT stored | String in payload | ❌ NOT stored | Parse: Stored as property in developerUser pointer object |
| `developer_user_id` | ❌ | N/A (internal only) | Always from `auth_response.developer_id` | ✅ | **Pointer** (`developerUser`) | ❌ NOT stored | ❌ NOT stored | ❌ NOT stored | Parse: Pointer to DeveloperUser class |

### c. ACL Fields Used

| Field | Can Pass | Location in Request | Default Behavior | Used | Set (Parse) | Set (Neo4j) | Set (Qdrant Qwen) | Set (Qdrant Property) | Parse ACL Conversion | Notes |
|-------|----------|---------------------|------------------|------|-------------|-------------|-------------------|----------------------|---------------------|-------|
| `user_read_access` | ✅ | In `metadata` object | Default: `[user_id]` | ✅ | ✅ Field + ACL | ✅ | ✅ (payload) | ✅ (payload) | ✅ `{userID: {read: true}}` | Stored as both field AND in ACL object |
| `user_write_access` | ✅ | In `metadata` object | Default: `[user_id]` | ✅ | ✅ Field + ACL | ✅ | ✅ (payload) | ✅ (payload) | ✅ `{userID: {write: true}}` | Stored as both field AND in ACL object |
| `workspace_read_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Stored ONLY as field (not in ACL object) |
| `workspace_write_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Stored ONLY as field (not in ACL object) |
| `role_read_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field + ACL | ✅ | ✅ (payload) | ✅ (payload) | ✅ `{role:ID: {read: true}}` | Stored as both field AND in ACL with `role:` prefix |
| `role_write_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field + ACL | ✅ | ✅ (payload) | ✅ (payload) | ✅ `{role:ID: {write: true}}` | Stored as both field AND in ACL with `role:` prefix |
| `namespace_read_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Field only, USED in Property search filters |
| `namespace_write_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Field only |
| `organization_read_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Field only, USED in Property search filters |
| `organization_write_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Field only |
| `external_user_read_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Field only (not in ACL object) |
| `external_user_write_access` | ✅ | In `metadata` object | Default: `[]` | ✅ | ✅ Field only | ✅ | ✅ (payload) | ✅ (payload) | ❌ Not in ACL | Field only (not in ACL object) |

### d. Parse Server Storage

See [Reference: Parse Server Storage](#reference-parse-server-storage) for detailed explanation.

**Key Points**:
- IDs stored as **Pointers** + extracted `objectId` in ACL arrays
- ACL arrays stored as **separate fields** (NOT in Parse ACL object for workspace/namespace/organization/external_user)
- `user_read_access` and `role_read_access` stored as **both field AND in Parse ACL object**

### e. Qdrant Qwen Storage Summary

**Qdrant Qwen Collection**: Stores memory embeddings with metadata for semantic search

| Field Type | Location | Indexed? | Notes |
|------------|----------|----------|-------|
| **IDs (Top-level)** | `user_id`, `workspace_id`, `organization_id`, `namespace_id` | ✅ | Fast filtering |
| **IDs (Metadata)** | `metadata.user_id`, `metadata.external_user_id` | ✅ | Nested for compatibility |
| **ACL Read (Top-level)** | `user_read_access`, `workspace_read_access`, `role_read_access`, `organization_read_access`, `namespace_read_access` | ✅ | Fast ACL filtering |
| **ACL Write (Top-level)** | `user_write_access`, `workspace_write_access`, `role_write_access`, `organization_write_access`, `namespace_write_access` | ✅ | Fast ACL filtering |
| **ACL Read (Metadata)** | `metadata.user_read_access`, `metadata.workspace_read_access`, `metadata.role_read_access`, `metadata.organization_read_access`, `metadata.namespace_read_access` | ✅ | Backward compatibility |
| **ACL Write (Metadata)** | `metadata.user_write_access`, `metadata.workspace_write_access`, `metadata.role_write_access`, `metadata.organization_write_access`, `metadata.namespace_write_access` | ✅ | Backward compatibility |
| **ACL External User (Metadata)** | `metadata.external_user_read_access`, `metadata.external_user_write_access` | ✅ | Nested only |
| **Content Metadata** | `topics`, `hierarchical_structures`, `location`, `emoji_tags`, `emotion_tags`, `conversationId`, `chunk_id` | ✅ | Indexed for filtering |
| **Classification** | `metadata.relatedGoals`, `metadata.useCaseClassificationScores`, `metadata.stepClassificationScores` | ✅ | Nested metadata |
| **Timestamps** | `metadata.createdAt` | ✅ | Indexed |
| **Custom** | All fields from `customMetadata` | ✅ | Flattened and added to payload |
| **Chunk Info** | `chunk_index`, `total_chunks`, `content` | ✅ | Per-chunk data |
| **Vector** | Embedding vector | ✅ | For similarity search |

**Why both top-level AND nested?**
- **Top-level**: Faster filtering for common queries (direct field access)
- **Nested in metadata**: Backward compatibility and detailed metadata access

**Code Location**: `memory/memory_graph.py:1034-1051` (index fields), `memory/memory_graph.py:2120-2145` (payload creation)

### f. Qdrant Qwen Search for Grouped Memories

**When**: During all memory additions (for finding similar/related memories to group)  
**Collection**: Qwen Collection (main memory storage)  
**Purpose**: Find semantically similar memories to link/group with new memory  
**Method**: `find_related_memory_items_async()` - `memory/memory_graph.py:5087-6254`

**Fields Used in Search Filters**:
| ID Field (from request) | ACL Array Field (checked in Qdrant) | Used In Filter | Filter Logic | Purpose |
|-------------------------|--------------------------------------|----------------|--------------|---------|
| `user_id` | `user_id` (direct match) | ✅ SHOULD (OR) | `user_id = "<value>"` | Check if user owns the memory |
| `user_id` | `user_read_access` | ✅ SHOULD (OR) | Check if `user_id` is IN `user_read_access` array | User-level access check |
| `workspace_id` | `workspace_read_access` | ✅ SHOULD (OR) | Check if `workspace_id` is IN `workspace_read_access` array (if ≤10 workspaces) | Workspace-level access check |
| `user_roles` | `role_read_access` | ✅ SHOULD (OR) | Check if ANY role is IN `role_read_access` array (if ≤10 roles) | Role-based access check |
| `organization_id` | `organization_read_access` | ✅ SHOULD (OR) | Check if `organization_id` is IN `organization_read_access` array | Organization-level access check |
| `namespace_id` | `namespace_read_access` | ✅ SHOULD (OR) | Check if `namespace_id` is IN `namespace_read_access` array | Namespace-level access check |
| `external_user_id` | `external_user_id` (direct match) | ✅ SHOULD (OR) | `external_user_id = "<value>"` | External user filter |
| `organization_id` | N/A (scoping) | ✅ MUST (AND) - Soft | Organization scoping (includes legacy without org_id) | Tenant isolation (soft) |
| `namespace_id` | N/A (scoping) | ✅ MUST (AND) - Soft | Namespace scoping (includes legacy without ns_id) | Tenant isolation (soft) |

**Filter Type**: 
- **SHOULD (OR)**: User has access if ANY condition matches
- **MUST (AND) - Soft**: Scoping filter that includes legacy memories without these IDs
- **Speed Optimization**: Only checks workspace/role if ≤10 items

**Returns**: List of similar memories for grouping/linking

**Code Location**: `memory/memory_graph.py:5155-5230`

### g. Qdrant Property Search for Graph Generation

**When**: During memory addition with `graph_generation` enabled  
**Collection**: Property Collection (`neo4j_properties`)  
**Purpose**: Detect duplicate entities before creating in Neo4j

#### Search 1: Unique Identifier Search

**Purpose**: Find exact or near-exact matches for unique properties (e.g., email, username)

**Fields Used in Search Filters**:
| ID Field (from request) | ACL Array Field (checked in Qdrant) | Used In Filter | Filter Logic | Purpose |
|-------------------------|--------------------------------------|----------------|--------------|---------|
| `namespace_id` | N/A (direct match) | ✅ MUST (AND) | `namespace_id = "<value>"` | **REQUIRED** - Exact match for tenant isolation |
| `user_id` | N/A (direct match) | ✅ MUST (AND) | `user_id = "<value>"` | **REQUIRED** - Per-user isolation (consistent with Neo4j MERGE) |
| `namespace_id` | `namespace_read_access` | ✅ SHOULD (OR) | Check if `namespace_id` is IN `namespace_read_access` array | Namespace-level access check |
| `user_id` | `user_read_access` | ✅ SHOULD (OR) | Check if `user_id` is IN `user_read_access` array | User-level access check |
| `workspace_id` | `workspace_read_access` | ✅ SHOULD (OR) | Check if `workspace_id` is IN `workspace_read_access` array | Workspace-level access check |
| `organization_id` | `organization_read_access` | ✅ SHOULD (OR) | Check if `organization_id` is IN `organization_read_access` array | Organization-level access check |
| `role_read_access` | `role_read_access` | ✅ SHOULD (OR) | Check if ANY role from request is IN `role_read_access` array | Role-based access check |

**Additional MUST Condition**: `property_key = "NodeType.uniqueIdentifierName"` (e.g., `"Person.email"`)

**Threshold**: 0.95 (very high - must be near-exact match)

**Code Location**: `memory/memory_graph.py:9257-9304`

#### Search 2: Content-Based Search

**Purpose**: Find semantically similar content to detect duplicate entities

**Fields Used in Search Filters**: Same as Unique Identifier Search (see table above)

**Additional MUST Conditions**: 
- `property_key IN ["NodeType.name", "NodeType.title", "NodeType.description", "NodeType.content"]`
- `namespace_id = "<namespace_id>"` (REQUIRED for tenant isolation)
- `user_id = "<user_id>"` (REQUIRED for per-user isolation)

**Threshold**: 0.90 (high similarity required)

**Code Location**: `memory/memory_graph.py:9752-9799`

#### Search 3: Find All Properties for Entity

**Purpose**: Retrieve all properties associated with a canonical entity

**Fields Used in Search Filters**: Same as Unique Identifier Search (see table above)

**Additional MUST Conditions**: 
- `source_node_id = "<canonical_node_id>"` (The Neo4j node ID)
- `namespace_id = "<namespace_id>"` (REQUIRED for tenant isolation)
- `user_id = "<user_id>"` (REQUIRED for per-user isolation)

**Method**: Scroll (paginated retrieval, not similarity search)

**Code Location**: `services/property_indexing_service.py:1049-1084`

### h. Neo4j Storage Summary

**Memory Node Properties** (stored via `memory_item_to_node` → `_create_node`):

| Field Type | Fields | Stored? | Notes |
|------------|--------|---------|-------|
| **Core** | `id`, `content`, `type`, `memoryChunkIds` | ✅ | Always stored |
| **IDs** | `user_id`, `workspace_id`, `organization_id`, `namespace_id` | ✅ | String properties |
| **IDs** | `external_user_id` | ❌ | **NOT stored** (inconsistency) |
| **ACL - User** | `user_read_access`, `user_write_access` | ✅ | Array properties |
| **ACL - Workspace** | `workspace_read_access`, `workspace_write_access` | ✅ | Array properties |
| **ACL - Role** | `role_read_access`, `role_write_access` | ✅ | Array properties |
| **ACL - Namespace** | `namespace_read_access`, `namespace_write_access` | ✅ | Array properties |
| **ACL - Organization** | `organization_read_access`, `organization_write_access` | ✅ | Array properties |
| **ACL - External User** | `external_user_read_access`, `external_user_write_access` | ✅ | Array properties |
| **Metadata** | `topics`, `emoji_tags`, `emotion_tags`, `hierarchical_structures`, etc. | ✅ | Various properties |
| **Role/Category** | `memory_role`, `memory_category` | ✅ | Stored with `memory_` prefix |
| **Timestamps** | `createdAt`, `updatedAt` | ✅ | DateTime properties |
| **Custom** | All fields from `customMetadata` | ✅ | Merged into node properties |

**⚠️ Inconsistency**: `external_user_id` is **NOT** stored in Neo4j but its ACL arrays ARE stored.

**Code Location**: `memory/memory_graph.py:4889-4915` (metadata_fields list defines what gets stored), `models/memory_models.py:59-129` (`memory_item_to_node`), `memory/memory_graph.py:10252-10290` (`_create_node`)

### i. Qdrant Property Storage Summary

**Qdrant Property Collection**: Stores property values from Neo4j nodes for fast property-based search

| Field Type | Location | Indexed? | Notes |
|------------|----------|----------|-------|
| **IDs** | `user_id`, `workspace_id`, `organization_id`, `namespace_id` | ✅ | String in payload - copied from source memory |
| **ACL - User** | `user_read_access`, `user_write_access` | ✅ | Inherited from source memory that created the Neo4j node |
| **ACL - Workspace** | `workspace_read_access`, `workspace_write_access` | ✅ | Inherited from source memory that created the Neo4j node |
| **ACL - Role** | `role_read_access`, `role_write_access` | ✅ | Inherited from source memory that created the Neo4j node |
| **ACL - Organization** | `organization_read_access`, `organization_write_access` | ✅ | Inherited from source memory that created the Neo4j node |
| **ACL - Namespace** | `namespace_read_access`, `namespace_write_access` | ✅ | Inherited from source memory that created the Neo4j node |
| **ACL - External User** | `external_user_read_access`, `external_user_write_access` | ✅ | Inherited from source memory that created the Neo4j node |
| **Property-Specific** | `property_name`, `property_value`, `node_type`, `node_id`, `property_key` | ✅ | Format: `NodeType.property_name` |
| **Sync Metadata** | `was_created`, `sync_operation`, `canonical_node_id`, `source_node_id` | ✅ | Property sync tracking |
| **Vector** | Embedding of property value | ✅ | For similarity search |

**What "Inherited from source memory" means**:
- When a memory creates a Neo4j node with properties, those properties are indexed in the Property Collection
- Each property inherits **ALL** ACL fields from the original memory that created the node
- Example: If Memory A (with `user_read_access=["user_123"]`) creates a Person node with `name="John"`, the property `Person.name="John"` will also have `user_read_access=["user_123"]`
- This ensures property-based searches respect the same access controls as the original memory

**Code Location**: `services/property_indexing_service.py:585-612` (property memory creation - lines 594-605 copy all ACL arrays)

### j. Neo4j Node Merge: ACL and ID Updates

**When**: During graph generation when duplicate entities are detected (deduplication)  
**Process**: Neo4j uses `MERGE` to either create a new node or match an existing one based on unique identifiers

#### What Happens During MERGE?

**1. MERGE Matching Criteria** (`_merge_node_by_unique_props` - `memory/memory_graph.py:9420-9575`):

⚠️ **IMPORTANT**: MERGE uses **AND logic** - ALL fields must match EXACTLY for a node to be matched.

The `MERGE` clause matches nodes based on:
- **Unique identifiers** (e.g., `email` for Person, `url` for Website)
- **ALL IDs**: `user_id`, `workspace_id`, `organization_id`, `namespace_id` (for multi-tenant isolation)
- **ALL ACL arrays**: All 12 ACL fields (user/workspace/role/namespace/organization/external_user read/write access)

```cypher
MERGE (n:Person {
  email: "john@example.com",      // Unique identifier
  user_id: "user_123",             // ALL must match (AND logic)
  workspace_id: "ws_456",          // ALL must match (AND logic)
  organization_id: "org_789",      // ALL must match (AND logic)
  namespace_id: "ns_001",          // ALL must match (AND logic)
  user_read_access: ["user_123"], // ALL must match (AND logic)
  user_write_access: ["user_123"], // ALL must match (AND logic)
  workspace_read_access: ["ws_456"], // ALL must match (AND logic)
  workspace_write_access: ["ws_456"],
  role_read_access: [],
  role_write_access: [],
  namespace_read_access: ["ns_001"],
  namespace_write_access: ["ns_001"],
  organization_read_access: ["org_789"],
  organization_write_access: ["org_789"],
  external_user_read_access: [],
  external_user_write_access: []
})
ON CREATE SET n += $all_properties, n.was_created_flag = true
ON MATCH SET n += $update_properties, n.was_created_flag = false
```

**Match Logic**: A node is matched **ONLY IF** all properties in the MERGE clause are identical. If ANY field differs (even by one element in an ACL array), a **NEW node is created** instead.

**2. ON CREATE (New Node)**:
- **ALL properties** from new memory are set (including all IDs and ACLs)
- `was_created_flag = true` (used for property indexing)

**3. ON MATCH (Existing Node)**:
- **Update properties** are set (excludes properties used in MERGE clause)
- **IDs and ACLs in MERGE clause are NOT updated** (they're matching criteria, not update targets)
- Only non-unique properties (like `name`, `description`, etc.) are updated
- `was_created_flag = false`

#### ACL and ID Fields Used in MERGE

**All fields included in MERGE matching** (`memory/memory_graph.py:9453-9534`):

| Field Type | Fields | Purpose | Updated on MATCH? |
|------------|--------|---------|-------------------|
| **IDs** | `user_id`, `workspace_id`, `organization_id`, `namespace_id` | Multi-tenant isolation | ❌ NO - Used for matching |
| **ACL - User** | `user_read_access`, `user_write_access` | Access control | ❌ NO - Used for matching |
| **ACL - Workspace** | `workspace_read_access`, `workspace_write_access` | Access control | ❌ NO - Used for matching |
| **ACL - Role** | `role_read_access`, `role_write_access` | Access control | ❌ NO - Used for matching |
| **ACL - Namespace** | `namespace_read_access`, `namespace_write_access` | Access control | ❌ NO - Used for matching |
| **ACL - Organization** | `organization_read_access`, `organization_write_access` | Access control | ❌ NO - Used for matching |
| **ACL - External User** | `external_user_read_access`, `external_user_write_access` | Access control | ❌ NO - Used for matching |
| **Content Properties** | `name`, `description`, `title`, etc. | Node data | ✅ YES - Updated on match |

**⚠️ Important**: ACL and ID fields are used as **matching criteria** with **AND logic** (exact match required). This means:
- ALL properties in MERGE must match EXACTLY (not OR - must match ALL)
- If ANY field differs (even one ACL array element), a NEW node is created
- Each tenant/user/ACL combination maintains separate nodes even with same unique identifiers

**Examples**:

**Example 1: Different Users** (creates 2 separate nodes):
```
Memory A: email="john@example.com", user_id="user_123" → Node 1
Memory B: email="john@example.com", user_id="user_456" → Node 2 (NEW)
```

**Example 2: Same User, Different ACLs** (creates 2 separate nodes):
```
Memory A: 
  email="john@example.com"
  user_id="user_123"
  workspace_read_access=["ws_456"]
  → Node 1

Memory B:
  email="john@example.com"
  user_id="user_123"  
  workspace_read_access=["ws_456", "ws_789"]  ← Different!
  → Node 2 (NEW, not merged)
```

**Example 3: Exact Match** (matches existing node):
```
Memory A:
  email="john@example.com"
  user_id="user_123"
  workspace_read_access=["ws_456"]
  all other ACLs match exactly
  → Node 1 created

Memory B:
  email="john@example.com"
  user_id="user_123"
  workspace_read_access=["ws_456"]
  all other ACLs match exactly
  → Node 1 MATCHED (updates content properties only)
```

#### Qdrant Property Collection Updates During MERGE

**When nodes are matched** (`was_created_flag = false`):
- Existing node's properties are updated (content properties only)
- Property Collection entries for those properties are re-indexed with:
  - **Updated property values**
  - **Same ACL fields** (inherited from original source memory that created the node)
  - `sync_operation = "update"`
  - `was_created = false`

**When new nodes are created** (`was_created_flag = true`):
- New property entries are added to Property Collection
- All ACL fields inherited from current memory
- `sync_operation = "create"`
- `was_created = true`

**Code Location**: 
- MERGE logic: `memory/memory_graph.py:9420-9575` (`_merge_node_by_unique_props`)
- Property sync: `services/property_indexing_service.py:245-350` (`sync_node_properties_to_qdrant`)

#### Summary

| Operation | Neo4j Node IDs | Neo4j Node ACLs | Neo4j Content Properties | Qdrant Property ACLs | Qdrant Property Values |
|-----------|----------------|-----------------|--------------------------|----------------------|------------------------|
| **MERGE (CREATE)** | ✅ Set from new memory | ✅ Set from new memory | ✅ Set from new memory | ✅ Inherited from source memory | ✅ Set from new memory |
| **MERGE (MATCH)** | ❌ NOT updated | ❌ NOT updated | ✅ Updated from new memory | ❌ NOT updated (stays with original) | ✅ Updated from new memory |

**Key Insight**: ACL and ID fields act as a **composite key** for deduplication, ensuring tenant isolation. They're never updated during MERGE to prevent cross-tenant data merging.

### k. Deduplication Strategy: Trade-offs and Inconsistencies

#### Current Issues

**Issue 1: Redundant ACLs in MERGE**

ACLs are included in MERGE alongside `user_id`:
```cypher
MERGE (n:Person {
  email: "john@example.com",
  user_id: "user_123",           // Already isolates per user
  user_read_access: ["user_123"] // ❌ REDUNDANT - always matches if user_id matches
  // ... other ACLs also redundant
})
```

**Why redundant**: If `user_id` already matches, ACLs will naturally match too. ACLs don't add isolation value.

**Issue 2: Inconsistency Between Qdrant and Neo4j** ✅ **FIXED**

| System | Deduplication Logic | Result |
|--------|---------------------|--------|
| **Qdrant Property Search** | MUST: `namespace_id` + `user_id` (AND logic) <br> SHOULD: ACL arrays (OR logic) | ✅ Finds properties for same user in namespace |
| **Neo4j MERGE** | MUST: `user_id` + `namespace_id` + all ACLs (AND logic) | Only matches if same user |

**Status**: ✅ **Consistent** - Both require exact `user_id` + `namespace_id` match for per-user isolation.

**Issue 3: Duplicate Entities from Multiple Users**

```
User A creates: Project "Website Redesign" (status="In Progress", user_id="A")
  → Neo4j Node 1

User B updates: Project "Website Redesign" (status="Completed", user_id="B")
  → Neo4j Node 2 (SEPARATE NODE - not updated!)

Search retrieves: BOTH nodes
LLM must: Determine which is more recent/accurate
```

**Result**: Fragmented knowledge graph with duplicate entities.

#### Design Trade-offs

**Current Approach: Per-User Isolation**
- **Isolation Boundary**: `user_id` + `namespace_id` + all ACLs
- **Pros**: 
  - ✅ Maximum data isolation
  - ✅ No cross-user conflicts
  - ✅ Users can't accidentally overwrite each other's data
- **Cons**:
  - ❌ No collaboration - each user has separate graph
  - ❌ Duplicate entities ("John Smith" exists N times)
  - ❌ Fragmented relationships
  - ❌ Search returns duplicates - LLM must reconcile

**Alternative: Namespace-Level Sharing**
- **Isolation Boundary**: `namespace_id` only (remove `user_id` from MERGE)
- **Pros**:
  - ✅ Collaborative graph - users share entities
  - ✅ One source of truth per entity
  - ✅ No duplicate entities
  - ✅ Better for team collaboration
- **Cons**:
  - ⚠️ Last write wins (need versioning for history)
  - ⚠️ Users can overwrite each other's updates
  - ⚠️ Requires proper namespace-per-customer setup

#### Recommendations

**Option 1: Fix Current Approach (Minimum Changes)** ✅ **PARTIALLY IMPLEMENTED**

1. ✅ **Make Qdrant consistent** - `user_id` added to MUST alongside `namespace_id`
2. ⏳ **Remove redundant ACLs from MERGE** - keep only `user_id` + `namespace_id`
3. ✅ **Document behavior** - each user has their own graph copy

**Current State**:
- ✅ Qdrant: `user_id` + `namespace_id` in MUST
- ⏳ Neo4j MERGE: Still has redundant ACL fields (should be removed)

**Recommended MERGE** (after cleanup):
```cypher
MERGE (n:Person {
  email: "john@example.com",
  user_id: "user_123",
  namespace_id: "ns_001"
  // ❌ TODO: REMOVE all ACLs, workspace_id, organization_id
})
```

**Option 2: Enable Collaboration (Better Long-term)**

1. **Remove `user_id` from MERGE** - use only `namespace_id`
2. **Track updates** - add `updated_by`, `updatedAt`, `version`
3. **Use ACLs for search filtering** - not deduplication
4. **One namespace per customer** - proper multi-tenancy

```cypher
MERGE (n:Person {
  email: "john@example.com",
  namespace_id: "ns_001"
  // ❌ REMOVE user_id, all ACLs
})
ON CREATE SET n += $all_properties, n.created_by = $user_id
ON MATCH SET 
  n += $update_properties,
  n.updated_by = $user_id,
  n.updatedAt = datetime(),
  n.version = n.version + 1
```

**Option 3: Hybrid Approach (Configurable)**

- Some node types shared (Person, Company, Product)
- Other node types private (PersonalNote, PrivateTask)
- Configure per schema which isolation level to use

#### Current Status

The system currently uses **per-user isolation** with:
- ✅ **Qdrant and Neo4j now consistent** - both use `user_id` + `namespace_id` for deduplication
- ⏳ **Redundant ACLs still in Neo4j MERGE** - should be removed for cleaner code
- ✅ **Documented design choice** - per-user isolation strategy is now clear

**Impact**: Each user maintains a separate copy of the knowledge graph, even when collaborating in the same namespace/organization. This ensures maximum data isolation but prevents collaborative graph building.

---

## 2. Search Memory (POST /v1/memory/search)

**Route**: `POST /v1/memory/search`  
**Handler**: `search_v1`  
**Location**: `routers/v1/memory_routes_v1.py:2154`

### a. Request Structure

**Top-Level Fields in API Request**:
- `query` (required): str - Search query text
- `metadata` (optional): MemoryMetadata - Used for filtering
- `organization_id` (optional): str - Filter by organization
- `namespace_id` (optional): str - Filter by namespace
- `user_id` (optional): str - Filter by specific user
- `external_user_id` (optional): str - Filter by external user
- `enable_agentic_graph` (optional): bool - Whether to use Neo4j graph search
- `rank_results` (optional): bool
- `schema_id` (optional): str
- `simple_schema_mode` (optional): bool
- `search_override` (optional): SearchOverrideSpecification

**Fields in `metadata` Object** - Used for filtering:
- **ACL Fields**: All ACL arrays (for filtering results)
- **Content Filters**: `role`, `category`, `topics`, `conversationId`, `location`
- **IDs**: `user_id`, `workspace_id`, `organization_id`, `namespace_id`, `external_user_id`

**Response Model (`SearchResponse`)**:
- `code`: int
- `status`: str
- `data`: SearchResult
  - `memories`: List[Memory] - Each Memory has all metadata fields including ACLs
  - `nodes`: List[Dict] - Graph nodes if `enable_agentic_graph=true`
- `search_id`: str - QueryLog objectId

### b. IDs Used

| Field | Source | Used | Filter (Qdrant) | Filter (Neo4j) | In Request | In Response | Notes |
|-------|--------|------|-----------------|----------------|------------|-------------|-------|
| `user_id` | `auth_response.end_user_id` | ✅ | ✅ | ✅ | ❌ | ❌ | Used for filtering in both Qdrant and Neo4j |
| `workspace_id` | `auth_response.workspace_id` | ✅ | ✅ | ✅ | ❌ | ❌ | Used for filtering |
| `organization_id` | `auth_response.organization_id` OR `search_request.organization_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Can be passed in request |
| `namespace_id` | `auth_response.namespace_id` OR `search_request.namespace_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Can be passed in request |
| `external_user_id` | `search_request.external_user_id` | ✅ | ✅ | ❌ | ✅ | ❌ | Optional filter |

### c. ACL Fields Used (for filtering)

| Field | Source | Used | Filter (Qdrant) | Filter (Neo4j) | In Request | In Response | Notes |
|-------|--------|------|-----------------|----------------|------------|-------------|-------|
| `user_read_access` | Derived from `user_id` | ✅ | ✅ (MatchAny) | ✅ | ✅ | ✅ | Qdrant filter: MatchAny |
| `workspace_read_access` | Derived from `workspace_id` | ✅ | ✅ (MatchAny) | ✅ | ✅ | ✅ | Qdrant filter: MatchAny |
| `role_read_access` | `auth_response.user_roles` | ✅ | ✅ (MatchAny) | ✅ | ✅ | ✅ | Qdrant filter: MatchAny |
| `namespace_read_access` | Derived from `namespace_id` | ✅ | ✅ (MatchAny) | ✅ | ✅ | ✅ | Qdrant filter: MatchAny |
| `organization_read_access` | Derived from `organization_id` | ✅ | ✅ (MatchAny) | ✅ | ✅ | ✅ | Qdrant filter: MatchAny |

### d. Qdrant Search Filtering

**Filter Conditions** (applied in `_search_qdrant_for_similar_content`):
- User access: `user_id = $user_id OR user_read_access MATCH ANY [$user_id]`
- Workspace access: `workspace_read_access MATCH ANY [$workspace_id]`
- Role access: `role_read_access MATCH ANY $user_roles`
- Namespace access: `namespace_read_access MATCH ANY [$namespace_id]`
- Organization access: `organization_read_access MATCH ANY [$organization_id]`
- Metadata filters: `metadata.role`, `metadata.category` (if provided)

**Code Location**: `memory/memory_graph.py:9760-9797` (Qdrant filtering)

### e. Neo4j Search Filtering

**Cypher Query Filters** (when `enable_agentic_graph=true`):
```cypher
WHERE (
  node.user_id = $user_id 
  OR any(x IN coalesce(node.user_read_access, []) WHERE x IN $user_read_access)
  OR any(x IN coalesce(node.workspace_read_access, []) WHERE x IN $workspace_read_access)
  OR any(x IN coalesce(node.role_read_access, []) WHERE x IN $role_read_access)
  OR any(x IN coalesce(node.organization_read_access, []) WHERE x IN $organization_read_access)
  OR any(x IN coalesce(node.namespace_read_access, []) WHERE x IN $namespace_read_access)
)
```

**Code Location**: `memory/memory_graph.py:4500-4540` (Neo4j access filtering)

---

## 3. Update Memory (PUT /v1/memory/{memory_id})

**Route**: `PUT /v1/memory/{memory_id}`  
**Handler**: `update_memory_v1`  
**Location**: `routers/v1/memory_routes_v1.py:501`

### a. Request Structure

**Top-Level Fields in API Request**:
- `content` (optional): str - Updated content
- `type` (optional): MemoryType
- `metadata` (optional): MemoryMetadata - Same as Add Memory
- `organization_id` (optional): str
- `namespace_id` (optional): str
- `context` (optional): List[ContextItem]
- `relationships_json` (optional): List[RelationshipItem]

**Fields in `metadata` Object**: Same as [Add Memory](#c-acl-fields-used)

**Response Model (`UpdateMemoryResponse`)**:
- `code`: int
- `status`: str
- `memory_items`: List[UpdateMemoryItem]
- `status_obj`: SystemUpdateStatus

### b. IDs Used

| Field | Source | Used | Set (Parse) | Set (Neo4j) | Set (Qdrant) | In Request | Notes |
|-------|--------|------|-------------|-------------|--------------|------------|-------|
| `user_id` | `auth_response.end_user_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Preserved from original |
| `workspace_id` | `auth_response.workspace_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Can be updated |
| `organization_id` | `auth_response.organization_id` OR `update_request.organization_id` | ✅ | ✅ | ✅ | ✅ | ✅ | Can be updated |
| `namespace_id` | `auth_response.namespace_id` OR `update_request.namespace_id` | ✅ | ✅ | ✅ | ✅ | ✅ | Can be updated |

### c. ACL Fields Used

**All ACL fields**: Same as [Add Memory](#c-acl-fields-used)
- Source: `update_request.metadata` OR preserved from original
- Set in: Parse ✅, Neo4j ✅, Qdrant ✅

**Important**: ACL preservation logic at `routers/v1/memory_routes_v1.py:649-664`

### d. Storage Updates

- **Neo4j Update**: `memory/memory_graph.py:4867-4929` (`update_memory_item_in_neo4j`)
- **Qdrant Update**: Chunks re-indexed with updated payload if content/metadata changes
- **Parse Update**: `services/memory_management.py:4430-4466`

---

## 4. Get Memory (GET /v1/memory/{memory_id})

**Route**: `GET /v1/memory/{memory_id}`  
**Handler**: `get_memory_v1`  
**Location**: `routers/v1/memory_routes_v1.py` (specific line not provided in original)

### Overview

This route retrieves a single memory by its ID. The response includes all stored metadata, ACL fields, and content.

**Request**: 
- Path parameter: `memory_id` (str)
- No body

**Response Model**:
- Returns a `Memory` object with all fields from Parse Server
- Includes all ACL arrays in the response
- Includes all metadata fields

**IDs and ACLs**: 
- **Used**: Memory ID for lookup
- **Returned**: All IDs and ACL fields stored in Parse Server

---

## 5. Batch Add Memory (POST /v1/memory/batch)

**Route**: `POST /v1/memory/batch`  
**Handler**: `add_memory_batch_v1`  
**Location**: `routers/v1/memory_routes_v1.py:858`

### a. Request Structure

**Top-Level Fields in API Request**:
- `memories` (required): List[AddMemoryRequest] - Each memory has its own metadata/ACLs
- `user_id` (optional): str - Applied to all memories if not in individual memory metadata
- `external_user_id` (optional): str - Resolved to internal user_id
- `organization_id` (optional): str - Applied to all memories
- `namespace_id` (optional): str - Applied to all memories
- `graph_generation` (optional): GraphGeneration - Applied to all memories
- `batch_size` (optional): int - Parallel processing batch size
- `webhook_url` (optional): str
- `webhook_secret` (optional): str

**Fields in Each Memory's `metadata` Object**: Same as [Add Memory](#1-add-memory-post-v1memory)

**Response Model (`AddMemoryResponse`)**: Same as Add Memory

### b. IDs, ACLs, and Storage

**All patterns follow [Add Memory](#1-add-memory-post-v1memory)** with these additions:

- **Batch-level IDs**: `user_id`, `organization_id`, `namespace_id` applied to ALL memories
- **Per-memory override**: Each memory can override with its own metadata
- **Batch processing**: Uses `batch_create_memory_nodes` for Neo4j (`memory/memory_graph.py:4768-4865`)

See Add Memory sections for detailed tables on:
- [IDs Used](#b-ids-used)
- [ACL Fields](#c-acl-fields-used)
- [Qdrant Storage](#e-qdrant-qwen-storage-summary)
- [Qdrant Searches](#f-qdrant-qwen-search-for-grouped-memories)
- [Neo4j Storage](#h-neo4j-storage-summary)

---

## 6. Upload Document (POST /v1/documents)

**Route**: `POST /v1/documents`  
**Handler**: Document upload workflow  
**Location**: `routers/v1/document_routes_v2.py:74`

### a. Request Structure

**Top-Level Fields in API Request**:
- `type` (required): MemoryType - Must be "document"
- `metadata` (optional): MemoryMetadata - Same as Add Memory
- `preferred_provider` (optional): PreferredProvider
- `hierarchical_enabled` (optional): bool
- File upload via multipart/form-data

**Fields in `metadata` Object**: Same as [Add Memory](#1-add-memory-post-v1memory)

**Response Model (`DocumentUploadResponse`)**:
- `code`: int
- `status`: str
- `document_status`: DocumentUploadStatus
- `memory_items`: List[AddMemoryItem]

### b. IDs Used

| Field | Source | Used | Set (Parse) | Set (Neo4j) | Set (Qdrant) | In Request | Notes |
|-------|--------|------|-------------|-------------|--------------|------------|-------|
| `user_id` | `auth_response.end_user_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memories from pages |
| `workspace_id` | `auth_response.workspace_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memories from pages |
| `organization_id` | `auth_response.organization_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memories |
| `namespace_id` | `auth_response.namespace_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memories |

### c. ACL Fields Used

**All ACL fields**: Same as [Add Memory](#c-acl-fields-used)
- Source: `document_metadata` OR merged from existing document
- ACLs are **merged** with user-provided ACLs if updating document

### d. Storage Pattern

- Each document page creates a memory following same pattern as [Add Memory](#1-add-memory-post-v1memory)
- Processing via Temporal workflow
- ACL merging at `cloud_plugins/temporal/activities/document_activities.py:1055-1092`

---

## 7. Messages Route (POST /v1/messages)

**Route**: `POST /v1/messages`  
**Handler**: `store_message`  
**Location**: `routers/v1/message_routes.py:62`

### a. Request Structure

**Top-Level Fields in API Request**:
- `content` (required): str - Message content
- `role` (required): MessageRole - "user" or "assistant"
- `metadata` (optional): MemoryMetadata - Same as Add Memory
- `conversationId` (optional): str
- Other message-specific fields

**Fields in `metadata` Object**: Same as [Add Memory](#1-add-memory-post-v1memory)

**Response Model (`MessageResponse`)**:
- `code`: int
- `status`: str
- `message_id`: str
- `memory_id`: Optional[str]

### b. IDs Used

| Field | Source | Used | Set (Parse) | Set (Neo4j) | Set (Qdrant) | In Request | Notes |
|-------|--------|------|-------------|-------------|--------------|------------|-------|
| `user_id` | `auth_response.end_user_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memory |
| `workspace_id` | `auth_response.workspace_id` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memory |
| `organization_id` | `multi_tenant_context.get("organization_id")` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memory |
| `namespace_id` | `multi_tenant_context.get("namespace_id")` | ✅ | ✅ | ✅ | ✅ | ❌ | Set when creating memory |

### c. ACL Fields Used

**All ACL fields**: Passed through to memory creation (same pattern as [Add Memory](#c-acl-fields-used))

### d. Storage Pattern

- Messages create memories via background task
- Memory storage follows same pattern as [Add Memory](#1-add-memory-post-v1memory)
- Code: `services/message_service.py:62-160`, `services/message_processing_pipeline.py`

---

## Reference: Parse Server Storage

### How Parse Server Stores IDs

**All IDs are stored as Pointers** to their respective classes:
- `user` → Pointer to `_User` class
- `workspace` → Pointer to `WorkSpace` class
- `organization` → Pointer to `Organization` class
- `namespace` → Pointer to `Namespace` class
- `developerUser` → Pointer to `DeveloperUser` class

**The string IDs come from TWO places**:
- **ACL Arrays**: `user_read_access`, `workspace_read_access`, etc. contain the `objectId` values extracted from the pointers
- **Deprecated Fields** (org/namespace only): `organization_id`, `namespace_id` string fields for backward compatibility

**For `external_user_id`**:
- Stored INSIDE the `developerUser` pointer as `developerUser.external_id`
- Also stored in `external_user_read_access` array

**Example Parse Server Memory Object**:
```json
{
  "user": {
    "__type": "Pointer",
    "className": "_User",
    "objectId": "user_123"
  },
  "user_read_access": ["user_123"],  // objectId from user pointer
  "workspace": {
    "__type": "Pointer",
    "className": "WorkSpace",
    "objectId": "ws_456"
  },
  "workspace_read_access": ["ws_456"],  // objectId from workspace pointer
  "organization": {
    "__type": "Pointer",
    "className": "Organization",
    "objectId": "org_789"
  },
  "organization_id": "org_789",  // DEPRECATED string field
  "organization_read_access": ["org_789"],  // objectId from organization pointer
  "developerUser": {
    "__type": "Pointer",
    "className": "DeveloperUser",
    "objectId": "dev_001",
    "external_id": "external_abc"  // external_user_id stored here
  },
  "external_user_read_access": ["external_abc"]  // external_id from developerUser
}
```

**Code**: See `models/parse_server.py:619-692` (`ParseStoredMemory` class)

### Field vs ACL Object Explanation

Parse Server has a special `ACL` object for access control. We store ACL information in **TWO different ways**:

#### 1. Field Only (NOT in ACL object)
These are stored as top-level array columns in the Memory class:

```json
{
  "workspace_read_access": ["ws_123", "ws_456"],
  "workspace_write_access": ["ws_123"],
  "namespace_read_access": ["ns_789"],
  "namespace_write_access": [],
  "organization_read_access": ["org_001"],
  "organization_write_access": [],
  "external_user_read_access": ["ext_user_1"],
  "external_user_write_access": []
}
```

**Why?** These fields are custom to our system and Parse Server's native ACL doesn't support them.

#### 2. Field + ACL Object (Stored in both places)
These are stored BOTH as array fields AND in Parse's special ACL object:

```json
{
  "user_read_access": ["user_123", "user_456"],
  "user_write_access": ["user_123"],
  "role_read_access": ["admin", "editor"],
  "role_write_access": ["admin"],
  
  "ACL": {
    "user_123": {"read": true, "write": true},
    "user_456": {"read": true, "write": false},
    "role:admin": {"read": true, "write": true},
    "role:editor": {"read": true, "write": false}
  }
}
```

**Why?** This provides **backward compatibility** with Parse Server's native ACL system while also allowing our system to query these fields as arrays.

**Code Location**: `services/memory_management.py:546-601` (`convert_acl` function)

---

## Reference: Central ACL Processing Patterns

### During Memory ADD Operations

**Step 1: ACL Fields Collection** (`services/memory_service.py:471-502`):
```python
# Define which ACL fields to merge from post/page/workspace
acl_fields = [
    'user_read_access', 'user_write_access',
    'workspace_read_access', 'workspace_write_access',
    'role_read_access', 'role_write_access'
]

# ⚠️ Note: namespace/organization ACLs are NOT in this list
# They are only set if explicitly passed in metadata
```

**Step 2: ACL Conversion for Parse Server** (`services/memory_management.py:546-601`):
```python
parse_acl = convert_acl(metadata)

# Converts:
# user_read_access=['user1'] → ACL: {"user1": {"read": true}}
# role_read_access=['admin'] → ACL: {"role:admin": {"read": true}}
# workspace/namespace/organization → Field only (NOT in ACL object)
```

**Step 3: Storage in All Systems**:
- **Parse Server**: ACL arrays as fields + Parse ACL object (user/role only)
- **Neo4j**: ALL ACL arrays as node properties
- **Qdrant**: ALL ACL arrays in payload

### During Memory SEARCH Operations

**Step 1: Extract User IDs** (from auth_response):
```python
user_id = auth_response.end_user_id
workspace_id = auth_response.workspace_id
organization_id = auth_response.organization_id
namespace_id = auth_response.namespace_id
user_roles = user_instance.roles
```

**Step 2: Build ACL Conditions** (`memory/memory_graph.py:5067-5183`):
```python
# Build SHOULD conditions (OR logic - user has access if ANY match)
should_conditions = []

# Direct ownership
should_conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

# ACL arrays - check if user_id is IN the array
should_conditions.append(FieldCondition(key="user_read_access", match=MatchAny(any=[user_id])))

# Workspace/Organization/Namespace/Role access
if workspace_id:
    should_conditions.append(FieldCondition(key="workspace_read_access", match=MatchAny(any=[workspace_id])))
if organization_id:
    should_conditions.append(FieldCondition(key="organization_read_access", match=MatchAny(any=[organization_id])))
if namespace_id:
    should_conditions.append(FieldCondition(key="namespace_read_access", match=MatchAny(any=[namespace_id])))
if user_roles:
    should_conditions.append(FieldCondition(key="role_read_access", match=MatchAny(any=user_roles)))
```

**Pattern**: Take the **ID value** from auth → Check if it's **IN** the corresponding `*_read_access` array

### Key Differences: ADD vs SEARCH

| Operation | ADD Memory | SEARCH Memory |
|-----------|------------|---------------|
| **ACL Source** | From metadata in request (or merged from post/page/workspace) | From auth_response (user's own IDs) |
| **ACL Format** | Arrays of IDs: `["user1", "user2"]` | Same arrays, but checking if auth user's ID is IN them |
| **Central Function** | `convert_acl()` - converts arrays to Parse ACL | Build `should_conditions` - converts IDs to filter conditions |
| **Logic** | Store: "These IDs have access" | Filter: "Does my ID exist in those arrays?" |
| **Location** | `services/memory_management.py:546-601` | `memory/memory_graph.py:5067-5183` |
| **Namespace/Org** | NOT merged from post/page (only if explicitly passed) | Checked in SHOULD conditions (OR logic) |

---

## Legend

- **Source**: Where the value comes from (auth, request body, derived)
- **Used**: Whether the field is used in the operation
- **Set**: Whether the field is set/stored in Parse Server/Neo4j/Qdrant
- **Filter**: Whether the field is used for filtering/search
- **Can Pass**: Whether developer can pass this field in API request
- **Location in Request**: Where field goes in request JSON (top-level vs in metadata object)
- **Default Behavior**: What happens if developer doesn't pass the field
- **In Response**: Whether the field is returned in the API response
- **MUST (AND)**: ALL conditions must be satisfied
- **SHOULD (OR)**: At least ONE condition must be satisfied
