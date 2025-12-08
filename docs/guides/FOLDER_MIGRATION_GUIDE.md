# Folder Migration Guide

## Current vs Target Structure

### Current Structure (Mixed)
```
memory/
├── api_handlers/           # Mixed - keep for now
├── background_tasks/       # Keep - background processing
├── datastore/             # RENAME → core/datastores/
├── memory/                # MOVE → core/domain/memory/
├── models/                # MOVE → core/models/
├── routers/               # MOVE → core/routers/
├── routes/                # MOVE → core/routes/ (consolidate with routers)
├── services/              # SPLIT:
│   ├── memory_service.py  → core/services/
│   ├── stripe_service.py  → cloud_plugins/stripe/ ✅ (already done)
│   ├── auth_utils.py      → core/services/
│   └── ...
├── scripts/               # SPLIT:
│   ├── stripe/            → cloud_scripts/ ✅ (already done)
│   └── ...                → keep scripts/
├── core/                  # ✅ Started (telemetry, subscription, first_run)
├── cloud_plugins/         # ✅ Created (stripe, temporal)
├── plugins/               # ✅ Created (for community plugins)
└── config/                # ✅ Done
```

### Target Structure (Clean)
```
memory/
├── core/                          # ✅ CORE BUSINESS LOGIC
│   ├── services/                  # Core services
│   │   ├── memory_service.py      # From services/
│   │   ├── search_service.py      # From services/
│   │   ├── auth_utils.py          # From services/
│   │   ├── telemetry.py          # ✅ Done
│   │   ├── subscription.py        # ✅ Done
│   │   └── first_run.py          # ✅ Done
│   │
│   ├── models/                    # Data models
│   │   └── ...                    # From models/
│   │
│   ├── routers/                   # API routes (FastAPI routers)
│   │   ├── memories.py            # From routers/v1/
│   │   ├── search.py
│   │   └── ...
│   │
│   ├── datastores/                # Pluggable datastores
│   │   ├── vector/                # From datastore/
│   │   │   ├── base.py           # New interface
│   │   │   └── qdrant.py         # Wrap existing qdrant
│   │   ├── graph/
│   │   │   ├── base.py           # New interface
│   │   │   └── neo4j.py          # Wrap existing neo4j
│   │   └── cache/
│   │       └── redis.py           # From datastore/
│   │
│   └── domain/                    # Domain logic (optional)
│       └── memory/                # From memory/
│
├── cloud_plugins/                 # ✅ CLOUD-ONLY
│   ├── stripe/                    # ✅ Done
│   ├── temporal/                  # ✅ Done
│   ├── amplitude/                 # TODO
│   ├── azure/                     # TODO
│   └── auth0/                     # TODO
│
├── plugins/                       # ✅ COMMUNITY PLUGINS
│   └── posthog/
│
├── api_handlers/                  # Keep (API utilities)
├── background_tasks/              # Keep (background processing)
├── scripts/                       # Keep (OSS scripts only)
├── utils/                         # Keep (utilities)
└── config/                        # ✅ Done
```

---

## Migration Strategy: Incremental, Not Big Bang

### ⚠️ **Don't Do This All At Once!**

Instead, migrate **incrementally** so your app keeps working:

1. ✅ **Phase 1 (Done)**: Feature flags, cloud plugins (stripe, temporal)
2. **Phase 2**: Create interfaces, wrap existing code
3. **Phase 3**: Gradually move files, test after each move
4. **Phase 4**: Clean up duplicates

---

## Phase 2: Create Interfaces (Don't Move Yet!)

### Step 1: Wrap Existing Services (No Breaking Changes)

```bash
# Create interfaces WITHOUT moving existing code
mkdir -p core/datastores/vector
mkdir -p core/datastores/graph
mkdir -p core/datastores/cache
```

#### Create Vector Store Interface

```python
# core/datastores/vector/base.py (NEW)
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """Base interface for vector databases"""
    
    @abstractmethod
    async def insert(self, vectors, ids, metadata): pass
    
    @abstractmethod
    async def search(self, query_vector, top_k=10): pass
```

#### Wrap Existing Qdrant

```python
# core/datastores/vector/qdrant.py (NEW - wraps existing)
from .base import VectorStore
from datastore.qdrant_connection import get_qdrant_client  # Your existing code!

class QdrantVectorStore(VectorStore):
    """Wrapper around your existing Qdrant code"""
    
    def __init__(self):
        self.client = get_qdrant_client()  # Use existing!
    
    async def insert(self, vectors, ids, metadata):
        # Use your existing Qdrant code
        return await self.client.upsert(...)
    
    async def search(self, query_vector, top_k=10):
        # Use your existing Qdrant code
        return await self.client.search(...)
```

#### Factory Pattern

```python
# core/datastores/__init__.py (NEW)
from .vector.qdrant import QdrantVectorStore

def get_vector_store():
    """Get vector store (currently just Qdrant)"""
    return QdrantVectorStore()  # Later: support multiple providers
```

#### Use in Your Code (Gradual Migration)

```python
# In services/memory_service.py (EXISTING FILE - small change)

# Old way (keep for now if you want):
from datastore.qdrant_connection import qdrant_client

# New way (add this):
from core.datastores import get_vector_store

vector_store = get_vector_store()
await vector_store.insert(vectors, ids, metadata)
```

---

## Phase 3: Gradual File Migration (One at a Time)

### Example: Migrate `services/memory_service.py`

```bash
# 1. Copy (don't move) first
cp services/memory_service.py core/services/memory_service.py

# 2. Update imports in the new file
# Edit core/services/memory_service.py:
#   - Update relative imports
#   - Test thoroughly

# 3. Update ONE route to use new location
# In routers/v1/memory_routes_v1.py:
# from services.memory_service import MemoryService  # Old
from core.services.memory_service import MemoryService  # New

# 4. Test that route works

# 5. Update all other imports gradually

# 6. Only when ALL imports updated, delete old file:
# rm services/memory_service.py
```

---

## What to Do Right Now (Practical Steps)

### Option A: Minimal Migration (Recommended)

**Just organize what matters, don't touch working code:**

1. ✅ **Already Done:**
   - `core/services/telemetry.py` ✅
   - `core/services/subscription.py` ✅
   - `cloud_plugins/stripe/` ✅
   - `cloud_plugins/temporal/` ✅
   - `config/` ✅

2. **Leave As-Is (No Migration Needed):**
   - `services/` → Keep all existing services here
   - `models/` → Keep all models here
   - `routers/` → Keep all routes here
   - `routes/` → Keep all routes here
   - `datastore/` → Keep datastore connections
   - `memory/` → Keep memory logic

3. **Only Add New Code to `core/`:**
   - New services → `core/services/`
   - New cloud features → `cloud_plugins/`
   - New community features → `plugins/`

**Result**: Clean structure for new code, existing code works as-is.

---

### Option B: Full Migration (Long Term Project)

**Do this incrementally over weeks/months:**

#### Week 1: Services

```bash
# Create structure
mkdir -p core/services

# Move non-cloud services ONE AT A TIME
# Test after each move!

# 1. memory_service.py (core business logic)
git mv services/memory_service.py core/services/
# Fix imports, test

# 2. auth_utils.py (shared auth logic)
git mv services/auth_utils.py core/services/
# Fix imports, test

# 3. Continue one by one...
```

#### Week 2: Models

```bash
mkdir -p core/models

# Move models one at a time
git mv models/memory_models.py core/models/
# Fix imports, test
```

#### Week 3: Datastores

```bash
mkdir -p core/datastores/{vector,graph,cache}

# 1. Create interfaces first
# core/datastores/vector/base.py

# 2. Wrap existing implementations
# core/datastores/vector/qdrant.py (wrapper)

# 3. Add factory
# core/datastores/__init__.py

# 4. Gradually switch to factory pattern
```

---

## Recommended Approach: **Option A (Minimal)**

### Why?

1. **Your app works** - don't break it!
2. **Time is valuable** - focus on features, not reorganization
3. **Incremental is safer** - migrate when you touch files naturally
4. **OSS contributors** will understand your structure over time

### What This Means for Folders:

```
memory/
├── core/              # NEW code goes here
│   ├── services/      # New services (telemetry, subscription, etc.)
│   └── datastores/    # New pluggable datastores (when ready)
│
├── cloud_plugins/     # Cloud-only features
│   ├── stripe/        # ✅
│   └── temporal/      # ✅
│
├── services/          # EXISTING - keep as-is for now
│   ├── memory_service.py
│   ├── search_service.py
│   └── ...
│
├── models/            # EXISTING - keep as-is
├── routers/           # EXISTING - keep as-is  
├── routes/            # EXISTING - keep as-is
├── datastore/         # EXISTING - keep as-is
└── memory/            # EXISTING - keep as-is
```

**Migration happens naturally:**
- Refactoring a service? Move it to `core/services/`
- Adding auth plugin? Put in `cloud_plugins/auth0/`
- Adding vector DB? Create `core/datastores/vector/chroma.py`

---

## Files That MUST Move (For OSS Distribution)

These are already done or can stay where they are:

### ✅ Already Moved to cloud_plugins/
- `services/stripe_service.py` → `cloud_plugins/stripe/service.py`
- `scripts/stripe/` → `cloud_scripts/stripe/`

### Should Move to cloud_plugins/ (Optional)
```bash
# If you want to keep cloud code separate:
mkdir -p cloud_plugins/azure
mv services/azure_webhook_consumer.py cloud_plugins/azure/

mkdir -p cloud_plugins/amplitude
# (amplitude code is in services/utils.py - can extract later)

mkdir -p cloud_plugins/auth0
# (auth0 integration - can extract later)
```

### Can Stay Where They Are
- `services/memory_service.py` → Core logic, fine in `services/`
- `models/` → Data models, fine where they are
- `routers/` → API routes, fine where they are

---

## TL;DR: What You Should Do Now

### ✅ Phase 1 Complete! (Just Finished)
- Feature flags ✅
- Temporal plugin ✅
- Batch validation ✅

### Next: **Don't Reorganize Everything!**

**Instead:**

1. **Keep existing structure** - it works!
2. **Use `core/` for new code** - establishes pattern
3. **Move files naturally** - when you refactor them anyway
4. **Open source script** - `prepare_open_source.py` handles exclusions

### When You DO Reorganize (Eventually):

**Priority Migration (if you want):**
1. Extract Azure/Amplitude to `cloud_plugins/` (clean separation)
2. Create datastore interfaces (pluggable backends)
3. Move core services to `core/services/` (one at a time)
4. Consolidate `routes/` and `routers/` (confusing to have both)

**But honestly?** Leave it for now. Focus on **features and users**, not **folder aesthetics**.

---

## Questions?

**Q: Should I move `services/` to `core/services/` now?**  
A: No. Only move files when you're refactoring them anyway. Copy-paste method ensures nothing breaks.

**Q: What about `routers/` vs `routes/`?**  
A: Keep both for now. Eventually consolidate to `core/routers/`, but not urgent.

**Q: Do I need to move `models/`?**  
A: No. `models/` at the root is fine. Only move if you want a cleaner structure.

**Q: What about `datastore/`?**  
A: Create `core/datastores/` for NEW pluggable interfaces. Keep `datastore/` for existing connections.

---

**Bottom Line**: ✅ **Phase 1 is complete!** Your folder structure is "good enough" for open source. Refactor later when needed.

