# Node Constraint Behavior Rules

## 🎯 Understanding `set`, `update`, and `match`

### The Three Fields

```python
{
    "node_type": "Project",
    "match": {"name": "Alpha"},      # WHEN to apply
    "set": {"team": "backend"},      # FORCE these values
    "update": ["status"]             # UPDATE these from AI
}
```

---

## 📐 Behavior Matrix

### Scenario 1: Only `set` (No `update`)

**Configuration:**
```python
{
    "node_type": "Project",
    "set": {"workspace_id": "ws_123", "team": "backend"}
}
```

**AI Extracts:**
```python
Project(name="Alpha", status="completed", priority="high")
```

**Result:**
```python
# Final node properties:
{
    "name": "Alpha",           # From AI
    "status": "completed",     # From AI
    "priority": "high",        # From AI
    "workspace_id": "ws_123",  # From 'set' (forced)
    "team": "backend"          # From 'set' (forced)
}
```

**Key:** 
- ✅ AI-extracted properties kept
- ✅ `set` properties added/overridden
- ✅ No updates to existing nodes (unless they're found during dedup)

---

### Scenario 2: Only `update` (No `set`)

**Configuration:**
```python
{
    "node_type": "Project",
    "update": ["status", "priority"]
}
```

**AI Extracts:**
```python
Project(name="Alpha", status="completed", priority="high")
```

**Behavior:**
- **If new node:** Creates with AI values (status="completed", priority="high")
- **If existing node found:** Updates status="completed", priority="high" on existing

**Key:**
- ✅ Updates ONLY apply to existing matched nodes
- ✅ New nodes just get AI values (no "update" needed since it's new)

---

### Scenario 3: Both `set` AND `update`

**Configuration:**
```python
{
    "node_type": "Project",
    "set": {"workspace_id": "ws_123", "team": "backend"},
    "update": ["status", "priority"]
}
```

**AI Extracts:**
```python
Project(name="Alpha", status="completed", priority="high")
```

**Behavior:**

#### If **NEW** node created:
```python
{
    "name": "Alpha",           # From AI
    "status": "completed",     # From AI
    "priority": "high",        # From AI
    "workspace_id": "ws_123",  # From 'set' (forced)
    "team": "backend"          # From 'set' (forced)
}
```
- ✅ `set` applied (forced values)
- ⚠️ `update` ignored (no existing node to update)

#### If **EXISTING** node found:
```python
# Existing node before:
{
    "name": "Alpha",
    "status": "in_progress",
    "priority": "medium",
    "workspace_id": "ws_999",
    "team": "frontend"
}

# After linking + applying constraints:
{
    "name": "Alpha",           # Kept (existing)
    "status": "completed",     # Updated from AI (via 'update')
    "priority": "high",        # Updated from AI (via 'update')
    "workspace_id": "ws_123",  # Forced (via 'set' - overrides existing!)
    "team": "backend"          # Forced (via 'set' - overrides existing!)
}
```

**Key:**
- ✅ `set` **always** applies (both new and existing)
- ✅ `update` **only** applies to existing nodes
- ⚠️ `set` **wins** if both modify same property

---

### Scenario 4: Conflict Between `set` and `update`

**Configuration:**
```python
{
    "node_type": "Project",
    "set": {"status": "archived"},      # Force status to "archived"
    "update": ["status"]                # Update status from AI
}
```

**AI Extracts:**
```python
Project(name="Alpha", status="completed")
```

**Precedence: `set` wins!**

#### If **NEW** node:
```python
{
    "name": "Alpha",
    "status": "archived"  # From 'set' (not "completed" from AI)
}
```

#### If **EXISTING** node:
```python
{
    "name": "Alpha",
    "status": "archived"  # From 'set' (not "completed" from AI)
}
```

**Rule:** `set` **always** wins over `update` and AI extraction

---

### Scenario 5: With `match` Condition

**Configuration:**
```python
{
    "node_type": "Project",
    "match": {"name": "Alpha"},         # ONLY apply to Project "Alpha"
    "set": {"team": "backend"}
}
```

**AI Extracts Multiple Projects:**
```python
[
    Project(name="Alpha", status="active"),
    Project(name="Beta", status="active"),
    Project(name="Gamma", status="active")
]
```

**Result:**
```python
# Project Alpha:
{
    "name": "Alpha",
    "status": "active",
    "team": "backend"  # ✅ Constraint applied (matches)
}

# Project Beta:
{
    "name": "Beta",
    "status": "active"
    # ❌ No team field (doesn't match)
}

# Project Gamma:
{
    "name": "Gamma",
    "status": "active"
    # ❌ No team field (doesn't match)
}
```

**Key:** Constraint only applies to nodes matching the condition

---

### Scenario 6: `match` with Multiple Conditions

**Configuration:**
```python
{
    "node_type": "Task",
    "match": {"priority": "high", "assignee": "Alice"},
    "set": {"urgent": true}
}
```

**AI Extracts:**
```python
[
    Task(title="Fix bug", priority="high", assignee="Alice"),    # Matches both
    Task(title="Review PR", priority="high", assignee="Bob"),    # Only priority matches
    Task(title="Write docs", priority="low", assignee="Alice")   # Only assignee matches
]
```

**Result:**
```python
# Task 1 (matches both):
{"title": "Fix bug", "priority": "high", "assignee": "Alice", "urgent": true}  # ✅

# Task 2 (partial match):
{"title": "Review PR", "priority": "high", "assignee": "Bob"}  # ❌ No urgent

# Task 3 (partial match):
{"title": "Write docs", "priority": "low", "assignee": "Alice"}  # ❌ No urgent
```

**Rule:** ALL conditions in `match` must be true (AND logic)

---

## 🎨 Real-World Examples

### Example 1: Force workspace_id on ALL Projects

```python
{
    "node_type": "Project",
    "set": {"workspace_id": "ws_123"}
    # No 'match' → applies to ALL Projects
}
```

**Result:** Every Project gets workspace_id="ws_123"

---

### Example 2: Update status ONLY for Project "Alpha"

```python
{
    "node_type": "Project",
    "match": {"name": "Alpha"},
    "update": ["status"]
}
```

**Result:** 
- Project Alpha: status updated from AI
- Other projects: unchanged

---

### Example 3: Complex - Force team, update status, only for high priority

```python
{
    "node_type": "Project",
    "match": {"priority": "high"},
    "set": {"team": "backend"},
    "update": ["status"]
}
```

**AI Extracts:**
```python
[
    Project(name="Alpha", priority="high", status="completed"),
    Project(name="Beta", priority="low", status="active")
]
```

**Result:**
```python
# Alpha (high priority - matches):
{
    "name": "Alpha",
    "priority": "high",
    "status": "completed",  # From AI (via 'update')
    "team": "backend"       # Forced (via 'set')
}

# Beta (low priority - doesn't match):
{
    "name": "Beta",
    "priority": "low",
    "status": "active"
    # No team, no status update
}
```

---

## 📊 Precedence Summary

```
Property value precedence (highest to lowest):

1. node_constraints.set        ← Always wins
2. node_constraints.update     ← Only for existing nodes
3. AI extraction               ← Default
4. Schema defaults             ← Fallback
```

**Within same node_constraint:**
```
set > update > AI extraction
```

---

## 🔧 Decision Tree

```
For each extracted node:

1. Check if node_constraint exists for this node_type
   ├─ No → Use AI values ✅
   └─ Yes → Continue to step 2

2. Check 'match' condition (if specified)
   ├─ Doesn't match → Skip this constraint, use AI values ✅
   └─ Matches (or no match specified) → Continue to step 3

3. Search for existing node (based on mode)
   ├─ mode="auto" → Search, create if not found
   ├─ mode="link_only" → Search, skip if not found
   └─ mode="use_existing" → Use use_node_id directly

4. Apply 'set' (if specified)
   ├─ Force these values on node (new or existing)
   └─ Overrides AI extraction

5. Apply 'update' (if specified AND node is existing)
   ├─ Update these properties from AI extraction
   └─ Only applies to existing matched nodes
   
6. Check precedence on conflicts
   └─ set wins over update
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Expecting `update` to Work on New Nodes

**❌ Wrong Assumption:**
```python
{
    "node_type": "Project",
    "update": ["status"]
}
# Thinking: "This will set status on new nodes"
```

**✅ Reality:**
- `update` only applies to **existing** nodes
- New nodes just get AI-extracted status
- If you want to force status on ALL nodes, use `set`

---

### Pitfall 2: Forgetting `set` Overrides `update`

**Configuration:**
```python
{
    "node_type": "Project",
    "set": {"status": "archived"},
    "update": ["status"]
}
```

**❌ Wrong Expectation:**
"Update will use AI value"

**✅ Reality:**
status = "archived" (from `set`, not AI)

---

### Pitfall 3: `match` Partial Matches

**Configuration:**
```python
{
    "node_type": "Task",
    "match": {"priority": "high", "assignee": "Alice"}
}
```

**❌ Wrong Expectation:**
"Applies if priority=high OR assignee=Alice"

**✅ Reality:**
Only applies if priority=high AND assignee=Alice (both must match)

---

## 🎓 Best Practices

### 1. Use `set` for Fixed Values
```python
# Good: Force workspace_id on all nodes
{
    "node_type": "Project",
    "set": {"workspace_id": "ws_123"}
}
```

### 2. Use `update` for Dynamic Values from AI
```python
# Good: Update status based on AI analysis
{
    "node_type": "Project",
    "update": ["status", "priority"]
}
```

### 3. Use `match` for Conditional Logic
```python
# Good: Only force team for Project Alpha
{
    "node_type": "Project",
    "match": {"name": "Alpha"},
    "set": {"team": "backend"}
}
```

### 4. Don't Overlap `set` and `update` on Same Property
```python
# ❌ Bad: Confusing, set wins anyway
{
    "node_type": "Project",
    "set": {"status": "archived"},
    "update": ["status"]  # Pointless, set wins
}

# ✅ Good: Choose one
{
    "node_type": "Project",
    "set": {"status": "archived"}  # Force this exact value
}
# OR
{
    "node_type": "Project",
    "update": ["status"]  # Use AI-analyzed value
}
```

---

## 📝 Quick Reference

| Field | When Applied | Value Source | Applies To |
|-------|-------------|--------------|------------|
| `set` | Always | YOU specify | All nodes (new + existing) |
| `update` | Only existing | AI extraction | Only existing matched nodes |
| `match` | Filter condition | N/A | Determines if constraint applies |

**Precedence:** `set` > `update` > AI extraction

**Match Logic:** ALL conditions must be true (AND logic)

