# Deployment Guide: v0.2.0 - Batch Processing by Default

## What Changed in v0.2.0

This release makes batch memory processing the **default path** for all new workflow executions:

- ✅ New workflows use `batch_add_memory_quick` activity (processes memories in batches)
- ✅ Legacy `add_memory_quick` kept as fallback for old workflow replays
- ✅ Worker versioning implemented via build IDs (format: `v{semver}+{feature}.{timestamp}`)
- ✅ Temporal patch gate `default-batch-processing` ensures deterministic execution

**Build ID for this release:** `v0.2.0+batch-default.20251117`

---

## Deployment Steps

### 1. Stop all existing workers

```bash
# Kill all worker processes
pkill -f start_all_workers.py
pkill -f start_temporal_worker.py
pkill -f start_document_worker.py

# Verify nothing is running
ps aux | grep -E "start_(all|temporal|document)_worker" | grep -v grep
# ↑ Should return nothing
```

### 2. Pull latest code and install dependencies

```bash
cd /Users/shawkatkabbara/Documents/GitHub/memory

# If using Poetry (recommended)
poetry install

# Verify version
poetry run python -c "from version import __version__; print(__version__)"
# ↑ Should print: 0.2.0
```

### 3. Start workers with new build ID

```bash
# Option A: Start all workers (recommended)
nohup poetry run python start_all_workers.py > logs/start_all_workers.log 2> logs/start_all_workers.err &

# Option B: Start workers separately
nohup poetry run python start_temporal_worker.py > logs/start_temporal_worker.log 2>&1 &
nohup poetry run python start_document_worker.py > logs/start_document_worker.log 2>&1 &
```

### 4. Verify workers are running

```bash
# Check logs for build ID
tail -f logs/start_all_workers.log | grep "Worker build ID"
# ↑ Should show: 🏗️  Worker build ID: v0.2.0+batch-default.20251117

# Check for batch processing logs (after triggering a test workflow)
tail -f logs/start_all_workers.log | grep "default-batch-processing"
# ↑ Should show: ✅ default-batch-processing patch active: using batch processing
```

### 5. (Optional) Configure Temporal Cloud Worker Versioning

If you want Temporal Cloud to only route new workflows to the new build:

**Note:** The official `temporal` CLI (not `tctl`) is recommended for Temporal Cloud.

```bash
# Install Temporal CLI if needed
brew install temporal

# Or download from: https://docs.temporal.io/cli

# Describe current task queue
temporal task-queue describe \
  --namespace papr-memory.pq3ak \
  --task-queue memory-processing

# Set the new build ID as default (routes new workflows to v0.2.0 workers)
temporal task-queue update-build-ids add-new-default \
  --namespace papr-memory.pq3ak \
  --task-queue memory-processing \
  --build-id "v0.2.0+batch-default.20251117"

# Do the same for document processing queue
temporal task-queue update-build-ids add-new-default \
  --namespace papr-memory.pq3ak \
  --task-queue document-processing-v2 \
  --build-id "v0.2.0+batch-default.20251117"
```

---

## Verification

### Test batch processing is active

1. Trigger a new memory batch workflow (e.g., via API or document upload)
2. Check logs for the batch path:

```bash
grep "default-batch-processing patch active" logs/start_all_workers.log
grep "Processing.*memories in batches" logs/start_all_workers.log
grep "batch_add_memory_quick" logs/start_all_workers.log
```

3. Check Temporal Cloud UI:
   - Navigate to workflow execution
   - Look for `batch_add_memory_quick` activity (not `add_memory_quick`)
   - Verify build ID is `v0.2.0+batch-default.20251117`

### Rollback (if needed)

If you need to rollback to the old version:

```bash
# 1. Stop new workers
pkill -f start_all_workers.py

# 2. Checkout previous code
git checkout <previous-commit>

# 3. Start workers (they'll use old build ID)
nohup poetry run python start_all_workers.py > logs/start_all_workers.log 2>&1 &
```

---

## Future Version Upgrades

When releasing future versions (e.g., v0.3.0):

1. **Update `pyproject.toml`:**
   ```toml
   version = "0.3.0"
   ```

2. **Update worker scripts** (if adding new features):
   ```python
   feature_id = "new-feature-name"  # e.g., "graph-memory"
   timestamp = "YYYYMMDD"           # Date of release
   ```

3. The `version.py` helper will automatically read the new semver from `pyproject.toml`

4. Build ID will automatically become: `v0.3.0+new-feature-name.YYYYMMDD`

---

## Troubleshooting

### Workers still using `add_memory_quick`

**Cause:** Old worker process still running, or logs are from pre-upgrade workflows

**Fix:**
```bash
# Force kill all workers
pkill -9 -f start_all_workers.py
pkill -9 -f start_temporal_worker.py
pkill -9 -f start_document_worker.py

# Clear old logs (optional)
> logs/start_all_workers.log

# Restart
nohup poetry run python start_all_workers.py > logs/start_all_workers.log 2>&1 &
```

### Build ID not showing in Temporal Cloud

**Cause:** Worker started without `build_id` parameter (likely from cached code)

**Fix:**
```bash
# Verify you have latest code
git pull origin main

# Restart workers
pkill -f start_all_workers.py
nohup poetry run python start_all_workers.py > logs/start_all_workers.log 2>&1 &

# Check build ID in logs
grep "Worker build ID" logs/start_all_workers.log
```

### Nondeterminism errors in Temporal

**Cause:** Old workflow history replaying against new code without proper versioning gates

**Fix:** This should **not** happen with v0.2.0 because we use `workflow.patched("default-batch-processing")`. If it does:

1. Check the workflow was started AFTER the v0.2.0 deployment
2. Old workflows (started before v0.2.0) should automatically use the legacy path
3. If issue persists, check for other code changes that might affect determinism

---

## Monitoring

Key metrics to watch after deployment:

- **Activity type distribution:** Ratio of `batch_add_memory_quick` vs `add_memory_quick`
- **Workflow execution time:** Should improve with batch processing
- **MongoDB connection pool usage:** Should decrease (batch activities reuse connections)
- **Error rates:** Watch for any spikes in activity failures

Query examples (in logs):

```bash
# Count batch vs individual calls
grep -c "batch_add_memory_quick" logs/start_all_workers.log
grep -c "add_memory_quick" logs/start_all_workers.log

# Check for errors
grep -i "error\|exception\|failed" logs/start_all_workers.log | tail -20
```

