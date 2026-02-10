# Running Tests - Memory Open Source

## Quick Start for Contributors

Running tests is simple - just one command:

```bash
# Run core tests (recommended for PR validation)
./run_tests.sh

# Or run a specific test
./run_tests.sh single tests/test_add_memory_fastapi.py::test_v1_add_memory_1

# Or run the full suite
./run_tests.sh all
```

**That's it!** The script handles everything automatically:
- ✅ Stops the server temporarily (frees memory for tests)
- ✅ Runs tests inside Docker (uses already-loaded services)
- ✅ Restarts the server when done
- ✅ No manual setup required!

## Why This Approach?

### The Memory Challenge

The embedding model (Qwen 0.6B) uses ~1GB RAM. When you run tests:
- **Server process**: Has model loaded (~1GB)
- **Test process**: Tries to load model again (~1GB)
- **Total**: 2GB in an 8GB container = Out of Memory 💥

### The Solution

The `run_tests.sh` script temporarily stops the FastAPI server, runs tests (which load the model), then restarts the server. This way only one copy of the model is in memory at a time.

```
Before Tests:  Server (1GB model) ✅
During Tests:  [Server stopped] → Tests (1GB model) ✅  
After Tests:   Server (1GB model) ✅
```

## Test Modes

### Quick Mode (Default - ~2-3 minutes)
Tests core V1 functionality:
- Add memory
- Add memory with API key
- Search memory

```bash
./run_tests.sh
# or
./run_tests.sh quick
```

### Single Test Mode
Run one specific test:

```bash
./run_tests.sh single tests/test_add_memory_fastapi.py::test_v1_add_memory_1
```

### Full Suite (~15-30 minutes)
Runs all 100+ V1 endpoint tests:

```bash
./run_tests.sh all
```

## Requirements

**Docker Resources** (minimum):
- **Memory**: 8GB
- **CPUs**: 4 cores
- **Swap**: 2GB
- **Disk**: 20GB

These are already the defaults for most Docker Desktop installations. If tests fail with OOM errors, check Docker Desktop → Settings → Resources.

## Alternative: Run Tests Locally

If you prefer to test outside Docker (faster iteration):

```bash
# Install dependencies
poetry install

# Start Docker services only (not the FastAPI server)
docker compose up -d mongodb neo4j qdrant redis parse-server

# Run tests locally (connects to Docker services via localhost)
poetry run pytest tests/test_add_memory_fastapi.py::test_v1_add_memory_1 -v
```

**Note**: This requires ~12GB RAM total (your machine + Docker services).

## Troubleshooting

### Test fails with "Out of Memory" (exit code 137)

The server might have auto-restarted. Run the script again - it handles this automatically.

### Test times out or hangs

The embedding model takes ~2 minutes to load the first time. Subsequent tests are faster because the model is cached.

### "Container not found"

Start the stack first:
```bash
docker compose up -d
```

### Neo4j connection errors

This is non-critical - tests will pass in fallback mode without Neo4j graph storage.

## CI/CD Integration

For GitHub Actions or other CI:

```yaml
- name: Run Tests
  run: ./run_tests.sh quick
  timeout-minutes: 10
```

## Architecture Note

The test script uses Docker's process isolation:
- **Container**: Shared among all processes (services, tests)
- **Processes**: Separate memory spaces
- **Python singletons**: Only work within a single process

This is why we can't have both server and tests running simultaneously - they're separate Python processes that each need to load the model.

## Future Improvements

Potential optimizations (not blocking):
1. **Shared memory models**: Use Unix domain sockets or shared memory to pass model between processes
2. **Increase Docker RAM**: Allow both server + tests to run (requires 12GB+ Docker memory)
3. **Remote model server**: Run embedding model as a separate service that both server and tests call

For now, the `run_tests.sh` script provides a simple, working solution that requires no additional setup! 🎉
