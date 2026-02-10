# Testing Guide for Memory OpenSource

This guide explains how to run tests for the Memory OpenSource project.

## Quick Start

### Run All V1 Tests

```bash
# Simple one-command test execution
./scripts/run_tests_docker.sh
```

This script will:
- ✅ Run the complete V1 test suite in Docker
- ✅ Automatically save test reports to `tests/test_reports/`
- ✅ Work with or without Docker volume mounting configured
- ✅ Display a summary when complete

### Run a Single Test

```bash
# Run one specific test
./tests/run_single_test.sh "tests/test_add_memory_fastapi.py::test_v1_add_memory_1"
```

## Test Reports

After running tests, reports are saved to:
```
tests/test_reports/
├── v1_endpoints_opensource_report_YYYYMMDD_HHMMSS.json  # Detailed JSON results
└── v1_endpoints_opensource_log_YYYYMMDD_HHMMSS.txt      # Human-readable summary
```

### View Test Results

```bash
# View latest test summary
cat tests/test_reports/v1_endpoints_opensource_log_*.txt | tail -50

# View full JSON report
cat tests/test_reports/v1_endpoints_opensource_report_*.json | jq '.summary'
```

## Test Suite Overview

The V1 test suite includes **~119 tests** covering:

| Test Group | Count | Description |
|------------|-------|-------------|
| Add Memory | 8 | Basic memory creation |
| Batch Add Memory | 10 | Bulk operations + webhooks |
| Update Memory | 3 | Memory updates |
| Get Memory | 1 | Memory retrieval |
| Search Memory | 17 | Search, ACL, filters |
| Memory Policy | 24 | Graph policies, link_to DSL |
| Schema Policy | 8 | Schema-level policies |
| OMO Safety | 6 | Privacy/consent enforcement |
| Delete Memory | 5 | Memory deletion |
| User Management | 10 | Create/update/delete users |
| Feedback | 2 | User feedback API |
| Query Log | 17 | Search analytics |
| Multi-tenant | 4 | Organization/namespace scoping |
| Document Processing | 5 | PDF/document handling |
| Messages | 1 | Message endpoint |

## Running Tests Locally (Without Docker)

If you prefer to run tests directly with Poetry:

```bash
# Set up environment variables to prevent OpenBLAS hangs
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run all tests
poetry run python tests/run_v1_tests_opensource.py

# Run specific test file
poetry run pytest tests/test_add_memory_fastapi.py -v

# Run one test
poetry run pytest tests/test_add_memory_fastapi.py::test_v1_add_memory_1 -v
```

**Note**: Local testing requires:
- All services running (MongoDB, Neo4j, Qdrant, Parse Server, Redis)
- Local embedding model (~500MB) will be downloaded on first run
- Proper environment variables configured in `.env`

## Docker Volume Mounting (Optional)

For instant test report access, configure Docker Desktop file sharing:

1. Open **Docker Desktop**
2. Go to **Settings** → **Resources** → **File Sharing**
3. Add: `/Users/your-username/Documents/GitHub` (or parent directory)
4. Click **Apply & Restart**

This allows test reports to appear instantly in `tests/test_reports/` without manual copying.

**Without this configuration**: The test script automatically copies reports after tests complete.

## Troubleshooting

### Tests Hang During Startup

If tests hang when loading the embedding model:

```bash
# Check if threading environment variables are set
echo $OPENBLAS_NUM_THREADS  # Should be 1
echo $OMP_NUM_THREADS       # Should be 1
echo $MKL_NUM_THREADS       # Should be 1

# These are automatically set in Docker, but needed for local runs
```

### Reports Not Appearing

```bash
# Check if Docker services are running
docker compose ps

# Check test-runner logs
docker compose logs test-runner

# Manually copy reports from last run
docker cp papr-test-runner:/app/tests/test_reports/. tests/test_reports/
```

### Parse Server Authentication Errors

```bash
# Verify test credentials are current
curl -X GET http://localhost:1337/parse/users/me \
  -H "X-Parse-Application-Id: papr-oss-app-id" \
  -H "X-Parse-Session-Token: YOUR_SESSION_TOKEN"

# Update credentials in .env and docker-compose.yaml if needed
```

## Contributing - Running Tests for PRs

Before submitting a PR:

1. **Run the full test suite**:
   ```bash
   ./scripts/run_tests_docker.sh
   ```

2. **Check results**:
   ```bash
   # View summary
   tail -50 tests/test_reports/v1_endpoints_opensource_log_*.txt
   
   # Look for this section:
   # Total Tests: 114
   # Passed: XX ✅
   # Failed: XX ❌
   # Success Rate: XX%
   ```

3. **Include test results in PR description**:
   - Copy the test summary (Total/Passed/Failed/Success Rate)
   - Attach the JSON report if helpful
   - Note any expected failures (if fixing specific bugs)

4. **For bug fixes, run the specific failing test**:
   ```bash
   ./tests/run_single_test.sh "tests/test_file.py::test_name"
   ```

## CI/CD Integration

The test suite is designed to run in CI environments:

```yaml
# Example GitHub Actions
- name: Run V1 Test Suite
  run: ./scripts/run_tests_docker.sh
  
- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: tests/test_reports/
```

## Advanced Testing

### Run Tests with Custom Options

```bash
# Run with verbose output
docker compose run --rm test-runner poetry run pytest tests/ -v -s

# Run only failed tests from last run
docker compose run --rm test-runner poetry run pytest tests/ --lf

# Run with coverage
docker compose run --rm test-runner poetry run pytest tests/ --cov=. --cov-report=html
```

### Debug Single Test

```bash
# Run single test with full debug output
docker compose run --rm test-runner poetry run pytest \
  tests/test_add_memory_fastapi.py::test_v1_add_memory_1 \
  -v -s --tb=long --log-cli-level=DEBUG
```

## Need Help?

- **Test failures**: Check logs in `tests/test_reports/` for detailed error messages
- **Setup issues**: See main [README.md](README.md) for Docker setup
- **Questions**: Open an issue on GitHub with your test output

---

**Happy Testing!** 🧪✨
