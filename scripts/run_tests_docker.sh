#!/bin/bash
# Comprehensive test runner for memory-opensource
# Handles volume mounting + fallback report copying automatically
# Works for all contributors regardless of Docker Desktop file sharing settings

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPORTS_DIR="./tests/test_reports"
CONTAINER_NAME="memory-oss-test-run-$(date +%s)"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Memory OpenSource - V1 Test Suite Runner                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Ensure reports directory exists
mkdir -p "$REPORTS_DIR"

# Check if volume mounting will work (optional check)
if docker run --rm -v "$PWD:/test" alpine sh -c "test -d /test" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Docker volume mounting is working"
    VOLUME_MOUNT_WORKS=true
else
    echo -e "${YELLOW}⚠${NC}  Docker volume mounting not configured - will copy reports manually"
    VOLUME_MOUNT_WORKS=false
fi

echo ""
echo -e "${BLUE}Starting test suite...${NC}"
echo ""

# Run the tests with volume mount attempt
# Even if volume mount fails, tests will still run and reports will be in container
if docker compose --profile test run --name "$CONTAINER_NAME" --rm test-runner poetry run python tests/run_v1_tests_opensource.py; then
    TEST_EXIT_CODE=0
    echo -e "${GREEN}✓ Tests completed${NC}"
else
    TEST_EXIT_CODE=$?
    echo -e "${YELLOW}⚠ Tests completed with some failures (exit code: $TEST_EXIT_CODE)${NC}"
fi

echo ""

# Check if reports are already on host (volume mount worked)
LATEST_REPORT=$(ls -t "$REPORTS_DIR"/v1_endpoints_opensource_report_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    REPORT_TIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST_REPORT" 2>/dev/null || stat -c "%y" "$LATEST_REPORT" 2>/dev/null | cut -d'.' -f1)
    CURRENT_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Check if report was created in the last 2 minutes (indicates volume mount worked)
    if [ "$REPORT_TIME" \> "$(date -v-2M "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d '2 minutes ago' "+%Y-%m-%d %H:%M:%S" 2>/dev/null)" ]; then
        echo -e "${GREEN}✓${NC} Test reports are already available in: ${BLUE}$REPORTS_DIR${NC}"
        echo ""
        echo -e "${GREEN}Latest reports:${NC}"
        ls -lh "$REPORTS_DIR"/*$(date +%Y%m%d)* 2>/dev/null || echo "  (Reports from previous run)"
    fi
else
    # Volume mount didn't work - try to copy reports from container
    # Since we used --rm, container is already removed, so we need a different approach
    echo -e "${YELLOW}⚠${NC}  Volume mount didn't work. Re-running test to capture reports..."
    echo -e "${BLUE}→${NC} Tip: To avoid this, configure Docker file sharing: Docker Desktop → Settings → Resources → File Sharing"
    echo ""
    
    # Run again without --rm so we can copy files out
    docker compose --profile test run --name "$CONTAINER_NAME" test-runner poetry run python tests/run_v1_tests_opensource.py >/dev/null 2>&1 || true
    
    # Copy reports from container
    if docker cp "$CONTAINER_NAME:/app/tests/test_reports/." "$REPORTS_DIR/" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Test reports copied to: ${BLUE}$REPORTS_DIR${NC}"
        echo ""
        echo -e "${GREEN}Latest reports:${NC}"
        ls -lh "$REPORTS_DIR"/*$(date +%Y%m%d)* 2>/dev/null || ls -lh "$REPORTS_DIR" | tail -5
    else
        echo -e "${YELLOW}⚠${NC}  Could not copy reports from container"
    fi
    
    # Clean up container
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Test Reports Location:${NC}"
echo -e "  📁 ${BLUE}$REPORTS_DIR/${NC}"
echo ""
echo -e "${GREEN}View Results:${NC}"
echo -e "  📄 cat $REPORTS_DIR/v1_endpoints_opensource_log_*.txt | tail -50"
echo -e "  📊 open $REPORTS_DIR/v1_endpoints_opensource_report_*.json"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

exit $TEST_EXIT_CODE
