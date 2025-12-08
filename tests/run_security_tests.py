#!/usr/bin/env python3
"""
Simple runner for Security Schema Tests only

This script runs only the security schema tests from the v1 test suite.
It properly initializes the app lifespan before running tests.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root and tests directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

async def main_security_only():
    """Run only security schema tests with proper lifespan initialization."""
    from main import app
    import httpx
    from httpx import ASGITransport, AsyncClient

    # Import test functions
    from tests.test_security_schema_v1 import (
        test_v1_create_security_schema,
        test_v1_add_memory_with_schema_id,
        test_v1_wait_for_memory_processing,
        test_v1_search_verify_neo4j_nodes,
        test_v1_search_with_agentic_graph,
        test_v1_add_memory_with_graph_override,
        test_v1_security_schema_full_workflow,
    )

    print("🚀 Starting Security Schema Tests Only")
    print("=" * 60)

    tests = [
        ("Create Security Schema", test_v1_create_security_schema),
        ("Add Memory with schema_id", test_v1_add_memory_with_schema_id),
        ("Wait for Memory Processing", test_v1_wait_for_memory_processing),
        ("Search Verify Neo4j Nodes", test_v1_search_verify_neo4j_nodes),
        ("Search with Agentic Graph", test_v1_search_with_agentic_graph),
        ("Add Memory with graph_override", test_v1_add_memory_with_graph_override),
        ("Full Workflow Validation", test_v1_security_schema_full_workflow),
    ]

    passed = 0
    failed = 0
    start_time = time.time()

    # Use async context manager to trigger lifespan events
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
        # The lifespan is now initialized! We can run tests

        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            try:
                # Call the test function with the app
                await test_func(app)
                print(f"✅ Test passed: {test_name}")
                passed += 1
            except Exception as e:
                print(f"❌ Test failed: {test_name}")
                print(f"   Error: {str(e)}")
                failed += 1

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Duration: {duration:.2f}s")
    print(f"{'='*60}")

    return failed == 0

if __name__ == "__main__":
    try:
        success = asyncio.run(main_security_only())
        if success:
            print("\n✅ All security schema tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
