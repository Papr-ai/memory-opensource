#!/usr/bin/env python3
"""
Delete All Memories Test Runner

This script runs the delete_all_memories tests individually for development and debugging.

Usage:
    python tests/run_delete_all_memories_tests.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_delete_all_memories import (
    test_delete_all_memories_complete_workflow,
    test_delete_all_memories_with_external_user_id,
    test_delete_all_memories_no_memories_found
)
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

async def run_all_tests():
    """Run all delete_all_memories tests."""
    tests = [
        ("Complete Workflow Test", test_delete_all_memories_complete_workflow),
        ("External User ID Test", test_delete_all_memories_with_external_user_id),
        ("No Memories Found Test", test_delete_all_memories_no_memories_found)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            await test_func()
            end_time = asyncio.get_event_loop().time()
            
            result = {
                'name': test_name,
                'status': 'PASSED',
                'duration': end_time - start_time,
                'error': None
            }
            logger.info(f"✅ {test_name} PASSED in {result['duration']:.2f}s")
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            result = {
                'name': test_name,
                'status': 'FAILED',
                'duration': end_time - start_time,
                'error': str(e)
            }
            logger.error(f"❌ {test_name} FAILED: {e}")
            logger.error(f"   Duration: {result['duration']:.2f}s")
        
        results.append(result)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = len([r for r in results if r['status'] == 'PASSED'])
    failed = len([r for r in results if r['status'] == 'FAILED'])
    total_time = sum(r['duration'] for r in results)
    
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total Time: {total_time:.2f}s")
    
    for result in results:
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        logger.info(f"{status_icon} {result['name']}: {result['status']} ({result['duration']:.2f}s)")
        if result['error']:
            logger.info(f"   Error: {result['error']}")
    
    if failed > 0:
        logger.error(f"\n❌ {failed} test(s) failed!")
        sys.exit(1)
    else:
        logger.info(f"\n🎉 All {passed} tests passed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 