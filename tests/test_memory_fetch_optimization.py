import asyncio
import time
import json
from typing import List
import pytest
from fastapi.testclient import TestClient
from app_factory import create_app
from memory.memory_graph import MemoryGraph
from models.memory_models import ParseStoredMemory
import logging

# Set up logging to capture detailed output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMemoryFetchOptimization:
    """Test suite to compare original vs optimized memory fetch methods"""
    
    async def setup_test_environment(self):
        """Set up test environment with real app and memory graph"""
        app = create_app()
        client = TestClient(app)
        
        # Get the memory graph instance directly - it's created during app startup
        # but may not be in app.state with TestClient
        from memory.memory_graph import MemoryGraph
        memory_graph = MemoryGraph()
        
        # Test data - using real memory IDs from our test environment
        test_memory_ids = [
            "b5ea40ae-5b36-42de-978d-96587277db4a_0",
            "b5ea40ae-5b36-42de-978d-96587277db4a"
        ]
        
        test_session_token = "YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd"  # From our test
        test_user_id = "Gs3yb8f851"  # From our test
        
        return {
            'memory_graph': memory_graph,
            'memory_ids': test_memory_ids,
            'session_token': test_session_token,
            'user_id': test_user_id,
            'client': client
        }
    
    async def test_fetch_methods_comparison(self):
        """Compare original vs optimized fetch methods"""
        test_env = await self.setup_test_environment()
        memory_graph = test_env['memory_graph']
        memory_ids = test_env['memory_ids']
        session_token = test_env['session_token']
        user_id = test_env['user_id']
        
        # Ensure connections are ready
        await memory_graph.ensure_async_connection()
        
        # Get a Neo4j session for the original method
        async with memory_graph.async_neo_conn.get_session() as neo_session:
            # Test original method
            logger.info("Testing original fetch_memory_items_from_sources_async...")
            start_time = time.time()
            
            original_results = await memory_graph.fetch_memory_items_from_sources_async(
                session_token=session_token,
                memory_item_ids=memory_ids,
                user_id=user_id,
                neo_session=neo_session,
                api_key=session_token
            )
            
            original_duration = time.time() - start_time
            logger.info(f"Original method took: {original_duration:.3f}s")
            logger.info(f"Original method returned {len(original_results)} items")
            
        # Test optimized method (no Neo4j session needed)
        logger.info("Testing optimized fetch_memory_items_from_sources_async_fast...")
        start_time = time.time()
        
        optimized_results = await memory_graph.fetch_memory_items_from_sources_async_fast(
            session_token=session_token,
            memory_item_ids=memory_ids,
            user_id=user_id,
            api_key=session_token
        )
        
        optimized_duration = time.time() - start_time
        logger.info(f"Optimized method took: {optimized_duration:.3f}s")
        logger.info(f"Optimized method returned {len(optimized_results)} items")
        
        # Compare results
        assert len(original_results) == len(optimized_results), \
            f"Result count mismatch: original={len(original_results)}, optimized={len(optimized_results)}"
        
        # Compare memory IDs (the most important part)
        original_ids = set(item.memoryId for item in original_results)
        optimized_ids = set(item.memoryId for item in optimized_results)
        
        assert original_ids == optimized_ids, \
            f"Memory ID mismatch: original={original_ids}, optimized={optimized_ids}"
        
        # Compare content (should be identical)
        for orig_item, opt_item in zip(
            sorted(original_results, key=lambda x: x.memoryId),
            sorted(optimized_results, key=lambda x: x.memoryId)
        ):
            assert orig_item.memoryId == opt_item.memoryId
            assert orig_item.content == opt_item.content
            assert orig_item.user.objectId == opt_item.user.objectId
            logger.info(f"✅ Memory {orig_item.memoryId} content matches")
        
        # Calculate performance improvement
        improvement_percent = ((original_duration - optimized_duration) / original_duration) * 100
        logger.info(f"🚀 Performance improvement: {improvement_percent:.1f}% faster")
        logger.info(f"   Original: {original_duration:.3f}s")
        logger.info(f"   Optimized: {optimized_duration:.3f}s")
        logger.info(f"   Time saved: {original_duration - optimized_duration:.3f}s")
        
        # Performance should be better (or at least not significantly worse)
        assert optimized_duration <= original_duration * 1.1, \
            f"Optimized method is slower: {optimized_duration:.3f}s vs {original_duration:.3f}s"
        
        return {
            'original_duration': original_duration,
            'optimized_duration': optimized_duration,
            'improvement_percent': improvement_percent,
            'original_count': len(original_results),
            'optimized_count': len(optimized_results),
            'results_match': True
        }
    
    async def test_multiple_iterations(self):
        """Run multiple iterations to get average performance"""
        test_env = await self.setup_test_environment()
        memory_graph = test_env['memory_graph']
        memory_ids = test_env['memory_ids']
        session_token = test_env['session_token']
        user_id = test_env['user_id']
        
        iterations = 3
        original_times = []
        optimized_times = []
        
        await memory_graph.ensure_async_connection()
        
        for i in range(iterations):
            logger.info(f"Iteration {i + 1}/{iterations}")
            
            # Original method with Neo4j session
            async with memory_graph.async_neo_conn.get_session() as neo_session:
                start_time = time.time()
                original_results = await memory_graph.fetch_memory_items_from_sources_async(
                    session_token=session_token,
                    memory_item_ids=memory_ids,
                    user_id=user_id,
                    neo_session=neo_session,
                    api_key=session_token
                )
                original_times.append(time.time() - start_time)
            
            # Optimized method (no Neo4j session needed)
            start_time = time.time()
            optimized_results = await memory_graph.fetch_memory_items_from_sources_async_fast(
                session_token=session_token,
                memory_item_ids=memory_ids,
                user_id=user_id,
                api_key=session_token
            )
            optimized_times.append(time.time() - start_time)
            
            # Verify results are consistent
            assert len(original_results) == len(optimized_results)
        
        # Calculate averages
        avg_original = sum(original_times) / len(original_times)
        avg_optimized = sum(optimized_times) / len(optimized_times)
        avg_improvement = ((avg_original - avg_optimized) / avg_original) * 100
        
        logger.info(f"📊 Average Performance (over {iterations} iterations):")
        logger.info(f"   Original: {avg_original:.3f}s")
        logger.info(f"   Optimized: {avg_optimized:.3f}s")
        logger.info(f"   Average improvement: {avg_improvement:.1f}%")
        
        return {
            'avg_original': avg_original,
            'avg_optimized': avg_optimized,
            'avg_improvement': avg_improvement,
            'iterations': iterations
        }


# Simple test runner function
async def run_memory_fetch_tests():
    """Simple test runner to compare memory fetch methods"""
    test_instance = TestMemoryFetchOptimization()
    
    logger.info("🧪 Starting Memory Fetch Optimization Tests")
    logger.info("=" * 60)
    
    try:
        # Single test
        logger.info("Running single comparison test...")
        single_result = await test_instance.test_fetch_methods_comparison()
        
        logger.info("✅ Single test completed successfully!")
        
        # Multiple iterations test
        logger.info("\nRunning multiple iterations test...")
        multi_result = await test_instance.test_multiple_iterations()
        
        logger.info("✅ Multiple iterations test completed successfully!")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🏆 FINAL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Results are identical between methods")
        logger.info(f"🚀 Average performance improvement: {multi_result['avg_improvement']:.1f}%")
        logger.info(f"⏱️  Average time saved: {multi_result['avg_original'] - multi_result['avg_optimized']:.3f}s")
        logger.info(f"🎯 Optimization successful - Neo4j removal achieved the goal!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the tests
    result = asyncio.run(run_memory_fetch_tests())
    if result:
        print("\n🎉 All tests passed! The optimization is working correctly.")
    else:
        print("\n❌ Tests failed. Check the logs above for details.") 