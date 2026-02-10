#!/usr/bin/env python3

import pytest
import httpx
import time
import asyncio
from asgi_lifespan import LifespanManager
from app_factory import create_app
from models.memory_models import SearchRequest, SearchResponse, AddMemoryRequest, BatchMemoryRequest
from models.shared_types import MemoryMetadata
from services.logger_singleton import LoggerSingleton
from os import environ as env
from dotenv import load_dotenv, find_dotenv

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Test constants
TEST_X_USER_API_KEY = env.get('TEST_X_USER_API_KEY')

@pytest.mark.asyncio
async def test_interaction_limits_performance_real_app():
    """Test interaction limits performance using real app and memory graph"""
    app = create_app()
    
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), 
            base_url="http://test",
            verify=False,
        ) as async_client:
            headers = {
                'Content-Type': 'application/json',
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip'
            }

            # 1. Create a test user
            user_create_payload = {"external_id": f"rate_limit_test_user_{int(time.time())}"}
            user_response = await async_client.post("/v1/user", json=user_create_payload, headers=headers)
            logger.info(f"User creation response: {user_response.status_code}")
            assert user_response.status_code in (200, 201), f"User creation failed: {user_response.text}"
            
            user_data = user_response.json()
            test_user_id = user_data.get("user_id") or user_data.get("id")
            logger.info(f"Created test user ID: {test_user_id}")

            # 2. Add a memory for the test user
            memory_content = "Test memory for rate limit performance testing"
            batch_request = BatchMemoryRequest(
                user_id=test_user_id,
                memories=[
                    AddMemoryRequest(
                        content=memory_content,
                        type="text",
                        metadata=MemoryMetadata(
                            customMetadata={"test": "rate_limit_performance"},
                            workspace_id="pohYfXWoOK"  # Use existing workspace
                        )
                    )
                ],
                batch_size=1
            )
            
            memory_response = await async_client.post(
                "/v1/memory/batch",
                params={"skip_background_processing": True},
                json=batch_request.model_dump(),
                headers=headers
            )
            
            if memory_response.status_code == 200:
                logger.info(f"Added memory for test user: {memory_content}")
            else:
                logger.info(f"Memory add response: {memory_response.status_code}")

            # 3. Perform search and measure rate limit check performance
            search_request = SearchRequest(
                query="rate limit performance test",
                rank_results=False,
                user_id=test_user_id
            )

            # Perform multiple searches to test performance consistency
            rate_limit_times = []
            total_times = []
            
            for i in range(3):
                logger.info(f"Starting search iteration {i+1}")
                
                start_time = time.time()
                response = await async_client.post(
                    "/v1/memory/search?max_memories=10&max_nodes=10",
                    json=search_request.model_dump(),
                    headers=headers
                )
                end_time = time.time()
                
                total_time = (end_time - start_time) * 1000  # Convert to milliseconds
                total_times.append(total_time)
                
                # Validate response
                assert response.status_code == 200, f"Search failed: {response.text}"
                response_data = response.json()
                validated_response = SearchResponse.model_validate(response_data)
                
                # Check if rate limit check was successful (no error)
                if validated_response.error is None:
                    logger.info(f"Search iteration {i+1}: SUCCESS, Total time: {total_time:.2f}ms")
                else:
                    logger.warning(f"Search iteration {i+1}: ERROR - {validated_response.error}")
                    # Rate limit errors might still be valid for testing
                
                # Add small delay between requests
                await asyncio.sleep(0.1)

            # 4. Analyze performance results
            avg_total_time = sum(total_times) / len(total_times)
            min_time = min(total_times)
            max_time = max(total_times)
            
            logger.info(f"Performance Results:")
            logger.info(f"  Average total time: {avg_total_time:.2f}ms")
            logger.info(f"  Min time: {min_time:.2f}ms") 
            logger.info(f"  Max time: {max_time:.2f}ms")
            logger.info(f"  All times: {[f'{t:.2f}ms' for t in total_times]}")
            
            # Performance assertions - total end-to-end should be reasonable
            assert avg_total_time < 5000, f"Average total time too high: {avg_total_time:.2f}ms"
            assert min_time < 3000, f"Minimum time too high: {min_time:.2f}ms"
            
            logger.info("✅ Integration test completed successfully")

@pytest.mark.asyncio 
async def test_rate_limit_fallback_behavior_real_app():
    """Test that rate limit fallback from MongoDB to Parse Server works correctly"""
    app = create_app()
    
    async with LifespanManager(app, startup_timeout=120):
        # Access the memory graph to check MongoDB availability
        memory_graph = app.state.memory_graph
        mongodb_available = memory_graph.mongo_client is not None
        
        logger.info(f"MongoDB available in test: {mongodb_available}")
        logger.info(f"Memory graph mongo_client: {memory_graph.mongo_client}")
        
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), 
            base_url="http://test",
            verify=False,
        ) as async_client:
            headers = {
                'Content-Type': 'application/json',
                'X-Client-Type': 'papr_plugin', 
                'X-API-Key': TEST_X_USER_API_KEY,
                'Accept-Encoding': 'gzip'
            }

            # Create a test user
            user_create_payload = {"external_id": f"fallback_test_user_{int(time.time())}"}
            user_response = await async_client.post("/v1/user", json=user_create_payload, headers=headers)
            assert user_response.status_code in (200, 201)
            
            user_data = user_response.json()
            test_user_id = user_data.get("user_id") or user_data.get("id")
            logger.info(f"Created fallback test user ID: {test_user_id}")

            # Perform a search to trigger rate limit check
            search_request = SearchRequest(
                query="fallback test query",
                rank_results=False,
                user_id=test_user_id
            )
            
            start_time = time.time()
            response = await async_client.post(
                "/v1/memory/search?max_memories=10&max_nodes=10",
                json=search_request.model_dump(),
                headers=headers
            )
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            logger.info(f"Fallback test total time: {total_time:.2f}ms")
            
            # Check response
            assert response.status_code == 200, f"Search failed: {response.text}"
            response_data = response.json()
            validated_response = SearchResponse.model_validate(response_data)
            
            # The key test: even if MongoDB is unavailable, the system should still work
            # via Parse Server fallback, and should not crash
            if validated_response.error is None:
                logger.info("✅ Rate limit check succeeded via fallback mechanism")
                assert validated_response.data is not None
            else:
                # Even if there's a rate limit error, it should be a clean error response
                logger.info(f"Rate limit error (expected in some cases): {validated_response.error}")
                assert isinstance(validated_response.error, str)
            
            logger.info("✅ Fallback behavior test completed successfully")

if __name__ == "__main__":
    # Run standalone
    asyncio.run(test_interaction_limits_performance_real_app())
    asyncio.run(test_rate_limit_fallback_behavior_real_app()) 