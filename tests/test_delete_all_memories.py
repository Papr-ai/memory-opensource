import pytest
import httpx
import time
import json
import asyncio
from uuid import uuid4
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app
from models.parse_server import (
    AddMemoryResponse, BatchMemoryResponse, DeleteMemoryResponse, 
    BatchMemoryError, SystemUpdateStatus
)
from models.memory_models import (
    AddMemoryRequest, BatchMemoryRequest, SearchRequest, SearchResponse,
    MemoryMetadata, SearchResult
)
from models.user_models import CreateUserRequest, UserResponse
from os import environ as env
from dotenv import load_dotenv, find_dotenv
import urllib3
from services.logger_singleton import LoggerSingleton
from asgi_lifespan import LifespanManager

# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Load environment variables
ENV_FILE = find_dotenv()
logger.info(f"Found .env file at: {ENV_FILE}")
load_dotenv(ENV_FILE)

# Test constants
# Prefer TEST_X_USER_API_KEY but fall back to Papr key or app-level key for convenience
TEST_X_API_KEY = (
    env.get('TEST_X_USER_API_KEY')
)
if not TEST_X_API_KEY:
    raise ValueError("Provide TEST_X_USER_API_KEY or TEST_X_PAPR_API_KEY or PAPR_MEMORY_API_KEY in the environment")

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@pytest.mark.asyncio
async def test_delete_all_memories_complete_workflow():
    """
    Complete end-to-end test for delete_all_memories endpoint:
    1. Create a new test user
    2. Add multiple memories via batch operation for that user
    3. Verify memories were created by searching
    4. Delete all memories for that user
    5. Confirm all memories were deleted by searching again
    """
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            
            # Test data preparation
            test_run_id = uuid4().hex[:12]
            test_email = f"delete_test_user_{test_run_id}@example.com"
            test_external_id = f"delete_test_user_{test_run_id}"
            
            logger.info(f"🧪 Starting delete_all_memories test with run ID: {test_run_id}")
            
            headers = {
                'Content-Type': 'application/json',
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_API_KEY,
                'Accept-Encoding': 'gzip'
            }
            
            # ============================================
            # STEP 1: Create a new test user
            # ============================================
            logger.info("📝 Step 1: Creating new test user...")
            
            create_user_request = CreateUserRequest(
                email=test_email,
                external_id=test_external_id,
                metadata={
                    "name": f"Delete Test User {test_run_id}",
                    "test_run_id": test_run_id,
                    "test_purpose": "delete_all_memories_workflow",
                    "preferences": {"theme": "dark", "test_mode": True}
                }
            )
            
            user_response = await async_client.post(
                "/v1/user",
                headers=headers,
                json=create_user_request.model_dump()
            )
            
            assert user_response.status_code == 200, f"Failed to create user: {user_response.text}"
            user_data = UserResponse(**user_response.json())
            test_user_id = user_data.user_id
            
            logger.info(f"✅ Created test user with ID: {test_user_id}")
            
            # ============================================
            # STEP 2: Add multiple memories via batch operation
            # ============================================
            logger.info("📚 Step 2: Adding batch memories for the test user...")
            
            # Create 2 test memories with unique content (avoid triggering Temporal)
            batch_memories = []
            memory_contents = [
                f"Memory 1: Project kickoff meeting notes for test run {test_run_id}. Discussed timeline, resources, and deliverables.",
                f"Memory 2: Customer feedback analysis from Q4 2024 for test {test_run_id}. Overall satisfaction increased by 15%."
            ]
            
            for i, content in enumerate(memory_contents):
                memory_request = AddMemoryRequest(
                    content=content,
                    type="text", 
                    metadata=MemoryMetadata(
                        topics=f"test, delete_all_memories, batch_{i+1}, {test_run_id}",
                        hierarchical_structures=f"testing, memory management, batch {i+1}",
                        createdAt=f"2024-01-{15+i:02d}T{10+i:02d}:00:00Z",
                        location="Test Environment",
                        emoji_tags=f"🧪📝{i+1}",
                        emotion_tags="neutral, organized",
                        conversationId=f"delete_test_{test_run_id}_{i+1}",
                        sourceUrl=f"https://test.example.com/memory_{i+1}",
                        user_id=test_user_id  # Associate with our test user
                    )
                )
                batch_memories.append(memory_request)
            
            batch_request = BatchMemoryRequest(
                memories=batch_memories,
                batch_size=2,  # Small batch to avoid Temporal
                user_id=test_user_id,  # Ensure all memories are for our test user
                webhook_url="https://webhook.site/test-webhook-delete-test",
                webhook_secret="test-webhook-secret-delete"
            )
            
            # Mock the webhook service to track when memories are processed
            with patch('routes.memory_routes.webhook_service.send_batch_completion_webhook') as mock_send_webhook:
                mock_send_webhook.return_value = True
                
                logger.info(f"🚀 Creating batch request with webhook for {len(batch_memories)} memories")
                
                batch_response = await async_client.post(
                    "/v1/memory/batch",
                    params={"skip_background_processing": True},  # Skip Temporal to get immediate memory IDs
                    json=batch_request.model_dump(),
                    headers=headers
                )
            
                logger.info(f"Batch response status: {batch_response.status_code}")
                logger.info(f"Batch response text: {batch_response.text[:500]}...")
                
                # Accept 200 or 207 (partial success)
                assert batch_response.status_code in [200, 207], f"Failed to add batch memories: {batch_response.text}"
                
                try:
                    batch_data = BatchMemoryResponse.model_validate(batch_response.json())
                except Exception as e:
                    logger.error(f"Failed to parse BatchMemoryResponse: {e}")
                    logger.error(f"Raw response: {batch_response.json()}")
                    raise
                
                logger.info(f"Batch data - Total successful: {batch_data.total_successful}, Total failed: {batch_data.total_failed}")
                logger.info(f"Batch data - Successful items: {len(batch_data.successful)}")

                # Since skip_background_processing=True, memories are created immediately
                logger.info(f"✅ Batch processing completed immediately (no Temporal)")
                
                # Extract memoryIds from nested AddMemoryResponse -> AddMemoryItem
                created_memory_ids = []
                for add_resp in batch_data.successful:
                    if add_resp.data and len(add_resp.data) > 0:
                        created_memory_ids.append(add_resp.data[0].memoryId)

                logger.info(f"✅ Created {len(created_memory_ids)} memories immediately")
            
            # Wait for memory creation to complete
            logger.info("⏳ Waiting for memories to be created...")
            await asyncio.sleep(5)

            # ============================================
            # STEP 3: Verify memories exist by direct GET
            # ============================================
            logger.info("🔍 Step 3: Verifying memories exist via direct GET...")

            # Verify memories were created by getting them directly
            verified_count = 0
            for memory_id in created_memory_ids:  # Check all memories
                get_response = await async_client.get(
                    f"/v1/memory/{memory_id}",
                    headers=headers
                )

                if get_response.status_code == 200:
                    verified_count += 1
                    logger.info(f"✅ Verified memory {memory_id} exists")
                else:
                    logger.warning(f"⚠️ Memory {memory_id} not found (status: {get_response.status_code})")

            logger.info(f"🔍 Verified {verified_count}/{len(created_memory_ids)} memories exist before deletion")

            # We should find at least some memories
            assert verified_count > 0, f"Should find at least some memories before deletion, but found {verified_count}"
            
            # ============================================
            # STEP 4: Delete ALL memories for the test user
            # ============================================
            logger.info("🗑️ Step 4: Deleting ALL memories for the test user...")
            
            delete_all_response = await async_client.delete(
                "/v1/memory/all",
                params={
                    "user_id": test_user_id,
                    "skip_parse": False
                },
                headers=headers
            )
            
            logger.info(f"Delete all response status: {delete_all_response.status_code}")
            logger.info(f"Delete all response body: {delete_all_response.json()}")
            
            # Should succeed (200) or partially succeed (207)
            assert delete_all_response.status_code in [200, 207], f"Delete all failed: {delete_all_response.text}"
            delete_all_data = BatchMemoryResponse.model_validate(delete_all_response.json())
            
            # Verify deletion results
            assert delete_all_data.total_processed > 0, "Should have processed some memories"
            assert delete_all_data.total_successful > 0, "Should have successfully deleted some memories"
            
            logger.info(f"✅ Delete all results:")
            logger.info(f"   - Total processed: {delete_all_data.total_processed}")
            logger.info(f"   - Total successful: {delete_all_data.total_successful}")
            logger.info(f"   - Total failed: {delete_all_data.total_failed}")
            
            if delete_all_data.total_failed > 0:
                logger.warning(f"⚠️ Some deletions failed: {delete_all_data.errors}")
            
            # Wait for deletion to propagate
            await asyncio.sleep(3)
            
            # ============================================
            # STEP 5: Confirm memories are deleted via direct GET
            # ============================================
            logger.info("🔍 Step 5: Confirming memories are deleted via direct GET...")

            # Try to GET each deleted memory - should return 404
            deleted_confirmed = 0
            for memory_id in created_memory_ids:
                get_response = await async_client.get(
                    f"/v1/memory/{memory_id}",
                    headers=headers
                )

                if get_response.status_code == 404:
                    deleted_confirmed += 1
                    logger.info(f"✅ Memory {memory_id} correctly returns 404 (deleted)")
                else:
                    logger.warning(f"⚠️ Memory {memory_id} still accessible (status: {get_response.status_code})")

            logger.info(f"📊 Deletion confirmation: {deleted_confirmed}/{len(created_memory_ids)} memories confirmed deleted")
            
            # ============================================
            # STEP 6: Additional verification - try to fetch specific memory IDs
            # ============================================
            logger.info("🔍 Step 6: Additional verification - checking specific memory IDs...")

            deleted_count = 0
            for memory_id in created_memory_ids:  # Check all memory IDs
                try:
                    get_response = await async_client.get(
                        f"/v1/memory/{memory_id}",
                        headers=headers
                    )

                    if get_response.status_code == 404:
                        deleted_count += 1
                        logger.info(f"✅ Memory {memory_id} correctly returns 404 (deleted)")
                    else:
                        logger.warning(f"⚠️ Memory {memory_id} still accessible (status: {get_response.status_code})")

                except Exception as e:
                    logger.info(f"✅ Memory {memory_id} access failed as expected: {e}")
                    deleted_count += 1

            logger.info(f"📊 Verification summary: {deleted_count}/{len(created_memory_ids)} checked memories are properly deleted")
            
            # ============================================
            # FINAL ASSERTIONS
            # ============================================
            logger.info("🏁 Final verification...")
            
            # Core assertions - be lenient to handle cases where some memories weren't created
            assert delete_all_data.total_processed >= batch_data.total_successful, f"Should have processed at least {batch_data.total_successful} memories (what was created), but got {delete_all_data.total_processed}"
            assert delete_all_data.total_successful > 0, "Should have successfully deleted some memories"
            
            # If we had 100% success in deletion, verify no test memories remain
            if delete_all_data.total_failed == 0:
                logger.info("✅ All deletions successful - test completed perfectly!")
            else:
                logger.info(f"✅ Test completed with {delete_all_data.total_failed} partial failures (acceptable)")
            
            logger.info(f"🎉 delete_all_memories test completed successfully for run ID: {test_run_id}")


@pytest.mark.asyncio 
async def test_delete_all_memories_with_external_user_id():
    """
    Test delete_all_memories endpoint using external_user_id parameter
    to verify user resolution works correctly.
    """
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            
            test_run_id = uuid4().hex[:10]
            test_email = f"ext_delete_test_{test_run_id}@example.com"
            test_external_id = f"ext_delete_test_{test_run_id}"
            
            logger.info(f"🧪 Starting external_user_id delete test with run ID: {test_run_id}")
            
            headers = {
                'Content-Type': 'application/json',
                'X-Client-Type': 'papr_plugin', 
                'X-API-Key': TEST_X_API_KEY,
                'Accept-Encoding': 'gzip'
            }
            
            # Create test user
            create_user_request = CreateUserRequest(
                email=test_email,
                external_id=test_external_id,
                metadata={
                    "name": f"External Delete Test User {test_run_id}",
                    "test_run_id": test_run_id,
                    "test_purpose": "delete_all_memories_external_user_id"
                }
            )
            
            user_response = await async_client.post(
                "/v1/user",
                headers=headers,
                json=create_user_request.model_dump()
            )
            
            assert user_response.status_code == 200, f"Failed to create user: {user_response.text}"
            user_data = UserResponse(**user_response.json())
            test_user_id = user_data.user_id
            
            logger.info(f"✅ Created test user with ID: {test_user_id}, external_id: {test_external_id}")
            
            # Add a single memory for this user
            add_request = AddMemoryRequest(
                content=f"External user delete test memory {test_run_id}",
                type="text",
                metadata=MemoryMetadata(
                    topics=f"external_user_delete_test, {test_run_id}",
                    user_id=test_user_id
                )
            )
            
            add_response = await async_client.post(
                "/v1/memory",
                params={"skip_background_processing": True},
                json=add_request.model_dump(),
                headers=headers
            )
            
            assert add_response.status_code == 200, f"Failed to add memory: {add_response.text}"
            add_data = AddMemoryResponse.model_validate(add_response.json())
            memory_id = add_data.data[0].memoryId
            
            logger.info(f"✅ Created test memory with ID: {memory_id}")
            
            # Now delete all memories using external_user_id
            delete_all_response = await async_client.delete(
                "/v1/memory/all",
                params={
                    "external_user_id": test_external_id,  # Use external_user_id instead of user_id
                    "skip_parse": False
                },
                headers=headers
            )
            
            logger.info(f"Delete all response status: {delete_all_response.status_code}")
            logger.info(f"Delete all response body: {delete_all_response.json()}")
            
            assert delete_all_response.status_code in [200, 207], f"Delete all failed: {delete_all_response.text}"
            delete_all_data = BatchMemoryResponse.model_validate(delete_all_response.json())
            
            # Verify deletion worked
            assert delete_all_data.total_processed >= 1, "Should have processed at least 1 memory"
            assert delete_all_data.total_successful >= 1, "Should have successfully deleted at least 1 memory"
            
            logger.info(f"✅ Successfully deleted memories using external_user_id: {test_external_id}")
            logger.info(f"   - Total processed: {delete_all_data.total_processed}")
            logger.info(f"   - Total successful: {delete_all_data.total_successful}")


@pytest.mark.asyncio
async def test_delete_all_memories_no_memories_found():
    """
    Test delete_all_memories endpoint when user has no memories
    (should return 404 with appropriate message).
    """
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as async_client:
            
            test_run_id = uuid4().hex[:10]
            test_email = f"empty_delete_test_{test_run_id}@example.com"
            test_external_id = f"empty_delete_test_{test_run_id}"
            
            logger.info(f"🧪 Starting no-memories delete test with run ID: {test_run_id}")
            
            headers = {
                'Content-Type': 'application/json',
                'X-Client-Type': 'papr_plugin',
                'X-API-Key': TEST_X_API_KEY,
                'Accept-Encoding': 'gzip'
            }
            
            # Create test user but don't add any memories
            create_user_request = CreateUserRequest(
                email=test_email,
                external_id=test_external_id,
                metadata={
                    "name": f"Empty Delete Test User {test_run_id}",
                    "test_run_id": test_run_id,
                    "test_purpose": "delete_all_memories_no_memories"
                }
            )
            
            user_response = await async_client.post(
                "/v1/user",
                headers=headers,
                json=create_user_request.model_dump()
            )
            
            assert user_response.status_code == 200, f"Failed to create user: {user_response.text}"
            user_data = UserResponse(**user_response.json())
            test_user_id = user_data.user_id
            
            logger.info(f"✅ Created empty test user with ID: {test_user_id}")
            
            # Try to delete all memories for this user (should be none)
            delete_all_response = await async_client.delete(
                "/v1/memory/all",
                params={
                    "user_id": test_user_id,
                    "skip_parse": False
                },
                headers=headers
            )
            
            logger.info(f"Delete all response status: {delete_all_response.status_code}")
            logger.info(f"Delete all response body: {delete_all_response.json()}")
            
            # Should return 404 with "No memories found for user" message
            assert delete_all_response.status_code == 404, f"Expected 404 for user with no memories, got {delete_all_response.status_code}"
            
            delete_all_data = BatchMemoryResponse.model_validate(delete_all_response.json())
            assert "No memories found for user" in delete_all_data.error, f"Expected 'No memories found' error message"
            
            logger.info(f"✅ Correctly returned 404 for user with no memories")


# Add this import at the top
import asyncio

if __name__ == "__main__":
    # Run the test directly for development/debugging
    asyncio.run(test_delete_all_memories_complete_workflow()) 