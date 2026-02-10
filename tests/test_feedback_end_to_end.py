import pytest
import httpx
from asgi_lifespan import LifespanManager
from main import app
from models.memory_models import SearchRequest, SearchResponse
from models.feedback_models import FeedbackRequest, FeedbackResponse, FeedbackData, ParsePointer
from os import environ as env
from dotenv import load_dotenv, find_dotenv
import asyncio
import time
from models.shared_types import FeedbackType, FeedbackSource
from services.memory_management import get_query_log_by_id_async


# Load environment variables
ENV_FILE = find_dotenv()
load_dotenv(ENV_FILE)

TEST_X_USER_API_KEY = env.get('TEST_X_USER_API_KEY')

async def wait_for_query_log(search_id: str, timeout_seconds: int = 30, poll_interval: float = 2.0) -> dict:
    """Poll Parse for QueryLog creation so feedback validation won't 404."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        query_log = await get_query_log_by_id_async(search_id)
        if query_log:
            return query_log
        await asyncio.sleep(poll_interval)
    raise AssertionError(f"QueryLog {search_id} not found within {timeout_seconds}s")

@pytest.mark.asyncio
async def test_feedback_end_to_end():
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

            # 1. Run a search
            
            
            search_request = SearchRequest(
                query="Show me my most recent memories",
                rank_results=True
            )
            search_resp = await async_client.post(
                "/v1/memory/search",
                json=search_request.model_dump(),
                headers=headers
            )
            assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
            search_body = search_resp.json()
            validated_search = SearchResponse.model_validate(search_body)
            assert validated_search.status == "success"
            assert validated_search.search_id is not None
            search_id = validated_search.search_id

            # Wait for the background QueryLog creation to complete
            await wait_for_query_log(search_id)

            # 2. Submit feedback for this search
            # Create FeedbackData object
            feedback_data = FeedbackData(
                feedbackType=FeedbackType.THUMBS_UP,
                feedbackValue="helpful",
                feedbackScore=1.0,
                feedbackText="This was helpful!",
                feedbackSource=FeedbackSource.INLINE,
                citedMemoryIds=[],
                citedNodeIds=[],
                feedbackProcessed=False,
                feedbackImpact=None
            )
            
            feedback_req = FeedbackRequest(
                search_id=search_id,
                feedbackData=feedback_data,
                user_id=None,  # Will be resolved from auth
                external_user_id=None
            )
            feedback_resp = await async_client.post(
                "/v1/feedback",
                json=feedback_req.model_dump(),
                headers=headers
            )
            assert feedback_resp.status_code == 200, f"Feedback failed: {feedback_resp.text}"
            feedback_body = feedback_resp.json()
            validated_feedback = FeedbackResponse.model_validate(feedback_body)
            assert validated_feedback.status == "success"
            assert validated_feedback.feedback_id is not None
            assert validated_feedback.message.lower().startswith("feedback submitted")

            # 3. (Optional) If you have an endpoint to fetch feedback by ID, fetch and validate it here
            # Otherwise, this confirms end-to-end creation and validation 

@pytest.mark.asyncio
async def test_get_feedback_by_id_v1():
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

            # 1. Run a search first to get a search_id  
            search_request = SearchRequest(
                query="Show me my most recent memories",
                enable_agentic_graph=False,
                rank_results=False
            )
            search_resp = await async_client.post(
                "/v1/memory/search",
                json=search_request.model_dump(),
                headers=headers
            )
            assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
            search_body = search_resp.json()
            validated_search = SearchResponse.model_validate(search_body)
            assert validated_search.status == "success"
            assert validated_search.search_id is not None
            search_id = validated_search.search_id

            # Wait for the background QueryLog creation to complete
            await wait_for_query_log(search_id)

            # 2. Submit feedback for this search
            feedback_data = FeedbackData(
                feedbackType=FeedbackType.THUMBS_UP,
                feedbackValue="helpful",
                feedbackScore=1.0,
                feedbackText="This was very helpful for my testing!",
                feedbackSource=FeedbackSource.INLINE,
                citedMemoryIds=[],
                citedNodeIds=[],
                feedbackProcessed=False,
                feedbackImpact=None
            )
            
            feedback_req = FeedbackRequest(
                search_id=search_id,
                feedbackData=feedback_data,
                user_id=None,  # Will be resolved from auth
                external_user_id=None
            )
            feedback_resp = await async_client.post(
                "/v1/feedback",
                json=feedback_req.model_dump(),
                headers=headers
            )
            assert feedback_resp.status_code == 200, f"Feedback failed: {feedback_resp.text}"
            feedback_body = feedback_resp.json()
            validated_feedback = FeedbackResponse.model_validate(feedback_body)
            assert validated_feedback.status == "success"
            assert validated_feedback.feedback_id is not None
            feedback_id = validated_feedback.feedback_id

            # 3. Poll for the feedback to be stored (background task completion)
            # Retry up to 15 times with 2 second delays (30 seconds total)
            max_attempts = 15
            get_feedback_resp = None
            retrieved_feedback = None
            
            for attempt in range(max_attempts):
                await asyncio.sleep(2)  # Wait 2 seconds between attempts
                
                get_feedback_resp = await async_client.get(
                    f"/v1/feedback/{feedback_id}",
                    headers=headers
                )
                
                if get_feedback_resp.status_code == 200:
                    get_feedback_body = get_feedback_resp.json()
                    retrieved_feedback = FeedbackResponse.model_validate(get_feedback_body)
                    if retrieved_feedback.status == "success":
                        break  # Successfully retrieved
                
                # If we get a 404, the background task might not have completed yet
                if get_feedback_resp.status_code == 404 and attempt < max_attempts - 1:
                    continue  # Try again
                elif get_feedback_resp.status_code != 200:
                    # Some other error occurred
                    break
            
            # Assert that we successfully retrieved the feedback
            assert get_feedback_resp is not None, "Failed to get feedback response"
            assert get_feedback_resp.status_code == 200, f"Get feedback failed after {max_attempts} attempts: {get_feedback_resp.text}"
            assert retrieved_feedback is not None, "Failed to parse feedback response"
            
            # 4. Validate the retrieved feedback
            assert retrieved_feedback.status == "success"
            assert retrieved_feedback.feedback_id == feedback_id
            assert retrieved_feedback.error is None
            assert "feedback retrieved successfully" in retrieved_feedback.message.lower() 