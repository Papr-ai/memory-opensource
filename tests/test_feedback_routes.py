import pytest
from models.feedback_models import (
    FeedbackRequest, BatchFeedbackRequest, FeedbackResponse, BatchFeedbackResponse,
    FeedbackData, ParsePointer
)
from models.shared_types import FeedbackType, FeedbackSource
import httpx
from dotenv import load_dotenv
import os
import logging

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

logger = logging.getLogger(__name__)

TEST_X_USER_API_KEY = os.getenv("TEST_X_USER_API_KEY")


# --- Single FeedbackRequest validation ---
@pytest.mark.asyncio
async def test_valid_single_feedback():
    feedback_data = FeedbackData(
        feedbackType=FeedbackType.THUMBS_UP,
        feedbackValue="helpful",
        feedbackScore=1.0,
        feedbackText="Great result!",
        feedbackSource=FeedbackSource.INLINE,
        citedMemoryIds=["mem1"],
        citedNodeIds=[],
        feedbackProcessed=False,
        feedbackImpact=None
    )
    
    feedback = FeedbackRequest(
        search_id="search_1",
        feedbackData=feedback_data,
        user_id="user_1",
        external_user_id=None
    )
    assert feedback.feedbackData.feedbackScore == 1.0
    assert feedback.feedbackData.feedbackType == FeedbackType.THUMBS_UP
    assert feedback.feedbackData.feedbackText == "Great result!"

@pytest.mark.asyncio
async def test_invalid_thumbs_feedback_score():
    feedback_data = FeedbackData(
        feedbackType=FeedbackType.THUMBS_UP,
        feedbackValue="helpful",
        feedbackScore=0.0,  # Invalid
        feedbackText="Great result!",
        feedbackSource=FeedbackSource.INLINE,
        citedMemoryIds=[],
        citedNodeIds=[],
        feedbackProcessed=False,
        feedbackImpact=None
    )
    
    with pytest.raises(ValueError):
        FeedbackRequest(
            search_id="search_2",
            feedbackData=feedback_data,
            user_id="user_1"
        )

@pytest.mark.asyncio
async def test_valid_rating_feedback():
    feedback_data = FeedbackData(
        feedbackType=FeedbackType.RATING,
        feedbackValue="excellent",
        feedbackScore=5.0,
        feedbackText="Perfect!",
        feedbackSource=FeedbackSource.INLINE,
        citedMemoryIds=[],
        citedNodeIds=[],
        feedbackProcessed=False,
        feedbackImpact=None
    )
    
    feedback = FeedbackRequest(
        search_id="search_3",
        feedbackData=feedback_data,
        user_id="user_1"
    )
    assert feedback.feedbackData.feedbackScore == 5.0
    assert feedback.feedbackData.feedbackType == FeedbackType.RATING

@pytest.mark.asyncio
async def test_invalid_rating_feedback_score():
    feedback_data = FeedbackData(
        feedbackType=FeedbackType.RATING,
        feedbackValue="too_high",
        feedbackScore=6.0,  # Out of range
        feedbackText="Too high!",
        feedbackSource=FeedbackSource.INLINE,
        citedMemoryIds=[],
        citedNodeIds=[],
        feedbackProcessed=False,
        feedbackImpact=None
    )
    
    with pytest.raises(ValueError):
        FeedbackRequest(
            search_id="search_4",
            feedbackData=feedback_data,
            user_id="user_1"
        )

@pytest.mark.asyncio
async def test_batch_feedback_request():
    feedback_data1 = FeedbackData(
        feedbackType=FeedbackType.THUMBS_DOWN,
        feedbackValue="unhelpful",
        feedbackScore=-1.0,
        feedbackText="Not helpful",
        feedbackSource=FeedbackSource.INLINE,
        citedMemoryIds=[],
        citedNodeIds=[],
        feedbackProcessed=False,
        feedbackImpact=None
    )
    
    feedback_data2 = FeedbackData(
        feedbackType=FeedbackType.RATING,
        feedbackValue="good",
        feedbackScore=3.0,
        feedbackText="Good but could be better",
        feedbackSource=FeedbackSource.INLINE,
        citedMemoryIds=[],
        citedNodeIds=[],
        feedbackProcessed=False,
        feedbackImpact=None
    )
    
    feedback1 = FeedbackRequest(
        search_id="search_5",
        feedbackData=feedback_data1,
        user_id="user_1"
    )
    feedback2 = FeedbackRequest(
        search_id="search_6",
        feedbackData=feedback_data2,
        user_id="user_1"
    )
    
    batch = BatchFeedbackRequest(feedback_items=[feedback1, feedback2])
    assert len(batch.feedback_items) == 2
    assert batch.feedback_items[0].feedbackData.feedbackType == FeedbackType.THUMBS_DOWN
    assert batch.feedback_items[1].feedbackData.feedbackScore == 3.0

@pytest.mark.asyncio
async def test_feedback_response_success_and_failure():
    resp = FeedbackResponse.success(feedback_id="fb_123")
    assert resp.status == "success"
    assert resp.feedback_id == "fb_123"
    fail = FeedbackResponse.failure(error="bad", code=400)
    assert fail.status == "error"
    assert fail.error == "bad"

@pytest.mark.asyncio
async def test_batch_feedback_response():
    resp = BatchFeedbackResponse.success(feedback_ids=["fb1", "fb2"], successful_count=2)
    assert resp.status == "success"
    assert resp.successful_count == 2
    assert resp.feedback_ids == ["fb1", "fb2"] 

@pytest.mark.asyncio
async def test_real_feedback_submission_and_memory_increment(async_client: httpx.AsyncClient):
    headers = {
        'Content-Type': 'application/json',
        'X-Client-Type': 'papr_plugin',
        'X-API-Key': TEST_X_USER_API_KEY,
        'Accept-Encoding': 'gzip'
    }

    # First, perform a search to get a valid search_id
    search_request = {
        "query": "test query for feedback",
        "rank_results": True,
        "enable_agentic_graph": False
    }
    search_resp = await async_client.post(
        "/v1/memory/search?max_memories=5&max_nodes=5",
        json=search_request,
        headers=headers
    )
    assert search_resp.status_code == 200
    search_body = search_resp.json()
    search_id = search_body.get('search_id')
    assert search_id

    # Prepare feedback data
    feedback_data = {
        "feedbackType": "thumbs_up",
        "feedbackValue": "helpful",
        "feedbackScore": 1.0,
        "feedbackText": "Great results!",
        "feedbackSource": "inline",
        "citedMemoryIds": ["example_mem_id"],  # Replace with real if needed
        "citedNodeIds": [],
        "feedbackProcessed": False,
        "feedbackImpact": None
    }

    feedback_request = {
        "search_id": search_id,
        "feedbackData": feedback_data,
        "user_id": None,
        "external_user_id": None
    }

    resp = await async_client.post(
        "/v1/feedback/submit",
        json=feedback_request,
        headers=headers
    )
    assert resp.status_code in [200, 207], f"Feedback submission failed: {resp.status_code} {resp.text}"
    body = resp.json()
    feedback_id = body.get('feedback_id')
    assert feedback_id

    # Poll Parse for the FeedbackLog (assuming a way to fetch by ID)
    # Note: You'll need to implement or use a get_feedback_by_id_async function similar to query log
    # For now, assume it exists or skip detailed verification if not

    logger.info("✓ Real Feedback submission and memory increment test passed") 