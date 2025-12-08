import pytest
import asyncio
from models.parse_server import QueryLog, UserFeedbackLog, MemoryRetrievalLog, ParsePointer
from services.memory_management import (
    store_memory_query_log_async,
    store_user_feedback_log_async,
    store_memory_retrieval_log_async,
    get_all_query_logs,
    get_feedback_logs_for_query,
    get_memory_retrieval_log_for_query,
    update_memory_retrieval_log,
)
from scripts.update_feedback_analytics import update_cited_memory_confidence_scores

@pytest.mark.asyncio
async def test_feedback_analytics_end_to_end():
    # 1. Create a QueryLog
    query_log = QueryLog(
        user=ParsePointer(objectId="user1", className="_User"),
        workspace=ParsePointer(objectId="ws1", className="WorkSpace"),
        queryText="test query"
    )
    qlog_result = await store_memory_query_log_async(query_log)
    query_log_id = qlog_result["objectId"]

    # 2. Create a MemoryRetrievalLog
    retrieval_log = MemoryRetrievalLog(
        user=ParsePointer(objectId="user1", className="_User"),
        workspace=ParsePointer(objectId="ws1", className="WorkSpace"),
        queryLog=ParsePointer(objectId=query_log_id, className="QueryLog"),
        retrievedMemories=[],
        citedMemories=[],
    )
    rlog_result = await store_memory_retrieval_log_async(retrieval_log)
    retrieval_log_id = rlog_result["objectId"]

    # 3. Create UserFeedbackLogs with different feedback types and cited memories
    feedbacks = [
        UserFeedbackLog(
            queryLog=ParsePointer(objectId=query_log_id, className="QueryLog"),
            user=ParsePointer(objectId="user1", className="_User"),
            workspace=ParsePointer(objectId="ws1", className="WorkSpace"),
            feedbackType="thumbs_up",
            feedbackSource="inline",
            citedMemoryIds=["memA", "memB"]
        ),
        UserFeedbackLog(
            queryLog=ParsePointer(objectId=query_log_id, className="QueryLog"),
            user=ParsePointer(objectId="user1", className="_User"),
            workspace=ParsePointer(objectId="ws1", className="WorkSpace"),
            feedbackType="copy_action",
            feedbackSource="inline",
            citedMemoryIds=["memA"]
        ),
        UserFeedbackLog(
            queryLog=ParsePointer(objectId=query_log_id, className="QueryLog"),
            user=ParsePointer(objectId="user1", className="_User"),
            workspace=ParsePointer(objectId="ws1", className="WorkSpace"),
            feedbackType="rating",
            feedbackScore=5.0,
            feedbackSource="inline",
            citedMemoryIds=["memB"]
        ),
    ]
    print(f"Creating feedback logs for query_log_id: {query_log_id}")
    for i, fb in enumerate(feedbacks):
        result = await store_user_feedback_log_async(fb)
        print(f"Created feedback log {i+1}: {result}")

    # 4. Run the analytics update
    print(f"Running analytics update for query_log_id: {query_log_id}")
    
    # Process only our specific QueryLog
    await update_cited_memory_confidence_scores(query_log_id)

    # 5. Fetch the updated MemoryRetrievalLog and check citedMemoryConfidenceScores
    updated_log = await get_memory_retrieval_log_for_query(query_log_id)
    scores = updated_log.citedMemoryConfidenceScores
    # memA: thumbs_up (1) + copy_action (2) = 3
    # memB: thumbs_up (1) + rating (1) = 2
    # total = 5
    assert abs(scores["memA"] - 0.6) < 0.01
    assert abs(scores["memB"] - 0.4) < 0.01 