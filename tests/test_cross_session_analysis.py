#!/usr/bin/env python3
"""
Test to verify cross-session message processing

This test validates the critical user experience:
"If a user shares a preference in one session then goes to another session,
we need to make sure his preferences are captured and used in the second session"

Test Flow:
1. Create < 15 messages in Session 1 (e.g., 10 messages with user preferences)
2. Verify messages are marked as "pending" (not analyzed yet - batch not triggered)
3. Start Session 2 by creating a message
4. Verify Session 1 messages are analyzed when Session 2 starts
5. Search for memories to confirm preferences from Session 1 were captured
"""
import asyncio
import httpx
import time
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_API_KEY = "f80c5a2940f21882420b41690522cb2c"
TEST_SESSION_TOKEN = "r:578db0db09b3159b7ec98e0043b2af9a"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": TEST_API_KEY,
    "Authorization": f"Bearer {TEST_SESSION_TOKEN}"
}


async def test_cross_session_analysis():
    """Test that preferences from session 1 are captured when session 2 starts"""

    print("=" * 80)
    print("Testing Cross-Session Message Processing")
    print("=" * 80)

    # Generate unique session IDs
    timestamp = int(time.time())
    session_1_id = f"cross_session_test_1_{timestamp}"
    session_2_id = f"cross_session_test_2_{timestamp}"

    print(f"\n📝 Session 1 ID: {session_1_id}")
    print(f"📝 Session 2 ID: {session_2_id}")

    # ========================================================================
    # STEP 1: Create 10 messages in Session 1 (below the 15 message threshold)
    # ========================================================================
    print(f"\n🔄 Step 1: Creating 10 preference messages in Session 1...")
    print("   (Below 15 message threshold - batch analysis should NOT trigger)")

    session_1_messages = [
        {"content": [{"type": "text", "text": "I prefer dark mode in all my applications"}], "role": "user"},
        {"content": [{"type": "text", "text": "I'll help you set your dark mode preferences"}], "role": "assistant"},
        {"content": [{"type": "text", "text": "My favorite programming language is Python"}], "role": "user"},
        {"content": [{"type": "text", "text": "Great choice! Python is excellent for many use cases"}], "role": "assistant"},
        {"content": [{"type": "text", "text": "I work as a machine learning engineer at TechCorp"}], "role": "user"},
        {"content": [{"type": "text", "text": "That's exciting! ML engineering is a growing field"}], "role": "assistant"},
        {"content": [{"type": "text", "text": "I have team standup meetings every day at 9am"}], "role": "user"},
        {"content": [{"type": "text", "text": "I'll note that for your daily schedule"}], "role": "assistant"},
        {"content": [{"type": "text", "text": "I'm allergic to peanuts and shellfish"}], "role": "user"},
        {"content": [{"type": "text", "text": "Important to remember for meal planning"}], "role": "assistant"}
    ]

    session_1_message_ids = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        for i, msg_data in enumerate(session_1_messages, 1):
            message_payload = {
                "content": msg_data["content"],
                "role": msg_data["role"],
                "sessionId": session_1_id,
                "process_messages": True,  # Enable processing
                "metadata": {
                    "topics": ["preferences", "personal"],
                    "test": "cross_session"
                }
            }

            response = await client.post("/v1/messages", headers=HEADERS, json=message_payload)

            if response.status_code == 200:
                result = response.json()
                message_id = result.get("objectId")
                session_1_message_ids.append(message_id)
                content_preview = msg_data["content"][0]["text"][:50]
                print(f"   ✅ Message {i}/10: [{msg_data['role']}] {content_preview}...")
            else:
                print(f"   ❌ Failed to create message {i}: {response.status_code} - {response.text}")
                return False

        print(f"\n✅ Created {len(session_1_message_ids)} messages in Session 1")

        # ========================================================================
        # STEP 2: Verify Session 1 messages are pending (not analyzed yet)
        # ========================================================================
        print(f"\n🔍 Step 2: Checking Session 1 status (should be pending)...")
        await asyncio.sleep(2)  # Brief wait for status updates

        status_response = await client.get(
            f"/v1/messages/sessions/{session_1_id}/status",
            headers=HEADERS
        )

        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Session 1 Status:")
            print(f"   - Total messages: {status.get('message_count', 0)}")
            print(f"   - Processing summary: {status.get('processing_summary', {})}")

            processing_summary = status.get("processing_summary", {})
            pending_count = processing_summary.get("pending", 0) + processing_summary.get("queued", 0)

            if pending_count == 10:
                print(f"   ✅ All 10 messages are pending (batch not triggered yet)")
            else:
                print(f"   ⚠️  Expected 10 pending messages, found {pending_count}")
        else:
            print(f"   ❌ Failed to get session status: {status_response.status_code}")

        # ========================================================================
        # STEP 3: Start Session 2 by creating a message
        # ========================================================================
        print(f"\n🚀 Step 3: Starting Session 2 (this should trigger Session 1 processing)...")
        print("   Creating first message in Session 2...")

        session_2_message_payload = {
            "content": [{"type": "text", "text": "Hello! This is a new conversation session"}],
            "role": "user",
            "sessionId": session_2_id,
            "process_messages": True,
            "metadata": {
                "topics": ["general"],
                "test": "cross_session"
            }
        }

        session_2_response = await client.post("/v1/messages", headers=HEADERS, json=session_2_message_payload)

        if session_2_response.status_code == 200:
            session_2_msg = session_2_response.json()
            print(f"   ✅ Session 2 started with message: {session_2_msg.get('objectId')}")
        else:
            print(f"   ❌ Failed to start Session 2: {session_2_response.status_code}")
            return False

        # ========================================================================
        # STEP 4: Wait for cross-session processing to complete
        # ========================================================================
        print(f"\n⏳ Step 4: Waiting for cross-session processing...")
        print("   The pipeline should detect Session 1 has <15 unprocessed messages")
        print("   and process them when Session 2 starts...")

        max_wait_time = 90  # 1.5 minutes
        start_time = time.time()
        processing_complete = False

        while time.time() - start_time < max_wait_time:
            await asyncio.sleep(5)  # Check every 5 seconds

            # Check Session 1 status
            status_response = await client.get(
                f"/v1/messages/sessions/{session_1_id}/status",
                headers=HEADERS
            )

            if status_response.status_code == 200:
                status = status_response.json()
                processing_summary = status.get("processing_summary", {})

                pending = processing_summary.get("pending", 0)
                queued = processing_summary.get("queued", 0)
                analyzing = processing_summary.get("analyzing", 0)
                completed = processing_summary.get("completed", 0)
                failed = processing_summary.get("failed", 0)

                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Session 1 Status: pending={pending}, queued={queued}, analyzing={analyzing}, completed={completed}, failed={failed}")

                # Check if processing is complete
                if pending + queued + analyzing == 0 and (completed + failed) > 0:
                    processing_complete = True
                    print(f"\n   ✅ Session 1 processing complete! ({completed} completed, {failed} failed)")
                    break
            else:
                print(f"   ⚠️  Failed to get status: {status_response.status_code}")

        if not processing_complete:
            print(f"\n   ⚠️  Processing did not complete within {max_wait_time}s")
            print("   This may indicate an issue with cross-session processing")

        # ========================================================================
        # STEP 5: Search for memories from Session 1 preferences
        # ========================================================================
        print(f"\n🔍 Step 5: Searching for memories created from Session 1 preferences...")

        # Search for various preferences we mentioned in Session 1
        search_queries = [
            "dark mode preferences",
            "Python programming language",
            "machine learning engineer",
            "peanuts allergies"
        ]

        total_memories_found = 0
        memories_by_query = {}

        for query in search_queries:
            search_response = await client.post(
                "/v1/memory/search",
                headers=HEADERS,
                json={
                    "query": query,
                    "limit": 5
                }
            )

            if search_response.status_code == 200:
                search_result = search_response.json()
                memories = search_result.get("data", [])

                if memories and isinstance(memories, list):
                    memories_by_query[query] = len(memories)
                    total_memories_found += len(memories)
                    print(f"\n   ✅ Query '{query}': Found {len(memories)} memories")
                    for mem in memories[:2]:  # Show first 2
                        memory_text = mem.get("memory", "")[:80]
                        memory_id = mem.get("objectId", "unknown")
                        source_type = mem.get("sourceType", "unknown")
                        category = mem.get("category", "unknown")
                        print(f"      - [{memory_id}] {memory_text}...")
                        print(f"        sourceType: {source_type}, category: {category}")
                else:
                    memories_by_query[query] = 0
                    print(f"   ⚠️  Query '{query}': No memories found")
            else:
                print(f"   ❌ Search failed for '{query}': {search_response.status_code}")

        # ========================================================================
        # STEP 6: Final verification and verdict
        # ========================================================================
        print(f"\n{'=' * 80}")
        print("📊 Test Results Summary")
        print(f"{'=' * 80}")
        print(f"Session 1 Messages Created: {len(session_1_message_ids)}")
        print(f"Session 2 Started: ✅")
        print(f"Cross-Session Processing Complete: {'✅ Yes' if processing_complete else '⚠️  Unknown'}")
        print(f"Total Memories Found via Search: {total_memories_found}")
        print(f"\nMemories by Query:")
        for query, count in memories_by_query.items():
            print(f"  - '{query}': {count} memories")

        # Verify expected behavior
        success = True
        issues = []

        if not processing_complete:
            success = False
            issues.append("Cross-session processing did not complete")

        if total_memories_found == 0:
            success = False
            issues.append("No memories found from Session 1 preferences")
        elif total_memories_found < 4:  # We expect at least some memories from 10 messages
            issues.append(f"Only {total_memories_found} memories found - expected more from 10 preference messages")

        if success:
            print(f"\n✅ SUCCESS: Cross-session analysis works correctly!")
            print(f"   Session 1 preferences were captured when Session 2 started")
            print(f"   Found {total_memories_found} searchable memories from Session 1")
            return True
        else:
            print(f"\n❌ ISSUES DETECTED:")
            for issue in issues:
                print(f"   - {issue}")
            print(f"\n   This test validates: 'If a user shares a preference in one session")
            print(f"   then goes to another session, preferences should be captured'")
            return False


if __name__ == "__main__":
    result = asyncio.run(test_cross_session_analysis())
    exit(0 if result else 1)
