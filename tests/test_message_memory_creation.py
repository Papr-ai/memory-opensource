#!/usr/bin/env python3
"""
Test to verify that message processing actually creates memories

This test:
1. Creates 15+ messages to trigger batch processing
2. Waits for processing to complete
3. Searches for memories to confirm they were created
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

async def test_message_to_memory_creation():
    """Test that messages actually create memories"""

    print("="*80)
    print("Testing Message → Memory Creation")
    print("="*80)

    # Generate unique session ID
    test_session_id = f"memory_test_{int(time.time())}"
    print(f"\n📝 Test Session ID: {test_session_id}")

    # Step 1: Create 15 messages to trigger batch processing
    print(f"\n🔄 Step 1: Creating 15 messages (to trigger batch processing)...")

    message_contents = [
        "I prefer dark mode in all my applications",
        "My favorite programming language is Python",
        "I work as a data scientist at a tech company",
        "I have a meeting every Monday at 10am with the product team",
        "I'm learning about machine learning and neural networks",
        "My email is john.doe@example.com",
        "I live in San Francisco, California",
        "I'm allergic to peanuts and shellfish",
        "I graduated from Stanford University with a CS degree",
        "I speak English, Spanish, and basic Mandarin",
        "I play guitar and piano in my free time",
        "I'm working on a side project about climate data analysis",
        "I prefer React over Vue for frontend development",
        "I have two cats named Luna and Sol",
        "I'm training for a marathon in October"
    ]

    stored_message_ids = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        for i, content in enumerate(message_contents, 1):
            message_data = {
                "content": content,
                "role": "user",
                "sessionId": test_session_id,
                "process_messages": True,
                "metadata": {
                    "topics": ["personal", "preferences"],
                    "test": True
                }
            }

            response = await client.post("/v1/messages", headers=HEADERS, json=message_data)

            if response.status_code == 200:
                result = response.json()
                message_id = result.get("objectId")
                stored_message_ids.append(message_id)
                print(f"   ✅ Message {i}/15: {content[:50]}... (ID: {message_id})")
            else:
                print(f"   ❌ Failed to create message {i}: {response.status_code} - {response.text}")
                return False

        print(f"\n✅ Successfully stored {len(stored_message_ids)} messages")

        # Step 2: Wait for background processing to complete
        print(f"\n⏳ Step 2: Waiting for background processing (batch analysis)...")
        print("   This may take 30-60 seconds as the batch processor analyzes 15 messages...")

        # Poll the session status endpoint
        max_wait_time = 120  # 2 minutes max
        start_time = time.time()
        processing_complete = False

        while time.time() - start_time < max_wait_time:
            await asyncio.sleep(5)  # Check every 5 seconds

            status_response = await client.get(
                f"/v1/messages/sessions/{test_session_id}/status",
                headers=HEADERS
            )

            if status_response.status_code == 200:
                status = status_response.json()
                status_breakdown = status.get("status_breakdown", {})

                queued = status_breakdown.get("queued", 0)
                analyzing = status_breakdown.get("analyzing", 0)
                completed = status_breakdown.get("completed", 0)
                failed = status_breakdown.get("failed", 0)

                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Status: queued={queued}, analyzing={analyzing}, completed={completed}, failed={failed}")

                if queued + analyzing == 0 and (completed + failed) > 0:
                    processing_complete = True
                    print(f"\n   ✅ Processing complete! ({completed} completed, {failed} failed)")
                    break
            else:
                print(f"   ⚠️  Failed to get status: {status_response.status_code}")

        if not processing_complete:
            print(f"\n   ⚠️  Processing did not complete within {max_wait_time}s")
            print("   This might be expected - batch processing only triggers at 15+ messages")

        # Step 3: Search for memories that should have been created
        print(f"\n🔍 Step 3: Searching for memories created from messages...")

        # Search with various keywords from our messages
        search_queries = [
            "dark mode preferences",
            "Python programming",
            "data scientist",
            "machine learning"
        ]

        total_memories_found = 0

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
                    print(f"\n   ✅ Query '{query}': Found {len(memories)} memories")
                    for mem in memories[:2]:  # Show first 2
                        memory_text = mem.get("memory", "")[:80]
                        memory_id = mem.get("objectId", "unknown")
                        print(f"      - [{memory_id}] {memory_text}...")
                    total_memories_found += len(memories)
                else:
                    print(f"   ⚠️  Query '{query}': No memories found")
            else:
                print(f"   ❌ Search failed for '{query}': {search_response.status_code}")

        # Step 4: Final verdict
        print(f"\n{'='*80}")
        print("📊 Test Results Summary")
        print(f"{'='*80}")
        print(f"Messages Created: {len(stored_message_ids)}")
        print(f"Processing Complete: {'✅ Yes' if processing_complete else '⚠️  Unknown (may still be processing)'}")
        print(f"Memories Found via Search: {total_memories_found}")

        if total_memories_found > 0:
            print(f"\n✅ SUCCESS: Message analysis created searchable memories!")
            print(f"   Found {total_memories_found} memories from message content")
            return True
        else:
            print(f"\n❌ ISSUE: No memories found via search")
            print("   Possible reasons:")
            print("   1. Batch processing requires 15+ messages (we created 15)")
            print("   2. Processing may still be running (check logs)")
            print("   3. Messages may not be 'memory-worthy' per AI analysis")
            print("   4. There may be an issue in the processing pipeline")
            return False

if __name__ == "__main__":
    result = asyncio.run(test_message_to_memory_creation())
    exit(0 if result else 1)
