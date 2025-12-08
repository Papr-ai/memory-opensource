#!/usr/bin/env python3
"""
Diagnostic script to check processingStatus field storage and retrieval
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

# Parse Server configuration
PARSE_SERVER_URL = os.getenv("PARSE_SERVER_URL") or os.getenv("PARSE_SERVER_URL")
PARSE_APPLICATION_ID = os.getenv("PARSE_APPLICATION_ID")
PARSE_MASTER_KEY = os.getenv("PARSE_MASTER_KEY")

if PARSE_SERVER_URL and not PARSE_SERVER_URL.startswith(('http://', 'https://')):
    PARSE_SERVER_URL = f"https://{PARSE_SERVER_URL}"

HEADERS = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-Master-Key": PARSE_MASTER_KEY,
    "Content-Type": "application/json"
}

# Get test credentials from environment variables
TEST_API_KEY = os.getenv("TEST_X_USER_API_KEY")
TEST_SESSION_TOKEN = os.getenv("TEST_SESSION_TOKEN")

if not TEST_API_KEY:
    raise ValueError("TEST_X_USER_API_KEY environment variable is required")
if not TEST_SESSION_TOKEN:
    raise ValueError("TEST_SESSION_TOKEN environment variable is required")

API_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": TEST_API_KEY,
    "Authorization": f"Bearer {TEST_SESSION_TOKEN}"
}


async def diagnose_processing_status():
    """Check if processingStatus is being stored and retrieved correctly"""

    print("="*80)
    print("Diagnosing processingStatus Field")
    print("="*80)

    async with httpx.AsyncClient(timeout=60.0) as client:

        # Step 1: Create a test message via the API
        print("\n1. Creating a test message via /v1/messages endpoint...")

        test_session_id = f"diagnostic_test_{int(asyncio.get_event_loop().time())}"
        message_data = {
            "content": [{"type": "text", "text": "This is a diagnostic test message"}],
            "role": "user",
            "sessionId": test_session_id,
            "process_messages": True,
            "metadata": {"topics": ["test"]}
        }

        response = await client.post(
            "http://localhost:8000/v1/messages",
            headers=API_HEADERS,
            json=message_data
        )

        if response.status_code == 200:
            result = response.json()
            message_id = result.get("objectId")
            print(f"   ✅ Message created: {message_id}")
            print(f"   Response: {result}")
        else:
            print(f"   ❌ Failed to create message: {response.status_code}")
            print(f"   Error: {response.text}")
            return

        # Step 2: Query Parse Server directly for this message
        print(f"\n2. Querying Parse Server directly for message {message_id}...")

        parse_response = await client.get(
            f"{PARSE_SERVER_URL}/parse/classes/PostMessage/{message_id}",
            headers=HEADERS
        )

        if parse_response.status_code == 200:
            parse_data = parse_response.json()
            print(f"   ✅ Retrieved from Parse Server")
            print(f"   Fields present: {list(parse_data.keys())}")
            print(f"   processingStatus: {parse_data.get('processingStatus', 'FIELD NOT PRESENT')}")
            print(f"   messageRole: {parse_data.get('messageRole', 'FIELD NOT PRESENT')}")
            print(f"   message: {parse_data.get('message', 'FIELD NOT PRESENT')[:50]}...")
        else:
            print(f"   ❌ Failed to query Parse: {parse_response.status_code}")
            print(f"   Error: {parse_response.text}")
            return

        # Step 3: Query via the /v1/messages/sessions/{sessionId} endpoint
        print(f"\n3. Querying via /v1/messages/sessions/{test_session_id} endpoint...")

        session_response = await client.get(
            f"http://localhost:8000/v1/messages/sessions/{test_session_id}",
            headers=API_HEADERS
        )

        if session_response.status_code == 200:
            session_data = session_response.json()
            messages = session_data.get("messages", [])
            print(f"   ✅ Retrieved {len(messages)} messages from session")

            if messages:
                msg = messages[0]
                print(f"   Message data: {msg}")
                print(f"   processing_status: {msg.get('processing_status', 'FIELD NOT PRESENT')}")
        else:
            print(f"   ❌ Failed to query session: {session_response.status_code}")
            print(f"   Error: {session_response.text}")

        # Step 4: Query session status
        print(f"\n4. Checking session status endpoint...")

        status_response = await client.get(
            f"http://localhost:8000/v1/messages/sessions/{test_session_id}/status",
            headers=API_HEADERS
        )

        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"   ✅ Session status: {status_data}")
        else:
            print(f"   ❌ Failed to get status: {status_response.status_code}")

        # Step 5: Query Parse Server with keys parameter (like our code does)
        print(f"\n5. Querying Parse Server with keys parameter (mimicking our code)...")

        import json
        query_params = {
            "where": json.dumps({"objectId": message_id}),
            "keys": "objectId,message,content,messageRole,createdAt,processingStatus",
            "limit": 1
        }

        keys_response = await client.get(
            f"{PARSE_SERVER_URL}/parse/classes/PostMessage",
            headers=HEADERS,
            params=query_params
        )

        if keys_response.status_code == 200:
            keys_data = keys_response.json()
            results = keys_data.get("results", [])
            print(f"   ✅ Query with keys returned {len(results)} results")
            if results:
                msg = results[0]
                print(f"   Fields returned: {list(msg.keys())}")
                print(f"   processingStatus: {msg.get('processingStatus', 'FIELD NOT PRESENT')}")
        else:
            print(f"   ❌ Failed: {keys_response.status_code}")

        print(f"\n{'='*80}")
        print("Diagnosis Complete")
        print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(diagnose_processing_status())
