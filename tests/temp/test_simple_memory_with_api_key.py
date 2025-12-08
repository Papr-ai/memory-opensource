import asyncio
import httpx
import json
from os import environ as env
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

# Load environment
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

API_KEY = env.get("PAPR_API_KEY")
PARSE_SERVER_URL = env.get("PARSE_SERVER_URL")

async def test_memory_with_api_key():
    url = f"{PARSE_SERVER_URL}/parse/v1/memory"

    # Test without external_user_id - just using API key authentication
    payload = {
        "content": "Security policy test without external_user_id",
        "type": "text",
        "metadata": {
            "topics": ["security", "policy"],
            "createdAt": datetime.now().isoformat() + "Z",
            "location": "test",
            "emoji_tags": ["🛡️"],
            "conversationId": "test_session_1",
            "customMetadata": {
                "source": "security_test",
                "category": "test"
            }
            # NOTE: No external_user_id here
        },
        "graph_override": {
            "nodes": [
                {
                    "label": "Conversation",
                    "properties": {
                        "conversation_id": "test_session_1",
                        "topic": "Security Policy"
                    }
                },
                {
                    "label": "Speaker",
                    "properties": {
                        "speaker_id": "security_admin",
                        "role": "admin"
                    }
                }
            ],
            "relationships": [
                {
                    "type": "PARTICIPATES_IN",
                    "start_node": {"label": "Speaker", "properties": {"speaker_id": "security_admin"}},
                    "end_node": {"label": "Conversation", "properties": {"conversation_id": "test_session_1"}},
                    "properties": {}
                }
            ]
        }
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print("=== Test: Add Memory with API Key (no external_user_id) ===")
    print(f"URL: {url}")
    print(f"Headers: X-API-Key: {API_KEY[:20]}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        print(f"\nStatus: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")

        if resp.status_code == 200 or resp.status_code == 201:
            print("\n✅ SUCCESS: Memory created without external_user_id!")
            return True
        else:
            print("\n❌ FAILED: Memory creation failed")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_with_api_key())
    exit(0 if success else 1)
