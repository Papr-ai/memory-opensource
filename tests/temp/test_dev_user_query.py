import asyncio
import httpx
import json
from os import environ as env
from dotenv import load_dotenv, find_dotenv

# Load environment
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

PARSE_SERVER_URL = env.get("PARSE_SERVER_URL")
PARSE_APPLICATION_ID = env.get("PARSE_APPLICATION_ID")
PARSE_MASTER_KEY = env.get("PARSE_MASTER_KEY")

async def test_query():
    url = f"{PARSE_SERVER_URL}/parse/classes/DeveloperUser"

    # Test 1: Query with both developer pointer and external_id
    params1 = {
        "where": json.dumps({
            "developer": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": "jtKplF3Gft"
            },
            "external_id": {"$in": ["security_user_456"]}
        }),
        "limit": 1,
        "include": "user"
    }

    # Test 2: Query with just external_id
    params2 = {
        "where": json.dumps({"external_id": "security_user_456"}),
        "include": "user,developer"
    }

    # Test 3: Query with just developer pointer
    params3 = {
        "where": json.dumps({
            "developer": {
                "__type": "Pointer",
                "className": "_User",
                "objectId": "jtKplF3Gft"
            }
        }),
        "include": "user"
    }

    headers = {
        "X-Parse-Application-Id": PARSE_APPLICATION_ID,
        "X-Parse-Master-Key": PARSE_MASTER_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        print("\n=== Test 1: Query with developer pointer + external_id ===")
        print(f"URL: {url}")
        print(f"Params: {json.dumps(params1, indent=2)}")
        resp1 = await client.get(url, headers=headers, params=params1)
        print(f"Status: {resp1.status_code}")
        print(f"Response: {json.dumps(resp1.json(), indent=2)}")

        print("\n=== Test 2: Query with just external_id ===")
        print(f"Params: {json.dumps(params2, indent=2)}")
        resp2 = await client.get(url, headers=headers, params=params2)
        print(f"Status: {resp2.status_code}")
        print(f"Response: {json.dumps(resp2.json(), indent=2)}")

        print("\n=== Test 3: Query with just developer pointer ===")
        print(f"Params: {json.dumps(params3, indent=2)}")
        resp3 = await client.get(url, headers=headers, params=params3)
        print(f"Status: {resp3.status_code}")
        data = resp3.json()
        print(f"Found {len(data.get('results', []))} records")
        for record in data.get('results', [])[:5]:  # Show first 5
            print(f"  - objectId: {record.get('objectId')}, external_id: {record.get('external_id')}")

if __name__ == "__main__":
    asyncio.run(test_query())
