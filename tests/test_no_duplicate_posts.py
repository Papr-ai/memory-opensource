"""
Test to verify that document uploads no longer create duplicate Posts.

After our fix, document processing should create only ONE Post with:
- type: "batch_memories" (merged)
- uploadId: <upload_id>
- extractionResultFile: <file>
- batch_memories_file: <file>

Instead of the old behavior (TWO Posts):
1. type: "page" (document)
2. type: "batch_memories" (standalone batch)
"""

import pytest
import httpx
import json
from datetime import datetime, UTC
from typing import List, Dict, Any


@pytest.mark.anyio
async def test_document_upload_creates_single_post(
    test_client,
    test_api_key,
    sample_pdf_file
):
    """
    Test that document upload creates only ONE Post (not two).
    
    Expected behavior:
    - Document processing creates a Post with type="batch_memories"
    - That SAME Post contains both document fields and batch memory fields
    - No separate "page" Post is created
    """
    # Upload document
    response = await test_client.post(
        "/v1/document",
        files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
        headers={"X-API-Key": test_api_key, "X-Client-Type": "test"}
    )
    
    assert response.status_code == 200
    result = response.json()
    
    upload_id = result.get("document_status", {}).get("upload_id")
    assert upload_id, "Upload ID should be returned"
    
    # Wait for processing to complete (simplified - in real test, poll status)
    import asyncio
    await asyncio.sleep(30)  # Adjust based on your processing time
    
    # Query Parse for Posts related to this upload
    parse_url = f"{PARSE_SERVER_URL}/parse/classes/Post"
    where_clause = {"uploadId": upload_id}
    params = {
        "where": json.dumps(where_clause),
        "keys": "objectId,type,uploadId,extractionResultFile,batch_memories_file,batchMetadata"
    }
    
    async with httpx.AsyncClient() as client:
        parse_response = await client.get(
            parse_url,
            params=params,
            headers={
                "X-Parse-Application-Id": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY
            }
        )
        
        assert parse_response.status_code == 200
        posts = parse_response.json().get("results", [])
    
    # ASSERTION 1: Only ONE Post should exist for this upload
    assert len(posts) == 1, f"Expected 1 Post, found {len(posts)}: {posts}"
    
    post = posts[0]
    
    # ASSERTION 2: Post should be type="batch_memories" (merged)
    assert post.get("type") == "batch_memories", \
        f"Expected type='batch_memories', got '{post.get('type')}'"
    
    # ASSERTION 3: Post should have document fields
    assert post.get("uploadId") == upload_id, "Post should have uploadId"
    assert post.get("extractionResultFile") is not None, \
        "Post should have extractionResultFile (document field)"
    
    # ASSERTION 4: Post should have batch memory fields
    assert post.get("batch_memories_file") is not None, \
        "Post should have batch_memories_file (batch field)"
    assert post.get("batchMetadata") is not None, \
        "Post should have batchMetadata (batch field)"
    
    print("✅ SUCCESS: Document upload created only ONE merged Post")
    print(f"   Post ID: {post.get('objectId')}")
    print(f"   Type: {post.get('type')}")
    print(f"   Upload ID: {post.get('uploadId')}")
    print(f"   Has document fields: ✅")
    print(f"   Has batch fields: ✅")


@pytest.mark.anyio
async def test_no_standalone_batch_posts_created(
    test_client,
    test_api_key,
    sample_pdf_file
):
    """
    Test that no standalone batch_memories Posts are created.
    
    Before our fix:
    - Post A: type="page", has extractionResultFile, no batch_memories_file
    - Post B: type="batch_memories", has batch_memories_file, no extractionResultFile
    
    After our fix:
    - Post A: type="batch_memories", has BOTH extractionResultFile AND batch_memories_file
    - Post B: DOES NOT EXIST
    """
    # Upload document
    response = await test_client.post(
        "/v1/document",
        files={"file": ("test.pdf", sample_pdf_file, "application/pdf")},
        headers={"X-API-Key": test_api_key, "X-Client-Type": "test"}
    )
    
    assert response.status_code == 200
    result = response.json()
    upload_id = result.get("document_status", {}).get("upload_id")
    
    # Wait for processing
    import asyncio
    await asyncio.sleep(30)
    
    # Query for standalone batch_memories Posts (no uploadId, no extractionResultFile)
    parse_url = f"{PARSE_SERVER_URL}/parse/classes/Post"
    where_clause = {
        "type": "batch_memories",
        "uploadId": {"$exists": False},
        "extractionResultFile": {"$exists": False},
        "createdAt": {"$gte": {"__type": "Date", "iso": datetime.now(UTC).isoformat()}}
    }
    params = {
        "where": json.dumps(where_clause),
        "limit": 100
    }
    
    async with httpx.AsyncClient() as client:
        parse_response = await client.get(
            parse_url,
            params=params,
            headers={
                "X-Parse-Application-ID": PARSE_APPLICATION_ID,
                "X-Parse-Master-Key": PARSE_MASTER_KEY
            }
        )
        
        standalone_batch_posts = parse_response.json().get("results", [])
    
    # ASSERTION: No standalone batch_memories Posts should be created
    assert len(standalone_batch_posts) == 0, \
        f"Found {len(standalone_batch_posts)} standalone batch_memories Posts (should be 0)"
    
    print("✅ SUCCESS: No standalone batch_memories Posts created")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-xvs"])

