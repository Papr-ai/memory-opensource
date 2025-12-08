"""
Test document upload with schema_id parameter.

This script:
1. Uploads a document with a schema_id in metadata
2. Verifies the upload succeeds
3. Shows how to check if schema_id was propagated

Note: For this test, we'll use a dummy schema_id to verify propagation.
In production, you'd first create the schema via /v1/schemas.
"""

import asyncio
import httpx
import json
from dotenv import load_dotenv
import os
import sys

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

# Get credentials
API_KEY = os.getenv("TEST_X_USER_API_KEY", "YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd")
BASE_URL = os.getenv("PARSE_SERVER_URL", "http://localhost:8000").replace("/parse", "")

# Use a test schema_id (in real scenario, this would come from creating a schema first)
TEST_SCHEMA_ID = "test-workflow-schema-001"


async def upload_document_with_schema(file_path: str, schema_id: str):
    """Upload a document with schema_id in metadata"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print("=" * 80)
    print("📤 UPLOADING DOCUMENT WITH SCHEMA_ID")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Schema ID: {schema_id}")
    print(f"API Key: {API_KEY[:20]}...")
    print(f"Base URL: {BASE_URL}")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        headers = {
            "X-API-Key": API_KEY,
            "X-Client-Type": "test_client"
        }
        
        # Read file
        with open(file_path, "rb") as f:
            pdf_content = f.read()
        
        files = {"file": (os.path.basename(file_path), pdf_content, "application/pdf")}
        
        # Metadata with schema_id
        metadata = {
            "schema_id": schema_id,
            "source": "test_schema_propagation",
            "test": True
        }
        
        form_data = {
            "metadata": json.dumps(metadata),
            "preferred_provider": "reducto",
            "hierarchical_enabled": "true"
        }
        
        print(f"\n🚀 Sending request...")
        print(f"   Metadata: {json.dumps(metadata, indent=2)}")
        
        response = await client.post(
            f"{BASE_URL}/v1/document",
            files=files,
            data=form_data,
            headers=headers
        )
        
        print(f"\n📡 Response: {response.status_code}")
        
        if response.status_code in [200, 202]:
            result = response.json()
            print(f"✅ Upload successful!")
            print(json.dumps(result, indent=2))
            
            upload_id = result.get("document_status", {}).get("upload_id")
            
            if upload_id:
                print(f"\n📊 Upload ID: {upload_id}")
                print(f"\n🔍 To verify schema_id propagation:")
                print(f"   1. Check Temporal workflow logs for upload_id: {upload_id}")
                print(f"   2. Check Parse Memory records: upload_id={upload_id}")
                print(f"   3. Query Neo4j:")
                print(f"      MATCH (n) WHERE n.upload_id = '{upload_id}'")
                print(f"      RETURN DISTINCT labels(n) as node_types, count(n) as count")
                print(f"\n   Expected (if schema enforcement was implemented):")
                print(f"      - CallSession, Agent, Workflow, Step, Tool")
                print(f"   Actual (current - no enforcement):")
                print(f"      - Any node types (schema_id propagated but not enforced)")
                
                return upload_id
            else:
                print(f"⚠️  No upload_id in response")
                return None
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return None


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test document upload with schema_id parameter"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="/Users/shawkatkabbara/Documents/GitHub/memory/tests/call_answering_sop.pdf",
        help="Path to PDF file to upload"
    )
    parser.add_argument(
        "--schema-id",
        type=str,
        default=TEST_SCHEMA_ID,
        help="Schema ID to use (default: test-workflow-schema-001)"
    )
    
    args = parser.parse_args()
    
    try:
        upload_id = await upload_document_with_schema(args.file, args.schema_id)
        
        if upload_id:
            print(f"\n" + "=" * 80)
            print(f"✅ TEST PASSED: Document uploaded with schema_id")
            print(f"=" * 80)
            print(f"\nNext steps:")
            print(f"1. Wait for workflow to complete (check Temporal UI)")
            print(f"2. Verify schema_id in logs/database")
            print(f"3. Check Neo4j for node types")
            return 0
        else:
            print(f"\n❌ TEST FAILED: Upload did not succeed")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

