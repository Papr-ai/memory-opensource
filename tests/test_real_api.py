#!/usr/bin/env python3
"""
Test the document processing API endpoint with real TensorLake API key
"""

import asyncio
import httpx
import os
from pathlib import Path

async def test_document_upload():
    """Test document upload with real API"""

    # Create a simple PDF-like content for testing
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n50\n%%EOF"

    # API endpoint
    base_url = "http://localhost:8000"  # Assuming FastAPI runs on port 8000
    url = f"{base_url}/v1/document"

    # Headers with test credentials
    headers = {
        "X-API-Key": os.getenv("TEST_X_USER_API_KEY", "YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd"),
        "X-Client-Type": "test_client"
    }

    # Form data
    files = {
        "file": ("test.pdf", pdf_content, "application/pdf")
    }

    data = {
        "preferred_provider": "tensorlake",
        "metadata": '{"source": "api_test", "type": "pdf"}'
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("Testing document upload with TensorLake provider...")
            print(f"API Key: {headers['X-API-Key'][:10]}...")
            print(f"TensorLake API Key: {os.getenv('TENSORLAKE_API_KEY', 'NOT_SET')[:20]}...")

            response = await client.post(
                url,
                headers=headers,
                files=files,
                data=data
            )

            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")

            if response.status_code in [200, 202]:
                result = response.json()
                print(f"Success! Upload ID: {result.get('document_status', {}).get('upload_id')}")
                print(f"Provider used: {result.get('details', {}).get('provider')}")
                return True
            else:
                print(f"Failed with status {response.status_code}")
                return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    success = asyncio.run(test_document_upload())
    print(f"Test {'PASSED' if success else 'FAILED'}")