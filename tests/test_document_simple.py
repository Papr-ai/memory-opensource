#!/usr/bin/env python3
"""
Simple test to verify the /document/v2 route is working
"""

import requests
import os
from pathlib import Path
import json
from dotenv import load_dotenv

def test_document_route_simple():
    """Simple test for document route"""

    # Load environment
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    # Base URL (assuming server is running)
    base_url = "http://localhost:8000"

    # Test headers
    headers = {
        "X-API-Key": os.getenv("TEST_X_USER_API_KEY", "YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd"),
        "X-Client-Type": "test_client"
    }

    print(f"🧪 Testing document route at {base_url}/v1/document")
    print(f"🔑 Using API key: {headers['X-API-Key'][:15]}...")

    # Try to find a PDF file
    possible_paths = [
        Path.home() / "Documents" / "Papr Articles of Incroporation_2.pdf",
        Path.home() / "Documents" / "eVISA_Kabbara_BassamShaowkat_610a8bb1bdc61 2.pdf",
    ]

    pdf_file = None
    for pdf_path in possible_paths:
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            pdf_file = {
                "content": pdf_content,
                "filename": pdf_path.name,
                "size": len(pdf_content)
            }
            print(f"📄 Found PDF: {pdf_file['filename']} ({pdf_file['size']:,} bytes)")
            break

    if not pdf_file:
        print("❌ No PDF file found, creating a simple test PDF")
        # Create a minimal valid PDF
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document for PAPR Memory) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000234 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
292
%%EOF"""
        pdf_file = {
            "content": pdf_content,
            "filename": "test_document.pdf",
            "size": len(pdf_content)
        }

    # Test upload
    print(f"\n📤 Uploading document...")

    try:
        files = {
            "file": (pdf_file["filename"], pdf_file["content"], "application/pdf")
        }

        data = {
            "preferred_provider": "tensorlake",
            "metadata": json.dumps({
                "source": "simple_test",
                "test_type": "basic_upload"
            }),
            "namespace": "test-namespace"
        }

        response = requests.post(
            f"{base_url}/v1/document",
            headers=headers,
            files=files,
            data=data,
            timeout=30,
            verify=False
        )

        print(f"📤 Response status: {response.status_code}")
        print(f"📤 Response: {response.text[:500]}...")

        if response.status_code in [200, 202]:
            result = response.json()
            upload_id = result.get("document_status", {}).get("upload_id")
            use_temporal = result.get("details", {}).get("use_temporal", False)

            print(f"✅ Upload successful!")
            print(f"📋 Upload ID: {upload_id}")
            print(f"⚡ Using Temporal: {use_temporal}")

            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {base_url}")
        print(f"ℹ️  Make sure the server is running with: poetry run uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_document_route_simple()
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}")