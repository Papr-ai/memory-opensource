#!/usr/bin/env python3
"""
Debug TensorLake API responses to see actual status format
"""

import asyncio
import httpx
import os
import json
from dotenv import load_dotenv

async def debug_tensorlake_status():
    """Debug TensorLake status responses"""

    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        load_dotenv()

    api_key = os.getenv("TENSORLAKE_API_KEY")
    base_url = os.getenv("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai")

    print(f"Debug TensorLake status...")
    print(f"API Key: {api_key[:20]}...")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        # First, let's start a new parse job
        print("Creating a new parse job...")

        # Upload a simple file first
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n50\n%%EOF"
        upload_files = {"file_bytes": ("test.pdf", pdf_content)}
        upload_data = {"labels": '{"debug": "true"}'}

        upload_response = await client.put(
            f"{base_url}/documents/v2/files",
            files=upload_files,
            data=upload_data,
            headers=headers
        )

        if upload_response.status_code != 200:
            print(f"Upload failed: {upload_response.status_code} - {upload_response.text}")
            return

        upload_result = upload_response.json()
        file_id = upload_result.get("file_id")
        print(f"File uploaded, file_id: {file_id}")

        # Start parsing
        parse_data = {
            "file_id": file_id,
            "mime_type": "application/pdf",
            "file_name": "test.pdf",
            "parsing_options": {
                "chunking_strategy": "none"
            }
        }

        parse_response = await client.post(
            f"{base_url}/documents/v2/read",
            json=parse_data,
            headers=headers
        )

        if parse_response.status_code != 200:
            print(f"Parse failed: {parse_response.status_code} - {parse_response.text}")
            return

        parse_result = parse_response.json()
        parse_id = parse_result.get("parse_id")
        print(f"Parsing started, parse_id: {parse_id}")
        print(f"Parse response: {json.dumps(parse_result, indent=2)}")

        # Now check status a few times and show full response
        for i in range(5):
            await asyncio.sleep(2)
            print(f"\n--- Status check #{i+1} ---")

            status_response = await client.get(
                f"{base_url}/documents/v2/parse/{parse_id}",
                headers=headers
            )

            print(f"Status code: {status_response.status_code}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Full response: {json.dumps(status_data, indent=2)}")

                # Check what fields are available
                status = status_data.get("status")
                print(f"Status field: {status}")

                if status == "successful" or "content" in status_data:
                    print("✅ Parsing completed!")
                    break
                elif status == "failed":
                    print("❌ Parsing failed!")
                    break
                else:
                    print(f"⏳ Still processing (status: {status})")
            else:
                print(f"Error response: {status_response.text}")

if __name__ == "__main__":
    asyncio.run(debug_tensorlake_status())