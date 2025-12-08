#!/usr/bin/env python3
"""
Test script for upload_file_to_parse function
Run this to test file uploads to Parse Server in isolation

Usage:
    python test_upload_file.py [file_path]
    
If no file_path is provided, uses a small test file.
"""
import asyncio
import sys
from pathlib import Path
import mimetypes
from os import environ as env
from dotenv import find_dotenv, load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

from services.memory_management import upload_file_to_parse

async def test_upload_file(file_path: str = None):
    """Test the upload_file_to_parse function"""
    print("=" * 80)
    print("Testing upload_file_to_parse function")
    print("=" * 80)
    
    # Get environment variables
    parse_server_url = env.get("PARSE_SERVER_URL")
    parse_application_id = env.get("PARSE_APPLICATION_ID")
    parse_master_key = env.get("PARSE_MASTER_KEY")
    parse_rest_api_key = env.get("PARSE_REST_API_KEY")
    
    # Check environment variables
    print(f"\nParse Server URL: {parse_server_url}")
    print(f"Parse Application ID: {parse_application_id}")
    print(f"Parse Master Key: {'SET' if parse_master_key else 'NOT SET'}")
    print(f"Parse REST API Key: {'SET' if parse_rest_api_key else 'NOT SET'}")
    
    # Read file or use default test content
    if file_path:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"\n❌ ERROR: File not found: {file_path}")
            return False
        
        print(f"\nReading file: {file_path}")
        with open(file_path_obj, 'rb') as f:
            test_content = f.read()
        
        test_filename = file_path_obj.name
        # Detect MIME type
        test_content_type, _ = mimetypes.guess_type(str(file_path_obj))
        if not test_content_type:
            test_content_type = "application/octet-stream"
    else:
        # Create a simple test file (text file)
        test_content = b"Hello, this is a test file upload to Parse Server!\nThis file is used to test the upload_file_to_parse function."
        test_filename = "test_upload.txt"
        test_content_type = "text/plain"
    
    print(f"\nTest file details:")
    print(f"  Filename: {test_filename}")
    print(f"  Size: {len(test_content)} bytes")
    print(f"  Content Type: {test_content_type}")
    
    # Try to get session token from environment or use Master Key
    # Note: If api_key is provided (truthy), the function uses PARSE_MASTER_KEY internally
    # If api_key is None, it uses session_token
    session_token = env.get("PARSE_SESSION_TOKEN", "")
    api_key = env.get("PARSE_API_KEY")
    
    # If we have parse_master_key, use it by passing any truthy api_key value
    # The function will use PARSE_MASTER_KEY when api_key is provided
    if parse_master_key:
        use_api_key = True
        api_key = api_key or "master"  # Any truthy value triggers master key auth
        print(f"\nAuthentication:")
        print(f"  Using Master Key authentication (PARSE_MASTER_KEY is set)")
    elif session_token:
        use_api_key = False
        print(f"\nAuthentication:")
        print(f"  Using Session Token authentication")
    else:
        print("\n⚠️  WARNING: Need either:")
        print("   1. PARSE_SESSION_TOKEN in environment, OR")
        print("   2. PARSE_MASTER_KEY in environment")
        print("\n   Attempting with dummy session token (will likely fail)...")
        session_token = "dummy_token"
        use_api_key = False
    
    print(f"\n{'='*80}")
    print("Starting upload...")
    print(f"{'='*80}\n")
    
    try:
        result = await upload_file_to_parse(
            file_content=test_content,
            filename=test_filename,
            content_type=test_content_type,
            session_token=session_token if not use_api_key else "dummy",
            api_key=api_key if use_api_key else None
        )
        
        if result:
            print(f"\n✅ SUCCESS! File uploaded successfully")
            print(f"   File URL: {result.file_url}")
            print(f"   Source URL: {result.source_url}")
            print(f"   Name: {result.name}")
            print(f"   MIME Type: {result.mime_type}")
            return True
        else:
            print(f"\n❌ FAILED: upload_file_to_parse returned None")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Exception occurred during upload")
        print(f"   Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Parse Server File Upload Test")
    print("="*80)
    
    # Get file path from command line if provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if file_path:
        print(f"Using file: {file_path}\n")
    
    # Run the test
    success = asyncio.run(test_upload_file(file_path))
    
    print(f"\n{'='*80}")
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    print(f"{'='*80}\n")
    
    sys.exit(0 if success else 1)

