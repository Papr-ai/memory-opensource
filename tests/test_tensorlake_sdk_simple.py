"""
Simple test to verify TensorLake SDK integration
"""
import asyncio
import os
from pathlib import Path

async def test_tensorlake_provider():
    """Test TensorLake provider with SDK"""
    print("\n" + "="*80)
    print("TESTING: TensorLake Provider with SDK")
    print("="*80)
    
    try:
        from core.document_processing.providers.tensorlake import TensorLakeProvider
        
        api_key = os.getenv("TENSORLAKE_API_KEY")
        if not api_key:
            print("❌ TENSORLAKE_API_KEY not set")
            return False
        
        # Initialize provider
        config = {
            "api_key": api_key,
            "base_url": "https://api.tensorlake.ai",
            "timeout": 300
        }
        
        provider = TensorLakeProvider(config)
        print("✅ TensorLake provider initialized")
        
        # Check if test PDF exists
        test_pdf = "tests/call_answering_sop.pdf"
        if not Path(test_pdf).exists():
            print(f"❌ Test PDF not found: {test_pdf}")
            return False
        
        # Read test file
        with open(test_pdf, "rb") as f:
            file_content = f.read()
        
        print(f"✅ Test PDF loaded: {len(file_content)} bytes")
        
        # Process document
        print("\n📤 Processing document with TensorLake SDK...")
        result = await provider.process_document(
            file_content=file_content,
            filename="call_answering_sop.pdf",
            upload_id="test-123"
        )
        
        print("\n📊 PROCESSING RESULT:")
        print(f"   Total pages: {result.total_pages}")
        print(f"   Processing time: {result.processing_time}")
        print(f"   Confidence: {result.confidence}")
        
        if result.pages:
            first_page = result.pages[0]
            print(f"\n📄 FIRST PAGE:")
            print(f"   Page number: {first_page.page_number}")
            print(f"   Content length: {len(first_page.content)} chars")
            print(f"   First 500 chars:\n{first_page.content[:500]}")
            
            # Check provider_specific data
            if result.provider_specific:
                print(f"\n📦 PROVIDER_SPECIFIC:")
                print(f"   Keys: {list(result.provider_specific.keys())}")
                if "content" in result.provider_specific:
                    content = result.provider_specific["content"]
                    print(f"   ✅ Has 'content' field: {len(content)} chars")
                else:
                    print(f"   ❌ Missing 'content' field")
                    
                if "parse_id" in result.provider_specific:
                    print(f"   parse_id: {result.provider_specific['parse_id']}")
        
        print("\n✅ TEST PASSED - TensorLake SDK integration working!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fetch_parse_result():
    """Test fetching parse result with SDK"""
    print("\n" + "="*80)
    print("TESTING: TensorLake fetch_parse_result with SDK")
    print("="*80)
    
    try:
        from core.document_processing.providers.tensorlake import TensorLakeProvider
        
        api_key = os.getenv("TENSORLAKE_API_KEY")
        if not api_key:
            print("❌ TENSORLAKE_API_KEY not set")
            return False
        
        # For this test, we need a valid parse_id
        # This would normally come from a previous parsing operation
        parse_id = os.getenv("TEST_TENSORLAKE_PARSE_ID")
        if not parse_id:
            print("⚠️  TEST_TENSORLAKE_PARSE_ID not set, skipping fetch test")
            return True
        
        config = {
            "api_key": api_key,
            "base_url": "https://api.tensorlake.ai",
            "timeout": 300
        }
        
        provider = TensorLakeProvider(config)
        print(f"✅ TensorLake provider initialized")
        print(f"   Fetching parse_id: {parse_id}")
        
        # Fetch parse result
        result = await provider.fetch_parse_result(parse_id)
        
        print("\n📊 FETCH RESULT:")
        print(f"   Type: {type(result)}")
        print(f"   Keys: {list(result.keys())}")
        
        if "content" in result:
            print(f"   ✅ Has 'content': {len(result['content'])} chars")
            print(f"   First 200 chars: {result['content'][:200]}")
        else:
            print(f"   ❌ Missing 'content' field")
        
        if "chunks" in result:
            print(f"   ✅ Has 'chunks': {len(result['chunks'])} chunks")
        
        print("\n✅ FETCH TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ FETCH TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TensorLake SDK Integration Tests")
    print("="*80)
    
    # Test 1: Process document
    test1_passed = await test_tensorlake_provider()
    
    # Test 2: Fetch parse result (optional)
    test2_passed = await test_fetch_parse_result()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"   Process document: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Fetch parse result: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

