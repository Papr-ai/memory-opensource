"""
Simple test script for PaddleOCR provider to verify output and content extraction
"""

import asyncio
import os
from pathlib import Path
from os import environ as env

async def test_paddleocr_provider():
    """Test PaddleOCR provider content extraction"""
    print("\n" + "="*80)
    print("TESTING: PaddleOCR Provider")
    print("="*80)
    
    try:
        # Check if PaddleOCR is installed
        try:
            import paddleocr
            print("✅ PaddleOCR package available")
        except ImportError:
            print("❌ PaddleOCR not installed. Install with: pip install paddleocr")
            return False
        
        from core.document_processing.providers.paddleocr import PaddleOCRProvider
        
        # Initialize provider (PaddleOCR runs locally, no API key needed)
        config = {
            "lang": "en",
            "use_gpu": False  # Use CPU for testing
        }
        
        provider = PaddleOCRProvider(config)
        print("✅ PaddleOCR provider initialized")
        
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
        print("\n📤 Processing document with PaddleOCR...")
        result = await provider.process_document(
            file_content=file_content,
            filename="call_answering_sop.pdf",
            upload_id="test-paddle-123"
        )
        
        print("\n📊 PROCESSING RESULT:")
        print(f"   Total pages: {result.total_pages}")
        print(f"   Processing time: {result.processing_time}s")
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
                print(f"   Type: {type(result.provider_specific)}")
                if hasattr(result.provider_specific, 'model_dump'):
                    ps_dict = result.provider_specific.model_dump()
                elif isinstance(result.provider_specific, dict):
                    ps_dict = result.provider_specific
                else:
                    ps_dict = {}
                
                print(f"   Keys: {list(ps_dict.keys())}")
                
                # Check for actual content
                if "results" in ps_dict:
                    print(f"   ✅ Has 'results' field")
                if "text" in ps_dict:
                    print(f"   ✅ Has 'text' field: {len(ps_dict['text'])} chars")
                if "content" in ps_dict:
                    print(f"   ✅ Has 'content' field: {len(ps_dict['content'])} chars")
                    
                # PaddleOCR typically returns OCR results as structured data
                actual_content = ps_dict.get("text") or ps_dict.get("content") or ""
                if actual_content:
                    print(f"\n📝 ACTUAL CONTENT SAMPLE (first 300 chars):")
                    print(f"{actual_content[:300]}")
                else:
                    print("\n⚠️  No direct text/content field, may need to extract from results")
        
        print("\n✅ TEST PASSED - PaddleOCR provider working!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run PaddleOCR provider test"""
    print("\n" + "="*80)
    print("PaddleOCR Provider Test")
    print("="*80)
    
    success = await test_paddleocr_provider()
    
    print("\n" + "="*80)
    print(f"RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

