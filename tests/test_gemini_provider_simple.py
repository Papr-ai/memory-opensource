"""
Simple test script for Gemini provider to verify SDK output and content extraction
"""

import asyncio
from pathlib import Path

# Load .env file first
try:
    from dotenv import find_dotenv, load_dotenv
    _ENV_FILE = find_dotenv()
    if _ENV_FILE:
        load_dotenv(_ENV_FILE)
        print("✅ Environment variables loaded from .env")
except Exception as e:
    print(f"⚠️  Could not load .env: {e}")

from os import environ as env

async def test_gemini_provider():
    """Test Gemini provider content extraction"""
    print("\n" + "="*80)
    print("TESTING: Gemini Provider (NEW google-genai SDK)")
    print("="*80)
    
    try:
        
        # Check for required packages first
        try:
            import google.genai
            print("✅ google-genai package available (NEW SDK)")
        except ImportError:
            print("❌ google-genai not installed. Install with: poetry add google-genai")
            return False
        
        try:
            import fitz  # PyMuPDF
            print("✅ PyMuPDF package available")
        except ImportError:
            print("❌ PyMuPDF not installed. Install with: poetry add pymupdf")
            return False
        
        from core.document_processing.providers.gemini import GeminiVisionProvider
        
        # Get API key using env.get()
        api_key = env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY or GEMINI_API_KEY not set in .env")
            return False
        
        print(f"✅ API key loaded: {api_key[:10]}...")
        
        # Initialize provider
        config = {
            "api_key": api_key,
            "model": env.get("GEMINI_MODEL", "gemini-2.0-flash-exp")  # Using latest model
        }
        
        provider = GeminiVisionProvider(config)
        print("✅ Gemini Vision provider initialized")
        
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
        print("\n📤 Processing document with Gemini...")
        result = await provider.process_document(
            file_content=file_content,
            filename="call_answering_sop.pdf",
            upload_id="test-gemini-123"
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
                if "text" in ps_dict:
                    print(f"   ✅ Has 'text' field: {len(ps_dict['text'])} chars")
                if "content" in ps_dict:
                    print(f"   ✅ Has 'content' field: {len(ps_dict['content'])} chars")
                    
                # Gemini typically returns response in 'text' field
                actual_content = ps_dict.get("text") or ps_dict.get("content") or ""
                if actual_content:
                    print(f"\n📝 ACTUAL CONTENT SAMPLE (first 300 chars):")
                    print(f"{actual_content[:300]}")
                else:
                    print("\n❌ No actual content found in provider_specific!")
        
        print("\n✅ TEST PASSED - Gemini provider working!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run Gemini provider test"""
    print("\n" + "="*80)
    print("Gemini Provider Test")
    print("="*80)
    
    success = await test_gemini_provider()
    
    print("\n" + "="*80)
    print(f"RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

