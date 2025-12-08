"""
Simple test script for DeepSeek-OCR provider to verify SDK output and content extraction
"""

import asyncio
import os
from pathlib import Path
from os import environ as env

async def test_deepseek_ocr_provider():
    """Test DeepSeek-OCR provider content extraction"""
    print("\n" + "="*80)
    print("TESTING: DeepSeek-OCR Provider")
    print("="*80)
    
    try:
        from core.document_processing.providers.deepseek import DeepSeekOCRProvider
        
        # Check for API key (DeepInfra is the primary supported platform)
        api_key = env.get("DEEPINFRA_TOKEN") or env.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ No API key found.")
            print("\n📝 DeepSeek-OCR requires DeepInfra API:")
            print("   1. Sign up at: https://deepinfra.com")
            print("   2. Get your API token")
            print("   3. Set environment variable:")
            print("      export DEEPINFRA_TOKEN='your_token_here'")
            print("\n💡 Note: DeepSeek models are NOT on HuggingFace Inference API")
            print("   You need either DeepInfra or a self-hosted deployment")
            return False
        
        # Use DeepInfra (the primary supported platform)
        print("✅ Using DeepInfra API")
        base_url = env.get("DEEPSEEK_API_URL", "https://api.deepinfra.com/v1/inference")
        
        # Initialize provider
        config = {
            "api_key": api_key,
            "base_url": base_url,
            "model_id": env.get("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-Janus-1.3B")
        }
        
        provider = DeepSeekOCRProvider(config)
        print("✅ DeepSeek-OCR provider initialized")
        
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
        print("\n📤 Processing document with DeepSeek-OCR...")
        result = await provider.process_document(
            file_content=file_content,
            filename="call_answering_sop.pdf",
            upload_id="test-deepseek-123"
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
                if "response" in ps_dict:
                    print(f"   ✅ Has 'response' field")
                    
                # DeepSeek typically returns response in structured format
                actual_content = ps_dict.get("text") or ps_dict.get("content") or ""
                if actual_content:
                    print(f"\n📝 ACTUAL CONTENT SAMPLE (first 300 chars):")
                    print(f"{actual_content[:300]}")
                else:
                    print("\n❌ No actual content found in provider_specific!")
        
        print("\n✅ TEST PASSED - DeepSeek-OCR provider working!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run DeepSeek-OCR provider test"""
    print("\n" + "="*80)
    print("DeepSeek-OCR Provider Test")
    print("="*80)
    
    success = await test_deepseek_ocr_provider()
    
    print("\n" + "="*80)
    print(f"RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

