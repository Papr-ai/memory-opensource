"""
Test scripts to understand provider SDK outputs

Run each test individually to see what data structures each provider returns.
This will help us implement proper Pydantic models and extraction logic.
"""

import asyncio
import os
from pathlib import Path

# Test file path
TEST_PDF = "tests/call_answering_sop.pdf"
TEST_IMAGE = "tests/test_image.png"  # Create a simple test image if needed


async def test_gemini_output():
    """Test Google Gemini Vision API output format"""
    print("\n" + "="*80)
    print("TESTING: Google Gemini Vision API")
    print("="*80)
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY or GEMINI_API_KEY not set")
            return
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Test with simple text extraction
        if Path(TEST_PDF).exists():
            with open(TEST_PDF, "rb") as f:
                file_bytes = f.read()
            
            # Upload file to Gemini
            print("\n📤 Uploading file to Gemini...")
            file = genai.upload_file(TEST_PDF)
            print(f"✅ File uploaded: {file.name}")
            
            # Generate content
            print("\n🤖 Generating content...")
            prompt = "Extract all text from this document. Return the text in markdown format."
            response = model.generate_content([prompt, file])
            
            print("\n📊 RESPONSE STRUCTURE:")
            print(f"Type: {type(response)}")
            print(f"Dir: {[x for x in dir(response) if not x.startswith('_')]}")
            
            print("\n📝 RESPONSE TEXT:")
            print(f"Type of text: {type(response.text)}")
            print(f"Text length: {len(response.text)} chars")
            print(f"First 500 chars:\n{response.text[:500]}")
            
            # Check if we can request structured output
            print("\n🔍 Testing structured output with JSON schema...")
            try:
                from pydantic import BaseModel
                from typing import List
                
                class DocumentStructure(BaseModel):
                    """Structured document output"""
                    title: str
                    sections: List[str]
                    full_text: str
                
                structured_prompt = """
Extract the document structure with:
- title: The document title
- sections: List of section headings
- full_text: All text content

Return as JSON matching the schema.
"""
                
                # Try using response_schema parameter (newer Gemini API)
                try:
                    structured_response = model.generate_content(
                        [structured_prompt, file],
                        generation_config={
                            "response_mime_type": "application/json"
                        }
                    )
                    print("✅ Structured output supported!")
                    print(f"Structured response type: {type(structured_response.text)}")
                    print(f"Structured response:\n{structured_response.text[:500]}")
                except Exception as e:
                    print(f"⚠️  Structured output not supported or failed: {e}")
                    
            except Exception as e:
                print(f"❌ Structured output test failed: {e}")
            
            # Check candidates structure
            print("\n🎯 CANDIDATES STRUCTURE:")
            if hasattr(response, 'candidates'):
                print(f"Number of candidates: {len(response.candidates)}")
                if response.candidates:
                    candidate = response.candidates[0]
                    print(f"Candidate type: {type(candidate)}")
                    print(f"Candidate dir: {[x for x in dir(candidate) if not x.startswith('_')]}")
                    
                    if hasattr(candidate, 'content'):
                        print(f"Content type: {type(candidate.content)}")
                        print(f"Content dir: {[x for x in dir(candidate.content) if not x.startswith('_')]}")
            
        else:
            print(f"❌ Test file not found: {TEST_PDF}")
            
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("   Install with: pip install google-generativeai")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_tensorlake_output():
    """Test TensorLake SDK output format"""
    print("\n" + "="*80)
    print("TESTING: TensorLake SDK")
    print("="*80)
    
    try:
        from tensorlake.documentai import DocumentAI
        
        api_key = os.getenv("TENSORLAKE_API_KEY")
        if not api_key:
            print("❌ TENSORLAKE_API_KEY not set")
            return
        
        if not Path(TEST_PDF).exists():
            print(f"❌ Test file not found: {TEST_PDF}")
            return
        
        print("\n🔧 Initializing TensorLake DocumentAI client...")
        doc_ai = DocumentAI(api_key=api_key)
        
        # Step 1: Upload and parse file
        print(f"\n📤 Uploading and parsing file: {TEST_PDF}")
        with open(TEST_PDF, "rb") as f:
            # The SDK handles upload and parsing in one call
            parse_response = doc_ai.parse_document(file=f, filename=Path(TEST_PDF).name)
        
        print(f"✅ Parse initiated")
        print(f"   Response type: {type(parse_response)}")
        print(f"   Response dir: {[x for x in dir(parse_response) if not x.startswith('_')]}")
        
        # Get parse_id
        if hasattr(parse_response, 'parse_id'):
            parse_id = parse_response.parse_id
            print(f"\n📝 Parse ID: {parse_id}")
        elif hasattr(parse_response, 'id'):
            parse_id = parse_response.id
            print(f"\n📝 Parse ID: {parse_id}")
        else:
            print(f"\n⚠️  Cannot find parse_id in response")
            print(f"   Available attributes: {[x for x in dir(parse_response) if not x.startswith('_')]}")
            return
        
        # Step 2: Wait for parsing to complete and get result
        print("\n⏳ Waiting for parse to complete...")
        import time
        max_wait = 60
        start = time.time()
        
        while time.time() - start < max_wait:
            # Get parse result using SDK
            result = doc_ai.get_parse_result(parse_id=parse_id)
            
            print(f"   Status: {result.status}")
            
            if result.status == 'successful' or result.status == 'completed':
                print("\n📊 PARSE RESULT STRUCTURE (SDK):")
                print(f"Type: {type(result)}")
                print(f"Attributes: {[x for x in dir(result) if not x.startswith('_')]}")
                
                # Check for chunks attribute
                if hasattr(result, 'chunks'):
                    print(f"\n✅ Has 'chunks' attribute")
                    print(f"   Type: {type(result.chunks)}")
                    print(f"   Length: {len(result.chunks) if result.chunks else 0}")
                    
                    if result.chunks and len(result.chunks) > 0:
                        first_chunk = result.chunks[0]
                        print(f"\n📦 First chunk structure:")
                        print(f"   Type: {type(first_chunk)}")
                        print(f"   Attributes: {[x for x in dir(first_chunk) if not x.startswith('_')]}")
                        
                        if hasattr(first_chunk, 'content'):
                            print(f"\n✅ Chunk has 'content' attribute")
                            print(f"   Type: {type(first_chunk.content)}")
                            print(f"   Length: {len(first_chunk.content)} chars")
                            print(f"   First 500 chars:\n{first_chunk.content[:500]}")
                        
                        if hasattr(first_chunk, 'text'):
                            print(f"\n✅ Chunk has 'text' attribute")
                            print(f"   Type: {type(first_chunk.text)}")
                            print(f"   Length: {len(first_chunk.text)} chars")
                    
                    # Combine all chunks
                    all_text = []
                    for chunk in result.chunks:
                        if hasattr(chunk, 'content'):
                            all_text.append(chunk.content)
                        elif hasattr(chunk, 'text'):
                            all_text.append(chunk.text)
                    
                    if all_text:
                        combined = "\n".join(all_text)
                        print(f"\n📄 COMBINED TEXT FROM ALL CHUNKS:")
                        print(f"   Total chunks: {len(all_text)}")
                        print(f"   Combined length: {len(combined)} chars")
                        print(f"   First 500 chars:\n{combined[:500]}")
                else:
                    print("\n❌ No 'chunks' attribute in result")
                
                # Check for other common fields
                for attr in ['content', 'text', 'pages', 'metadata', 'raw_response']:
                    if hasattr(result, attr):
                        value = getattr(result, attr)
                        print(f"\n✅ Has '{attr}' attribute: {type(value)}")
                        if isinstance(value, str):
                            print(f"   Length: {len(value)} chars")
                            print(f"   First 200 chars: {value[:200]}")
                
                break
            elif result.status == 'failed':
                print(f"\n❌ Parse failed")
                if hasattr(result, 'error'):
                    print(f"   Error: {result.error}")
                break
            
            await asyncio.sleep(2)
        else:
            print(f"\n⏰ Timeout waiting for parse to complete")
        
    except ImportError as ie:
        print("❌ tensorlake package not installed")
        print("   Install with: pip install tensorlake")
        print(f"   Import error: {ie}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_paddleocr_output():
    """Test PaddleOCR output format"""
    print("\n" + "="*80)
    print("TESTING: PaddleOCR")
    print("="*80)
    
    try:
        from paddleocr import PaddleOCR
        from PIL import Image
        import io
        
        print("\n🔧 Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Create a simple test image if needed
        test_file = TEST_IMAGE if Path(TEST_IMAGE).exists() else TEST_PDF
        
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            return
        
        print(f"\n📄 Processing file: {test_file}")
        
        # Run OCR
        result = ocr.ocr(test_file, cls=True)
        
        print("\n📊 RESULT STRUCTURE:")
        print(f"Type: {type(result)}")
        print(f"Length: {len(result) if result else 0}")
        
        if result and len(result) > 0:
            print(f"\nFirst element type: {type(result[0])}")
            print(f"First element length: {len(result[0]) if result[0] else 0}")
            
            if result[0] and len(result[0]) > 0:
                print(f"\nFirst detection type: {type(result[0][0])}")
                print(f"First detection length: {len(result[0][0])}")
                print(f"First detection structure: {result[0][0]}")
                
                # PaddleOCR returns: [bbox_coords, (text, confidence)]
                if len(result[0][0]) >= 2:
                    bbox, text_conf = result[0][0][0], result[0][0][1]
                    text, conf = text_conf
                    
                    print(f"\n✅ DETECTION FORMAT:")
                    print(f"   BBox: {bbox}")
                    print(f"   Text: {text}")
                    print(f"   Confidence: {conf}")
        
        print(f"\n📋 FULL RESULT (first 3 detections):")
        for i, page_result in enumerate(result[:1]):  # First page only
            print(f"\nPage {i+1}:")
            if page_result:
                for j, detection in enumerate(page_result[:3]):  # First 3 detections
                    if len(detection) >= 2:
                        bbox, (text, conf) = detection[0], detection[1]
                        print(f"  {j+1}. '{text}' (conf: {conf:.3f})")
        
    except ImportError:
        print("❌ paddleocr package not installed")
        print("   Install with: pip install paddleocr")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_deepseek_ocr_output():
    """Test DeepSeek-OCR HuggingFace API output format"""
    print("\n" + "="*80)
    print("TESTING: DeepSeek-OCR (HuggingFace API)")
    print("="*80)
    
    try:
        import httpx
        
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            print("❌ HUGGINGFACE_API_KEY not set")
            return
        
        model_id = "deepseek-ai/DeepSeek-OCR"
        base_url = "https://api-inference.huggingface.co/models"
        
        if not Path(TEST_PDF).exists():
            print(f"❌ Test file not found: {TEST_PDF}")
            return
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            print("\n📤 Sending request to HuggingFace...")
            
            with open(TEST_PDF, "rb") as f:
                file_data = f.read()
            
            headers = {"Authorization": f"Bearer {api_key}"}
            
            response = await client.post(
                f"{base_url}/{model_id}",
                content=file_data,
                headers=headers
            )
            
            print(f"\n📊 RESPONSE:")
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\nResponse type: {type(result)}")
                
                if isinstance(result, list):
                    print(f"List length: {len(result)}")
                    if result:
                        print(f"First element type: {type(result[0])}")
                        print(f"First element: {result[0]}")
                        
                        if isinstance(result[0], dict):
                            print(f"Keys: {result[0].keys()}")
                            
                            # Common fields in HF API responses
                            for field in ['generated_text', 'text', 'label', 'score', 'confidence']:
                                if field in result[0]:
                                    value = result[0][field]
                                    print(f"\n✅ Has '{field}': {type(value)}")
                                    if isinstance(value, str):
                                        print(f"   Length: {len(value)} chars")
                                        print(f"   First 500 chars:\n{value[:500]}")
                                    else:
                                        print(f"   Value: {value}")
                
                elif isinstance(result, dict):
                    print(f"Keys: {list(result.keys())}")
                    for key, value in result.items():
                        print(f"\n'{key}': {type(value)}")
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  First 200 chars: {value[:200]}")
                        else:
                            print(f"  Value: {value}")
                
                else:
                    print(f"Unexpected type: {type(result)}")
                    print(f"Value: {result}")
            else:
                print(f"❌ Error response:")
                print(f"   {response.text}")
                
    except ImportError:
        print("❌ httpx package not installed")
        print("   Install with: pip install httpx")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all provider tests"""
    print("\n" + "="*80)
    print("PROVIDER OUTPUT FORMAT TESTING")
    print("="*80)
    print("\nThis script tests each provider to understand their output formats")
    print("so we can implement proper Pydantic models and extraction logic.\n")
    
    # Test each provider
    await test_gemini_output()
    await test_tensorlake_output()
    await test_paddleocr_output()
    await test_deepseek_ocr_output()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the output structures above")
    print("2. Create Pydantic models for each provider's output")
    print("3. Update provider_adapter.py to properly extract content")
    print("4. Update the provider classes to use typed responses")


if __name__ == "__main__":
    asyncio.run(main())

