#!/usr/bin/env python3
"""
Test to verify complete image URL preservation flow:
1. Provider extraction (Reducto) -> ImageElement with markdown content
2. LLM enhancement -> Preserves image_url in metadata
3. Final memory -> Has markdown in content + URL in metadata
"""
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()
os.environ.setdefault("TESTING", "1")

from core.document_processing.provider_adapter import extract_structured_elements
from core.document_processing.llm_memory_generator import generate_memory_structure_for_content
from models.memory_models import MemoryMetadata

async def main():
    print("=" * 80)
    print("IMAGE URL PRESERVATION FLOW TEST")
    print("=" * 80)
    
    # Load real Reducto response
    reducto_file = Path("/Users/shawkatkabbara/Downloads/b1ee8b3479b29f40964bdaa830163b19_provider_result_f8141f7d-88ba-4145-925c-0b025b22d6c7.json")
    
    if not reducto_file.exists():
        print(f"❌ File not found: {reducto_file}")
        return
    
    with open(reducto_file) as f:
        provider_response = json.load(f)
    
    print(f"\n✅ Loaded Reducto response from: {reducto_file.name}")
    
    # Step 1: Extract structured elements
    print("\n" + "=" * 80)
    print("STEP 1: Provider Extraction (provider_adapter.py)")
    print("=" * 80)
    
    elements = extract_structured_elements(provider_response, "reducto")
    
    # Find first image element
    image_elements = [e for e in elements if e.content_type.value == "image"]
    
    if not image_elements:
        print("❌ No image elements found")
        return
    
    first_image = image_elements[0]
    
    print(f"\n📸 Found {len(image_elements)} image elements")
    print(f"\nFirst image element:")
    print(f"  - element_id: {first_image.element_id}")
    print(f"  - image_url: {first_image.image_url}")
    print(f"  - image_description: {first_image.image_description}")
    print(f"  - content (first 200 chars):\n    {first_image.content[:200]}")
    
    # Check if markdown syntax is present
    has_markdown = "![" in first_image.content and "](" in first_image.content
    print(f"\n  ✅ Markdown syntax present: {has_markdown}")
    
    # Step 2: LLM Enhancement
    print("\n" + "=" * 80)
    print("STEP 2: LLM Enhancement (llm_memory_generator.py)")
    print("=" * 80)
    
    base_metadata = MemoryMetadata(
        organization_id="test_org",
        namespace_id="test_namespace"
    )
    
    # Generate memory for first image (this will call LLM)
    print("\n🤖 Calling LLM to enhance first image element...")
    enhanced_memory = await generate_memory_structure_for_content(
        content_element=first_image,
        base_metadata=base_metadata,
        domain=None
    )
    
    if not enhanced_memory:
        print("❌ LLM enhancement failed")
        return
    
    print(f"\n✅ LLM enhanced memory created")
    print(f"\nContent (first 300 chars):")
    print(f"  {enhanced_memory.content[:300]}")
    
    # Check metadata
    custom_meta = enhanced_memory.metadata.customMetadata
    
    print(f"\n📋 Custom Metadata:")
    print(f"  - image_url: {custom_meta.get('image_url', 'NOT FOUND')}")
    print(f"  - image_description: {custom_meta.get('image_description', 'NOT FOUND')}")
    print(f"  - bbox: {custom_meta.get('bbox', 'NOT FOUND')}")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    checks = {
        "Markdown in content": "![" in enhanced_memory.content,
        "image_url in metadata": "image_url" in custom_meta,
        "image_description in metadata": "image_description" in custom_meta,
        "bbox in metadata": "bbox" in custom_meta,
        "Content not just description": len(enhanced_memory.content) > len(first_image.image_description or "") + 20
    }
    
    for check_name, passed in checks.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {check_name}: {passed}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Image URL preservation is working!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review output above")
    print("=" * 80)
    
    # Show full memory request for inspection
    print("\n\n📄 FULL MEMORY REQUEST (for manual inspection):")
    print("=" * 80)
    print(json.dumps(enhanced_memory.model_dump(), indent=2, default=str)[:2000])
    print("...")

if __name__ == "__main__":
    asyncio.run(main())

