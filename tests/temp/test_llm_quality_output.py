"""
Quick script to test LLM memory generation quality with REAL Reducto PDF data
"""
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

from models.memory_models import AddMemoryRequest
from models.shared_types import MemoryMetadata
from models.hierarchical_models import TextElement, TableElement, ImageElement

async def main():
    # Create base metadata
    base_metadata = MemoryMetadata(
        organization_id=env.get('TEST_ORGANIZATION_ID', 'test_org'),
        namespace_id=env.get('TEST_NAMESPACE_ID', 'test_ns'),
        user_id=env.get('TEST_USER_ID', 'test_user'),
        workspace_id=env.get('TEST_WORKSPACE_ID', 'test_workspace'),
        customMetadata={'test': 'real_pdf_hierarchical'}
    )
    
    # Load the REAL Reducto PDF output
    real_pdf_path = Path('/Users/shawkatkabbara/Downloads/b1ee8b3479b29f40964bdaa830163b19_provider_result_f8141f7d-88ba-4145-925c-0b025b22d6c7.json')
    
    if not real_pdf_path.exists():
        print(f"❌ Real PDF file not found at {real_pdf_path}")
        return
    
    print('\n' + '='*80)
    print('LLM-GENERATED MEMORY QUALITY TEST - REAL REDUCTO PDF')
    print('='*80)
    print(f'\n📄 Using real PDF output: {real_pdf_path.name}')
    
    # First extract structured content from the real PDF
    from cloud_plugins.temporal.activities.document_activities import extract_structured_content_from_provider
    
    with open(real_pdf_path, 'r') as f:
        real_reducto_data = json.load(f)
    
    print(f'\n🔄 Step 1: Extracting structured content from Reducto output...')
    
    extraction = await extract_structured_content_from_provider(
        provider_specific=real_reducto_data,
        provider_name="reducto",
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id
    )
    
    print(f'✅ Extraction completed:')
    print(f'   - Decision: {extraction["decision"]}')
    print(f'   - Total elements: {len(extraction.get("structured_elements", []))}')
    print(f'   - Element types: {extraction.get("element_summary", {})}')
    print(f'   - Structure: {extraction.get("structure_analysis", {})}')
    
    # Use first 10 elements for LLM test (to keep it fast)
    structured_elements = extraction.get('structured_elements', [])[:10]
    
    if not structured_elements:
        print("❌ No structured elements extracted!")
        return
    
    print(f'\n🤖 Step 2: Generating LLM-optimized memories for first {len(structured_elements)} elements...')
    
    from cloud_plugins.temporal.activities.document_activities import generate_llm_optimized_memory_structures
    
    result = await generate_llm_optimized_memory_structures(
        content_elements=structured_elements,
        domain=None,  # Let LLM infer domain from content
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        use_llm=True
    )
    
    memories = result.get('memory_requests', [])
    
    print(f'\n✅ Generated {len(memories)} LLM-optimized memories\n')
    
    # Show first 3 memories in detail
    for idx, mem in enumerate(memories[:3], 1):
        separator = '='*80
        print(f'\n{separator}')
        print(f'MEMORY #{idx} (from real PDF)')
        print(separator)
        print(f'\nContent:\n{mem["content"]}\n')
        
        metadata = mem.get('metadata', {})
        custom = metadata.get('customMetadata', {}) if isinstance(metadata, dict) else {}
        
        print('Original Element Info:')
        print(f'  - Element ID: {custom.get("element_id", "N/A")}')
        print(f'  - Content Type: {custom.get("content_type", "N/A")}')
        print(f'  - Provider: {custom.get("provider", "N/A")}')
        print(f'  - Chunk/Block: {custom.get("chunk_index", "N/A")}/{custom.get("block_index", "N/A")}')
        
        print('\nLLM Enhancements:')
        print(f'  - LLM Enhanced: {custom.get("llm_enhanced", False)}')
        
        relationships = custom.get("llm_relationships")
        if relationships:
            rel_preview = relationships[:200] if isinstance(relationships, str) else str(relationships)[:200]
            print(f'  - Relationships: {rel_preview}...')
        
        topics = metadata.get("topics", []) if isinstance(metadata, dict) else []
        if topics:
            print(f'  - Topics: {topics}')
        
        # Show comparison with original
        print(f'\n📊 Content Enhancement:')
        orig_elem = structured_elements[idx-1]
        orig_content = orig_elem.get('content', '')
        print(f'  - Original length: {len(orig_content)} chars')
        print(f'  - Enhanced length: {len(mem["content"])} chars')
        print(f'  - Original preview: {orig_content[:150]}...')
    
    print(f'\n{separator}')
    print('SUMMARY - REAL PDF PROCESSING')
    print(separator)
    print(f'📄 PDF: {real_pdf_path.name}')
    print(f'📊 Total pages: {extraction.get("structure_analysis", {}).get("total_pages", "N/A")}')
    print(f'📝 Total elements extracted: {len(extraction.get("structured_elements", []))}')
    print(f'🤖 Memories generated (sample): {len(memories)}')
    print(f'🚀 LLM model: Groq (openai/gpt-oss-20b)')
    print(f'⚡ Batch size used: Auto-detected based on content')
    
    # Count LLM enhancements
    enhanced = sum(1 for m in memories if m.get('metadata', {}).get('customMetadata', {}).get('llm_enhanced'))
    print(f'✨ Memories with LLM enhancements: {enhanced}/{len(memories)} ({100*enhanced//len(memories) if memories else 0}%)')
    
    # Show element type breakdown
    print(f'\n📈 Element Type Breakdown:')
    for elem_type, count in extraction.get('element_summary', {}).items():
        print(f'   - {elem_type}: {count}')
    
    print(f'\n{separator}\n')

if __name__ == "__main__":
    asyncio.run(main())

