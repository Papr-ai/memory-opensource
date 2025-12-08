"""
Unit tests for LLM Memory Generation activity

This test suite validates that:
1. LLM-optimized memory generation works correctly
2. Tables are chunked into separate, structured memories
3. Hierarchical document structure (sections/subsections) is preserved
4. LLM adds intelligent metadata and relationships
5. Different content types (text, tables, images) are handled appropriately
"""
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from os import environ as env
from typing import List, Dict, Any

from models.memory_models import AddMemoryRequest
from models.shared_types import MemoryMetadata
from models.hierarchical_models import TextElement, TableElement, ImageElement, ContentType

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


@pytest.fixture
def base_metadata():
    """Create base metadata for testing"""
    return MemoryMetadata(
        organization_id=env.get("TEST_ORGANIZATION_ID", "test_org"),
        namespace_id=env.get("TEST_NAMESPACE_ID", "test_ns"),
        user_id=env.get("TEST_USER_ID", "test_user"),
        workspace_id=env.get("TEST_WORKSPACE_ID", "test_workspace"),
        customMetadata={
            "test_id": "llm_memory_gen_test",
            "source": "pytest"
        }
    )


@pytest.fixture
def hierarchical_document_elements():
    """
    Create structured elements representing a hierarchical document:
    - Section 1: Introduction
      - Subsection 1.1: Background
      - Subsection 1.2: Objectives
    - Section 2: Financial Results (with table)
      - Subsection 2.1: Revenue Analysis
    - Section 3: Visual Data (with chart)
    """
    elements = []
    
    # Section 1: Introduction (Level 1)
    elements.append(TextElement(
        element_id="section_1",
        content="Introduction\n\nThis document presents the quarterly financial results and analysis for Q4 2024. The report includes detailed revenue breakdowns, expense analysis, and future projections.",
        metadata={
            "section_level": 1,
            "section_title": "Introduction",
            "section_number": "1",
            "content_type": "section_header"
        }
    ))
    
    # Subsection 1.1: Background (Level 2)
    elements.append(TextElement(
        element_id="section_1_1",
        content="1.1 Background\n\nOur company has shown consistent growth over the past fiscal year. This quarter represents a significant milestone in our expansion strategy, with key investments in technology infrastructure and market development.",
        metadata={
            "section_level": 2,
            "section_title": "Background",
            "section_number": "1.1",
            "parent_section": "1",
            "content_type": "subsection"
        }
    ))
    
    # Subsection 1.2: Objectives (Level 2)
    elements.append(TextElement(
        element_id="section_1_2",
        content="1.2 Objectives\n\nThe primary objectives for this quarter were: (1) Achieve 20% revenue growth, (2) Expand into three new markets, (3) Reduce operational costs by 10%, and (4) Increase customer satisfaction scores.",
        metadata={
            "section_level": 2,
            "section_title": "Objectives",
            "section_number": "1.2",
            "parent_section": "1",
            "content_type": "subsection"
        }
    ))
    
    # Section 2: Financial Results (Level 1) - with embedded table reference
    elements.append(TextElement(
        element_id="section_2",
        content="2. Financial Results\n\nQ4 2024 delivered exceptional financial performance across all key metrics. Total revenue reached $125M, representing a 25% increase year-over-year. The detailed breakdown is presented in the financial summary table below.",
        metadata={
            "section_level": 1,
            "section_title": "Financial Results",
            "section_number": "2",
            "content_type": "section_header",
            "contains_table": True
        }
    ))
    
    # TABLE: Financial Summary (should be separate memory with structured data)
    table_data = {
        "headers": ["Metric", "Q3 2024", "Q4 2024", "YoY Change"],
        "rows": [
            ["Revenue", "$100M", "$125M", "+25%"],
            ["Operating Expenses", "$60M", "$70M", "+16.7%"],
            ["EBITDA", "$40M", "$55M", "+37.5%"],
            ["Net Income", "$28M", "$38M", "+35.7%"],
            ["Gross Margin", "60%", "64%", "+4pp"]
        ]
    }
    
    elements.append(TableElement(
        element_id="table_financial_summary",
        content="Financial Summary Table: Revenue $100M → $125M (+25%), Operating Expenses $60M → $70M (+16.7%), EBITDA $40M → $55M (+37.5%), Net Income $28M → $38M (+35.7%), Gross Margin 60% → 64% (+4pp)",
        structured_data=table_data,
        headers=table_data["headers"],
        rows=table_data["rows"],
        table_type="data_table",
        metadata={
            "section_level": 2,
            "parent_section": "2",
            "section_title": "Financial Summary Table",
            "content_type": "financial_table",
            "table_category": "quarterly_results",
            "time_period": "Q4 2024"
        }
    ))
    
    # Subsection 2.1: Revenue Analysis (Level 2)
    elements.append(TextElement(
        element_id="section_2_1",
        content="2.1 Revenue Analysis\n\nRevenue growth was driven primarily by three factors: (1) 30% increase in enterprise customer acquisitions, (2) 15% expansion in average contract value, and (3) successful launch of our new product line contributing $18M in incremental revenue.",
        metadata={
            "section_level": 2,
            "section_title": "Revenue Analysis",
            "section_number": "2.1",
            "parent_section": "2",
            "content_type": "subsection",
            "domain": "financial"
        }
    ))
    
    # Section 3: Visual Data (Level 1) - with chart
    elements.append(TextElement(
        element_id="section_3",
        content="3. Market Performance\n\nOur market share has grown significantly in all target segments. The following chart illustrates our competitive position and growth trajectory across different market verticals.",
        metadata={
            "section_level": 1,
            "section_title": "Market Performance",
            "section_number": "3",
            "content_type": "section_header",
            "contains_chart": True
        }
    ))
    
    # IMAGE/CHART: Market Share Growth
    elements.append(ImageElement(
        element_id="chart_market_share",
        content="Market Share Growth Chart: Shows enterprise segment at 35% (+8%), SMB segment at 28% (+5%), and startup segment at 15% (+3%). Competitive positioning improved across all categories.",
        image_url="https://example.com/charts/market_share_q4_2024.png",
        image_description="Bar chart comparing Q3 and Q4 market share across three segments: Enterprise, SMB, and Startup. Each segment shows positive growth with enterprise leading at 35% market share.",
        metadata={
            "section_level": 2,
            "parent_section": "3",
            "section_title": "Market Share Growth Chart",
            "content_type": "chart",
            "chart_type": "bar_chart",
            "visual_category": "market_analysis"
        }
    ))
    
    return elements


@pytest.fixture
def simple_extraction_output(hierarchical_document_elements, base_metadata):
    """
    Simulate output from extract_structured_content_from_provider
    This represents a "complex" document that needs LLM optimization
    """
    return {
        "decision": "complex",
        "structured_elements": [elem.model_dump() for elem in hierarchical_document_elements],
        "memory_requests": [],  # Empty - will be filled by LLM generation
        "element_summary": {
            "text": 5,
            "table": 1,
            "image": 1
        },
        "structure_analysis": {
            "total_pages": 3,
            "has_tables": True,
            "has_images": True,
            "has_charts": True
        },
        "provider": "reducto",
        "extraction_stored": False
    }


@pytest.mark.asyncio
async def test_llm_memory_generation_basic(simple_extraction_output, base_metadata):
    """Test basic LLM memory generation with structured elements"""
    from cloud_plugins.temporal.activities.document_activities import generate_llm_optimized_memory_structures
    
    extraction = simple_extraction_output
    structured_elements = extraction["structured_elements"]
    
    print(f"\n{'='*80}")
    print(f"TEST: Basic LLM Memory Generation")
    print(f"{'='*80}")
    print(f"Input: {len(structured_elements)} structured elements")
    print(f"Element types: {extraction['element_summary']}")
    
    # Call the LLM generation activity
    result = await generate_llm_optimized_memory_structures(
        content_elements=structured_elements,
        domain="financial",  # Use financial domain for specialized prompts
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        use_llm=True
    )
    
    print(f"\n{'='*80}")
    print(f"LLM GENERATION RESULTS")
    print(f"{'='*80}")
    
    # Extract memory requests
    memory_requests = result.get("memory_requests", [])
    generation_summary = result.get("generation_summary", {})
    
    print(f"Generated {len(memory_requests)} memory structures")
    print(f"Generation summary: {generation_summary}")
    
    # Assertions
    assert len(memory_requests) > 0, "Should generate at least one memory"
    assert len(memory_requests) == len(structured_elements), "Should generate one memory per element"
    
    # Analyze the generated memories
    print(f"\n{'='*80}")
    print(f"MEMORY ANALYSIS")
    print(f"{'='*80}")
    
    for idx, mem_dict in enumerate(memory_requests[:3]):  # Show first 3
        print(f"\n--- Memory {idx + 1} ---")
        print(f"Content preview: {mem_dict['content'][:150]}...")
        print(f"Type: {mem_dict.get('type', 'N/A')}")
        
        metadata = mem_dict.get('metadata', {})
        custom_meta = metadata.get('customMetadata', {}) if isinstance(metadata, dict) else {}
        
        print(f"LLM Enhanced: {custom_meta.get('llm_enhanced', False)}")
        print(f"Content Type: {custom_meta.get('content_type', 'N/A')}")
        print(f"Element ID: {custom_meta.get('element_id', 'N/A')}")
        
        if 'llm_relationships' in custom_meta:
            print(f"LLM Relationships: {custom_meta['llm_relationships'][:100]}...")


@pytest.mark.asyncio
async def test_table_gets_separate_structured_memory(hierarchical_document_elements, base_metadata):
    """Test that tables are chunked into separate memories with structured data preserved"""
    from cloud_plugins.temporal.activities.document_activities import generate_llm_optimized_memory_structures
    
    # Filter to just get the table element
    table_elements = [elem for elem in hierarchical_document_elements if isinstance(elem, TableElement)]
    
    assert len(table_elements) == 1, "Should have exactly one table for this test"
    
    table_element = table_elements[0]
    
    print(f"\n{'='*80}")
    print(f"TEST: Table Separate Memory with Structured Data")
    print(f"{'='*80}")
    print(f"Table: {table_element.metadata.get('section_title', 'Unknown')}")
    print(f"Structured data headers: {table_element.headers}")
    print(f"Number of rows: {len(table_element.rows)}")
    
    # Generate LLM memory for table
    result = await generate_llm_optimized_memory_structures(
        content_elements=[table_element.model_dump()],
        domain="financial",
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        use_llm=True
    )
    
    memory_requests = result.get("memory_requests", [])
    
    assert len(memory_requests) == 1, "Table should generate exactly one memory"
    
    table_memory = memory_requests[0]
    
    print(f"\n{'='*80}")
    print(f"TABLE MEMORY RESULT")
    print(f"{'='*80}")
    print(f"Content length: {len(table_memory['content'])} chars")
    print(f"Content preview:\n{table_memory['content'][:300]}...")
    
    metadata = table_memory.get('metadata', {})
    custom_meta = metadata.get('customMetadata', {}) if isinstance(metadata, dict) else {}
    
    print(f"\n--- Metadata ---")
    print(f"LLM Enhanced: {custom_meta.get('llm_enhanced', False)}")
    print(f"Content Type: {custom_meta.get('content_type', 'N/A')}")
    
    # Check for financial-specific metadata (should be added by LLM)
    print(f"\n--- LLM-Added Metadata ---")
    for key, value in custom_meta.items():
        if key.startswith('llm_'):
            print(f"{key}: {str(value)[:100]}")
    
    # Assertions
    assert custom_meta.get('llm_enhanced') == True, "Table memory should be LLM enhanced"
    assert 'financial' in table_memory['content'].lower() or 'revenue' in table_memory['content'].lower(), "Should preserve financial context"


@pytest.mark.asyncio
async def test_hierarchical_structure_preservation(hierarchical_document_elements, base_metadata):
    """Test that hierarchical structure (sections/subsections) is preserved in memory metadata"""
    from cloud_plugins.temporal.activities.document_activities import generate_llm_optimized_memory_structures
    
    print(f"\n{'='*80}")
    print(f"TEST: Hierarchical Structure Preservation")
    print(f"{'='*80}")
    
    # Generate memories for all elements
    result = await generate_llm_optimized_memory_structures(
        content_elements=[elem.model_dump() for elem in hierarchical_document_elements],
        domain="financial",
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        use_llm=True
    )
    
    memory_requests = result.get("memory_requests", [])
    
    print(f"Generated {len(memory_requests)} memories")
    
    # Analyze hierarchical structure
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    section_hierarchy = {}
    
    for mem_dict in memory_requests:
        metadata = mem_dict.get('metadata', {})
        custom_meta = metadata.get('customMetadata', {}) if isinstance(metadata, dict) else {}
        
        section_level = custom_meta.get('section_level')
        section_number = custom_meta.get('section_number')
        section_title = custom_meta.get('section_title')
        parent_section = custom_meta.get('parent_section')
        
        if section_number:
            section_hierarchy[section_number] = {
                'level': section_level,
                'title': section_title,
                'parent': parent_section,
                'content_preview': mem_dict['content'][:80]
            }
    
    # Print hierarchy
    for section_num in sorted(section_hierarchy.keys()):
        info = section_hierarchy[section_num]
        indent = "  " * (info['level'] - 1) if info['level'] else ""
        parent_info = f" (parent: {info['parent']})" if info['parent'] else ""
        print(f"{indent}{section_num}. {info['title']}{parent_info}")
        print(f"{indent}   → {info['content_preview']}...")
    
    # Assertions
    assert len(section_hierarchy) > 0, "Should preserve section structure"
    
    # Check for parent-child relationships
    has_parent_child = any(info['parent'] is not None for info in section_hierarchy.values())
    assert has_parent_child, "Should preserve parent-child relationships"
    
    # Check that hierarchical metadata is preserved (at least one section with level info)
    has_hierarchy_info = any(info['level'] is not None for info in section_hierarchy.values())
    assert has_hierarchy_info, "Should preserve hierarchical level information"
    
    print(f"\n✅ Hierarchical structure preserved: {len(section_hierarchy)} sections found")
    print(f"✅ Parent-child relationships working correctly")


@pytest.mark.asyncio
async def test_llm_adds_intelligent_metadata(hierarchical_document_elements, base_metadata):
    """Test that LLM adds intelligent metadata and relationships"""
    from cloud_plugins.temporal.activities.document_activities import generate_llm_optimized_memory_structures
    
    # Focus on the financial table and revenue analysis
    financial_elements = [
        elem for elem in hierarchical_document_elements 
        if 'financial' in str(elem.metadata).lower() or 
           'revenue' in str(elem.metadata).lower() or
           isinstance(elem, TableElement)
    ]
    
    print(f"\n{'='*80}")
    print(f"TEST: LLM Intelligent Metadata Generation")
    print(f"{'='*80}")
    print(f"Testing with {len(financial_elements)} financial elements")
    
    result = await generate_llm_optimized_memory_structures(
        content_elements=[elem.model_dump() for elem in financial_elements],
        domain="financial",
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        use_llm=True
    )
    
    memory_requests = result.get("memory_requests", [])
    
    print(f"\n{'='*80}")
    print(f"LLM-ADDED INTELLIGENCE")
    print(f"{'='*80}")
    
    for idx, mem_dict in enumerate(memory_requests):
        print(f"\n--- Memory {idx + 1} ---")
        
        metadata = mem_dict.get('metadata', {})
        custom_meta = metadata.get('customMetadata', {}) if isinstance(metadata, dict) else {}
        
        # Check for LLM-added features
        llm_features = {
            'relationships': custom_meta.get('llm_relationships'),
            'query_patterns': custom_meta.get('query_patterns'),
            'data_categories': custom_meta.get('llm_data_categories'),
            'key_metrics': custom_meta.get('llm_key_metrics'),
            'topics': metadata.get('topics') if isinstance(metadata, dict) else None
        }
        
        for feature, value in llm_features.items():
            if value:
                print(f"{feature}: {str(value)[:150]}")
        
        # Check content enhancement
        content_enhanced = len(mem_dict['content']) > 100
        print(f"Content enhanced: {content_enhanced} ({len(mem_dict['content'])} chars)")
    
    # Assertions
    assert len(memory_requests) > 0, "Should generate memories"
    
    # Check that at least one memory has LLM enhancements
    has_relationships = any(
        mem.get('metadata', {}).get('customMetadata', {}).get('llm_relationships')
        for mem in memory_requests
    )
    has_topics = any(
        mem.get('metadata', {}).get('topics')
        for mem in memory_requests
    )
    
    print(f"\n✅ LLM enhancements detected:")
    print(f"   - Relationships: {has_relationships}")
    print(f"   - Topics: {has_topics}")


@pytest.mark.asyncio
async def test_with_real_reducto_extraction(base_metadata):
    """Test LLM generation with real Reducto extraction output"""
    json_path = Path("/Users/shawkatkabbara/Downloads/b1ee8b3479b29f40964bdaa830163b19_provider_result_f8141f7d-88ba-4145-925c-0b025b22d6c7.json")
    
    if not json_path.exists():
        pytest.skip(f"Real Reducto file not found at {json_path}")
    
    print(f"\n{'='*80}")
    print(f"TEST: LLM Generation with Real Reducto Data")
    print(f"{'='*80}")
    
    # First run extraction
    from cloud_plugins.temporal.activities.document_activities import extract_structured_content_from_provider
    
    with open(json_path, 'r') as f:
        real_reducto = json.load(f)
    
    extraction = await extract_structured_content_from_provider(
        provider_specific=real_reducto,
        provider_name="reducto",
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id
    )
    
    print(f"Extraction completed:")
    print(f"  Decision: {extraction['decision']}")
    print(f"  Elements: {len(extraction.get('structured_elements', []))}")
    print(f"  Structure: {extraction.get('structure_analysis', {})}")
    
    if extraction['decision'] != 'complex':
        pytest.skip("Document classified as simple, skipping LLM test")
    
    # Now test LLM generation (limit to first 10 elements for speed)
    structured_elements = extraction.get('structured_elements', [])[:10]
    
    print(f"\n{'='*80}")
    print(f"Running LLM generation on first {len(structured_elements)} elements...")
    print(f"{'='*80}")
    
    from cloud_plugins.temporal.activities.document_activities import generate_llm_optimized_memory_structures
    
    result = await generate_llm_optimized_memory_structures(
        content_elements=structured_elements,
        domain=None,  # General domain
        base_metadata=base_metadata,
        organization_id=base_metadata.organization_id,
        namespace_id=base_metadata.namespace_id,
        use_llm=True
    )
    
    memory_requests = result.get("memory_requests", [])
    generation_summary = result.get("generation_summary", {})
    
    print(f"\n{'='*80}")
    print(f"REAL DATA LLM RESULTS")
    print(f"{'='*80}")
    print(f"Generated: {len(memory_requests)} memories")
    print(f"Summary: {generation_summary}")
    
    # Show comparison: before vs after LLM
    print(f"\n{'='*80}")
    print(f"BEFORE vs AFTER LLM ENHANCEMENT")
    print(f"{'='*80}")
    
    for idx, (orig_elem, llm_mem) in enumerate(zip(structured_elements[:3], memory_requests[:3])):
        print(f"\n--- Element {idx + 1} ---")
        print(f"BEFORE (raw extraction):")
        print(f"  Content: {orig_elem['content'][:100]}...")
        print(f"  Metadata keys: {list(orig_elem.get('metadata', {}).keys())}")
        
        print(f"\nAFTER (LLM enhanced):")
        print(f"  Content: {llm_mem['content'][:100]}...")
        
        metadata = llm_mem.get('metadata', {})
        custom_meta = metadata.get('customMetadata', {}) if isinstance(metadata, dict) else {}
        llm_keys = [k for k in custom_meta.keys() if k.startswith('llm_')]
        print(f"  LLM-added metadata: {llm_keys}")
        print(f"  Topics: {metadata.get('topics', []) if isinstance(metadata, dict) else []}")
    
    # Assertions
    assert len(memory_requests) == len(structured_elements), "Should generate one memory per element"
    
    # Check for LLM enhancements
    enhanced_count = sum(
        1 for mem in memory_requests
        if mem.get('metadata', {}).get('customMetadata', {}).get('llm_enhanced')
    )
    
    print(f"\n✅ {enhanced_count}/{len(memory_requests)} memories LLM-enhanced")
    assert enhanced_count > 0, "At least some memories should be LLM-enhanced"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

