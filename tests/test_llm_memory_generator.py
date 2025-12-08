import asyncio
from models.hierarchical_models import ContentElement, ContentType
from core.document_processing.llm_memory_generator import generate_optimized_memory_structures
from models.shared_types import MemoryMetadata


def test_generate_optimized_memory_structures_general():
    elements = [
        ContentElement(content_type=ContentType.TEXT, element_id="e1", content="Some content", metadata={}),
    ]

    memories = asyncio.run(generate_optimized_memory_structures(elements, domain=None, base_metadata=MemoryMetadata()))
    assert isinstance(memories, list)
    assert len(memories) >= 1
    for m in memories:
        assert hasattr(m, "content") and m.content


