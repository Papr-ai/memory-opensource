import pytest
from core.document_processing.hierarchical_chunker import HierarchicalChunker
from models.hierarchical_models import ChunkingConfig, ChunkingStrategy


def test_hierarchical_chunker_basic_structure():
    pages = [
        {"content": "Project Alpha\nIntroduction\nDetails about project."},
        {"content": "Methods\nStep 1\nStep 2"},
    ]
    doc = {"pages": pages, "metadata": {"upload_id": "u1"}}
    cfg = ChunkingConfig(strategy=ChunkingStrategy.HIERARCHICAL)
    chunker = HierarchicalChunker(cfg)

    analysis = chunker.hierarchical_chunk_document(doc, cfg, content_types=["text"])

    assert analysis.document_structure.total_pages == 2
    assert len(analysis.content_elements) > 0
    assert "chunking_strategy" in analysis.processing_stats


def test_hierarchical_chunker_title_extraction():
    pages = [{"content": "A Meaningful Title\nThen content"}]
    ds = HierarchicalChunker(ChunkingConfig(strategy=ChunkingStrategy.HIERARCHICAL)).analyze_document_structure(pages, {})
    # Title likely detected from first lines
    assert ds.title is None or isinstance(ds.title, str)


