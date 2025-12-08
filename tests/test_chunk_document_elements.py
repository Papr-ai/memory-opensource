"""Tests for the chunk_document_elements Temporal activity."""

from pathlib import Path
import json

import pytest


def _provider_blocks_to_elements(provider_data):
    """Convert provider JSON blocks into simplified structured elements."""

    type_map = {
        "header": "text",
        "title": "text",
        "text": "text",
        "table": "table",
        "figure": "image",
    }

    elements = []
    counter = 0

    parse_section = provider_data
    if "parse" in parse_section:
        parse_section = parse_section.get("parse", {})
    else:
        parse_section = provider_data.get("result", {}).get("parse", {})

    for chunk in parse_section.get("result", {}).get("chunks", []):
        for block in chunk.get("blocks", []):
            counter += 1

            raw_type = block.get("type", "text")
            content_type = type_map.get(raw_type.lower(), "text")

            bbox = block.get("bbox", {})
            metadata = {
                "page": bbox.get("page"),
                "source_bbox": bbox,
                "provider_block_type": raw_type,
            }

            element_dict = {
                "element_id": f"provider-{counter:03d}",
                "content_type": content_type,
                "content": block.get("content", ""),
                "metadata": metadata,
            }

            if content_type == "image" and block.get("image_url"):
                element_dict["image_url"] = block["image_url"]

            # If structured data exists, keep it; otherwise omit to reproduce real-world gaps
            structured_data = block.get("structured_data")
            if content_type == "table" and structured_data:
                element_dict["structured_data"] = structured_data

            elements.append(element_dict)

    return elements


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PROVIDER_RESULT_SAMPLE = json.loads(
    (FIXTURE_DIR / "provider_result_sample.json").read_text()
)
PROVIDER_STRUCTURED_ELEMENTS = _provider_blocks_to_elements(PROVIDER_RESULT_SAMPLE)


SAMPLE_STRUCTURED_ELEMENTS = [
    {
        "element_id": "text-001",
        "content_type": "text",
        "content": "This is a sample paragraph from the manual.",
        "metadata": {"page_number": 1},
    },
    {
        "element_id": "table-001",
        "content_type": "table",
        "content": "Column A,Column B\nValue 1,Value 2",
        "metadata": {"page_number": 1},
        # Note: structured_data is intentionally omitted to reproduce the regression scenario
    },
    {
        "element_id": "image-001",
        "content_type": "image",
        "content": "",
        "metadata": {"page_number": 1},
        "image_url": "https://example.com/image.png",
    },
]


@pytest.mark.asyncio
async def test_chunk_document_elements_inline_handles_missing_table_structured_data():
    """Inline elements without structured_data on tables should not raise validation errors."""

    from cloud_plugins.temporal.activities.document_activities import (
        chunk_document_elements,
    )

    result = await chunk_document_elements(
        content_elements=SAMPLE_STRUCTURED_ELEMENTS,
        chunking_config=None,
        post_id=None,
        extraction_stored=False,
    )

    assert "chunked_elements" in result
    assert result["stats"]["original_count"] == len(SAMPLE_STRUCTURED_ELEMENTS)
    assert result["stats"]["chunked_count"] >= 1


@pytest.mark.asyncio
async def test_chunk_document_elements_fetches_and_stores(monkeypatch):
    """When extraction is stored in Parse, the activity should fetch and process it."""

    from cloud_plugins.temporal.activities.document_activities import (
        chunk_document_elements,
    )

    async def fake_fetch(post_id: str):  # pragma: no cover - simple async stub
        assert post_id == "RV16B7tW3b"
        return {"structured_elements": PROVIDER_STRUCTURED_ELEMENTS}

    monkeypatch.setattr(
        "services.memory_management.fetch_extraction_result_from_post",
        fake_fetch,
    )

    class DummyMemoryGraph:  # pragma: no cover - simple stub
        async def ensure_async_connection(self):
            return None

    monkeypatch.setattr(
        "memory.memory_graph.MemoryGraph",
        lambda: DummyMemoryGraph(),
    )

    class DummyParseIntegration:  # pragma: no cover - simple stub
        def __init__(self, _memory_graph):
            pass

        async def upload_file(self, *_args, **_kwargs):
            return "https://example.com/chunked.json.gz"

        async def update_post(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "core.document_processing.parse_integration.ParseDocumentIntegration",
        DummyParseIntegration,
    )

    def fake_compress(_data):  # pragma: no cover - simple stub
        return b"compressed", 42.0

    monkeypatch.setattr(
        "services.memory_management.compress_extraction",
        fake_compress,
        raising=False,
    )

    result = await chunk_document_elements(
        content_elements=[],
        chunking_config=None,
        post_id="RV16B7tW3b",
        extraction_stored=True,
    )

    assert result["extraction_stored"] is True
    assert result["post_id"] == "RV16B7tW3b"
    assert result["stats"]["original_count"] == len(PROVIDER_STRUCTURED_ELEMENTS)
    assert result["stats"]["chunked_count"] >= 1

