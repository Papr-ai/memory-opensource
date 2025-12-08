"""
Provider adapter that normalizes provider-specific outputs into
structured elements for hierarchical processing and LLM generation.
"""

from typing import Dict, Any, List
import re

from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


def extract_structured_elements(
    provider_name: str,
    provider_specific: Dict[str, Any],
    base_metadata: Dict[str, Any],
    organization_id: str,
    namespace_id: str
) -> List[Any]:
    """Return a list of ContentElement (from models.hierarchical_models)
    normalized from the provider-specific payload.

    For Reducto: we leverage reducto_memory_transformer to create AddMemoryRequests
    then convert each to a TextElement to keep a uniform path for downstream
    LLM generation.

    For TensorLake: we rely on ProviderContentExtractor to build elements.
    """
    from models.hierarchical_models import TextElement

    name = (provider_name or "").lower()

    if name == "reducto":
        try:
            # Use Reducto SDK's typed response for proper traversal
            from models.hierarchical_models import TextElement, TableElement, ImageElement
            from reducto.types.shared.pipeline_response import PipelineResponse
            
            elements: List[Any] = []
            
            # Parse with Reducto SDK if we have a raw dict
            if isinstance(provider_specific, dict):
                try:
                    pipeline = PipelineResponse(**provider_specific)
                except Exception as parse_err:
                    logger.warning(f"Failed to parse as PipelineResponse, trying raw dict: {parse_err}")
                    pipeline = provider_specific
            else:
                pipeline = provider_specific
            
            # Navigate to chunks using Reducto SDK structure
            chunks = []
            if hasattr(pipeline, 'result') and pipeline.result:
                # PipelineResponse.result.parse.result.chunks
                if hasattr(pipeline.result, 'parse') and pipeline.result.parse:
                    if hasattr(pipeline.result.parse, 'result') and pipeline.result.parse.result:
                        if hasattr(pipeline.result.parse.result, 'chunks'):
                            chunks = pipeline.result.parse.result.chunks or []
            
            # Fallback to dict navigation if SDK parsing didn't work
            if not chunks and isinstance(provider_specific, dict):
                result = provider_specific.get("result", {})
                if isinstance(result, dict):
                    parse = result.get("parse", {})
                    if isinstance(parse, dict):
                        parse_result = parse.get("result", {})
                        if isinstance(parse_result, dict):
                            chunks = parse_result.get("chunks", [])
            
            logger.info(f"Processing Reducto response with {len(chunks)} chunks")
            
            # Iterate through all chunks (usually semantic sections or pages)
            for chunk_idx, chunk_data in enumerate(chunks):
                # Get blocks from this chunk using SDK accessor
                blocks = []
                if hasattr(chunk_data, 'blocks'):
                    blocks = chunk_data.blocks or []
                elif isinstance(chunk_data, dict):
                    blocks = chunk_data.get('blocks', [])
                
                if not blocks:
                    continue
                
                logger.debug(f"  Chunk {chunk_idx}: {len(blocks)} blocks")
                
                # Process each block within the chunk
                for block_idx, block in enumerate(blocks):
                    # Extract block data using Reducto SDK accessors
                    if hasattr(block, 'content'):
                        content = block.content
                        block_type = str(block.type) if hasattr(block, 'type') else 'text'
                        # Handle Reducto confidence (can be string like "high", "medium", "low" or numeric)
                        raw_confidence = block.confidence if hasattr(block, 'confidence') else 0.9
                        if isinstance(raw_confidence, (int, float)):
                            confidence = float(raw_confidence)
                        elif isinstance(raw_confidence, str):
                            # Map string confidence to numeric
                            confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.5}
                            confidence = confidence_map.get(raw_confidence.lower(), 0.9)
                        else:
                            confidence = 0.9
                    elif isinstance(block, dict):
                        content = block.get('content', '')
                        block_type = str(block.get('type', 'text'))
                        raw_confidence = block.get('confidence', 0.9)
                        if isinstance(raw_confidence, (int, float)):
                            confidence = float(raw_confidence)
                        elif isinstance(raw_confidence, str):
                            confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.5}
                            confidence = confidence_map.get(raw_confidence.lower(), 0.9)
                        else:
                            confidence = 0.9
                    else:
                        continue
                    
                    # Skip empty content
                    if not content or not str(content).strip():
                        continue
                    
                    # Build element metadata
                    elem_metadata = {
                        "chunk_index": chunk_idx,
                        "block_index": block_idx,
                        "block_type": block_type,
                        "confidence": confidence,
                        "provider": "reducto",
                        "organization_id": organization_id,
                        "namespace_id": namespace_id,
                        **base_metadata
                    }
                    
                    # Create appropriate element type based on block type
                    element_id = f"reducto_c{chunk_idx}_b{block_idx}"
                    content_str = str(content)
                    
                    # Map Reducto block types to our element types
                    block_type_lower = block_type.lower()
                    
                    if block_type_lower in ("table", "table_cell"):
                        # Try to parse table structure from content
                        structured_data = {"raw_content": content_str}
                        elements.append(TableElement(
                            element_id=element_id,
                            content=content_str,
                            metadata=elem_metadata,
                            structured_data=structured_data,
                            table_type="data_table"
                        ))
                    elif block_type_lower in ("image", "figure"):
                        # Extract image URL if available from block
                        image_url = None
                        if hasattr(block, 'url'):
                            image_url = block.url
                        elif hasattr(block, 'image_url'):
                            image_url = block.image_url
                        elif isinstance(block, dict):
                            image_url = block.get('url') or block.get('image_url')
                        
                        # Extract bbox (bounding box) coordinates for image location
                        bbox = None
                        if hasattr(block, 'bbox'):
                            bbox = block.bbox
                        elif isinstance(block, dict):
                            bbox = block.get('bbox')
                        
                        # Flatten bbox to avoid nested dict (simpler type system)
                        if bbox:
                            bbox_dict = bbox if isinstance(bbox, dict) else getattr(bbox, '__dict__', {})
                            # Flatten bbox coordinates into top-level metadata fields
                            if 'height' in bbox_dict:
                                elem_metadata['bbox_height'] = float(bbox_dict['height'])
                            if 'width' in bbox_dict:
                                elem_metadata['bbox_width'] = float(bbox_dict['width'])
                            if 'left' in bbox_dict:
                                elem_metadata['bbox_left'] = float(bbox_dict['left'])
                            if 'top' in bbox_dict:
                                elem_metadata['bbox_top'] = float(bbox_dict['top'])
                            if 'page' in bbox_dict:
                                elem_metadata['page_number'] = int(bbox_dict['page'])
                            if 'original_page' in bbox_dict:
                                elem_metadata['bbox_original_page'] = int(bbox_dict['original_page'])
                        
                        # Create image content with markdown format
                        # Always use markdown image syntax, even if URL is null
                        if image_url:
                            # Add markdown image link with actual URL
                            image_content = f"![{content_str}]({image_url})"
                        else:
                            # Add markdown image placeholder without URL (still shows as image in viewers)
                            image_content = f"![{content_str}](#)"
                        
                        # Add description as text below the image markdown
                        if content_str:
                            image_content = f"{image_content}\n\n*{content_str}*"
                        
                        elements.append(ImageElement(
                            element_id=element_id,
                            content=image_content,
                            metadata=elem_metadata,
                            image_url=image_url,
                            image_description=content_str
                        ))
                    else:
                        # Default to text element for text, heading, paragraph, list item, etc.
                        elements.append(TextElement(
                            element_id=element_id,
                            content=content_str,
                            metadata=elem_metadata
                        ))
            
            logger.info(f"Extracted {len(elements)} content elements from Reducto response")
            return elements
            
        except Exception as e:
            logger.warning(f"Reducto adapter failed, falling back to generic extractor: {e}", exc_info=True)
            # Fall back to generic path below

    # Handle other providers
    if name == "tensorlake":
        try:
            from models.hierarchical_models import ProviderContentExtractor
            return ProviderContentExtractor.extract_from_tensorlake(provider_specific)
        except Exception as e:
            logger.error(f"TensorLake adapter failed: {e}")
            return []
    elif name == "deepseek-ocr":
        return _extract_deepseek_elements(provider_specific, base_metadata, organization_id, namespace_id)
    elif name == "paddleocr":
        return _extract_paddleocr_elements(provider_specific, base_metadata, organization_id, namespace_id)
    elif name == "gemini":
        return _extract_gemini_elements(provider_specific, base_metadata, organization_id, namespace_id)
    else:
        try:
            from models.hierarchical_models import ProviderContentExtractor
            return ProviderContentExtractor.extract_from_reducto(provider_specific)
        except Exception as e:
            logger.error(f"Provider adapter failed for {provider_name}: {e}")
            return []



def provider_to_markdown(provider_name: str, provider_specific: Dict[str, Any]) -> str:
    """Render provider JSON into a human-readable Markdown string.

    Best-effort, provider-aware rendering. For Reducto, we traverse result.chunks[].blocks[]
    and emit text paragraphs, simple markdown tables (when detectable), and image links
    when URLs are present in the payload. Fallbacks produce plain text.
    """
    name = (provider_name or "").lower()
    lines: List[str] = []

    if name == "tensorlake":
        # TensorLake: extract content field if available
        content = provider_specific.get("content") or provider_specific.get("text", "")
        if content:
            logger.info(f"TensorLake: using content field ({len(content)} chars)")
            return content
        
        # Check full_result if present (from dereferenced parse_id)
        full_result = provider_specific.get("full_result", {})
        if full_result:
            content = full_result.get("content") or full_result.get("text", "")
            if content:
                logger.info(f"TensorLake: using full_result.content field ({len(content)} chars)")
                return content
        
        # Fallback: if we only have parse_id reference, create minimal content
        parse_id = provider_specific.get("parse_id")
        if parse_id:
            logger.warning(f"TensorLake: no content found, only parse_id reference: {parse_id}")
            return f"# Document\n\nTensorLake parse_id: {parse_id}\n\n*(Content not yet fetched)*"
    
    elif name in ("gemini", "geminivision"):
        # Gemini: extract content field if available
        content = provider_specific.get("content") or provider_specific.get("text", "")
        if content:
            logger.info(f"Gemini: using content field ({len(content)} chars)")
            return content
        
        # Fallback: if we only have metadata, create minimal content
        model = provider_specific.get("model", "unknown")
        logger.warning(f"Gemini: no content found, only metadata (model: {model})")
        return f"# Document\n\nProcessed with Gemini (model: {model})\n\n*(Content not available)*"
    
    elif name == "paddleocr":
        # PaddleOCR: extract text from results array
        results = provider_specific.get("results", [])
        if results:
            content_parts = [result.get("text", "") for result in results if result.get("text")]
            if content_parts:
                content = "\n\n".join(content_parts)
                logger.info(f"PaddleOCR: using results text ({len(content)} chars)")
                return content
        
        logger.warning("PaddleOCR: no text found in results")
        return f"# Document\n\nProcessed with PaddleOCR\n\n*(No text extracted)*"
    
    elif name == "deepseek-ocr":
        # DeepSeek-OCR: extract text from ocr_results array
        ocr_results = provider_specific.get("ocr_results", [])
        if ocr_results:
            content_parts = [result.get("text", "") for result in ocr_results if result.get("text")]
            if content_parts:
                content = "\n\n".join(content_parts)
                logger.info(f"DeepSeek-OCR: using ocr_results text ({len(content)} chars)")
                return content
        
        logger.warning("DeepSeek-OCR: no text found in ocr_results")
        return f"# Document\n\nProcessed with DeepSeek-OCR\n\n*(No text extracted)*"
    
    elif name == "reducto":
        result = (provider_specific or {}).get("result") or {}
        # Reducto SDK returns nested structure: result.parse.result.chunks
        parse_result = result.get("parse", {}).get("result", {})
        chunks = parse_result.get("chunks") or []
        
        if isinstance(chunks, list) and len(chunks) > 0:
            logger.info(f"Reducto: processing {len(chunks)} chunks from parse.result")
            for idx, ch in enumerate(chunks):
                # Optional section header per chunk
                lines.append(f"\n### Section {idx + 1}\n")
                blocks = ch.get("blocks") or []
                for blk in blocks:
                    btype = (blk.get("type") or "").lower()
                    content = blk.get("content") or ""
                    # Try table-like content detection
                    if btype in ("table",) or _looks_like_table(content):
                        lines.extend(_render_markdown_table(content))
                    elif btype in ("image", "figure"):
                        url = blk.get("url") or blk.get("image_url") or blk.get("src")
                        caption = (blk.get("caption") or content or "Image").strip()
                        if url:
                            lines.append(f"![{caption}]({url})\n")
                        else:
                            lines.append(caption + "\n")
                    else:
                        if content:
                            lines.append(content.strip() + "\n")
        else:
            logger.warning(f"Reducto: no chunks found in parse.result, available keys: {list(parse_result.keys())}")

        # Also include any top-level text if present
        top_content = provider_specific.get("content")
        if isinstance(top_content, str) and top_content.strip():
            lines.append("\n" + top_content.strip() + "\n")

        return "\n".join(lines).strip()

    # Default fallback: stringify
    try:
        import json
        return json.dumps(provider_specific, ensure_ascii=False, indent=2)
    except Exception:
        return str(provider_specific)


def _looks_like_table(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    # Simple heuristics: presence of multiple columns separators
    if "|" in text and re.search(r"\|.*\|", text):
        return True
    # multiple consecutive spaces across lines suggest columns
    lines = text.splitlines()
    hits = sum(1 for ln in lines if re.search(r"\s{3,}", ln))
    return hits >= 2


def _render_markdown_table(text: str) -> List[str]:
    """Render a crude markdown table from text. If the text already contains pipes,
    keep as-is; otherwise split by multi-spaces as columns."""
    out: List[str] = []
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return out
    # If already pipe-based, emit directly
    if any("|" in ln for ln in lines):
        out.extend(lines)
        if not lines[-1].endswith("\n"):
            out.append("")
        return out
    # Else split by multiple spaces
    rows = [[cell.strip() for cell in re.split(r"\s{2,}", ln) if cell.strip()] for ln in lines]
    if not rows:
        return out
    headers = rows[0]
    out.append("|" + "|".join(headers) + "|")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows[1:]:
        out.append("|" + "|".join(r) + "|")
    out.append("")
    return out


def _extract_deepseek_elements(
    provider_specific: Dict[str, Any],
    base_metadata: Dict[str, Any],
    organization_id: str,
    namespace_id: str
) -> List[Any]:
    """Extract content elements from DeepSeek-OCR response."""
    from models.hierarchical_models import TextElement
    
    elements = []
    ocr_results = provider_specific.get("ocr_results", [])
    
    for idx, page in enumerate(ocr_results):
        page_num = page.get("page_number", idx + 1)
        full_text = page.get("full_text", "")
        
        if full_text:
            elements.append(TextElement(
                element_id=f"deepseek_page_{page_num}",
                content=full_text,
                metadata={
                    "page_number": page_num,
                    "provider": "deepseek-ocr",
                    "organization_id": organization_id,
                    "namespace_id": namespace_id
                }
            ))
    
    return elements


def _extract_paddleocr_elements(
    provider_specific: Dict[str, Any],
    base_metadata: Dict[str, Any],
    organization_id: str,
    namespace_id: str
) -> List[Any]:
    """Extract content elements from PaddleOCR response."""
    from models.hierarchical_models import TextElement, TableElement
    
    elements = []
    results = provider_specific.get("results", [])
    
    for idx, result in enumerate(results):
        page_num = result.get("page_number", idx + 1)
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        
        # Check if this has table data
        if "tables" in metadata and metadata["tables"]:
            for table_idx, table in enumerate(metadata["tables"]):
                elements.append(TableElement(
                    element_id=f"paddleocr_table_{page_num}_{table_idx}",
                    content=table.get("html", ""),
                    metadata={
                        "page_number": page_num,
                        "provider": "paddleocr",
                        "table_index": table_idx,
                        "organization_id": organization_id,
                        "namespace_id": namespace_id
                    }
                ))
        
        # Add text content
        if text:
            elements.append(TextElement(
                element_id=f"paddleocr_page_{page_num}",
                content=text,
                metadata={
                    "page_number": page_num,
                    "provider": "paddleocr",
                    "confidence": result.get("confidence", 0.9),
                    "organization_id": organization_id,
                    "namespace_id": namespace_id
                }
            ))
    
    return elements


def _extract_gemini_elements(
    provider_specific: Dict[str, Any],
    base_metadata: Dict[str, Any],
    organization_id: str,
    namespace_id: str
) -> List[Any]:
    """Extract content elements from Gemini response."""
    from models.hierarchical_models import TextElement
    
    elements = []
    pages = provider_specific.get("pages", [])
    
    for page in pages:
        page_num = page.get("page_number", 1)
        content = page.get("content", "")
        
        if content:
            elements.append(TextElement(
                element_id=f"gemini_page_{page_num}",
                content=content,
                metadata={
                    "page_number": page_num,
                    "provider": "gemini",
                    "confidence": page.get("confidence", 0.9),
                    "organization_id": organization_id,
                    "namespace_id": namespace_id
                }
            ))
    
    return elements

