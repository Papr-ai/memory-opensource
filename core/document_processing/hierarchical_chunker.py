"""
Hierarchical document chunking processor that preserves semantic meaning and document structure
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from models.hierarchical_models import (
    ContentElement, TextElement, TableElement, ImageElement, ChartElement,
    HierarchicalSection, DocumentStructure, ContentType, ChunkingConfig,
    ExtractionConfig, ContentAnalysis, MultiModalContent
)
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

# Optional semantic similarity (falls back if not installed)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SEM_MODEL_AVAILABLE = True
except Exception:
    _SEM_MODEL_AVAILABLE = False


# Context extraction configuration
CONTEXT_CONFIG = {
    "context_chars_before": 400,  # Research-backed optimal size (Anthropic 2024)
    "context_chars_after": 400,
    "include_section_header": True,
    "max_context_elements": 3  # Max elements to look back/forward
}


def extract_element_with_context(
    elements: List[ContentElement],
    target_index: int,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract element with surrounding context (400 chars before/after).

    Based on 2025 research (Contextual Retrieval - Anthropic):
    - Tables and images need surrounding text context for better retrieval
    - 400 characters is the optimal context window size
    - Preserves semantic relationships between elements

    Args:
        elements: List of all content elements
        target_index: Index of the target element
        config: Optional configuration (uses CONTEXT_CONFIG defaults)

    Returns:
        Dictionary with:
        - element: The target element
        - context_before: Text before element (up to 400 chars)
        - context_after: Text after element (up to 400 chars)
        - section_context: Section header if available
        - context_elements_before: IDs of elements used for context_before
        - context_elements_after: IDs of elements used for context_after
    """
    if config is None:
        config = CONTEXT_CONFIG

    element = elements[target_index]

    # Extract text before
    context_before = ""
    context_elements_before = []
    chars_collected = 0

    for i in range(target_index - 1, -1, -1):
        if elements[i].content_type == ContentType.TEXT:
            text = elements[i].content
            needed_chars = config['context_chars_before'] - chars_collected

            if len(text) <= needed_chars:
                context_before = text + "\n\n" + context_before
                context_elements_before.append(elements[i].element_id)
                chars_collected += len(text)
            else:
                # Take last N characters
                context_before = "..." + text[-needed_chars:] + "\n\n" + context_before
                context_elements_before.append(elements[i].element_id)
                break

            if chars_collected >= config['context_chars_before']:
                break
            if len(context_elements_before) >= config['max_context_elements']:
                break

    # Extract text after
    context_after = ""
    context_elements_after = []
    chars_collected = 0

    for i in range(target_index + 1, len(elements)):
        if elements[i].content_type == ContentType.TEXT:
            text = elements[i].content
            needed_chars = config['context_chars_after'] - chars_collected

            if len(text) <= needed_chars:
                context_after = context_after + "\n\n" + text
                context_elements_after.append(elements[i].element_id)
                chars_collected += len(text)
            else:
                # Take first N characters
                context_after = context_after + "\n\n" + text[:needed_chars] + "..."
                context_elements_after.append(elements[i].element_id)
                break

            if chars_collected >= config['context_chars_after']:
                break
            if len(context_elements_after) >= config['max_context_elements']:
                break

    # Build section context
    section_context = ""
    if config['include_section_header']:
        section_title = element.metadata.get('section_title', '')
        section_level = element.metadata.get('section_level', 1)
        if section_title:
            section_context = f"{'#' * section_level} {section_title}\n\n"

    return {
        "element": element,
        "context_before": context_before.strip(),
        "context_after": context_after.strip(),
        "section_context": section_context.strip(),
        "context_elements_before": context_elements_before,
        "context_elements_after": context_elements_after,
        "has_context": bool(context_before or context_after)
    }


class HierarchicalChunker:
    """
    Advanced document chunker that preserves semantic meaning and document structure
    """

    def __init__(self, chunking_config: ChunkingConfig):
        self.config = chunking_config
        self.element_id_counter = 0
        self.section_id_counter = 0
        self._embedding_model = None  # Lazy init when/if used

    def _get_domain(self) -> Optional[str]:
        """Best-effort domain lookup from config without breaking existing types"""
        return getattr(self.config, "domain", None)

    def analyze_document_structure(self, pages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> DocumentStructure:
        """
        Analyze document pages and extract hierarchical structure
        """
        logger.info(f"Analyzing document structure for {len(pages)} pages")

        document_id = metadata.get("upload_id", f"doc_{datetime.now().isoformat()}")
        title = self._extract_document_title(pages)

        # Extract sections from all pages
        sections = self._extract_hierarchical_sections(pages)

        document_structure = DocumentStructure(
            document_id=document_id,
            title=title,
            total_pages=len(pages),
            sections=sections,
            metadata=metadata
        )

        logger.info(f"Extracted {len(sections)} top-level sections")
        return document_structure

    def hierarchical_chunk_document(
        self,
        document_content: Dict[str, Any],
        chunking_strategy: ChunkingConfig,
        content_types: List[str]
    ) -> ContentAnalysis:
        """
        Process document using hierarchical chunking strategy
        """
        logger.info("Starting hierarchical chunking process")

        pages = document_content.get("pages", [])
        metadata = document_content.get("metadata", {})

        # Step 1: Analyze document structure
        document_structure = self.analyze_document_structure(pages, metadata)

        # Step 2: Extract content elements from each section
        content_elements = []
        element_relationships = []

        for section in document_structure.sections:
            section_elements, section_relationships = self._process_section_elements(
                section, pages, content_types
            )
            content_elements.extend(section_elements)
            element_relationships.extend(section_relationships)

        # Step 3: Apply chunking strategy
        chunked_elements = self._apply_chunking_strategy(content_elements, chunking_strategy)

        # Step 4: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(chunked_elements, document_structure)

        processing_stats = {
            "total_elements": len(content_elements),
            "chunked_elements": len(chunked_elements),
            "processing_time": datetime.now().isoformat(),
            "chunking_strategy": chunking_strategy.strategy.value
        }

        return ContentAnalysis(
            document_structure=document_structure,
            content_elements=chunked_elements,
            element_relationships=element_relationships,
            processing_stats=processing_stats,
            quality_metrics=quality_metrics
        )

    def _extract_document_title(self, pages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract document title from first page"""
        if not pages:
            return None

        first_page_content = pages[0].get("content", "")
        lines = first_page_content.strip().split('\n')

        # Look for title-like content in first few lines
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                words = line.split()
                if len(words) >= 2 and len(words) <= 12:
                    return line

        return None

    def _extract_hierarchical_sections(self, pages: List[Dict[str, Any]]) -> List[HierarchicalSection]:
        """Extract hierarchical sections from document pages"""
        sections = []
        current_section = None
        section_stack = []

        for page_num, page in enumerate(pages):
            content = page.get("content", "")
            page_sections = self._identify_sections_in_page(content, page_num + 1)

            for section_info in page_sections:
                level = section_info["level"]
                title = section_info["title"]
                content_text = section_info["content"]

                # Create new section
                section = HierarchicalSection(
                    section_id=self._generate_section_id(),
                    title=title,
                    level=level,
                    elements=[],
                    subsections=[],
                    metadata={
                        "start_page": page_num + 1,
                        "content_length": len(content_text)
                    }
                )

                # Handle section hierarchy
                if level == 1:
                    # Top-level section
                    sections.append(section)
                    section_stack = [section]
                else:
                    # Find appropriate parent section
                    while len(section_stack) >= level:
                        section_stack.pop()

                    if section_stack:
                        parent_section = section_stack[-1]
                        parent_section.subsections.append(section)
                        section.parent_section_id = parent_section.section_id
                    else:
                        # Fallback to top level if hierarchy is broken
                        sections.append(section)

                    section_stack.append(section)

                current_section = section

        return sections

    def _identify_sections_in_page(self, content: str, page_num: int) -> List[Dict[str, Any]]:
        """Identify sections within a page based on headers and content structure"""
        sections = []
        lines = content.split('\n')

        current_section = {
            "level": 1,
            "title": f"Page {page_num}",
            "content": "",
            "start_line": 0
        }

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Detect headers (simple heuristics)
            header_level = self._detect_header_level(line, i, lines)

            if header_level > 0:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section.copy())

                # Start new section
                current_section = {
                    "level": header_level,
                    "title": line,
                    "content": "",
                    "start_line": i
                }
            else:
                # Add line to current section content
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        # If no headers detected, treat entire page as one section
        if not sections:
            sections.append({
                "level": 1,
                "title": f"Page {page_num} Content",
                "content": content,
                "start_line": 0
            })

        return sections

    def _detect_header_level(self, line: str, line_index: int, all_lines: List[str]) -> int:
        """Detect if a line is a header and determine its level"""
        line = line.strip()

        # 0) Domain-specific header patterns first
        domain_level = self._detect_domain_header_level(line)
        if domain_level > 0:
            return domain_level

        # Skip very short or very long lines
        if len(line) < 3 or len(line) > 200:
            return 0

        # Check for numbered headers (1., 1.1, A., etc.)
        numbered_header_patterns = [
            r'^\d+\.\s*[A-Z]',  # 1. Title
            r'^\d+\.\d+\s*[A-Z]',  # 1.1 Subtitle
            r'^[A-Z]\.\s*[A-Z]',  # A. Title
            r'^[IVX]+\.\s*[A-Z]',  # Roman numerals
        ]

        for i, pattern in enumerate(numbered_header_patterns):
            if re.match(pattern, line):
                return i + 1

        # Check for capitalized headers
        if line.isupper() and len(line.split()) <= 8:
            return 1

        # Check for title case headers
        words = line.split()
        if len(words) <= 8 and all(word[0].isupper() for word in words if word):
            # Look at surrounding context
            if line_index > 0 and line_index < len(all_lines) - 1:
                prev_line = all_lines[line_index - 1].strip()
                next_line = all_lines[line_index + 1].strip()

                # Header if surrounded by empty lines or different formatting
                if not prev_line or not next_line:
                    return 2

        return 0

    def _detect_domain_header_level(self, line: str) -> int:
        """Domain-aware header detection using lightweight regex/keyword rules.

        Returns 1/2 for probable headers or 0 if not matched.
        """
        domain = (self._get_domain() or "").lower()
        text = line.lower()

        if domain in ("finance", "financial"):
            finance_keys_lvl1 = [
                "management's discussion", "md&a", "risk factors",
                "consolidated financial statements", "financial statements",
                "notes to consolidated", "balance sheet", "income statement",
                "cash flows", "results of operations"
            ]
            finance_keys_lvl2 = [
                "revenues", "operating expenses", "r&d", "research and development",
                "ebitda", "gross margin", "operating margin", "profit margin"
            ]
            if any(k in text for k in finance_keys_lvl1):
                return 1
            if any(k in text for k in finance_keys_lvl2):
                return 2

        if domain in ("healthcare", "medical"):
            hc_lvl1 = ["patient history", "test results", "imaging", "mri", "ct", "x-ray"]
            hc_lvl2 = ["blood test", "labs", "hemoglobin", "cbc", "panel", "findings", "impression"]
            if any(k in text for k in hc_lvl1):
                return 1
            if any(k in text for k in hc_lvl2):
                return 2

        if domain in ("science", "scientific", "research"):
            sci_lvl1 = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
            sci_lvl2 = ["figure", "table", "experiment", "evaluation", "metrics"]
            if any(text.startswith(k) for k in sci_lvl1):
                return 1
            if any(k in text for k in sci_lvl2):
                return 2

        return 0

    def _process_section_elements(
        self,
        section: HierarchicalSection,
        pages: List[Dict[str, Any]],
        content_types: List[str]
    ) -> Tuple[List[ContentElement], List[Dict[str, Any]]]:
        """Process elements within a section"""
        elements = []
        relationships = []

        # For now, create a text element for the section content
        # In a full implementation, this would analyze the content and extract
        # tables, images, etc. based on the content_types parameter

        if section.title or any(page.get("content", "") for page in pages):
            # Create text element for section
            content_text = ""
            if section.title:
                content_text += section.title + "\n\n"

            # Add content from relevant pages (simplified for now)
            for page in pages:
                page_content = page.get("content", "")
                if page_content and section.title and section.title.lower() in page_content.lower():
                    content_text += page_content

            if content_text.strip():
                text_element = TextElement(
                    element_id=self._generate_element_id(),
                    content=content_text.strip(),
                    semantic_role="section_content",
                    metadata={
                        "section_id": section.section_id,
                        "section_title": section.title,
                        "section_level": section.level
                    }
                )
                elements.append(text_element)

        # Process subsections recursively
        for subsection in section.subsections:
            sub_elements, sub_relationships = self._process_section_elements(
                subsection, pages, content_types
            )
            elements.extend(sub_elements)
            relationships.extend(sub_relationships)

            # Create relationships between parent and child sections
            if elements:
                relationships.append({
                    "type": "contains",
                    "source": section.section_id,
                    "target": subsection.section_id,
                    "metadata": {"relationship": "parent_section"}
                })

        return elements, relationships

    def _enrich_elements_with_context(
        self,
        content_elements: List[ContentElement]
    ) -> List[ContentElement]:
        """
        Enrich tables and images with surrounding text context.

        Based on 2025 research: Tables and images need context for better retrieval.
        Adds 400 chars before/after from surrounding text elements.

        Returns:
            List of elements with enriched content for tables/images
        """
        enriched_elements = []

        for i, element in enumerate(content_elements):
            # Only enrich tables and images
            if element.content_type not in [ContentType.TABLE, ContentType.IMAGE, ContentType.CHART]:
                enriched_elements.append(element)
                continue

            # Extract context
            context_data = extract_element_with_context(content_elements, i)

            if not context_data['has_context']:
                # No context available, keep element as-is
                enriched_elements.append(element)
                logger.debug(f"No context found for {element.content_type.value} element {element.element_id}")
                continue

            # Build enriched content with context wrapper
            enriched_content = ""

            # Add section context if available
            if context_data['section_context']:
                enriched_content += context_data['section_context'] + "\n"

            # Add context before
            if context_data['context_before']:
                enriched_content += "[Context before: " + context_data['context_before'] + "]\n\n"

            # Add original content
            enriched_content += element.content

            # Add context after
            if context_data['context_after']:
                enriched_content += "\n\n[Context after: " + context_data['context_after'] + "]"

            # Create new element with enriched content
            # Preserve all original attributes
            element_dict = {
                "element_id": element.element_id,
                "content_type": element.content_type,
                "content": enriched_content,
                "metadata": element.metadata.copy() if element.metadata else {}
            }

            # Store original content and context info in metadata
            element_dict["metadata"]["original_content"] = element.content
            element_dict["metadata"]["has_context_enrichment"] = True
            element_dict["metadata"]["context_before_length"] = len(context_data['context_before'])
            element_dict["metadata"]["context_after_length"] = len(context_data['context_after'])
            element_dict["metadata"]["context_elements_before"] = context_data['context_elements_before']
            element_dict["metadata"]["context_elements_after"] = context_data['context_elements_after']

            # Preserve type-specific fields
            if hasattr(element, 'structured_data') and element.structured_data:
                element_dict["structured_data"] = element.structured_data
            if hasattr(element, 'image_url') and element.image_url:
                element_dict["image_url"] = element.image_url
            if hasattr(element, 'image_description') and element.image_description:
                element_dict["image_description"] = element.image_description
            if hasattr(element, 'page_number'):
                element_dict["page_number"] = element.page_number
            if hasattr(element, 'position'):
                element_dict["position"] = element.position

            # Reconstruct element with proper type
            if element.content_type == ContentType.TABLE:
                enriched_element = TableElement(**element_dict)
            elif element.content_type == ContentType.IMAGE:
                enriched_element = ImageElement(**element_dict)
            elif element.content_type == ContentType.CHART:
                enriched_element = ChartElement(**element_dict)
            else:
                enriched_element = ContentElement(**element_dict)

            enriched_elements.append(enriched_element)
            logger.debug(f"Enriched {element.content_type.value} element {element.element_id} with {len(context_data['context_before']) + len(context_data['context_after'])} chars of context")

        # Log summary
        enriched_count = sum(1 for e in enriched_elements if e.metadata.get('has_context_enrichment', False))
        logger.info(f"Enriched {enriched_count}/{len(content_elements)} elements with surrounding context")

        return enriched_elements

    def _apply_chunking_strategy(
        self,
        content_elements: List[ContentElement],
        chunking_strategy: ChunkingConfig
    ) -> List[ContentElement]:
        """
        Apply the specified chunking strategy to content elements.

        NOTE: Elements should be enriched with context BEFORE chunking.
        """
        # Enrich tables and images with context first
        enriched_elements = self._enrich_elements_with_context(content_elements)

        # Apply chunking strategy to enriched elements
        if chunking_strategy.strategy.value == "semantic":
            return self._semantic_chunking(enriched_elements, chunking_strategy)
        elif chunking_strategy.strategy.value == "structural":
            return self._structural_chunking(enriched_elements, chunking_strategy)
        elif chunking_strategy.strategy.value == "hierarchical":
            return self._hierarchical_chunking(enriched_elements, chunking_strategy)
        else:  # hybrid
            return self._hybrid_chunking(enriched_elements, chunking_strategy)

    def _semantic_chunking(
        self,
        elements: List[ContentElement],
        config: ChunkingConfig
    ) -> List[ContentElement]:
        """Apply semantic chunking strategy - groups small elements and splits large ones"""
        chunked_elements = []
        current_chunk_content = []
        current_chunk_length = 0

        target_size = getattr(config, 'max_chunk_size', 6000)
        min_size = getattr(config, 'min_chunk_size', 1000)
        preserve_tables = getattr(config, 'preserve_tables', True)
        preserve_images = getattr(config, 'preserve_images', True)

        for element in elements:
            # Always preserve tables and images as separate chunks if configured
            if preserve_tables and element.content_type == ContentType.TABLE:
                # Flush current text chunk if any
                if current_chunk_content:
                    chunked_elements.append(self._merge_elements(current_chunk_content))
                    current_chunk_content = []
                    current_chunk_length = 0
                # Add table as separate chunk
                chunked_elements.append(element)
                continue

            if preserve_images and element.content_type == ContentType.IMAGE:
                # Flush current text chunk if any
                if current_chunk_content:
                    chunked_elements.append(self._merge_elements(current_chunk_content))
                    current_chunk_content = []
                    current_chunk_length = 0
                # Add image as separate chunk
                chunked_elements.append(element)
                continue

            element_length = len(element.content)

            # If element is larger than max_chunk_size, split it
            if element_length > target_size:
                # Flush current chunk first
                if current_chunk_content:
                    chunked_elements.append(self._merge_elements(current_chunk_content))
                    current_chunk_content = []
                    current_chunk_length = 0

                # Split large element
                chunks = self._split_element_semantically(element, config)
                chunked_elements.extend(chunks)
                continue

            # For normal-sized text elements: group until reaching target size
            # If adding this element would exceed target_size and we already have min_size
            if current_chunk_length + element_length > target_size and current_chunk_length >= min_size:
                # Flush current chunk
                chunked_elements.append(self._merge_elements(current_chunk_content))
                current_chunk_content = []
                current_chunk_length = 0

            # Add element to current chunk
            current_chunk_content.append(element)
            current_chunk_length += element_length

        # Flush remaining chunk
        if current_chunk_content:
            if len(current_chunk_content) == 1:
                # Single element, add as-is
                chunked_elements.append(current_chunk_content[0])
            else:
                # Merge multiple elements
                chunked_elements.append(self._merge_elements(current_chunk_content))

        return chunked_elements

    def _merge_elements(self, elements: List[ContentElement]) -> ContentElement:
        """Merge multiple ContentElements into a single chunked element"""
        if len(elements) == 1:
            return elements[0]

        # Combine content with double newline separator
        merged_content = "\n\n".join([e.content for e in elements])

        # Merge metadata
        merged_metadata = {
            "merged_from": [e.element_id for e in elements],
            "chunk_type": "semantic_group",
            "element_count": len(elements),
            **elements[0].metadata  # Use first element's metadata as base
        }

        # Create merged element
        merged_element = ContentElement(
            element_id=f"merged_{elements[0].element_id}",
            content_type=elements[0].content_type,
            content=merged_content,
            metadata=merged_metadata
        )

        return merged_element

    def _structural_chunking(
        self,
        elements: List[ContentElement],
        config: ChunkingConfig
    ) -> List[ContentElement]:
        """Apply structural chunking strategy"""
        # Respect document structure boundaries
        return elements  # Simplified - already structured by sections

    def _hierarchical_chunking(
        self,
        elements: List[ContentElement],
        config: ChunkingConfig
    ) -> List[ContentElement]:
        """Apply hierarchical chunking strategy - maintains section boundaries"""
        # Group elements by section to maintain hierarchical structure
        chunked_elements = []

        target_size = getattr(config, 'max_chunk_size', 6000)
        min_size = getattr(config, 'min_chunk_size', 1000)
        preserve_tables = getattr(config, 'preserve_tables', True)
        preserve_images = getattr(config, 'preserve_images', True)

        # Group elements by section
        section_groups: Dict[str, List[ContentElement]] = {}
        for element in elements:
            section_id = element.metadata.get('section_id', 'default')
            if section_id not in section_groups:
                section_groups[section_id] = []
            section_groups[section_id].append(element)

        # Process each section group
        for section_id, section_elements in section_groups.items():
            current_chunk_content = []
            current_chunk_length = 0

            for element in section_elements:
                # Always preserve tables and images as separate chunks if configured
                if preserve_tables and element.content_type == ContentType.TABLE:
                    # Flush current text chunk if any
                    if current_chunk_content:
                        chunked_elements.append(self._merge_elements(current_chunk_content))
                        current_chunk_content = []
                        current_chunk_length = 0
                    # Add table as separate chunk
                    chunked_elements.append(element)
                    continue

                if preserve_images and element.content_type == ContentType.IMAGE:
                    # Flush current text chunk if any
                    if current_chunk_content:
                        chunked_elements.append(self._merge_elements(current_chunk_content))
                        current_chunk_content = []
                        current_chunk_length = 0
                    # Add image as separate chunk
                    chunked_elements.append(element)
                    continue

                element_length = len(element.content)

                # If element is larger than max_chunk_size, split it
                if element_length > target_size:
                    # Flush current chunk first
                    if current_chunk_content:
                        chunked_elements.append(self._merge_elements(current_chunk_content))
                        current_chunk_content = []
                        current_chunk_length = 0

                    # Split large element while preserving section metadata
                    chunks = self._split_element_semantically(element, config)
                    chunked_elements.extend(chunks)
                    continue

                # For normal-sized text elements: group within section until reaching target size
                # If adding this element would exceed target_size and we already have min_size
                if current_chunk_length + element_length > target_size and current_chunk_length >= min_size:
                    # Flush current chunk (maintaining section boundary)
                    merged = self._merge_elements(current_chunk_content)
                    # Ensure section metadata is preserved
                    merged.metadata['section_id'] = section_id
                    merged.metadata['chunk_type'] = 'hierarchical_section'
                    chunked_elements.append(merged)
                    current_chunk_content = []
                    current_chunk_length = 0

                # Add element to current chunk
                current_chunk_content.append(element)
                current_chunk_length += element_length

            # Flush remaining chunk for this section
            if current_chunk_content:
                merged = self._merge_elements(current_chunk_content)
                # Ensure section metadata is preserved
                merged.metadata['section_id'] = section_id
                merged.metadata['chunk_type'] = 'hierarchical_section'
                chunked_elements.append(merged)

        logger.info(f"Hierarchical chunking: {len(elements)} elements → {len(chunked_elements)} chunks (grouped by section)")
        return chunked_elements

    def _hybrid_chunking(
        self,
        elements: List[ContentElement],
        config: ChunkingConfig
    ) -> List[ContentElement]:
        """Apply hybrid chunking strategy combining multiple approaches"""
        # First apply structural chunking, then semantic within structures
        structured_elements = self._structural_chunking(elements, config)
        return self._semantic_chunking(structured_elements, config)

    def _split_element_semantically(
        self,
        element: ContentElement,
        config: ChunkingConfig
    ) -> List[ContentElement]:
        """Split a large element into smaller chunks while preserving semantics"""
        content = element.content

        # Tokenize into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
        if not sentences:
            return [element]

        # If semantic model available, use similarity-aware splitting
        if _SEM_MODEL_AVAILABLE:
            try:
                if self._embedding_model is None:
                    # Lightweight, widely available model
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

                embeddings = self._embedding_model.encode(sentences, normalize_embeddings=True)

                max_size = getattr(config, 'max_chunk_size', 4000)  # Target 1-2 pages with Qwen 2650-dim embeddings
                overlap = getattr(config, 'overlap_size', 200)
                sim_threshold = getattr(config, 'semantic_similarity_threshold', 0.65)

                chunks_text: List[str] = []
                current = sentences[0]
                current_len = len(current)

                for i in range(1, len(sentences)):
                    sim = float(st_util.cos_sim(embeddings[i - 1], embeddings[i]))
                    next_sentence = sentences[i]
                    # Start a new chunk if similarity drops or we would exceed size
                    if sim < sim_threshold or (current_len + 1 + len(next_sentence)) > max_size:
                        chunks_text.append(current.strip())
                        # Overlap tail from current as context
                        overlap_text = self._get_overlap_text(current, overlap)
                        current = (overlap_text + " " + next_sentence).strip() if overlap_text else next_sentence
                        current_len = len(current)
                    else:
                        current += ". " + next_sentence
                        current_len += 2 + len(next_sentence)

                if current.strip():
                    chunks_text.append(current.strip())

                # Build ContentElements
                out: List[ContentElement] = []
                for idx, ch in enumerate(chunks_text):
                    out.append(self._create_chunk_element(element, ch, idx, config))
                return out
            except Exception as e:
                logger.warning(f"Semantic splitting fallback due to error: {e}")

        # Fallback: length-aware sentence packing without embeddings
        chunks: List[ContentElement] = []
        max_size = getattr(config, 'max_chunk_size', 4000)  # Target 1-2 pages with Qwen 2650-dim embeddings
        overlap = getattr(config, 'overlap_size', 200)
        current_chunk = ""
        chunk_count = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 2 > max_size and current_chunk:
                # Create chunk
                chunk_element = self._create_chunk_element(
                    element, current_chunk, chunk_count, config
                )
                chunks.append(chunk_element)

                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk, overlap)
                chunk_count += 1

            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence

        # Add final chunk
        if current_chunk.strip():
            chunk_element = self._create_chunk_element(
                element, current_chunk, chunk_count, config
            )
            chunks.append(chunk_element)

        return chunks

    def _create_chunk_element(
        self,
        original_element: ContentElement,
        chunk_content: str,
        chunk_index: int,
        config: ChunkingConfig
    ) -> ContentElement:
        """Create a new chunk element from original element"""
        chunk_metadata = original_element.metadata.copy()
        chunk_metadata.update({
            "original_element_id": original_element.element_id,
            "chunk_index": chunk_index,
            "is_chunk": True
        })

        # Create new element of same type
        if isinstance(original_element, TextElement):
            return TextElement(
                element_id=self._generate_element_id(),
                content=chunk_content.strip(),
                semantic_role=original_element.semantic_role,
                language=original_element.language,
                confidence=original_element.confidence,
                metadata=chunk_metadata,
                page_number=original_element.page_number,
                position=original_element.position,
                parent_element_id=original_element.element_id
            )
        else:
            # Default to generic ContentElement
            return ContentElement(
                element_id=self._generate_element_id(),
                content_type=original_element.content_type,
                content=chunk_content.strip(),
                metadata=chunk_metadata,
                page_number=original_element.page_number,
                position=original_element.position,
                parent_element_id=original_element.element_id
            )

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= overlap_size:
            return text

        # Try to get overlap at sentence boundary
        sentences = text.split('. ')
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= overlap_size:
                overlap_text = sentence + ". " + overlap_text
            else:
                break

        return overlap_text.strip()

    def _calculate_quality_metrics(
        self,
        elements: List[ContentElement],
        document_structure: DocumentStructure
    ) -> Dict[str, float]:
        """Calculate quality metrics for the chunking process"""
        total_content_length = sum(len(element.content) for element in elements)
        avg_chunk_size = total_content_length / len(elements) if elements else 0

        # Calculate coherence score (simplified)
        coherence_score = 0.8  # Placeholder - would use semantic analysis

        # Calculate structure preservation score
        structure_score = len(document_structure.sections) / max(len(elements), 1)
        structure_score = min(structure_score, 1.0)

        return {
            "average_chunk_size": avg_chunk_size,
            "total_chunks": len(elements),
            "coherence_score": coherence_score,
            "structure_preservation_score": structure_score,
            "processing_quality": (coherence_score + structure_score) / 2
        }

    def _generate_element_id(self) -> str:
        """Generate unique element ID"""
        self.element_id_counter += 1
        return f"element_{self.element_id_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _generate_section_id(self) -> str:
        """Generate unique section ID"""
        self.section_id_counter += 1
        return f"section_{self.section_id_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class MultiModalContentExtractor:
    """
    Extractor for multi-modal content (tables, images, charts)
    """

    def __init__(self, extraction_config: ExtractionConfig):
        self.config = extraction_config

    def extract_multimodal_content(
        self,
        page_content: Dict[str, Any],
        extraction_config: ExtractionConfig
    ) -> MultiModalContent:
        """
        Extract and classify different content types from a page
        """
        logger.info("Extracting multi-modal content from page")

        text_elements = []
        table_elements = []
        image_elements = []
        chart_elements = []

        content = page_content.get("content", "")
        page_num = page_content.get("page_number", 1)

        # Extract text elements (basic implementation)
        if content.strip():
            text_element = TextElement(
                element_id=f"text_{page_num}_{datetime.now().strftime('%H%M%S')}",
                content=content,
                page_number=page_num,
                metadata={"extracted_from": "page_content"}
            )
            text_elements.append(text_element)

        # Extract tables (placeholder - would integrate with actual table detection)
        detected_tables = self._detect_tables(content, page_num)
        table_elements.extend(detected_tables)

        # Extract images and charts (placeholder)
        if extraction_config.extract_images:
            images = self._detect_images(page_content, page_num)
            image_elements.extend(images)

        if extraction_config.analyze_charts:
            charts = self._detect_charts(page_content, page_num)
            chart_elements.extend(charts)

        return MultiModalContent(
            text_elements=text_elements,
            table_elements=table_elements,
            image_elements=image_elements,
            chart_elements=chart_elements,
            metadata={
                "page_number": page_num,
                "extraction_config": extraction_config.model_dump(),
                "extraction_timestamp": datetime.now().isoformat()
            }
        )

    def _detect_tables(self, content: str, page_num: int) -> List[TableElement]:
        """Detect and extract tables from content"""
        tables = []

        # Simple table detection based on patterns
        # In practice, this would use more sophisticated table detection
        lines = content.split('\n')
        potential_table_lines = []

        for line in lines:
            # Look for lines with multiple columns (simplified heuristic)
            if '\t' in line or '|' in line or re.search(r'\s{3,}', line):
                potential_table_lines.append(line)
            elif potential_table_lines:
                # End of potential table
                if len(potential_table_lines) >= 3:  # Minimum table size
                    table = self._parse_table_lines(potential_table_lines, page_num)
                    if table:
                        tables.append(table)
                potential_table_lines = []

        # Check for final table
        if len(potential_table_lines) >= 3:
            table = self._parse_table_lines(potential_table_lines, page_num)
            if table:
                tables.append(table)

        return tables

    def _parse_table_lines(self, lines: List[str], page_num: int) -> Optional[TableElement]:
        """Parse detected table lines into structured data"""
        if not lines:
            return None

        # Simple parsing - split by common delimiters
        rows = []
        headers = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Split by tabs, pipes, or multiple spaces
            if '\t' in line:
                row = [cell.strip() for cell in line.split('\t')]
            elif '|' in line:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
            else:
                # Split by multiple spaces
                row = [cell.strip() for cell in re.split(r'\s{2,}', line) if cell.strip()]

            if not row:
                continue

            if i == 0:
                headers = row
            else:
                rows.append(row)

        if not headers or not rows:
            return None

        # Create structured data
        structured_data = {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers)
        }

        table_content = "\n".join(lines)

        return TableElement(
            element_id=f"table_{page_num}_{len(rows)}x{len(headers)}",
            content=table_content,
            structured_data=structured_data,
            headers=headers,
            rows=rows,
            page_number=page_num,
            metadata={
                "detected_method": "pattern_matching",
                "confidence": 0.7,  # Placeholder confidence
                "extraction_method": "simple_parsing"
            }
        )

    def _detect_images(self, page_content: Dict[str, Any], page_num: int) -> List[ImageElement]:
        """Detect and extract images from page content"""
        # Placeholder - would integrate with actual image detection
        images = []

        # Check if page content indicates images
        content = page_content.get("content", "")
        if "image" in content.lower() or "figure" in content.lower():
            # Create placeholder image element
            image_element = ImageElement(
                element_id=f"image_{page_num}_{datetime.now().strftime('%H%M%S')}",
                content=f"Image detected on page {page_num}",
                page_number=page_num,
                metadata={
                    "detection_method": "keyword_based",
                    "confidence": 0.5
                }
            )
            images.append(image_element)

        return images

    def _detect_charts(self, page_content: Dict[str, Any], page_num: int) -> List[ChartElement]:
        """Detect and extract charts from page content"""
        # Placeholder - would integrate with actual chart detection
        charts = []

        content = page_content.get("content", "")
        chart_keywords = ["chart", "graph", "plot", "diagram", "figure"]

        for keyword in chart_keywords:
            if keyword in content.lower():
                chart_element = ChartElement(
                    element_id=f"chart_{page_num}_{keyword}",
                    content=f"Chart detected on page {page_num}: {keyword}",
                    chart_type="unknown",
                    page_number=page_num,
                    metadata={
                        "detection_method": "keyword_based",
                        "detected_keyword": keyword,
                        "confidence": 0.4
                    }
                )
                charts.append(chart_element)
                break  # Only one chart per page for now

        return charts