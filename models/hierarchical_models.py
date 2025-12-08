"""
Data models for hierarchical chunking and multi-modal content processing

This module extends the existing Memory type system to support hierarchical
content extraction and multi-modal document processing.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from datetime import datetime

# Import existing types for integration
from models.shared_types import MemoryType, MemoryMetadata
from models.memory_models import AddMemoryRequest


class ExtendedMemoryType(str, Enum):
    """Extended memory types that include hierarchical content types

    This extends the existing MemoryType enum to support multi-modal content
    while maintaining backward compatibility.
    """
    # Existing types (for compatibility)
    TEXT = "text"
    CODE_SNIPPET = "code_snippet"
    DOCUMENT = "document"

    # New hierarchical content types
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    SECTION = "section"


class ContentType(str, Enum):
    """Types of content that can be processed (legacy support)"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    METADATA = "metadata"


class ChunkingStrategy(str, Enum):
    """Chunking strategies for different content types"""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


class StorageStrategy(str, Enum):
    """Storage strategies for different content types"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    DOCUMENT_ONLY = "document_only"
    HYBRID = "hybrid"
    TIME_SERIES = "time_series"


class ContentElement(BaseModel):
    """Base model for any content element"""
    element_id: str = Field(..., description="Unique identifier for this element")
    content_type: ContentType = Field(..., description="Type of content")
    content: str = Field(..., description="Raw content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Element-specific metadata")
    page_number: Optional[int] = Field(default=None, description="Page number where element appears")
    position: Optional[Dict[str, float]] = Field(default=None, description="Position on page (x, y, width, height)")
    parent_element_id: Optional[str] = Field(default=None, description="Parent element in hierarchy")
    child_element_ids: List[str] = Field(default_factory=list, description="Child elements")


class TextElement(ContentElement):
    """Text-based content element"""
    content_type: ContentType = ContentType.TEXT
    semantic_role: Optional[str] = Field(default=None, description="Semantic role (paragraph, title, etc.)")
    language: Optional[str] = Field(default="en", description="Language of text")
    confidence: Optional[float] = Field(default=None, description="OCR/extraction confidence")


class TableElement(ContentElement):
    """Table content element with structured data"""
    content_type: ContentType = ContentType.TABLE
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="Parsed table structure")
    headers: List[str] = Field(default_factory=list, description="Table column headers")
    rows: List[List[str]] = Field(default_factory=list, description="Table row data")
    table_type: Optional[str] = Field(default=None, description="Type of table (financial, clinical, etc.)")
    has_time_series: bool = Field(default=False, description="Whether table contains time-series data")


class ImageElement(ContentElement):
    """Image content element"""
    content_type: ContentType = ContentType.IMAGE
    image_url: Optional[str] = Field(default=None, description="URL to stored image")
    image_hash: Optional[str] = Field(default=None, description="Hash of image content")
    ocr_text: Optional[str] = Field(default=None, description="Text extracted from image")
    image_description: Optional[str] = Field(default=None, description="AI-generated image description")
    detected_objects: List[Dict[str, Any]] = Field(default_factory=list, description="Objects detected in image")


class ChartElement(ContentElement):
    """Chart/diagram content element"""
    content_type: ContentType = ContentType.CHART
    chart_type: Optional[str] = Field(default=None, description="Type of chart (bar, line, pie, etc.)")
    extracted_data: Optional[Dict[str, Any]] = Field(default=None, description="Data extracted from chart")
    chart_description: Optional[str] = Field(default=None, description="AI-generated chart description")
    data_series: List[Dict[str, Any]] = Field(default_factory=list, description="Chart data series")


class HierarchicalSection(BaseModel):
    """Hierarchical section of a document"""
    section_id: str = Field(..., description="Unique section identifier")
    title: Optional[str] = Field(default=None, description="Section title")
    level: int = Field(..., description="Hierarchical level (1=top level)")
    elements: List[ContentElement] = Field(default_factory=list, description="Content elements in section")
    subsections: List["HierarchicalSection"] = Field(default_factory=list, description="Child sections")
    parent_section_id: Optional[str] = Field(default=None, description="Parent section ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Section metadata")


class DocumentStructure(BaseModel):
    """Complete document structure"""
    document_id: str = Field(..., description="Unique document identifier")
    title: Optional[str] = Field(default=None, description="Document title")
    total_pages: int = Field(..., description="Total number of pages")
    sections: List[HierarchicalSection] = Field(default_factory=list, description="Top-level sections")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processing_timestamp: datetime = Field(default_factory=datetime.now, description="When structure was created")


class ChunkingConfig(BaseModel):
    """Configuration for hierarchical chunking"""
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.HIERARCHICAL, description="Chunking strategy")
    max_chunk_size: int = Field(default=4000, description="Maximum chunk size in characters")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")
    overlap_size: int = Field(default=200, description="Overlap between chunks")
    preserve_tables: bool = Field(default=True, description="Keep tables as single units")
    preserve_images: bool = Field(default=True, description="Keep images with context")
    semantic_threshold: float = Field(default=0.8, description="Semantic similarity threshold for grouping")
    respect_boundaries: bool = Field(default=True, description="Respect section boundaries")


class ExtractionConfig(BaseModel):
    """Configuration for multi-modal content extraction"""
    extract_tables: bool = Field(default=True, description="Extract and parse tables")
    extract_images: bool = Field(default=True, description="Process images and charts")
    ocr_images: bool = Field(default=True, description="Perform OCR on images")
    analyze_charts: bool = Field(default=True, description="Analyze charts and extract data")
    detect_headers: bool = Field(default=True, description="Detect section headers")
    preserve_formatting: bool = Field(default=True, description="Preserve text formatting")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for extraction")


class DomainContext(BaseModel):
    """Domain-specific context for processing"""
    domain: str = Field(..., description="Domain type (financial, healthcare, legal, etc.)")
    subdomain: Optional[str] = Field(default=None, description="Specific subdomain")
    entity_types: List[str] = Field(default_factory=list, description="Expected entity types")
    table_schemas: Dict[str, Any] = Field(default_factory=dict, description="Expected table structures")
    terminology: Dict[str, str] = Field(default_factory=dict, description="Domain-specific terminology")
    compliance_requirements: List[str] = Field(default_factory=list, description="Regulatory requirements")


class MemoryStructure(BaseModel):
    """Optimized memory structure for storage"""
    memory_id: str = Field(..., description="Unique memory identifier")
    content: str = Field(..., description="Memory content")
    title: Optional[str] = Field(default=None, description="Memory title")
    content_type: ContentType = Field(..., description="Type of content")
    storage_strategy: StorageStrategy = Field(..., description="Optimal storage strategy")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Relationships to other memories")
    embeddings_metadata: Dict[str, Any] = Field(default_factory=dict, description="Vector embedding metadata")
    graph_metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph storage metadata")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document storage metadata")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing information")


class HierarchicalProcessingConfig(BaseModel):
    """Complete configuration for hierarchical processing"""
    chunking_strategy: ChunkingConfig = Field(default_factory=ChunkingConfig, description="Chunking configuration")
    extraction_config: ExtractionConfig = Field(default_factory=ExtractionConfig, description="Extraction configuration")
    storage_strategy: StorageStrategy = Field(default=StorageStrategy.HYBRID, description="Storage strategy")
    domain_context: Optional[DomainContext] = Field(default=None, description="Domain-specific context")
    enable_llm_optimization: bool = Field(default=True, description="Use LLM for structure optimization")
    batch_size: int = Field(default=50, description="Batch size for memory creation")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")


class ContentAnalysis(BaseModel):
    """Analysis result of document content"""
    document_structure: DocumentStructure = Field(..., description="Hierarchical document structure")
    content_elements: List[ContentElement] = Field(..., description="All extracted content elements")
    element_relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Relationships between elements")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality assessment metrics")


class MultiModalContent(BaseModel):
    """Multi-modal content extraction result"""
    text_elements: List[TextElement] = Field(default_factory=list, description="Text content elements")
    table_elements: List[TableElement] = Field(default_factory=list, description="Table content elements")
    image_elements: List[ImageElement] = Field(default_factory=list, description="Image content elements")
    chart_elements: List[ChartElement] = Field(default_factory=list, description="Chart content elements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")


class ProcessingResult(BaseModel):
    """Result of structured data processing"""
    memory_structures: List[MemoryStructure] = Field(..., description="Generated memory structures")
    storage_assignments: Dict[str, StorageStrategy] = Field(..., description="Storage strategy assignments")
    relationship_graph: Dict[str, Any] = Field(default_factory=dict, description="Relationship graph")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class EnhancedProcessingResult(BaseModel):
    """Enhanced processing result with hierarchical content"""
    hierarchical_chunks: List[HierarchicalSection] = Field(..., description="Hierarchical content chunks")
    multimodal_content: List[MultiModalContent] = Field(..., description="Multi-modal content elements")
    structured_data: ProcessingResult = Field(..., description="Structured processing results")
    memory_structures: List[MemoryStructure] = Field(..., description="Optimized memory structures")
    creation_result: Dict[str, Any] = Field(..., description="Memory creation results")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


# Domain-specific models

class FinancialTableMemory(MemoryStructure):
    """Specialized memory structure for financial tables"""
    content_type: ContentType = ContentType.TABLE
    financial_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Financial entities")
    time_period: Optional[str] = Field(default=None, description="Time period covered")
    currency: Optional[str] = Field(default=None, description="Currency used")
    financial_metrics: Dict[str, float] = Field(default_factory=dict, description="Extracted financial metrics")


class MedicalTableMemory(MemoryStructure):
    """Specialized memory structure for medical/healthcare tables"""
    content_type: ContentType = ContentType.TABLE
    medical_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Medical entities")
    patient_identifiers: List[str] = Field(default_factory=list, description="Patient identifiers (anonymized)")
    clinical_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Clinical insights")
    compliance_tags: List[str] = Field(default_factory=list, description="Compliance/privacy tags")


# Integration with existing Memory system

class ProviderContentExtractor(BaseModel):
    """Extracts structured content from provider-specific results"""

    @staticmethod
    def extract_from_reducto(provider_specific: Dict[str, Any]) -> List['ContentElement']:
        """Extract structured content from Reducto provider results"""
        elements = []

        # Reducto likely provides structured data - examine the provider_specific field
        # This is where we would parse tables, charts, images that Reducto detects
        # For now, create a placeholder implementation

        if isinstance(provider_specific, dict):
            # Look for table data
            if 'tables' in provider_specific:
                for table_data in provider_specific['tables']:
                    elements.append(TableElement(
                        element_id=f"table_{len(elements)}",
                        content=str(table_data),
                        structured_data=table_data,
                        metadata={'source': 'reducto', 'type': 'table'}
                    ))

            # Look for image data
            if 'images' in provider_specific:
                for image_data in provider_specific['images']:
                    elements.append(ImageElement(
                        element_id=f"image_{len(elements)}",
                        content=image_data.get('description', ''),
                        image_url=image_data.get('url'),
                        image_description=image_data.get('description'),
                        metadata={'source': 'reducto', 'type': 'image'}
                    ))

        return elements

    @staticmethod
    def extract_from_tensorlake(provider_specific: Dict[str, Any]) -> List['ContentElement']:
        """Extract structured content from TensorLake provider results"""
        elements = []

        # TensorLake likely provides structured data - examine the provider_specific field
        # This is where we would parse tables, charts, images that TensorLake detects

        if isinstance(provider_specific, dict):
            # 1) Preferred: unified 'elements' format
            if 'elements' in provider_specific and isinstance(provider_specific['elements'], list):
                for element_data in provider_specific['elements']:
                    element_type = element_data.get('type', 'text')
                    if element_type == 'table':
                        elements.append(TableElement(
                            element_id=f"table_{len(elements)}",
                            content=str(element_data.get('content', '')),
                            structured_data=element_data,
                            metadata={'source': 'tensorlake', 'type': 'table'}
                        ))
                    elif element_type == 'image':
                        elements.append(ImageElement(
                            element_id=f"image_{len(elements)}",
                            content=element_data.get('description', '') or '',
                            image_url=element_data.get('url'),
                            image_description=element_data.get('description'),
                            metadata={'source': 'tensorlake', 'type': 'image'}
                        ))
                    else:
                        # Treat as text element
                        elements.append(TextElement(
                            element_id=f"text_{len(elements)}",
                            content=str(element_data.get('content', '')),
                            metadata={'source': 'tensorlake', 'type': 'text'}
                        ))

            # 2) Tables/images at top-level
            if not elements:
                if 'tables' in provider_specific and isinstance(provider_specific['tables'], list):
                    for table_data in provider_specific['tables']:
                        elements.append(TableElement(
                            element_id=f"table_{len(elements)}",
                            content=str(table_data),
                            structured_data=table_data if isinstance(table_data, dict) else {'raw': table_data},
                            metadata={'source': 'tensorlake', 'type': 'table'}
                        ))
                if 'images' in provider_specific and isinstance(provider_specific['images'], list):
                    for image_data in provider_specific['images']:
                        desc = image_data.get('description') if isinstance(image_data, dict) else None
                        url = image_data.get('url') if isinstance(image_data, dict) else None
                        elements.append(ImageElement(
                            element_id=f"image_{len(elements)}",
                            content=desc or '',
                            image_url=url,
                            image_description=desc,
                            metadata={'source': 'tensorlake', 'type': 'image'}
                        ))

            # 3) Blocks/pages style payloads
            if not elements:
                blocks = provider_specific.get('blocks') or provider_specific.get('segments')
                if isinstance(blocks, list):
                    for blk in blocks:
                        blk_type = (blk.get('type') if isinstance(blk, dict) else None) or 'text'
                        blk_text = blk.get('text') if isinstance(blk, dict) else str(blk)
                        elements.append(TextElement(
                            element_id=f"text_{len(elements)}",
                            content=str(blk_text or ''),
                            metadata={'source': 'tensorlake', 'type': blk_type}
                        ))
                pages = provider_specific.get('pages')
                if isinstance(pages, list):
                    for idx, pg in enumerate(pages, 1):
                        pg_text = pg.get('content') or pg.get('text') or ''
                        if pg_text:
                            elements.append(TextElement(
                                element_id=f"page_{idx}",
                                content=str(pg_text),
                                metadata={'source': 'tensorlake', 'type': 'page'},
                                page_number=idx
                            ))

            # 4) Plain content/text
            if not elements:
                content = provider_specific.get('content') or provider_specific.get('text')
                if content:
                    elements.append(TextElement(
                        element_id=f"text_{len(elements)}",
                        content=str(content),
                        metadata={'source': 'tensorlake', 'type': 'text'}
                    ))

        return elements


class MemoryTransformer(BaseModel):
    """Transforms hierarchical content elements into AddMemoryRequest objects"""

    @staticmethod
    def content_element_to_memory_request(
        element: 'ContentElement',
        base_metadata: Optional[MemoryMetadata] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> AddMemoryRequest:
        """Convert a ContentElement to an AddMemoryRequest"""

        # Determine the memory type based on content type
        memory_type_mapping = {
            ContentType.TEXT: MemoryType.TEXT,
            ContentType.TABLE: MemoryType.DOCUMENT,  # Use document type for tables until we extend MemoryType
            ContentType.IMAGE: MemoryType.DOCUMENT,   # Use document type for images
            ContentType.CHART: MemoryType.DOCUMENT,   # Use document type for charts
            ContentType.DIAGRAM: MemoryType.DOCUMENT,
            ContentType.LIST: MemoryType.TEXT,
            ContentType.HEADER: MemoryType.TEXT,
            ContentType.FOOTER: MemoryType.TEXT,
            ContentType.METADATA: MemoryType.TEXT
        }

        memory_type = memory_type_mapping.get(element.content_type, MemoryType.TEXT)

        # Create enhanced metadata
        enhanced_metadata = base_metadata.model_copy() if base_metadata else MemoryMetadata()

        # Add content-type specific metadata
        if not enhanced_metadata.customMetadata:
            enhanced_metadata.customMetadata = {}

        enhanced_metadata.customMetadata.update({
            'content_type': element.content_type.value,
            'element_id': element.element_id,
            'hierarchical_type': 'structured_content'
        })

        # Add element-specific metadata (flatten to avoid nested dicts)
        if element.metadata:
            # Only add scalar values from element.metadata (skip nested dicts/lists)
            for key, value in element.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    enhanced_metadata.customMetadata[key] = value
                elif value is None:
                    pass  # Skip None values
                # Skip complex types (dicts, lists) - they should be handled explicitly above

        # Add page information if available
        if element.page_number:
            enhanced_metadata.customMetadata['page_number'] = element.page_number

        # Add position information if available
        if element.position:
            enhanced_metadata.customMetadata['position'] = element.position

        # For structured content like tables, add the structured data to metadata
        if isinstance(element, TableElement) and element.structured_data:
            # Store structured_data as JSON string to avoid nested dict
            import json
            enhanced_metadata.customMetadata['structured_data_json'] = json.dumps(element.structured_data)
            enhanced_metadata.customMetadata['table_type'] = element.table_type
            enhanced_metadata.customMetadata['has_time_series'] = element.has_time_series

        elif isinstance(element, ImageElement):
            if element.image_url:
                enhanced_metadata.customMetadata['image_url'] = element.image_url
            if element.image_hash:
                enhanced_metadata.customMetadata['image_hash'] = element.image_hash
            if element.ocr_text:
                enhanced_metadata.customMetadata['ocr_text'] = element.ocr_text
            if element.image_description:
                enhanced_metadata.customMetadata['image_description'] = element.image_description

        elif isinstance(element, ChartElement):
            if element.chart_type:
                enhanced_metadata.customMetadata['chart_type'] = element.chart_type
            if element.extracted_data:
                enhanced_metadata.customMetadata['extracted_data'] = element.extracted_data
            if element.chart_description:
                enhanced_metadata.customMetadata['chart_description'] = element.chart_description

        return AddMemoryRequest(
            content=element.content,
            type=memory_type,
            metadata=enhanced_metadata,
            organization_id=organization_id,
            namespace_id=namespace_id,
            context=[],  # Add context if needed
            relationships_json=[]  # Add relationships if needed
        )

    @staticmethod
    def hierarchical_section_to_memory_requests(
        section: 'HierarchicalSection',
        base_metadata: Optional[MemoryMetadata] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> List[AddMemoryRequest]:
        """Convert a HierarchicalSection to multiple AddMemoryRequest objects"""

        memory_requests = []

        # Create a memory request for the section itself if it has content
        if section.title or section.elements:
            section_content = section.title or ""
            if section.elements:
                # Add a summary of the elements in the section
                element_summaries = []
                for element in section.elements:
                    element_summaries.append(f"- {element.content_type.value}: {element.content[:100]}...")
                section_content += "\n\nSection contains:\n" + "\n".join(element_summaries)

            section_metadata = base_metadata.model_copy() if base_metadata else MemoryMetadata()
            if not section_metadata.customMetadata:
                section_metadata.customMetadata = {}

            section_metadata.customMetadata.update({
                'section_id': section.section_id,
                'section_level': section.level,
                'section_title': section.title,
                'hierarchical_type': 'section',
                'element_count': len(section.elements)
            })

            memory_requests.append(AddMemoryRequest(
                content=section_content,
                type=MemoryType.DOCUMENT,
                metadata=section_metadata,
                organization_id=organization_id,
                namespace_id=namespace_id
            ))

        # Create memory requests for each element in the section
        for element in section.elements:
            memory_request = MemoryTransformer.content_element_to_memory_request(
                element, base_metadata, organization_id, namespace_id
            )

            # Add section context to the element metadata
            if memory_request.metadata and memory_request.metadata.customMetadata:
                memory_request.metadata.customMetadata.update({
                    'parent_section_id': section.section_id,
                    'parent_section_title': section.title,
                    'parent_section_level': section.level
                })

            memory_requests.append(memory_request)

        # Recursively process subsections
        for subsection in section.subsections:
            subsection_requests = MemoryTransformer.hierarchical_section_to_memory_requests(
                subsection, base_metadata, organization_id, namespace_id
            )
            memory_requests.extend(subsection_requests)

        return memory_requests


class DocumentToMemoryTransformer(BaseModel):
    """High-level transformer for converting processed documents to memory requests"""

    @staticmethod
    def process_document_result(
        processing_result: Dict[str, Any],  # From DocumentProvider.process_document
        base_metadata: Optional[MemoryMetadata] = None,
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> List[AddMemoryRequest]:
        """Convert a document processing result to memory requests"""

        memory_requests = []

        # Extract provider name and specific data
        provider_specific = processing_result.get('provider_specific', {})
        provider_name = processing_result.get('provider', 'unknown')

        # Extract structured content from provider results
        structured_elements = []

        if provider_name.lower() == 'reducto':
            structured_elements = ProviderContentExtractor.extract_from_reducto(provider_specific)
        elif provider_name.lower() == 'tensorlake':
            structured_elements = ProviderContentExtractor.extract_from_tensorlake(provider_specific)

        # If we found structured elements, create memory requests for them
        for element in structured_elements:
            memory_request = MemoryTransformer.content_element_to_memory_request(
                element, base_metadata, organization_id, namespace_id
            )
            memory_requests.append(memory_request)

        # Also create memory requests for the basic page content
        pages = processing_result.get('pages', [])
        for page in pages:
            page_metadata = base_metadata.model_copy() if base_metadata else MemoryMetadata()
            if not page_metadata.customMetadata:
                page_metadata.customMetadata = {}

            page_metadata.customMetadata.update({
                'page_number': page.get('page_number', 1),
                'confidence': page.get('confidence', 0.9),
                'hierarchical_type': 'page_content',
                'provider': provider_name
            })

            memory_requests.append(AddMemoryRequest(
                content=page.get('content', ''),
                type=MemoryType.DOCUMENT,
                metadata=page_metadata,
                organization_id=organization_id,
                namespace_id=namespace_id
            ))

        return memory_requests


# Update forward references
HierarchicalSection.model_rebuild()