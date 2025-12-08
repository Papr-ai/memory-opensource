"""
Pydantic models for TensorLake Document AI API responses.
Based on: https://docs.tensorlake.ai/document-ingestion/parsing/parsed-document-reference
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ParseStatus(str, Enum):
    """Status of a parse job"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESSFUL = "successful"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class Chunk(BaseModel):
    """A chunk of layout text extracted from the document"""
    content: str = Field(description="The markdown content of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the chunk")
    
    class Config:
        extra = "allow"  # Allow additional fields from API


class PageFragment(BaseModel):
    """A fragment (object) on a page"""
    fragment_type: str = Field(description="Type of fragment: section_header, title, text, table, figure, etc.")
    reading_order: int = Field(description="Reading order of the fragment")
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    content: str = Field(description="Actual content of the fragment")
    
    class Config:
        extra = "allow"


class PageDimensions(BaseModel):
    """Page dimensions"""
    width: float
    height: float


class Page(BaseModel):
    """Layout information for a single page"""
    page_number: int = Field(description="Page number")
    dimensions: Optional[PageDimensions] = Field(default=None, description="Width and height in pixels")
    page_fragments: Optional[List[PageFragment]] = Field(default=None, description="Objects on the page")
    
    class Config:
        extra = "allow"


class PageClass(BaseModel):
    """Page classification information"""
    page_class: str = Field(description="Classification name")
    page_numbers: List[int] = Field(description="Page numbers matching this classification")


class StructuredData(BaseModel):
    """Structured data extracted from document"""
    data: Dict[str, Any] = Field(description="Extracted structured data matching schema")
    page_numbers: List[int] = Field(description="Pages where data was extracted")
    schema_name: str = Field(description="Name of the schema provided by user")


class Usage(BaseModel):
    """Usage statistics for the parse job"""
    pages_parsed: Optional[int] = None
    signature_detected_pages: Optional[int] = None
    strikethrough_detected_pages: Optional[int] = None
    ocr_input_tokens_used: Optional[int] = None
    ocr_output_tokens_used: Optional[int] = None
    extraction_input_tokens_used: Optional[int] = None
    extraction_output_tokens_used: Optional[int] = None
    summarization_input_tokens_used: Optional[int] = None
    summarization_output_tokens_used: Optional[int] = None
    
    class Config:
        extra = "allow"


class ParseResult(BaseModel):
    """
    Complete parsed document result from TensorLake Document AI.
    Based on their official API documentation.
    """
    # Parsed document specific fields
    chunks: Optional[List[Chunk]] = Field(
        default=None,
        description="Chunks of layout text extracted from the document. Contains markdown content."
    )
    pages: Optional[List[Page]] = Field(
        default=None,
        description="Layout of the document with bounding boxes and structure information."
    )
    page_classes: Optional[List[PageClass]] = Field(
        default=None,
        description="Page classifications extracted from the document."
    )
    structured_data: Optional[List[StructuredData]] = Field(
        default=None,
        description="Structured data extracted according to provided schemas."
    )
    
    # Parse details
    parse_id: str = Field(description="Unique identifier for the parse job")
    parsed_pages_count: int = Field(description="Number of pages parsed successfully", ge=0)
    total_pages: Optional[int] = Field(default=None, description="Total number of pages")
    status: ParseStatus = Field(description="Status of the parse job")
    created_at: str = Field(description="Creation timestamp in RFC 3339 format")
    finished_at: Optional[str] = Field(default=None, description="Completion timestamp in RFC 3339 format")
    error: Optional[str] = Field(default=None, description="Error message if any")
    labels: Optional[Dict[str, Any]] = Field(default=None, description="Labels associated with parse job")
    usage: Optional[Usage] = Field(default=None, description="Usage statistics")
    
    class Config:
        extra = "allow"  # Allow additional fields from API
    
    def get_full_text(self) -> str:
        """
        Extract full text content from all chunks.
        This is the primary way to get document content from TensorLake.
        """
        if not self.chunks:
            return ""
        
        return "\n".join(chunk.content for chunk in self.chunks if chunk.content)
    
    def get_chunk_contents(self) -> List[str]:
        """Get list of all chunk contents as separate strings"""
        if not self.chunks:
            return []
        
        return [chunk.content for chunk in self.chunks if chunk.content]

