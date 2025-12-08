"""
Enhanced Reducto Memory Transformer

This module transforms Reducto AI responses into optimized memory objects
that can be effectively stored and queried in Papr Memory.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from models.memory_models import AddMemoryRequest
from models.shared_types import MemoryMetadata, MemoryType
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


@dataclass
class ReductoBlock:
    """Represents a block from Reducto response"""
    content: str
    type: str
    confidence: float
    bounding_box: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReductoChunk:
    """Represents a chunk from Reducto response"""
    content: str
    blocks: List[ReductoBlock]
    metadata: Optional[Dict[str, Any]] = None


class ReductoMemoryTransformer:
    """
    Transforms Reducto AI responses into optimized memory objects
    """

    def __init__(self):
        self.content_type_weights = {
            'title': 1.0,
            'heading': 0.9,
            'text': 0.7,
            'table': 0.8,
            'code': 0.6,
            'formula': 0.7,
            'list': 0.6,
            'paragraph': 0.5
        }

    def transform_reducto_response_to_memories(
        self,
        reducto_response: Dict[str, Any],
        base_metadata: MemoryMetadata,  # Accept MemoryMetadata object directly
        organization_id: str,
        namespace_id: str,
        user_id: Optional[str] = None  # Keep for backward compat but optional
    ) -> List[AddMemoryRequest]:
        """
        Transform Reducto response into a list of optimized memory requests
        
        Args:
            reducto_response: Raw Reducto API response
            base_metadata: MemoryMetadata object with user IDs already set correctly
            organization_id: Organization ID
            namespace_id: Namespace ID  
            user_id: (Deprecated) Use base_metadata.user_id or base_metadata.external_user_id instead
        """
        logger.info("Transforming Reducto response to memory objects")

        memories = []
        
        # Normalize/sanitize base metadata custom fields
        if hasattr(base_metadata, 'customMetadata') and base_metadata.customMetadata:
            base_custom = self._sanitize_custom_metadata(base_metadata.customMetadata)
        else:
            base_custom = {}
        
        # Extract chunks from response
        chunks = self._extract_chunks_from_response(reducto_response)
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunk_memories = self._process_chunk_to_memories(
                chunk, chunk_idx, base_metadata, base_custom, organization_id, namespace_id
            )
            memories.extend(chunk_memories)
        
        # Create summary memory
        summary_memory = self._create_document_summary_memory(
            reducto_response, base_metadata, base_custom, organization_id, namespace_id
        )
        if summary_memory:
            memories.insert(0, summary_memory)  # Add summary first
        
        logger.info(f"Created {len(memories)} memory objects from Reducto response")
        return memories

    def _extract_chunks_from_response(self, response: Dict[str, Any]) -> List[ReductoChunk]:
        """Extract chunks from Reducto response"""
        chunks = []
        
        # Navigate the response structure
        result = response.get("result", {})
        if isinstance(result, dict):
            result_chunks = result.get("chunks", [])
        else:
            # Handle case where result is the chunks directly
            result_chunks = result if isinstance(result, list) else []
        
        for chunk_data in result_chunks:
            if hasattr(chunk_data, 'blocks'):
                blocks = chunk_data.blocks
            elif isinstance(chunk_data, dict) and 'blocks' in chunk_data:
                blocks = chunk_data['blocks']
            else:
                # Fallback: treat as single block
                blocks = [chunk_data]
            
            # Convert blocks to ReductoBlock objects
            reducto_blocks = []
            for block in blocks:
                if hasattr(block, 'content'):
                    content = block.content
                    block_type = getattr(block, 'type', 'text')
                    confidence = getattr(block, 'confidence', 0.9)
                    bounding_box = getattr(block, 'bounding_box', None)
                    metadata = getattr(block, 'metadata', {})
                elif isinstance(block, dict):
                    content = block.get('content', '')
                    block_type = block.get('type', 'text')
                    confidence = block.get('confidence', 0.9)
                    bounding_box = block.get('bounding_box')
                    metadata = block.get('metadata', {})
                else:
                    continue
                
                reducto_blocks.append(ReductoBlock(
                    content=content,
                    type=block_type,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    metadata=metadata
                ))
            
            if reducto_blocks:
                chunks.append(ReductoChunk(
                    content="\n".join([b.content for b in reducto_blocks]),
                    blocks=reducto_blocks,
                    metadata=chunk_data if hasattr(chunk_data, '__dict__') else {}
                ))
        
        return chunks

    def _process_chunk_to_memories(
        self,
        chunk: ReductoChunk,
        chunk_idx: int,
        base_metadata: MemoryMetadata,
        base_custom: Dict[str, Any],
        organization_id: str,
        namespace_id: str
    ) -> List[AddMemoryRequest]:
        """Process a single chunk into memory objects
        
        Args:
            chunk: Reducto chunk to process
            chunk_idx: Index of chunk in document
            base_metadata: MemoryMetadata with user IDs already set
            base_custom: Sanitized customMetadata dict
            organization_id: Organization ID
            namespace_id: Namespace ID
        """
        memories = []
        
        # Group blocks by type for better memory organization
        block_groups = self._group_blocks_by_type(chunk.blocks)
        
        for group_type, blocks in block_groups.items():
            if not blocks:
                continue
            
            # Create memory based on content type
            if group_type == 'table':
                memory = self._create_table_memory(
                    blocks, chunk_idx, base_metadata, organization_id, namespace_id
                )
            elif group_type in ['title', 'heading']:
                memory = self._create_title_memory(
                    blocks, chunk_idx, base_metadata, organization_id, namespace_id
                )
            elif group_type == 'code':
                memory = self._create_code_memory(
                    blocks, chunk_idx, base_metadata, organization_id, namespace_id
                )
            elif group_type == 'formula':
                memory = self._create_formula_memory(
                    blocks, chunk_idx, base_metadata, organization_id, namespace_id
                )
            else:
                memory = self._create_text_memory(
                    blocks, chunk_idx, base_metadata, organization_id, namespace_id
                )
            
            if memory:
                memories.append(memory)
        
        return memories

    def _group_blocks_by_type(self, blocks: List[ReductoBlock]) -> Dict[str, List[ReductoBlock]]:
        """Group blocks by their type"""
        groups = {}
        for block in blocks:
            block_type = block.type or 'text'
            if block_type not in groups:
                groups[block_type] = []
            groups[block_type].append(block)
        return groups

    def _create_table_memory(
        self,
        blocks: List[ReductoBlock],
        chunk_idx: int,
        base_metadata: MemoryMetadata,
        organization_id: str,
        namespace_id: str
    ) -> AddMemoryRequest:
        """Create memory for table content"""
        table_content = "\n".join([block.content for block in blocks])
        
        # Extract table structure
        table_structure = self._parse_table_structure(table_content)
        
        # Create enhanced content with table analysis
        enhanced_content = f"Table Data:\n{table_content}\n\nTable Structure: {json.dumps(table_structure, ensure_ascii=False)}"
        
        # Use helper to create metadata with preserved user IDs
        metadata = self._create_metadata_from_base(
            base_metadata=base_metadata,
            base_custom={},  # Will be merged below
            organization_id=organization_id,
            namespace_id=namespace_id,
            additional_custom={
                "content_type": "table",
                "chunk_index": chunk_idx,
                "table_structure": table_structure,
                "row_count": table_structure.get("row_count", 0),
                "column_count": table_structure.get("column_count", 0),
                "source": "reducto_table_extraction"
            }
        )
        
        return AddMemoryRequest(
            content=enhanced_content,
            type=MemoryType.DOCUMENT,
            metadata=metadata
        )

    def _create_title_memory(
        self,
        blocks: List[ReductoBlock],
        chunk_idx: int,
        base_metadata: MemoryMetadata,
        organization_id: str,
        namespace_id: str
    ) -> AddMemoryRequest:
        """Create memory for title/heading content"""
        title_content = "\n".join([block.content for block in blocks])
        
        # Extract key concepts from title
        concepts = self._extract_concepts_from_text(title_content)
        
        enhanced_content = f"Document Section: {title_content}\n\nKey Concepts: {', '.join(concepts)}"
        
        metadata = self._create_metadata_from_base(
            base_metadata=base_metadata,
            base_custom={},
            organization_id=organization_id,
            namespace_id=namespace_id,
            additional_custom={
                "content_type": "title",
                "chunk_index": chunk_idx,
                "concepts": concepts,
                "section_level": self._determine_section_level(title_content),
                "source": "reducto_title_extraction"
            }
        )
        
        return AddMemoryRequest(
            content=enhanced_content,
            type=MemoryType.DOCUMENT,
            metadata=metadata
        )

    def _create_code_memory(
        self,
        blocks: List[ReductoBlock],
        chunk_idx: int,
        base_metadata: MemoryMetadata,
        organization_id: str,
        namespace_id: str
    ) -> AddMemoryRequest:
        """Create memory for code content"""
        code_content = "\n".join([block.content for block in blocks])
        
        # Detect programming language
        language = self._detect_programming_language(code_content)
        
        enhanced_content = f"Code Block ({language}):\n```{language}\n{code_content}\n```"
        
        metadata = self._create_metadata_from_base(
            base_metadata=base_metadata,
            base_custom={},
            organization_id=organization_id,
            namespace_id=namespace_id,
            additional_custom={
                "content_type": "code",
                "chunk_index": chunk_idx,
                "programming_language": language,
                "code_length": len(code_content),
                "source": "reducto_code_extraction"
            }
        )
        
        return AddMemoryRequest(
            content=enhanced_content,
            type=MemoryType.DOCUMENT,
            metadata=metadata
        )

    def _create_formula_memory(
        self,
        blocks: List[ReductoBlock],
        chunk_idx: int,
        base_metadata: MemoryMetadata,
        organization_id: str,
        namespace_id: str
    ) -> AddMemoryRequest:
        """Create memory for mathematical formula content"""
        formula_content = "\n".join([block.content for block in blocks])
        
        # Extract mathematical concepts
        math_concepts = self._extract_math_concepts(formula_content)
        
        enhanced_content = f"Mathematical Formula:\n{formula_content}\n\nMathematical Concepts: {', '.join(math_concepts)}"
        
        metadata = self._create_metadata_from_base(
            base_metadata=base_metadata,
            base_custom={},
            organization_id=organization_id,
            namespace_id=namespace_id,
            additional_custom={
                "content_type": "formula",
                "chunk_index": chunk_idx,
                "math_concepts": math_concepts,
                "formula_complexity": self._assess_formula_complexity(formula_content),
                "source": "reducto_formula_extraction"
            }
        )
        
        return AddMemoryRequest(
            content=enhanced_content,
            type=MemoryType.DOCUMENT,
            metadata=metadata
        )

    def _create_text_memory(
        self,
        blocks: List[ReductoBlock],
        chunk_idx: int,
        base_metadata: MemoryMetadata,
        organization_id: str,
        namespace_id: str
    ) -> AddMemoryRequest:
        """Create memory for general text content"""
        text_content = "\n".join([block.content for block in blocks])
        
        # Extract key entities and concepts
        entities = self._extract_entities_from_text(text_content)
        concepts = self._extract_concepts_from_text(text_content)
        
        enhanced_content = f"{text_content}\n\nKey Entities: {', '.join(entities)}\nKey Concepts: {', '.join(concepts)}"
        
        metadata = self._create_metadata_from_base(
            base_metadata=base_metadata,
            base_custom={},
            organization_id=organization_id,
            namespace_id=namespace_id,
            additional_custom={
                "content_type": "text",
                "chunk_index": chunk_idx,
                "entities": entities,
                "concepts": concepts,
                "text_length": len(text_content),
                "source": "reducto_text_extraction"
            }
        )
        
        return AddMemoryRequest(
            content=enhanced_content,
            type=MemoryType.DOCUMENT,
            metadata=metadata
        )

    def _create_metadata_from_base(
        self,
        base_metadata: MemoryMetadata,
        base_custom: Dict[str, Any],
        organization_id: str,
        namespace_id: str,
        additional_custom: Dict[str, Any]
    ) -> MemoryMetadata:
        """Create new MemoryMetadata preserving user IDs from base_metadata
        
        Args:
            base_metadata: Base metadata with user IDs already set
            base_custom: Base customMetadata dict
            organization_id: Organization ID
            namespace_id: Namespace ID
            additional_custom: Additional custom metadata to merge
            
        Returns:
            New MemoryMetadata with preserved user IDs and merged custom metadata
        """
        # Merge custom metadata
        merged_custom = {**base_custom, **additional_custom}
        
        # Create new metadata preserving user IDs from base
        return MemoryMetadata(
            organization_id=organization_id,
            namespace_id=namespace_id,
            user_id=base_metadata.user_id,  # Preserve internal user_id if set
            external_user_id=base_metadata.external_user_id,  # Preserve external_user_id if set
            customMetadata=self._sanitize_custom_metadata(merged_custom)
        )

    def _create_document_summary_memory(
        self,
        response: Dict[str, Any],
        base_metadata: MemoryMetadata,
        base_custom: Dict[str, Any],
        organization_id: str,
        namespace_id: str
    ) -> Optional[AddMemoryRequest]:
        """Create a summary memory for the entire document"""
        try:
            # Extract document metadata
            usage = response.get("usage", {})
            total_pages = usage.get("num_pages", 0)
            credits_used = usage.get("credits", 0)
            
            # Create summary content
            summary_content = f"Document Summary:\n"
            summary_content += f"- Total Pages: {total_pages}\n"
            summary_content += f"- Processing Credits Used: {credits_used}\n"
            summary_content += f"- Processing Provider: Reducto AI\n"
            summary_content += f"- Processing Timestamp: {datetime.now().isoformat()}\n"
            
            # Add content type breakdown
            chunks = self._extract_chunks_from_response(response)
            content_types = {}
            for chunk in chunks:
                for block in chunk.blocks:
                    block_type = block.type or 'text'
                    content_types[block_type] = content_types.get(block_type, 0) + 1
            
            if content_types:
                summary_content += f"- Content Types: {json.dumps(content_types, indent=2)}\n"
            
            # Create metadata preserving user IDs from base
            metadata = self._create_metadata_from_base(
                base_metadata=base_metadata,
                base_custom=base_custom,
                organization_id=organization_id,
                namespace_id=namespace_id,
                additional_custom={
                    "content_type": "document_summary",
                    "total_pages": total_pages,
                    "credits_used": credits_used,
                    "content_type_breakdown": content_types,
                    "source": "reducto_document_summary"
                }
            )
            
            return AddMemoryRequest(
                content=summary_content,
                type=MemoryType.DOCUMENT,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create document summary memory: {e}")
            return None

    def _sanitize_custom_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata so it conforms to MemoryMetadata constraints.
        - Drop None values
        - Flatten nested customMetadata if present
        - Convert dict/tuple/set to JSON strings
        - Keep only primitives and lists of primitives
        """
        if not isinstance(meta, dict):
            return {}
        result: Dict[str, Any] = {}
        # Flatten nested customMetadata
        nested = meta.get("customMetadata")
        if isinstance(nested, dict):
            meta = {**{k: v for k, v in meta.items() if k != "customMetadata"}, **nested}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                result[k] = v
            elif isinstance(v, list):
                cleaned_list = []
                for item in v:
                    if isinstance(item, (str, int, float, bool)):
                        cleaned_list.append(item)
                    else:
                        cleaned_list.append(str(item))
                result[k] = cleaned_list
            elif isinstance(v, (dict, tuple, set)):
                try:
                    result[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    result[k] = str(v)
            else:
                result[k] = str(v)
        return result

    # Helper methods for content analysis
    def _parse_table_structure(self, table_content: str) -> Dict[str, Any]:
        """Parse table structure from content"""
        lines = table_content.split('\n')
        if not lines:
            return {"row_count": 0, "column_count": 0}
        
        # Simple table parsing
        rows = [line.split('\t') if '\t' in line else line.split('|') for line in lines if line.strip()]
        if not rows:
            return {"row_count": 0, "column_count": 0}
        
        max_cols = max(len(row) for row in rows) if rows else 0
        
        return {
            "row_count": len(rows),
            "column_count": max_cols,
            "has_headers": len(rows) > 1,
            "structure": "tabular"
        }

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction - in practice would use NLP
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts = list(set(words))[:5]  # Top 5 concepts
        return concepts

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text"""
        # Simple entity extraction - in practice would use NER
        entities = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
        return list(set(entities))[:5]  # Top 5 entities

    def _determine_section_level(self, title: str) -> int:
        """Determine section level from title"""
        if title.isupper():
            return 1
        elif title.startswith(('1.', '2.', '3.')):
            return 2
        elif title.startswith(('a.', 'b.', 'c.')):
            return 3
        else:
            return 2

    def _detect_programming_language(self, code: str) -> str:
        """Detect programming language from code"""
        # Simple language detection
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code or 'const ' in code:
            return 'javascript'
        elif 'public class' in code or 'private ' in code:
            return 'java'
        elif '#include' in code or 'int main' in code:
            return 'c'
        else:
            return 'unknown'

    def _extract_math_concepts(self, formula: str) -> List[str]:
        """Extract mathematical concepts from formula"""
        math_keywords = ['derivative', 'integral', 'sum', 'limit', 'function', 'equation', 'matrix', 'vector']
        concepts = [kw for kw in math_keywords if kw in formula.lower()]
        return concepts

    def _assess_formula_complexity(self, formula: str) -> str:
        """Assess formula complexity"""
        if len(formula) > 100:
            return 'high'
        elif len(formula) > 50:
            return 'medium'
        else:
            return 'low'


# Factory function for easy usage
def transform_reducto_to_memories(
    reducto_response: Dict[str, Any],
    base_metadata: Dict[str, Any],
    organization_id: str,
    namespace_id: str,
    user_id: str
) -> List[AddMemoryRequest]:
    """Factory function to transform Reducto response to memory objects
    
    Args:
        reducto_response: Full Reducto API response JSON
        base_metadata: Either a MemoryMetadata object or a dict to construct one
        organization_id: Organization ID
        namespace_id: Namespace ID  
        user_id: User ID (for backward compat; prefer setting in base_metadata)
    """
    transformer = ReductoMemoryTransformer()
    
    # Coerce base_metadata to MemoryMetadata if it's a dict
    if isinstance(base_metadata, dict):
        base_metadata = MemoryMetadata(**base_metadata)
    
    return transformer.transform_reducto_response_to_memories(
        reducto_response, base_metadata, organization_id, namespace_id, user_id
    )
