import tiktoken
from typing import List, Dict, Any
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

def count_query_embedding_tokens(query_text: str) -> int:
    """
    Calculate token count for query embedding
    This approximates the tokens used for sentence embedding
    """
    try:
        # Use cl100k_base encoding (used by GPT-4 and similar models)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(query_text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error counting query embedding tokens: {e}")
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(query_text) // 4

def count_retrieved_memory_tokens(memory_items: List[Dict[str, Any]]) -> int:
    """
    Calculate total token count for retrieved memories
    """
    try:
        total_tokens = 0
        encoding = tiktoken.get_encoding("cl100k_base")
        
        for memory in memory_items:
            # Count tokens in content
            content = memory.get('content')
            if content and content is not None:
                content_tokens = len(encoding.encode(str(content)))
                total_tokens += content_tokens
            
            # Count tokens in metadata fields that might be relevant
            metadata_fields = ['title', 'topics', 'location', 'sourceUrl']
            for field in metadata_fields:
                field_value = memory.get(field)
                if field_value and field_value is not None:
                    if isinstance(field_value, list):
                        field_text = ' '.join(str(item) for item in field_value if item is not None)
                    else:
                        field_text = str(field_value)
                    if field_text:  # Additional check to ensure field_text is not empty
                        field_tokens = len(encoding.encode(field_text))
                        total_tokens += field_tokens
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"Error counting retrieved memory tokens: {e}")
        # Fallback: rough approximation
        total_chars = 0
        for memory in memory_items:
            content = memory.get('content')
            if content and content is not None:
                total_chars += len(str(content))
            # Add some chars for metadata
            total_chars += 100  # Rough estimate for metadata
        return total_chars // 4  # 1 token ≈ 4 characters

def count_neo_nodes_tokens(neo_nodes: List[Dict[str, Any]]) -> int:
    """
    Calculate total token count for Neo4j nodes
    """
    try:
        total_tokens = 0
        encoding = tiktoken.get_encoding("cl100k_base")
        
        for node in neo_nodes:
            # Count tokens in node properties
            node_properties = node.get('properties')
            if node_properties:
                for key, value in node_properties.items():
                    if value and value is not None:
                        if isinstance(value, list):
                            value_text = ' '.join(str(item) for item in value if item is not None)
                        else:
                            value_text = str(value)
                        if value_text:  # Additional check to ensure value_text is not empty
                            value_tokens = len(encoding.encode(value_text))
                            total_tokens += value_tokens
            
            # Count tokens in labels
            node_labels = node.get('labels')
            if node_labels and node_labels is not None:
                labels_text = ' '.join(str(label) for label in node_labels if label is not None)
                if labels_text:  # Additional check to ensure labels_text is not empty
                    labels_tokens = len(encoding.encode(labels_text))
                    total_tokens += labels_tokens
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"Error counting Neo4j nodes tokens: {e}")
        # Fallback: rough approximation
        total_chars = 0
        for node in neo_nodes:
            node_properties = node.get('properties')
            if node_properties and node_properties is not None:
                total_chars += len(str(node_properties))
            node_labels = node.get('labels')
            if node_labels and node_labels is not None:
                total_chars += len(' '.join(str(label) for label in node_labels if label is not None))
        return total_chars // 4  # 1 token ≈ 4 characters 