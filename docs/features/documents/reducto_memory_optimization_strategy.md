# Reducto Memory Optimization Strategy

## Overview

This document outlines the best approach for transforming Reducto AI responses into optimized memory objects that can be effectively stored and queried in Papr Memory (Qdrant/MongoDB/Neo4j).

## Current Reducto Response Analysis

From our testing, Reducto returns rich structured data:

```json
{
  "job_id": "c28d3532-c4c3-46a8-831a-7c0a20368bdb",
  "result": {
    "type": "full",
    "chunks": [
      {
        "blocks": [
          {
            "content": "Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies",
            "type": "title",
            "confidence": 0.95,
            "bounding_box": {...}
          },
          {
            "content": "Abstract\nThis paper presents...",
            "type": "text",
            "confidence": 0.9
          },
          {
            "content": "Table 1: Performance Metrics\n| Agent | Accuracy | Speed |\n|-------|----------|-------|",
            "type": "table",
            "confidence": 0.85
          }
        ]
      }
    ]
  },
  "usage": {
    "num_pages": 30,
    "credits": 15.0
  }
}
```

## Recommended Memory Transformation Strategy

### 1. **Enhanced Reducto Integration**

We've created `ReductoMemoryTransformer` that:

- **Groups blocks by type** (tables, titles, code, formulas, text)
- **Creates specialized memories** for each content type
- **Extracts metadata** (concepts, entities, structure)
- **Generates queryable content** with enhanced context

### 2. **Memory Types Created**

#### **Table Memories**
```json
{
  "content": "Table Data:\n| Agent | Accuracy | Speed |\n|-------|----------|-------|\n\nTable Structure: {\"row_count\": 3, \"column_count\": 3}",
  "metadata": {
    "content_type": "table",
    "table_structure": {...},
    "row_count": 3,
    "column_count": 3
  }
}
```

#### **Title/Heading Memories**
```json
{
  "content": "Document Section: Multi-Agent Design\n\nKey Concepts: Multi, Agent, Design, Optimizing",
  "metadata": {
    "content_type": "title",
    "concepts": ["Multi", "Agent", "Design"],
    "section_level": 1
  }
}
```

#### **Code Memories**
```json
{
  "content": "Code Block (python):\n```python\ndef optimize_agent():\n    return best_agent\n```",
  "metadata": {
    "content_type": "code",
    "programming_language": "python",
    "code_length": 45
  }
}
```

#### **Formula Memories**
```json
{
  "content": "Mathematical Formula:\nE = mc²\n\nMathematical Concepts: equation, physics",
  "metadata": {
    "content_type": "formula",
    "math_concepts": ["equation", "physics"],
    "formula_complexity": "low"
  }
}
```

### 3. **Reducto Pipeline Options**

Reducto offers three main pipelines:

#### **Parse Pipeline** (Current)
- **Use Case**: General document parsing
- **Output**: Structured blocks with content types
- **Best For**: Mixed content documents

#### **Extract Pipeline** (Recommended for specific use cases)
- **Use Case**: Schema-based extraction
- **Output**: Structured data matching your schema
- **Best For**: When you know exactly what data you want

#### **Split Pipeline** (Similar to our hierarchical chunker)
- **Use Case**: Document chunking
- **Output**: Semantic chunks
- **Best For**: When you want to preserve semantic boundaries

### 4. **Optimal Memory Storage Strategy**

#### **For Qdrant (Vector Database)**
```python
# Store with rich metadata for filtering
memory_vector = {
    "content": enhanced_content,
    "metadata": {
        "content_type": "table",
        "chunk_index": 0,
        "concepts": ["performance", "metrics"],
        "entities": ["Agent", "Accuracy"],
        "source": "reducto_table_extraction"
    }
}
```

#### **For MongoDB (Document Database)**
```python
# Store with full document structure
memory_document = {
    "content": enhanced_content,
    "metadata": metadata,
    "relationships": [
        {"type": "part_of", "target": "document_summary"},
        {"type": "contains", "target": "performance_data"}
    ],
    "indexed_fields": {
        "content_type": "table",
        "concepts": ["performance", "metrics"],
        "entities": ["Agent", "Accuracy"]
    }
}
```

#### **For Neo4j (Graph Database)**
```python
# Create nodes and relationships
CREATE (m:Memory {
    content: enhanced_content,
    content_type: "table",
    concepts: ["performance", "metrics"]
})
CREATE (d:Document {title: "Multi-Agent Design"})
CREATE (m)-[:PART_OF]->(d)
CREATE (m)-[:CONTAINS]->(c:Concept {name: "performance"})
```

## Best Practices for Queryable Memories

### 1. **Content Enhancement**
- **Add context**: Include surrounding information
- **Extract entities**: Identify key people, places, concepts
- **Generate concepts**: Create searchable topic tags
- **Preserve structure**: Maintain table/formula formatting

### 2. **Metadata Optimization**
- **Content type classification**: table, title, code, formula, text
- **Confidence scores**: From Reducto processing
- **Positional information**: Page numbers, chunk indices
- **Relationship mapping**: Links between related content

### 3. **Query Pattern Support**
```python
# Example queries that should work well:
queries = [
    "What are the performance metrics in the document?",
    "Show me all code examples",
    "What mathematical formulas are present?",
    "Find tables with more than 5 rows",
    "What are the main concepts in section 2?"
]
```

## Implementation Recommendations

### 1. **Use Enhanced Reducto Transformer**
- Leverages the new `ReductoMemoryTransformer`
- Creates specialized memories for different content types
- Extracts rich metadata for better searchability

### 2. **Combine with LLM Enhancement**
- Use Gemini 2.5 Flash for content enhancement
- Generate additional context and insights
- Create question-answer pairs for better retrieval

### 3. **Implement Hierarchical Chunking**
- Use Reducto's split pipeline for semantic chunking
- Combine with our hierarchical chunker for complex documents
- Preserve document structure and relationships

### 4. **Multi-Modal Support**
- Handle tables, images, formulas, and code separately
- Create specialized memories for each content type
- Enable cross-modal queries and relationships

## Testing the Enhanced Approach

Run the updated test to see the enhanced memory creation:

```bash
poetry run pytest tests/test_document_processing_v2.py -k "test_reducto_provider_simple_upload_and_parse" -v -s
```

This will show:
- Enhanced content extraction from Reducto blocks
- Specialized memory creation for different content types
- Rich metadata for better searchability
- Document summary with content type breakdown

## Next Steps

1. **Test the enhanced transformer** with real documents
2. **Optimize memory storage** based on your database choice
3. **Implement query patterns** that leverage the rich metadata
4. **Consider Reducto's extract pipeline** for specific use cases
5. **Integrate with LLM enhancement** for even better context

The enhanced approach should provide much better memory objects that can answer complex questions with proper context from your document chunks.
