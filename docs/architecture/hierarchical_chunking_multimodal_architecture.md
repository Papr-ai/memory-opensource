# PAPR Hierarchical Chunking with Multi-Modal Support Architecture

## Overview

This document outlines the architecture for implementing hierarchical, semantic-aware chunking with multi-modal content support in the PAPR memory system. The design addresses the limitations of basic recursive character splitting by preserving document structure, context, and enabling sophisticated handling of tables, charts, and other structured content.

## Current State Analysis

### Existing Architecture
- **Document Processing**: File reference-based processing via Temporal workflows
- **Storage**: Parse Server for files, Neo4j for relationships, vector DBs for embeddings
- **Memory Creation**: `/v1/memory/batch` endpoint for bulk memory ingestion
- **Providers**: TensorLake, Reducto, Gemini Vision for document parsing

### Current Limitations
1. **Basic Chunking**: Simple character-based splitting loses semantic context
2. **Limited Structure Preservation**: No hierarchical relationship tracking
3. **Poor Table Handling**: Tables processed as flat text, losing structure
4. **No Multi-Modal Memory**: Images, charts, tables not optimally stored
5. **Missing Context**: No cross-chunk relationship preservation

## Hierarchical Chunking Architecture

### Core Principles

1. **Semantic Preservation**: Maintain document hierarchy and meaning
2. **Context Awareness**: Preserve relationships between content sections
3. **Multi-Modal Support**: Handle text, tables, images, charts as distinct content types
4. **Flexible Storage**: Adaptive storage strategy based on content type and use case

### Document Structure Hierarchy

```
Document
├── Metadata (title, author, creation date, etc.)
├── Sections
│   ├── Section Header
│   ├── Paragraphs
│   ├── Tables
│   ├── Images/Charts
│   └── Subsections (recursive)
├── Cross-References
└── Appendices
```

## Multi-Modal Content Strategy

### Content Type Classification

#### 1. Textual Content
- **Paragraphs**: Semantic chunking by topic boundaries
- **Headers**: Preserved as structural markers
- **Lists**: Maintained as structured units
- **Footnotes**: Linked to parent content

#### 2. Tabular Content
```json
{
  "type": "table",
  "structure": {
    "headers": ["Column1", "Column2", "Column3"],
    "rows": [...],
    "metadata": {
      "title": "Financial Summary Q1 2024",
      "source_page": 15,
      "table_id": "table_001"
    }
  },
  "extracted_insights": [...],
  "relationships": ["links to paragraph above", "referenced in conclusion"]
}
```

#### 3. Visual Content
- **Charts**: OCR + chart understanding + data extraction
- **Images**: Image embedding + OCR text extraction
- **Diagrams**: Structural analysis + relationship mapping

### Storage Strategy Analysis

#### Option 1: Embedding-Only Approach
**Pros:**
- Unified search across all content types
- Semantic similarity matching
- Simple implementation

**Cons:**
- Loss of structured data queryability
- Poor performance for precise data retrieval
- Limited analytical capabilities

#### Option 2: Graph Schema Approach
**Pros:**
- Preserves relationships between content
- Enables complex queries across content types
- Maintains document structure hierarchy

**Cons:**
- Complex schema management
- Performance overhead for large datasets
- Limited embedding search capabilities

#### Option 3: Hybrid Multi-Store Approach (RECOMMENDED)
```
Content Type → Storage Strategy

Text Chunks → Vector DB (embeddings) + Neo4j (relationships)
Tables → MongoDB collections + Vector DB (table descriptions)
Time Series Data → MongoDB time-series + Graph relationships
Images → Parse Server files + Vector DB (image embeddings)
Metadata → Neo4j (document structure graph)
```

## Implementation Architecture

### New Temporal Activities

#### 1. Hierarchical Chunking Activity
```python
@activity.defn
async def hierarchical_chunk_document(
    document_content: Dict[str, Any],
    chunking_strategy: ChunkingStrategy,
    content_types: List[str]
) -> Dict[str, Any]:
    """
    Process document using hierarchical chunking strategy
    """
```

#### 2. Multi-Modal Content Extraction Activity
```python
@activity.defn
async def extract_multimodal_content(
    page_content: Dict[str, Any],
    extraction_config: ExtractionConfig
) -> MultiModalContent:
    """
    Extract and classify different content types from a page
    """
```

#### 3. Structured Data Processing Activity
```python
@activity.defn
async def process_structured_data(
    structured_content: List[StructuredElement],
    storage_strategy: StorageStrategy,
    organization_id: str,
    namespace_id: str
) -> ProcessingResult:
    """
    Process tables, charts, and other structured content
    """
```

#### 4. LLM Memory Structure Generation Activity
```python
@activity.defn
async def generate_memory_structures(
    content_analysis: ContentAnalysis,
    domain_context: DomainContext,
    user_preferences: UserPreferences
) -> List[MemoryStructure]:
    """
    Use LLM to generate optimized memory structures from content analysis
    """
```

### Enhanced Document Processing Workflow

```python
@workflow.defn
class HierarchicalDocumentProcessingWorkflow:
    """Enhanced workflow with hierarchical chunking support"""

    async def run(
        self,
        file_reference: DocumentFileReference,
        processing_config: HierarchicalProcessingConfig,
        # ... other parameters
    ) -> EnhancedProcessingResult:

        # Step 1: Basic document parsing (existing)
        document_analysis = await workflow.execute_activity(
            "analyze_document_structure",
            args=[file_reference, processing_config],
            start_to_close_timeout=timedelta(minutes=10)
        )

        # Step 2: Hierarchical chunking
        hierarchical_chunks = await workflow.execute_activity(
            "hierarchical_chunk_document",
            args=[document_analysis, processing_config.chunking_strategy],
            start_to_close_timeout=timedelta(minutes=15)
        )

        # Step 3: Multi-modal content extraction
        multimodal_content = []
        for chunk in hierarchical_chunks:
            content = await workflow.execute_activity(
                "extract_multimodal_content",
                args=[chunk, processing_config.extraction_config],
                start_to_close_timeout=timedelta(minutes=5)
            )
            multimodal_content.append(content)

        # Step 4: Structured data processing
        structured_results = await workflow.execute_activity(
            "process_structured_data",
            args=[multimodal_content, processing_config.storage_strategy],
            start_to_close_timeout=timedelta(minutes=20)
        )

        # Step 5: LLM-based memory structure generation
        memory_structures = await workflow.execute_activity(
            "generate_memory_structures",
            args=[structured_results, processing_config.domain_context],
            start_to_close_timeout=timedelta(minutes=10)
        )

        # Step 6: Batch memory creation
        memory_creation_result = await workflow.execute_activity(
            "create_hierarchical_memory_batch",
            args=[memory_structures, processing_config],
            start_to_close_timeout=timedelta(minutes=15)
        )

        return EnhancedProcessingResult(
            hierarchical_chunks=hierarchical_chunks,
            multimodal_content=multimodal_content,
            structured_data=structured_results,
            memory_structures=memory_structures,
            creation_result=memory_creation_result
        )
```

## Content Type Specific Strategies

### Table Processing Strategy

#### Financial Use Case Example
```python
class FinancialTableProcessor:
    """Specialized processor for financial tables"""

    async def process_financial_table(self, table_data: TableData) -> FinancialTableMemory:
        # 1. Extract structured data
        structured_data = self.extract_financial_structure(table_data)

        # 2. Generate time-series data if applicable
        if self.is_time_series(structured_data):
            time_series_records = self.create_time_series_records(structured_data)
            await self.store_in_mongodb_timeseries(time_series_records)

        # 3. Create graph relationships
        graph_relationships = self.create_financial_relationships(structured_data)
        await self.store_in_neo4j(graph_relationships)

        # 4. Generate searchable embeddings
        table_description = await self.generate_table_description(structured_data)
        embedding = await self.create_embedding(table_description)

        # 5. Create memory structure
        return FinancialTableMemory(
            content=table_description,
            structured_data=structured_data,
            relationships=graph_relationships,
            embedding=embedding,
            metadata={
                "table_type": "financial_summary",
                "period": self.extract_period(structured_data),
                "entities": self.extract_entities(structured_data)
            }
        )
```

#### Healthcare Use Case Example
```python
class HealthcareTableProcessor:
    """Specialized processor for healthcare/medical tables"""

    async def process_medical_table(self, table_data: TableData) -> MedicalTableMemory:
        # 1. Identify medical entities (drugs, dosages, lab values)
        medical_entities = await self.extract_medical_entities(table_data)

        # 2. Create structured medical records
        if self.is_patient_data(table_data):
            patient_records = self.create_patient_records(table_data, medical_entities)
            await self.store_in_secure_collection(patient_records)

        # 3. Generate clinical insights
        clinical_insights = await self.generate_clinical_insights(table_data)

        # 4. Create temporal relationships for longitudinal data
        if self.has_temporal_data(table_data):
            temporal_graph = self.create_temporal_graph(table_data)
            await self.store_temporal_relationships(temporal_graph)

        return MedicalTableMemory(
            content=self.generate_medical_summary(table_data),
            entities=medical_entities,
            insights=clinical_insights,
            metadata={
                "medical_domain": self.classify_medical_domain(table_data),
                "data_sensitivity": "high",
                "compliance_tags": ["HIPAA", "medical_records"]
            }
        )
```

### LLM-Based Memory Structure Generation

#### Strategy for Processing Reducto Output
```python
class LLMMemoryStructureGenerator:
    """Use LLM to generate optimized memory structures from document analysis"""

    async def generate_from_reducto_output(
        self,
        reducto_output: ReductoOutput,
        domain_context: DomainContext
    ) -> List[MemoryStructure]:

        # 1. Analyze content patterns
        content_analysis = await self.analyze_content_patterns(reducto_output)

        # 2. Generate domain-specific prompts
        generation_prompts = self.create_domain_prompts(content_analysis, domain_context)

        # 3. Use LLM to generate memory structures
        memory_structures = []
        for content_section in content_analysis.sections:
            structure = await self.llm_generate_memory_structure(
                content=content_section,
                prompt=generation_prompts.get(content_section.type),
                context=domain_context
            )
            memory_structures.append(structure)

        # 4. Optimize for batch memory creation
        optimized_batch = self.optimize_for_batch_creation(memory_structures)

        return optimized_batch

    async def llm_generate_memory_structure(
        self,
        content: ContentSection,
        prompt: str,
        context: DomainContext
    ) -> MemoryStructure:
        """
        LLM prompt example for financial domain:

        "Given this financial table data: {content}

        Generate a memory structure that:
        1. Preserves numerical accuracy
        2. Creates queryable metadata for financial analysis
        3. Establishes relationships to other financial entities
        4. Includes time-period context
        5. Enables natural language querying about financial performance

        Return as structured JSON for memory batch creation."
        """

        llm_response = await self.call_llm(prompt, content, context)
        return self.parse_llm_response_to_memory_structure(llm_response)
```

## Storage Architecture Recommendations

### Hybrid Storage Strategy

```python
class HybridStorageManager:
    """Manages multi-store strategy for different content types"""

    def __init__(self):
        self.vector_db = QdrantClient()  # For embeddings and semantic search
        self.graph_db = Neo4jConnection()  # For relationships and structure
        self.document_db = MongoDBConnection()  # For structured data and time series
        self.file_storage = ParseServerStorage()  # For binary files and images

    async def store_hierarchical_content(self, content: HierarchicalContent) -> StorageResult:
        storage_tasks = []

        # Text content → Vector DB + Graph relationships
        if content.text_chunks:
            storage_tasks.append(self.store_text_chunks(content.text_chunks))

        # Tables → MongoDB + Vector descriptions + Graph relationships
        if content.tables:
            storage_tasks.append(self.store_tables_hybrid(content.tables))

        # Time series data → MongoDB time-series collections
        if content.time_series:
            storage_tasks.append(self.store_time_series(content.time_series))

        # Images → Parse Server + Vector embeddings
        if content.images:
            storage_tasks.append(self.store_images_hybrid(content.images))

        # Document structure → Graph DB
        storage_tasks.append(self.store_document_structure(content.structure))

        return await asyncio.gather(*storage_tasks)

    async def store_tables_hybrid(self, tables: List[TableContent]) -> StorageResult:
        """Hybrid storage strategy for tables"""
        results = []

        for table in tables:
            # 1. Store structured data in MongoDB
            mongo_result = await self.document_db.store_table_data(
                collection=f"tables_{table.domain}",
                data=table.structured_data,
                metadata=table.metadata
            )

            # 2. Create searchable description and store embedding
            description = await self.generate_table_description(table)
            embedding_result = await self.vector_db.store_embedding(
                content=description,
                metadata={
                    "content_type": "table",
                    "mongo_ref": mongo_result.id,
                    "table_type": table.table_type
                }
            )

            # 3. Store relationships in graph
            graph_result = await self.graph_db.create_table_relationships(
                table_id=mongo_result.id,
                relationships=table.relationships,
                parent_document=table.parent_document_id
            )

            results.append(HybridStorageResult(
                mongo_id=mongo_result.id,
                vector_id=embedding_result.id,
                graph_nodes=graph_result.nodes
            ))

        return results
```

## Integration with Existing Memory Batch Endpoint

### Enhanced Memory Batch Creation

```python
class HierarchicalMemoryBatchCreator:
    """Creates memory batches with hierarchical and multi-modal support"""

    async def create_hierarchical_batch(
        self,
        memory_structures: List[MemoryStructure],
        organization_id: str,
        namespace_id: str,
        user_id: str
    ) -> BatchCreationResult:

        # Group memories by type and strategy
        grouped_memories = self.group_memories_by_strategy(memory_structures)

        batch_requests = []

        # Text-based memories (existing flow)
        if grouped_memories.text_memories:
            text_batch = self.create_text_memory_batch(grouped_memories.text_memories)
            batch_requests.append(text_batch)

        # Table-based memories (new hybrid flow)
        if grouped_memories.table_memories:
            table_batch = await self.create_table_memory_batch(grouped_memories.table_memories)
            batch_requests.append(table_batch)

        # Time-series memories (new time-aware flow)
        if grouped_memories.timeseries_memories:
            ts_batch = await self.create_timeseries_memory_batch(grouped_memories.timeseries_memories)
            batch_requests.append(ts_batch)

        # Execute all batches via existing /v1/memory/batch endpoint
        batch_results = []
        for batch_request in batch_requests:
            result = await self.call_memory_batch_endpoint(
                batch_request=batch_request,
                organization_id=organization_id,
                namespace_id=namespace_id,
                user_id=user_id
            )
            batch_results.append(result)

        return BatchCreationResult(
            total_memories_created=sum(r.count for r in batch_results),
            batch_results=batch_results,
            hierarchical_relationships=self.create_cross_batch_relationships(batch_results)
        )
```

## Performance and Scalability Considerations

### Chunking Performance
- **Parallel Processing**: Process chunks in parallel within Temporal activities
- **Adaptive Chunk Sizes**: Based on content complexity and type
- **Caching**: Cache intermediate results for repeated processing patterns

### Storage Performance
- **Write Distribution**: Distribute writes across multiple storage systems
- **Read Optimization**: Query routing based on content type and query pattern
- **Indexing Strategy**: Optimized indexes for each storage system

### Memory Creation Optimization
- **Batch Size Optimization**: Dynamic batch sizes based on content complexity
- **Priority Queuing**: Prioritize critical content types (e.g., tables in financial docs)
- **Error Recovery**: Robust error handling with partial success recovery

## Monitoring and Observability

### Key Metrics
- **Chunking Quality**: Semantic coherence scores
- **Processing Latency**: Time per content type and document size
- **Storage Distribution**: Data distribution across storage systems
- **Query Performance**: Response times for different query patterns

### Alerting
- **Processing Failures**: Failed chunking or storage operations
- **Quality Degradation**: Declining semantic coherence scores
- **Performance Issues**: Increased latency or timeouts

## Migration Strategy

### Phase 1: Foundation (2-3 weeks)
1. Implement basic hierarchical chunking activities
2. Create multi-modal content extraction framework
3. Design storage strategy interfaces

### Phase 2: Core Implementation (3-4 weeks)
1. Implement table processing strategies
2. Build LLM-based memory structure generation
3. Create hybrid storage manager

### Phase 3: Integration (2-3 weeks)
1. Integrate with existing document processing workflow
2. Enhance memory batch creation
3. Performance optimization

### Phase 4: Domain Specialization (3-4 weeks)
1. Implement financial domain processors
2. Implement healthcare domain processors
3. Add domain-specific storage optimizations

## Next Steps

1. **Create detailed technical specifications** for each new Temporal activity
2. **Implement proof-of-concept** for one content type (tables)
3. **Design domain-specific processors** for target use cases
4. **Create storage strategy prototypes** and performance benchmarks
5. **Develop LLM prompt engineering** for memory structure generation

This architecture provides a comprehensive foundation for implementing hierarchical chunking with multi-modal support while maintaining compatibility with existing PAPR systems and enabling sophisticated domain-specific processing capabilities.