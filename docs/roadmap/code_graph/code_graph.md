# Code Knowledge Graph Architecture

## Overview

This document outlines a revolutionary approach to code intelligence that transforms how we interact with codebases. Instead of primitive text-based search (grep), we propose a semantic knowledge graph that understands code structure, relationships, and architectural patterns.

## Current Problem: Context-Poor Code Search

When searching for `BatchMemoryRequest` with grep, we get:
- Raw text matches without context
- No understanding of type relationships
- Missing architectural context
- No usage patterns or best practices
- Fragmented results requiring manual investigation

**Example Problem:**
```bash
# Current approach
grep -r "BatchMemoryRequest" --include="*.py" .
# Result: 15 scattered matches, 45 minutes of detective work
```

## Vision: Code-as-Knowledge-Graph Architecture

Transform code into a queryable semantic graph that understands:
- **Type System**: Pydantic models, inheritance, composition
- **API Architecture**: Endpoints, request/response patterns, validation
- **Business Logic**: Services, workflows, dependencies
- **Data Flow**: How data moves through the system
- **Best Practices**: Patterns, anti-patterns, examples

## Custom Ontology Design

### Node Types

```yaml
# Core Type System
PydanticModel:
  properties: [name, module, fields, validators, inheritance_chain]
  relationships: [INHERITS_FROM, COMPOSES, USED_BY, VALIDATES]

TypeAnnotation:
  properties: [type_name, generic_params, optional, default_value]
  relationships: [ANNOTATES, REFERENCES_TYPE]

# API Layer
APIEndpoint:
  properties: [path, method, handler_function, auth_required]
  relationships: [ACCEPTS, RETURNS, VALIDATES_WITH, ROUTES_TO]

RequestModel:
  properties: [fields, validation_rules, serialization_format]
  relationships: [VALIDATED_BY, PROCESSED_BY, STORED_AS]

ResponseModel:
  properties: [success_fields, error_fields, status_codes]
  relationships: [RETURNED_BY, SERIALIZED_FROM]

# Business Logic
Service:
  properties: [name, responsibility, dependencies]
  relationships: [CALLS, DEPENDS_ON, IMPLEMENTS, ORCHESTRATES]

Workflow:
  properties: [steps, temporal_config, retry_policy]
  relationships: [EXECUTES, TRIGGERS, HANDLES_FAILURE]

# Data Layer
DatabaseModel:
  properties: [table, fields, indexes, constraints]
  relationships: [PERSISTS, QUERIES, JOINS_WITH]

# Architecture
Module:
  properties: [name, purpose, exported_symbols]
  relationships: [IMPORTS, EXPORTS, CONTAINS]
```

### Relationship Types

```yaml
relationships:
  # Type Relationships
  INHERITS_FROM: "Type inheritance hierarchy"
  COMPOSES: "Type composition (has-a relationship)"
  VALIDATES: "Validation rules and logic"

  # Data Flow
  ACCEPTS: "API endpoint accepts this type"
  RETURNS: "API endpoint returns this type"
  PROCESSES: "Service processes this data"
  TRANSFORMS: "Data transformation pipeline"

  # Architecture
  DEPENDS_ON: "Service/module dependencies"
  IMPLEMENTS: "Interface implementation"
  ORCHESTRATES: "Workflow orchestration"

  # Usage Patterns
  BEST_PRACTICE: "Recommended usage pattern"
  ANTI_PATTERN: "Usage to avoid"
  EXAMPLE_USAGE: "Code example or test case"
```

## Semantic Search Examples

### Current vs Semantic Approach

**Current Grep Approach:**
```bash
# Question: "How do I use BatchMemoryRequest correctly?"
grep -r "BatchMemoryRequest" --include="*.py" .
# Result: Raw text matches, no context, fragmented understanding
```

**Semantic Knowledge Graph Approach:**
```cypher
# Same question with semantic understanding
MATCH (model:PydanticModel {name: "BatchMemoryRequest"})
OPTIONAL MATCH (model)-[r:USED_BY]->(endpoint:APIEndpoint)
OPTIONAL MATCH (model)-[v:VALIDATES]->(validation:ValidationRule)
OPTIONAL MATCH (model)-[e:EXAMPLE_USAGE]->(example:CodeExample)
OPTIONAL MATCH (model)-[bp:BEST_PRACTICE]->(practice:Pattern)
RETURN {
  definition: model,
  endpoints_that_use_it: collect(endpoint),
  validation_rules: collect(validation),
  usage_examples: collect(example),
  best_practices: collect(practice),
  related_types: [(model)-[:COMPOSES|INHERITS_FROM]->(related) | related]
}
```

## Custom Schema Routes API

```python
@router.post("/code/ontology/create")
async def create_code_ontology(ontology: CodeOntologyRequest):
    """
    Create semantic schema for codebase analysis

    Request:
    {
      "repository_path": "/Users/me/memory-server",
      "languages": ["python"],
      "analysis_depth": "deep",  # shallow|medium|deep
      "include_patterns": ["*.py", "*.yaml"],
      "custom_node_types": [
        {
          "name": "PydanticModel",
          "detection_rules": [
            "class.*BaseModel",
            "@dataclass",
            "Field\\("
          ],
          "properties": ["name", "fields", "validators"],
          "relationships": ["INHERITS_FROM", "USED_BY"]
        }
      ]
    }
    """
    pass

@router.post("/code/analyze")
async def analyze_codebase(request: CodeAnalysisRequest):
    """
    Parse codebase into knowledge graph using Tree-sitter + custom rules

    Process:
    1. Parse AST for each file
    2. Extract entities (classes, functions, types)
    3. Build relationships (calls, imports, inheritance)
    4. Apply semantic annotations
    5. Generate embeddings for semantic search
    """
    pass

@router.post("/code/search/semantic")
async def semantic_code_search(query: SemanticSearchQuery):
    """
    Natural language code search with contextual understanding

    Examples:
    - "How do I properly validate BatchMemoryRequest?"
    - "What are all the Temporal workflow patterns?"
    - "Show me anti-patterns in our API design"
    - "Find all Pydantic models that handle authentication"
    """
    pass

@router.get("/code/context/{symbol}")
async def get_code_context(symbol: str):
    """
    Get complete context for a code symbol

    Returns:
    - Type definition and hierarchy
    - All usage locations with semantic context
    - Validation rules and constraints
    - Best practices and examples
    - Related symbols and patterns
    - Test coverage and examples
    """
    pass
```

## Example: BatchMemoryRequest Complete Context

What the system would return for `BatchMemoryRequest`:

```json
{
  "symbol": "BatchMemoryRequest",
  "definition": {
    "type": "PydanticModel",
    "file": "models/memory_models.py:250",
    "inheritance": ["BaseModel"],
    "fields": [
      {
        "name": "memories",
        "type": "List[AddMemoryRequest]",
        "required": true,
        "validation": "min_length=1, max_length=50"
      },
      {
        "name": "webhook_url",
        "type": "Optional[str]",
        "description": "Completion notification endpoint"
      }
    ]
  },
  "usage_patterns": {
    "api_endpoints": [
      {
        "path": "/v1/memories/batch",
        "method": "POST",
        "handler": "memory_routes_v1.py:719",
        "authentication": "required",
        "rate_limits": "30/minute"
      }
    ],
    "services": [
      {
        "name": "batch_processor.process_batch_with_temporal",
        "purpose": "Routes large batches to Temporal workflows",
        "conditions": "batch_size > temporal_threshold"
      }
    ],
    "validations": [
      {
        "rule": "validate_batch_sizes",
        "checks": ["content_length", "total_batch_size", "storage_limits"],
        "error_handling": "ValidationError with specific field errors"
      }
    ]
  },
  "best_practices": [
    "Always use proper Pydantic types, never Dict[str, Any]",
    "Validate batch size before processing",
    "Use webhook_url for async processing notification",
    "Handle validation errors gracefully with specific messages"
  ],
  "related_types": {
    "composes": ["AddMemoryRequest", "OptimizedAuthResponse"],
    "used_by": ["BatchMemoryResponse", "TemporalWorkflowData"],
    "validates_with": ["validate_batch_sizes", "field_validator"]
  },
  "examples": [
    {
      "context": "test_add_memory_fastapi.py:150",
      "pattern": "correct_usage",
      "code": "BatchMemoryRequest(memories=[...], webhook_url=None)"
    }
  ],
  "common_errors": [
    {
      "error": "'dict' object has no attribute 'memories'",
      "cause": "Passing dict instead of Pydantic model",
      "solution": "Use proper type conversion in batch_processor.py:49-58"
    }
  ]
}
```

## Advanced Query Examples

### Find Validation Anti-Patterns
```cypher
MATCH (model:PydanticModel)-[:VALIDATES_WITH]->(validation)
WHERE validation.pattern = "anti_pattern"
RETURN model, validation
```

### Show Complete Data Flow for Batch Processing
```cypher
MATCH path = (endpoint:APIEndpoint)-[:ACCEPTS]->(request)
-[:PROCESSED_BY]->(service)-[:CALLS]->(workflow)
WHERE endpoint.path CONTAINS "batch"
RETURN path
```

### Find All Temporal-Related Code with Proper Typing
```cypher
MATCH (temporal:Service)-[:USES_TYPE]->(type:PydanticModel)
WHERE temporal.name CONTAINS "temporal"
RETURN temporal, type, type.best_practices
```

## Current vs Semantic Approach Comparison

### Scenario: "Fix the Pydantic type error in Temporal workflow"

**Current Grep Approach:**
```bash
# Step 1: Find the error
grep -r "dict.*has no attribute" .
# Returns: scattered log entries, no context

# Step 2: Find related code
grep -r "BatchMemoryRequest" --include="*.py" .
# Returns: 15 files with random matches

# Step 3: Understand relationships
grep -r "temporal.*workflow" --include="*.py" .
# Returns: fragmented results

# Result: 45 minutes of detective work, context switching, potential mistakes
```

**Semantic Knowledge Graph Approach:**
```cypher
// Single query gets complete context
MATCH (error:RuntimeError {message: "dict object has no attribute memories"})
-[:OCCURS_IN]->(workflow:Workflow)
-[:USES_TYPE]->(type:PydanticModel)
-[:VALIDATION_ERROR]->(solution:BestPractice)
RETURN {
  error_location: error.file_line,
  root_cause: error.cause,
  affected_type: type,
  correct_usage: solution.example,
  related_fixes: [(error)-[:SIMILAR_TO]->(similar) | similar]
}

// Result: Complete context in seconds, with solutions
```

### Scenario: "How do I properly implement a new batch API endpoint?"

**Current Approach:** Hunt through multiple files, piece together patterns, make assumptions

**Semantic Approach:**
```cypher
MATCH (pattern:ArchitecturalPattern {name: "batch_api_endpoint"})
-[:IMPLEMENTED_BY]->(examples:APIEndpoint)
-[:USES_TYPES]->(request_models:PydanticModel)
-[:VALIDATED_WITH]->(validations)
-[:FOLLOWS_PATTERN]->(best_practices)
RETURN {
  template: pattern.implementation_template,
  required_types: collect(request_models),
  validation_rules: collect(validations),
  best_practices: collect(best_practices),
  example_implementations: collect(examples)
}
```

## Implementation Strategy

### Phase 1: AST Parsing & Graph Construction
```python
import ast
import tree_sitter
from models.temporal_models import CodeEntity, CodeRelationship

class CodeGraphBuilder:
    def parse_codebase(self, repo_path: str):
        # Use Tree-sitter for robust parsing
        # Extract entities: classes, functions, types, imports
        # Build relationships: inheritance, composition, usage
        # Apply semantic annotations based on patterns
        pass

    def extract_pydantic_models(self, ast_node):
        # Detect Pydantic models
        # Extract field definitions and validators
        # Map inheritance relationships
        # Identify usage patterns
        pass
```

### Phase 2: Semantic Enhancement
```python
class SemanticEnhancer:
    def enhance_with_context(self, entities: List[CodeEntity]):
        # Add architectural context (service boundaries)
        # Identify design patterns and anti-patterns
        # Generate best practice recommendations
        # Create usage examples from tests
        pass
```

### Phase 3: Query Interface
```python
class SemanticCodeQuery:
    async def natural_language_search(self, query: str):
        # Convert NL to graph query
        # Execute semantic search
        # Return structured results with context
        pass
```

## Research Foundation

### Latest Publications (2024-2025)

**GNN-Coder (February 2025)**: Novel framework using Graph Neural Networks with AST to capture structural and semantic information for code retrieval.

**Deep Code Curator – code2graph**: Graph learning pipelines that embed code elements using various approaches including treating them as knowledge graphs.

**GitLab Knowledge Graph**: Public beta system that creates queryable maps of code repositories, providing deep insights into codebases for AI-driven features.

### Technical Approaches

**Code2Vec Methodology**: Decomposes code into AST paths, learns atomic representations, and aggregates them for semantic understanding.

**PathPair2Vec**: AST path pair-based code representation using attention mechanisms, showing 17.88% improvement in F1 score over existing methods.

**Tree-sitter Integration**: Robust parsing across multiple languages with concrete syntax tree preservation.

## Architecture Benefits

### Intelligence Amplification
- **Context-Aware Coding**: Complete usage context, validation rules, and best practices
- **Proactive Error Prevention**: System warns about common mistakes before they happen
- **Pattern Recognition**: Identifies architectural patterns and suggests consistent implementations

### ROI Analysis
- **Current**: 30-45 minutes per context-switching task
- **With Code Graph**: 30 seconds to 2 minutes per task
- **Productivity Gain**: ~15-20x faster development iteration
- **Quality Improvement**: Proactive pattern enforcement, fewer bugs

## Implementation Roadmap

### Phase 1: Proof of Concept (4 weeks)
1. AST parsing for Python files
2. Basic entity extraction (classes, functions)
3. Simple relationship mapping
4. Basic query interface

### Phase 2: Semantic Enhancement (6 weeks)
1. Pydantic model detection and analysis
2. API endpoint pattern recognition
3. Service dependency mapping
4. Best practice identification

### Phase 3: Production System (8 weeks)
1. Full codebase analysis
2. Natural language query interface
3. Integration with development tools
4. Performance optimization

### Phase 4: Advanced Features (12 weeks)
1. Real-time code change detection
2. Proactive pattern suggestions
3. Anti-pattern warnings
4. Automated documentation generation

## Conclusion

This code knowledge graph architecture represents a fundamental shift from text-based code search to semantic understanding. By leveraging the latest research in AST analysis, graph neural networks, and semantic code search, we can create an intelligent development environment that amplifies developer productivity by 15-20x while preventing common errors and enforcing architectural best practices.

The system transforms the codebase from a collection of text files into a queryable, context-rich knowledge system that understands relationships, patterns, and architectural principles. This enables developers to work at the level of intent rather than implementation details, dramatically improving both productivity and code quality.

---

*Created: January 2025*
*Last Updated: January 2025*
*Status: Design Phase*