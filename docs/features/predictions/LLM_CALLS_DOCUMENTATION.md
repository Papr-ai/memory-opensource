# LLM Calls Documentation

This document provides a comprehensive overview of all LLM (Large Language Model) calls in the codebase, including model configurations, token limits, and use cases.

## Table of Contents
- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [LLM Calls by Provider](#llm-calls-by-provider)
- [Detailed Call Documentation](#detailed-call-documentation)
- [Token Budget Management](#token-budget-management)
- [Fallback Strategy](#fallback-strategy)

---

## Overview

The codebase uses a multi-provider LLM strategy to optimize for:
- **Cost**: Using cheaper models for simple operations
- **Speed**: Leveraging Groq for fast inference
- **Reliability**: Multiple fallback providers
- **Quality**: Higher-quality models for complex reasoning

### Provider Distribution
- **OpenAI**: Primary for structured operations, content generation
- **Groq**: Fast inference for Cypher queries, message analysis
- **Google Gemini**: Free tier with high limits for document processing

---

## Environment Variables

### Primary Model Configuration

| Environment Variable | Default Value | Priority | Purpose |
|---------------------|---------------|----------|---------|
| `LLM_MODEL` | `gpt-4.1-nano` | **HIGH** | Primary model for most operations |
| `LLM_MODEL_MINI` | `gpt-5-mini` | **HIGH** | Higher-quality model for schema/search operations |
| `LLM_MODEL_NANO` | `gpt-5-nano` | MEDIUM | Document processing metadata generation |
| `GROQ_PATTERN_SELECTOR_MODEL` | `openai/gpt-oss-20b` | **HIGH** | Groq model for pattern selection and message analysis |
| `GROQ_LLM_MODEL` | `openai/gpt-oss-20b` | MEDIUM | Groq model for document processing |
| `GEMINI_MODEL_FAST` | `gemini-2.5-flash` | MEDIUM | Google Gemini model for document processing |

### API Keys

| Environment Variable | Required | Used For |
|---------------------|----------|----------|
| `OPENAI_API_KEY` | ✅ Yes | All OpenAI API calls |
| `GROQ_API_KEY` | ✅ Yes | Groq API calls (Cypher, message analysis) |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | ⚠️ Optional | Gemini fallback for document processing |

### Configuration Flags

| Environment Variable | Default | Purpose |
|---------------------|---------|---------|
| `LLM_LOCATION_CLOUD` | `true` | Use cloud models (true) or local Ollama (false) |
| `USE_DOTENV` | `true` | Load environment variables from .env file |
| `GROQ_NEO_CYPHER` | Not set | Enable Groq-specific Cypher generation path |

### Priority Notes
- **HIGH Priority**: These variables directly impact core functionality and are used frequently
- **MEDIUM Priority**: Used for specific operations or as fallback options
- Environment variables always override default values when set

---

## LLM Calls by Provider

### OpenAI Models

#### 1. **Primary Model** (`LLM_MODEL`)
- **Default**: `gpt-4.1-nano`
- **Env Variable**: `LLM_MODEL` (overrides default)
- **Location**: `api_handlers/chat_gpt_completion.py` (line 98)
- **Context Window**: 128K tokens
- **Max Output**: Dynamic (min 1024, max 4096)
- **Temperature**: 0.5-0.7
- **Use Cases**:
  - Content generation with memory retrieval
  - Review and feedback generation
  - Memory item generation
  - Function calling with `get_memory` tool

#### 2. **Mini Model** (`LLM_MODEL_MINI`)
- **Default**: `gpt-5-mini`
- **Env Variable**: `LLM_MODEL_MINI` (overrides default)
- **Location**: `api_handlers/chat_gpt_completion.py` (line 105)
- **Context Window**: 128K tokens
- **Max Output**: Dynamic (calculated based on input)
- **Temperature**: Not set (default)
- **Use Cases**:
  - Query classification with goals/use cases/steps
  - Tier prediction for routing
  - Schema generation operations
  - Complex reasoning tasks

#### 3. **Nano Model** (`LLM_MODEL_NANO`)
- **Default**: `gpt-5-nano`
- **Env Variable**: `LLM_MODEL_NANO` (overrides default)
- **Location**: `core/document_processing/llm_memory_generator.py` (line 422)
- **Context Window**: 128K tokens
- **Max Output**: 2048 tokens (metadata only)
- **Temperature**: 0.3
- **Use Cases**:
  - Document content metadata generation
  - Structured metadata extraction
  - Title, topics, and query patterns generation

### Groq Models

#### 4. **Pattern Selector Model** (`GROQ_PATTERN_SELECTOR_MODEL`)
- **Default**: `openai/gpt-oss-20b`
- **Env Variable**: `GROQ_PATTERN_SELECTOR_MODEL` (overrides default)
- **Locations**:
  - `api_handlers/chat_gpt_completion.py` (line 138)
  - `services/message_analysis.py` (line 20)
  - `services/message_batch_analysis.py` (line 100)
- **Context Window**: 128K tokens
- **Max Output**: Not explicitly set
- **Temperature**: 0.1
- **Use Cases**:
  - **Cypher Query Generation**: Pattern selection for Neo4j queries
  - **Message Analysis**: Individual message memory-worthiness analysis
  - **Batch Message Analysis**: Multiple messages processed together
  - Tool calling with structured outputs

#### 5. **Document Processing Model** (`GROQ_LLM_MODEL`)
- **Default**: `openai/gpt-oss-20b`
- **Env Variable**: `GROQ_LLM_MODEL` (overrides default)
- **Location**: `core/document_processing/llm_memory_generator.py` (line 642)
- **Context Window**: 128K tokens
- **Max Output**: 1024 tokens
- **Temperature**: 0.3
- **Use Cases**:
  - Document metadata generation (tertiary fallback)
  - JSON schema enforcement for structured outputs
  - Rate limit handling with exponential backoff

### Google Gemini Models

#### 6. **Fast Model** (`GEMINI_MODEL_FAST`)
- **Default**: `gemini-2.5-flash`
- **Env Variable**: `GEMINI_MODEL_FAST` (overrides default)
- **Locations**:
  - `api_handlers/chat_gpt_completion.py` (line 142)
  - `core/document_processing/llm_memory_generator.py` (line 484)
- **Context Window**: 1M tokens
- **Max Output**: 8192 tokens (schema generation), 1024 tokens (document processing)
- **Temperature**: 0.3
- **Use Cases**:
  - **UseCase Generation Fallback**: When OpenAI structured outputs fail
  - **Document Processing Primary**: Free tier with high rate limits
  - Native JSON schema support via `responseSchema`

### Local Models (Ollama)

#### 7. **Local Llama Model**
- **Default**: `meta-llama/llama-4-maverick-17b-128e-instruct`
- **Env Variable**: Model passed to constructor
- **Location**: `services/query_log_service.py` (line 511)
- **Context Window**: 128K tokens
- **Max Output**: Dynamic
- **Temperature**: Not set
- **Use Cases**:
  - Query classification when `model_location_cloud=false`
  - Local inference for privacy-sensitive operations

---

## Detailed Call Documentation

### 1. Content Generation with Memory (`generate_content_with_memories`)

**File**: `api_handlers/chat_gpt_completion.py:3420`

```python
Model: LLM_MODEL (default: gpt-4.1-nano)
Env Override: LLM_MODEL
Max Input: 128K tokens
Max Output: min(4096, 128000 - input_tokens - 7)
Temperature: 0.5
```

**Purpose**: Generate content using retrieved memories with function calling

**Token Budget Logic**:
```python
max_tokens_for_completion = min(4096, 128000 - token_count_prompt - 7)
if max_tokens_for_completion < 1024:
    raise ValueError("Available tokens for completion are too few")
```

**Features**:
- Tool calling with `get_memory` function
- Two-stage generation: memory retrieval → content generation
- JSON response format for structured output

---

### 2. Query Classification (`_classify_query_with_user_data_internal`)

**File**: `services/query_log_service.py:290`

```python
Model: LLM_MODEL_MINI (default: gpt-5-mini)
Env Override: LLM_MODEL_MINI
Max Input: 128K tokens
Max Output: Dynamic (calculated based on input budget)
Temperature: Not set (default)
```

**Purpose**: Classify queries with goals, use cases, steps, and tier prediction

**Token Budget Logic**:
```python
MODEL_CONTEXT_WINDOW = 128_000
MODEL_MAX_OUTPUT = 16_384
buffer_tokens = 2000

available_output_tokens = MODEL_CONTEXT_WINDOW - prompt_token_estimate - buffer_tokens
max_completion_tokens = min(MODEL_MAX_OUTPUT, max(300, available_output_tokens))
```

**Features**:
- Structured output with `QueryClassification` Pydantic model
- Includes reasoning tokens logging for o1/gpt-5 models
- Fallback to simpler JSON mode with 1/4 token limit on failure
- 20-second timeout for primary attempt, 10-second for fallback

**Fallback Strategy**:
```python
# Primary: Typed parse with response_format
response = await chat_gpt.async_client.beta.chat.completions.parse(
    model=chat_gpt.model_mini,
    response_format=QueryClassification,
    max_completion_tokens=max_completion_tokens
)

# Fallback: JSON object mode with reduced tokens
fallback_tokens = max(300, max_completion_tokens // 4)
raw = await chat_gpt._create_completion_with_fallback_async(
    model=chat_gpt.model_mini,
    response_format={"type": "json_object"},
    max_completion_tokens=fallback_tokens
)
```

---

### 3. Cypher Query Generation (`generate_cypher_from_user_query_async`)

**File**: `api_handlers/chat_gpt_completion.py:1400`

```python
Primary Model: GROQ_PATTERN_SELECTOR_MODEL (default: openai/gpt-oss-20b)
Fallback Model: LLM_MODEL (default: gpt-4.1-nano)
Env Override: GROQ_PATTERN_SELECTOR_MODEL, LLM_MODEL
Max Input: 128K tokens
Max Output: Not explicitly set
Temperature: 0.1 (deterministic for query generation)
```

**Purpose**: Generate Neo4j Cypher queries for graph traversal using pattern templates

**Features**:
- Tool calling with dynamic schema generation
- Pattern-based query construction
- Property enhancement with vector matching
- Multiple fallback paths

**Execution Flow**:
```
1. Try Groq with tool calling (openai/gpt-oss-20b)
   ↓ (on failure)
2. Try Groq with Instructor library
   ↓ (on failure)
3. Try OpenAI fallback (LLM_MODEL)
   ↓ (on failure)
4. Use simple base query
```

**Tool Call Format**:
```python
cypher_tool = {
    "type": "function",
    "function": {
        "name": "generate_cypher_query",
        "description": "Generate a Cypher query AST for Neo4j",
        "parameters": dynamic_schema  # Generated based on available patterns
    }
}
```

---

### 4. Message Analysis (`analyze_message_for_memory`)

**File**: `services/message_analysis.py:45`

```python
Model: GROQ_PATTERN_SELECTOR_MODEL (default: openai/gpt-oss-20b)
Env Override: GROQ_PATTERN_SELECTOR_MODEL
Max Input: 128K tokens
Max Output: Not explicitly set
Temperature: 0.1
```

**Purpose**: Analyze individual chat messages to determine memory-worthiness

**Features**:
- JSON mode for structured output
- Role-specific category validation
- Confidence scoring
- `AddMemoryRequest` generation for worthy messages

**Response Schema**:
```json
{
  "is_memory_worthy": boolean,
  "confidence_score": number (0.0-1.0),
  "reasoning": "string",
  "memory_request": {
    "content": "refined memory content",
    "role": "user|assistant",
    "category": "appropriate category",
    "metadata": {...}
  }
}
```

---

### 5. Batch Message Analysis (`analyze_message_batch_for_memory`)

**File**: `services/message_batch_analysis.py:111`

```python
Model: GROQ_PATTERN_SELECTOR_MODEL (default: openai/gpt-oss-20b)
Env Override: GROQ_PATTERN_SELECTOR_MODEL
Max Input: 128K tokens
Max Output: Not explicitly set
Temperature: 0.1
```

**Purpose**: Analyze multiple messages in a conversation for memory extraction

**Features**:
- Batch processing for efficiency
- Context-aware analysis across messages
- Per-message confidence scoring
- Automatic memory creation for worthy messages

---

### 6. UseCase Memory Generation (`generate_usecase_memory_item_async`)

**File**: `api_handlers/chat_gpt_completion.py:3780`

```python
Primary Model: self.model (LLM_MODEL)
Fallback Model: GEMINI_MODEL_FAST (default: gemini-2.5-flash)
Env Override: LLM_MODEL, GEMINI_MODEL_FAST
Max Input: 128K tokens (OpenAI), 1M tokens (Gemini)
Max Output: Not explicitly set (OpenAI), 8192 tokens (Gemini)
Temperature: Not set (OpenAI), 0.3 (Gemini)
```

**Purpose**: Generate goals and use cases from memory items with structured outputs

**Features**:
- OpenAI Structured Outputs (primary)
- Gemini fallback with JSON schema enforcement
- Token limit handling with content trimming
- Metrics tracking (input/output tokens, cost)

**Fallback Logic**:
```python
try:
    # Try OpenAI first
    completion = await self.async_client.beta.chat.completions.parse(
        model=self.model,
        messages=messages,
        response_format=UseCaseMemoryItem
    )
except Exception as e:
    # Fallback to Gemini
    result = await self._call_gemini_structured_async(messages, UseCaseMemoryItem)
```

---

### 7. Memory Graph Schema Generation (`generate_memory_graph_schema`)

**File**: `api_handlers/chat_gpt_completion.py:599`

```python
Model: self.model (LLM_MODEL, default: gpt-4.1-nano)
Env Override: LLM_MODEL
Max Input: 128K tokens
Max Output: Not explicitly set
Temperature: Not set
```

**Purpose**: Generate memory graph schemas with nodes and relationships

**Features**:
- Structured outputs with `MemoryGraphSchema` model
- Content trimming to fit token limits
- Existing schema consideration
- Metrics tracking

---

### 8. Document Memory Generation

**File**: `core/document_processing/llm_memory_generator.py:200`

This operation uses a three-tier provider strategy:

#### Primary Route: Google Gemini
```python
Model: GEMINI_MODEL_FAST (default: gemini-2.5-flash)
Env Override: GEMINI_MODEL_FAST
Max Input: 1M tokens
Max Output: 1024 tokens (metadata only)
Temperature: 0.3
```

**Why Primary**: Free tier, high rate limits (1500 RPM), massive context window

#### Secondary Route: OpenAI
```python
Model: LLM_MODEL_NANO (default: gpt-5-nano)
Env Override: LLM_MODEL_NANO
Max Input: 128K tokens
Max Output: 2048 tokens (metadata only)
Temperature: 0.3
```

**Why Secondary**: Reliable, good quality, paid but reasonable rates

#### Tertiary Route: Groq
```python
Model: GROQ_LLM_MODEL (default: openai/gpt-oss-20b)
Env Override: GROQ_LLM_MODEL
Max Input: 128K tokens
Max Output: 1024 tokens (metadata only)
Temperature: 0.3
```

**Why Tertiary**: Fast inference, rate limit handling with exponential backoff

**Purpose**: Generate rich metadata for document chunks to enable semantic search

**Features**:
- Contextual metadata generation (considers document title, section, page)
- Chunking validation (assesses semantic coherence)
- JSON schema enforcement across all providers
- Entity extraction, topic classification, query pattern generation
- Document position tracking

**Prompt Template** (simplified):
```
Document Title: {document_title}
Section: {section_title}
Page: {page_number}
Domain: {domain}

Content: {content}

Generate metadata:
- title: Descriptive title (50-100 chars)
- topics: 3-7 relevant topics
- entities: Named entities with types
- search_keywords: 5-10 keywords
- query_patterns: 3-5 natural language queries
- relationships: Document-level relationships
- chunking_validation: Semantic coherence assessment
```

---

### 9. Review and Feedback Generation (`review_page`)

**File**: `api_handlers/chat_gpt_completion.py:2759`

```python
Model: LLM_MODEL (default: gpt-4.1-nano)
Env Override: LLM_MODEL
Max Input: 128K tokens
Max Output: min(4096, 128000 - input_tokens - 7)
Temperature: 0.7 (higher for creative feedback)
```

**Purpose**: Review content and provide structured feedback with memory context

**Features**:
- Agent-based instructions (Product Agent, Engineering Agent, etc.)
- Two-stage generation: memory retrieval → feedback generation
- Up to 3 relevant memories included
- Structured JSON response with feedback areas, questions, examples, rewrites

---

### 10. Image Prompt Generation (`generate_image_prompt`)

**File**: `api_handlers/chat_gpt_completion.py:326`

```python
Model: LLM_MODEL (default: gpt-4.1-nano)
Env Override: LLM_MODEL
Max Input: 8K tokens (validated)
Max Output: Not explicitly set
Temperature: Not set
```

**Purpose**: Generate DALL-E prompts from memory content

**Features**:
- DALL-E content policy compliance
- Concise prompts (< 27 words)
- Token limit validation

---

## Token Budget Management

### Dynamic Token Budget Calculation

Most operations use dynamic token budgeting to maximize output quality:

```python
# Standard pattern
max_tokens_for_completion = min(
    MAX_OUTPUT,  # Model's max output limit
    CONTEXT_WINDOW - input_tokens - buffer
)

# Minimum threshold check
if max_tokens_for_completion < MIN_THRESHOLD:
    raise ValueError("Available tokens too few")
```

### Model-Specific Limits

| Model | Context Window | Max Output | Buffer |
|-------|---------------|------------|--------|
| `gpt-4.1-nano` | 128K | 4096 | 7-2000 |
| `gpt-5-mini` | 128K | 16K | 2000 |
| `gpt-5-nano` | 128K | 2048 | N/A |
| `openai/gpt-oss-20b` (Groq) | 128K | Not set | N/A |
| `gemini-2.5-flash` | 1M | 8192 | N/A |

### Parameter Normalization

For o-series and gpt-5 models, parameters are automatically normalized:

```python
def _normalize_chat_kwargs(self, kwargs: dict) -> dict:
    """
    - o-series and gpt-5 models: drop temperature
    - Map max_tokens → max_completion_tokens for these models
    """
    if model_name.startswith("o") or model_name.startswith("gpt-5"):
        kwargs.pop("temperature", None)
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    return kwargs
```

---

## Fallback Strategy

### Multi-Provider Resilience

The codebase implements sophisticated fallback chains:

#### Cypher Query Generation Chain
```
1. Groq (openai/gpt-oss-20b) - Fast, cheap
   ↓
2. Groq with Instructor - Alternative structured output
   ↓
3. OpenAI (LLM_MODEL) - Reliable fallback
   ↓
4. Simple base query - Final fallback
```

#### Document Processing Chain
```
1. Gemini 2.5 Flash - Free, high limits
   ↓
2. OpenAI (gpt-5-nano) - Reliable, paid
   ↓
3. Groq (openai/gpt-oss-20b) - Fast alternative
   ↓
4. Deterministic fallback - Rule-based
```

#### Query Classification Chain
```
1. OpenAI typed parse (gpt-5-mini) - Structured outputs
   ↓
2. OpenAI JSON mode (gpt-5-mini) - Simpler format, 1/4 tokens
   ↓
3. Default response - Minimal safe response
```

### Rate Limit Handling

Groq calls implement exponential backoff:

```python
if response.status_code == 429:
    if retry_count < max_retries:
        retry_delay = 2 ** retry_count  # 1s, 2s, 4s
        await asyncio.sleep(retry_delay)
        return await self._call_groq_structured(
            prompt, 
            retry_count=retry_count + 1
        )
```

### Timeout Management

Critical operations have timeout protection:

```python
# Query classification: 20s primary, 10s fallback
response = await asyncio.wait_for(
    chat_gpt.async_client.beta.chat.completions.parse(...),
    timeout=20.0
)

# Fallback with shorter timeout
raw = await asyncio.wait_for(
    chat_gpt._create_completion_with_fallback_async(...),
    timeout=10.0
)
```

---

## Cost Optimization

### Model Selection by Cost

Models are selected based on operation complexity and cost:

| Operation Type | Model Choice | Reasoning |
|----------------|-------------|-----------|
| Simple metadata | Gemini 2.5 Flash | Free tier, high limits |
| Pattern selection | Groq OSS-20B | Very cheap, fast |
| Content generation | gpt-4.1-nano | Balanced quality/cost |
| Complex reasoning | gpt-5-mini | Worth the cost for accuracy |

### Cost Tracking

Operations track token usage and calculate costs:

```python
output_tokens = self.count_tokens(json.dumps(result))
total_cost = self.calculate_cost(token_count, output_tokens)

# Cost calculation
input_cost = input_tokens * self.cost_per_input_token
output_cost = output_tokens * self.cost_per_output_token
```

Current rates (for cloud models):
- Input: $0.0000001 per token
- Output: $0.0000004 per token

---

## Configuration Examples

### Production Configuration (.env)

```bash
# Primary Models
LLM_MODEL=gpt-4.1-nano
LLM_MODEL_MINI=gpt-5-mini
LLM_MODEL_NANO=gpt-5-nano

# Provider-Specific Models
GROQ_PATTERN_SELECTOR_MODEL=openai/gpt-oss-20b
GROQ_LLM_MODEL=openai/gpt-oss-20b
GEMINI_MODEL_FAST=gemini-2.5-flash

# API Keys
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AI...

# Configuration
LLM_LOCATION_CLOUD=true
USE_DOTENV=true
```

### Local Development Configuration

```bash
# Use local Ollama for cost savings
LLM_LOCATION_CLOUD=false

# Still need cloud keys for certain operations
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

### Cost-Optimized Configuration

```bash
# Maximize use of free tiers
LLM_MODEL=gpt-4.1-nano              # Cheapest OpenAI option
GEMINI_MODEL_FAST=gemini-2.5-flash  # Free tier
GROQ_PATTERN_SELECTOR_MODEL=openai/gpt-oss-20b  # Cheap via Groq
```

---

## Summary Statistics

### Total LLM Call Types: 10 distinct operations
### Total Providers: 4 (OpenAI, Groq, Google Gemini, Ollama)
### Total Models Configured: 7 unique models
### Environment Variables: 6 model variables, 3 API keys, 2 config flags

### Operations by Frequency (estimated):
1. **High Frequency**: Document processing, message analysis, query classification
2. **Medium Frequency**: Content generation, Cypher query generation
3. **Low Frequency**: UseCase generation, schema generation, image prompts

### Provider Usage:
- **OpenAI**: 60% of operations (primary workhorse)
- **Groq**: 30% of operations (fast inference, pattern selection)
- **Gemini**: 10% of operations (document processing, fallback)

---

## Best Practices

### When Adding New LLM Calls

1. **Choose the Right Model**:
   - Simple operations → Groq or Gemini
   - Complex reasoning → OpenAI Mini or Main
   - Document processing → Gemini (free tier)

2. **Implement Fallbacks**:
   - Always have at least one fallback provider
   - Include timeout handling
   - Log failures for monitoring

3. **Manage Token Budgets**:
   - Calculate dynamic limits based on input
   - Set minimum thresholds
   - Trim content before sending

4. **Use Structured Outputs**:
   - Prefer `response_format` with Pydantic models
   - Validate responses before use
   - Handle parsing errors gracefully

5. **Track Metrics**:
   - Log token usage
   - Calculate costs
   - Monitor performance

---

## Troubleshooting

### Common Issues

#### "Available tokens too few" Error
**Cause**: Input too large for context window
**Solution**: Increase token trimming or split into smaller chunks

#### Groq Tool Call Failures
**Cause**: Known Groq reliability issue with tool calling
**Solution**: Automatic fallback to OpenAI implemented

#### Rate Limit Errors (429)
**Cause**: Exceeded provider rate limits
**Solution**: 
- Groq: Exponential backoff implemented
- OpenAI: Increase delays or use different API key tier

#### JSON Parsing Errors
**Cause**: Model returned malformed JSON
**Solution**: 
- Use `response_format={"type": "json_object"}`
- Validate with Pydantic models
- Implement fallback to alternative provider

---

## Version History

- **v1.0** (2025-11-18): Initial documentation of all LLM calls
- Environment variables documented with defaults and priorities
- Multi-provider fallback strategy documented
- Token budget management explained

---

## References

- **OpenAI API Docs**: https://platform.openai.com/docs/api-reference
- **Groq API Docs**: https://console.groq.com/docs
- **Google Gemini Docs**: https://ai.google.dev/docs
- **Instructor Library**: https://github.com/jxnl/instructor

---

*Last Updated: 2025-11-18*
*Maintainer: Development Team*

