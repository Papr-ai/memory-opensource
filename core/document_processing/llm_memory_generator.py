"""
LLM-based Memory Structure Generator

This module uses large language models to intelligently generate memory structures
from document analysis, creating optimized memory representations for different
content types and domains.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from models.hierarchical_models import (
    ContentElement, DocumentStructure, DomainContext,
    MemoryStructure, MemoryTransformer, HierarchicalSection,
    TableElement  # For adaptive batch sizing
)
from models.shared_types import MemoryMetadata
from models.memory_models import AddMemoryRequest
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


class LLMPromptTemplates:
    """Template prompts for different domains and content types"""

    FINANCIAL_TABLE_PROMPT = """
    You are analyzing a financial table. Generate ONLY metadata for searchability - DO NOT rewrite the table content.

    Table Content: {content}
    Table Metadata: {metadata}

    IMPORTANT:
    - Analyze the table above to extract financial metadata
    - DO NOT include the "content" field in your response
    - We will preserve the original table exactly as-is
    - Focus on making the table searchable via natural language

    Generate metadata that enables:
    1. Descriptive title for this financial data
    2. Key financial metrics and trends identified
    3. Queryable metadata for financial analysis
    4. Time-period context if applicable
    5. Natural language queries users might ask

    Return ONLY metadata as JSON (DO NOT include content field):
    {{
        "title": "Descriptive title (e.g., 'Q4 2024 Revenue by Product Line')",
        "topics": ["financial", "revenue", "quarterly"],
        "financial_metrics": ["revenue", "growth", "margin"],
        "time_period": "Q4 2024",
        "currency": "USD",
        "query_patterns": ["What was the revenue growth?", "Show me Q4 performance"],
        "relationships": ["part_of_annual_report", "related_to_financial_analysis"],
        "metadata": {{
            "table_type": "financial_summary"
        }}
    }}
    """

    HEALTHCARE_TABLE_PROMPT = """
    You are analyzing a healthcare/medical table from a document. Your task is to create a memory structure optimized for medical information retrieval.

    Table Content: {content}
    Table Metadata: {metadata}

    Create a memory structure that:
    1. Identifies medical entities (drugs, dosages, lab values, conditions)
    2. Preserves clinical context and relationships
    3. Creates temporal relationships for longitudinal data
    4. Ensures compliance with medical data standards
    5. Enables clinical insights generation
    6. Supports questions about patient care, treatment outcomes, or medical trends

    Return a JSON structure with:
    - title: Clinical description of the data
    - content: Medical summary with key clinical insights
    - metadata: Medical entities and clinical context
    - compliance_tags: Relevant compliance markers (HIPAA, etc.)
    - temporal_relationships: Time-based medical data relationships
    - clinical_insights: Generated insights about the medical data
    """

    GENERAL_CONTENT_PROMPT = """
    You are analyzing content from a document. Generate ONLY metadata to enhance searchability - DO NOT rewrite or return the content itself.

    # DOCUMENT CONTEXT (Based on 2025 Contextual Retrieval Research)
    Document Title: {document_title}
    Document Type: {document_type}
    Total Pages: {total_pages}
    Domain: {domain}

    # SECTION CONTEXT
    Current Section: {section_title}
    Page Number: {page_number}

    # CONTENT TO ANALYZE
    Content Type: {content_type}
    Content: {content}

    # ADDITIONAL CONTEXT
    {context}

    IMPORTANT:
    - Analyze the content above IN THE CONTEXT of the overall document "{document_title}"
    - Consider how this content relates to the document's main themes and topics
    - Generate metadata that helps users find this when searching about the document
    - DO NOT include the "content" field in your response
    - We will preserve the original content exactly as-is
    - Focus on generating rich metadata for search and discovery
    - ALSO validate whether this content forms a semantically coherent chunk

    Generate metadata that enables:
    1. A concise, descriptive title (50-100 characters) that captures what this content is about IN CONTEXT of the document
    2. Key entities and concepts identified in the content
    3. Topic classification (3-7 relevant topics) - consider document-level themes
    4. Search keywords for improved findability (5-10 keywords)
    5. Natural language query patterns users might ask to find this content (3-5 questions)
    6. Relationships to other content in the document
    7. Additional structured metadata
    8. **CHUNKING VALIDATION**: Assess if this content is well-bounded semantically

    Chunking Analysis:
    - Does this content represent a complete semantic unit (introduction, problem statement, solution, conclusion)?
    - If content seems to end mid-thought or start mid-section, note it
    - If content mixes unrelated topics (e.g., problem statement + unrelated table + solution), suggest split points
    - For very large content (>5000 chars), identify natural semantic boundaries

    Return ONLY metadata as JSON (DO NOT include content field):
    {{
        "title": "Descriptive title considering document context (50-100 characters)",
        "topics": ["topic1", "topic2", "topic3"],
        "entities": [{{"name": "entity1", "type": "person"}}, {{"name": "entity2", "type": "concept"}}],
        "search_keywords": ["keyword1", "keyword2", "keyword3"],
        "query_patterns": ["How do I...", "What is...", "Where can I find..."],
        "relationships": ["related_to_section_x", "part_of_chapter_y", "supports_document_theme_z"],
        "document_position": {{
            "section": "{section_title}",
            "page": "{page_number}",
            "semantic_role": "introduction|analysis|conclusion|supporting_data|example"
        }},
        "chunking_validation": {{
            "is_coherent": true,
            "completeness": "complete|incomplete_start|incomplete_end|mixed_topics",
            "suggested_split_points": [],
            "reasoning": "Brief explanation of chunk quality"
        }},
        "metadata": {{
            "key": "value"
        }}
    }}
    """

    IMAGE_ANALYSIS_PROMPT = """
    You are analyzing an image/chart from a document. Create a memory structure that captures visual information and makes it searchable.

    Image Description: {description}
    OCR Text: {ocr_text}
    Image Type: {image_type}
    Image URL: {image_url}
    Context: {context}

    Create a memory structure that:
    1. Describes visual content comprehensively
    2. Extracts data from charts/graphs if present
    3. Identifies objects, text, and relationships in the image
    4. Creates searchable metadata for visual content
    5. Enables questions about visual information
    6. Links to related textual content
    7. PRESERVES the image URL in the content field as markdown: ![description](url) if available

    Return a JSON structure with:
    - title: Descriptive title for the visual content
    - content: Comprehensive description of visual information
    - visual_elements: Identified visual components
    - extracted_data: Data extracted from charts/graphs
    - search_metadata: Metadata for visual content search
    - related_concepts: Concepts related to the visual content
    """


class LLMMemoryStructureGenerator(BaseModel):
    """Generates optimized memory structures using LLM analysis"""

    def __init__(self, **data):
        super().__init__(**data)
        # Load environment variables conditionally (respects USE_DOTENV)
        try:
            from dotenv import load_dotenv
            from os import environ as env
            use_dotenv = env.get("USE_DOTENV", "true").lower() == "true"
            if use_dotenv:
                load_dotenv()
        except Exception:
            pass  # Ignore errors in case dotenv is not available

    async def generate_memory_structure_for_content(
        self,
        content_element: ContentElement,
        domain_context: Optional[DomainContext] = None,
        base_metadata: Optional[MemoryMetadata] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> AddMemoryRequest:
        """
        Generate an optimized memory structure for a content element using LLM.

        Args:
            content_element: The content to analyze
            domain_context: Optional domain context
            base_metadata: Base metadata to include
            document_metadata: Document-level context (title, type, pages, domain, etc.)
                              for contextual retrieval (2025 research)
        """

        try:
            logger.info(f"Generating LLM-optimized memory structure for {content_element.content_type.value}")

            # Select appropriate prompt based on content type and domain
            prompt = self._select_prompt_template(content_element, domain_context)

            # Generate LLM response with document context
            llm_response = await self._call_llm(prompt, content_element, domain_context, document_metadata)

            # Parse and validate LLM response
            memory_structure = self._parse_llm_response(llm_response, content_element, base_metadata)

            # Log with correct attribute access (memory_structure is AddMemoryRequest, not dict)
            title_preview = (memory_structure.content[:50] + "...") if memory_structure.content else "Untitled"
            logger.info(f"Successfully generated LLM memory structure: {title_preview}")

            return memory_structure

        except Exception as e:
            # Enhanced error logging with full context
            logger.error(f"LLM memory generation failed: {e}", exc_info=True)
            logger.error(f"Failed element details - Type: {type(content_element)}, ID: {getattr(content_element, 'element_id', 'UNKNOWN')}, content_type: {getattr(content_element, 'content_type', 'MISSING')}")
            logger.error(f"Content length: {len(getattr(content_element, 'content', ''))} chars")

            # Fallback to basic transformation
            return MemoryTransformer.content_element_to_memory_request(
                content_element, base_metadata
            )

    def _select_prompt_template(
        self,
        content_element: ContentElement,
        domain_context: Optional[DomainContext]
    ) -> str:
        """Select the appropriate prompt template based on content and domain"""

        # Domain-specific prompts
        if domain_context:
            domain = domain_context.domain.lower()

            if domain == "financial" and content_element.content_type.value == "table":
                return LLMPromptTemplates.FINANCIAL_TABLE_PROMPT

            elif domain in ["healthcare", "medical"] and content_element.content_type.value == "table":
                return LLMPromptTemplates.HEALTHCARE_TABLE_PROMPT

        # Content-type specific prompts
        if content_element.content_type.value in ["image", "chart", "diagram"]:
            return LLMPromptTemplates.IMAGE_ANALYSIS_PROMPT

        # Default general prompt
        return LLMPromptTemplates.GENERAL_CONTENT_PROMPT

    async def _call_llm(
        self,
        prompt_template: str,
        content_element: ContentElement,
        domain_context: Optional[DomainContext],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Call LLM with the formatted prompt including document context.

        Args:
            prompt_template: The prompt template to use
            content_element: The content element to analyze
            domain_context: Optional domain context
            document_metadata: Document-level context for contextual retrieval
        """
        import os
        import httpx

        # Extract document context (defaults for missing fields)
        if document_metadata is None:
            document_metadata = {}

        doc_title = document_metadata.get('title', content_element.metadata.get('document_title', 'Unknown Document'))
        doc_type = document_metadata.get('type', 'document')
        total_pages = document_metadata.get('total_pages', content_element.metadata.get('total_pages', 'Unknown'))
        domain = document_metadata.get('domain', domain_context.domain if domain_context else 'general')
        section_title = content_element.metadata.get('section_title', 'Unknown Section')
        page_number = content_element.metadata.get('page_number', 'Unknown')

        # Format prompt with content data and document context
        if hasattr(content_element, 'structured_data') and content_element.structured_data:
            # For tables and structured content
            # Support both table-specific prompts and general prompts
            try:
                prompt = prompt_template.format(
                    content=content_element.content,
                    metadata=json.dumps(content_element.metadata or {}, indent=2),
                    structured_data=json.dumps(content_element.structured_data, indent=2),
                    content_type=content_element.content_type.value,
                    context=json.dumps(content_element.metadata or {}, indent=2),
                    # Document context fields
                    document_title=doc_title,
                    document_type=doc_type,
                    total_pages=total_pages,
                    domain=domain,
                    section_title=section_title,
                    page_number=page_number
                )
            except KeyError as e:
                # Fallback if template doesn't use all variables
                logger.warning(f"Template missing some variables: {e}, using fallback formatting")
                prompt = prompt_template.format(
                    content=content_element.content,
                    metadata=json.dumps(content_element.metadata or {}, indent=2)
                )
        elif hasattr(content_element, 'image_description'):
            # For images
            try:
                prompt = prompt_template.format(
                    description=content_element.image_description or content_element.content,
                    ocr_text=getattr(content_element, 'ocr_text', ''),
                    image_type=content_element.content_type.value,
                    image_url=getattr(content_element, 'image_url', '') or '',
                    context=json.dumps(content_element.metadata or {}, indent=2),
                    # Document context fields
                    document_title=doc_title,
                    document_type=doc_type,
                    total_pages=total_pages,
                    domain=domain,
                    section_title=section_title,
                    page_number=page_number
                )
            except KeyError:
                prompt = prompt_template.format(
                    description=content_element.image_description or content_element.content,
                    ocr_text=getattr(content_element, 'ocr_text', ''),
                    image_type=content_element.content_type.value,
                    image_url=getattr(content_element, 'image_url', '') or '',
                    context=json.dumps(content_element.metadata or {}, indent=2)
                )
        else:
            # For general content
            try:
                prompt = prompt_template.format(
                    content=content_element.content,
                    content_type=content_element.content_type.value,
                    context=json.dumps(content_element.metadata or {}, indent=2),
                    # Document context fields
                    document_title=doc_title,
                    document_type=doc_type,
                    total_pages=total_pages,
                    domain=domain,
                    section_title=section_title,
                    page_number=page_number
                )
            except KeyError as e:
                logger.warning(f"Template missing variables: {e}, using fallback")
                prompt = prompt_template.format(
                    content=content_element.content,
                    content_type=content_element.content_type.value,
                    context=json.dumps(content_element.metadata or {}, indent=2)
                )

        logger.info("Calling LLM for memory structure generation...")

        # Token limit checking (rough estimate: 1 token ≈ 4 chars)
        estimated_input_tokens = len(prompt) // 4
        max_input_limit = 30000  # Conservative limit for most models (Groq: 32K, GPT-4o-mini: 128K, Gemini: 32K)

        if estimated_input_tokens > max_input_limit:
            logger.warning(f"Prompt too long ({estimated_input_tokens} tokens, limit {max_input_limit}) - using deterministic fallback")
            return await self._simulate_llm_response(content_element, domain_context)

        # Try models in order of preference: Gemini > Groq > OpenAI (last resort)
        # Gemini 1.5 Flash: 1M context, 8K output, free (with limits), VERY RELIABLE
        # Groq openai/gpt-oss-20b: 8K context, FREE, fast, supports structured outputs, HIGH rate limits
        # OpenAI GPT-4o-mini: Last resort fallback (may hit rate limits with parallel processing)
        try:
            # Try Gemini first (free, reliable, huge context window)
            return await self._call_gemini_pro(prompt)
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}, falling back to Groq")
            try:
                return await self._call_groq(prompt)
            except Exception as e2:
                logger.warning(f"Groq call failed: {e2}, falling back to OpenAI (last resort)")
                try:
                    return await self._call_openai_mini(prompt)
                except Exception as e3:
                    logger.error(f"All LLM providers failed: {e3}, using deterministic fallback")
                    return await self._simulate_llm_response(content_element, domain_context)

    async def _call_openai_mini(self, prompt: str) -> str:
        """Call OpenAI GPT-4o-mini API (primary route - high limits, low cost)

        GPT-4o-mini specs:
        - 128K context window
        - 16K max output tokens
        - $0.150 per 1M input tokens, $0.600 per 1M output tokens
        - Fast and reliable
        """
        import httpx
        from os import environ as env

        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not found")

        # Use configurable model (default to gpt-5-nano)
        # Fallback to gpt-4o-mini if gpt-5-nano not available
        model = env.get("LLM_MODEL_NANO", "gpt-5-nano")

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing content and generating structured metadata. Always respond with valid JSON containing only metadata fields."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2048,  # Metadata only, shouldn't need more
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return content
            else:
                raise Exception(f"No valid response from OpenAI: {result}")

    async def _call_gemini_pro(self, prompt: str) -> str:
        """Call Google Gemini 2.5 Flash API with structured JSON schema (primary LLM)
        
        Gemini 2.5 Flash (gemini-2.0-flash-exp) specs:
        - 1M token context window
        - 8K max output tokens
        - Free tier with high rate limits (1500 RPM)
        - Native JSON schema support (responseSchema)
        - Improved structured output reliability vs 1.5
        
        Note:
            Uses responseSchema to enforce strict JSON structure with required fields.
            Validates response structure and JSON format before returning.
            Falls back to Groq/OpenAI if response is malformed or blocked.
        """
        import httpx
        from os import environ as env

        api_key = env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_API_KEY or GEMINI_API_KEY not found")
        
        # Use GEMINI_MODEL_FAST from env (gemini-2.5-flash by default)
        # Gemini 2.5 Flash has improved structured output support
        model = env.get("GEMINI_MODEL_FAST", "gemini-2.5-flash")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        headers = {
            "Content-Type": "application/json",
        }

        # Define strict JSON schema for Gemini (same structure as Groq)
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "title": {
                    "type": "STRING",
                    "description": "A concise, descriptive title for the memory (max 100 chars)"
                },
                "topics": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "List of main topics or themes (max 5)"
                },
                "query_patterns": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Example queries that would retrieve this memory (max 3)"
                },
                "key_concepts": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Key concepts and entities mentioned (max 10)"
                }
            },
            "required": ["title"]
        }

        # Add system instruction to clarify expectations
        system_instruction = {
            "parts": [{
                "text": "You are an expert at analyzing content and generating structured metadata for memory retrieval. "
                       "Generate ONLY metadata fields (title, topics, query_patterns, key_concepts). "
                       "Do NOT rewrite or summarize the content itself - it is preserved separately. "
                       "Focus on making the memory easily discoverable through semantic search."
            }]
        }

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "systemInstruction": system_instruction,
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,  # Reduced since we only need metadata
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{url}?key={api_key}",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                result = response.json()

                # Check for safety filters or blocked content
                if 'candidates' not in result or len(result['candidates']) == 0:
                    logger.warning(f"Gemini returned no candidates. Response: {result}")
                    raise Exception(f"No candidates in Gemini response: {result}")
                
                candidate = result['candidates'][0]
                
                # Check finish reason (could be SAFETY, MAX_TOKENS, etc.)
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                if finish_reason not in ['STOP', 'MAX_TOKENS']:
                    logger.warning(f"Gemini finished with reason: {finish_reason}. Candidate: {candidate}")
                    raise Exception(f"Gemini finished with unexpected reason: {finish_reason}")
                
                # Check if content exists and has parts
                if 'content' not in candidate:
                    logger.error(f"Gemini response missing 'content' key. Full response: {result}")
                    raise Exception(f"Gemini response missing 'content' key")
                
                if 'parts' not in candidate['content']:
                    logger.error(f"Gemini response missing 'parts' key. Content: {candidate['content']}")
                    raise Exception(f"Gemini response missing 'parts' key")
                
                if len(candidate['content']['parts']) == 0:
                    logger.error(f"Gemini response has empty 'parts' array. Content: {candidate['content']}")
                    raise Exception(f"Gemini response has empty 'parts' array")
                
                # Extract text from first part
                first_part = candidate['content']['parts'][0]
                if 'text' not in first_part:
                    logger.error(f"Gemini part missing 'text' key. Part: {first_part}")
                    raise Exception(f"Gemini part missing 'text' key")
                
                content = first_part['text']
                
                # Validate it's valid JSON
                import json
                try:
                    parsed = json.loads(content)
                    if 'title' not in parsed:
                        logger.warning(f"Gemini JSON missing required 'title' field: {content}")
                        raise Exception("Gemini response missing required 'title' field")
                except json.JSONDecodeError as e:
                    logger.error(f"Gemini returned invalid JSON: {content}")
                    raise Exception(f"Gemini returned invalid JSON: {e}")
                
                return content
                
        except httpx.HTTPStatusError as e:
            logger.warning(f"Gemini HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.warning(f"Gemini request error: {e}")
            raise
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")
            raise

    async def _call_groq(self, prompt: str, retry_count: int = 0, max_retries: int = 3) -> str:
        """Call Groq API with structured JSON schema output
        
        Args:
            prompt: The prompt to send to the LLM
            retry_count: Current retry attempt (for exponential backoff)
            max_retries: Maximum number of retries on rate limit errors
            
        Note:
            Groq rate limits (free tier):
            - 14,400 requests/day
            - 30 requests/minute
            - openai/gpt-oss-20b: Supports structured outputs with JSON schema
            
            If rate limited (429), retries with exponential backoff: 1s, 2s, 4s
        """
        import httpx
        import asyncio
        from os import environ as env

        # Check for Groq API key
        api_key = env.get("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not found")

        # Use openai/gpt-oss-20b which supports structured outputs
        # See: https://console.groq.com/docs/structured-outputs#supported-models
        # llama-3.3-70b-versatile does NOT support json_schema, only json_object mode
        model = env.get("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Define JSON schema for structured output (simplified for Groq compatibility)
        # NOTE: We preserve original content, so we only need metadata from LLM
        json_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A concise, descriptive title for the memory (max 100 chars)"
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of main topics or themes (max 5)"
                },
                "query_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Example queries that would retrieve this memory (max 3)"
                },
                "key_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key concepts and entities mentioned (max 10)"
                }
            },
            "required": ["title"],
            "additionalProperties": False
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at analyzing documents and extracting metadata. Generate a concise title, relevant topics, example search queries, and key concepts. Do NOT rewrite or summarize the content - we preserve the original text. Focus only on generating helpful metadata for search and retrieval."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1024,  # Reduced since we only need metadata
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_metadata",
                    "description": "Metadata for document memory (title, topics, queries, concepts)",
                    "schema": json_schema,
                    "strict": False  # Allow flexibility for better compatibility
                }
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                
                # Log error details if request fails
                if response.status_code == 400:
                    error_detail = response.text
                    logger.error(f"Groq 400 Bad Request - Response: {error_detail}")
                    logger.error(f"Groq 400 Bad Request - Payload model: {payload['model']}")
                    raise Exception(f"Groq API rejected request: {error_detail[:500]}")
                
                # Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    if retry_count < max_retries:
                        retry_delay = 2 ** retry_count  # Exponential: 1s, 2s, 4s
                        logger.warning(f"Groq rate limit hit (429). Retrying in {retry_delay}s... (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        return await self._call_groq(prompt, retry_count + 1, max_retries)
                    else:
                        raise Exception(f"Groq rate limit exceeded after {max_retries} retries")
                
                response.raise_for_status()

                result = response.json()

                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content
                else:
                    raise Exception(f"No valid response from Groq: {result}")
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Already handled above, but catch again for safety
                raise Exception(f"Groq rate limit exceeded: {e}")
            raise

    async def _simulate_llm_response(
        self,
        content_element: ContentElement,
        domain_context: Optional[DomainContext]
    ) -> str:
        """Simulate LLM response for testing purposes"""

        # Simulate realistic LLM responses based on content type
        if content_element.content_type.value == "table":
            return json.dumps({
                "title": f"Data Table: {content_element.content[:50]}...",
                "content": f"This table contains structured data with key insights about {content_element.content[:100]}...",
                "metadata": {
                    "content_type": "structured_table",
                    "data_categories": ["financial", "performance"],
                    "time_period": "quarterly",
                    "key_metrics": ["revenue", "growth", "performance"]
                },
                "relationships": ["related_to_quarterly_report", "part_of_financial_analysis"],
                "query_patterns": [
                    "What are the key financial metrics?",
                    "Show me quarterly performance data",
                    "What trends are visible in this data?"
                ]
            })

        elif content_element.content_type.value in ["image", "chart"]:
            return json.dumps({
                "title": f"Visual Content: {content_element.content[:50]}...",
                "content": f"This visual element shows {content_element.content}. Key visual insights include graphical representation of data trends and patterns.",
                "visual_elements": ["chart", "data_visualization", "trends"],
                "extracted_data": {"chart_type": "bar_chart", "data_points": 12},
                "search_metadata": {
                    "visual_type": "chart",
                    "contains_data": True,
                    "chart_category": "performance"
                },
                "related_concepts": ["data_analysis", "performance_metrics", "trends"]
            })

        else:
            return json.dumps({
                "title": f"Content: {content_element.content[:50]}...",
                "content": f"Enhanced content with insights: {content_element.content}",
                "topics": ["document_content", "analysis"],
                "entities": [{"type": "concept", "name": "main_topic"}],
                "relationships": ["part_of_document"],
                "search_keywords": ["content", "document", "information"]
            })

    def _parse_llm_response(
        self,
        llm_response: str,
        content_element: ContentElement,
        base_metadata: Optional[MemoryMetadata]
    ) -> AddMemoryRequest:
        """Parse LLM response and create AddMemoryRequest"""

        try:
            # Parse JSON response from LLM
            response_data = json.loads(llm_response)

            # VALIDATION: Ensure response_data is a dict
            if not isinstance(response_data, dict):
                logger.error(f"LLM returned non-dict response: {type(response_data)}. Element: {content_element.element_id}")
                raise ValueError(f"LLM response must be a dict, got {type(response_data)}")

            # VALIDATION: Warn about missing expected keys and provide fallbacks
            expected_keys = ["title", "content"]
            missing_keys = [k for k in expected_keys if k not in response_data]
            if missing_keys:
                logger.warning(f"LLM response missing keys {missing_keys} for element {content_element.element_id}")
                
                # Fallback: Use original element content if LLM didn't provide it
                if "content" not in response_data:
                    response_data["content"] = content_element.content
                    logger.info(f"Using original element content as fallback for {content_element.element_id}")
                
                # Fallback: Generate a title if LLM didn't provide one
                if "title" not in response_data:
                    content_preview = content_element.content[:50] if content_element.content else "Untitled"
                    response_data["title"] = f"{content_element.content_type.value.title()}: {content_preview}..."
                    logger.info(f"Generated fallback title for {content_element.element_id}")

            # VALIDATION: Ensure metadata is dict if present
            if "metadata" in response_data and not isinstance(response_data["metadata"], (dict, list, type(None))):
                logger.warning(f"LLM metadata has unexpected type {type(response_data['metadata'])}, converting to string")
                response_data["metadata"] = str(response_data["metadata"])

            # Create enhanced metadata
            enhanced_metadata = base_metadata.model_copy() if base_metadata else MemoryMetadata()
            if not enhanced_metadata.customMetadata:
                enhanced_metadata.customMetadata = {}

            # FIRST: Preserve original element metadata (hierarchical structure, etc.)
            if hasattr(content_element, 'metadata') and content_element.metadata:
                # Preserve all original metadata from the element
                for key, value in content_element.metadata.items():
                    # Only add if not already in customMetadata (don't overwrite base_metadata)
                    if key not in enhanced_metadata.customMetadata:
                        enhanced_metadata.customMetadata[key] = value

            # Preserve ImageElement specific attributes (always preserve, even if None/empty)
            if hasattr(content_element, 'image_url'):
                enhanced_metadata.customMetadata['image_url'] = content_element.image_url
            if hasattr(content_element, 'image_description'):
                enhanced_metadata.customMetadata['image_description'] = content_element.image_description
            if hasattr(content_element, 'image_hash'):
                enhanced_metadata.customMetadata['image_hash'] = content_element.image_hash
            if hasattr(content_element, 'ocr_text'):
                enhanced_metadata.customMetadata['ocr_text'] = content_element.ocr_text

            # THEN: Add LLM-generated metadata (these will take precedence)
            enhanced_metadata.customMetadata.update({
                "llm_generated": True,
                "content_type": content_element.content_type.value,
                "element_id": content_element.element_id,
                "llm_enhanced": True,
                "generation_timestamp": datetime.now().isoformat()
            })

            # Add LLM response metadata (with type safety)
            llm_metadata = response_data.get("metadata", {})
            if isinstance(llm_metadata, dict):
                for key, value in llm_metadata.items():
                    enhanced_metadata.customMetadata[f"llm_{key}"] = value
            elif isinstance(llm_metadata, list):
                # LLM returned list instead of dict - store as array
                logger.warning(f"LLM returned metadata as list instead of dict: {llm_metadata}")
                enhanced_metadata.customMetadata["llm_metadata_tags"] = llm_metadata
            else:
                logger.warning(f"LLM returned unexpected metadata type: {type(llm_metadata)}")

            # Add topics if provided
            if "topics" in response_data:
                enhanced_metadata.topics = response_data["topics"]

            # Extract query patterns (will be added to content for embedding)
            query_patterns = response_data.get("query_patterns", [])
            if query_patterns:
                # Store in metadata for reference
                enhanced_metadata.customMetadata["query_patterns"] = query_patterns

            # Add relationships (serialize as JSON string to match Pydantic string type expectation)
            if "relationships" in response_data:
                rels = response_data["relationships"]
                if isinstance(rels, (list, dict)):
                    enhanced_metadata.customMetadata["llm_relationships"] = json.dumps(rels)
                else:
                    enhanced_metadata.customMetadata["llm_relationships"] = str(rels)

            # ALWAYS use original content - LLM provides metadata only
            content = content_element.content

            # Warn if LLM incorrectly returned content (violating instructions)
            if "content" in response_data:
                logger.warning(f"LLM returned content field despite instructions not to - using original content anyway")
                enhanced_metadata.customMetadata["llm_note"] = "LLM returned content but we preserved original"

            # Use LLM-generated title (fallback to content preview if missing)
            title = response_data.get("title", content[:100])

            # Mark that content was preserved (not LLM-generated)
            enhanced_metadata.customMetadata["content_source"] = "original_preserved"

            # APPEND QUERY PATTERNS TO CONTENT for embedding
            # This improves semantic search by embedding the questions users might ask
            if query_patterns and isinstance(query_patterns, list):
                query_text = "\n\n---\nRelated Questions:\n" + "\n".join(f"- {q}" for q in query_patterns)
                content = content + query_text
                logger.debug(f"Appended {len(query_patterns)} query patterns to content for embedding")

            return AddMemoryRequest(
                content=content,
                type="document",  # Using document type for rich content
                metadata=enhanced_metadata,
                context=[],
                relationships_json=[]
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback to basic transformation
            return MemoryTransformer.content_element_to_memory_request(
                content_element, base_metadata
            )

    def _consolidate_small_elements(
        self,
        content_elements: List[ContentElement],
        max_combined_chars: int = 6000  # Target: 1-2 pages per memory (increased to support full 2-page chunks)
    ) -> List[ContentElement]:
        """Consolidate small text elements to create richer, less fragmented memories.

        Tables and images are kept separate (they already have context enrichment from hierarchical chunker).
        Text elements are consolidated regardless of their position in the document.
        
        Key improvement: Group by type FIRST to avoid tables/images breaking consolidation batches.
        """

        # Separate elements by type to enable better consolidation
        # Tables/images already have 400 chars context enrichment from hierarchical chunker
        text_elements = [e for e in content_elements if e.content_type.value == "text"]
        table_elements = [e for e in content_elements if e.content_type.value == "table"]
        image_elements = [e for e in content_elements if e.content_type.value == "image"]
        chart_elements = [e for e in content_elements if e.content_type.value == "chart"]
        
        logger.info(f"Grouping elements by type: {len(text_elements)} text, {len(table_elements)} tables, {len(image_elements)} images, {len(chart_elements)} charts")

        # Consolidate text elements (no interruptions from tables/images!)
        consolidated_text = []
        current_batch = []
        current_chars = 0

        for element in text_elements:
            element_size = len(element.content)

            # If single element is already large (>4500 chars ~1.5 pages), keep it separate
            if element_size > 4500 and not current_batch:
                consolidated_text.append(element)
                continue
            
            # If adding this element would exceed limit, flush current batch
            if current_chars + element_size > max_combined_chars and current_batch:
                consolidated_text.append(self._merge_text_elements(current_batch))
                current_batch = []
                current_chars = 0
            
            # Add element to current batch
            current_batch.append(element)
            current_chars += element_size
        
        # Flush remaining text batch
        if current_batch:
            consolidated_text.append(self._merge_text_elements(current_batch))
        
        # Combine: consolidated text + all tables/images/charts (which already have context)
        # Order: text first (most common), then tables, images, charts
        consolidated = consolidated_text + table_elements + image_elements + chart_elements
        
        reduction_pct = ((len(content_elements) - len(consolidated)) / len(content_elements) * 100) if content_elements else 0
        logger.info(f"Consolidated {len(content_elements)} elements into {len(consolidated)} memories (reduction: {reduction_pct:.1f}%)")
        logger.info(f"  - Text: {len(text_elements)} → {len(consolidated_text)} memories ({len(text_elements) - len(consolidated_text)} merged)")
        logger.info(f"  - Tables: {len(table_elements)} memories (preserved with context)")
        logger.info(f"  - Images: {len(image_elements)} memories (preserved with context)")
        logger.info(f"  - Charts: {len(chart_elements)} memories (preserved with context)")
        
        return consolidated
    
    def _merge_text_elements(self, elements: List[ContentElement]) -> ContentElement:
        """Merge multiple text elements into a single consolidated element"""
        
        if len(elements) == 1:
            return elements[0]
        
        # Combine content with clear separators
        combined_content = "\n\n".join(e.content for e in elements)
        
        # Merge metadata
        combined_metadata = elements[0].metadata.copy() if elements[0].metadata else {}
        combined_metadata["consolidated_from"] = [e.element_id for e in elements]
        combined_metadata["consolidation_count"] = len(elements)
        
        # Create merged element
        return ContentElement(
            element_id=f"consolidated_{elements[0].element_id}",
            content_type=elements[0].content_type,
            content=combined_content,
            metadata=combined_metadata,
            page_number=elements[0].page_number,
            position=elements[0].position
        )

    async def generate_batch_memory_structures(
        self,
        content_elements: List[ContentElement],
        domain_context: Optional[DomainContext] = None,
        base_metadata: Optional[MemoryMetadata] = None,
        batch_size: Optional[int] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[AddMemoryRequest]:
        """
        Generate memory structures for a batch of content elements.

        Args:
            content_elements: List of ContentElement objects to process
            domain_context: Optional domain-specific context
            base_metadata: Base metadata to apply to all memories
            batch_size: Optional batch size (auto-determined if None)
                - Large docs (>500 elements): 15
                - Documents with tables: 5 (tables use more tokens)
                - Default: 10
            document_metadata: Document-level context for contextual retrieval (2025 research)
        """

        # STEP 1: Consolidate small elements to reduce fragmentation
        consolidated_elements = self._consolidate_small_elements(content_elements)
        logger.info(f"After consolidation: {len(consolidated_elements)} elements (original: {len(content_elements)})")

        # Adaptive batch sizing based on document characteristics
        if batch_size is None:
            total_elements = len(consolidated_elements)
            has_tables = any(isinstance(e, TableElement) for e in consolidated_elements[:20])  # Check first 20

            if total_elements > 500:
                batch_size = 15  # Large documents - maximize throughput
                logger.info(f"Using batch_size=15 for large document ({total_elements} elements)")
            elif has_tables:
                batch_size = 5   # Tables use significantly more tokens
                logger.info(f"Using batch_size=5 due to table content")
            else:
                batch_size = 10  # Optimal default for most documents
                logger.info(f"Using batch_size=10 (default)")

        logger.info(f"Generating LLM memory structures for {len(consolidated_elements)} elements in batches of {batch_size}")
        if document_metadata:
            logger.info(f"Using document context: {document_metadata.get('title', 'Unknown')}")

        memory_requests = []

        # Process in batches to avoid overwhelming the LLM service
        for i in range(0, len(consolidated_elements), batch_size):
            batch = consolidated_elements[i:i + batch_size]

            # Process batch concurrently with document context
            batch_tasks = [
                self.generate_memory_structure_for_content(element, domain_context, base_metadata, document_metadata)
                for element in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                else:
                    memory_requests.append(result)

            # Brief pause between batches to respect rate limits
            if i + batch_size < len(consolidated_elements):
                await asyncio.sleep(1)

        logger.info(f"Generated {len(memory_requests)} LLM-optimized memory structures")
        return memory_requests


class DomainSpecificGenerator:
    """Domain-specific memory structure generators"""

    @staticmethod
    async def generate_financial_memories(
        content_elements: List[ContentElement],
        base_metadata: Optional[MemoryMetadata] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[AddMemoryRequest]:
        """Generate memory structures optimized for financial domain"""

        domain_context = DomainContext(
            domain="financial",
            subdomain="reporting",
            entity_types=["revenue", "expenses", "profit", "growth", "metrics"],
            terminology={"Q1": "First Quarter", "YoY": "Year over Year", "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization"}
        )

        generator = LLMMemoryStructureGenerator()
        return await generator.generate_batch_memory_structures(
            content_elements, domain_context, base_metadata, None, document_metadata
        )

    @staticmethod
    async def generate_healthcare_memories(
        content_elements: List[ContentElement],
        base_metadata: Optional[MemoryMetadata] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[AddMemoryRequest]:
        """Generate memory structures optimized for healthcare domain"""

        domain_context = DomainContext(
            domain="healthcare",
            subdomain="clinical",
            entity_types=["patient", "diagnosis", "treatment", "medication", "lab_results"],
            compliance_requirements=["HIPAA", "medical_records", "patient_privacy"]
        )

        generator = LLMMemoryStructureGenerator()
        return await generator.generate_batch_memory_structures(
            content_elements, domain_context, base_metadata, None, document_metadata
        )


# Factory function for easy usage
async def generate_optimized_memory_structures(
    content_elements: List[ContentElement],
    domain: Optional[str] = None,
    base_metadata: Optional[MemoryMetadata] = None,
    document_metadata: Optional[Dict[str, Any]] = None
) -> List[AddMemoryRequest]:
    """
    Factory function to generate optimized memory structures.

    Args:
        content_elements: List of content elements to process
        domain: Optional domain for domain-specific processing
        base_metadata: Base metadata to apply
        document_metadata: Document-level context (title, type, pages, etc.)
                          for contextual retrieval (2025 research)
    """

    if domain == "financial":
        return await DomainSpecificGenerator.generate_financial_memories(
            content_elements, base_metadata, document_metadata
        )
    elif domain in ["healthcare", "medical"]:
        return await DomainSpecificGenerator.generate_healthcare_memories(
            content_elements, base_metadata, document_metadata
        )
    else:
        # Use general LLM generator
        generator = LLMMemoryStructureGenerator()
        return await generator.generate_batch_memory_structures(
            content_elements, None, base_metadata, None, document_metadata
        )