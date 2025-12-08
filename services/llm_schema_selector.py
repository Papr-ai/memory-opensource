from typing import Dict, Any, List, Optional, Tuple
from services.schema_service import SchemaService
from services.logging_config import get_logger
from pydantic import BaseModel, Field
import json
from openai import AsyncOpenAI
from os import environ as env

logger = get_logger(__name__)

class SchemaSelectionResponse(BaseModel):
    """Response model for LLM schema selection"""
    selected_schema_name: Optional[str] = Field(
        None, 
        description="Name of the most relevant user-defined schema, or null if system schema should be used"
    )
    selected_schema_index: Optional[int] = Field(
        None,
        description="Index (1-based) of the selected schema from the provided list, or null if system schema should be used"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how certain the selection is"
    )
    reasoning: str = Field(
        description="Brief explanation of why this schema was selected"
    )

class LLMSchemaSelector:
    """Uses LLM to intelligently select the most relevant schema"""
    
    def __init__(self, schema_service: SchemaService):
        self.schema_service = schema_service
        self.client = AsyncOpenAI(api_key=env.get("OPENAI_API_KEY"))
        self.model = env.get("OPENAI_SCHEMA_SELECTOR_MODEL", "gpt-5-mini")
        self.enabled = env.get("ENABLE_LLM_SCHEMA_SELECTION", "true").lower() == "true"
        
    async def select_schema_for_content(
        self, 
        content: str, 
        user_id: str, 
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        operation_type: str = "add_memory",  # "add_memory" or "search_memory"
        organization_id: Optional[str] = None,
        namespace_id: Optional[str] = None
    ) -> Tuple[Optional[str], float, Optional[object]]:
        """
        Use LLM to select the most relevant schema for given content.
        This method fetches schemas from Parse Server.
        
        Returns:
            Tuple of (schema_id, confidence_score, schema_object)
        """
        logger.info(f"🔍 SCHEMA STEP 1: Fetching user schemas for user_id={user_id}, workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}")
        user_schemas = await self.schema_service.get_active_schemas(user_id, workspace_id, organization_id, namespace_id)
        logger.info(f"🔍 SCHEMA STEP 2: Found {len(user_schemas)} user schemas")
        for schema in user_schemas:
            logger.info(f"🔍 SCHEMA: {schema.name} (id={schema.id}) - {len(schema.node_types)} node types, {len(schema.relationship_types)} rel types")
        
        result = await self._select_from_schemas(content, user_schemas, metadata, operation_type)
        logger.info(f"🔍 SCHEMA STEP 3: LLM selected schema_id={result[0]}, confidence={result[1]}")
        return result
    
    async def select_from_existing_schemas(
        self,
        content: str,
        existing_schemas: List,
        metadata: Optional[Dict[str, Any]] = None,
        operation_type: str = "add_memory"
    ) -> Tuple[Optional[str], float, Optional[object]]:
        """
        Use LLM to select the most relevant schema from pre-fetched schemas.
        This method is optimized for search operations where schemas are already available.
        
        Args:
            content: The content/query to analyze
            existing_schemas: Pre-fetched list of UserGraphSchema objects
            metadata: Optional metadata for context
            operation_type: "add_memory" or "search_memory"
        
        Returns:
            Tuple of (schema_id, confidence_score, schema_object)
        """
        return await self._select_from_schemas(content, existing_schemas, metadata, operation_type)
    
    async def _select_from_schemas(
        self,
        content: str,
        user_schemas: List,
        metadata: Optional[Dict[str, Any]] = None,
        operation_type: str = "add_memory"
    ) -> Tuple[Optional[str], float, Optional[object]]:
        """
        Common method to select schema from a list of schemas using LLM.
        
        Returns:
            Tuple of (schema_id, confidence_score, schema_object)
        """
        try:
            # Check if LLM selection is enabled
            if not self.enabled:
                logger.info("LLM schema selection disabled, falling back to algorithmic selection")
                return None, 0.5, None
            
            if not user_schemas:
                logger.info("No user schemas available, using system schema")
                return None, 1.0, None
            
            # OPTIMIZATION: If only one schema, auto-select it without LLM call
            if len(user_schemas) == 1:
                selected_schema = user_schemas[0]
                logger.info(f"✅ AUTO-SELECTED: Only one schema available - {selected_schema.name} (id={selected_schema.id})")
                logger.info(f"  Node types: {list(selected_schema.node_types.keys())}")
                logger.info(f"  Relationship types: {list(selected_schema.relationship_types.keys())}")
                return selected_schema.id, 1.0, selected_schema
            
            # Prepare schema information for LLM
            schema_descriptions = []
            for i, schema in enumerate(user_schemas, 1):
                node_types = list(schema.node_types.keys())
                relationship_types = list(schema.relationship_types.keys())
                
                schema_info = {
                    "index": i,
                    "name": schema.name,
                    "description": schema.description or "No description provided",
                    "node_types": node_types,
                    "relationship_types": relationship_types,
                    "example_use_cases": self._generate_use_case_examples(schema)
                }
                schema_descriptions.append(schema_info)
            
            # Create LLM prompt
            system_prompt = self._create_selection_prompt(operation_type)
            user_prompt = self._create_user_prompt(content, metadata, schema_descriptions, operation_type)
            
            # Call LLM for schema selection
            logger.info(f"Using {self.model} for schema selection in {operation_type}")
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=SchemaSelectionResponse
                # Note: temperature parameter not supported by gpt-5-mini
            )
            
            result = completion.choices[0].message.parsed
            logger.info(f"LLM schema selection: {result.selected_schema_name} (confidence: {result.confidence})")
            logger.info(f"LLM reasoning: {result.reasoning}")
            
            # Find selected schema
            selected_schema_id = None
            selected_schema_object = None
            if result.selected_schema_index and 1 <= result.selected_schema_index <= len(user_schemas):
                selected_schema_object = user_schemas[result.selected_schema_index - 1]
                selected_schema_id = selected_schema_object.id
                logger.info(f"Selected schema: {selected_schema_object.name} (ID: {selected_schema_id})")
            elif result.selected_schema_name:
                # Fallback: find by name
                for schema in user_schemas:
                    if schema.name.lower() == result.selected_schema_name.lower():
                        selected_schema_id = schema.id
                        selected_schema_object = schema
                        break
            
            return selected_schema_id, result.confidence, selected_schema_object
            
        except Exception as e:
            logger.error(f"Error in LLM schema selection: {e}")
            # Fallback to first active schema
            if user_schemas:
                return user_schemas[0].id, 0.5, user_schemas[0]
            return None, 1.0, None
    
    def _create_selection_prompt(self, operation_type: str) -> str:
        """Create system prompt for schema selection"""
        operation_desc = {
            "add_memory": "adding a new memory item to the knowledge graph",
            "search_memory": "searching for existing memories in the knowledge graph"
        }.get(operation_type, "processing memory content")
        
        return f"""You are an expert at selecting the most appropriate knowledge graph schema for {operation_desc}.

Your task is to analyze the provided content/query and metadata, then select the most relevant user-defined schema from the available options.

SELECTION CRITERIA:
1. **Domain Match**: Which schema's domain (e-commerce, CRM, HR, etc.) best matches the content?
2. **Node Type Relevance**: Which schema has node types that are mentioned or implied in the content?
3. **Relationship Relevance**: Which schema has relationships that match the actions/connections in the content?
4. **Context Fit**: Consider the metadata and overall context to determine the most appropriate domain

GUIDELINES:
- If the content clearly fits one schema's domain, select that schema with high confidence
- If content mentions specific entities that match a schema's node types, prefer that schema
- If content describes actions that match a schema's relationships, consider that schema
- If no user schema is clearly relevant, return null to use the system schema
- If multiple schemas could work, pick the one with the strongest match

CONFIDENCE SCORING:
- 0.9-1.0: Perfect match, content clearly belongs to this schema's domain
- 0.7-0.8: Good match, most entities/concepts align with this schema
- 0.5-0.6: Moderate match, some alignment but could work with other schemas too
- 0.3-0.4: Weak match, minimal alignment
- 0.0-0.2: Poor match, content doesn't fit this schema well

Always provide clear reasoning for your selection."""

    def _create_user_prompt(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]], 
        schema_descriptions: List[Dict], 
        operation_type: str
    ) -> str:
        """Create user prompt with content and schema options"""
        
        operation_context = {
            "add_memory": "I want to add this memory content to my knowledge graph:",
            "search_memory": "I want to search my knowledge graph for:"
        }.get(operation_type, "Content to analyze:")
        
        prompt = f"""{operation_context}

CONTENT/QUERY:
{content}

METADATA:
{json.dumps(metadata, indent=2) if metadata else "None provided"}

AVAILABLE USER SCHEMAS:
"""
        
        for schema_info in schema_descriptions:
            prompt += f"""
{schema_info['index']}. **{schema_info['name']}**
   Description: {schema_info['description']}
   Node Types: {', '.join(schema_info['node_types'])}
   Relationships: {', '.join(schema_info['relationship_types'])}
   Use Cases: {schema_info['example_use_cases']}
"""
        
        prompt += """
TASK:
Analyze the content/query and select the most appropriate schema. Consider:
- What entities (people, objects, concepts) are mentioned?
- What actions or relationships are described?
- What domain/context does this content belong to?

Return the schema index (1-based) that best matches, or null if the system schema is most appropriate."""
        
        return prompt
    
    def _generate_use_case_examples(self, schema) -> str:
        """Generate example use cases for a schema"""
        examples = []
        
        # Generate examples based on node types
        node_types = list(schema.node_types.keys())
        if len(node_types) >= 2:
            examples.append(f"Managing {node_types[0]} and {node_types[1]} data")
        
        # Generate examples based on relationships
        rel_types = list(schema.relationship_types.keys())
        if rel_types:
            rel_example = rel_types[0].lower().replace('_', ' ')
            examples.append(f"Tracking {rel_example} relationships")
        
        # Domain-specific examples
        domain_keywords = {
            'product': 'E-commerce, inventory, sales',
            'customer': 'Customer management, CRM, support',
            'employee': 'HR, team management, org structure',
            'project': 'Project management, task tracking',
            'lead': 'Sales pipeline, lead management',
            'campaign': 'Marketing, advertising, outreach'
        }
        
        for node_type in node_types:
            for keyword, domain in domain_keywords.items():
                if keyword in node_type.lower():
                    examples.append(domain)
                    break
        
        return '; '.join(examples) if examples else "General purpose schema"

# Dependency injection
def get_llm_schema_selector() -> LLMSchemaSelector:
    from services.schema_service import get_schema_service
    return LLMSchemaSelector(get_schema_service())