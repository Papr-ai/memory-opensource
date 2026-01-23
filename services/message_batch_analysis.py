"""
Batch message analysis service for processing multiple messages efficiently
"""
import json
import groq
import instructor
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from models.message_models import MessageAnalysisResult
from models.memory_models import AddMemoryRequest, GraphGeneration, GraphGenerationMode, AutoGraphGeneration
from models.shared_types import MessageRole, UserMemoryCategory, AssistantMemoryCategory, MemoryMetadata, PropertyOverrideRule
from services.logger_singleton import LoggerSingleton
from services.message_service import update_message_processing_status
from memory.memory_graph import MemoryGraph, AsyncSession
from fastapi import BackgroundTasks
from models.parse_server import AddMemoryItem
from services.memory_service import handle_incoming_memory
import os

logger = LoggerSingleton.get_logger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class ConversationSummaries(BaseModel):
    """Hierarchical conversation summaries"""
    short_term: str = Field(
        description="Concise summary of the last 15 messages (current batch)"
    )
    medium_term: str = Field(
        description="Summary of the last ~100 messages, synthesized from previous medium-term + current short-term"
    )
    long_term: str = Field(
        description="Full session summary capturing main themes, outcomes, and progress"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Key topics discussed in the conversation"
    )

async def add_message_to_memory_task(
    memory_request: AddMemoryRequest,
    user_id: str,
    session_token: str,
    neo_session: Optional[AsyncSession],
    memory_graph: Optional[MemoryGraph],
    background_tasks: Optional[BackgroundTasks],
    client_type: str = 'message_processing',
    user_workspace_ids: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    legacy_route: bool = True,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
) -> List[AddMemoryItem]:
    """Add a message to memory using the same path as document processing"""
    try:
        # Create instances if not provided (same as document processing pattern)
        if memory_graph is None:
            memory_graph = MemoryGraph()
        if background_tasks is None:
            background_tasks = BackgroundTasks()
        
        # Ensure async connection
        await memory_graph.ensure_async_connection()
        
        # Use handle_incoming_memory to ensure consistent ACL and metadata handling
        # (exact same pattern as add_page_to_memory_task)
        if neo_session is None:
            async with memory_graph.async_neo_conn.get_session() as session:
                response = await handle_incoming_memory(
                    memory_request=memory_request,
                    end_user_id=user_id,
                    developer_user_id=user_id,
                    sessionToken=session_token,
                    neo_session=session,
                    user_info=None,  
                    client_type=client_type,
                    memory_graph=memory_graph,
                    background_tasks=background_tasks,
                    skip_background_processing=False,  # Same as document processing
                    user_workspace_ids=user_workspace_ids,
                    api_key=api_key,
                    legacy_route=legacy_route,
                    workspace_id=workspace_id,
                    api_key_id=api_key_id
                )
        else:
            response = await handle_incoming_memory(
                memory_request=memory_request,
                end_user_id=user_id,
                developer_user_id=user_id,
                sessionToken=session_token,
                neo_session=neo_session,
                user_info=None,  
                client_type=client_type,
                memory_graph=memory_graph,
                background_tasks=background_tasks,
                skip_background_processing=False,  # Same as document processing
                user_workspace_ids=user_workspace_ids,
                api_key=api_key,
                legacy_route=legacy_route,
                workspace_id=workspace_id,
                api_key_id=api_key_id
            )

        if not response or not response.data:
            raise RuntimeError(f"Failed to add memory item for user {user_id}")

        return response.data

    except Exception as e:
        logger.error(f"Error in add_message_to_memory_task: {str(e)}", exc_info=True)
        raise

# Groq configuration with instructor for structured outputs
groq_client = instructor.from_groq(
    groq.AsyncGroq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON
)
groq_model = os.getenv("GROQ_PATTERN_SELECTOR_MODEL", "openai/gpt-oss-20b")

# Learning detection prompts
USER_PREFERENCE_LEARNING_PROMPT = """
**USER PREFERENCE LEARNING DETECTION:**

Identify learnings about the USER's preferences, behaviors, and patterns:

1. **Behavioral Patterns**: How the user works, communicates, makes decisions
2. **Communication Preferences**: Tone, level of detail, format preferences
3. **Work Patterns**: When they're most productive, how they organize work
4. **Technical Preferences**: Tools, languages, frameworks they prefer
5. **Feedback Patterns**: How they give/receive feedback, what resonates
6. **Success Patterns**: What approaches work best for this user

Extract learning if:
- User explicitly states preferences ("I prefer...", "I like...", "I work best when...")
- User corrects or refines previous statements (reveals preference evolution)
- User shows consistent patterns across multiple interactions
- User provides feedback on what worked/didn't work

For detected user preference learnings, provide:
- user_learning_content: Clear statement of the preference/pattern
- user_learning_type: Category (behavioral, communication, work_pattern, technical, feedback, success)
- user_learning_confidence: 0.0-1.0 confidence score
- user_learning_evidence: Quote or observation supporting the learning
"""

AGENT_PERFORMANCE_LEARNING_PROMPT = """
**AGENT PERFORMANCE LEARNING DETECTION:**

Identify learnings about HOW THE AGENT performs tasks and what approaches work best:

1. **Project Stack Discoveries**: Technologies, frameworks, architecture patterns in use
2. **Execution Efficiency**: Tool call optimizations, better sequences, shortcuts
3. **Planning Strategies**: Effective vs ineffective planning approaches
4. **Project Workflows**: Build processes, deployment patterns, testing strategies
5. **Code Organization**: File structure, naming conventions, module patterns
6. **Error Recovery**: Effective debugging approaches, common pitfalls
7. **Context Optimization**: What information is most useful when
8. **Decision Points**: Key factors that lead to better outcomes

Extract learning if:
- Agent discovers project-specific stack/architecture (FastAPI, React, etc.)
- Agent finds more efficient approach after initial attempts
- Agent identifies workflow/process that works for this project
- Agent learns code organization or file structure patterns
- Agent discovers effective debugging/error-handling approach
- Agent optimizes from many tool calls to fewer tool calls
- Agent refines plan based on execution results

For detected agent performance learnings, provide:
- performance_learning_content: Clear statement of the learning
- performance_learning_type: Category (stack_discovery, execution_efficiency, planning_strategy, workflow, code_organization, error_recovery, context_optimization, decision_point)
- performance_learning_confidence: 0.0-1.0 confidence score
- inefficient_approach: What didn't work well (if applicable)
- efficient_approach: What works better
- performance_context: Specific context where this applies
- performance_scope: Scope level (project, goal, user, global)
"""


class MemoryMetadataSchema(BaseModel):
    """Schema for memory metadata"""
    role: Optional[str] = None
    category: Optional[str] = None
    sourceType: Optional[str] = None
    sessionId: Optional[str] = None
    topics: Optional[List[str]] = None
    hierarchical_structures: Optional[Union[str, List]] = None
    customMetadata: Optional[Dict[str, Any]] = None

class MemoryRequestSchema(BaseModel):
    """Schema for memory creation request"""
    content: str
    type: str = "text"
    metadata: Optional[MemoryMetadataSchema] = None

class MessageAnalysisSchema(BaseModel):
    """Schema for individual message analysis"""
    message_index: int
    is_memory_worthy: bool
    confidence_score: float
    reasoning: str
    memory_request: Optional[MemoryRequestSchema] = None
    
    # User preference learning fields
    has_user_preference_learning: bool = False
    user_learning_content: Optional[str] = None
    user_learning_type: Optional[str] = None  # communication, work_pattern, technical, etc.
    user_learning_confidence: float = 0.0
    user_learning_evidence: Optional[str] = None
    
    # Agent performance learning fields
    has_performance_learning: bool = False
    performance_learning_content: Optional[str] = None
    performance_learning_type: Optional[str] = None  # tool_usage, execution, discovery, etc.
    performance_learning_confidence: float = 0.0
    inefficient_approach: Optional[str] = None
    efficient_approach: Optional[str] = None
    performance_context: Optional[str] = None
    performance_scope: Optional[str] = None  # project, user, general

class BatchMessageAnalysisSchema(BaseModel):
    """Structured output schema for batch message analysis"""
    analyses: List[MessageAnalysisSchema] = Field(
        ...,
        description="List of analysis results for each message"
    )
    summaries: ConversationSummaries = Field(
        description="Hierarchical summaries of the conversation (short/medium/long-term)"
    )


async def get_session_summaries(
    session_id: str,
    user_id: str,
    organization_id: Optional[str],
    namespace_id: Optional[str]
) -> Optional[Dict[str, str]]:
    """
    Fetch existing MessageSession summaries from Neo4j to provide context for new summary generation.
    
    Args:
        session_id: Session identifier
        user_id: User ID for access control
        organization_id: Organization ID for multi-tenant scoping
        namespace_id: Namespace ID for multi-tenant scoping
        
    Returns:
        Dict with medium_term and long_term summaries, or None if not found
    """
    try:
        from datastore.neo4jconnection import Neo4jConnection
        import os
        
        uri = os.getenv('NEO4J_URL', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        pwd = os.getenv('NEO4J_SECRET', os.getenv('NEO4J_PASSWORD', 'password'))
        
        neo4j_conn = Neo4jConnection(uri=uri, user=user, pwd=pwd)
        driver = neo4j_conn.get_driver()
        
        query = """
        MATCH (s:MessageSession {sessionId: $session_id})
        WHERE s.user_id = $user_id
        AND ($org_id IS NULL OR s.organization_id = $org_id)
        RETURN s.medium_term_summary as medium_term,
               s.long_term_summary as long_term,
               s.message_count as message_count
        LIMIT 1
        """
        
        with driver.session() as session:
            result = session.run(query, 
                session_id=session_id,
                user_id=user_id,
                org_id=organization_id
            )
            record = result.single()
            
            if record and (record['medium_term'] or record['long_term']):
                summaries = {
                    'medium_term': record['medium_term'] or '',
                    'long_term': record['long_term'] or '',
                    'message_count': record['message_count'] or 0
                }
                logger.info(f"Found existing summaries for session {session_id}: {summaries['message_count']} messages")
                driver.close()
                return summaries
            
            driver.close()
            logger.info(f"No existing summaries found for session {session_id}")
            return None
            
    except Exception as e:
        logger.warning(f"Could not fetch session summaries: {e}")
        return None


async def get_agent_learning_schema_id(
    user_id: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Optional[str]:
    """
    Look up AgentLearning schema by name or tag.
    
    Args:
        user_id: User ID for schema lookup
        workspace_id: Optional workspace ID
        organization_id: Optional organization ID
        namespace_id: Optional namespace ID
        
    Returns:
        Schema ID if found, None otherwise
    """
    try:
        from services.schema_service import SchemaService
        
        schema_service = SchemaService()
        all_schemas = await schema_service.get_active_schemas(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id
        )
        
        # Look for schema by name or tag
        for schema in all_schemas:
            if schema.name == "AgentLearning" or "agent_learning" in getattr(schema, 'tags', []):
                logger.info(f"Found AgentLearning schema: {schema.id}")
                return schema.id
        
        logger.warning(f"AgentLearning schema not found for user {user_id}, organization {organization_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error looking up AgentLearning schema: {e}", exc_info=True)
        return None


async def analyze_message_batch_for_memory(
    messages: List[Dict[str, Any]],
    session_context: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None
) -> Tuple[List[MessageAnalysisResult], ConversationSummaries]:
    """
    Analyze a batch of messages to determine which should become memories AND generate summaries
    
    Args:
        messages: List of message dictionaries from Parse Server
        session_context: Optional context about the conversation session
        session_id: Session ID to fetch previous summaries
        user_id: User ID for access control
        organization_id: Organization ID for multi-tenant scoping
        namespace_id: Namespace ID for multi-tenant scoping
        
    Returns:
        Tuple of (List of MessageAnalysisResult objects, ConversationSummaries)
    """
    if not messages:
        # Return empty results with empty summaries
        empty_summaries = ConversationSummaries(
            short_term="No messages to summarize",
            medium_term="No messages yet",
            long_term="No messages yet",
            topics=[]
        )
        return [], empty_summaries
    
    try:
        # Fetch previous summaries for context (if session_id provided)
        previous_summaries = None
        if session_id and user_id:
            previous_summaries = await get_session_summaries(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
        
        # Build conversation context for the LLM, including tool call info
        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("message", "")
            conversation_text += f"Message {i+1} ({role}): {content}\n"
            
            # Include tool call information for assistant messages
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                conversation_text += f"  [Tool calls: {len(msg['tool_calls'])} calls]\n"
        
        # Create analysis prompt for batch processing
        system_prompt = f"""You are an AI assistant that analyzes chat conversations to:
1. Identify messages worth storing as long-term memories
2. Detect learning signals (user preferences & agent performance)
3. Generate hierarchical conversation summaries

Analyze each message in the conversation and determine:
1. Whether it contains information worth storing as a long-term memory
2. The appropriate memory category based on the message role
3. Whether it contains USER PREFERENCE LEARNING signals
4. Whether it contains AGENT PERFORMANCE LEARNING signals
5. Confidence in your analysis (0.0 to 1.0)
6. Brief reasoning for your decision

**CRITICAL: Role-based Categories (MUST match exactly):**
- **User messages categories**: preference, task, goal, fact, context
- **Assistant messages categories**: skills, learning, task, goal, fact, context

CATEGORY DEFINITIONS:
User categories:
- preference: Personal preferences, settings, likes/dislikes
- task: Specific tasks, todos, action items
- goal: Objectives, targets, aspirations
- fact: Important factual information to remember
- context: Background information, situational context

Assistant categories:
- skills: Capabilities, techniques, methods demonstrated
- learning: Knowledge, insights, or educational content shared
- task: Tasks or action items for the assistant
- goal: Goals or objectives for the assistant
- fact: Factual information shared by the assistant
- context: Contextual information provided by the assistant

**Memory-worthy criteria:**
- Contains factual information, preferences, or insights
- Represents tasks, goals, or decisions
- Shows learning or skill development
- Has long-term relevance beyond the immediate conversation

**NOT memory-worthy:**
- Greetings, confirmations, or casual chat
- Temporary status updates
- Questions without substantive content
- Purely procedural exchanges

{USER_PREFERENCE_LEARNING_PROMPT}

{AGENT_PERFORMANCE_LEARNING_PROMPT}

**SUMMARY GENERATION:**

Generate hierarchical summaries for the conversation:

1. **short_term**: Concise summary of THIS batch (last 15 messages)
   - Focus on: Key decisions, progress, new information
   - Length: 2-3 sentences
   - Optimize for: Quick context grasp

2. **medium_term**: Summary of last ~100 messages
   - Synthesize: Previous medium-term summary + current short-term
   - Focus on: Recent context, ongoing work, current themes
   - Length: 4-5 sentences
   - Update strategy: Integrate new developments, drop stale details

3. **long_term**: Full session summary
   - Capture: Main arc, overall progress, key outcomes
   - Focus on: Big picture, major themes, achievements
   - Length: 5-7 sentences
   - Update strategy: Only update if significant new themes/outcomes emerge

4. **topics**: Array of key topics discussed (3-7 topics)

Return a JSON object with an "analyses" array and a "summaries" object.
Each analysis should include:
- message_index: The index of the message (0-based)
- is_memory_worthy: boolean
- confidence_score: float (0.0-1.0)
- reasoning: string explanation
- memory_request: AddMemoryRequest object if memory-worthy (null otherwise)

LEARNING FIELDS (include for EACH message, even if no learning detected):
- has_user_preference_learning: boolean
- user_learning_content: string (null if not detected)
- user_learning_type: string (null if not detected)
- user_learning_confidence: float (0.0 if not detected)
- user_learning_evidence: string (null if not detected)
- has_performance_learning: boolean
- performance_learning_content: string (null if not detected)
- performance_learning_type: string (null if not detected)
- performance_learning_confidence: float (0.0 if not detected)
- inefficient_approach: string (null if not applicable)
- efficient_approach: string (null if not applicable)
- performance_context: string (null if not detected)
- performance_scope: string (null if not detected)

For memory_request, MUST include:
- content: The memory content (may be refined from original message)
- type: "text"
- metadata: MUST contain:
  - role: "user" or "assistant" (based on message role)
  - category: appropriate category from the list above for this role
  - sourceType: "chatMessages"
  - sessionId: the chat session ID
  - topics: array of relevant topics
  - hierarchical_structures: navigation hierarchy if relevant
  - customMetadata: analysis confidence and reasoning
"""

        # Build user prompt with previous summaries if available
        previous_context = ""
        if previous_summaries:
            previous_context = f"""
PREVIOUS SUMMARIES (for context):
Long-term (full session): {previous_summaries.get('long_term', 'N/A')}
Medium-term (last ~100 messages): {previous_summaries.get('medium_term', 'N/A')}
Current message count: {previous_summaries.get('message_count', 0)}

"""

        user_prompt = f"""Analyze this conversation for memory-worthy content and generate updated summaries.

{previous_context}CURRENT BATCH ({len(messages)} new messages):
{conversation_text}

{f"Session context: {session_context}" if session_context else ""}

YOUR TASKS:
1. Analyze each message for memories and learnings
2. Generate NEW short-term summary for these {len(messages)} messages
3. UPDATE medium-term summary by synthesizing previous medium + new short
4. UPDATE long-term summary if new significant themes/outcomes emerged (otherwise keep similar)
5. Extract key topics discussed

Return complete analysis with both "analyses" array and "summaries" object."""

        # Call Groq API with instructor for structured outputs
        try:
            batch_schema = await groq_client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_model=BatchMessageAnalysisSchema  # ✅ Instructor handles structured outputs!
            )
            
            logger.info(f"✅ Groq structured batch analysis: {len(batch_schema.analyses)} messages analyzed")
            
        except Exception as e:
            logger.error(f"Failed to call Groq with structured outputs: {e}", exc_info=True)
            # Return empty results with fallback summaries
            fallback_summaries = ConversationSummaries(
                short_term=f"Batch of {len(messages)} messages (analysis failed)",
                medium_term=previous_summaries.get('medium_term', 'N/A') if previous_summaries else 'N/A',
                long_term=previous_summaries.get('long_term', 'N/A') if previous_summaries else 'N/A',
                topics=[]
            )
            return [], fallback_summaries
        
        # Convert to MessageAnalysisResult objects
        results = []
        for analysis in batch_schema.analyses:
            try:
                message_index = analysis.message_index
                if message_index >= len(messages):
                    continue
                    
                message = messages[message_index]
                
                # Extract memory content and category from structured schema
                memory_content = ""
                category = ""
                topics = []
                hierarchical_structures = ""
                
                if analysis.is_memory_worthy and analysis.memory_request:
                    memory_content = analysis.memory_request.content
                    if analysis.memory_request.metadata:
                        category = analysis.memory_request.metadata.category or ""
                        topics = analysis.memory_request.metadata.topics or []
                        hierarchical_structures = analysis.memory_request.metadata.hierarchical_structures or ""
                
                # Create MessageAnalysisResult with learning fields
                result = MessageAnalysisResult(
                    is_memory_worthy=analysis.is_memory_worthy,
                    memory_content=memory_content,
                    category=category,
                    role=MessageRole(message.get("messageRole", "user")),  # Use messageRole from Parse
                    confidence_score=analysis.confidence_score,
                    reasoning=analysis.reasoning,
                    topics=topics,
                    hierarchical_structures=hierarchical_structures,
                    message_id=message.get("objectId", ""),
                    session_id=session_context.split(" ")[-1] if session_context and "Session " in session_context else "",  # Extract from session_context
                    # User preference learning fields
                    has_user_preference_learning=analysis.has_user_preference_learning,
                    user_learning_content=analysis.user_learning_content,
                    user_learning_type=analysis.user_learning_type,
                    user_learning_confidence=analysis.user_learning_confidence,
                    user_learning_evidence=analysis.user_learning_evidence,
                    # Agent performance learning fields
                    has_performance_learning=analysis.has_performance_learning,
                    performance_learning_content=analysis.performance_learning_content,
                    performance_learning_type=analysis.performance_learning_type,
                    performance_learning_confidence=analysis.performance_learning_confidence,
                    inefficient_approach=analysis.inefficient_approach,
                    efficient_approach=analysis.efficient_approach,
                    performance_context=analysis.performance_context,
                    performance_scope=analysis.performance_scope
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing analysis result {message_index}: {e}")
                continue
        
        learning_count = sum(1 for r in results if r.has_user_preference_learning or r.has_performance_learning)
        logger.info(f"Batch analysis complete: {len(results)} messages analyzed, {sum(1 for r in results if r.is_memory_worthy)} memory-worthy, {learning_count} with learning signals")
        logger.info(f"✅ Generated summaries - Topics: {batch_schema.summaries.topics}")
        
        return results, batch_schema.summaries
        
    except Exception as e:
        logger.error(f"Error in batch message analysis: {e}")
        # Return empty results with fallback summaries
        fallback_summaries = ConversationSummaries(
            short_term="Analysis error",
            medium_term="N/A",
            long_term="N/A",
            topics=[]
        )
        return [], fallback_summaries


async def process_batch_analysis_results(
    analysis_results: List[MessageAnalysisResult],
    summaries: ConversationSummaries,
    user_id: str,
    session_token: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    project_id: Optional[str] = None,
    goal_id: Optional[str] = None,
    session_id: Optional[str] = None,
    parent_background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, Any]:
    """
    Process batch analysis results and create memories for worthy messages.
    Also creates learning memories when learning signals are detected.
    Creates MessageSession node with hierarchical summaries.
    
    Args:
        analysis_results: List of analysis results from batch processing
        summaries: Hierarchical conversation summaries
        user_id: User ID
        session_token: Session token for authentication
        workspace_id: Optional workspace ID
        organization_id: Optional organization ID
        namespace_id: Optional namespace ID
        api_key_id: Optional API key ID
        project_id: Optional project ID for performance learning context
        goal_id: Optional goal ID for performance learning context
        session_id: Session ID for MessageSession node
        parent_background_tasks: BackgroundTasks from HTTP request
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_analyzed": len(analysis_results),
        "memories_created": 0,
        "learnings_created": 0,
        "messages_processed": 0,
        "errors": 0
    }
    
    # Cache schema_id for this batch (lazy loading when first learning detected)
    agent_learning_schema_id = None
    
    for result in analysis_results:
        try:
            # Update message status to processing
            if result.message_id:
                await update_message_processing_status(result.message_id, "processing")
            
            # 1. Create regular memory if memory-worthy
            if result.is_memory_worthy and result.memory_content and result.category:
                # Create AddMemoryRequest from analysis
                memory_metadata = MemoryMetadata(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    topics=result.topics,
                    hierarchical_structures=result.hierarchical_structures,
                    conversationId=result.session_id,
                    # Role and category as primary metadata fields
                    role=result.role,
                    category=result.category,
                    customMetadata={
                        "confidence_score": result.confidence_score,
                        "analysis_reasoning": result.reasoning,
                        "source_message_id": result.message_id,
                        "batch_processed": True
                    }
                )
                
                memory_request = AddMemoryRequest(
                    content=result.memory_content,
                    type="text",
                    metadata=memory_metadata
                )
                
                # Create memory using the exact same pattern as document processing
                logger.info(f"Creating memory for message {result.message_id}: {memory_request.content[:100]}...")
                
                try:
                    # Use parent BackgroundTasks for graph generation (auto-executes after HTTP response)
                    memory_items = await add_message_to_memory_task(
                        memory_request=memory_request,
                        user_id=user_id,
                        session_token=session_token,
                        neo_session=None,  # Will be handled internally by handle_incoming_memory
                        memory_graph=None,  # Will be created internally
                        background_tasks=parent_background_tasks,  # ✅ Use parent's BackgroundTasks!
                        client_type="message_processing",
                        user_workspace_ids=None,
                        api_key=None,
                        legacy_route=True,
                        workspace_id=workspace_id,
                        organization_id=organization_id,
                        namespace_id=namespace_id,
                        api_key_id=api_key_id
                    )
                    
                    if memory_items and len(memory_items) > 0:
                        logger.info(f"✅ Successfully created memory {memory_items[0].memoryId} for message {result.message_id}")
                        stats["memories_created"] += 1
                    else:
                        logger.error(f"❌ Failed to create memory for message {result.message_id} - no memory item returned")
                        stats["errors"] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error creating memory for message {result.message_id}: {e}")
                    stats["errors"] += 1
            
            # 2. Create USER PREFERENCE learning if detected
            if result.has_user_preference_learning and result.user_learning_content:
                # Lazy lookup of schema ID (only when first learning detected)
                if agent_learning_schema_id is None:
                    agent_learning_schema_id = await get_agent_learning_schema_id(
                        user_id=user_id,
                        workspace_id=workspace_id,
                        organization_id=organization_id,
                        namespace_id=namespace_id
                    )
                
                if agent_learning_schema_id:
                    logger.info(f"Creating user preference learning for message {result.message_id}")
                    
                    learning_request = AddMemoryRequest(
                        content=result.user_learning_content,
                        type="text",
                        metadata=MemoryMetadata(
                            user_id=user_id,
                            workspace_id=workspace_id,
                            organization_id=organization_id,
                            namespace_id=namespace_id,
                            role=MessageRole.ASSISTANT,
                            category=AssistantMemoryCategory.LEARNING,
                            conversationId=result.session_id,
                            customMetadata={
                                "role": "user_preference",
                                "learning_type": result.user_learning_type,
                                "confidence": result.user_learning_confidence,
                                "evidence": result.user_learning_evidence,
                                "source_message_id": result.message_id
                            }
                        ),
                        graph_generation=GraphGeneration(
                            mode=GraphGenerationMode.AUTO,
                            auto=AutoGraphGeneration(schema_id=agent_learning_schema_id)
                        )
                    )
                    
                    try:
                        # Use parent BackgroundTasks from HTTP request for graph generation
                        # This allows FastAPI to auto-execute tasks after HTTP response
                        memory_items = await add_message_to_memory_task(
                            memory_request=learning_request,
                            user_id=user_id,
                            session_token=session_token,
                            neo_session=None,
                            memory_graph=None,
                            background_tasks=parent_background_tasks,  # ✅ Use parent's BackgroundTasks!
                            client_type="message_processing",
                            user_workspace_ids=None,
                            api_key=None,
                            legacy_route=True,
                            workspace_id=workspace_id,
                            organization_id=organization_id,
                            namespace_id=namespace_id,
                            api_key_id=api_key_id
                        )
                        
                        if memory_items and len(memory_items) > 0:
                            logger.info(f"✅ Successfully created user preference learning {memory_items[0].memoryId}")
                            stats["learnings_created"] += 1
                        else:
                            logger.error(f"❌ Failed to create user preference learning for message {result.message_id}")
                            stats["errors"] += 1
                    except Exception as e:
                        logger.error(f"❌ Error creating user preference learning: {e}")
                        stats["errors"] += 1
                else:
                    logger.warning(f"AgentLearning schema not found, skipping user preference learning for message {result.message_id}")
            
            # 3. Create AGENT PERFORMANCE learning if detected
            if result.has_performance_learning and result.performance_learning_content:
                # Reuse cached schema_id (or look up if not cached yet)
                if agent_learning_schema_id is None:
                    agent_learning_schema_id = await get_agent_learning_schema_id(
                        user_id=user_id,
                        workspace_id=workspace_id,
                        organization_id=organization_id,
                        namespace_id=namespace_id
                    )
                
                if agent_learning_schema_id:
                    logger.info(f"Creating agent performance learning for message {result.message_id}")
                    
                    # Build property_overrides for context injection
                    property_overrides = []
                    if project_id or goal_id:
                        override_set = {}
                        if project_id:
                            override_set["project_id"] = project_id
                        if goal_id:
                            override_set["goal_id"] = goal_id
                        
                        property_overrides.append(
                            PropertyOverrideRule(
                                nodeLabel="Learning",
                                set=override_set
                            )
                        )
                    
                    perf_request = AddMemoryRequest(
                        content=result.performance_learning_content,
                        type="text",
                        metadata=MemoryMetadata(
                            user_id=user_id,
                            workspace_id=workspace_id,
                            organization_id=organization_id,
                            namespace_id=namespace_id,
                            role=MessageRole.ASSISTANT,
                            category=AssistantMemoryCategory.LEARNING,
                            conversationId=result.session_id,
                            customMetadata={
                                "role": "agent_performance",
                                "learning_type": result.performance_learning_type,
                                "confidence": result.performance_learning_confidence,
                                "scope": result.performance_scope,
                                "context": (
                                    f"{result.performance_context or ''}\n"
                                    f"Inefficient: {result.inefficient_approach or 'N/A'}\n"
                                    f"Efficient: {result.efficient_approach or 'N/A'}"
                                ).strip(),
                                "source_message_id": result.message_id,
                                "project_id": project_id,
                                "goal_id": goal_id
                            }
                        ),
                        graph_generation=GraphGeneration(
                            mode=GraphGenerationMode.AUTO,
                            auto=AutoGraphGeneration(
                                schema_id=agent_learning_schema_id,
                                property_overrides=property_overrides if property_overrides else None
                            )
                        )
                    )
                    
                    try:
                        # Use parent BackgroundTasks from HTTP request for graph generation
                        # This allows FastAPI to auto-execute tasks after HTTP response
                        memory_items = await add_message_to_memory_task(
                            memory_request=perf_request,
                            user_id=user_id,
                            session_token=session_token,
                            neo_session=None,
                            memory_graph=None,
                            background_tasks=parent_background_tasks,  # ✅ Use parent's BackgroundTasks!
                            client_type="message_processing",
                            user_workspace_ids=None,
                            api_key=None,
                            legacy_route=True,
                            workspace_id=workspace_id,
                            organization_id=organization_id,
                            namespace_id=namespace_id,
                            api_key_id=api_key_id
                        )
                        
                        if memory_items and len(memory_items) > 0:
                            logger.info(f"✅ Successfully created agent performance learning {memory_items[0].memoryId}")
                            stats["learnings_created"] += 1
                        else:
                            logger.error(f"❌ Failed to create agent performance learning for message {result.message_id}")
                            stats["errors"] += 1
                    except Exception as e:
                        logger.error(f"❌ Error creating agent performance learning: {e}")
                        stats["errors"] += 1
                else:
                    logger.warning(f"AgentLearning schema not found, skipping agent performance learning for message {result.message_id}")
            
            # Update message status to completed
            if result.message_id:
                await update_message_processing_status(result.message_id, "completed")
            
            stats["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing analysis result for message {result.message_id}: {e}")
            stats["errors"] += 1
            
            # Update message status to failed
            if result.message_id:
                await update_message_processing_status(result.message_id, "failed", str(e))
    
    # Create/Update MessageSession node with summaries
    if session_id and summaries:
        try:
            logger.info(f"Creating/updating MessageSession node for session {session_id}")
            
            # Lazy lookup for schema (only once per batch)
            if agent_learning_schema_id is None:
                agent_learning_schema_id = await get_agent_learning_schema_id(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id
                )
            
            if agent_learning_schema_id:
                # Create MessageSession memory request
                session_memory_request = AddMemoryRequest(
                    content=summaries.short_term,  # Main content is short-term summary
                    type="text",
                    metadata=MemoryMetadata(
                        role=MessageRole.ASSISTANT.value,
                        category=AssistantMemoryCategory.CONTEXT.value,
                        sourceType="conversationSummary",
                        sessionId=session_id,
                        topics=summaries.topics,
                        customMetadata={
                            "sessionId": session_id,
                            "short_term_summary": summaries.short_term,
                            "medium_term_summary": summaries.medium_term,
                            "long_term_summary": summaries.long_term,
                            "message_count": len(analysis_results),
                            "topics": ",".join(summaries.topics),
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    ),
                    graph_generation=GraphGeneration(
                        mode=GraphGenerationMode.AUTO,
                        auto_graph_generation=AutoGraphGeneration(
                            schema_id=agent_learning_schema_id,
                            enable_background_processing=True
                        ),
                        property_overrides=[
                            PropertyOverrideRule(
                                node_label="MessageSession",
                                property_name="sessionId",
                                property_value=session_id
                            ),
                            PropertyOverrideRule(
                                node_label="MessageSession",
                                property_name="short_term_summary",
                                property_value=summaries.short_term
                            ),
                            PropertyOverrideRule(
                                node_label="MessageSession",
                                property_name="medium_term_summary",
                                property_value=summaries.medium_term
                            ),
                            PropertyOverrideRule(
                                node_label="MessageSession",
                                property_name="long_term_summary",
                                property_value=summaries.long_term
                            ),
                            PropertyOverrideRule(
                                node_label="MessageSession",
                                property_name="message_count",
                                property_value=str(len(analysis_results))
                            ),
                            PropertyOverrideRule(
                                node_label="MessageSession",
                                property_name="topics",
                                property_value=",".join(summaries.topics)
                            )
                        ]
                    )
                )
                
                # Create MessageSession node
                session_memory_items = await add_message_to_memory_task(
                    memory_request=session_memory_request,
                    user_id=user_id,
                    session_token=session_token,
                    neo_session=None,
                    memory_graph=None,
                    background_tasks=parent_background_tasks,  # ✅ Use parent's BackgroundTasks!
                    client_type="message_processing",
                    user_workspace_ids=None,
                    api_key=None,
                    legacy_route=True,
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    api_key_id=api_key_id
                )
                
                if session_memory_items and len(session_memory_items) > 0:
                    logger.info(f"✅ Successfully created/updated MessageSession node for {session_id}")
                else:
                    logger.error(f"❌ Failed to create MessageSession node for {session_id}")
                    
                # Also store summaries in Parse Server for quick API access
                from services.message_service import update_chat_summaries
                await update_chat_summaries(
                    session_id=session_id,
                    user_id=user_id,
                    summaries={
                        "short_term": summaries.short_term,
                        "medium_term": summaries.medium_term,
                        "long_term": summaries.long_term,
                        "topics": summaries.topics
                    },
                    workspace_id=workspace_id
                )
                logger.info(f"✅ Updated Chat summaries in Parse Server for {session_id}")
            else:
                logger.warning(f"AgentLearning schema not found, skipping MessageSession creation for {session_id}")
                
        except Exception as e:
            logger.error(f"Error creating MessageSession node: {e}", exc_info=True)
    
    logger.info(f"Batch processing complete: {stats}")
    return stats
