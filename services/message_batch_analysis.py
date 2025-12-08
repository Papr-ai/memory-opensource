"""
Batch message analysis service for processing multiple messages efficiently
"""
import json
import groq
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from models.message_models import MessageAnalysisResult
from models.memory_models import AddMemoryRequest
from models.shared_types import MessageRole, UserMemoryCategory, AssistantMemoryCategory, MemoryMetadata
from services.logger_singleton import LoggerSingleton
from services.message_service import update_message_processing_status
from memory.memory_graph import MemoryGraph, AsyncSession
from fastapi import BackgroundTasks
from models.parse_server import AddMemoryItem
from services.memory_service import handle_incoming_memory
import os

logger = LoggerSingleton.get_logger(__name__)

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

# Groq configuration
groq_client = groq.AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
groq_model = os.getenv("GROQ_PATTERN_SELECTOR_MODEL", "openai/gpt-oss-20b")


class BatchMessageAnalysisSchema(BaseModel):
    """Structured output schema for batch message analysis"""
    analyses: List[Dict[str, Any]] = Field(
        ...,
        description="List of analysis results for each message, with message_index and analysis data"
    )


async def analyze_message_batch_for_memory(
    messages: List[Dict[str, Any]],
    session_context: Optional[str] = None
) -> List[MessageAnalysisResult]:
    """
    Analyze a batch of messages to determine which should become memories
    
    Args:
        messages: List of message dictionaries from Parse Server
        session_context: Optional context about the conversation session
        
    Returns:
        List of MessageAnalysisResult objects
    """
    if not messages:
        return []
    
    try:
        # Build conversation context for the LLM
        conversation_text = ""
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("message", "")
            conversation_text += f"Message {i+1} ({role}): {content}\n"
        
        # Create analysis prompt for batch processing
        system_prompt = f"""You are an AI assistant that analyzes chat conversations to identify messages worth storing as long-term memories.

Analyze each message in the conversation and determine:
1. Whether it contains information worth storing as a long-term memory
2. The appropriate memory category based on the message role
3. Confidence in your analysis (0.0 to 1.0)
4. Brief reasoning for your decision

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

Return a JSON object with an "analyses" array containing analysis results for each message.
Each analysis should include:
- message_index: The index of the message (0-based)
- is_memory_worthy: boolean
- confidence_score: float (0.0-1.0)
- reasoning: string explanation
- memory_request: AddMemoryRequest object if memory-worthy (null otherwise)

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

        user_prompt = f"""Analyze this conversation for memory-worthy content:

{conversation_text}

{f"Session context: {session_context}" if session_context else ""}

Return analysis results for each message in JSON format."""

        # Call Groq API
        response = await groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        logger.info(f"Groq batch analysis response: {response_text}")
        
        try:
            analysis_data = json.loads(response_text)
            batch_schema = BatchMessageAnalysisSchema(**analysis_data)
        except Exception as e:
            logger.error(f"Failed to parse Groq batch response: {e}")
            return []
        
        # Convert to MessageAnalysisResult objects
        results = []
        for analysis in batch_schema.analyses:
            try:
                message_index = analysis.get("message_index", 0)
                if message_index >= len(messages):
                    continue
                    
                message = messages[message_index]
                
                # Create MessageAnalysisResult
                result = MessageAnalysisResult(
                    is_memory_worthy=analysis.get("is_memory_worthy", False),
                    memory_content=analysis.get("memory_request", {}).get("content", "") if analysis.get("is_memory_worthy") else "",
                    category=analysis.get("memory_request", {}).get("category", "") if analysis.get("is_memory_worthy") else "",
                    role=MessageRole(message.get("messageRole", "user")),  # Use messageRole from Parse
                    confidence_score=analysis.get("confidence_score", 0.0),
                    reasoning=analysis.get("reasoning", ""),
                    topics=analysis.get("memory_request", {}).get("metadata", {}).get("topics", []) if analysis.get("is_memory_worthy") else [],
                    hierarchical_structures=analysis.get("memory_request", {}).get("metadata", {}).get("hierarchical_structures", "") if analysis.get("is_memory_worthy") else "",
                    message_id=message.get("objectId", ""),
                    session_id=session_context.split(" ")[-1] if session_context and "Session " in session_context else ""  # Extract from session_context
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing analysis result {message_index}: {e}")
                continue
        
        logger.info(f"Batch analysis complete: {len(results)} messages analyzed, {sum(1 for r in results if r.is_memory_worthy)} memory-worthy")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch message analysis: {e}")
        return []


async def process_batch_analysis_results(
    analysis_results: List[MessageAnalysisResult],
    user_id: str,
    session_token: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process batch analysis results and create memories for worthy messages
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_analyzed": len(analysis_results),
        "memories_created": 0,
        "messages_processed": 0,
        "errors": 0
    }
    
    for result in analysis_results:
        try:
            # Update message status to processing
            if result.message_id:
                await update_message_processing_status(result.message_id, "processing")
            
            if result.is_memory_worthy and result.memory_content:
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
                    # Create a dedicated function similar to add_page_to_memory_task
                    memory_items = await add_message_to_memory_task(
                        memory_request=memory_request,
                        user_id=user_id,
                        session_token=session_token,
                        neo_session=None,  # Will be handled internally by handle_incoming_memory
                        memory_graph=None,  # Will be created internally
                        background_tasks=None,  # Will be created internally
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
                
                # Update message status to completed
                if result.message_id:
                    await update_message_processing_status(result.message_id, "completed")
            else:
                # Mark as processed but no memory created
                if result.message_id:
                    await update_message_processing_status(result.message_id, "completed")
            
            stats["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing analysis result for message {result.message_id}: {e}")
            stats["errors"] += 1
            
            # Update message status to failed
            if result.message_id:
                await update_message_processing_status(result.message_id, "failed", str(e))
    
    logger.info(f"Batch processing complete: {stats}")
    return stats
