"""
AI analysis service for chat messages
Determines if messages should become memories and extracts structured memory data
"""
import json
import groq
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from models.message_models import MessageAnalysisResult, MessageRequest
from models.memory_models import AddMemoryRequest
from models.shared_types import MessageRole, UserMemoryCategory, AssistantMemoryCategory, MemoryMetadata
from services.logger_singleton import LoggerSingleton
import os

logger = LoggerSingleton.get_logger(__name__)

# Groq configuration (optional - only if GROQ_API_KEY is set)
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = groq.AsyncGroq(api_key=groq_api_key)
else:
    groq_client = None
    logger.debug("GROQ_API_KEY not set - Groq features will be disabled")
groq_model = os.getenv("GROQ_PATTERN_SELECTOR_MODEL", "openai/gpt-oss-20b")


class MessageAnalysisSchema(BaseModel):
    """Structured output schema for message analysis - wraps AddMemoryRequest"""
    is_memory_worthy: bool = Field(
        ..., 
        description="Whether this message contains information worth storing as a long-term memory"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis from 0.0 to 1.0"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this message is or isn't memory-worthy and the chosen category"
    )
    memory_request: Optional[AddMemoryRequest] = Field(
        None,
        description="If memory-worthy, the complete AddMemoryRequest with all fields including customMetadata"
    )


async def analyze_message_for_memory(
    message_request: MessageRequest,
    session_context: Optional[str] = None
) -> MessageAnalysisResult:
    """
    Analyze a chat message to determine if it should become a memory using Groq
    
    Args:
        message_request: The message to analyze
        session_context: Optional context from the conversation session
        
    Returns:
        MessageAnalysisResult with analysis details
    """
    try:
        # Build the analysis prompt
        role_categories = _get_role_categories(message_request.role)
        
        system_prompt = f"""You are an AI assistant that analyzes chat messages to determine if they contain information worth storing as long-term memories.

ROLE CONTEXT:
- Message role: {message_request.role.value}
- Available categories for this role: {', '.join(role_categories)}

MEMORY-WORTHY CRITERIA:
For USER messages, consider memory-worthy if it contains:
- Personal preferences or settings
- Tasks to be done or completed
- Goals or objectives
- Important facts or information to remember
- Contextual information that might be useful later

For ASSISTANT messages, consider memory-worthy if it contains:
- Skills or capabilities demonstrated
- Learning or knowledge shared
- Techniques or methods explained

NOT memory-worthy:
- Simple greetings or pleasantries
- Purely conversational filler
- Temporary/transient information
- Questions without substantial context

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

IMPORTANT: If the message IS memory-worthy, you MUST provide:
- content: Refined/extracted memory content
- role: MUST be set to "{message_request.role.value}"
- category: MUST be one of the valid categories for this role: {', '.join(role_categories)}
- metadata: MUST include:
  - role: The message role ("{message_request.role.value}")
  - category: The appropriate category
  - sourceType: "chatMessages"
  - sessionId: The chat session ID
  - topics: Array of relevant topics
  - hierarchical_structures: Navigation hierarchy
  - customMetadata: Analysis confidence and reasoning

Analyze the message and provide structured output."""

        user_prompt = f"""Message to analyze:
Role: {message_request.role.value}
Content: "{message_request.content}"
Session ID: {message_request.sessionId}

{f"Session context: {session_context}" if session_context else ""}

Analyze this message and determine if it's memory-worthy. If yes, create a complete AddMemoryRequest."""

        # Use Groq with JSON mode for structured output
        enhanced_prompt = f"""{system_prompt}

IMPORTANT: Respond with a valid JSON object that matches this exact schema:
{{
    "is_memory_worthy": boolean,
    "confidence_score": number (0.0 to 1.0),
    "reasoning": "string explaining the decision",
    "memory_request": {{
        "content": "refined memory content if memory-worthy, otherwise null",
        "type": "text",
        "role": "{message_request.role.value}",
        "category": "appropriate category or null",
        "metadata": {{
            "topics": ["array", "of", "topics"],
            "hierarchical_structures": "Category/Subcategory/Specific",
            "sourceType": "chat_message",
            "conversationId": "{message_request.sessionId}",
            "customMetadata": {{
                "analysis_confidence": confidence_score,
                "analysis_reasoning": "reasoning text",
                "original_role": "{message_request.role.value}"
            }}
        }}
    }}
}}

If the message is NOT memory-worthy, set memory_request to null."""

        if not groq_client:
            logger.warning("Groq client not initialized - GROQ_API_KEY not set. Skipping message analysis.")
            # Return a default result indicating no memory should be created
            return MessageAnalysisResult(
                is_memory_worthy=False,
                confidence=0.0,
                reasoning="Groq API key not configured - cannot analyze message"
            )
        
        response = await groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        response_content = response.choices[0].message.content
        try:
            analysis_data = json.loads(response_content)
            analysis = MessageAnalysisSchema(**analysis_data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq JSON response: {response_content}")
            raise Exception(f"Invalid JSON response from Groq: {e}")
        except Exception as e:
            logger.error(f"Failed to create MessageAnalysisSchema from response: {analysis_data}")
            raise Exception(f"Schema validation error: {e}")
        
        # Extract data for MessageAnalysisResult
        if analysis.is_memory_worthy and analysis.memory_request:
            memory_content = analysis.memory_request.content
            category = analysis.memory_request.category
            topics = analysis.memory_request.metadata.topics if analysis.memory_request.metadata else None
            hierarchical_structures = analysis.memory_request.metadata.hierarchical_structures if analysis.memory_request.metadata else None
        else:
            memory_content = None
            category = None
            topics = None
            hierarchical_structures = None
        
        # Validate category for role
        if category and category not in role_categories:
            logger.warning(f"Invalid category '{category}' for role '{message_request.role}'. Setting to None.")
            category = None
        
        logger.info(f"Message analysis complete: memory_worthy={analysis.is_memory_worthy}, category={category}, confidence={analysis.confidence_score}")
        
        return MessageAnalysisResult(
            is_memory_worthy=analysis.is_memory_worthy,
            memory_content=memory_content,
            category=category,
            role=message_request.role,
            confidence_score=analysis.confidence_score,
            reasoning=analysis.reasoning,
            topics=topics,
            hierarchical_structures=hierarchical_structures
        )
        
    except Exception as e:
        logger.error(f"Error analyzing message: {str(e)}")
        # Return a safe default
        return MessageAnalysisResult(
            is_memory_worthy=False,
            memory_content=None,
            category=None,
            role=message_request.role,
            confidence_score=0.0,
            reasoning=f"Analysis failed: {str(e)}",
            topics=None,
            hierarchical_structures=None
        )


async def create_memory_request_from_analysis(
    message_request: MessageRequest,
    analysis: MessageAnalysisResult,
    message_id: str
) -> Optional[AddMemoryRequest]:
    """
    Create an AddMemoryRequest from message analysis results

    Args:
        message_request: Original message request
        analysis: Analysis results
        message_id: PostMessage objectId for linking

    Returns:
        AddMemoryRequest if memory should be created, None otherwise
    """
    if not analysis.is_memory_worthy or not analysis.memory_content:
        return None

    try:
        # Import ContextItem here to avoid circular imports
        from models.memory_models import ContextItem

        # Build metadata
        metadata = message_request.metadata or MemoryMetadata()

        # Set role and category in metadata (primary metadata fields)
        metadata.role = analysis.role
        metadata.category = analysis.category

        # Update metadata with analysis results
        if analysis.topics:
            metadata.topics = analysis.topics

        if analysis.hierarchical_structures:
            metadata.hierarchical_structures = analysis.hierarchical_structures

        # Add source information - set sourceType to "chatMessages" and sessionId
        metadata.sourceType = "chatMessages"
        metadata.sessionId = message_request.sessionId  # Chat session ID
        metadata.conversationId = message_request.sessionId  # Also set conversationId for compatibility

        # Store the PostMessage link in customMetadata
        if not metadata.customMetadata:
            metadata.customMetadata = {}

        metadata.customMetadata.update({
            "analysis_confidence": analysis.confidence_score,
            "analysis_reasoning": analysis.reasoning,
            "original_role": analysis.role.value,
            "message_id": message_id,  # PostMessage objectId
            "chat_session_id": message_request.sessionId
        })

        # Build context array with current message
        # TODO: In the future, we should fetch previous messages from the session for richer context
        context = [
            ContextItem(
                content=message_request.content if isinstance(message_request.content, str) else str(message_request.content),
                role=message_request.role.value
            )
        ]

        return AddMemoryRequest(
            content=analysis.memory_content,
            type="text",
            metadata=metadata,
            context=context
        )

    except Exception as e:
        logger.error(f"Error creating memory request from analysis: {str(e)}")
        return None


def _get_role_categories(role: MessageRole) -> list[str]:
    """Get valid categories for a given role"""
    if role == MessageRole.USER:
        return [cat.value for cat in UserMemoryCategory]
    elif role == MessageRole.ASSISTANT:
        return [cat.value for cat in AssistantMemoryCategory]
    else:
        return []


async def get_session_context(session_id: str, limit: int = 5) -> Optional[str]:
    """
    Get recent context from a conversation session for better analysis
    
    Args:
        session_id: Session ID to get context for
        limit: Number of recent messages to include
        
    Returns:
        String summary of recent conversation context
    """
    try:
        # This would integrate with the message service to get recent messages
        # For now, return None - can be implemented later for enhanced context
        return None
        
    except Exception as e:
        logger.error(f"Error getting session context: {str(e)}")
        return None
