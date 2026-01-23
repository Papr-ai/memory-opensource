"""
Message processing pipeline structured for easy Temporal migration
Handles the background processing of chat messages into memories
"""
import asyncio
from typing import Optional, Dict, Any
from fastapi import BackgroundTasks

from models.message_models import MessageRequest, MessageResponse
from services.message_service import (
    update_message_processing_status, 
    should_trigger_analysis, 
    get_unprocessed_messages_for_session,
    get_previous_chat_needing_processing
)
from services.message_analysis import analyze_message_for_memory, create_memory_request_from_analysis
from services.message_batch_analysis import analyze_message_batch_for_memory, process_batch_analysis_results
from services.memory_service import handle_incoming_memory
from services.logger_singleton import LoggerSingleton
from memory.memory_graph import MemoryGraph
from services.utils import get_memory_graph

logger = LoggerSingleton.get_logger(__name__)


class MessageProcessingPipeline:
    """
    Message processing pipeline that can be easily migrated to Temporal workflows
    
    This class structures the processing steps in a way that maps cleanly to Temporal:
    - Each method represents a potential Temporal activity
    - Clear separation of concerns for each processing step
    - Proper error handling and status tracking
    - Structured for durable execution patterns
    """
    
    def __init__(self):
        self.memory_graph = None
    
    async def _ensure_memory_graph(self):
        """Ensure memory graph is initialized"""
        if self.memory_graph is None:
            self.memory_graph = get_memory_graph()
    
    async def process_message_workflow(
        self,
        message_request: MessageRequest,
        message_response: MessageResponse,
        user_id: str,
        session_token: str,
        workspace_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main workflow for processing a message
        
        This method orchestrates the entire pipeline and can be easily converted
        to a Temporal workflow with each step becoming an activity.
        
        Args:
            message_request: Original message request
            message_response: Stored message response
            user_id: User ID for memory creation
            session_token: Session token for authentication
            workspace_id: Optional workspace ID
            api_key: Optional API key
            
        Returns:
            Dict with processing results
        """
        workflow_result = {
            "message_id": message_response.objectId,
            "session_id": message_response.sessionId,
            "steps_completed": [],
            "memory_created": False,
            "memory_id": None,
            "error": None
        }
        
        try:
            # Step 1: Update status to analyzing
            await self._update_processing_status(
                message_response.objectId, 
                "analyzing"
            )
            workflow_result["steps_completed"].append("status_updated_analyzing")
            
            # Step 2: Analyze message for memory worthiness
            analysis_result = await self._analyze_message_content(
                message_request,
                message_response.sessionId
            )
            workflow_result["steps_completed"].append("message_analyzed")
            workflow_result["analysis"] = {
                "is_memory_worthy": analysis_result.is_memory_worthy,
                "category": analysis_result.category,
                "confidence": analysis_result.confidence_score
            }
            
            # Step 3: Create memory if worthy
            if analysis_result.is_memory_worthy:
                memory_id = await self._create_memory_from_analysis(
                    message_request,
                    analysis_result,
                    message_response.objectId,
                    user_id,
                    session_token,
                    workspace_id,
                    api_key,
                    api_key_id
                )
                
                if memory_id:
                    workflow_result["memory_created"] = True
                    workflow_result["memory_id"] = memory_id
                    workflow_result["steps_completed"].append("memory_created")
                    
                    # Step 4: Link message to memory
                    await self._link_message_to_memory(
                        message_response.objectId,
                        memory_id
                    )
                    workflow_result["steps_completed"].append("message_linked")
            
            # Step 5: Update final status
            await self._update_processing_status(
                message_response.objectId,
                "completed"
            )
            workflow_result["steps_completed"].append("status_updated_completed")
            
            logger.info(f"Message processing workflow completed for {message_response.objectId}")
            return workflow_result
            
        except Exception as e:
            logger.error(f"Error in message processing workflow: {str(e)}")
            workflow_result["error"] = str(e)
            
            # Update status to failed
            await self._update_processing_status(
                message_response.objectId,
                "failed",
                str(e)
            )
            
            return workflow_result
    
    async def _update_processing_status(
        self,
        message_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Activity: Update message processing status
        
        This maps directly to a Temporal activity for status updates.
        """
        try:
            return await update_message_processing_status(
                message_id,
                status,
                error_message
            )
        except Exception as e:
            logger.error(f"Failed to update processing status: {str(e)}")
            return False
    
    async def _analyze_message_content(
        self,
        message_request: MessageRequest,
        session_id: str
    ) -> Any:
        """
        Activity: Analyze message content for memory extraction
        
        This maps directly to a Temporal activity for AI analysis.
        """
        try:
            # Get session context for better analysis (future enhancement)
            session_context = None  # Could be implemented later
            
            return await analyze_message_for_memory(
                message_request,
                session_context
            )
        except Exception as e:
            logger.error(f"Failed to analyze message content: {str(e)}")
            raise
    
    async def _create_memory_from_analysis(
        self,
        message_request: MessageRequest,
        analysis_result: Any,
        message_id: str,
        user_id: str,
        session_token: str,
        workspace_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Activity: Create memory from analysis results
        
        This maps directly to a Temporal activity for memory creation.
        """
        try:
            await self._ensure_memory_graph()
            
            # Create memory request from analysis
            memory_request = await create_memory_request_from_analysis(
                message_request,
                analysis_result,
                message_id
            )
            
            if not memory_request:
                logger.info("No memory request created from analysis")
                return None
            
            # memory_request is already an AddMemoryRequest from the analysis
            
            # Use existing memory creation pipeline
            # Create a mock neo session for the memory creation
            await self.memory_graph.ensure_async_connection()
            async with self.memory_graph.async_neo_conn.get_session() as neo_session:
                response = await handle_incoming_memory(
                    memory_request,
                    user_id,  # end_user_id
                    user_id,  # developer_user_id
                    session_token,
                    neo_session,
                    None,  # user_info
                    "messages_api",  # client_type
                    self.memory_graph,
                    None,  # background_tasks - not needed for this flow
                    skip_background_processing=True,  # We're already in background processing
                    api_key=api_key,
                    legacy_route=False,
                    workspace_id=workspace_id,
                    api_key_id=api_key_id
                )
            
            if response and response.data and len(response.data) > 0:
                memory_id = response.data[0].objectId
                logger.info(f"Successfully created memory {memory_id} from message {message_id}")
                return memory_id
            else:
                logger.warning("Memory creation returned no data")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create memory from analysis: {str(e)}")
            raise
    
    async def _link_message_to_memory(
        self,
        message_id: str,
        memory_id: str
    ) -> bool:
        """
        Activity: Link message to created memory
        
        This maps directly to a Temporal activity for relationship creation.
        """
        try:
            # Update the PostMessage with a reference to the created memory
            from services.message_service import PARSE_SERVER_URL, HEADERS
            import httpx
            
            update_data = {
                "createdMemoryId": memory_id,
                "memoryLinked": True
            }
            
            url = f"{PARSE_SERVER_URL}/parse/classes/PostMessage/{message_id}"
            
            async with httpx.AsyncClient() as client:
                response = await client.put(url, headers=HEADERS, json=update_data)
                response.raise_for_status()
                
                logger.info(f"Successfully linked message {message_id} to memory {memory_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to link message to memory: {str(e)}")
            return False


# Global pipeline instance
_pipeline_instance = None

def get_message_processing_pipeline() -> MessageProcessingPipeline:
    """Get or create the message processing pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MessageProcessingPipeline()
    return _pipeline_instance


def process_message_background(
    message_request: MessageRequest,
    message_response: MessageResponse,
    user_id: str,
    session_token: str,
    workspace_id: Optional[str] = None,
    api_key: Optional[str] = None,
    api_key_id: Optional[str] = None
):
    """
    Background task wrapper for message processing
    
    This function can be easily replaced with a Temporal workflow starter
    when migrating to Temporal.
    """
    async def _process():
        pipeline = get_message_processing_pipeline()
        await pipeline.process_message_workflow(
            message_request,
            message_response,
            user_id,
            session_token,
            workspace_id,
            api_key,
            api_key_id
        )
    
    # Run the async processing
    asyncio.create_task(_process())


async def process_previous_session_messages(
    current_session_id: str,
    user_id: str,
    session_token: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process unprocessed messages from the previous session
    This implements the correct logic: only check the most recent previous session
    """
    workflow_result = {
        "previous_session_processing": True,
        "processed_session": None,
        "messages_processed": 0,
        "memories_created": 0,
        "errors": []
    }
    
    try:
        # Get the previous session that needs processing
        previous_chat = await get_previous_chat_needing_processing(user_id, current_session_id)
        
        if not previous_chat:
            logger.info("No previous session needs processing")
            return workflow_result
            
        prev_session_id = previous_chat.get("sessionId")
        prev_unprocessed = previous_chat.get("messageCount", 0) - previous_chat.get("lastProcessedMessageIndex", 0)
        
        logger.info(f"Processing previous session {prev_session_id} with {prev_unprocessed} unprocessed messages")
        
        # Get unprocessed messages for the previous session
        unprocessed_messages = await get_unprocessed_messages_for_session(prev_session_id, user_id)
        
        if unprocessed_messages:
            try:
                # Mark messages as processing
                for msg in unprocessed_messages:
                    await update_message_processing_status(msg["objectId"], "processing")
                
                # Analyze batch (with summaries)
                analysis_results, summaries = await analyze_message_batch_for_memory(
                    unprocessed_messages,
                    session_context=f"Previous Session {prev_session_id}",
                    session_id=prev_session_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id
                )
                
                # Process results and create memories (for previous session processing)
                batch_result = await process_batch_analysis_results(
                    analysis_results,
                    summaries,  # ✅ Pass summaries
                    user_id,
                    session_token,
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    api_key_id=api_key_id,
                    session_id=prev_session_id,  # ✅ Pass session_id
                    parent_background_tasks=None  # No parent context for previous session cleanup
                )
                
                # Update workflow result
                workflow_result["processed_session"] = prev_session_id
                workflow_result["messages_processed"] = len(unprocessed_messages)
                workflow_result["memories_created"] = batch_result.get("memories_created", 0)
                
                # Mark messages as completed
                for msg in unprocessed_messages:
                    await update_message_processing_status(msg["objectId"], "completed")
                
                logger.info(f"Previous session processing completed: {len(unprocessed_messages)} messages processed")
                
            except Exception as e:
                error_msg = f"Error processing previous session {prev_session_id}: {e}"
                logger.error(error_msg)
                workflow_result["errors"].append(error_msg)
        
    except Exception as e:
        error_msg = f"Error in previous session processing: {e}"
        logger.error(error_msg)
        workflow_result["errors"].append(error_msg)
    
    return workflow_result


async def smart_message_processing_workflow(
    message_request: MessageRequest,
    message_response: MessageResponse,
    user_id: str,
    session_token: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    parent_background_tasks: Optional[BackgroundTasks] = None
) -> Dict[str, Any]:
    """
    Smart message processing workflow that analyzes messages in batches
    
    This workflow:
    1. Checks if analysis should be triggered (every 15 messages or new session)
    2. If triggered, processes unprocessed messages in batch
    3. Otherwise, just stores the message for later processing
    """
    logger.info(f"🔄 Smart message processing workflow started for message {message_response.objectId} in session {message_request.sessionId}")
    
    workflow_result = {
        "message_id": message_response.objectId,
        "session_id": message_request.sessionId,
        "analysis_triggered": False,
        "batch_processed": False,
        "processing_enabled": message_request.process_messages,
        "error": None
    }
    
    try:
        # Check if processing is enabled
        if not message_request.process_messages:
            # Just mark as stored without processing
            await update_message_processing_status(message_response.objectId, "stored_only")
            logger.info(f"Message {message_response.objectId} stored without processing (process_messages=false)")
            return workflow_result

        # Detect if this is a new session (first few messages)
        # Get the chat session to check message count
        from services.message_service import get_or_create_chat_session
        chat = await get_or_create_chat_session(message_request.sessionId, user_id, None, None, None)
        message_count = chat.get("messageCount", 0) if chat else 0
        is_new_session = message_count <= 2  # Consider it new if <=2 messages

        logger.info(f"Session {message_request.sessionId} has {message_count} messages, is_new_session={is_new_session}")

        # Check if we should trigger analysis
        should_analyze = await should_trigger_analysis(
            message_request.sessionId,
            user_id,
            is_new_session=is_new_session  # Now properly detected
        )
        
        if should_analyze:
            logger.info(f"Triggering batch analysis - checking current session and cross-session opportunities")
            workflow_result["analysis_triggered"] = True
            
            # Check if we should also process the previous session (when new session starts)
            previous_session_result = await process_previous_session_messages(
                message_request.sessionId,
                user_id,
                session_token,
                workspace_id,
                organization_id,
                namespace_id,
                api_key_id=api_key_id
            )
            
            if previous_session_result.get("messages_processed", 0) > 0:
                workflow_result["previous_session_processing"] = previous_session_result
                logger.info(f"Processed previous session: {previous_session_result['processed_session']}")
            
            # Process current session if it has enough messages
            unprocessed_messages = await get_unprocessed_messages_for_session(
                message_request.sessionId,
                user_id
            )
            
            if unprocessed_messages:
                # Perform batch analysis for current session (with summaries)
                analysis_results, summaries = await analyze_message_batch_for_memory(
                    unprocessed_messages,
                    session_context=f"Session {message_request.sessionId}",
                    session_id=message_request.sessionId,
                    user_id=user_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id
                )
                
                # Process the analysis results
                batch_stats = await process_batch_analysis_results(
                    analysis_results,
                    summaries,  # ✅ Pass summaries
                    user_id,
                    session_token,
                    workspace_id,
                    organization_id,
                    namespace_id,
                    api_key_id=api_key_id,
                    session_id=message_request.sessionId,  # ✅ Pass session_id
                    parent_background_tasks=parent_background_tasks  # ✅ Pass it down
                )
                
                workflow_result["batch_processed"] = True
                workflow_result["batch_stats"] = batch_stats
                
                logger.info(f"Batch processing completed for session {message_request.sessionId}: {batch_stats}")
            else:
                logger.info(f"No unprocessed messages found for session {message_request.sessionId}")
        else:
            # Just mark the message as pending for future batch processing
            await update_message_processing_status(message_response.objectId, "pending")
            logger.info(f"Message {message_response.objectId} queued for batch processing")
        
        return workflow_result
        
    except Exception as e:
        logger.error(f"Error in smart message processing workflow: {str(e)}")
        workflow_result["error"] = str(e)
        
        # Update status to failed
        await update_message_processing_status(
            message_response.objectId,
            "failed",
            str(e)
        )
        
        return workflow_result


def add_message_processing_task(
    background_tasks: BackgroundTasks,
    message_request: MessageRequest,
    message_response: MessageResponse,
    user_id: str,
    session_token: str,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    namespace_id: Optional[str] = None,
    api_key_id: Optional[str] = None
):
    """
    Add smart message processing to FastAPI background tasks
    
    Uses the smart analysis workflow that processes messages in batches
    instead of analyzing every single message.
    
    IMPORTANT: Passes parent BackgroundTasks through so graph generation
    tasks get added to the same instance and auto-execute.
    """
    async def _async_wrapper():
        try:
            logger.info(f"🚀 Starting background processing for message {message_response.objectId} in session {message_request.sessionId}")
            await smart_message_processing_workflow(
                message_request,
                message_response,
                user_id,
                session_token,
                workspace_id,
                organization_id,
                namespace_id,
                api_key_id=api_key_id,
                parent_background_tasks=background_tasks  # ✅ Thread it through!
            )
            logger.info(f"✅ Completed background processing for message {message_response.objectId}")
        except Exception as e:
            logger.error(f"❌ Error in background processing for message {message_response.objectId}: {e}", exc_info=True)
    
    # Add to FastAPI background tasks
    logger.info(f"📋 Adding background task for message {message_response.objectId} in session {message_request.sessionId}")
    background_tasks.add_task(_async_wrapper)
