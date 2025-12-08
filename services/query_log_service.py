import asyncio
import time
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
import httpx
from pydantic import BaseModel, Field
from services.logger_singleton import LoggerSingleton
from models.parse_server import QueryLog, ParsePointer
from services.auth_utils import get_user_from_token
from services.user_utils import User
from api_handlers.chat_gpt_completion import ChatGPTCompletion
from os import environ as env
from dotenv import find_dotenv, load_dotenv
from models.parse_server import (
    ParseStoredMemory, AddMemoryResponse, ErrorDetail, DeletionStatus, BatchMemoryResponse, BatchMemoryError, DeleteMemoryResponse, UpdateMemoryResponse, UpdateMemoryItem, SystemUpdateStatus, DocumentUploadResponse, DocumentUploadStatus, AddMemoryItem, Memory, ParsePointer, QueryLog, MemoryRetrievalLog
)
from models.memory_models import SearchResponse, SearchRequest, MemoryMetadata, SearchResult, RelationshipItem, NeoNode, RelatedMemoryResult


# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


logger = LoggerSingleton.get_logger(__name__)
logger.info("Logger initialized at top of main.py!")

class QueryClassification(BaseModel):
    """Model for query classification response"""
    goals: List[str] = Field(default_factory=list, description="Related goals")
    use_cases: List[str] = Field(default_factory=list, description="Related use cases")
    steps: List[str] = Field(default_factory=list, description="Related steps")
    predicted_tier: str = Field(default="2", description="Predicted router tier (0, 1, or 2)")
    confidence: float = Field(default=0.5, description="Confidence score (0.0 to 1.0)")
    goal_scores: List[float] = Field(default_factory=list, description="Confidence scores for each goal (0.0 to 1.0), in same order as goals")
    use_case_scores: List[float] = Field(default_factory=list, description="Confidence scores for each use case (0.0 to 1.0), in same order as use_cases")
    step_scores: List[float] = Field(default_factory=list, description="Confidence scores for each step (0.0 to 1.0), in same order as steps")

class QueryLogService:
    """Service for handling QueryLog operations including background classification and logging"""
    
    def __init__(self):
        # Don't initialize ChatGPTCompletion here - get it from app state when needed
        pass
    
    async def create_query_log_background(
        self,
        query_log: QueryLog,
        session_token: Optional[str],
        api_key: Optional[str],
        client_type: str = 'papr_plugin',
        chat_gpt=None,  # Pass chat_gpt from app state
        search_id: str = None  # NEW: accept search_id
    ) -> Optional[str]:
        """
        Background task to create QueryLog entry with optional classification
        """
        try:
            # Perform classification if not provided in metadata and chat_gpt is available
            logger.info(f"Checking if classification is needed - has_data: {self._has_classification_data(query_log)}, chat_gpt: {chat_gpt is not None}")
            if not self._has_classification_data(query_log) and chat_gpt:
                logger.info("Starting classification process...")
                import asyncio
                # Add timeout wrapper around entire classification to prevent hanging tests
                try:
                    classification_data = await asyncio.wait_for(
                        self._classify_query_with_user_data(
                            query_log.queryText, 
                            chat_gpt, 
                            session_token, 
                            api_key,
                            user_id=query_log.user.objectId
                        ),
                        timeout=25.0  # 25 second timeout for entire classification process
                    )
                except asyncio.TimeoutError:
                    logger.warning("Classification process timed out after 25 seconds, skipping classification")
                    classification_data = {
                        'predictedTier': '2',
                        'tierPredictionConfidence': 0.5,
                        'relatedGoals': [],
                        'relatedUseCases': [],
                        'relatedSteps': [],
                        'goalClassificationScores': [],
                        'useCaseClassificationScores': []
                    }
                logger.info(f"Classification data received: {classification_data}")
                
                # Update QueryLog object with classification data
                if classification_data.get('predictedTier'):
                    query_log.predictedTier = classification_data['predictedTier']
                if classification_data.get('tierPredictionConfidence'):
                    query_log.tierPredictionConfidence = classification_data['tierPredictionConfidence']
                if classification_data.get('relatedGoals'):
                    query_log.relatedGoals = classification_data['relatedGoals']
                if classification_data.get('relatedUseCases'):
                    query_log.relatedUseCases = classification_data['relatedUseCases']
                if classification_data.get('relatedSteps'):
                    query_log.relatedSteps = classification_data['relatedSteps']
                if classification_data.get('goalClassificationScores'):
                    query_log.goalClassificationScores = classification_data['goalClassificationScores']
                    logger.info(f"Set goalClassificationScores: {query_log.goalClassificationScores}")
                if classification_data.get('useCaseClassificationScores'):
                    query_log.useCaseClassificationScores = classification_data['useCaseClassificationScores']
                    logger.info(f"Set useCaseClassificationScores: {query_log.useCaseClassificationScores}")

                logger.info(f"QueryLog with classification data: {query_log}")
            else:
                logger.info("Skipping classification - either data already exists or chat_gpt not available")
            
            # Create QueryLog in Parse Server
            logger.info(f"About to save QueryLog to Parse Server with goalClassificationScores: {query_log.goalClassificationScores}")
            logger.info(f"About to save QueryLog to Parse Server with useCaseClassificationScores: {query_log.useCaseClassificationScores}")
            # Pass search_id as objectId if provided
            if search_id:
                query_log.objectId = search_id
            result = await self._save_query_log_to_parse(
                query_log=query_log,
                session_token=session_token,
                api_key=api_key,
                client_type=client_type
            )
            
            if result and result.get('objectId'):
                logger.info(f"QueryLog created successfully for query: {query_log.queryText[:50]}... with ID: {result['objectId']}")
                return result['objectId']
            else:
                logger.error("Failed to create QueryLog - no objectId returned")
                return None
            
        except Exception as e:
            logger.error(f"Error creating QueryLog in background: {e}", exc_info=True)
            return None
    
    def _has_classification_data(self, query_log: QueryLog) -> bool:
        """Check if classification data is already provided in QueryLog"""
        has_data = bool(
            (query_log.relatedGoals and len(query_log.relatedGoals) > 0) or 
            (query_log.relatedUseCases and len(query_log.relatedUseCases) > 0) or 
            (query_log.relatedSteps and len(query_log.relatedSteps) > 0)
        )
        logger.info(f"QueryLog has classification data: {has_data}")
        logger.info(f"relatedGoals: {query_log.relatedGoals} (length: {len(query_log.relatedGoals) if query_log.relatedGoals else 0})")
        logger.info(f"relatedUseCases: {query_log.relatedUseCases} (length: {len(query_log.relatedUseCases) if query_log.relatedUseCases else 0})")
        logger.info(f"relatedSteps: {query_log.relatedSteps} (length: {len(query_log.relatedSteps) if query_log.relatedSteps else 0})")
        return has_data
    
    async def _get_user_goals_and_usecases(self, user_id: str, session_token: str, api_key: Optional[str] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Get existing goals, use cases, and steps for a user
        Similar to the logic in _index_memories_and_process
        Returns full objects for proper objectId mapping
        """
        try:
            # Import the functions from memory_management
            from services.memory_management import get_user_goals_async, get_user_usecases_async
            
            logger.info(f"Fetching goals and use cases for user {user_id}")
            logger.info(f"Using session_token: {session_token[:20] if session_token else 'None'}...")
            logger.info(f"Using api_key: {api_key[:20] if api_key else 'None'}...")
            
            # Get most recent goals (limit small for classification)
            existing_goals = await get_user_goals_async(user_id, session_token, api_key=api_key, limit=5)
            logger.info(f"Raw goals response: {existing_goals}")
            
            # Get most recent use cases (limit small for classification)
            existing_use_cases = await get_user_usecases_async(user_id, session_token, api_key=api_key, limit=5)
            logger.info(f"Raw use cases response: {existing_use_cases}")
            
            # For now, we'll use an empty list for steps since we don't have a steps API yet
            # TODO: Add get_user_steps_async when available
            existing_steps = []
            
            logger.debug(f'Existing goals: {existing_goals}')
            logger.debug(f'Existing use cases: {existing_use_cases}')
            logger.debug(f'Existing steps: {existing_steps}')
            
            return existing_goals, existing_use_cases, existing_steps
            
        except Exception as e:
            logger.error(f"Error getting user goals and use cases: {e}", exc_info=True)
            return [], [], []
    
    def _extract_goal_titles(self, goals: List[Any]) -> List[str]:
        """Extract goal titles from goal objects"""
        extracted_goals = []
        if goals:
            for goal in goals:
                if isinstance(goal, dict) and 'title' in goal:
                    extracted_goals.append(goal['title'])
                elif hasattr(goal, 'title'):
                    extracted_goals.append(goal.title)
        return extracted_goals
    
    def _extract_usecases(self, use_cases: List[Any]) -> List[str]:
        """Extract use case titles from use case objects"""
        extracted_use_cases = []
        if use_cases:
            for uc in use_cases:
                if isinstance(uc, dict) and 'name' in uc:
                    extracted_use_cases.append(uc['name'])
                elif hasattr(uc, 'name'):
                    extracted_use_cases.append(uc.name)
        return extracted_use_cases
    
    def _find_object_by_title(self, title: str, objects: List[Any], field_name: str = 'title') -> Optional[str]:
        """Find objectId by matching title/name in a list of objects"""
        for obj in objects:
            obj_title = None
            obj_id = None
            
            if isinstance(obj, dict):
                obj_title = obj.get(field_name)
                obj_id = obj.get('objectId')
            elif hasattr(obj, field_name) and hasattr(obj, 'objectId'):
                obj_title = getattr(obj, field_name)
                obj_id = obj.objectId
            
            if obj_title and obj_id and obj_title.lower() == title.lower():
                return obj_id
        
        return None
    
    def _suggest_potential_goals(self, query_text: str, goal_titles: List[str]) -> List[str]:
        """Suggest potential goal matches based on query keywords"""
        query_lower = query_text.lower()
        suggested_goals = []
        
        # Keywords that might indicate relevant goals
        launch_keywords = ['launch', 'release', 'deploy', 'go live', 'rollout']
        task_keywords = ['task', 'work', 'project', 'assignment', 'todo']
        team_keywords = ['team', 'collaboration', 'leadership', 'management']
        business_keywords = ['business', 'growth', 'revenue', 'acquisition', 'marketing']
        
        for goal in goal_titles:
            goal_lower = goal.lower()
            
            # Check for launch-related goals
            if any(keyword in query_lower for keyword in launch_keywords):
                if any(keyword in goal_lower for keyword in ['launch', 'marketing', 'campaign', 'acquisition', 'user', 'product']):
                    suggested_goals.append(goal)
            
            # Check for task-related goals
            if any(keyword in query_lower for keyword in task_keywords):
                if any(keyword in goal_lower for keyword in ['task', 'workflow', 'automation', 'process', 'management']):
                    suggested_goals.append(goal)
            
            # Check for team-related goals
            if any(keyword in query_lower for keyword in team_keywords):
                if any(keyword in goal_lower for keyword in ['team', 'leadership', 'collaboration', 'management']):
                    suggested_goals.append(goal)
            
            # Check for business-related goals
            if any(keyword in query_lower for keyword in business_keywords):
                if any(keyword in goal_lower for keyword in ['business', 'growth', 'revenue', 'acquisition', 'marketing']):
                    suggested_goals.append(goal)
        
        return list(set(suggested_goals))  # Remove duplicates
    
    async def _classify_query_with_user_data(
        self, 
        query_text: str, 
        chat_gpt, 
        session_token: str, 
        api_key: Optional[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Use LLM to classify query into goals, use cases, and steps using user's existing data
        Similar to the logic in _index_memories_and_process
        """
        import asyncio
        
        try:
            # Add timeout to prevent hanging
            return await asyncio.wait_for(
                self._classify_query_with_user_data_internal(
                    query_text, chat_gpt, session_token, api_key, user_id
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("Classification timed out after 30 seconds")
            return {
                'predictedTier': '2',
                'tierPredictionConfidence': 0.5,
                'relatedGoals': [],
                'relatedUseCases': [],
                'relatedSteps': [],
                'goalClassificationScores': [],
                'useCaseClassificationScores': []
            }
        except Exception as e:
            logger.error(f"Error in classification with timeout: {e}", exc_info=True)
            return {
                'predictedTier': '2',
                'tierPredictionConfidence': 0.5,
                'relatedGoals': [],
                'relatedUseCases': [],
                'relatedSteps': [],
                'goalClassificationScores': [],
                'useCaseClassificationScores': []
            }
    
    async def _classify_query_with_user_data_internal(
        self, 
        query_text: str, 
        chat_gpt, 
        session_token: str, 
        api_key: Optional[str],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Internal method for classification (without timeout wrapper)
        """
        try:
            # Get user's most recent goals/use cases to keep prompt small and relevant
            existing_goals, existing_use_cases, existing_steps = await self._get_user_goals_and_usecases(
                user_id, session_token, api_key
            )
            
            logger.info(f"Found {len(existing_goals)} existing goals for user {user_id}")
            logger.info(f"Found {len(existing_use_cases)} existing use cases for user {user_id}")
            logger.info(f"Found {len(existing_steps)} existing steps for user {user_id}")
            
            # If no goals/use cases found, return early with defaults
            if not existing_goals and not existing_use_cases:
                logger.info("No goals or use cases found for user, returning default classification")
                return {
                    'predictedTier': '2',
                    'tierPredictionConfidence': 0.5,
                    'relatedGoals': [],
                    'relatedUseCases': [],
                    'relatedSteps': [],
                    'goalClassificationScores': [],
                    'useCaseClassificationScores': []
                }
            
            # Log the actual goal and use case titles for debugging (trim for prompt size)
            full_goal_titles = self._extract_goal_titles(existing_goals)
            full_use_case_titles = self._extract_usecases(existing_use_cases)
            full_step_titles = self._extract_usecases(existing_steps)

            # Trim lists to avoid token limits; include only top N for prompt context
            MAX_ITEMS_FOR_PROMPT = 5
            goal_titles = full_goal_titles[:MAX_ITEMS_FOR_PROMPT]
            use_case_titles = full_use_case_titles[:MAX_ITEMS_FOR_PROMPT]
            step_titles = full_step_titles[:MAX_ITEMS_FOR_PROMPT]
            
            logger.info(f"Available goal titles: {goal_titles}")
            logger.info(f"Available use case titles: {use_case_titles}")
            logger.info(f"Available step titles: {step_titles}")
            logger.info(f"Num goals: {len(goal_titles)}, use cases: {len(use_case_titles)}, steps: {len(step_titles)}")
            
            # Debug the structure of the data
            if existing_goals:
                logger.info(f"First goal structure: {existing_goals[0]}")
            if existing_use_cases:
                logger.info(f"First use case structure: {existing_use_cases[0]}")
            
            # Suggest potential goal matches
            suggested_goals = self._suggest_potential_goals(query_text, goal_titles)
            logger.info(f"Suggested potential goals for query '{query_text}': {suggested_goals}")
            
            # Build richer context including recent goal details
            def summarize_goal(g: dict) -> str:
                if isinstance(g, dict):
                    title = g.get('title')
                    desc = (g.get('description') or '')
                    kr = g.get('keyResults') or []
                    kr_summ = ", ".join([str(k) for k in kr][:3]) if isinstance(kr, list) else str(kr)
                    return f"{title} | {desc[:140]} | KR: {kr_summ}"
                return str(g)

            def summarize_usecase(u: dict) -> str:
                if isinstance(u, dict):
                    name = u.get('name')
                    desc = (u.get('description') or '')
                    return f"{name} | {desc[:140]}"
                return str(u)

            recent_goal_summaries = [summarize_goal(g) for g in existing_goals[:MAX_ITEMS_FOR_PROMPT]]
            recent_usecase_summaries = [summarize_usecase(u) for u in existing_use_cases[:MAX_ITEMS_FOR_PROMPT]]

            classification_prompt = f"""
            Analyze this query and classify it into the user's existing goals, use cases, and steps. Also predict which router tier would be most appropriate.

            Query: "{query_text}"

            Most recent goals (title | description | top key results): {recent_goal_summaries}
            Most recent use cases (name | description): {recent_usecase_summaries}
            User's existing goals (titles): {goal_titles}
            User's existing use cases (names): {use_case_titles}
            User's existing steps: {step_titles}

            Note: Only the first {MAX_ITEMS_FOR_PROMPT} items are shown per list (of potentially many).

            SUGGESTED POTENTIAL GOALS (based on keyword matching):
            {suggested_goals}

            IMPORTANT GUIDELINES:
            1. Look for goals that are semantically related to the query, even if the exact words don't match
            2. Consider broader themes and concepts - a query about "tasks for launch" might relate to goals about "user acquisition", "marketing campaigns", "product launch", etc.
            3. Think about what the user is trying to accomplish and find goals that align with that intent
            4. If the query mentions specific people or projects, look for goals that might involve those areas
            5. Be generous in matching - it's better to include a potentially relevant goal than miss it
            6. Consider the suggested goals above as starting points, but don't limit yourself to just those

            For the query "{query_text}", consider these potential matches:
            - If it mentions "launch", look for goals about launching, marketing, user acquisition, product development
            - If it mentions specific people (like "Shawkat"), look for goals about team management, leadership, collaboration
            - If it mentions "tasks", look for goals about project management, workflow optimization, task automation
            - If it mentions business objectives, look for goals about business growth, revenue, customer acquisition

            Please provide:
            1. Related goals from the user's existing goals (list of titles)
            2. Related use cases from the user's existing use cases (list of titles)
            3. Related steps from the user's existing steps (list of titles)
            4. For each goal, provide a confidence score (0.0-1.0) indicating how relevant it is to the query
            5. For each use case, provide a confidence score (0.0-1.0) indicating how relevant it is to the query
            6. For each step, provide a confidence score (0.0-1.0) indicating how relevant it is to the query
            7. Predicted router tier (1, 2, or 3)
            8. Overall confidence score (0.0 to 1.0) for the entire classification

            Only return goals, use cases, and steps that actually exist in the user's data.
            If no relevant items exist, return empty lists.

            Format your response as JSON:
            {{
                "goals": ["goal1", "goal2"],
                "goal_scores": [0.8, 0.6],
                "use_cases": ["usecase1", "usecase2"],
                "use_case_scores": [0.9, 0.7],
                "steps": ["step1", "step2"],
                "step_scores": [0.7, 0.6],
                "predicted_tier": "1",
                "confidence": 0.85
            }}
            """
            prompt_len = len(classification_prompt)
            logger.info(f"Classification prompt length: {prompt_len} chars")
            logger.info(f"Classification prompt preview: {classification_prompt[:500]}")
            
            messages = [
                {"role": "system", "content": "You are a query classification assistant."},
                {"role": "user", "content": classification_prompt}
            ]
            
            # Calculate token usage dynamically based on model capabilities
            # gpt-5-mini: 400K context window, 128K max output
            # o1-mini: 128K context window, 65K max output
            model_name = chat_gpt.model_mini
            is_reasoning_model = "gpt-5" in model_name or model_name.startswith("o1-")
            
            # Model-specific limits
            if "gpt-5" in model_name:
                MODEL_CONTEXT_WINDOW = 400_000
                MODEL_MAX_OUTPUT = 128_000
            elif model_name.startswith("o1-"):
                MODEL_CONTEXT_WINDOW = 128_000
                MODEL_MAX_OUTPUT = 65_000
            else:
                # Default for gpt-4o-mini and similar
                MODEL_CONTEXT_WINDOW = 128_000
                MODEL_MAX_OUTPUT = 16_000
            
            # Estimate input tokens
            try:
                prompt_token_estimate = chat_gpt.estimate_message_tokens(messages)
                logger.info(f"Classification prompt token estimate: {prompt_token_estimate}")
                
                # Calculate dynamic max_completion_tokens
                # Reserve some buffer for safety (1-2% of context window)
                buffer_tokens = max(500, int(MODEL_CONTEXT_WINDOW * 0.02))
                available_output_tokens = MODEL_CONTEXT_WINDOW - prompt_token_estimate - buffer_tokens
                
                # Cap at model's max output limit
                max_completion_tokens = min(available_output_tokens, MODEL_MAX_OUTPUT)
                
                # For reasoning models (gpt-5-mini, o1-mini, o1-preview), there's a fixed reasoning token budget
                # gpt-5-mini and o1 models use reasoning tokens internally which have a fixed limit
                # The reasoning tokens are PART of the completion tokens, not separate
                # So we don't need to artificially limit max_completion_tokens - the model will handle it
                # The warning about hitting the reasoning token limit means the model needs more reasoning
                # tokens to complete the task, which we can't increase (it's fixed at ~500-65K depending on model)
                
                # COST OPTIMIZATION: Classification output should be small (JSON with goal/usecase names + scores)
                # Typical output: 200-800 tokens for a classification response
                # For reasoning models: need room for reasoning tokens (500+) + actual output
                # Set a reasonable upper bound to control costs while allowing enough room
                # Reuse CLASSIFY_MAX_TOTAL_TOKENS but now it means max output tokens (more intuitive)
                CLASSIFICATION_OUTPUT_LIMIT = int(env.get("CLASSIFY_MAX_TOTAL_TOKENS", "16000"))
                
                if is_reasoning_model:
                    # Reasoning models need extra room for internal reasoning (500-5000 tokens)
                    # Set higher limit to ensure reasoning + output fits
                    max_completion_tokens = min(max_completion_tokens, CLASSIFICATION_OUTPUT_LIMIT)
                else:
                    # Non-reasoning models: much smaller limit is fine
                    max_completion_tokens = min(max_completion_tokens, 4000)
                
                logger.info(f"  Cost-optimized max_completion_tokens: {max_completion_tokens:,} tokens")
                logger.info(f"  (Reasoning model: {is_reasoning_model}, limit: {CLASSIFICATION_OUTPUT_LIMIT:,})")
                
                logger.info(f"📊 Token Budget Calculation:")
                logger.info(f"  Model: {model_name}")
                logger.info(f"  Context Window: {MODEL_CONTEXT_WINDOW:,} tokens")
                logger.info(f"  Input Tokens: {prompt_token_estimate:,} tokens")
                logger.info(f"  Buffer: {buffer_tokens:,} tokens")
                logger.info(f"  Available for Output: {available_output_tokens:,} tokens")
                logger.info(f"  Max Output (model limit): {MODEL_MAX_OUTPUT:,} tokens")
                logger.info(f"  Dynamic max_completion_tokens: {max_completion_tokens:,} tokens")
                
            except Exception as estimate_err:
                logger.warning(f"Failed to estimate tokens, using default: {estimate_err}")
                max_completion_tokens = 4000  # Safe default
                prompt_token_estimate = 2000
            
            # Use the correct pattern based on model location
            try:
                if chat_gpt.model_location_cloud == False:
                    # Local Ollama with instructor
                    logger.info("Using Local Ollama for classification")
                    logger.info(f"🚀 Local Ollama Parameters: max_tokens={max_completion_tokens:,}")
                    response = await chat_gpt.instructor_groq_client.chat.completions.create(
                        model="meta-llama/llama-4-maverick-17b-128e-instruct",
                        messages=messages,
                        response_model=QueryClassification,
                        max_tokens=max_completion_tokens,  # Dynamic based on input
                    )
                    response_json = response.model_dump_json(indent=2)
                    response = json.loads(response_json)
                else:
                    # Cloud OpenAI with typed parse first
                    logger.info("Using Cloud OpenAI (mini) for classification (typed parse first)")
                    try:
                        # Add timeout to individual API call to prevent hanging
                        import asyncio
                        
                        # Build API parameters with dynamic max_completion_tokens
                        api_params = {
                            "model": chat_gpt.model_mini,
                            "messages": messages,
                            "response_format": QueryClassification,
                            "max_completion_tokens": max_completion_tokens
                        }
                        
                        logger.info(f"🚀 API Call Parameters: model={api_params['model']}, max_completion_tokens={api_params['max_completion_tokens']:,}")
                        
                        response = await asyncio.wait_for(
                            chat_gpt.async_client.beta.chat.completions.parse(**api_params),
                            timeout=10.0  # 10 second timeout for first attempt (reduced for faster failure)
                        )
                        
                        # Log actual token usage from the API response
                        if hasattr(response, 'usage') and response.usage:
                            usage = response.usage
                            logger.info(f"✅ API Response Token Usage:")
                            logger.info(f"  Prompt Tokens: {usage.prompt_tokens:,}")
                            logger.info(f"  Completion Tokens: {usage.completion_tokens:,}")
                            logger.info(f"  Total Tokens: {usage.total_tokens:,}")
                            
                            # Log reasoning tokens if available (for gpt-5-mini and o1 models)
                            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                                details = usage.completion_tokens_details
                                if hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                                    logger.info(f"  Reasoning Tokens: {details.reasoning_tokens:,}")
                                    logger.info(f"  Note: Reasoning tokens are part of completion tokens, used for internal 'thinking'")
                        
                        classification_ast = response.choices[0].message.parsed
                        response = classification_ast.model_dump_json(indent=2)
                    except (Exception, asyncio.TimeoutError) as parse_err:
                        error_msg = str(parse_err)
                        # Check if it's a token limit error
                        if "length limit" in error_msg.lower() or "reasoning_tokens" in error_msg.lower():
                            logger.warning(f"Classification hit token limit: {parse_err}. Using simpler fallback.")
                        elif isinstance(parse_err, asyncio.TimeoutError):
                            logger.warning(f"Classification API call timed out: {parse_err}. Using fallback.")
                        else:
                            logger.warning(f"Typed parse classification failed, retrying with json_object on mini model: {parse_err}")
                        
                        # Fallback: simpler JSON response with much lower token limit
                        # Use 1/4 of the normal limit for fallback (simpler, more conservative)
                        fallback_tokens = max(300, max_completion_tokens // 4)
                        logger.info(f"Using fallback with max_completion_tokens={fallback_tokens:,}")
                        try:
                            raw = await asyncio.wait_for(
                                chat_gpt._create_completion_with_fallback_async(
                                    model=chat_gpt.model_mini,
                                    messages=messages,
                                    response_format={"type": "json_object"},
                                    max_completion_tokens=fallback_tokens
                                ),
                                timeout=10.0  # 10 second timeout for fallback
                            )
                            response = raw.choices[0].message.content or "{}"
                        except (Exception, asyncio.TimeoutError) as fallback_err:
                            logger.error(f"Fallback classification also failed: {fallback_err}")
                            # Return minimal default response
                            return {
                                'predictedTier': '2',
                                'tierPredictionConfidence': 0.5,
                                'relatedGoals': [],
                                'relatedUseCases': [],
                                'relatedSteps': [],
                                'goalClassificationScores': [],
                                'useCaseClassificationScores': []
                            }
                
                logger.info(f"Raw classification response length: {len(str(response))} chars")
                logger.info(f"Raw classification response preview: {str(response)[:500]}")
                logger.info(f"Raw classification response: {response}")
            except Exception as e:
                logger.error(f"Error during LLM classification call: {e}", exc_info=True)
                # Return default values if LLM call fails
                return {
                    'predictedTier': '2',
                    'tierPredictionConfidence': 0.5,
                    'relatedGoals': [],
                    'relatedUseCases': [],
                    'relatedSteps': [],
                    'goalClassificationScores': [],
                    'useCaseClassificationScores': []
                }
            
            # Parse response
            try:
                import json
                if not response:
                    logger.warning("Empty response from classification")
                    return {
                        'predictedTier': '2',
                        'tierPredictionConfidence': 0.5,
                        'relatedGoals': [],
                        'relatedUseCases': [],
                        'relatedSteps': [],
                        'goalClassificationScores': [],
                        'useCaseClassificationScores': []
                    }
                    
                classification = json.loads(response)
                logger.info(f"Parsed classification: {classification}")
                
                # Convert goal/use case/step titles to ParsePointers
                related_goals = []
                if classification.get('goals'):
                    logger.info(f"LLM returned goals: {classification['goals']}")
                    for goal_title in classification['goals']:
                        goal_id = self._find_object_by_title(goal_title, existing_goals, 'title')
                        if goal_id:
                            related_goals.append(ParsePointer(objectId=goal_id, className="Goal"))
                        else:
                            logger.warning(f"Could not find objectId for goal title: {goal_title}")
                
                related_use_cases = []
                if classification.get('use_cases'):
                    logger.info(f"LLM returned use cases: {classification['use_cases']}")
                    for use_case_title in classification['use_cases']:
                        use_case_id = self._find_object_by_title(use_case_title, existing_use_cases, 'name')
                        if use_case_id:
                            related_use_cases.append(ParsePointer(objectId=use_case_id, className="Usecase"))
                        else:
                            logger.warning(f"Could not find objectId for use case name: {use_case_title}")
                
                related_steps = []
                if classification.get('steps'):
                    logger.info(f"LLM returned steps: {classification['steps']}")
                    for step_title in classification['steps']:
                        step_id = self._find_object_by_title(step_title, existing_steps, 'name')
                        if step_id:
                            related_steps.append(ParsePointer(objectId=step_id, className="Step"))
                        else:
                            logger.warning(f"Could not find objectId for step name: {step_title}")
                
                # Parse confidence scores
                goal_scores = classification.get('goal_scores', [])
                use_case_scores = classification.get('use_case_scores', [])
                step_scores = classification.get('step_scores', [])
                logger.info(f"Goal classification scores: {goal_scores}")
                logger.info(f"Use case classification scores: {use_case_scores}")
                logger.info(f"Step classification scores: {step_scores}")
                
                logger.info(f"Final related goals: {related_goals}")
                logger.info(f"Final related use cases: {related_use_cases}")
                logger.info(f"Final related steps: {related_steps}")
                
                return {
                    'predictedTier': classification.get('predicted_tier', '2'),
                    'tierPredictionConfidence': classification.get('confidence', 0.5),
                    'relatedGoals': related_goals,
                    'relatedUseCases': related_use_cases,
                    'relatedSteps': related_steps,
                    'goalClassificationScores': goal_scores,
                    'useCaseClassificationScores': use_case_scores
                }
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse classification response: {response}")
                return {
                    'predictedTier': '2',
                    'tierPredictionConfidence': 0.5,
                    'relatedGoals': [],
                    'relatedUseCases': [],
                    'relatedSteps': [],
                    'goalClassificationScores': [],
                    'useCaseClassificationScores': []
                }
            except Exception as e:
                logger.warning(f"Error parsing classification response: {e}")
                return {
                    'predictedTier': '2',
                    'tierPredictionConfidence': 0.5,
                    'relatedGoals': [],
                    'relatedUseCases': [],
                    'relatedSteps': [],
                    'goalClassificationScores': [],
                    'useCaseClassificationScores': []
                }
                
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return {
                'predictedTier': '2',
                'tierPredictionConfidence': 0.5,
                'relatedGoals': [],
                'relatedUseCases': [],
                'relatedSteps': [],
                'goalClassificationScores': [],
                'useCaseClassificationScores': []
            }
    
    async def _classify_query(self, query_text: str, chat_gpt) -> Dict[str, Any]:
        """
        Use LLM to classify query into goals, use cases, and steps
        Also predict router tier
        """
        try:
            classification_prompt = f"""
            Analyze this query and classify it into goals, use cases, and steps. Also predict which router tier would be most appropriate.

            Query: "{query_text}"

            Please provide:
            1. Related goals (comma-separated)
            2. Related use cases (comma-separated) 
            3. Related steps (comma-separated)
            4. Predicted router tier (0, 1, or 2)
            5. Confidence score (0.0 to 1.0)

            Format your response as JSON:
            {{
                "goals": ["goal1", "goal2"],
                "use_cases": ["usecase1", "usecase2"],
                "steps": ["step1", "step2"],
                "predicted_tier": "1",
                "confidence": 0.85
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a query classification assistant."},
                {"role": "user", "content": classification_prompt}
            ]
            
            # Use the correct pattern based on model location
            if chat_gpt.model_location_cloud == False:
                # Local Ollama with instructor
                response = await chat_gpt.instructor_groq_client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=messages,
                    response_model=QueryClassification,
                    max_tokens=1000,  # Increased for classification response
                )
                response_json = response.model_dump_json(indent=2)
                response = json.loads(response_json)
            else:
                # Cloud OpenAI
                response = await chat_gpt.async_client.beta.chat.completions.parse(
                    model=chat_gpt.model,
                    messages=messages,
                    response_format=QueryClassification,
                    max_completion_tokens=1000  # Increased for classification response
                )
                classification_ast = response.choices[0].message.parsed
                response = classification_ast.model_dump_json(indent=2)
            
            # Parse response
            try:
                import json
                if not response:
                    logger.warning("Empty response from classification")
                    return {
                        'predictedTier': '2',
                        'tierPredictionConfidence': 0.5
                    }
                    
                classification = json.loads(response)
                
                return {
                    'predictedTier': classification.get('predicted_tier', '2'),
                    'tierPredictionConfidence': classification.get('confidence', 0.5),
                    # Note: We don't populate relatedGoals/relatedUseCases/relatedSteps here
                    # as we don't have the actual objectIds. This is just for tier prediction.
                }
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse classification response: {response}")
                return {
                    'predictedTier': '2',
                    'tierPredictionConfidence': 0.5
                }
            except Exception as e:
                logger.warning(f"Error parsing classification response: {e}")
                return {
                    'predictedTier': '2',
                    'tierPredictionConfidence': 0.5
                }
                
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            return {
                'predictedTier': '2',
                'tierPredictionConfidence': 0.5
            }
    
    async def _save_query_log_to_parse(
        self,
        query_log: QueryLog,
        session_token: Optional[str],
        api_key: Optional[str],
        client_type: Optional[str]
    ) -> Optional[dict]:
        """Save QueryLog to Parse Server"""
        try:
            # Use the memory_management function to store the QueryLog
            from services.memory_management import store_memory_query_log_async
            
            result = await store_memory_query_log_async(
                query_log=query_log,
                session_token=session_token,
                api_key=api_key
            )
            
            if result:
                logger.info("QueryLog saved successfully to Parse Server")
                return result
            else:
                logger.error("Failed to save QueryLog to Parse Server")
                return None
                    
        except Exception as e:
            logger.error(f"Error saving QueryLog to Parse Server: {e}", exc_info=True)
            return None

    async def prepare_and_create_query_log_background(
        self,
        query: str,
        search_request: "SearchRequest",
        metadata: "MemoryMetadata",
        resolved_user_id: str,
        workspace_id: str,
        memory_items: List["Memory"],
        neo_nodes: List["NeoNode"],
        retrieval_latency_ms: float,
        search_start_time: float,
        session_token: str,
        api_key: Optional[str],
        client_type: str,
        chat_gpt,
        search_id: str = None  # NEW: accept search_id
    ) -> Optional[str]:
        """
        Background method to prepare and create QueryLog entry
        This method handles all the logging preparation that was previously done synchronously
        """
        try:
            # Calculate response size in KB (for logging only)
            memory_items_size = len(json.dumps(
                [item.model_dump() for item in memory_items],
                default=self._serialize_datetime
            ).encode('utf-8')) / 1024
            
            neo_nodes_size = len(json.dumps(
                [node.model_dump() for node in neo_nodes],
                default=self._serialize_datetime
            ).encode('utf-8')) / 1024
            
            total_size = memory_items_size + neo_nodes_size

            logger.info(f"length of memory_items: {len(memory_items)} ({memory_items_size:.2f} KB)")
            logger.info(f"length of neo_nodes: {len(neo_nodes)} ({neo_nodes_size:.2f} KB)")
            logger.info(f"Total response size: {total_size:.2f} KB")

            # Calculate token metrics
            from services.token_utils import count_query_embedding_tokens, count_retrieved_memory_tokens, count_neo_nodes_tokens
            query_embedding_tokens = count_query_embedding_tokens(query)
            retrieved_memory_tokens = count_retrieved_memory_tokens([item.model_dump() for item in memory_items])
            neo_nodes_tokens = count_neo_nodes_tokens([node.model_dump() for node in neo_nodes])
            total_retrieved_tokens = retrieved_memory_tokens + neo_nodes_tokens

            # Determine tier sequence based on search options
            tier_sequence = [2]  # Default to tier 2 (cloud search)
            if search_request.enable_agentic_graph:
                tier_sequence.append(3)  # Add tier 3 (agentic graph)

            # Prepare QueryLog data for background task
            search_request_metadata = metadata.model_dump(exclude_none=True) if metadata else {}
            search_options = {
                'rank_results': search_request.rank_results,
                'enable_agentic_graph': search_request.enable_agentic_graph
            }
            performance_metrics = {
                'retrieval_latency_ms': retrieval_latency_ms
            }
            token_metrics = {
                'query_embedding_tokens': query_embedding_tokens,
                'retrieved_memory_tokens': total_retrieved_tokens
            }

            # Create QueryLog object
            user_pointer = ParsePointer(
                objectId=resolved_user_id,
                className="_User"
            )
            
            workspace_pointer = ParsePointer(
                objectId=workspace_id,
                className="WorkSpace"
            )
            
            # Handle optional pointers from metadata
            post_pointer = None
            if search_request_metadata.get('post'):
                post_pointer = ParsePointer(
                    objectId=search_request_metadata['post'],
                    className="Post"
                )
            
            user_message_pointer = None
            if search_request_metadata.get('userMessage'):
                user_message_pointer = ParsePointer(
                    objectId=search_request_metadata['userMessage'],
                    className="PostMessage"
                )
            
            assistant_message_pointer = None
            if search_request_metadata.get('assistantMessage'):
                assistant_message_pointer = ParsePointer(
                    objectId=search_request_metadata['assistantMessage'],
                    className="PostMessage"
                )
            
            # Handle goal/use case/step pointers from metadata (if provided)
            related_goals = []
            if search_request_metadata.get('relatedGoals'):
                for goal_id in search_request_metadata['relatedGoals']:
                    related_goals.append(ParsePointer(objectId=goal_id, className="Goal"))
            
            related_use_cases = []
            if search_request_metadata.get('relatedUseCases'):
                for use_case_id in search_request_metadata['relatedUseCases']:
                    related_use_cases.append(ParsePointer(objectId=use_case_id, className="Usecase"))
            
            related_steps = []
            if search_request_metadata.get('relatedSteps'):
                for step_id in search_request_metadata['relatedSteps']:
                    related_steps.append(ParsePointer(objectId=step_id, className="Step"))

            # Calculate total processing time
            total_processing_time_ms = (time.time() - search_start_time) * 1000

            query_log = QueryLog(
                user=user_pointer,
                workspace=workspace_pointer,
                post=post_pointer,
                userMessage=user_message_pointer,
                assistantMessage=assistant_message_pointer,
                sessionId=search_request_metadata.get('sessionId'),
                queryText=query,
                relatedGoals=related_goals,
                relatedUseCases=related_use_cases,
                relatedSteps=related_steps,
                # Don't set classification scores here - they will be populated during classification
                goalClassificationScores=[],
                useCaseClassificationScores=[],
                stepClassificationScores=[],
                rankingEnabled=search_options.get('rank_results'),
                enabledAgenticGraph=search_options.get('enable_agentic_graph'),
                tierSequence=tier_sequence,
                retrievalLatencyMs=performance_metrics.get('retrieval_latency_ms'),
                queryEmbeddingTokens=token_metrics.get('query_embedding_tokens'),
                retrievedMemoryTokens=token_metrics.get('retrieved_memory_tokens'),
                totalProcessingTimeMs=total_processing_time_ms,
                apiVersion='v1',
                infrastructureRegion=env.get('INFRASTRUCTURE_REGION', 'us-east-1'),
                objectId=search_id if search_id else None  # NEW: set objectId if provided
            )

            # Now call the query log service to handle classification and storage
            query_log_id = await self.create_query_log_background(
                query_log=query_log,
                session_token=session_token,
                api_key=api_key,
                client_type=client_type,
                chat_gpt=chat_gpt,
                search_id=search_id  # pass search_id
            )
            
            return query_log_id

        except Exception as e:
            logger.error(f"Error in background query log preparation: {e}", exc_info=True)
            return None

    def _serialize_datetime(self, obj):
        """Helper method to serialize datetime objects"""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    async def create_query_and_retrieval_logs_background(
        self,
        query: str,
        search_request: "SearchRequest",
        metadata: "MemoryMetadata",
        resolved_user_id: str,
        workspace_id: str,
        relevant_items: "RelatedMemoryResult",
        retrieval_latency_ms: float,
        search_start_time: float,
        session_token: str,
        api_key: Optional[str],
        client_type: str,
        chat_gpt,
        search_id: str = None  # NEW: accept search_id
    ) -> None:
        """
        Create both QueryLog and MemoryRetrievalLog in background, using full RelatedMemoryResult for accurate relations and scores.
        """
        from models.parse_server import ParseStoredMemory, MemoryRetrievalLog, QueryLog, ParsePointer
        from services.memory_management import store_memory_retrieval_log_async
        import traceback

        try:
            # --- First, create the QueryLog ---
            query_log_id = await self.prepare_and_create_query_log_background(
                query=query,
                search_request=search_request,
                metadata=metadata,
                resolved_user_id=resolved_user_id,
                workspace_id=workspace_id,
                memory_items=relevant_items.memory_items,
                neo_nodes=relevant_items.neo_nodes,
                retrieval_latency_ms=retrieval_latency_ms,
                search_start_time=search_start_time,
                session_token=session_token,
                api_key=api_key,
                client_type=client_type,
                chat_gpt=chat_gpt,
                search_id=search_id  # pass search_id
            )
            
            if not query_log_id:
                logger.error("Failed to create QueryLog, cannot create MemoryRetrievalLog")
                return

            # --- Build relations and scores for MemoryRetrievalLog ---
            logger.info(f"[QueryLog] Number of memory_items received: {len(relevant_items.memory_items)}")
            retrieved_memories = []
            predicted_grouped_memories = []
            retrieved_memory_scores = []
            retrieved_memory_similarity_scores = {}
            retrieved_memory_confidence_scores = {}

            # Map memoryId to source info for quick lookup
            memory_id_to_source = {}
            if relevant_items.memory_source_info:
                for src in relevant_items.memory_source_info.memory_id_source_location:
                    memory_id_to_source[src.memory_id] = src.source_location

            # Try to get original similarity scores if available (assume relevant_items has .similarity_scores_by_id)
            similarity_scores_by_id = getattr(relevant_items, 'similarity_scores_by_id', {})
            bigbird_memory_info = getattr(relevant_items, 'bigbird_memory_info', [])

            # Log the source of each memory item
            logger.info(f"Relevant items: {relevant_items}")
            logger.info(f"Memory source info: {relevant_items.memory_source_info}")
            logger.info(f"Similarity scores by id: {similarity_scores_by_id}")
            logger.info(f"BigBird memory info: {bigbird_memory_info}")
            logger.info(f"Number of memory items: {len(relevant_items.memory_items) if relevant_items.memory_items else 0}")
            logger.info(f"Number of similarity scores: {len(similarity_scores_by_id)}")
            logger.info(f"Number of BigBird items: {len(bigbird_memory_info)}")
            
            # Determine which retrieval tiers were used - with fallback logic
            used_qdrant = False
            used_qdrant_grouped = False
            used_neo = False
            
            # Track which memories came from grouped sources for predictedGroupedMemories
            grouped_memory_ids = set()
            
            # BigBird memories (legacy route): ALL are grouped by default
            for item in bigbird_memory_info:
                bigbird_id = item.get('id')
                if bigbird_id:
                    grouped_memory_ids.add(bigbird_id)
                    used_qdrant_grouped = True
                    logger.info(f"BigBird memory {bigbird_id} treated as grouped (legacy route)")
            
            # Method 1: Use memory_source_info if available for non-legacy detection
            if relevant_items.memory_source_info:
                logger.info("Using memory_source_info for source detection")
                for src in relevant_items.memory_source_info.memory_id_source_location:
                    if src.source_location.in_qdrant:
                        used_qdrant = True
                    if src.source_location.in_qdrant_grouped:
                        used_qdrant_grouped = True
                        # Strip chunk suffix to get base memory ID for grouping
                        base_memory_id = src.memory_id.split('_')[0] if '_' in src.memory_id else src.memory_id
                        grouped_memory_ids.add(base_memory_id)
                        logger.info(f"Memory {src.memory_id} (base: {base_memory_id}) detected as grouped Qdrant from memory_source_info")
                    if src.source_location.in_neo:
                        used_neo = True
                        
                # Log the source of each memory item
                for memory in relevant_items.memory_items:
                    memory_id = getattr(memory, 'memoryId', None)
                    if memory_id:
                        src_info = None
                        for src in relevant_items.memory_source_info.memory_id_source_location:
                            if src.memory_id == memory_id:
                                src_info = src.source_location
                                break
                        logger.info(f"Memory {memory_id} sources: "
                                    f"Qdrant={getattr(src_info, 'in_qdrant', False)}, "
                                    f"QdrantGrouped={getattr(src_info, 'in_qdrant_grouped', False)}, "
                                    f"Neo4j={getattr(src_info, 'in_neo', False)}")
            
            # Method 2: Fallback - detect from similarity scores and qdrant grouped info
            if not used_qdrant and not used_qdrant_grouped and not used_neo:
                logger.info("Using fallback source detection from similarity scores and Qdrant grouped info")
                
                # Check if we have Qdrant results (lower similarity scores typically)
                qdrant_ids = set()
                
                for memory in relevant_items.memory_items:
                    memory_id = getattr(memory, 'memoryId', None)
                    if memory_id:
                        # Try to find similarity score for this memory
                        # First try the exact memory_id
                        sim_score = similarity_scores_by_id.get(memory_id)
                        
                        # If not found, try to find any chunk ID that starts with this memory_id
                        if sim_score is None:
                            for chunk_id, score in similarity_scores_by_id.items():
                                if chunk_id.startswith(memory_id + '_'):
                                    sim_score = score
                                    logger.info(f"Found similarity score for memory {memory_id} via chunk ID {chunk_id}: {score}")
                                    break
                        
                        if sim_score is not None:
                            # Check if this memory is in grouped Qdrant results
                            in_qdrant_grouped = any(item.get('id') == memory_id and item.get('metadata', {}).get('isGroupedMemories', False) for item in bigbird_memory_info)
                            if in_qdrant_grouped:
                                grouped_memory_ids.add(memory_id)
                                used_qdrant_grouped = True
                                logger.info(f"Memory {memory_id} detected as grouped Qdrant (score: {sim_score})")
                            else:
                                qdrant_ids.add(memory_id)
                                used_qdrant = True
                                logger.info(f"Memory {memory_id} detected as regular Qdrant (score: {sim_score})")
                        else:
                            logger.warning(f"No similarity score found for memory {memory_id}")
            
            memory_retrieval_tiers = []
            if used_qdrant:
                memory_retrieval_tiers.append(2)  # Example: 2 = Qdrant
            if used_qdrant_grouped:
                memory_retrieval_tiers.append(3)  # Example: 3 = Qdrant Grouped
            if used_neo:
                memory_retrieval_tiers.append(4)  # Example: 4 = Neo4j
            used_predicted_grouping = used_qdrant_grouped  # Set True if grouped Qdrant was used
            logger.info(f"Used predicted grouping: {used_predicted_grouping}")
            logger.info(f"Memory retrieval tiers: {memory_retrieval_tiers}")
            logger.info(f"Used qdrant: {used_qdrant}")
            logger.info(f"Used qdrant_grouped: {used_qdrant_grouped}")
            logger.info(f"Used neo: {used_neo}")
            logger.info(f"Grouped memory IDs: {grouped_memory_ids}")


            for idx, memory in enumerate(relevant_items.memory_items):
                # Ensure correct structure
                if not isinstance(memory, ParseStoredMemory):
                    memory = ParseStoredMemory.from_dict(memory)
                pointer = {
                    "__type": "Pointer",
                    "className": "Memory",
                    "objectId": memory.objectId
                }
                
                # Determine source and add to appropriate list
                memory_id = getattr(memory, 'memoryId', None)
                # Always add to retrievedMemories (full set)
                retrieved_memories.append(pointer)
                logger.info(f"Added memory {memory_id} to retrievedMemories (all retrieved)")
                # Additionally mark grouped memories
                if memory_id and memory_id in grouped_memory_ids:
                    predicted_grouped_memories.append(pointer)
                    logger.info(f"Also added memory {memory_id} to predictedGroupedMemories (grouped)")
                
                # Similarity score (from retrieval)
                sim_score = similarity_scores_by_id.get(memory.memoryId, None)
                
                # If not found, try to find any chunk ID that starts with this memory_id
                if sim_score is None:
                    for chunk_id, score in similarity_scores_by_id.items():
                        if chunk_id.startswith(memory.memoryId + '_'):
                            sim_score = score
                            logger.info(f"Found similarity score for memory {memory.memoryId} via chunk ID {chunk_id}: {score}")
                            break
                
                # Similarity score stored in retrievedMemorySimilarityScores and retrievedMemoryScores (quality)
                if sim_score is None:
                    sim_score = 0.0
                # Use Memory objectId as the key for downstream counter updates
                retrieved_memory_similarity_scores[memory.objectId] = sim_score
                retrieved_memory_scores.append(sim_score)
                # Confidence score from reranking (if available)
                conf_score = (
                    relevant_items.confidence_scores[idx]
                    if relevant_items.confidence_scores and idx < len(relevant_items.confidence_scores)
                    else 0.5
                )
                retrieved_memory_confidence_scores[memory.objectId] = conf_score

            logger.info(f"[QueryLog] Number of retrieved_memories to log: {len(retrieved_memories)}")
            logger.info(f"[QueryLog] Number of predicted_grouped_memories to log: {len(predicted_grouped_memories)}")
            logger.info(f"Final predictedGroupedMemories count: {len(predicted_grouped_memories)}")
            logger.info(f"Final retrievedMemories count: {len(retrieved_memories)}")

            # --- Build MemoryRetrievalLog ---
            # Create proper ParsePointer objects
            user_pointer = ParsePointer(
                objectId=resolved_user_id,
                className="_User"
            )
            
            workspace_pointer = ParsePointer(
                objectId=workspace_id,
                className="WorkSpace"
            )
            
            # Create QueryLog pointer using the ID we just created
            query_log_pointer = ParsePointer(
                objectId=query_log_id,
                className="QueryLog"
            )
            
            retrieval_log = MemoryRetrievalLog(
                user=user_pointer,
                workspace=workspace_pointer,
                queryLog=query_log_pointer,
                totalMemoriesRetrieved=len(relevant_items.memory_items),
                retrievedMemories=retrieved_memories,
                retrievedMemoryScores=retrieved_memory_scores,
                retrievedMemorySimilarityScores=retrieved_memory_similarity_scores,
                retrievedMemoryConfidenceScores=retrieved_memory_confidence_scores,
                retrievalLatencyMs=retrieval_latency_ms,
                memoryRetrievalTiers=memory_retrieval_tiers,
                # Note: usedPredictedGrouping and predictionModelUsed will be set later when we know if the answer came from a predicted group
                usedPredictedGrouping=None,  # Will be set when answer is generated and we know if it used predicted groups
                predictedGroupedMemories=predicted_grouped_memories,
                groupedMemoriesDistribution=(
                    round(
                        (len(predicted_grouped_memories) / max(1, len(relevant_items.memory_items))),
                        4,
                    ) if len(predicted_grouped_memories) > 0 else 0.0
                ),
                predictionModelUsed=None  # Will be set when answer is generated and we know which model was used
            )
            
            # Store MemoryRetrievalLog in Parse
            await store_memory_retrieval_log_async(
                memory_retrieval_log=retrieval_log,
                session_token=session_token,
                api_key=api_key
            )

            # Update Memory counters (cache/citation) incrementally
            try:
                from services.memory_management import update_memory_counters_from_retrieval_async
                # Pointers above are dicts with keys: __type, className, objectId. Support both dicts and objects.
                def _extract_object_id(ptr):
                    if isinstance(ptr, dict):
                        return ptr.get("objectId")
                    return getattr(ptr, "objectId", None)

                retrieved_ids = [mid for mid in (_extract_object_id(p) for p in retrieved_memories) if mid]
                cited_ids = [mid for mid in (_extract_object_id(p) for p in predicted_grouped_memories) if mid]
                await update_memory_counters_from_retrieval_async(
                    retrieved_ids=retrieved_ids,
                    retrieved_similarity_scores=retrieved_memory_similarity_scores,
                    retrieved_confidence_scores=retrieved_memory_confidence_scores,
                    cited_ids=cited_ids,
                    cited_confidence_scores=None,
                    session_token=session_token,
                    api_key=api_key,
                    half_life_days=30.0,
                    retrieval_latency_ms=retrieval_latency_ms,
                    token_counts=None,
                    tier_confidences=None,
                    engagement_scores=None,
                )
            except Exception as up_err:
                logger.error(f"Failed to update memory counters from retrieval log: {up_err}")
            
            logger.info(f"Successfully created both QueryLog ({query_log_id}) and MemoryRetrievalLog")
            
        except Exception as e:
            logger.error(f"Error in create_query_and_retrieval_logs_background: {e}\n{traceback.format_exc()}")

    async def create_memory_retrieval_log_background(
        self,
        query: str,
        search_request: "SearchRequest",
        metadata: "MemoryMetadata",
        resolved_user_id: str,
        workspace_id: str,
        memory_items: List["Memory"],
        neo_nodes: List["NeoNode"],
        retrieval_latency_ms: float,
        search_start_time: float,
        session_token: str,
        api_key: Optional[str],
        client_type: str,
        query_log_id: Optional[str] = None
    ) -> None:
        """
        Background method to create MemoryRetrievalLog entry
        This method handles the memory retrieval logging that tracks which memories were retrieved
        """
        try:
            # Calculate response size in KB (for logging only)
            memory_items_size = len(json.dumps(
                [item.model_dump() for item in memory_items],
                default=self._serialize_datetime
            ).encode('utf-8')) / 1024
            
            neo_nodes_size = len(json.dumps(
                [node.model_dump() for node in neo_nodes],
                default=self._serialize_datetime
            ).encode('utf-8')) / 1024
            
            total_size = memory_items_size + neo_nodes_size

            logger.info(f"MemoryRetrievalLog - length of memory_items: {len(memory_items)} ({memory_items_size:.2f} KB)")
            logger.info(f"MemoryRetrievalLog - length of neo_nodes: {len(neo_nodes)} ({neo_nodes_size:.2f} KB)")
            logger.info(f"MemoryRetrievalLog - Total response size: {total_size:.2f} KB")

            # Prepare metadata for MemoryRetrievalLog
            search_request_metadata = metadata.model_dump(exclude_none=True) if metadata else {}
            
            # Create pointers
            user_pointer = ParsePointer(
                objectId=resolved_user_id,
                className="_User"
            )
            
            workspace_pointer = ParsePointer(
                objectId=workspace_id,
                className="WorkSpace"
            )
            
            # Handle optional pointers from metadata
            post_pointer = None
            if search_request_metadata.get('post'):
                post_pointer = ParsePointer(
                    objectId=search_request_metadata['post'],
                    className="Post"
                )
            
            # Create QueryLog pointer if we have the ID
            query_log_pointer = None
            if query_log_id:
                query_log_pointer = ParsePointer(
                    objectId=query_log_id,
                    className="QueryLog"
                )
            
            # Convert memory items to ParsePointers
            retrieved_memories = []
            for memory in memory_items:
                if hasattr(memory, 'memoryId') and memory.memoryId:
                    retrieved_memories.append(ParsePointer(
                        objectId=memory.memoryId,
                        className="Memory"
                    ))
            
            # For now, we'll set cited_memories to empty (this would be populated later when memories are actually cited)
            cited_memories = []
            
            # Calculate memory retrieval scores (for now, we'll use a simple scoring)
            retrieved_memory_scores = [0.8] * len(retrieved_memories)  # Placeholder scores
            
            # Determine which tiers were used for retrieval
            memory_retrieval_tiers = []
            if search_request.enable_agentic_graph:
                memory_retrieval_tiers = [2, 3]  # Cloud search + agentic graph
            else:
                memory_retrieval_tiers = [2]  # Cloud search only
            
            # Calculate total processing time
            total_processing_time_ms = (time.time() - search_start_time) * 1000

            # Create MemoryRetrievalLog object
            memory_retrieval_log = MemoryRetrievalLog(
                user=user_pointer,
                workspace=workspace_pointer,
                post=post_pointer,
                sessionId=search_request_metadata.get('sessionId'),
                queryLog=query_log_pointer,
                retrievedMemories=retrieved_memories,
                citedMemories=cited_memories,
                retrievedMemoryScores=retrieved_memory_scores,
                memoryRetrievalTiers=memory_retrieval_tiers,
                totalMemoriesRetrieved=len(retrieved_memories),
                totalMemoriesCited=len(cited_memories),
                retrievalLatencyMs=retrieval_latency_ms,
                embeddingLatencyMs=None,  # Not tracked yet
                usedPredictedGrouping=False,  # Not implemented yet
                predictedGroupedMemories=[],  # Not implemented yet
                predictionAccuracyScore=None,  # Not implemented yet
                predictionModelUsed=None  # Not implemented yet
            )

            # Store MemoryRetrievalLog in Parse Server
            await self._save_memory_retrieval_log_to_parse(
                memory_retrieval_log=memory_retrieval_log,
                session_token=session_token,
                api_key=api_key,
                client_type=client_type
            )
            
            logger.info(f"MemoryRetrievalLog created successfully for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error creating MemoryRetrievalLog in background: {e}", exc_info=True)

    async def _save_memory_retrieval_log_to_parse(
        self,
        memory_retrieval_log: "MemoryRetrievalLog",
        session_token: Optional[str],
        api_key: Optional[str],
        client_type: Optional[str]
    ) -> None:
        """Save MemoryRetrievalLog to Parse Server"""
        try:
            # Use the memory_management function to store the MemoryRetrievalLog
            from services.memory_management import store_memory_retrieval_log_async
            
            result = await store_memory_retrieval_log_async(
                memory_retrieval_log=memory_retrieval_log,
                session_token=session_token,
                api_key=api_key
            )
            
            if result:
                logger.info("MemoryRetrievalLog saved successfully to Parse Server")
            else:
                logger.error("Failed to save MemoryRetrievalLog to Parse Server")
                    
        except Exception as e:
            logger.error(f"Error saving MemoryRetrievalLog to Parse Server: {e}", exc_info=True)

# Global instance
query_log_service = QueryLogService() 