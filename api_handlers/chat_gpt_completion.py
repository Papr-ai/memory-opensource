import os
from enum import Enum
from tiktoken import encoding_for_model
import tiktoken
import json
import re
from openai import OpenAI, AsyncOpenAI
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from uuid import uuid4
from typing import Dict, List, Tuple
from models.structured_outputs import (
    UseCaseMemoryItem, MemoryGraphSchema
)
from models.shared_types import NodeLabel, RelationshipType
from models.acl import ACLFilter
# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import instructor
from pydantic import BaseModel, Field
from typing import List, Literal, TYPE_CHECKING, Dict, Any, Optional, TypedDict, Union
from models.structured_outputs import ModelFindRealtedmemmories

if TYPE_CHECKING:
    from memory.memory_graph import MemoryGraph

# Import for runtime use
try:
    from memory.memory_graph import MemoryGraph as _MemoryGraph
    MemoryGraph = _MemoryGraph
except ImportError:
    # Handle potential circular import
    MemoryGraph = None

from models.cipher_ast import CypherQuery, MatchClause, PatternElement, CipherNode, Edge, NodeAlias, NODE_PROPERTY_MAP
from models.memory_models import RelatedMemoryResult, ContextItem, RerankingConfig
from models.parse_server import UseCaseMetrics, UseCaseResponse, RelatedMemoriesMetrics, RelatedMemoriesSuccess, RelatedMemoriesError, ParseStoredMemory
from services.logging_config import get_logger
from services.logger_singleton import LoggerSingleton
import groq
# Create a logger instance for this module
logger = LoggerSingleton.get_logger(__name__)
import ollama
from typing import TYPE_CHECKING
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import get_args, get_origin, Union
from dotenv import load_dotenv, find_dotenv

if TYPE_CHECKING:
    from services.user_utils import User

# Load environment variables before using them anywhere in this module (conditionally based on USE_DOTENV)
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

def get_allowed_properties_table(node_labels):
    lines = [
        "🚧 ABSOLUTE PROPERTY RULE 🚧",
        "For each node label you may reference ONLY the properties listed below.",
        "If a property is not listed for that label, you must NOT use it.",
        ""
    ]
    for label in node_labels:
        model = NODE_PROPERTY_MAP.get(label)
        if model:
            prop_list = []
            for name, field in model.model_fields.items():
                allowed = None
                ann = field.annotation
                # Unwrap Optional/Union
                if get_origin(ann) is Union:
                    args = [a for a in get_args(ann) if a is not type(None)]
                    if args:
                        ann = args[0]
                if get_origin(ann) is Literal:
                    allowed = "|".join(str(v) for v in get_args(ann))
                elif isinstance(ann, type) and issubclass(ann, Enum):
                    allowed = "|".join([e.value for e in ann])
                if allowed:
                    prop_list.append(f"{name} ({allowed})")
                else:
                    prop_list.append(name)
            
            label_str = label.value if hasattr(label, 'value') else str(label)
            lines.append(f"• {label_str:<15}: {', '.join(prop_list)}")
            logger.info(f"🔧 PROPERTIES: Added properties for {label_str}: {prop_list}")
    return "\n".join(lines)

class ChatGPTCompletion:
    default_model = os.environ.get("LLM_MODEL", "gpt-4.1-nano")
    model_location_cloud = True 
    def __init__(self, api_key, organization_id=None, model=None, model_location_cloud=None, embedding_model= None):
        self.model = model if model else self.default_model  # Use the provided model or the default
        self.model_location_cloud = model_location_cloud if model_location_cloud else self.model_location_cloud
        self._memory_graph = None  # Initialize as None
        # Dedicated higher-quality model for schema/search operations
        self.model_mini = os.environ.get("LLM_MODEL_MINI", "gpt-5-mini")
    
        if (self.model_location_cloud == False):
            logger.info("Applying local settigns ")    
            self.client = ChatOllama(model=model)
            #self.client = Ollama(model=model)
            if embedding_model == "":
                logger.info ("empty embedding model")
            self.embedding_model =embedding_model if embedding_model else "text-embedding-3-small"
            self.embeddingclient = OllamaEmbeddings(model=self.embedding_model)            
            self.cost_per_input_token  = 0  # cost for local
            self.cost_per_output_token = 0  # cost for local
            self.OpenAIAPIOllama = OpenAI(
                       base_url = 'http://localhost:11434/v1',
                        api_key='ollama', # required, but unused
                        )
            self.modelfunctions = OllamaFunctions(model=model, format="json", response_format={ "type": "json_object" })
            self.clientinstructor = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
            )
        else:       
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            openai_organization = os.environ.get("OPENAI_ORGANIZATION")
            self.client = OpenAI(api_key=openai_api_key, organization=openai_organization)
            self.async_client = AsyncOpenAI(api_key=openai_api_key, organization=openai_organization)
            self.cost_per_input_token  = 0.0000001000  # cost for gpt-4o-mini
            self.cost_per_output_token = 0.0000004000  # cost for gpt-4o-mini
            self.groq_client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self.groq_async_client = groq.AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
            self.instructor_groq_client = instructor.from_groq(self.groq_async_client)
            # Groq model for pattern selection (using larger OSS model for better accuracy)
            self.groq_pattern_selector_model = os.environ.get("GROQ_PATTERN_SELECTOR_MODEL", "openai/gpt-oss-20b")
            
            # Initialize Gemini client for fallback (Gemini 2.5 Flash)
            self.gemini_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            self.gemini_model = os.environ.get("GEMINI_MODEL_FAST", "gemini-2.5-flash")

    async def _create_completion_async(self, **kwargs):
        """Async wrapper for creating chat completions"""
        # Normalize kwargs for model-specific compatibility (e.g., gpt-5, o-series)
        normalized_kwargs = self._normalize_chat_kwargs(kwargs)
        return await self.async_client.chat.completions.create(**normalized_kwargs)

    async def _create_completion_with_fallback_async(self, **kwargs):
        """Async wrapper with fallback handling for creating chat completions"""
        # Normalize kwargs for model-specific compatibility (e.g., gpt-5, o-series)
        normalized_kwargs = self._normalize_chat_kwargs(kwargs)
        return await self.async_client.chat.completions.create(**normalized_kwargs)

    def _create_completion_sync(self, **kwargs):
        """Sync wrapper for creating chat completions"""
        # Normalize kwargs for model-specific compatibility (e.g., gpt-5, o-series)
        normalized_kwargs = self._normalize_chat_kwargs(kwargs)
        return self.client.chat.completions.create(**normalized_kwargs)

    def _normalize_chat_kwargs(self, kwargs: dict) -> dict:
        """Normalize chat.completions kwargs for models with different parameter support.

        - For o-series and gpt-5 models, drop unsupported temperature and map
          max_tokens -> max_completion_tokens.
        """
        try:
            model_name = (kwargs.get("model") or getattr(self, "model", "") or "").lower()
        except Exception:
            model_name = (getattr(self, "model", "") or "").lower()

        is_o_series = model_name.startswith("o")
        is_gpt5 = model_name.startswith("gpt-5")

        # Create a shallow copy to avoid mutating caller dict
        out = dict(kwargs)

        # Temperature: some models (o-series, gpt-5-nano) only accept default, reject overrides
        if (is_o_series or is_gpt5) and "temperature" in out:
            out.pop("temperature", None)

        # Token param migration: max_tokens -> max_completion_tokens for o-series/gpt-5
        if (is_o_series or is_gpt5) and "max_tokens" in out:
            if "max_completion_tokens" not in out:
                out["max_completion_tokens"] = out.pop("max_tokens")
            else:
                out.pop("max_tokens", None)

        return out
    
    async def _call_gemini_structured_async(self, messages: List[Dict[str, str]], response_model: BaseModel) -> BaseModel:
        """
        Call Gemini API with structured output (JSON schema enforcement).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model defining the expected response structure
            
        Returns:
            Parsed Pydantic model instance
            
        Raises:
            Exception: If API call fails or response doesn't match schema
        """
        import httpx
        
        if not self.gemini_api_key:
            raise Exception("GEMINI_API_KEY not configured")
        
        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        # Get JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        
        # Build Gemini request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
                "responseSchema": schema
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{url}?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                
                if 'content' not in candidate:
                    raise Exception(f"Gemini response missing 'content' key. Full response: {result}")
                
                if 'parts' not in candidate['content']:
                    raise Exception(f"Gemini response missing 'parts' key. Content: {candidate['content']}")
                
                if len(candidate['content']['parts']) == 0:
                    raise Exception(f"Gemini response has empty 'parts' array. Content: {candidate['content']}")
                
                content_text = candidate['content']['parts'][0]['text']
                
                # Parse JSON response into Pydantic model
                import json
                parsed_json = json.loads(content_text)
                return response_model(**parsed_json)
            else:
                raise Exception(f"No valid response from Gemini: {result}") 

    @property
    def memory_graph(self):
        if self._memory_graph is None:
            from memory.memory_graph import MemoryGraph
            self._memory_graph = MemoryGraph()
        return self._memory_graph

    @staticmethod
    def get_tokenizer(model_name):
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            logger.warning(f"WARNING: Unknown model for tiktoken: {model_name}, using cl100k_base as fallback.")
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using tiktoken."""
        if (self.model_location_cloud == False):
            response = ollama.embeddings(model=self.embedding_model, prompt=text)            
            #response = self.embeddingclient.embed_documents(text)
            return len(response['embedding'])
        else:        
            encoding = self.get_tokenizer(self.model)  # Assuming you're using GPT-4
            return len(encoding.encode(text))
        
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the total cost based on input and output token counts."""
        if (self.model_location_cloud == False):
            return 0.000000
        else:
            input_cost = input_tokens * self.cost_per_input_token
            output_cost = output_tokens * self.cost_per_output_token
            total_cost = input_cost + output_cost
            
            # Ensure we return at least 6 decimal places for small numbers
            if total_cost < 0.000001:
                return format(total_cost, '.8f')
            return total_cost
        
    def generate_image_prompt(self, memory_item_content):
        if (self.model_location_cloud == False):
            logger.error("Function not supported locally yet")
      
        prompt = f"Please generate a DALL E prompt for this memory: {memory_item_content}. The prompt should be concise, simple, less than 27 words and should adhere to DALL E content policies and not violate them."
        
         # Ensure the prompt doesn't exceed the token limit
        if self.count_tokens(prompt) > 8000:
            raise ValueError("The prompt is too long and exceeds the token limit.")
        
        # Update parameters for chat completion API to include user role with content = prompt
        messages = [
            {"role": "system", "content": "You generate concise, simple, less than 27 word prompts for DALLE that adhere to DALLE's content policies"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,  # Assuming you want to use GPT-4 turbo
            messages=messages
        )
        logger.info(f"prompt generated from chatGPT chat completion API: {response}")

        return response.choices[0].message['content'].strip()
    
    def trim_content_to_token_limit(self, content, max_tokens=8000, buffer_tokens=1000):
        """
        Trim content to fit within token limit while maintaining a buffer for system messages and response.
        
        Args:
            content (str|dict|list): Content to trim
            max_tokens (int): Maximum total tokens allowed
            buffer_tokens (int): Buffer to reserve for system messages and response
            
        Returns:
            str|dict|list: Trimmed content that fits within token limit
        """
        if content is None:
            return ""
            
        # Convert content to string if it's not already
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content)
        else:
            content_str = str(content)
            
        content_tokens = self.count_tokens(content_str)
        if content_tokens <= (max_tokens - buffer_tokens):
            return content
            
        # If content is too long, trim it by reducing the size proportionally
        reduction_ratio = (max_tokens - buffer_tokens) / content_tokens
        
        if isinstance(content, (dict, list)):
            # For dict/list, trim the string representation and parse back
            words = content_str.split()
            new_length = int(len(words) * reduction_ratio)
            trimmed_str = " ".join(words[:new_length])
            try:
                return json.loads(trimmed_str)
            except json.JSONDecodeError:
                # If can't parse back to original type, return as string
                return trimmed_str
        else:
            # For strings, just trim the words
            words = content_str.split()
            new_length = int(len(words) * reduction_ratio)
            return " ".join(words[:new_length])

    def estimate_message_tokens(self, messages, model_name=None):
        """
        Roughly estimate token usage for a list of chat messages by tokenizing
        the JSON representation. This is an approximation but sufficient for
        budgeting and trimming purposes.
        """
        try:
            serialized = json.dumps(messages, ensure_ascii=False)
        except Exception:
            # Fall back to string conversion if serialization fails
            serialized = str(messages)
        # Use the provided model name if given, otherwise the instance's model
        original_model = getattr(self, "model", None)
        if model_name is not None:
            try:
                # Temporarily override for accurate tokenizer selection
                self.model = model_name
                return self.count_tokens(serialized)
            finally:
                self.model = original_model
        return self.count_tokens(serialized)

    def trim_messages_to_token_budget(
        self,
        messages,
        max_total_tokens,
        reserve_completion_tokens=0,
        buffer_tokens=0,
        model_name=None,
    ):
        """
        Trim chat messages so the prompt stays within the allowed token budget.

        Args:
            messages (list[dict]): Chat messages with keys like {"role", "content"}
            max_total_tokens (int): Total token budget for prompt + completion
            reserve_completion_tokens (int): Tokens reserved for the model's response
            buffer_tokens (int): Extra safety buffer for system/tool overhead
            model_name (str|None): Optional model name override for tokenization

        Returns:
            list[dict]: A possibly-trimmed list of messages within budget
        """
        if not isinstance(messages, list):
            return messages

        allowed_prompt_tokens = int(max_total_tokens) - int(reserve_completion_tokens) - int(buffer_tokens)
        if allowed_prompt_tokens <= 0:
            # Nothing can fit; return the last system + last user truncated aggressively
            pruned = []
            system_msg = next((m for m in messages if m.get("role") == "system"), None)
            user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if system_msg:
                pruned.append({"role": "system", "content": ""})
            if user_msg:
                pruned.append({"role": "user", "content": ""})
            return pruned if pruned else [{"role": "user", "content": ""}]

        def within_budget(msgs):
            return self.estimate_message_tokens(msgs, model_name=model_name) <= allowed_prompt_tokens

        if within_budget(messages):
            return messages

        # Prefer trimming user messages first, starting from the newest
        trimmed = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        # Helper to trim a single message's content proportionally
        def trim_message_content(msg, target_tokens):
            content = msg.get("content", "")
            # Use existing content trimmer to reduce content to target_tokens
            # We set a small buffer since this is per-message trimming
            trimmed_content = self.trim_content_to_token_limit(
                content,
                max_tokens=max(1, target_tokens),
                buffer_tokens=0,
            )
            msg["content"] = trimmed_content

        # Iteratively trim: newest user messages -> older user messages -> assistant messages
        roles_to_trim_order = ["user", "assistant"]

        # Compute fixed token cost of non-trimmable messages (e.g., system/tool)
        non_trimmable = [m for m in trimmed if m.get("role") not in roles_to_trim_order]
        base_tokens = self.estimate_message_tokens(non_trimmable, model_name=model_name) if non_trimmable else 0

        # Avoid negative/zero remaining budget
        remaining_budget = max(1, allowed_prompt_tokens - base_tokens)

        # Collect trimmable messages in order from newest to oldest within each role
        trimmable = []
        for role in roles_to_trim_order:
            trimmable.extend([m for m in reversed(trimmed) if m.get("role") == role])

        # If nothing to trim, drop oldest messages until within budget, preserving system
        if not trimmable:
            while not within_budget(trimmed) and len(trimmed) > 1:
                # Drop the oldest non-system message
                for i, m in enumerate(trimmed):
                    if m.get("role") != "system":
                        trimmed.pop(i)
                        break
                else:
                    break
            return trimmed

        # Distribute remaining budget across trimmable messages
        estimated_tokens_trimmable = self.estimate_message_tokens(trimmable, model_name=model_name)
        if estimated_tokens_trimmable <= remaining_budget and within_budget(trimmed):
            return trimmed

        # Proportional trimming loop with safety cap
        safety_iters = 0
        while not within_budget(trimmed) and safety_iters < 10 and remaining_budget > 0:
            safety_iters += 1
            # Recompute the tokens for trimmable content each iteration
            trimmable_tokens = self.estimate_message_tokens(trimmable, model_name=model_name)
            if trimmable_tokens <= 0:
                # If somehow zero, break to avoid division by zero
                break
            ratio = min(1.0, remaining_budget / trimmable_tokens)

            for msg in trimmable:
                current_tokens = max(1, self.estimate_message_tokens([msg], model_name=model_name))
                target_tokens = max(1, int(current_tokens * ratio))
                trim_message_content(msg, target_tokens)

        # Final hard cap: if still over, drop oldest non-system messages
        while not within_budget(trimmed) and len(trimmed) > 1:
            for i, m in enumerate(trimmed):
                if m.get("role") != "system":
                    trimmed.pop(i)
                    break
            else:
                break

        return trimmed

    def generate_usecase_memory_item(self, memory_item: dict, context=None, existing_goals=None, existing_use_cases=None, existing_nodes=None, existing_relationships=None):
        """
        Generate goals and use cases using OpenAI's Structured Outputs feature with proper token limit handling.
        """
        try:
            # Prepare the content with token limit handling
            memory_content = self.trim_content_to_token_limit(memory_item['content'], 3000)
            trimmed_goals = self.trim_content_to_token_limit(existing_goals or [], 2000)
            trimmed_use_cases = self.trim_content_to_token_limit(existing_use_cases or [], 2000)

            # Construct minimal prompt
            prompt = (
                f"Memory item: {json.dumps(memory_content)}\n"
                f"Review this memory item and suggest new goals and use cases, or select from existing ones.\n"
                f"For each goal or use case, indicate if it is 'new' or 'existing'.\n"
                f"Existing goals: {json.dumps(trimmed_goals)}\n"
                f"Existing use cases: {json.dumps(trimmed_use_cases)}"
            )

            # Count tokens
            messages = [
                {"role": "system", "content": "You help identify goals and use cases from memory items."},
                {"role": "user", "content": prompt}
            ]
            
            token_count = self.count_tokens(json.dumps(messages))
            logger.info(f"Token count after trimming: {token_count}")

            # Make API call using Structured Outputs
            if self.model_location_cloud:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=UseCaseMemoryItem
                )
                
                # Parse the response into our Pydantic model
                result = completion.choices[0].message.parsed
            else:
                # For local models (Ollama), continue using instructor
                response = self.clientinstructor.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=UseCaseMemoryItem
                )
                result = response

            # Calculate metrics
            output_tokens = self.count_tokens(json.dumps(result.model_dump() if isinstance(result, BaseModel) else result))
            total_cost = self.calculate_cost(token_count, output_tokens)

            return {
                "data": result.model_dump() if isinstance(result, BaseModel) else result,
                "metrics": {
                    "usecase_token_count_input": token_count,
                    "usecase_token_count_output": output_tokens,
                    "usecase_total_cost": total_cost
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_usecase_memory_item: {e}")
            raise

    def generate_memory_graph_schema(self, memory_item, usecase_memory_item, existing_schema=None, workspace_id=None):
        """
        Generate a memory graph schema using OpenAI's Structured Outputs feature.

        Args:
            memory_item (dict): The memory item to create a graph schema for
            context (str, optional): Additional context to inform schema generation
            goal (dict, optional): Related goal information to consider in schema
            use_case (dict, optional): Related use case information to consider in schema
            existing_nodes (list, optional): List of current nodes in the memory graph
            existing_relationships (list, optional): List of current relationships in the memory graph

        Returns:
            dict: Contains two keys:
                - data: Dictionary with 'nodes' and 'relationships' lists defining the schema
                - metrics: Dictionary with token counts and cost information
        """
        try:
            # Trim content to fit within token limits
            memory_content = self.trim_content_to_token_limit(memory_item, 3000)
            usecase_content = self.trim_content_to_token_limit(usecase_memory_item, 2000)
            schema_content = self.trim_content_to_token_limit(existing_schema, 2000) if existing_schema else None

            # Construct the prompt
            prompt = (
                f"Memory item: {json.dumps(memory_content)}\n"
                f"Use case info: {json.dumps(usecase_content)}\n"
                f"Analyze this memory item and suggest a graph schema with nodes and relationships.\n"
                f"For each node and relationship, indicate if it is 'new' or 'existing'.\n"
            )

            if schema_content:
                prompt += f"Existing schema: {json.dumps(schema_content)}"

            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": "You help design memory graph schemas with nodes and relationships."},
                {"role": "user", "content": prompt}
            ]

            # Count input tokens
            token_count = self.count_tokens(json.dumps(messages))
            logger.info(f"Token count after trimming: {token_count}")

            # Make API call using Structured Outputs
            if self.model_location_cloud:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=MemoryGraphSchema
                )
                
                # Parse the response into our Pydantic model
                result = completion.choices[0].message.parsed
            else:
                # For local models (Ollama), continue using instructor
                response = self.clientinstructor.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=MemoryGraphSchema
                )
                result = response

            # Calculate metrics
            output_tokens = self.count_tokens(json.dumps(result.model_dump() if isinstance(result, BaseModel) else result))
            total_cost = self.calculate_cost(token_count, output_tokens)

            return {
                "data": result.model_dump() if isinstance(result, BaseModel) else result,
                "metrics": {
                    "schema_token_count_input": token_count,
                    "schema_token_count_output": output_tokens,
                    "schema_total_cost": total_cost
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_memory_graph_schema: {e}")
            raise

    def generate_related_memories_old(self, memory_graph, memory_item, user_id, goal=None, use_case=None):
        """
        Generates queries to find related memories and then uses those queries to find and construct relationships
        with related memory items.
        """
        # Construct the queries
        queries = {
            "queries": [
                {
                "query": "Find memories related to the `handleChatStreamChunk` function to assess previous feedback, implementations, or identified issues."
                },
                {
                "query": "Retrieve memories of past code review sessions that involved `handleChatStreamChunk` or similar functions to understand the context and feedback provided."
                },
                {
                "query": "Search for memories related to technical debt items, specifically those mentioned in connection with the `handleChatStreamChunk` function or similar areas, to track progress and resolution efforts."
                }
            ]
        }


        queries_example = json.dumps(queries)

        # Constructing the prompt
        prompt = f"Given this memory graph in JSON format: {json.dumps(memory_graph)}, and this list of Goals: {json.dumps(goal)} and list of use_cases: {json.dumps(use_case)} create a query to find related memories that we need to build a Neo4js relationship with this memory item {json.dumps(memory_item)}. Format your response as JSON format using this example: {queries_example}"

        # Ensure the prompt doesn't exceed the token limit
        if self.count_tokens(prompt) > 8000:
            raise ValueError("The prompt is too long and exceeds the token limit.")

        # Sending the prompt to the Chat Completion API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )

        # logger the response
        #logger.info(f"Response from OpenAI API: {response}")

        # Extracting and verifying the content from the response
        try:
            memories_queries = json.loads(response.choices[0].message.content)
            return memories_queries
        except json.JSONDecodeError:
            raise ValueError("The response from the API is not in valid JSON format.")
   
    def generate_related_memories(self, session_token: str, memory_graph, memory_item, user_id: str, goal=None, use_case=None, metadata=None):
        """
        Uses ChatGPT to define queries for finding related memories and then executes these queries
        to construct relationships.
        """

        GET_MEMORY_COST = 0.0015164  # Fixed cost includes sentence-bert, big-bird embedding costs and pinecone retreive cost

        # Define the function for the API to call
        find_memory = {
            "type": "function",
            "function": {
                "name": "find_related_memories",
                "description": "Runs multiple queries to find related memories based on the provided queries list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "List of queries that we need to run to get related memories for the user. Up to 3 queries",
                            "items": {
                                "type": "string",
                                "description": "Query that we can use to find a related memory.",
                            },
                        }
                    },
                    "required": [],
                },
            },
        }


        # Constructing the prompt to generate queries
        prompt = f"Given this memory graph in JSON format: {json.dumps(memory_graph)}, and this list of Goals: {json.dumps(goal)} and list of use_cases: {json.dumps(use_case)}, create detailed queries made up of two sentences to find related memories that we need to build a Neo4js relationship with this memory item {json.dumps(memory_item)}. Exclude queries that would generate an actual result for this specific memory item since are just adding it now {json.dumps(memory_item)}."

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")

        # Prepare the initial messages for the conversation
        messages = [
            {"role": "system", "content": "You are a memory assistant."},
            {"role": "user", "content": prompt}
        ]
        tools = [find_memory]

        # Serialize the total input to JSON for token counting (messages and tools)
        total_input = json.dumps({"messages": messages, "tools": tools})
        total_input_token_count = self.count_tokens(total_input)
        logger.info(f"Total input token count (including messages and tools): {total_input_token_count}")


        # Call the OpenAI API with function calling
        try:
            logger.info(f"Calling the OpenAI API with function calling for generating related memories.")
            related_memories_token_count_input = total_input_token_count


            if (self.model_location_cloud == False):
                response = self.clientinstructor.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=ModelFindRealtedmemmories,
                )
                respjson = response.model_dump_json(indent=2)
                function_arguments = json.loads(respjson)                
                queries = function_arguments['queries']    

                # Calculate output tokens for local model
                related_memories_token_count_output = self.count_tokens(respjson)
                related_memories_total_cost = self.calculate_cost(related_memories_token_count_input, related_memories_token_count_output) + GET_MEMORY_COST
                
                # Now use these queries to find related memories
                related_memories = self.memory_graph.find_related_memories(
                    session_token=session_token,
                    memory_graph=memory_graph,
                    memory_item=memory_item,
                    queries=queries,
                    user_id=user_id,
                    chat_gpt=self,
                    metadata=metadata,
                    skip_neo=True,
                    legacy_route=True
                )                
                return {
                    "data": related_memories,
                    "generated_queries": queries,
                    "metrics": {
                        "related_memories_token_count_input": related_memories_token_count_input,
                        "related_memories_token_count_output": related_memories_token_count_output,
                        "related_memories_total_cost": related_memories_total_cost
                    }
                }
            else:                
                response = self._create_completion_sync(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "find_related_memories"}},
                    temperature=0.7
                )

                # Process the API's response
                response_message = response.choices[0].message
                logger.info(f"Response message: {response_message}")

                # Assuming response_message contains structured data and/or text, calculate output token count
                output_text = response_message.content if response_message.content else ""
                output_token_count = self.count_tokens(output_text)
                logger.info(f"Output token count: {output_token_count}")

                tool_calls = response_message.tool_calls
                if tool_calls:
                    # Get queries and calculate token counts
                    function_arguments = json.loads(tool_calls[0].function.arguments)
                    queries = function_arguments['queries']
                    token_count_arguments = self.count_tokens(json.dumps(function_arguments))
                    
                    # Calculate total output tokens and cost
                    related_memories_token_count_output = output_token_count + token_count_arguments
                    related_memories_total_cost = self.calculate_cost(related_memories_token_count_input, related_memories_token_count_output) + GET_MEMORY_COST
                    
                    # Get related memories
                    related_memories = self.memory_graph.find_related_memories(
                        session_token=session_token,
                        memory_graph=memory_graph,
                        memory_item=memory_item,
                        queries=queries,
                        user_id=user_id,
                        chat_gpt=self,
                        metadata=metadata,
                        skip_neo=True,
                        legacy_route=True
                    )

                    return {
                        "data": related_memories,
                        "generated_queries": queries,
                        "metrics": {
                            "related_memories_token_count_input": related_memories_token_count_input,
                            "related_memories_token_count_output": related_memories_token_count_output,
                            "related_memories_total_cost": related_memories_total_cost
                        }
                    }
                else:
                    logger.warning("No tool calls found in the response.")
                    return {
                        "error": "No tool call was made in the response.",
                        "generated_queries": [],
                        "metrics": {
                            "related_memories_token_count_input": related_memories_token_count_input,
                            "related_memories_token_count_output": 0,
                            "related_memories_total_cost": GET_MEMORY_COST  # Only including get_memory cost
                        }
                    }

        except Exception as e:
            logger.error(f"Error in generate_related_memories: {e}")
            raise

    async def generate_related_memories_async(
        self,
        session_token: str,
        memory_graph: Dict[str, Any],
        memory_item: Dict[str, Any],
        user_id: str,
        neo_session: AsyncSession,
        goal: Optional[Dict[str, Any]] = None,
        use_case: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        exclude_memory_id: Optional[str] = None,
        user_workspace_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        legacy_route: bool = True
    ) -> Union[RelatedMemoriesSuccess, RelatedMemoriesError]:
        """
        Uses ChatGPT to define queries for finding related memories and then executes these queries
        to construct relationships.

        Args:
            session_token (str): Authentication token for the session
            memory_graph (Dict[str, Any]): The memory graph structure
            memory_item (Dict[str, Any]): The memory item to find relations for
            user_id (str): The ID of the user
            goal (Optional[Dict[str, Any]]): Optional goal context
            use_case (Optional[Dict[str, Any]]): Optional use case context
            metadata (Optional[Dict[str, Any]]): Optional metadata
            exclude_memory_id (Optional[str]): Optional memory ID to exclude from results

        Returns:
            Union[RelatedMemoriesSuccess, RelatedMemoriesError]: Dictionary containing either:
                - Success case: data (List[ParseStoredMemory]), generated_queries, and metrics
                - Error case: error message, empty generated_queries, and metrics
                - user_workspace_ids (Optional[List[str]]): List of user workspace IDs
        """

        GET_MEMORY_COST = 0.0015164  # Fixed cost includes sentence-bert, big-bird embedding costs and pinecone retreive cost

        # Define the function for the API to call
        find_memory = {
            "type": "function",
            "function": {
                "name": "find_related_memories",
                "description": "Runs multiple queries to find related memories based on the provided queries list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "List of queries that we need to run to get related memories for the user. Up to 3 queries",
                            "items": {
                                "type": "string",
                                "description": "Query that we can use to find a related memory.",
                            },
                        }
                    },
                    "required": [],
                },
            },
        }


        # Constructing the prompt to generate queries
        prompt = f"Given this memory graph in JSON format: {json.dumps(memory_graph)}, and this list of Goals: {json.dumps(goal)} and list of use_cases: {json.dumps(use_case)}, create detailed queries made up of two sentences to find related memories that we need to build a Neo4js relationship with this memory item {json.dumps(memory_item)}. Exclude queries that would generate an actual result for this specific memory item since are just adding it now {json.dumps(memory_item)}."

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")

        # Prepare the initial messages for the conversation
        messages = [
            {"role": "system", "content": "You are a memory assistant."},
            {"role": "user", "content": prompt}
        ]
        tools = [find_memory]

        # Serialize the total input to JSON for token counting (messages and tools)
        total_input = json.dumps({"messages": messages, "tools": tools})
        total_input_token_count = self.count_tokens(total_input)
        logger.info(f"Total input token count (including messages and tools): {total_input_token_count}")


        # Call the OpenAI API with function calling
        try:
            logger.info(f"Calling the OpenAI API with function calling for generating related memories.")
            related_memories_token_count_input = total_input_token_count


            if (self.model_location_cloud == False):
                response = await self.clientinstructor.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=ModelFindRealtedmemmories,
                )
                respjson = response.model_dump_json(indent=2)
                function_arguments = json.loads(respjson)                
                queries = function_arguments['queries']    

                # Calculate output tokens for local model
                related_memories_token_count_output = self.count_tokens(respjson)
                related_memories_total_cost = self.calculate_cost(related_memories_token_count_input, related_memories_token_count_output) + GET_MEMORY_COST
                
                # Now use these queries to find related memories
                related_memories_result = await self.memory_graph.find_related_memories(
                    session_token=session_token,
                    memory_graph=memory_graph,
                    memory_item=memory_item,
                    queries=queries,
                    user_id=user_id,
                    chat_gpt=self,
                    metadata=metadata,
                    neo_session=neo_session,
                    skip_neo=True,
                    exclude_memory_id=exclude_memory_id,
                    user_workspace_ids=user_workspace_ids,
                    api_key=api_key,
                    reranking_config=RerankingConfig(
                        reranking_enabled=True,
                        reranking_model=os.environ.get("LLM_MODEL", "gpt-4.1-nano")
                    ),
                    legacy_route=legacy_route
                )
                
                # Unpack the result
                related_memories, confidence_scores = related_memories_result
                logger.info(f'related_memories: {related_memories}')
                logger.info(f'confidence_scores: {confidence_scores}')

                return RelatedMemoriesSuccess(
                    data=related_memories,
                    generated_queries=queries,
                    confidence_scores=confidence_scores,
                    metrics={
                        "related_memories_token_count_input": related_memories_token_count_input,
                        "related_memories_token_count_output": related_memories_token_count_output,
                        "related_memories_total_cost": related_memories_total_cost
                    }
                )
            else:                
                response = await self._create_completion_async(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "find_related_memories"}},
                    temperature=0.7
                )

                # Process the API's response
                response_message = response.choices[0].message
                logger.info(f"Response message: {response_message}")

                # Assuming response_message contains structured data and/or text, calculate output token count
                output_text = response_message.content if response_message.content else ""
                output_token_count = self.count_tokens(output_text)
                logger.info(f"Output token count: {output_token_count}")

                tool_calls = response_message.tool_calls
                if tool_calls:
                    # Get queries and calculate token counts
                    function_arguments = json.loads(tool_calls[0].function.arguments)
                    queries = function_arguments['queries']
                    token_count_arguments = self.count_tokens(json.dumps(function_arguments))
                    
                    # Calculate total output tokens and cost
                    related_memories_token_count_output = output_token_count + token_count_arguments
                    related_memories_total_cost = self.calculate_cost(related_memories_token_count_input, related_memories_token_count_output) + GET_MEMORY_COST
                    
                    # Get related memories
                    reranking_config = RerankingConfig(reranking_enabled=True, reranking_model="gpt-4.1-nano")
                    
                    related_memories_result = await self.memory_graph.find_related_memories(
                        session_token=session_token,
                        memory_graph=memory_graph,
                        memory_item=memory_item,
                        queries=queries,
                        user_id=user_id,
                        chat_gpt=self,
                        metadata=metadata,
                        neo_session=neo_session,
                        skip_neo=True,
                        exclude_memory_id=exclude_memory_id,
                        user_workspace_ids=user_workspace_ids,
                        api_key=api_key,
                        reranking_config=reranking_config,
                        legacy_route=legacy_route
                    )
                    
                    # Unpack the result
                    related_memories, confidence_scores = related_memories_result
                    logger.info(f'related_memories: {related_memories}')
                    logger.info(f'confidence_scores: {confidence_scores}')

                    return RelatedMemoriesSuccess(
                        data=related_memories,
                        generated_queries=queries,
                        confidence_scores=confidence_scores,
                        metrics={
                            "related_memories_token_count_input": related_memories_token_count_input,
                            "related_memories_token_count_output": related_memories_token_count_output,
                            "related_memories_total_cost": related_memories_total_cost
                        }
                    )
                else:
                    return RelatedMemoriesError(
                        error="No tool call was made in the response.",
                        generated_queries=[],
                        confidence_scores=[],
                        metrics={
                            "related_memories_token_count_input": related_memories_token_count_input,
                            "related_memories_token_count_output": 0,
                            "related_memories_total_cost": GET_MEMORY_COST
                        }
                    )

        except Exception as e:
            logger.error(f"Error in generate_related_memories_async: {e}")
            return RelatedMemoriesError(
                error=str(e),
                generated_queries=[],
                confidence_scores=[],
                metrics={
                    "related_memories_token_count_input": related_memories_token_count_input,
                    "related_memories_token_count_output": 0,
                    "related_memories_total_cost": GET_MEMORY_COST
                }
            )


    def define_memory_graph_node(self, memory_graph, memory_item, related_memories=None):
        """
        Define a Neo4j node and relationships for a memory item using the Chat Completion API.

        :param memory_graph: The existing memory graph in JSON format.
        :param memory_item: The memory item to be added to the graph.
        :param context: Optional context that may contain previous memory items.
        :return: JSON formatted definition of the new node and its relationships.
        """

            
        # Adjust the relationship and node example data as needed
        relationship_json_example = json.dumps({
            "node_name": "GitHubComment",
            "relationships_json": [{
                "related_item_id": "unique_id_here",
                "relation_type": "needs_refactoring",
                "related_item_type": "TextMemoryItem",
                "metadata": {
                    "code_owner": "Shawkat",
                    "tech_debt": "Specific improvements needed"
                }
            }]
        }, indent=2)

        # Construct the prompt with dynamic length adjustment
        base_prompt = f"Given this memory graph in JSON format: {json.dumps(memory_graph, indent=2)}, create a Neo4j node in JSON format for the memory item {json.dumps(memory_item, indent=2)}."
        related_memories_prompt = ""
        if related_memories:
            related_memories_prompt = f"These are related memories for the user that we need to define Neo4j relationships with. Select the top 3 most relevant memories to then define Neo4j relationships with, make sure to only use relationship types that are in the user's existing memory graph. This our memory item: {json.dumps(related_memories, indent=2)}. Format your response as JSON format using this example only include node_name and relationship_json:{relationship_json_example}"
        else:
            related_memories_prompt = f"Format your response as JSON format using this example only include node_name: {relationship_json_example}"

        full_prompt = f"{base_prompt} {related_memories_prompt}"
        
        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(full_prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")

        # Calculate available tokens for completion, ensuring not to exceed 128000 tokens
        # Subtracting a small buffer (e.g., 7 tokens) to account for any calculation discrepancies
        max_tokens_for_completion = min(4096, 128000 - token_count_prompt - 7)
        logger.info(f"max_tokens_for_completion: {max_tokens_for_completion}")

        # Check if the calculated max_tokens_for_completion is below the minimum required for a meaningful response
        if max_tokens_for_completion < 1024:  # Example threshold, adjust as needed
            raise ValueError("Available tokens for completion are too few for a meaningful response.")

        # Ensure the max_tokens parameter in the API request does not exceed the calculated limit
        try:
            # Save input token count
            graph_node_token_count_input = token_count_prompt

            if (self.model_location_cloud == False):
                response = self.OpenAIAPIOllama.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=max_tokens_for_completion,  # Adjusted dynamically
                    temperature=0.7,
                    response_format={ "type": "json_object" }
                )
                content = response.choices[0].message.content
            else:                
                response = self._create_completion_sync(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=max_tokens_for_completion,  # Will be normalized to max_completion_tokens if needed
                    temperature=0.7,
                    response_format={ "type": "json_object" }

                )
                content = response.choices[0].message.content
             # Check if the response has choices and message content
            if response.choices and response.choices[0].message.content:
                try:
                           
                    response_content = response.choices[0].message.content
                    logger.info(f"response_content: {response_content}")
                    # Parse the response content as JSON

                    # Calculate metrics
                    graph_node_token_count_output = self.count_tokens(response_content)
                    graph_node_total_cost = self.calculate_cost(graph_node_token_count_input, graph_node_token_count_output)
                    
                    logger.info(f"Token counts - Input: {graph_node_token_count_input}, Output: {graph_node_token_count_output}")
                    logger.info(f"Total cost: ${graph_node_total_cost:.4f}")

                    node_and_relationships = json.loads(response_content)

                    # Return structured response with data and metrics
                    return {
                        "data": node_and_relationships,
                        "metrics": {
                            "graph_node_token_count_input": graph_node_token_count_input,
                            "graph_node_token_count_output": graph_node_token_count_output,
                            "graph_node_total_cost": graph_node_total_cost
                        }
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}. Response content: ```json\n{response_content}\n```")
                    raise
            else:
                logger.error(f"Unexpected response or empty content. Response: {response}")
                return {
                    "data": None,
                    "metrics": {
                        "graph_node_token_count_input": graph_node_token_count_input,
                        "graph_node_token_count_output": 0,
                        "graph_node_total_cost": 0
                    }
                }
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise
   
    def acl_filter_to_cypher_conditions(self, acl_filter: ACLFilter, node_alias: str = "m") -> str:
        """
        Convert ACLFilter Pydantic model to Cypher conditions string for a specific node.
        
        IMPORTANT: This should be called for BOTH m and n nodes in relationship queries
        to ensure multi-tenant isolation and prevent leaking data across tenants.
        
        Args:
            acl_filter (ACLFilter): Pydantic model containing ACL conditions
            node_alias (str): Node variable name in Cypher query (e.g., "m", "n")
            
        Returns:
            str: Cypher conditions string for the specified node
        """
        conditions = []

        # Process each condition in the or_ list
        for condition in acl_filter.or_:
            if condition.user_id:
                # Handle user_id equality condition
                eq_value = condition.user_id.get("$eq")
                if eq_value:
                    conditions.append(f"{node_alias}.user_id IS NOT NULL AND {node_alias}.user_id = '{eq_value}'")
                    
            if condition.user_read_access:
                # Handle user_read_access IN condition
                in_values = condition.user_read_access.get("$in", [])
                if in_values:
                    in_values_formatted = [f"'{v}'" for v in in_values]
                    conditions.append(f"{node_alias}.user_read_access IS NOT NULL AND any(x IN {node_alias}.user_read_access WHERE x IN [{', '.join(in_values_formatted)}])")
                    
            if condition.workspace_read_access:
                # Handle workspace_read_access IN condition
                in_values = condition.workspace_read_access.get("$in", [])
                if in_values:
                    in_values_formatted = [f"'{v}'" for v in in_values]
                    conditions.append(f"{node_alias}.workspace_read_access IS NOT NULL AND any(x IN {node_alias}.workspace_read_access WHERE x IN [{', '.join(in_values_formatted)}])")
                    
            if condition.role_read_access:
                # Handle role_read_access IN condition
                in_values = condition.role_read_access.get("$in", [])
                if in_values:
                    in_values_formatted = [f"'{v}'" for v in in_values]
                    conditions.append(f"{node_alias}.role_read_access IS NOT NULL AND any(x IN {node_alias}.role_read_access WHERE x IN [{', '.join(in_values_formatted)}])")
                    
            if condition.organization_id:
                # Handle organization_id equality condition
                eq_value = condition.organization_id.get("$eq")
                if eq_value:
                    conditions.append(f"{node_alias}.organization_id IS NOT NULL AND {node_alias}.organization_id = '{eq_value}'")
                    
            if condition.organization_read_access:
                # Handle organization_read_access IN condition
                in_values = condition.organization_read_access.get("$in", [])
                if in_values:
                    in_values_formatted = [f"'{v}'" for v in in_values]
                    conditions.append(f"{node_alias}.organization_read_access IS NOT NULL AND any(x IN {node_alias}.organization_read_access WHERE x IN [{', '.join(in_values_formatted)}])")
                    
            if condition.namespace_id:
                # Handle namespace_id equality condition
                eq_value = condition.namespace_id.get("$eq")
                if eq_value:
                    conditions.append(f"{node_alias}.namespace_id IS NOT NULL AND {node_alias}.namespace_id = '{eq_value}'")
                    
            if condition.namespace_read_access:
                # Handle namespace_read_access IN condition
                in_values = condition.namespace_read_access.get("$in", [])
                if in_values:
                    in_values_formatted = [f"'{v}'" for v in in_values]
                    conditions.append(f"{node_alias}.namespace_read_access IS NOT NULL AND any(x IN {node_alias}.namespace_read_access WHERE x IN [{', '.join(in_values_formatted)}])")

        # Join all conditions with OR
        cypher_acl_condition = " OR ".join(conditions) if conditions else "true"
        return cypher_acl_condition

    def generate_neo4j_cipher_query(
        self, 
        user_query: str, 
        bigbird_memory_ids: list, 
        acl_filter: ACLFilter, 
        context: Optional[List[ContextItem]] = None, 
        project_id: str = None, 
        user_id: str = None, 
        memory_graph: dict = None, 
        memory_graph_schema: Dict[str, Any] = None,
        top_k: int = 10
    ) -> str:
        """Generate an optimized Neo4j Cypher query using structured tool calls."""
        try:
            # Define Memory node properties
            memory_node_properties = [
                'id', 'content', 'createdAt', 'user_id', 'type',
                'topics', 'emotion_tags', 'steps', 'current_step'
            ]

            cypher_tool = {
                "type": "function",
                "function": {
                    "name": "generate_cypher_query",
                    "description": "Generate a Neo4j Cypher query for Memory nodes and their relationships",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_parts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "clause": {
                                            "type": "string",
                                            "enum": ["MATCH", "WHERE", "WITH", "RETURN", "ORDER BY", "LIMIT", "OPTIONAL MATCH"]
                                        },
                                        "pattern": {"type": "string"},
                                        "conditions": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "properties": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": memory_node_properties
                                            }
                                        }
                                    },
                                    "required": ["clause"]
                                }
                            },
                            "parameters": {
                                "type": "object",
                                "additionalProperties": True
                            }
                        },
                        "required": ["query_parts"]
                    }
                }
            }

            # Available relationships from memory_graph
            available_relationships = memory_graph_schema.get('relationships', ['RELATION']) if memory_graph_schema else ['RELATION']
            logger.info(f'available_relationships: {available_relationships}')

            prompt = f"""
Given the following:
1. User Query: "{user_query}"
2. Node Type: Memory with properties: {memory_node_properties}
3. Available Relationship Types: {available_relationships}
4. Access Control: User ID '{acl_filter.get('user_id')}' must have access to returned nodes
5. Memory IDs to search within: $bigbird_memory_ids

Generate a Cypher query that:
- Works with Memory nodes and their properties
- Uses appropriate relationships from the available types
- Includes necessary access control conditions
- Returns relevant data based on the user's query
- Uses parameters for values where appropriate
- Always includes condition: m.id IN $bigbird_memory_ids
- Optimizes for performance

Examples of valid property usage:
- m.content: Memory content
- m.createdAt: Timestamp of creation
- m.topics: Array of topics
- m.emotion_tags: Array of emotion tags
- m.type: Type of memory
- m.steps: Array of steps
- m.current_step: Current step

Note: Always use 'm', 'r', 'n' as variable names for source Memory node, relationship, and target Memory node respectively.
"""

            # Get LLM response
            #response = self.client.chat.completions.create(
            #    model=self.model,
            #    messages=[
            #        {"role": "system", "content": "You are a Neo4j query expert that generates optimized Cypher queries for Memory graphs."},
            #        {"role": "user", "content": prompt}
            #    ],
            #    tools=[cypher_tool],
            #    tool_choice={"type": "function", "function": {"name": "generate_cypher_query"}}
            #)

            # Extract and process the query structure
            #tool_call = response.choices[0].message.tool_calls[0]
            #query_structure = json.loads(tool_call.function.arguments)
            #logger.warning(f"Generated llm query structure: {query_structure}")
            
            # Build the query from parts with proper spacing
            #query_parts = []
            #return_added = False
            #order_by_added = False
            #limit_added = False
            
            # Process MATCH and WHERE first
            #for part in query_structure['query_parts']:
            #    if part['clause'] == 'MATCH':
            #        pattern = part.get('pattern', '')
            #        query_parts.append(f"MATCH {pattern}".strip())
            #    elif part['clause'] == 'WHERE':
            #        conditions = part.get('conditions', [])
            #        required_conditions = [
            #            "m.id IN $bigbird_memory_ids",
            #            f"({self.acl_filter_to_cypher_conditions(acl_filter)})"
            #        ]
            #        all_conditions = required_conditions + [
            #            cond for cond in conditions 
            #            if "bigbird_memory_ids" not in cond 
            #            and "user_id" not in cond 
            #            and "user_read_access" not in cond
            #        ]
            #        query_parts.append(f"WHERE {' AND '.join(all_conditions)}")

            # Ensure WHERE clause exists
            #if not any(part.startswith('WHERE') for part in query_parts):
            #    match_index = next((i for i, part in enumerate(query_parts) if part.startswith('MATCH')), 0)
            #    acl_condition = self.acl_filter_to_cypher_conditions(acl_filter)
            #    where_clause = f"WHERE m.id IN $bigbird_memory_ids AND ({acl_condition})"
            #    query_parts.insert(match_index + 1, where_clause)

            # Add RETURN clause
            #for part in query_structure['query_parts']:
            #    if part['clause'] == 'RETURN':
            #        pattern = part.get('pattern', 'm, r, n')  # Default return pattern
            #        query_parts.append(f"RETURN {pattern}")
            #        return_added = True
            #        break

            # Ensure RETURN exists
            #if not return_added:
            #    query_parts.append("RETURN m, r, n")

            # Add ORDER BY if present
            #for part in query_structure['query_parts']:
            #    if part['clause'] == 'ORDER BY':
            #        pattern = part.get('pattern', 'm.createdAt DESC')  # Default ordering
            #        query_parts.append(f"ORDER BY {pattern}")
            #        order_by_added = True
            #        break

            # Add LIMIT
            #for part in query_structure['query_parts']:
            #    if part['clause'] == 'LIMIT':
            #        pattern = part.get('pattern', str(top_k))  # Use provided top_k
            #        query_parts.append(f"LIMIT {pattern}")
            #        limit_added = True
            #        break

            # Ensure LIMIT exists
            #if not limit_added:
            #    query_parts.append(f"LIMIT {top_k}")

            # Join all parts with proper spacing
            #query = "\n".join(query_parts)
            
            #logger.info(f"Generated Cypher query: {query}")
            
            # Validate the query
            #parameters = {'bigbird_memory_ids': bigbird_memory_ids}  # Define parameters
            #self.validate_cypher_query(query, parameters)
            
            return self.fallback_cipher_query(
                bigbird_memory_ids=bigbird_memory_ids,
                acl_filter=acl_filter,
                memory_graph=memory_graph,
                top_k=top_k,
                memory_graph_schema=memory_graph_schema
            )

        except Exception as e:
            logger.error(f"Error generating structured Cypher query: {str(e)}")
            return self.fallback_cipher_query(
                bigbird_memory_ids=bigbird_memory_ids,
                acl_filter=acl_filter,
                memory_graph=memory_graph,
                top_k=top_k,
                memory_graph_schema=memory_graph_schema
            )


    async def generate_neo4j_cipher_query_async(
        self, 
        user_query: str, 
        acl_filter: Dict[str, Any], 
        context: Optional[List[ContextItem]] = None, 
        project_id: str = None, 
        user_id: str = None, 
        memory_graph: "MemoryGraph" = None, 
        memory_graph_schema: Dict[str, Any] = None,
        top_k: int = 10,
        enhanced_schema_cache: Optional[Dict[str, Any]] = None,
        neo_session: Optional[AsyncSession] = None
    ) -> Tuple[str, bool]:  # Return tuple of (query, is_llm_generated)
        """Generate an optimized Neo4j Cypher query using structured tool calls. Returns (query, is_llm_generated). If query generation fails, returns ("", False)."""
        try:
            generated_query = ""
            # Define available node labels and relationships
            available_nodes = memory_graph_schema.get('nodes', ['Memory']) if memory_graph_schema else ['Memory']
            available_relationships = memory_graph_schema.get('relationships', []) if memory_graph_schema else []
            relationship_patterns = memory_graph_schema.get('patterns', []) if memory_graph_schema else []
            
            # 🔍 DEBUG: Log ActivePatterns integration
            logger.info(f"🔍 CYPHER GENERATION DEBUG:")
            logger.info(f"🔍   memory_graph_schema keys: {list(memory_graph_schema.keys()) if memory_graph_schema else 'None'}")
            logger.info(f"🔍   available_nodes: {available_nodes}")
            logger.info(f"🔍   available_relationships: {available_relationships}")
            logger.info(f"🔍   relationship_patterns count: {len(relationship_patterns)}")
            if relationship_patterns:
                logger.info(f"🔍   First 3 patterns: {relationship_patterns[:3]}")
            else:
                logger.warning(f"🔍   ⚠️ NO RELATIONSHIP PATTERNS FOUND - This will cause 0 neo nodes!")

            # DYNAMIC REGISTRATION: Register user's custom node properties (skip if already done in auth)
            if user_id and not memory_graph_schema.get('dynamic_registration_completed', False):
                try:
                    from services.schema_service import SchemaService
                    from models.cipher_ast import register_user_custom_properties
                    
                    schema_service = SchemaService()
                    # Get user's active schemas with full multi-tenant context
                    workspace_id = acl_filter.get('workspace_id')
                    organization_id = acl_filter.get('organization_id')
                    namespace_id = acl_filter.get('namespace_id')
                    
                    logger.info(f"🔧 DYNAMIC REGISTRATION: Fetching schemas with workspace_id={workspace_id}, org_id={organization_id}, namespace_id={namespace_id}")
                    user_schemas = await schema_service.get_active_schemas(
                        user_id, 
                        workspace_id=workspace_id,
                        organization_id=organization_id,
                        namespace_id=namespace_id
                    )
                    
                    # Dynamically register custom properties
                    register_user_custom_properties(user_schemas)
                    logger.info(f"🔧 DYNAMIC REGISTRATION: Processed {len(user_schemas)} user schemas for custom properties")
                    
                except Exception as e:
                    logger.warning(f"Failed to register dynamic custom properties: {e}")
            elif memory_graph_schema.get('dynamic_registration_completed', False):
                logger.info(f"🚀 SKIPPING REGISTRATION: Dynamic properties already registered in auth phase")

            # Coerce node labels to NodeLabel enums for property table validation
            available_node_enums = []
            for label in available_nodes:
                try:
                    available_node_enums.append(NodeLabel(label) if not isinstance(label, NodeLabel) else label)
                except Exception:
                    continue

            # If relationships are empty, allow all known relationships as fallback
            if not available_relationships:
                available_relationships = [r.value for r in RelationshipType]

            # Dynamically build allowed properties table for prompt
            allowed_properties_table = get_allowed_properties_table(available_node_enums)

            # Log input parameters
            logger.info(f"Generating Neo4j query with:")
            logger.info(f"User query: {user_query}")
            logger.info(f"Available nodes: {available_nodes}")
            logger.info(f"Available relationships: {available_relationships}")
            logger.info(f"Allowed properties table: {allowed_properties_table}")
            
            # Define messages for both Groq paths
            messages = [
                {
                    "role": "system",
                    "content": f"""
                        You are a graph pattern selector that chooses the most relevant relationship pattern 
                        from actual database patterns discovered from the user's data.
                        
                        CRITICAL: You MUST use the generate_cypher_query tool to provide your response. 
                        Do NOT provide a text response - ALWAYS use the tool call format.
                        
                        Your job is to:
                        1. Select the best pattern from the available options that matches the user's query intent
                        2. Identify relevant property filters from the user's question
                        3. Specify comparison operators (CONTAINS, EQUALS, etc.)
                        4. Provide reasoning for your selection
                        5. ALWAYS respond using the generate_cypher_query tool - never provide plain text
                        
                        You are NOT writing raw Cypher queries - a template system will build the final query
                        using your selected pattern and filters.
                        
                        PATTERN SELECTION RULES:
                        1. Choose the pattern that best captures the user's relationship intent
                        2. Prefer specific patterns over generic ones when available
                        3. Consider the entities mentioned in the user's query
                        4. Only add property filters that are explicitly mentioned or strongly implied
                        5. MINIMIZE property filters - only add what's clearly needed from the user query
                        6. Do NOT add ID conditions unless specifically requested
                        
                        ALLOWED COMPARISON OPERATORS:
                        - CONTAINS: for partial string matches (preferred for names, text content)
                        - STARTS WITH: for prefix matches
                        - EQUALS: use only when exact match is required
                        - IN: for list membership

                        {allowed_properties_table}

                        Example:
                        User: "Find Python functions that John created"
                        Good Selection: "Function -> CREATED_BY -> Person" 
                        + source_properties: [{{"property": "language", "operator": "CONTAINS", "value": "Python"}}]
                        + target_properties: [{{"property": "name", "operator": "CONTAINS", "value": "John"}}]
                        
                        Remember:
                        - ALWAYS use the generate_cypher_query tool to respond
                        - Focus on the main relationship the user is asking about
                        - Only add property filters that are explicitly mentioned in the query
                        - Provide clear reasoning for your pattern choice
                        - Never provide plain text responses - tool calls only"""
                },
                {
                    "role": "user",
                    "content": f"""
                    Select the most appropriate graph pattern for this query: "{user_query}"
                    
                    Available patterns from your database: {relationship_patterns[:10] if relationship_patterns else 'Using fallback patterns'}
                    Available nodes: {available_nodes}
                    Available relationships: {available_relationships}
                    
                    IMPORTANT: Use the generate_cypher_query tool to provide your response. 
                    Do not provide a text response - use the tool call format.
                    
                    Choose the pattern and filters that best match the user's intent.
                    Focus on the main relationship being queried and only add property filters 
                    that are clearly mentioned or strongly implied in the user's question.
                    """
                }
            ]

            # In the generate_neo4j_cipher_query_async method, before the API call:
            try:
                # Create dynamic schema with discovered node labels, relationships, and patterns
                dynamic_schema = self.create_dynamic_cypher_schema(available_nodes, available_relationships, relationship_patterns)
                logger.info(f"CypherQuery schema: {json.dumps(dynamic_schema, indent=2)}")
                import time
                if not os.environ.get("GROQ_NEO_CYPHER"):
                    # Create tools with the dynamic schema
                    # Using standard Groq tool calling format (no prefix needed)
                    cypher_tool = {
                        "type": "function",
                        "function": {
                            "name": "generate_cypher_query",
                            "description": "Generate a Cypher query AST for Neo4j",
                            "parameters": dynamic_schema
                        }
                    }

                    start_time = time.time()
                    # Use Groq OpenAI/GPT-OSS-20B for pattern selection
                    # Use "required" for proper tool calling, with OpenAI fallback if Groq fails
                    tool_choice_setting = "required" if relationship_patterns else "auto"
                    logger.info(f"🔧 GROQ TOOL CHOICE: Using '{tool_choice_setting}' (patterns: {len(relationship_patterns)})")
                    
                    # Log the complete messages being sent to Groq
                    logger.warning(f"🔧 GROQ MESSAGES: Complete system message:")
                    logger.warning(f"🔧 GROQ MESSAGES: {messages[0]['content']}")
                    logger.warning(f"🔧 GROQ MESSAGES: Complete user message:")
                    logger.warning(f"🔧 GROQ MESSAGES: {messages[1]['content']}")
                    logger.warning(f"🔧 GROQ MESSAGES: Dynamic schema being sent:")
                    logger.warning(f"🔧 GROQ MESSAGES: {json.dumps(dynamic_schema, indent=2)}")
                    
                    completion = await self.groq_async_client.chat.completions.create(
                        model=self.groq_pattern_selector_model,
                        messages=messages,
                        tools=[cypher_tool],
                        tool_choice=tool_choice_setting
                    )
                    end_time = time.time()
                    logger.warning(f"Groq OpenAI/GPT-OSS-20B Response time: {end_time - start_time} seconds")
                    
                    # Check if tool calls were made
                    message = completion.choices[0].message
                    if not message.tool_calls or len(message.tool_calls) == 0:
                        logger.warning(f"🔧 GROQ FALLBACK: No tool call made, attempting to parse text response")
                        logger.info(f"🔧 Response content: {message.content}")
                        
                        # Try to extract JSON from the text response with multiple patterns
                        try:
                            import re
                            tool_response = None
                            
                            # Pattern 1: JSON in code blocks
                            json_match = re.search(r'```json\s*(\{.*?\})\s*```', message.content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1).strip()
                                tool_response = json.loads(json_str)
                                logger.info(f"🔧 Successfully extracted JSON from code block")
                            
                            # Pattern 2: JSON without code blocks
                            if not tool_response:
                                json_match = re.search(r'(\{[^{}]*"chosen_pattern"[^{}]*\})', message.content, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(1).strip()
                                    tool_response = json.loads(json_str)
                                    logger.info(f"🔧 Successfully extracted JSON from text (no code blocks)")
                            
                            # Pattern 3: Try to find any valid JSON object in the response
                            if not tool_response:
                                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', message.content)
                                for json_obj in json_objects:
                                    try:
                                        parsed = json.loads(json_obj.strip())
                                        if 'chosen_pattern' in parsed:  # Validate it's our expected format
                                            tool_response = parsed
                                            logger.info(f"🔧 Successfully extracted JSON from general pattern")
                                            break
                                    except:
                                        continue
                            
                            if not tool_response:
                                logger.error(f"🚨 Could not extract JSON from Groq text response")
                                logger.error(f"🚨 Response content: {message.content}")
                                raise Exception(f"Groq failed to use tool call and no JSON found. Response: {message.content}")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"🚨 Failed to parse JSON from text response: {e}")
                            raise Exception(f"Groq failed to use tool call. Response: {message.content}")
                        except Exception as e:
                            logger.error(f"🚨 Unexpected error extracting JSON from text: {e}")
                            raise Exception(f"Groq failed to use tool call. Response: {message.content}")
                    else:
                        # Parse the tool call response
                        tool_call = message.tool_calls[0]
                        logger.info(f"🔧 Tool call function name: {tool_call.function.name}")
                        logger.info(f"🔧 Tool call arguments: {tool_call.function.arguments}")
                        
                        # Clean the JSON string before parsing (remove trailing newlines, etc.)
                        try:
                            json_str = tool_call.function.arguments.strip()
                            tool_response = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"🚨 Failed to parse tool call JSON: {e}")
                            logger.error(f"🚨 Raw arguments: {repr(tool_call.function.arguments)}")
                            # Try to fix common JSON issues
                            try:
                                cleaned_json = self._fix_malformed_json(tool_call.function.arguments)
                                tool_response = json.loads(cleaned_json)
                                logger.info(f"🔧 Successfully parsed JSON after cleaning")
                            except Exception as e2:
                                logger.error(f"🚨 Failed to parse JSON even after cleaning: {e2}")
                                raise Exception(f"Failed to parse Groq tool call JSON: {e}")
                        except Exception as e:
                            logger.error(f"🚨 Unexpected error parsing tool call: {e}")
                            raise
                    
                    # Parse the enhanced response format and use template system
                    try:
                        chosen_pattern = tool_response.get('chosen_pattern', '')
                        source_properties = tool_response.get('source_properties', [])
                        target_properties = tool_response.get('target_properties', [])
                        reasoning = tool_response.get('reasoning', '')
                        
                        logger.info(f"🔧 Groq chosen pattern: {chosen_pattern}")
                        logger.info(f"🔧 Source properties: {source_properties}")
                        logger.info(f"🔧 Target properties: {target_properties}")
                        logger.info(f"🔧 Groq reasoning: {reasoning}")
                        
                        # Convert property objects to filter format for template system
                        source_filters = []
                        target_filters = []
                        
                        # Parse source properties (now objects with property, operator, value)
                        for prop_obj in source_properties:
                            if isinstance(prop_obj, dict) and 'property' in prop_obj and 'value' in prop_obj:
                                source_filters.append({
                                    'property': prop_obj['property'],
                                    'operator': prop_obj.get('operator', 'CONTAINS'),
                                    'value': prop_obj['value']
                                })
                            elif isinstance(prop_obj, str):
                                # Backward compatibility: if it's still a string, handle it the old way
                                logger.warning(f"🔧 Received string property instead of object: {prop_obj}")
                                # Keep old logic as fallback
                                if prop_obj == 'name':
                                    import re
                                    # Use user_query as the source for name extraction
                                    potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_query)
                                    for name in potential_names:
                                        if len(name.split()) >= 2:
                                            source_filters.append({'property': 'name', 'operator': 'CONTAINS', 'value': name})
                                            break
                        
                        # Parse target properties (now objects with property, operator, value)
                        for prop_obj in target_properties:
                            if isinstance(prop_obj, dict) and 'property' in prop_obj and 'value' in prop_obj:
                                target_filters.append({
                                    'property': prop_obj['property'],
                                    'operator': prop_obj.get('operator', 'CONTAINS'),
                                    'value': prop_obj['value']
                                })
                            elif isinstance(prop_obj, str):
                                # Backward compatibility: if it's still a string, handle it the old way
                                logger.warning(f"🔧 Received string property instead of object: {prop_obj}")
                                # Keep old logic as fallback
                                if prop_obj == 'language':
                                    # Use user_query as the source for language detection
                                    query_lower = user_query.lower()
                                    languages = ['python', 'javascript', 'java', 'typescript', 'go', 'rust', 'c++', 'c#']
                                    for lang in languages:
                                        if lang.lower() in query_lower:
                                            target_filters.append({'property': 'language', 'operator': 'CONTAINS', 'value': lang})
                                            break
                        
                        logger.info(f"🔧 Generated source filters: {source_filters}")
                        logger.info(f"🔧 Generated target filters: {target_filters}")
                        
                        # Use the enhanced template system to build Cypher query
                        generated_query = self.build_cypher_from_pattern(
                            selected_pattern=chosen_pattern,
                            source_filters=source_filters,
                            target_filters=target_filters,
                            user_id=user_id
                        )
                        
                        logger.info(f"🔧 Template system generated query successfully")
                        
                        # OPTIONAL: Enhance query with property suggestions from Qdrant (if enhanced schema cache available)
                        # Allow enhancement with system schemas even if no user schemas exist
                        if enhanced_schema_cache and generated_query:
                            try:
                                enhanced_query = await self._enhance_query_with_property_suggestions(
                                    cypher_query=generated_query,
                                    user_query=user_query,
                                    acl_filter=acl_filter,
                                    enhanced_schema_cache=enhanced_schema_cache,
                                    memory_graph=memory_graph,
                                    neo_session=neo_session
                                )
                                if enhanced_query:
                                    generated_query = enhanced_query
                                    logger.info(f"🚀 PROPERTY ENHANCEMENT: Successfully enhanced query with Qdrant property suggestions")
                                else:
                                    logger.info(f"ℹ️ PROPERTY ENHANCEMENT: No enhancements needed")
                            except Exception as e:
                                logger.warning(f"🚀 PROPERTY ENHANCEMENT: Enhancement failed, continuing with original query: {e}")
                        elif enhanced_schema_cache and generated_query:
                            logger.info(f"ℹ️ PROPERTY ENHANCEMENT: Skipped - no UserGraphSchema data available (ActiveNodeRel only)")
                        
                    except Exception as parse_error:
                        logger.error(f"Template system failed: {parse_error}")
                        logger.info(f"Tool response: {tool_response}")
                        
                        # Final fallback - use template system with default pattern
                        try:
                            logger.warning("Using fallback pattern: Memory-RELATED_TO->Memory")
                            generated_query = self.build_cypher_from_pattern(
                                selected_pattern="Memory-RELATED_TO->Memory",
                                source_filters=[],
                                target_filters=[],
                                user_id=user_id
                            )
                            logger.info(f"Fallback template system successful")
                        except Exception as fallback_error:
                            logger.error(f"Fallback template system also failed: {fallback_error}")
                            # Use absolute fallback with ACL for BOTH m and n nodes
                            generated_query = """MATCH path = (m:Memory)-[:RELATED_TO*1..2]->(n:Memory)
WHERE (m.user_id = $user_id OR any(x IN coalesce(m.user_read_access, []) WHERE x IN $user_read_access) OR any(x IN coalesce(m.workspace_read_access, []) WHERE x IN $workspace_read_access) OR any(x IN coalesce(m.role_read_access, []) WHERE x IN $role_read_access) OR any(x IN coalesce(m.organization_read_access, []) WHERE x IN $organization_read_access) OR any(x IN coalesce(m.namespace_read_access, []) WHERE x IN $namespace_read_access))
  AND (n.user_id = $user_id OR any(x IN coalesce(n.user_read_access, []) WHERE x IN $user_read_access) OR any(x IN coalesce(n.workspace_read_access, []) WHERE x IN $workspace_read_access) OR any(x IN coalesce(n.role_read_access, []) WHERE x IN $role_read_access) OR any(x IN coalesce(n.organization_read_access, []) WHERE x IN $organization_read_access) OR any(x IN coalesce(n.namespace_read_access, []) WHERE x IN $namespace_read_access))
WITH DISTINCT path
RETURN {
    path: path,
    nodes: [n IN nodes(path) | { id: n.id, labels: labels(n), properties: properties(n) }],
    relationships: [r IN relationships(path) | {
        type: type(r), properties: properties(r),
        startNode: startNode(r).id, endNode: endNode(r).id
    }]
} AS result"""
                            logger.warning(f"Using absolute fallback query with ACL for both m and n nodes")
                else:
                    # NEW: Use enhanced schema for Instructor/Groq path too
                    if enhanced_schema_cache and relationship_patterns:
                        logger.info(f"🚀 GROQ ENHANCED SCHEMA: Using discovered patterns for Groq/Instructor path")
                        
                        # Create enhanced schema with discovered patterns
                        enhanced_schema = self.create_dynamic_cypher_schema(
                            available_nodes=available_nodes,
                            available_relationships=available_relationships,
                            relationship_patterns=relationship_patterns
                        )
                        
                        # Define cypher_tool for OpenAI fallback (same as Route 1)
                        cypher_tool = {
                            "type": "function",
                            "function": {
                                "name": "generate_cypher_query",
                                "description": "Generate a Cypher query AST for Neo4j",
                                "parameters": enhanced_schema
                            }
                        }
                        tool_choice_setting = "required" if relationship_patterns else "auto"
                        
                        # Update messages to include enhanced schema information
                        enhanced_messages = [
                            {
                                "role": "system",
                                "content": f"""
                                    You are a graph pattern selector that chooses the most relevant relationship pattern 
                                    from actual database patterns discovered from the user's data.
                                    
                                    Your job is to:
                                    1. Select the best pattern from the available options that matches the user's query intent
                                    2. Identify relevant property filters from the user's question
                                    3. Specify comparison operators (CONTAINS, EQUALS, etc.)
                                    4. Provide reasoning for your selection
                                    
                                    You are NOT writing raw Cypher queries - a template system will build the final query
                                    using your selected pattern and filters.
                                    
                                    PATTERN SELECTION RULES:
                                    1. Choose the pattern that best captures the user's relationship intent
                                    2. Prefer specific patterns over generic ones when available
                                    3. Consider the entities mentioned in the user's query
                                    4. Only add property filters that are explicitly mentioned or strongly implied
                                    5. MINIMIZE property filters - only add what's clearly needed from the user query
                                    6. Do NOT add ID conditions unless specifically requested
                                    
                                    ALLOWED COMPARISON OPERATORS:
                                    - CONTAINS: for partial string matches (preferred for names, text content)
                                    - STARTS WITH: for prefix matches
                                    - EQUALS: use only when exact match is required
                                    - IN: for list membership

                                    {allowed_properties_table}

                                    Example:
                                    User: "Find Python functions that John created"
                                    Good Selection: "Function -> CREATED_BY -> Person" 
                                    + source_properties: [{{"property": "language", "operator": "CONTAINS", "value": "Python"}}]
                                    + target_properties: [{{"property": "name", "operator": "CONTAINS", "value": "John"}}]
                                    
                                    Remember:
                                    - Focus on the main relationship the user is asking about
                                    - Only add property filters that are explicitly mentioned in the query
                                    - Provide clear reasoning for your pattern choice"""
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Select the most appropriate graph pattern for this query: "{user_query}"
                                
                                Available patterns from your database: {relationship_patterns[:10] if relationship_patterns else 'Using fallback patterns'}
                                Available nodes: {available_nodes}
                                Available relationships: {available_relationships}
                                
                                Choose the pattern and filters that best match the user's intent.
                                Focus on the main relationship being queried and only add property filters 
                                that are clearly mentioned or strongly implied in the user's question.
                                """
                            }
                        ]
                        
                        # Create dynamic response model from enhanced schema
                        from pydantic import create_model
                        from typing import List, Optional
                        
                        # Create the dynamic model based on enhanced schema
                        DynamicPatternSelection = create_model(
                            'DynamicPatternSelection',
                            query=(str, ...),
                            chosen_pattern=(str, ...),
                            source_properties=(List[dict], []),
                            target_properties=(List[dict], []),
                            reasoning=(str, ...)
                        )
                        
                        # Log the enhanced messages being sent to Groq (Instructor path)
                        logger.info(f"🔧 GROQ INSTRUCTOR MESSAGES: Complete system message:")
                        if enhanced_messages and len(enhanced_messages) > 0:
                            logger.info(f"🔧 GROQ INSTRUCTOR MESSAGES: {enhanced_messages[0]['content']}")
                        logger.info(f"🔧 GROQ INSTRUCTOR MESSAGES: Complete user message:")
                        if enhanced_messages and len(enhanced_messages) > 1:
                            logger.info(f"🔧 GROQ INSTRUCTOR MESSAGES: {enhanced_messages[1]['content']}")
                        logger.warning(f"🔧 GROQ INSTRUCTOR MESSAGES: Enhanced schema being sent:")
                        logger.info(f"🔧 GROQ INSTRUCTOR MESSAGES: {json.dumps(enhanced_schema, indent=2)}")
                        
                        start_time = time.time()
                        completion_groq = await self.instructor_groq_client.chat.completions.create(
                            model=self.groq_pattern_selector_model,
                            messages=enhanced_messages,
                            response_model=DynamicPatternSelection
                        )
                        end_time = time.time()
                        logger.warning(f"Groq OpenAI/GPT-OSS-20B Response time: {end_time - start_time} seconds")
                        
                        # Process the enhanced response using template system
                        chosen_pattern = completion_groq.chosen_pattern
                        source_properties = completion_groq.source_properties
                        target_properties = completion_groq.target_properties
                        reasoning = completion_groq.reasoning
                        
                        logger.info(f"🔧 Groq chosen pattern: {chosen_pattern}")
                        logger.info(f"🔧 Source properties: {source_properties}")
                        logger.info(f"🔧 Target properties: {target_properties}")
                        logger.info(f"🔧 Groq reasoning: {reasoning}")
                        
                        # Convert properties to filters
                        source_filters = [{"property": prop.get("property", ""), "operator": prop.get("operator", "="), "value": prop.get("value", "")} for prop in source_properties if isinstance(prop, dict)]
                        target_filters = [{"property": prop.get("property", ""), "operator": prop.get("operator", "="), "value": prop.get("value", "")} for prop in target_properties if isinstance(prop, dict)]
                        
                        # Use the enhanced template system to build Cypher query
                        generated_query = self.build_cypher_from_pattern(
                            selected_pattern=chosen_pattern,
                            source_filters=source_filters,
                            target_filters=target_filters,
                            user_id=user_id
                        )
                        
                        if generated_query:
                            logger.info("Template system generated query successfully")
                            
                            # NEW: Apply property enhancement to ALL LLM-generated queries (both template and instructor paths)
                            enhancement_params = {}  # Parameters from property enhancement
                            if enhanced_schema_cache and generated_query:
                                logger.info(f"🚀 PROPERTY ENHANCEMENT: Attempting to enhance LLM-generated query")
                                try:
                                    enhanced_query, enhancement_params = await self._enhance_query_with_property_suggestions(
                                        cypher_query=generated_query,
                                        user_query=user_query,
                                        acl_filter=acl_filter,
                                        enhanced_schema_cache=enhanced_schema_cache,
                                        memory_graph=memory_graph
                                    )
                                    if enhanced_query and enhanced_query != generated_query:
                                        generated_query = enhanced_query
                                        logger.info(f"✅ PROPERTY ENHANCEMENT SUCCESS: Query enhanced with vector-matched properties and {len(enhancement_params)} parameters")
                                    else:
                                        logger.info(f"ℹ️ PROPERTY ENHANCEMENT: No enhancements needed")
                                except Exception as e:
                                    logger.error(f"❌ PROPERTY ENHANCEMENT ERROR: {e}", exc_info=True)
                                    # Continue with original query if enhancement fails
                            
                            logger.info(f"Final LLM generated query: {generated_query}")
                            logger.info(f"Enhancement parameters: {list(enhancement_params.keys())}")
                            return generated_query, True, enhancement_params  # Return query, is_llm_generated, and parameters
                        else:
                            logger.warning("Template system failed, falling back to static CypherQuery model")
                    
                    # Fallback to original static model if enhanced schema not available
                    # Define cypher_tool for OpenAI fallback (same as Route 1)
                    cypher_tool = {
                        "type": "function",
                        "function": {
                            "name": "generate_cypher_query",
                            "description": "Generate a Cypher query AST for Neo4j",
                            "parameters": dynamic_schema
                        }
                    }
                    tool_choice_setting = "required" if relationship_patterns else "auto"
                    
                    # Log the messages being sent to Groq (Fallback path)
                    logger.warning(f"🔧 GROQ FALLBACK MESSAGES: Complete system message:")
                    if messages and len(messages) > 0:
                        logger.warning(f"🔧 GROQ FALLBACK MESSAGES: {messages[0]['content']}")
                    logger.warning(f"🔧 GROQ FALLBACK MESSAGES: Complete user message:")
                    if messages and len(messages) > 1:
                        logger.warning(f"🔧 GROQ FALLBACK MESSAGES: {messages[1]['content']}")
                    logger.warning(f"🔧 GROQ FALLBACK MESSAGES: Using static CypherQuery model")
                    
                    start_time = time.time()
                    completion_groq = await self.instructor_groq_client.chat.completions.create(
                        model=self.groq_pattern_selector_model,
                        messages=messages,
                        response_model=CypherQuery
                    )
                    end_time = time.time()
                    logger.warning(f"Groq Response time: {end_time - start_time} seconds")
                    generated_query = completion_groq.to_cypher()
                    logger.info(f"Generated base Cypher query from AST (Groq): {generated_query}")
                    logger.info(f"Match pattern: {completion_groq.match.pattern}")
                # Log the raw response
                #logger.info(f"OpenAI Response: {json.dumps(completion.model_dump(), indent=2)}")
                
            except Exception as e:
                logger.error(f"Error in Groq query generation: {str(e)}", exc_info=True)
                
                # Log more details about the error for debugging
                error_str = str(e)
                if "tool_use_failed" in error_str:
                    logger.error(f"🚨 GROQ TOOL CALL FAILURE: This is a known Groq reliability issue")
                    logger.error(f"🚨 See: https://community.groq.com/t/issue-tool-calling-failures-on-groq-llms/592")
                elif "Failed to parse tool call arguments as JSON" in error_str:
                    logger.error(f"🚨 GROQ JSON PARSING FAILURE: Tool call JSON was malformed")
                elif "Tool choice is required, but model did not call a tool" in error_str:
                    logger.error(f"🚨 GROQ TOOL CHOICE FAILURE: Model ignored required tool call")
                else:
                    logger.error(f"🚨 UNEXPECTED GROQ ERROR: {error_str}")
                
                # FALLBACK: Try OpenAI with LLM_MODEL as a reliable alternative
                logger.warning(f"🔄 FALLBACK: Attempting OpenAI fallback due to Groq failure")
                try:
                    fallback_start = time.time()
                    
                    # Use the same tool and messages with OpenAI using LLM_MODEL_NANO or LLM_MODEL from env
                    # Try LLM_MODEL_NANO first (user preference), then LLM_MODEL, then default
                    fallback_model = os.environ.get("LLM_MODEL_NANO") or os.environ.get("LLM_MODEL", "gpt-5-nano")
                    logger.info(f"🔄 FALLBACK: Using model {fallback_model} (from LLM_MODEL_NANO={os.environ.get('LLM_MODEL_NANO')} or LLM_MODEL={os.environ.get('LLM_MODEL')} or default)")
                    
                    # Add timeout to prevent hanging (20 seconds for OpenAI fallback)
                    # Use _create_completion_async to ensure proper normalization for gpt-5 models
                    import asyncio
                    openai_completion = await asyncio.wait_for(
                        self._create_completion_async(
                            model=fallback_model,  # Use LLM_MODEL_NANO or LLM_MODEL from environment (gpt-5-nano)
                            messages=messages,
                            tools=[cypher_tool],
                            tool_choice=tool_choice_setting
                        ),
                        timeout=20.0  # 20 second timeout for OpenAI fallback
                    )
                    
                    fallback_end = time.time()
                    logger.warning(f"🔄 OpenAI {fallback_model} fallback response time: {fallback_end - fallback_start:.2f}s")
                    
                    # Parse OpenAI response (same logic as Groq)
                    openai_message = openai_completion.choices[0].message
                    if openai_message.tool_calls and len(openai_message.tool_calls) > 0:
                        tool_call = openai_message.tool_calls[0]
                        logger.info(f"🔄 FALLBACK: OpenAI tool call successful")
                        logger.info(f"🔄 Tool call arguments: {tool_call.function.arguments}")
                        
                        # Clean and parse JSON
                        json_str = tool_call.function.arguments.strip()
                        tool_response = json.loads(json_str)
                        
                        # Process the tool response (same logic as Groq success path)
                        chosen_pattern = tool_response.get('chosen_pattern', '')
                        source_properties = tool_response.get('source_properties', [])
                        target_properties = tool_response.get('target_properties', [])
                        reasoning = tool_response.get('reasoning', '')
                        
                        logger.info(f"🔄 FALLBACK: OpenAI chosen pattern: {chosen_pattern}")
                        logger.info(f"🔄 FALLBACK: Source properties: {source_properties}")
                        logger.info(f"🔄 FALLBACK: Target properties: {target_properties}")
                        
                        # Convert to filters and build query
                        source_filters = []
                        target_filters = []
                        
                        for prop_obj in source_properties:
                            if isinstance(prop_obj, dict) and 'property' in prop_obj and 'value' in prop_obj:
                                source_filters.append({
                                    'property': prop_obj['property'],
                                    'operator': prop_obj.get('operator', 'CONTAINS'),
                                    'value': prop_obj['value']
                                })
                        
                        for prop_obj in target_properties:
                            if isinstance(prop_obj, dict) and 'property' in prop_obj and 'value' in prop_obj:
                                target_filters.append({
                                    'property': prop_obj['property'],
                                    'operator': prop_obj.get('operator', 'CONTAINS'),
                                    'value': prop_obj['value']
                                })
                        
                        # Build Cypher query using template system
                        generated_query = self.build_cypher_from_pattern(
                            selected_pattern=chosen_pattern,
                            source_filters=source_filters,
                            target_filters=target_filters,
                            user_id=user_id
                        )
                        
                        logger.info(f"🔄 FALLBACK SUCCESS: Generated query via OpenAI {fallback_model}: {generated_query}")
                        return generated_query, True, {}  # Successfully generated via OpenAI fallback (no enhancement params)
                        
                    else:
                        logger.error(f"🔄 FALLBACK FAILED: OpenAI also didn't make tool call")
                        raise Exception("OpenAI fallback also failed to make tool call")
                        
                except asyncio.TimeoutError:
                    logger.error(f"🔄 FALLBACK TIMEOUT: OpenAI fallback timed out after 20 seconds")
                    logger.error(f"🔄 FALLBACK TIMEOUT: This means OpenAI API call hung or took too long")
                    # Final fallback: simple base query
                    left_label = available_node_enums[0].value if available_node_enums else 'Memory'
                    right_label = available_node_enums[1].value if len(available_node_enums) > 1 else left_label
                    generated_query = f"MATCH (m:{left_label})-[r]-(n:{right_label}) RETURN m, r, n LIMIT {top_k}"
                    logger.warning(f"🔄 FINAL FALLBACK: Using simple base query with LIMIT: {generated_query}")
                    return generated_query, False, {}  # Final fallback query (no enhancement params)
                except Exception as fallback_error:
                    logger.error(f"🔄 FALLBACK FAILED: OpenAI fallback also failed: {fallback_error}")
                    logger.error(f"🔄 FALLBACK FAILED: Error type: {type(fallback_error).__name__}")
                    # Final fallback: simple base query
                    left_label = available_node_enums[0].value if available_node_enums else 'Memory'
                    right_label = available_node_enums[1].value if len(available_node_enums) > 1 else left_label
                    generated_query = f"MATCH (m:{left_label})-[r]-(n:{right_label}) RETURN m, r, n LIMIT {top_k}"
                    logger.warning(f"🔄 FINAL FALLBACK: Using simple base query with LIMIT: {generated_query}")
                    return generated_query, False, {}  # Final fallback query (no enhancement params)
            
            
            
            # Extract ACL values from the filter (robust to non-dict inputs)
            acl_conditions = []
            try:
                or_list = []
                if isinstance(acl_filter, dict):
                    or_list = acl_filter.get('$or', [])
                elif hasattr(acl_filter, 'model_dump'):
                    dumped = acl_filter.model_dump()
                    or_list = dumped.get('$or', [])
                for condition in or_list:
                    if not isinstance(condition, dict):
                        continue
                    if 'user_id' in condition:
                        acl_conditions.append("m.user_id = $user_id")
                    elif 'user_read_access' in condition:
                        acl_conditions.append("m.user_read_access IS NOT NULL AND any(x IN m.user_read_access WHERE x IN $user_read_access)")
                    elif 'workspace_read_access' in condition:
                        acl_conditions.append("m.workspace_read_access IS NOT NULL AND any(x IN m.workspace_read_access WHERE x IN $workspace_read_access)")
                    elif 'role_read_access' in condition:
                        acl_conditions.append("m.role_read_access IS NOT NULL AND any(x IN m.role_read_access WHERE x IN $role_read_access)")
            except Exception as e:
                logger.warning(f"Failed to parse acl_filter for Cypher WHERE: {e}")

            base_query = generated_query.strip() if generated_query else "MATCH (m:Memory)-[r]-(n:Memory)"
            
            # Template system already builds the complete query with ACL, RETURN, and all necessary clauses
            # Just add LIMIT if not already present
            if not generated_query.strip().upper().endswith('RESULT'):
                generated_query += f"\nLIMIT {top_k}"
            else:
                generated_query += f"\nLIMIT {top_k}"
            
            # NEW: Apply property enhancement to ALL LLM-generated queries (both template and instructor paths)
            if enhanced_schema_cache and generated_query:
                logger.info(f"🚀 PROPERTY ENHANCEMENT: Attempting to enhance LLM-generated query")
                try:
                    enhanced_query = await self._enhance_query_with_property_suggestions(
                        cypher_query=generated_query,
                        user_query=user_query,
                        acl_filter=acl_filter,
                        enhanced_schema_cache=enhanced_schema_cache,
                        memory_graph=memory_graph,
                        neo_session=neo_session
                    )
                    if enhanced_query:
                        generated_query = enhanced_query
                        logger.info(f"🚀 PROPERTY ENHANCEMENT: Successfully enhanced query with Qdrant property suggestions")
                    else:
                        logger.info(f"🚀 PROPERTY ENHANCEMENT: No enhancements applied (no matching properties found)")
                except Exception as enhancement_error:
                    logger.warning(f"🚀 PROPERTY ENHANCEMENT: Failed to enhance query: {enhancement_error}")
                    # Continue with original query if enhancement fails
            else:
                if not enhanced_schema_cache:
                    logger.info(f"🚀 PROPERTY ENHANCEMENT: Skipped (no enhanced_schema_cache)")
                if not generated_query:
                    logger.info(f"🚀 PROPERTY ENHANCEMENT: Skipped (no generated_query)")

            logger.info(f"Final LLM generated query: {generated_query}")
            return generated_query, True, {}  # True indicates LLM-generated query (no enhancement params for static path)

        except Exception as e:
            logger.warning(f"Error in query generation, returning empty query: {str(e)}")
            return "", False, {}  # False indicates no query generated

    def _generate_cypher_from_structure(
        self, 
        query_structure: dict, 
        bigbird_memory_ids: list, 
        acl_filter: ACLFilter
    ) -> str:
        """Convert structured query components into a Cypher query string."""
        try:
            query_parts = []
            
            # Process MATCH clause with all relationship types from memory graph
            for match in query_structure['match']:
                # Get all relationship types and join them with '|'
                rel_types = match.get('relationships', [])
                if rel_types:
                    rel = rel_types[0]  # Get the first relationship definition
                    # Create relationship pattern with all possible types
                    rel_pattern = f"-[{rel['variable']}:{rel['type']}]-"
                    if rel['direction'] == '->':
                        rel_pattern += '>'
                    elif rel['direction'] == '<-':
                        rel_pattern = '<' + rel_pattern
                    
                    # Create node patterns
                    source_node = f"({match['variable']}:{match['label']})"
                    target_node = f"({rel['target']['variable']}:{rel['target']['label']})"
                    
                    # Combine into full pattern
                    pattern = f"{source_node}{rel_pattern}{target_node}"
                else:
                    pattern = f"({match['variable']}:{match['label']})"
                query_parts.append(f"MATCH {pattern}")
            
            # Process WHERE clause with ACL conditions
            where_conditions = []
            cypher_acl_conditions = self.acl_filter_to_cypher_conditions(acl_filter)
            
            # Always include bigbird_memory_ids and ACL conditions
            where_conditions.append("m.id IN $bigbird_memory_ids")
            where_conditions.append(f"({cypher_acl_conditions})")
            
            # Add relationship type condition using all available types
            if rel_types:
                where_conditions.append(f"type(r) IN {str(rel_types[0]['type'])}")
            
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")
            
            # Process RETURN clause
            returns = ", ".join(query_structure['return'])
            query_parts.append(f"RETURN {returns}")
            
            # Process ORDER BY clause
            if 'order_by' in query_structure and query_structure['order_by']:
                order_parts = [
                    f"{item['expression']} {item['direction']}"
                    for item in query_structure['order_by']
                ]
                query_parts.append(f"ORDER BY {', '.join(order_parts)}")
            
            # Process LIMIT
            if 'limit' in query_structure:
                query_parts.append(f"LIMIT {query_structure['limit']}")
            
            query = "\n".join(query_parts)
            
            # Validate the query
            parameters = {
                'bigbird_memory_ids': bigbird_memory_ids
            }
            self.validate_cypher_query(query, parameters)
            
            return query
            
        except Exception as e:
            logger.error(f"Error generating Cypher query: {str(e)}")
            raise ValueError("Failed to generate valid Cypher query")

    def _generate_match_pattern(self, match_element: dict) -> str:
        """Generate a MATCH pattern for a single node and its relationships."""
        node_pattern = f"({match_element['variable']}:{match_element['label']})"
        
        if 'relationships' in match_element and match_element['relationships']:
            for rel in match_element['relationships']:
                rel_pattern = f"-[{rel['variable']}:{rel['type']}]-"
                if rel['direction'] == '->':
                    rel_pattern += '>'
                else:
                    rel_pattern = '<' + rel_pattern
                    
                target = rel['target']
                target_pattern = f"({target['variable']}:{target['label']})"
                node_pattern += f"{rel_pattern}{target_pattern}"
        
        return node_pattern
        
    def validate_query_properties(self, query: str, node_properties: list) -> bool:
        # Extract properties used in the query
        #used_properties = extract_properties_from_query(query)
        # Check if all used properties are in the node_properties list
        #for prop in used_properties:
         #   if prop not in node_properties:
          #      logger.warning(f"Property '{prop}' is not in node properties.")
           #     return False
        return True


    def fallback_cipher_query(
        self,
        bigbird_memory_ids: list,
        acl_filter: ACLFilter,
        memory_graph: dict,
        top_k: int,
        memory_graph_schema: Dict[str, Any] = None
    ) -> str:
        """
        Generate a fallback Cypher query if LLM fails, mirroring the original functionality.
        IMPORTANT: Applies ACL conditions to BOTH m and n nodes for multi-tenant isolation.
        """
        try:
            cipher_relationship_types = memory_graph_schema.get('relationships', []) if memory_graph_schema else []
            sanitized_relationship_types = [rel_type.replace('-', '_') for rel_type in cipher_relationship_types] if cipher_relationship_types else []
            
            # Generate ACL conditions for BOTH nodes
            cypher_acl_conditions_m = self.acl_filter_to_cypher_conditions(acl_filter, node_alias="m")
            cypher_acl_conditions_n = self.acl_filter_to_cypher_conditions(acl_filter, node_alias="n")

            # Build the MATCH clause
            if sanitized_relationship_types:
                # Create the relationship type string, including additional types if they exist
                relationship_type_string = "RELATION" + (f"|{'|'.join(sanitized_relationship_types)}")
                # Start with the base MATCH clause for the Memory nodes and RELATION
                cipher_query = f"""
                MATCH (m:Memory)-[r:{relationship_type_string}]->(n:Memory)
                WHERE m.id IN $bigbird_memory_ids
                """
                # Incorporate the ACL conditions for BOTH m and n
                if cypher_acl_conditions_m:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_m})
                    """
                if cypher_acl_conditions_n:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_n})
                    """
                cipher_query += f"""
                AND r.type IN $sanitized_relationship_types
                """
            else:
                # Use variable-length relationships as in your original code
                cipher_query = f"""
                MATCH (m:Memory)-[r:RELATION]->(n:Memory)
                WHERE m.id IN $bigbird_memory_ids
                """
                if cypher_acl_conditions_m:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_m})
                    """
                if cypher_acl_conditions_n:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_n})
                    """

            # Complete the query with the RETURN and ORDER BY clauses
            cipher_query += f"""
            RETURN m, r, n
            ORDER BY COALESCE(n.createdAt, n.id) DESC
            LIMIT {top_k}
            """

            logger.info(f"Fallback Cipher Query with ACL for both nodes: {cipher_query}")
            return cipher_query.strip()
        except Exception as e:
            logger.error(f"Error generating fallback Cypher query: {e}")
            return None

    async def fallback_cipher_query_async(
        self,
        bigbird_memory_ids: list,
        acl_filter: Dict[str, Any],  # Changed type hint to Dict
        memory_graph: dict,
        top_k: int,
        memory_graph_schema: Dict[str, Any] = None
    ) -> str:
        """
        Generate a fallback Cypher query if LLM fails, mirroring the original functionality.
        Async version mainly for consistency, as this is a computational function without I/O.
        """
        try:
            cipher_relationship_types = memory_graph_schema.get('relationships', []) if memory_graph_schema else []
            sanitized_relationship_types = [rel_type.replace('-', '_') for rel_type in cipher_relationship_types] if cipher_relationship_types else []
            
            # Build ACL conditions for BOTH nodes (m and n)
            acl_conditions_m = []
            acl_conditions_n = []
            acl_or_conditions = acl_filter.get('$or', [])
            
            for condition in acl_or_conditions:
                if 'user_id' in condition:
                    acl_conditions_m.append("m.user_id = $user_id")
                    acl_conditions_n.append("n.user_id = $user_id")
                elif 'user_read_access' in condition:
                    acl_conditions_m.append("m.user_read_access IN $user_read_access")
                    acl_conditions_n.append("n.user_read_access IN $user_read_access")
                elif 'workspace_read_access' in condition:
                    acl_conditions_m.append("m.workspace_read_access IN $workspace_read_access")
                    acl_conditions_n.append("n.workspace_read_access IN $workspace_read_access")
                elif 'role_read_access' in condition:
                    acl_conditions_m.append("m.role_read_access IN $role_read_access")
                    acl_conditions_n.append("n.role_read_access IN $role_read_access")
            
            cypher_acl_conditions_m = " OR ".join(acl_conditions_m) if acl_conditions_m else ""
            cypher_acl_conditions_n = " OR ".join(acl_conditions_n) if acl_conditions_n else ""

            # Build the MATCH clause
            if sanitized_relationship_types:
                # Create the relationship type string
                relationship_type_string = "|".join(sanitized_relationship_types)
                # Start with the base MATCH clause
                cipher_query = f"""
                MATCH (m:Memory)-[r:RELATION]-(n:Memory)
                MATCH path = (p)-[*1..2]-(n)
                WHERE m.id IN $bigbird_memory_ids
                """
                # Add ACL conditions for BOTH m and n
                if cypher_acl_conditions_m:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_m})
                    """
                if cypher_acl_conditions_n:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_n})
                    """
                cipher_query += f"""
                AND type(r) IN $sanitized_relationship_types
                """
            else:
                # Use basic RELATION relationship
                cipher_query = f"""
                MATCH (m:Memory)-[r]->(n:Memory)
                WHERE m.id IN $bigbird_memory_ids
                """
                if cypher_acl_conditions_m:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_m})
                    """
                if cypher_acl_conditions_n:
                    cipher_query += f"""
                    AND ({cypher_acl_conditions_n})
                    """

            # Complete the query
            cipher_query += f"""   
            ORDER BY COALESCE(n.createdAt, n.id) DESC
            RETURN DISTINCT path
            LIMIT {top_k}
            """

            logger.info(f"Fallback Cipher Query with ACL for both nodes (async): {cipher_query}")
            return cipher_query.strip()
        except Exception as e:
            logger.error(f"Error generating fallback Cypher query: {e}")
            return None


    def get_memory_for_text(self, sessionToken: str, text, context=None, project_id=None, user_id=None, use_case=None, memory_graph=None):
        """
        Analyze the text and get memory for highlighted keywords using ChatGPT completion API with function calling capability.

        :param text: The text selected by the user.
        :param context: Optional context for the query.
        :param user_id: Optional user ID for personalized queries.
        :param project_id: Optional project ID related to the query.
        :param use_case: Optional use case for the query.
        :param memory_graph: Optional existing memory graph.
        :return: JSON with highlighted text, location, and retrieved memory item.
        """
        logger.info("Starting get_memory_for_text method")

        # Define the function for the API to call
        get_memory = {
            "type": "function",
            "function": {
                "name": "get_memory",
                "description": "Retrieve a memory item from the user's personal memory graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search an item in memory.",
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Relationship type given query and context that defines relationship type between memory items you want to find in memory.",
                        },
                        "context": {
                            "type": "array",
                            "description": "Context can be conversation history or any relevant context for a memory item.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "Person who created the memory item in context.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of a memory item or conversation that is related to query.",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        }

                    },
                    "required": ["query"],
                },
            },
        }

        # Construct the prompt for analyzing the text
        prompt = f"Analyze the following text and identify the set of keywords to highlight for retrieving information from memory: '{text}'."
        if context:
            prompt += f" Use the context provided: {context}."
        if project_id:
            prompt += f" This is related to project ID {project_id}."
        if use_case:
            prompt += f" The use-case is {use_case}."
        if memory_graph:
            prompt += f" Given the memory graph: {json.dumps(memory_graph)}."
        
        logger.info(f"Constructed prompt: {prompt}")


        # Prepare the initial messages for the conversation
        messages = [
            {"role": "system", "content": "You are a memory assistant."},
            {"role": "user", "content": prompt}
        ]
        tools = [get_memory]


        # Process the API's response
        try:
            logger.info(f"Process the API's response: model: {self.model}, messages: {messages}, tools: {tools}")
            # Call the OpenAI API with function calling
            response = self.client.chat.completions.create(
                model=self.model,  # Use the appropriate model identifier
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_memory"}},
                temperature=0.7
                
            )

            logger.info(f"response from model: {response}")

            response_message = response.choices[0].message
            logger.info(f"Response message: {response_message}")

            tool_calls = response_message.tool_calls
            logger.info(f"Tool calls found: {tool_calls}")


            if tool_calls:

                # Note: the JSON response may not always be valid; be sure to handle errors
                available_functions = {
                    "get_memory": get_memory,
                }  # only one function in this example, but you can have multiple

                # Append the function response to the conversation
                messages.append(response_message)

                # Process tool_calls
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)

                    # Set default values for relation_type and context
                    relation_type = function_args.get('relation_type', None)
                    function_context = function_args.get('context', context if context else {})

                    logger.info(f"Calling get_memory with args: {function_args}")

                     # Prepare arguments for find_related_memory_items method
                    find_memory_args = {
                        'query': function_args['query'],
                        'context': function_context,
                        'user_id': user_id,
                        'chat_gpt': self,
                        'metadata': None,
                        'relation_type': relation_type,
                        'project_id': project_id,
                        'skip_neo': False
                    }
                    
                    context = context if context else {}  # Use the context as is
                    function_args['context'] = context

                    # Call the find_related_memory_items method with updated arguments
                    function_response = self.memory_graph.find_related_memory_items(sessionToken, **find_memory_args)
                    logger.info(f"got function_response: {function_response}")

                     # Format the function response as JSON string
                    formatted_function_response = json.dumps(function_response, indent=2)


                    # Add the function response to messages
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": formatted_function_response,
                    })

                    # Construct a follow-up prompt that asks for the response in JSON format with the given example
                    follow_up_prompt = {
                        "role": "user",
                        "content": "Use context from this user's selected text: {text} Please select ONLY ONE memory item from the retrieved list of memory items and make sure that memory item doesn't have exactly the same content as the selected text provided. Format your response as a JSON object similar to this example. If a field in memoryItem isn't listed please don't inlude it in the final response. " + json.dumps({
                            "highlightedText": "Top use-cases",
                            "UpdatedText":"Update user's selected text and enhance it using the memory item retreived",
                            "memoryItem": {
                                "content": "Top use-cases include RPG games, developers building apps, creative writing - books, daily diary and more!",
                                "metadata": {
                                    "createdAt": "2023-09-18",
                                    "emoji tags": "🏰🎮",
                                    "emotion tags": "anticipation",
                                    "hierarchical structures": "Gameplay Choices",
                                    "id": "1f66ea97-187d-40fb-a394-bb3eba64824c",
                                    "imageGenerationCategory": "['rpg_action']",
                                    "imageURL": "https://parseserverstoragewest.blob.core.windows.net/parse/d8dbf8453016bbc8356ca85f5e845aa2_1bc65b04-06c0-440e-bbcb-5d771aca9f55.png",
                                    "location": "Mojave Wasteland",
                                    "project_id": "None",
                                    "prompt": "\"Player choosing a castle location in the Mojave Wasteland for a Fallout RPG adventure.\"",
                                    "topics": "Fallout RPG, Game Location, Mojave Wasteland, Castle",
                                    "type": "TextMemoryItem",
                                    "user_id": "QgSyTCZJzO"
                                }
                                
                            }
                                                    
                        }, indent=2)  # The indent parameter is optional, just for better readability if you print or log it
                    }

                    messages.append(follow_up_prompt)

                    # Call the API again to process the function response
                    second_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={ "type": "json_object" },
                        temperature=0.7
                    )
                    logger.info(f"Received second response from OpenAI: {second_response}")
            
                    # Extract the final response
                    final_message = second_response.choices[0].message
                    logger.info(f"Received second response from OpenAI: {final_message.content}")

                    try:
                        # Parse the JSON string into a Python dictionary
                        response_content = json.loads(final_message.content)

                        # Extract the relevant data from the response_content
                        final_message_data = {
                            "highlightedText": response_content.get("highlightedText", ""),
                            "UpdatedText": response_content.get("UpdatedText", ""),  # Using .get() method with a default value
                            "memoryItem": response_content.get("memoryItem", "")
                        }

                        # Now final_message_data is a dictionary that can be serialized
                        logger.info(f"Received second response from OpenAI: {final_message_data}")

                        return final_message_data

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error: {e}")
                        # Handle the error or return an appropriate response

                    response_content = final_message.content
                        
                else:
                    logger.warning("No tool calls found in the response.")
                    return {"error": "No tool call was made in the response."}

                
        except Exception as e:
            logger.error(f"Error in get_memory_for_text: {e}")
            raise

    def get_instructions_for_agent(self, agent):
        """
        Return specific instructions based on the agent type using a dictionary to mimic a switch statement.

        :param agent: The type of agent.
        :return: Instructions string or None if invalid agent.
        """
        instructions = {
            'Product Agent': "You are an expert Product manager reviewing a problem definition document in detail. A successful document discusses the current state and justifies the priority of a problem that UX design and engineering can solve as a next step, and doesn't discuss the solution. *Instructions* Before we start the review let's get relevant memories that can help us improve the quality of this document. Write a detailed query that we can use to retreive customer conversations or analytics we have on customer usage and behavior. Respond only in JSON format ",
            'Engineering Agent': "You are an expert Engineering lead reviewing a problem definition document in detail. A successful document doesn't discuss the solution but gives engineers problem context that lets them better develop a product. *Instructions* Return suggestions in JSON format with: feedback_area, sentence, question, question_reason, example",
            'UX Design Agent': "You are an expert UX Designer reviewing a problem definition document in detail. A successful document doesn't discuss the solution but specifies the problem so designers can design an experience to solve it. *Instructions* Return suggestions in JSON format with: feedback_area, sentence, question, question_reason, example",
            'Project Manager Agent': "You are an expert Project manager reviewing a problem definition document in detail. A successful document doesn't discuss the solution but gives a team clear problem to solve. *Instructions* Return suggestions in JSON format with: feedback_area, sentence, question, question_reason, example",
            'Product Enablement Agent': "You are an expert Product Enablement manager reviewing a problem definition document in detail. A successful document doesn't discuss the solution in detail. Instead it justifies the problem and includes sections: problem, target customer, current state, desired future state, metrics we should track, and counterargument. *Instructions* Return suggestions in JSON format with: feedback_area, sentence, question, question_reason, example",
            'Analyst Agent': "Act as an expert data Analyst Agent reviewing a problem brief in detail. The goal of the problem brief is to define the problem only, not yet the solution so focus your review on whether the document properly articulates the problem and justifies it. A better problem brief will help the design and engineering team be more successful in discovering a solution to the problem in the brief. Follow the following steps when reviewing the document: Return suggestions in JSON format with the following keys: feedback_area, sentence, question, question_reason, example"
        }

        return instructions.get(agent, None)  # Return None if the agent is not specified in the dictionary


    def review_page_with_memories(self, sessionToken: str, user_id: str, text: any, agent: any, project_id: str = None):
        """
        Review the content of a page and provide structured feedback based on the agent's perspective.

        :param text: The text content of the page.
        :param agent: The type of agent (e.g., 'Product Agent', 'Engineering Agent').
        :return: JSON formatted feedback with suggestions.
        """
         # Define the function for the API to call
        get_memory = {
            "type": "function",
            "function": {
                "name": "get_memory",
                "description": "Retrieve a memory item from the user's personal memory graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Detailed query to search in the user's memory. Write the query in sentence format and include 2-3 sentences if needed.",
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Relationship type given query and context that defines relationship type between memory items you want to find in memory.",
                        },
                        "context": {
                            "type": "array",
                            "description": "Context can be conversation history or any relevant context for a memory item.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "Person who created the memory item in context.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of a memory item or conversation that is related to query.",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        }

                    },
                    "required": ["query", "context"],
                },
            },
        }

        
        # Define the instructions based on the agent type

        try:
            instructions = self.get_instructions_for_agent(agent)
            logger.info(f"Instructions: {instructions}")

            default_instructions = self.get_instructions_for_agent("Product Agent")
            logger.info(f"default_instructions: {default_instructions}")

        except Exception as e:
            logger.error("Error fetching instructions for agent:", str(e))

        messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": text}
                ]
        tools = [get_memory]
        logger.info("Starting review_page method")

         # Process the API's response
        try:
            
            # Call the OpenAI API with function calling
            response = self.client.chat.completions.create(
                model=self.model,  # Use the appropriate model identifier
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_memory"}},
                temperature=0.7
            )

            logger.info(f"response from model: {response}")

            response_message = response.choices[0].message
            logger.info(f"Response message: {response_message}")

            tool_calls = response_message.tool_calls
            logger.info(f"Tool calls found: {tool_calls}")


            if tool_calls:

                # Note: the JSON response may not always be valid; be sure to handle errors
                available_functions = {
                    "get_memory": get_memory,
                }  # only one function in this example, but you can have multiple

                # Append the function response to the conversation
                messages.append(response_message)

                # Process tool_calls
                for tool_call in tool_calls:
                    if tool_call.function.arguments:                    
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)

                        # Set default values for relation_type and context
                        relation_type = function_args.get('relation_type', None)
                        function_context = function_args.get('context', None)
                        logger.info(f"function_context: {function_context}")

                        logger.info(f"Calling get_memory with args: {function_args}")

                        # Prepare arguments for find_related_memory_items method
                        find_memory_args = {
                            'query': function_args['query'],
                            'context': function_context,
                            'user_id': user_id,
                            'chat_gpt': self,
                            'metadata': None,
                            'relation_type': relation_type,
                            'project_id': project_id,
                            'skip_neo': False
                        }

                        # Call the find_related_memory_items method with updated arguments
                        function_response = self.memory_graph.find_related_memory_items(sessionToken, **find_memory_args)
                        logger.info(f"got function_response: {function_response}")

                        # Format the function response as JSON string
                        formatted_function_response = json.dumps(function_response, indent=2)


                        # Add the function response to messages
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": formatted_function_response,
                        })

                        # Construct a follow-up prompt that asks for the response in JSON format with the given example
                        follow_up_prompt = {
                            "role": "user",
                            "content": "Use context from this user's selected text that we need to review: {text} Please include up to 3 relevant memories that you can use in the re-written text. Also suggest updates to up to 3 sentences in this text. Format your response as a JSON object similar to this example. If a field in memoryItem isn't listed please don't inlude it in the final response. " + json.dumps({ 
                                "review": [
                                    { 
                                        "feedback_area": "Problem Statement Structure",
                                        "highlightedText": "Select a sentence from the user's input text that has a feedback area that we also want to highlight to the user",
                                        "question": "Can the solution details be removed from the problem statement section?",
                                        "question_reason": "The problem statement section should focus on describing the problem in detail and justifying its importance, without suggesting or detailing a solution. Discussing the solution in the problem statement can lead to biased problem understanding.",
                                        "example": "Instead of mentioning the Chrome Extension, the document should elaborate on why current methods are cumbersome, perhaps by detailing the steps users currently take and the specific pain points they experience with those methods.",
                                        "rewrite":"Re-write the highlightedText using the feedback, questions and relevant memories to improve quality." ,
                                        "related_memories": [
                                            {
                                                "objectId": "n31SW9VEIY"
                                            }
                                        ]

                                    }                                
                                ]
                                
                                                                                
                            }, indent=2)  # The indent parameter is optional, just for better readability if you print or log it
                        }

                        messages.append(follow_up_prompt)

                        # Call the API again to process the function response
                        second_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={ "type": "json_object" },
                            temperature=0.7
                        )
                        logger.info(f"Received second response from OpenAI: {second_response}")
                
                        # Extract the final response
                        final_message = second_response.choices[0].message
                        logger.info(f"Received second response from OpenAI: {final_message.content}")
                        try:
                            # Parse the JSON string into a Python dictionary
                            response_content = json.loads(final_message.content)

                            # Check if function_response is a string and parse it into a dictionary if so
                            if isinstance(function_response, str):
                                function_response = json.loads(function_response)

                            # Ensure function_response is a list before proceeding
                            if isinstance(function_response, list):
                                # Directly add the list to response_content under the 'review' key
                                if 'review' in response_content:
                                    # Extend the existing 'review' list with the new items
                                    response_content['review'].extend(function_response)
                                else:
                                    # Or create a new 'review' list if it doesn't exist
                                    response_content['review'] = function_response
                            else:
                                logger.error("Expected function_response to be a list, received: {}".format(type(function_response)))
                                return {"error": "Invalid format for function_response"}

                            return response_content

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {e}")
                            return {"error": "Error parsing JSON data"}

                        except Exception as e:
                            logger.error(f"Unexpected error processing the response: {e}")
                            return {"error": "An error occurred processing the response"}



                            
                    else:
                        logger.warning("No tool calls found in the response.")
                        return {"error": "No tool call was made in the response."}

                
        except Exception as e:
            logger.error(f"Error in get_memory_for_text: {e}")
            raise


    def review_page(self, sessionToken: str, user_id: str, text: any, agent: any, project_id: str = None, reviewText: str = None):
        """
        Review the content of a page and provide structured feedback based on the agent's perspective.

        :param text: The text content of the page.
        :param agent: The type of agent (e.g., 'Product Agent', 'Engineering Agent').
        :return: JSON formatted feedback with suggestions.
        """

        # Construct a follow-up prompt that asks for the response in JSON format with the given example
        try:
            review_json = json.dumps({ 
                "review": [
                    { 
                        "feedback_area": "Problem Statement Structure",
                        "sentence": "Select a sentence from the user's input text that has a feedback area that we also want to highlight to the user",
                        "question": "Can the solution details be removed from the problem statement section?",
                        "question_reason": "The problem statement section should focus on describing the problem in detail and justifying its importance, without suggesting or detailing a solution. Discussing the solution in the problem statement can lead to biased problem understanding.",
                        "example": "Instead of mentioning the Chrome Extension, the document should elaborate on why current methods are cumbersome, perhaps by detailing the steps users currently take and the specific pain points they experience with those methods."
                    }                                
                ]   
            }, indent=2)
        except TypeError as e:
            print("A TypeError occurred:", str(e))

            
         
        # Define the instructions based on the agent type
        try:
            instructions = self.get_instructions_for_agent(agent)
            logger.info(f"Instructions: {instructions}")

        except Exception as e:
            print("Error fetching instructions for agent:", str(e))

         # Constructing the prompt
        prompt_parts = [
            f"Review Prompt: {reviewText}" if reviewText else "",
            f"Use context from this user's selected text that we need to review: {text} Please include up to 3 relevant memories that you can use in the re-written text. Also suggest updates to up to 3 sentences in this text. Format your response as a JSON object similar to this example. If a field in memoryItem isn't listed please don't inlude it in the final response. ",
            f"{review_json}"
            
        ]
        prompt = " ".join(filter(None, prompt_parts))  # Filter out any empty strings
        logger.info(f"prompt: {prompt}")


        messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
        ]
        logger.info(f"messages: {messages}")

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")

        # Calculate available tokens for completion, ensuring not to exceed 128000 tokens
        # Subtracting a small buffer (e.g., 7 tokens) to account for any calculation discrepancies
        max_tokens_for_completion = min(4096, 128000 - token_count_prompt - 7)
        logger.info(f"max_tokens_for_completion: {max_tokens_for_completion}")

        # Check if the calculated max_tokens_for_completion is below the minimum required for a meaningful response
        if max_tokens_for_completion < 1024:  # Example threshold, adjust as needed
            raise ValueError("Available tokens for completion are too few for a meaningful response.")

        # Ensure the max_tokens parameter in the API request does not exceed the calculated limit
        try:
            response = self._create_completion_sync(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens_for_completion,   
                temperature=0.7,
                response_format={ "type": "json_object" }

            )

            # Check if the response has choices and message content
            if response.choices and response.choices[0].message.content:
                try:
                            
                    response_content = response.choices[0].message.content
                    logger.info(f"response_content: {response_content}")
                    # Parse the response content as JSON

                    # Count output tokens
                    token_count_output = self.count_tokens(response_content)
                    logger.info(f"token_count_output: {token_count_output}")

                    # Calculate and log the total cost
                    total_cost = self.calculate_cost(token_count_prompt, token_count_output)
                    logger.info(f"Total cost: ${total_cost:.4f}")


                    final_response = json.loads(response_content)
                    return final_response
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}. Response content: ```json\n{response_content}\n```")
                    raise
            else:
                logger.error(f"Unexpected response or empty content. Response: {response}")
                return None
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise

    
    def generate_content(self, sessionToken: str, user_id: str, prompt: any, option: any, command: str):
        """
        Generate content based on the given params.

        :param prompt: the user's prompt.
        :param option: The option the user chose (e.g., 'Summarize', 'Rephrase').
        :return: markdown formatted feedback with content generated.
        """

        # Define prompt templates for each option
        options = {
            "continue": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that continues existing text based on context from prior text. "
                    "Give more weight/priority to the later characters than the beginning ones. "
                    "Limit your response to no more than 200 characters, but make sure to construct complete sentences. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": prompt}
            ],
            "improve": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that improves existing text. "
                    "Limit your response to no more than 200 characters, but make sure to construct complete sentences. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "shorter": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that shortens existing text. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "longer": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that lengthens existing text. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "fix": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that fixes grammar and spelling errors in existing text. "
                    "Limit your response to no more than 200 characters, but make sure to construct complete sentences. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "zap": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that generates text based on a prompt. "
                    "You take an input from the user and a command for manipulating the text "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"For this text: {prompt}. You have to respect the command: {command}"}
            ]
        }

        if option not in options:
                logger.error(f"Invalid option provided: {option}")

        messages = options[option]
        logger.info(f"messages: {messages}")

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")

        # Calculate available tokens for completion, ensuring not to exceed 128000 tokens
        # Subtracting a small buffer (e.g., 7 tokens) to account for any calculation discrepancies
        max_tokens_for_completion = min(4096, 128000 - token_count_prompt - 7)
        logger.info(f"max_tokens_for_completion: {max_tokens_for_completion}")

        # Check if the calculated max_tokens_for_completion is below the minimum required for a meaningful response
        if max_tokens_for_completion < 1024:  # Example threshold, adjust as needed
            raise ValueError("Available tokens for completion are too few for a meaningful response.")

        # Ensure the max_tokens parameter in the API request does not exceed the calculated limit
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens_for_completion,   
                temperature=0.5,
                # response_format={ "type": "json_object" }
            )

            # Check if the response has choices and message content
            if response.choices and response.choices[0].message.content:
                try:
                            
                    response_content = response.choices[0].message.content
                    logger.info(f"response_content: {response_content}")
                    # Parse the response content as JSON

                    # Count output tokens
                    token_count_output = self.count_tokens(response_content)
                    logger.info(f"token_count_output: {token_count_output}")

                    # Calculate and log the total cost
                    total_cost = self.calculate_cost(token_count_prompt, token_count_output)
                    logger.info(f"Total cost: ${total_cost:.4f}")


                    final_response = response_content
                    return final_response
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}. Response content: ```json\n{response_content}\n```")
                    raise
            else:
                logger.error(f"Unexpected response or empty content. Response: {response}")
                return None
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise

    def get_memories_for_a_review(self, sessionToken: str, user_id: str, text: any, agent: any, review_context: any, project_id: str = None):
        """
        Review the content of a page and provide structured feedback based on the agent's perspective.

        :param text: The text content of the page.
        :param agent: The type of agent (e.g., 'Product Agent', 'Engineering Agent').
        :return: JSON formatted feedback with suggestions.
        """
         # Define the function for the API to call
        get_memory = {
            "type": "function",
            "function": {
                "name": "get_memory",
                "description": "Retrieve a memory item from the user's personal memory graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Detailed query to search in the user's memory. Write the query in sentence format and include 2-3 sentences if needed.",
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Relationship type given query and context that defines relationship type between memory items you want to find in memory.",
                        },
                        "context": {
                            "type": "array",
                            "description": "Context can be conversation history or any relevant context for a memory item.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "Person who created the memory item in context.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of a memory item or conversation that is related to query.",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        }

                    },
                    "required": ["query"],
                },
            },
        }


        review_context_json = None

        try:
            # Attempt to create JSON string from review_context
            review_context_json = json.dumps(review_context, indent=2)
        except Exception as e:
            logger.error(f"Error serializing review_context to JSON: {e}")
            # Handle or return an error response if necessary
            return {"error": f"Failed to serialize review_context: {str(e)}"}

        instructions = self.get_instructions_for_agent(agent)
        logger.info(f"Instructions: {instructions}")

        # Constructing the prompt
        prompt_parts = [
            f"User's highlighted text: {text}",
            f"review_context: {review_context_json}" if review_context_json else ""
        ]
        prompt = " ".join(filter(None, prompt_parts))  # Filter out any empty strings

        messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                ]
        tools = [get_memory]

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")


        # Serialize the total input to JSON for token counting (messages and tools)
        total_input = json.dumps({"messages": messages, "tools": tools})
        total_input_token_count = self.count_tokens(total_input)
        logger.info(f"Total input token count (including messages and tools): {total_input_token_count}")
        logger.info("Starting review_page method")

         # Process the API's response
        try:
            
            # Call the OpenAI API with function calling
            response = self.client.chat.completions.create(
                model=self.model,  # Use the appropriate model identifier
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_memory"}},
                temperature=0.7
            )

            logger.info(f"response from model: {response}")

            response_message = response.choices[0].message
            logger.info(f"Response message: {response_message}")

            tool_calls = response_message.tool_calls
            logger.info(f"Tool calls found: {tool_calls}")


            if tool_calls:

                # Note: the JSON response may not always be valid; be sure to handle errors
                available_functions = {
                    "get_memory": get_memory,
                }  # only one function in this example, but you can have multiple

                # Append the function response to the conversation
                messages.append(response_message)

                # Process tool_calls
                for tool_call in tool_calls:
                    if tool_call.function.arguments:                    
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)

                        # Set default values for relation_type and context
                        relation_type = function_args.get('relation_type', None)
                        function_context = function_args.get('context', None)
                        logger.info(f"function_context: {function_context}")

                        logger.info(f"Calling get_memory with args: {function_args}")

                        # Prepare arguments for find_related_memory_items method
                        find_memory_args = {
                            'query': function_args['query'],
                            'context': function_context,
                            'user_id': user_id,
                            'chat_gpt': self,
                            'metadata': None,
                            'relation_type': relation_type,
                            'project_id': project_id,
                            'skip_neo': False
                        }

                        # Call the find_related_memory_items method with updated arguments
                        function_response = self.memory_graph.find_related_memory_items(sessionToken, **find_memory_args)
                        logger.info(f"got function_response: {function_response}")

                        # Format the function response as JSON string
                        formatted_function_response = json.dumps(function_response, indent=2)


                        # Add the function response to messages
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": formatted_function_response,
                        })

                        # Construct a follow-up prompt that asks for the response in JSON format with the given example
                        follow_up_prompt = {
                            "role": "user",
                            "content": "Use context from this user's selected text that we need to review: {text} and this review: {review_context} Please include up to 3 relevant memories that you can use in the re-written text prioritizing memories that include customer interviews. Also suggest updates to up to 3 sentences in this text. Ensure that the re-written text maintains the same tiptap schema used for the editor. Apply the following markup changes: 1) If words are removed, use 'strike' markup. 2) If words are added, use 'italic' markup. 3) Existing highlighted text should change from yellow to blue to indicate revisions needing approval. Format your response as a JSON object similar to this example." + json.dumps({
                                "review": 
                                    {
                                        "highlightedText": "Select a sentence from the user's input text that has a feedback area that we also want to highlight to the user",
                                        "rewrite_text": "Re-write the highlighted text using the feedback, questions, and relevant memories to improve quality.",
                                        "related_memories": [
                                            {
                                                "objectId": "n31SW9VEIY"
                                            }
                                        ]
                                    }
                                
                            }, indent=2)  # The indent parameter is optional, just for better readability if you print or log it
                        }


                        messages.append(follow_up_prompt)

                        # Call the API again to process the function response
                        second_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={ "type": "json_object" },
                            temperature=0.7
                        )
                        logger.info(f"Received second response from OpenAI: {second_response}")
                
                        # Extract the final response
                        final_message = second_response.choices[0].message
                        logger.info(f"Received second response from OpenAI: {final_message.content}")
                        try:
                            # Parse the JSON string into a Python dictionary
                            response_content = json.loads(final_message.content)

                            # Check if function_response is a string and parse it into a dictionary if so
                            if isinstance(function_response, str):
                                function_response = json.loads(function_response)

                            # Ensure function_response is a list before proceeding
                            if isinstance(function_response, list):
                                response_content['memories'] = function_response
                            else:
                                logger.error("Expected function_response to be a list, received: {}".format(type(function_response)))
                                return {"error": "Invalid format for function_response"}

                            return response_content

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {e}")
                            return {"error": "Error parsing JSON data"}

                        except Exception as e:
                            logger.error(f"Unexpected error processing the response: {e}")
                            return {"error": "An error occurred processing the response"}



                            
                    else:
                        logger.warning("No tool calls found in the response.")
                        return {"error": "No tool call was made in the response."}

                
        except Exception as e:
            logger.error(f"Error in get_memory_for_text: {e}")
            raise

    def generate_content_with_memories (self, sessionToken: str, user_id: str, prompt: any, option: any, command: any, project_id: str = None):
        """
        Generate content based on the given params.

        :param prompt: the user's prompt.
        :param option: The option the user chose (e.g., 'Summarize', 'Rephrase').
        :return: markdown formatted feedback with content generated.
        """
         # Define the function for the API to call
        get_memory = {
            "type": "function",
            "function": {
                "name": "get_memory",
                "description": "Retrieve a memory item from the user's personal memory graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Detailed query to search in the user's memory. Write the query in sentence format and include 2-3 sentences if needed.",
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "Relationship type given query and context that defines relationship type between memory items you want to find in memory.",
                        },
                        "context": {
                            "type": "array",
                            "description": "Context can be conversation history or any relevant context for a memory item.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "Person who created the memory item in context.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content of a memory item or conversation that is related to query.",
                                    },
                                },
                                "required": ["role", "content"],
                            },
                        }

                    },
                    "required": ["query", "context"],
                },
            },
        }

        # Define prompt templates for each option
        options = {
            "continue": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that continues existing text based on context from prior text. "
                    "Give more weight/priority to the later characters than the beginning ones. "
                    "Limit your response to no more than 200 characters, but make sure to construct complete sentences. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": prompt}
            ],
            "improve": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that improves existing text. "
                    "Limit your response to no more than 200 characters, but make sure to construct complete sentences. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "shorter": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that shortens existing text. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "longer": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that lengthens existing text. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "fix": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that fixes grammar and spelling errors in existing text. "
                    "Limit your response to no more than 200 characters, but make sure to construct complete sentences. "
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"The existing text is: {prompt}"}
            ],
            "zap": [
                {"role": "system", "content": (
                    "You are an AI writing assistant that generates text based on a prompt and related memories. "
                    "You take an input from the user and a command for manipulating the text and attempt to find related memories"
                    "Use Markdown formatting when appropriate."
                )},
                {"role": "user", "content": f"For this text: {prompt}. You have to respect the command: {command}"}
            ]
        }

        if option not in options:
                logger.error(f"Invalid option provided: {option}")

        messages = options[option]
        logger.info(f"messages: {messages}")

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")

        # Calculate available tokens for completion, ensuring not to exceed 128000 tokens
        # Subtracting a small buffer (e.g., 7 tokens) to account for any calculation discrepancies
        max_tokens_for_completion = min(4096, 128000 - token_count_prompt - 7)
        logger.info(f"max_tokens_for_completion: {max_tokens_for_completion}")

        # Check if the calculated max_tokens_for_completion is below the minimum required for a meaningful response
        if max_tokens_for_completion < 1024:  # Example threshold, adjust as needed
            raise ValueError("Available tokens for completion are too few for a meaningful response.")

        tools = [get_memory]

        # Count the number of tokens in the existing prompt
        token_count_prompt = self.count_tokens(prompt)
        logger.info(f"token_count_prompt: {token_count_prompt}")


        # Serialize the total input to JSON for token counting (messages and tools)
        total_input = json.dumps({"messages": messages, "tools": tools})
        total_input_token_count = self.count_tokens(total_input)
        logger.info(f"Total input token count (including messages and tools): {total_input_token_count}")
        logger.info("Starting generate content method")

         # Process the API's response
        try:
            
            # Call the OpenAI API with function calling
            response = self.client.chat.completions.create(
                model=self.model,  # Use the appropriate model identifier
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_memory"}},
                temperature=0.5
            )

            logger.info(f"response from model: {response}")

            response_message = response.choices[0].message
            logger.info(f"Response message: {response_message}")

            tool_calls = response_message.tool_calls
            logger.info(f"Tool calls found: {tool_calls}")


            if tool_calls:

                # Note: the JSON response may not always be valid; be sure to handle errors
                available_functions = {
                    "get_memory": get_memory,
                }  # only one function in this example, but you can have multiple

                # Append the function response to the conversation
                messages.append(response_message)

                # Process tool_calls
                for tool_call in tool_calls:
                    if tool_call.function.arguments:                    
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)

                        # Set default values for relation_type and context
                        relation_type = function_args.get('relation_type', None)
                        function_context = function_args.get('context', None)
                        logger.info(f"function_context: {function_context}")

                        logger.info(f"Calling get_memory with args: {function_args}")

                        # Prepare arguments for find_related_memory_items method
                        find_memory_args = {
                            'query': function_args['query'],
                            'context': function_context,
                            'user_id': user_id,
                            'chat_gpt': self,
                            'metadata': None,
                            'relation_type': relation_type,
                            'project_id': project_id,
                            'skip_neo': False
                        }

                        # Call the find_related_memory_items method with updated arguments
                        function_response = self.memory_graph.find_related_memory_items(sessionToken, **find_memory_args)
                        logger.info(f"got function_response: {function_response}")

                        # Format the function response as JSON string
                        formatted_function_response = json.dumps(function_response, indent=2)


                        # Add the function response to messages
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": formatted_function_response,
                        })

                        # Construct a follow-up prompt that asks for the response in JSON format with the given example
                        follow_up_prompt = {
                            "role": "user",
                            "content": "Use context from this user's selected text: {prompt} and this generate content command: {command} Please include up to 5 relevant memories that you can use to generate content based on the user's command. When generating text Ensure text maintains the same tiptap schema used for the editor. Apply the following markup changes: 1) If words are removed, use 'strike' markup. 2) If words are added, use 'italic' markup. 3) Existing highlighted text should change from yellow to blue to indicate revisions needing approval. Format your response as a JSON object similar to this example." + json.dumps({
                                "review": 
                                    {
                                        "highlightedText": "Select a sentence from the user's input text that has a feedback area that we also want to highlight to the user",
                                        "rewrite_text": "Create content to comply with this {command} by leveraging the 5 relevant memories related to the {prompt} and {command}. If the memories aren't too relevant use your best judgement to create the content and let the user know that you didn't find relevant memories.",
                                        "related_memories": [
                                            {
                                                "objectId": "n31SW9VEIY"

                                            }
                                        ]
                                    }
                                
                            }, indent=2)  # The indent parameter is optional, just for better readability if you print or log it
                        }


                        messages.append(follow_up_prompt)

                        # Call the API again to process the function response
                        second_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format={ "type": "json_object" },
                            temperature=0.7
                        )
                        logger.info(f"Received second response from OpenAI: {second_response}")
                
                        # Extract the final response
                        final_message = second_response.choices[0].message
                        logger.info(f"Received second response from OpenAI: {final_message.content}")
                        try:
                            # Parse the JSON string into a Python dictionary
                            final_response_data = json.loads(second_response.choices[0].message.content)  # Parses the JSON string into a Python dictionary

                            logger.info(f"Extracted rewrite_text: {final_response_data}")

                            # Check if function_response is a string and parse it into a dictionary if so
                            if isinstance(function_response, str):
                                function_response = json.loads(function_response)

                            # Ensure function_response is a list before proceeding
                            if isinstance(function_response, list):
                                final_response_data['memories'] = function_response
                            else:
                                logger.error("Expected function_response to be a list, received: {}".format(type(function_response)))
                                return {"error": "Invalid format for function_response"}

                            return final_response_data

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON parsing error: {e}")
                            return {"error": "Error parsing JSON data"}

                        except Exception as e:
                            logger.error(f"Unexpected error processing the response: {e}")
                            return {"error": "An error occurred processing the response"}
                    else:
                        logger.warning("No tool calls found in the response.")
                        return {"error": "No tool call was made in the response."}

        except Exception as e:
            logger.error(f"Error in get_memory_for_text: {e}")
            raise

    def _construct_cypher_prompt(
        self, 
        user_query: str, 
        memory_graph: dict, 
        bigbird_memory_ids: list, 
        acl_filter: ACLFilter
    ) -> str:
        """
        Construct a prompt for the LLM to generate a Cypher query.
        """
        # Extract node properties from your database schema
        memory_node_properties = [
            'id', 'content', 'createdAt', 'user_id', 'type', 
            'topics', 'emotion_tags', 'steps', 'current_step'
        ]

        # Convert ACL conditions to Cypher
        cypher_acl_conditions = self.acl_filter_to_cypher_conditions(acl_filter)

        # Define example structure that matches our tool schema
        example_structure = {
            "match": [{
                "variable": "m",
                "label": "Memory",
                "properties": {
                    "id": "$id",
                    "createdAt": "$timestamp"
                },
                "relationships": [{
                    "type": memory_graph.get('relationships', ['RELATION'])[0],
                    "direction": "->",
                    "variable": "r",
                    "target": {
                        "variable": "n",
                        "label": "Memory",
                        "properties": {
                            "type": "TextMemoryItem"
                        }
                    }
                }]
            }],
            "where": [
                "m.id IN $bigbird_memory_ids",
                f"({cypher_acl_conditions})"  # Include ACL conditions
            ],
            "return": ["m", "r", "n"],
            "order_by": [{
                "expression": "m.createdAt",
                "direction": "DESC"
            }],
            "limit": 10
        }

        prompt_parts = [
            f"Generate a Neo4j Cypher query for the following user query: {user_query}",
            "\nAvailable schema:",
            f"- Node label: Memory",
            f"- Node properties: {', '.join(memory_node_properties)}",
            f"- Property types:",
            "  - id: string",
            "  - content: string",
            "  - createdAt: string (ISO format date)",
            "  - user_id: string",
            "  - type: string",
            "  - topics: array",
            "  - emotion_tags: array",
            "  - steps: array",
            "  - current_step: string",
            f"- Valid relationship types: {', '.join(memory_graph.get('relationships', ['RELATION']))}",
            "\nQuery requirements:",
            f"- Must filter Memory nodes where id IN {bigbird_memory_ids}",
            f"- Must include these ACL conditions: ({cypher_acl_conditions})",
            "- Should return the Memory nodes and their relationships",
            "- Must order results by createdAt in descending order",
            "- Must use only the provided relationship types",
            "- Must use only the provided node properties",
            "- Avoid Cartesian products by using proper relationship patterns",
            "- Use parameters for variable data (prefix with $)",
            "\nYour response must be a structured query following this exact format:",
            json.dumps(example_structure, indent=2),
            "\nEnsure all required fields (match, return) are included and properly formatted.",
            "Use only the properties and relationship types defined in the schema."
        ]

        return "\n".join(prompt_parts)

    async def generate_usecase_memory_item_async(
        self, 
        memory_item: Dict[str, Any],
        context: Optional[List[ContextItem]] = None,
        existing_goals: Optional[List[Dict[str, Any]]] = None,
        existing_use_cases: Optional[List[Dict[str, Any]]] = None
    ) -> UseCaseResponse:
        """
        Generates use cases and goals from a memory item using OpenAI's structured outputs.
        The method analyzes the content and extracts relevant goals and use cases.

        Args:
            memory_item (Dict[str, Any]): The memory item to analyze, containing at minimum:
                - content: str - The main text content
                - metadata: Dict - Additional metadata about the memory
            context (Optional[Dict[str, Any]]): Additional context for analysis
            existing_goals (Optional[List[Dict[str, Any]]]): List of existing goals to consider
            existing_use_cases (Optional[List[Dict[str, Any]]]): List of existing use cases to consider

        Returns:
            UseCaseResponse: Dictionary containing:
                - data: Dict with 'goals' and 'use_cases' lists
                - metrics: Token counts and cost information
                - refusal: Optional string containing refusal message if model refuses to process
        """
        try:
            # Ensure memory content is a string
            memory_content = memory_item.get('content', '')
            if not isinstance(memory_content, str):
                memory_content = str(memory_content)

            # Prepare the content with token limit handling
            memory_content = self.trim_content_to_token_limit(memory_content, 3000)
            
            # Convert goals and use cases to lists if they're not already
            if existing_goals and not isinstance(existing_goals, list):
                existing_goals = [existing_goals]
            if existing_use_cases and not isinstance(existing_use_cases, list):
                existing_use_cases = [existing_use_cases]
                
            trimmed_goals = self.trim_content_to_token_limit(existing_goals or [], 2000)
            trimmed_use_cases = self.trim_content_to_token_limit(existing_use_cases or [], 2000)

            # Construct minimal prompt
            prompt = (
                f"Memory item content: {memory_content}\n"
                f"Review this memory item and suggest new goals and use cases, or select from existing ones.\n"
                f"For each goal or use case, indicate if it is 'new' or 'existing'.\n"
                f"If the input is not relevant for extracting goals or use cases, return empty lists.\n"
                f"Existing goals: {json.dumps(trimmed_goals)}\n"
                f"Existing use cases: {json.dumps(trimmed_use_cases)}"
            )

            # Count tokens
            messages = [
                {"role": "system", "content": "You help identify goals and use cases from memory items. Always return valid JSON matching the schema."},
                {"role": "user", "content": prompt}
            ]

            token_count = self.count_tokens(json.dumps(messages))
            logger.info(f"Token count after trimming async: {token_count}")

            # Log the messages before making the API call
            logger.info(f"Messages for API call async: {messages}")

            # Make API call using Structured Outputs with Gemini fallback
            if self.model_location_cloud:
                try:
                    # Try OpenAI first
                    completion = await self.async_client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=UseCaseMemoryItem
                    )
                    result = completion.choices[0].message.parsed
                    logger.info(f"✅ OpenAI schema generation successful")
                except Exception as e:
                    logger.warning(f"OpenAI schema generation failed: {e}, falling back to Gemini")
                    try:
                        # Fallback to Gemini
                        result = await self._call_gemini_structured_async(messages, UseCaseMemoryItem)
                        completion = None  # No OpenAI completion object for Gemini
                        logger.info(f"✅ Gemini schema generation successful (fallback)")
                    except Exception as e2:
                        logger.error(f"Both OpenAI and Gemini schema generation failed: {e2}")
                        raise
            else:
                # For local models (Ollama), continue using instructor
                response = await self.clientinstructor.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_model=UseCaseMemoryItem
                )
                result = response
                completion = None  # No OpenAI completion object for local models

            # Calculate metrics
            output_tokens = self.count_tokens(json.dumps(result.model_dump() if isinstance(result, BaseModel) else result))
            total_cost = self.calculate_cost(token_count, output_tokens)

            # Check for refusal (only for OpenAI completion)
            if completion and hasattr(completion.choices[0].message, 'refusal') and completion.choices[0].message.refusal:
                logger.warning(f"Model refused to process: {completion.choices[0].message.refusal}")
                return {
                    "data": {"goals": [], "use_cases": []},
                    "metrics": {
                        "usecase_token_count_input": token_count,
                        "usecase_token_count_output": 0,
                        "usecase_total_cost": total_cost
                    },
                    "refusal": completion.choices[0].message.refusal
                }

            # Handle None result
            if result is None:
                logger.warning("Received None result from model.")
                return {
                    "data": {"goals": [], "use_cases": []},
                    "metrics": {
                        "usecase_token_count_input": token_count,
                        "usecase_token_count_output": 0,
                        "usecase_total_cost": total_cost
                    },
                    "refusal": "Model returned None result"
                }

            return {
                "data": result.model_dump() if isinstance(result, BaseModel) else result,
                "metrics": {
                    "usecase_token_count_input": token_count,
                    "usecase_token_count_output": output_tokens,
                    "usecase_total_cost": total_cost
                }
            }

        except Exception as e:
            logger.error(f"Error in generate_usecase_memory_item_async: {e}")
            raise
    

    async def generate_node_ids(self, nodes: List[Dict], relationships: List[Dict], custom_labels: List[str] = None, custom_relationships: List[str] = None, property_overrides: Optional[List[Any]] = None, user_schema: Optional[Any] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Preserve LLM-generated semantic IDs and update relationship references accordingly.
        This prevents duplicate nodes by maintaining consistent IDs across runs.
        
        Args:
            nodes: List of node dictionaries
            relationships: List of relationship dictionaries
            custom_labels: List of custom node labels for validation
            custom_relationships: List of custom relationship types for validation
            property_overrides: Optional list of property override rules
            user_schema: Optional UserGraphSchema for required field validation
        
        Returns:
            Tuple containing updated nodes and relationships lists
        """
        try: 
            logger.info(f"🔧 GENERATE_NODE_IDS: property_overrides = {property_overrides}")
            # Create mapping of old IDs to preserved IDs
            id_mapping = {}
            
            # Process nodes while preserving LLM-generated IDs
            updated_nodes = []
            llm_gen_id_to_final_id = {}  # Map llmGenNodeId to final node ID
            
            for node in nodes:
                try:
                    # Extract llmGenNodeId (the ID the LLM generated for relationships)
                    llm_gen_node_id = node['properties'].get('llmGenNodeId')
                    
                    # Extract the actual content ID from the node properties
                    old_id = node['properties'].get('id')
                    if not old_id:
                        old_id = str(uuid4())  # Generate new ID if none exists
                    
                    # Validate node label (allow both system and custom labels)
                    if not NodeLabel.is_valid_label(node['label'], custom_labels):
                        raise ValueError(f"Invalid node label: {node['label']}")
                        
                    # Skip UUID generation for Memory nodes
                    if node['label'] == NodeLabel.Memory:
                        new_id = old_id  # Keep existing ID for Memory nodes
                    else:
                        # Keep LLM's semantic ID for non-Memory nodes (prevents duplicates)
                        new_id = old_id
                    
                    # Store mapping from llmGenNodeId to final ID for relationship processing
                    if llm_gen_node_id:
                        llm_gen_id_to_final_id[llm_gen_node_id] = {'label': node['label'], 'id': new_id}
                    
                    # Create the mapping key with proper error handling (for backward compatibility)
                    mapping_key = f"{node['label']}: {old_id}"
                    id_mapping[mapping_key] = f"{node['label']}: {new_id}"
                    
                    # Apply property overrides if provided
                    final_properties = {**node['properties'], 'id': new_id}
                    if property_overrides:
                        applied_overrides = self._apply_property_overrides(node, final_properties, property_overrides)
                        if applied_overrides:
                            logger.info(f"🔧 APPLIED OVERRIDES for {node['label']} (id: {new_id}): {applied_overrides}")
                    
                    # SCHEMA VALIDATION: Check required properties AFTER overrides are applied
                    # This allows property overrides to "rescue" nodes by providing missing required fields
                    if user_schema:
                        from memory.memory_graph import MemoryGraph
                        is_valid, missing_fields = MemoryGraph._validate_required_properties(
                            node['label'], final_properties, user_schema
                        )
                        if not is_valid:
                            logger.warning(
                                f"🚫 SKIPPING {node['label']} node (id: {new_id}) - "
                                f"missing required fields: {missing_fields}"
                            )
                            logger.warning(f"🚫 Node properties: {list(final_properties.keys())}")
                            # Skip this node - don't add it to updated_nodes
                            continue
                    
                    # Create new node with updated ID and applied overrides
                    updated_node = {
                        'label': node['label'],
                        'properties': final_properties
                    }
                    updated_nodes.append(updated_node)
                    
                except Exception as e:
                    logger.error(f"Error processing node: {node}")
                    logger.error(f"Error details: {str(e)}")
                    raise
            
            # Update relationship references
            updated_relationships = []
            for rel in relationships:
                try:
                    # Validate relationship type (allow both system and custom relationships)
                    if not RelationshipType.is_valid_relationship(rel['type'], custom_relationships):
                        raise ValueError(f"Invalid relationship type: {rel['type']}")
                    
                    source_obj = rel.get('source')
                    target_obj = rel.get('target')
                    
                    # Check format: type+llmGenNodeId objects (new format) or old formats for backward compatibility
                    if isinstance(source_obj, dict) and ('type' in source_obj or 'label' in source_obj):
                        # New type+llmGenNodeId format (or old label+id format for backward compatibility)
                        source_type = source_obj.get('type') or source_obj.get('label')
                        source_llm_id = source_obj.get('llmGenNodeId') or source_obj.get('id')  # Try new format first, fallback to old
                        target_type = target_obj.get('type') or target_obj.get('label')
                        target_llm_id = target_obj.get('llmGenNodeId') or target_obj.get('id')  # Try new format first, fallback to old
                        
                        # Look up actual nodes
                        if source_llm_id not in llm_gen_id_to_final_id:
                            logger.warning(f"Source llmGenNodeId not found: {source_llm_id}. Skipping relationship.")
                            continue
                        if target_llm_id not in llm_gen_id_to_final_id:
                            logger.warning(f"Target llmGenNodeId not found: {target_llm_id}. Skipping relationship.")
                            continue
                        
                        source_node = llm_gen_id_to_final_id[source_llm_id]
                        target_node = llm_gen_id_to_final_id[target_llm_id]
                        
                        # Validate declared types match actual node types
                        if source_node['label'] != source_type:
                            logger.error(
                                f"❌ TYPE MISMATCH: Source declared as '{source_type}' "
                                f"but node {source_llm_id} is actually '{source_node['label']}'. Skipping."
                            )
                            continue
                        
                        if target_node['label'] != target_type:
                            logger.error(
                                f"❌ TYPE MISMATCH: Target declared as '{target_type}' "
                                f"but node {target_llm_id} is actually '{target_node['label']}'. Skipping."
                            )
                            continue
                        
                        # VALIDATION: Check allowed source/target types from schema
                        if user_schema and hasattr(user_schema, 'relationship_types'):
                            rel_type = rel['type']
                            if rel_type in user_schema.relationship_types:
                                rel_def = user_schema.relationship_types[rel_type]
                                
                                # Check source type
                                if hasattr(rel_def, 'allowed_source_types') and rel_def.allowed_source_types:
                                    if source_type not in rel_def.allowed_source_types:
                                        logger.warning(
                                            f"🚫 SKIPPING: {rel_type} does not allow source type '{source_type}'. "
                                            f"Allowed: {rel_def.allowed_source_types}"
                                        )
                                        continue
                                
                                # Check target type
                                if hasattr(rel_def, 'allowed_target_types') and rel_def.allowed_target_types:
                                    if target_type not in rel_def.allowed_target_types:
                                        logger.warning(
                                            f"🚫 SKIPPING: {rel_type} does not allow target type '{target_type}'. "
                                            f"Allowed: {rel_def.allowed_target_types}"
                                        )
                                        continue
                                
                                logger.info(f"✅ VALID: {source_type} -{rel_type}-> {target_type}")
                        
                        updated_rel = {
                            'type': rel['type'],
                            'direction': '->',
                            'source': {
                                'label': source_node['label'],
                                'id': source_node['id']
                            },
                            'target': {
                                'label': target_node['label'],
                                'id': target_node['id']
                            }
                        }
                    
                    # Fallback: simple string IDs (old format for backward compatibility)
                    elif isinstance(source_obj, str) and isinstance(target_obj, str):
                        source_llm_id = source_obj
                        target_llm_id = target_obj
                        
                        if source_llm_id not in llm_gen_id_to_final_id:
                            logger.warning(f"Source llmGenNodeId not found: {source_llm_id}. Skipping relationship.")
                            continue
                        if target_llm_id not in llm_gen_id_to_final_id:
                            logger.warning(f"Target llmGenNodeId not found: {target_llm_id}. Skipping relationship.")
                            continue
                        
                        source_node = llm_gen_id_to_final_id[source_llm_id]
                        target_node = llm_gen_id_to_final_id[target_llm_id]
                        
                        # Validation for old format
                        if user_schema and hasattr(user_schema, 'relationship_types'):
                            rel_type = rel['type']
                            if rel_type in user_schema.relationship_types:
                                rel_def = user_schema.relationship_types[rel_type]
                                source_label = source_node['label']
                                target_label = target_node['label']
                                
                                if hasattr(rel_def, 'allowed_source_types') and rel_def.allowed_source_types:
                                    if source_label not in rel_def.allowed_source_types:
                                        logger.warning(
                                            f"🚫 SKIPPING: {rel_type} does not allow source type '{source_label}'. "
                                            f"Allowed: {rel_def.allowed_source_types}"
                                        )
                                        continue
                                
                                if hasattr(rel_def, 'allowed_target_types') and rel_def.allowed_target_types:
                                    if target_label not in rel_def.allowed_target_types:
                                        logger.warning(
                                            f"🚫 SKIPPING: {rel_type} does not allow target type '{target_label}'. "
                                            f"Allowed: {rel_def.allowed_target_types}"
                                        )
                                        continue
                                
                                logger.info(f"✅ VALID: {source_label} -{rel_type}-> {target_label}")
                        
                        updated_rel = {
                            'type': rel['type'],
                            'direction': '->',
                            'source': {
                                'label': source_node['label'],
                                'id': source_node['id']
                            },
                            'target': {
                                'label': target_node['label'],
                                'id': target_node['id']
                            }
                        }
                    
                    # Oldest format: source and target are objects with label and id
                    else:
                        source_key = f"{rel['source']['label']}: {rel['source']['id']}"
                        target_key = f"{rel['target']['label']}: {rel['target']['id']}"
                        
                        if source_key not in id_mapping:
                            raise KeyError(f"Source node not found in mapping: {source_key}")
                        if target_key not in id_mapping:
                            raise KeyError(f"Target node not found in mapping: {target_key}")
                        
                        updated_rel = {
                            'type': rel['type'],
                            'direction': rel.get('direction', '->'),
                            'source': {
                                'label': rel['source']['label'],
                                'id': id_mapping[source_key].split(': ')[1]
                            },
                            'target': {
                                'label': rel['target']['label'],
                                'id': id_mapping[target_key].split(': ')[1]
                            }
                        }
                    
                    updated_relationships.append(updated_rel)
                    
                except Exception as e:
                    logger.error(f"Error processing relationship: {rel}")
                    logger.error(f"Error details: {str(e)}")
                    logger.error(f"Current id_mapping: {id_mapping}")
                    raise
            
            return updated_nodes, updated_relationships
    
        except Exception as e:
            logger.error(f"Error in generate_node_ids: {str(e)}")
            logger.error(f"Nodes: {nodes}")
            logger.error(f"Relationships: {relationships}")
            raise
    
    def _apply_property_overrides(self, original_node, final_properties, property_overrides):
        """
        Apply property overrides with match conditions to a node.
        
        Args:
            original_node: The original node from LLM (with 'label' and 'properties')
            final_properties: The properties dict to modify (will be modified in-place)
            property_overrides: List of override rules in format:
                [
                    {"nodeLabel": "User", "match": {"name": "Alice"}, "set": {"id": "user_123"}},
                    {"nodeLabel": "Note", "set": {"pageId": "pg_123"}}  # no match = apply to all
                ]
        
        Returns:
            dict: The properties that were actually applied (for logging)
        """
        applied_props = {}
        
        try:
            # Enhanced format with Pydantic models: [PropertyOverrideRule(...)]
            for override_rule in property_overrides:
                # Handle both Pydantic models and dict (for backward compatibility during transition)
                if hasattr(override_rule, 'nodeLabel'):
                    # Pydantic model
                    node_label = override_rule.nodeLabel
                    match_conditions = override_rule.match or {}
                    set_properties = override_rule.set
                else:
                    # Dict format (fallback)
                    if not isinstance(override_rule, dict):
                        continue
                    node_label = override_rule.get('nodeLabel')
                    match_conditions = override_rule.get('match', {})
                    set_properties = override_rule.get('set', {})
                
                # Check if this rule applies to this node type
                if node_label != original_node['label']:
                    logger.debug(f"🔧 OVERRIDE SKIP: Rule for '{node_label}' doesn't match node type '{original_node['label']}'")
                    continue
                
                # Log the rule being evaluated
                node_id = final_properties.get('id', 'unknown')
                logger.info(f"🔧 OVERRIDE EVAL: Checking rule for {node_label} (node id: {node_id})")
                
                # Check if match conditions are satisfied (if any)
                if match_conditions:
                    logger.info(f"🔧 MATCH CHECK: Node '{node_id}' checking conditions: {match_conditions}")
                    logger.info(f"🔧 MATCH CHECK: Node properties: {original_node['properties']}")
                    
                    if not self._matches_conditions(original_node['properties'], match_conditions):
                        logger.info(f"🔧 MATCH FAIL: Node '{node_id}' does NOT match conditions {match_conditions} - SKIPPING")
                        continue  # Skip this rule - conditions not met
                    else:
                        logger.info(f"🔧 MATCH SUCCESS: Node '{node_id}' MATCHES conditions {match_conditions} - APPLYING overrides")
                else:
                    logger.info(f"🔧 NO MATCH: Rule has no match conditions - applying to all {node_label} nodes")
                
                # Apply the 'set' properties
                if set_properties:
                    logger.info(f"🔧 APPLYING: Setting properties {set_properties} on node '{node_id}'")
                    final_properties.update(set_properties)
                    applied_props.update(set_properties)
                        
        except Exception as e:
            logger.error(f"Error applying property overrides: {str(e)}")
            logger.error(f"Override rules: {property_overrides}")
            logger.error(f"Node: {original_node}")
        
        return applied_props
    
    def _matches_conditions(self, node_properties, match_conditions):
        """
        Check if node properties match the given conditions.
        
        Args:
            node_properties: Dict of node properties
            match_conditions: Dict of conditions to match
            
        Returns:
            bool: True if all conditions are satisfied
        """
        try:
            for key, expected_value in match_conditions.items():
                if key not in node_properties:
                    logger.info(f"🔧 MATCH DETAIL: Property '{key}' not found in node properties - FAIL")
                    return False
                
                actual_value = node_properties[key]
                logger.info(f"🔧 MATCH DETAIL: Comparing '{key}': expected='{expected_value}', actual='{actual_value}'")
                
                # Exact match for now - could extend to support regex, contains, etc.
                if actual_value != expected_value:
                    logger.info(f"🔧 MATCH DETAIL: Values don't match for '{key}' - FAIL")
                    return False
                else:
                    logger.info(f"🔧 MATCH DETAIL: Values match for '{key}' - PASS")
                    
            logger.info(f"🔧 MATCH DETAIL: All conditions satisfied - OVERALL MATCH")
            return True
        except Exception as e:
            logger.error(f"Error matching conditions: {str(e)}")
            return False
                

    async def generate_memory_graph_schema_async(
            self, 
            memory_item: Dict[str, Any],
            usecase_memory_item: Dict[str, Any],
            neo_session: AsyncSession,
            workspace_id: Optional[str] = None,
            related_memories: Optional[List[Dict[str, Any]]] = None,
            user_id: Optional[str] = None,
            schema_ids: Optional[List[str]] = None,
            property_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
            developer_user_id: Optional[str] = None,
            developer_workspace_id: Optional[str] = None,
            organization_id: Optional[str] = None,
            namespace_id: Optional[str] = None
        ) -> Dict[str, Union[MemoryGraphSchema, Dict[str, float]]]:
        """
        Generate a memory graph schema using OpenAI's Structured Outputs feature with direct JSON schema.

        Args:
            memory_item (Dict[str, Any]): The memory item to analyze
            usecase_memory_item (Dict[str, Any]): Previously extracted use cases and goals
            workspace_id (Optional[str]): Workspace identifier if applicable
            related_memories (Optional[List[Dict[str, Any]]]): List of related memories
            neo_session (AsyncSession): Neo4j session
            user_id (Optional[str]): User ID for schema selection
            schema_ids (Optional[List[str]]): Custom schema IDs to enforce

        Returns:
            Dict[str, Union[MemoryGraphSchema, Dict[str, float]]]: Contains:
                - data: MemoryGraphSchema with 'nodes' and 'relationships' lists
                - metrics: Dictionary with token counts and cost information
        """
        total_cost = 0  # Initialize at the start of the method

        try:
            creator_name = memory_item.get('metadata', {}).get('creator_name')
            company = memory_item.get('metadata', {}).get('company')
            logger.info(f"Creator name: {creator_name}")
            logger.info(f"🔧 GRAPH_SCHEMA_ASYNC: property_overrides parameter = {property_overrides}")
            logger.info(f"memory_item inside generate_memory_graph_schema_async: {memory_item}")

            # Step 1: Generate nodes and relationships together
            node_prompt_parts = [
                "=== MEMORY CONTENT TO ANALYZE ===",
                f"{memory_item}",
                "",
                "=== CONTEXT (DO NOT EXTRACT FROM THIS SECTION - it's just for your context) ===",
                f"Use case info (for context only - these are high-level patterns based on previous memories, NOT part of the memory content): {json.dumps(usecase_memory_item)}",
                f"Related memories (for context only): {json.dumps(related_memories)}",
                "",
                "=== EXTRACTION INSTRUCTIONS ===",
                "CRITICAL: Only extract nodes and relationships from the MEMORY CONTENT section above.",
                "DO NOT create nodes or relationships based on the CONTEXT section (use cases, related memories).",
                "The context section is provided only to help you understand the user's workflow and priorities.",
                "",
                "We already have one node of type 'Memory' representing the provided memory item.",
                "Analyze the MEMORY CONTENT and identify appropriate node types based on what is explicitly mentioned.",
                "",
                "EXTRACTION COMPLETENESS:",
                "- Extract ALL entities of each type mentioned in the content",
                "- If 3 entities of the same type are mentioned, create 3 nodes (not just 1 - i.e. if there are three tasks with a person's name and we have a person and task node we should create three task nodes and three person nodes for each then connect them to each other)",
                "- Do not stop after extracting the first entity - continue until all are captured",
                "- For checkbox/list items (- [ ] or - [x]), create a node for each item if appropriate",
                "- If entities are connected in the content, create nodes for ALL entities involved",
                "",
                "For each node, include all required properties based on its type.",
                "Choose property values that will be meaningful for future searches.",
                "Focus on selecting nodes that will help a user find something important or relevant from this memory in the future.",
                "",
                "After identifying ALL nodes, create ALL relevant relationships between them that are evident from the content.",
                "RELATIONSHIP COMPLETENESS:",
                "- CRITICAL: Every node you create should connect to at least one other node via a relationship",
                "- If you create a node that has no relationships, either remove the node OR find a connection",
                "- Identify EVERY connection mentioned or strongly implied between entities",
                "- If Entity A is related to Entity B, create the appropriate relationship based on available types",
                "",
                "RELATIONSHIP FORMAT:",
                "- Each relationship uses a type+id format for source and target",
                "- source: {\"type\": \"NodeType\", \"id\": \"llmGenNodeId\"}",
                "- target: {\"type\": \"NodeType\", \"id\": \"llmGenNodeId\"}",
                "- The \"type\" field specifies the node type and must match one of the allowed types for that relationship",
                "- The \"id\" field contains the llmGenNodeId you generated for that node",
                "- Example: {\"source\": {\"type\": \"Task\", \"id\": \"task_1\"}, \"target\": {\"type\": \"Person\", \"id\": \"person_1\"}, \"type\": \"ASSIGNED_TO\"}",
                "- If multiple entities connect to the same entity, create a relationship for EACH connection",
                "- Do not skip any relationships - extract all connections from the content",
                "- Review your nodes list and ensure each node appears in at least one relationship (as source or target)",
                "",
                "IMPORTANT REQUIREMENTS FOR NODES:",
                "- Assign a unique llmGenNodeId to each node (e.g., 'task_1', 'project_main', 'user_john')",
                "- These IDs are used to connect nodes via relationships",
                "- CRITICAL: Include ALL unique identifier properties defined in the schema (e.g., workflow_id, step_id, task_id)",
                "- Unique identifiers are REQUIRED for entity matching and deduplication - extract them from the content if available",
                "- If a unique identifier property is mentioned in the content, you MUST include it in the node properties",
                "- IMPORTANT: If a unique identifier is NOT available in the content, use null - this is acceptable and property_overrides can fill it later",
                "- Null unique identifiers are acceptable - we don't force values. If they remain null, nodes may be skipped (which is OK)",
                "",
                "IMPORTANT REQUIREMENTS FOR RELATIONSHIPS:",
                "- Create MULTIPLE relationships - there is no limit and it's encouraged to not leave any node handing and connect it to the appropraite target/source node",
                "- Use llmGenNodeId values as source and target",
                "- Each relationship should represent a distinct connection",
                "- Create relationships in BOTH directions if bidirectional connections exist",
                "",
                "EXAMPLES:",
                "",
                "Example 1 - Multiple entities with connections:",
                "If content mentions: 'Alice to create deck by Nov 20, Bob to add features by Nov 15, Carol to enable support'",
                "Extract ALL entities:",
                "  - Entity type 1: node 1, node 2, node 3 (create all 3, not just 1)",
                "  - Entity type 2: node A, node B, node C (create all 3, not just 1)",
                "Create ALL relationships:",
                "  - If node 1 connects to node A, create that relationship",
                "  - If node 2 connects to node B, create that relationship",
                "  - If node 3 connects to node C, create that relationship",
                "",
                "Example 2 - Container with multiple items:",
                "If content has: 'Project X includes: item 1, item 2, item 3'",
                "Nodes: ContainerNode[llmGenNodeId='container_x'], ItemNode[llmGenNodeId='item_1'], ItemNode[llmGenNodeId='item_2'], ItemNode[llmGenNodeId='item_3']",
                "Relationships:",
                "  {source: 'item_1', target: 'container_x', type: 'BELONGS_TO'} (or appropriate relationship type)",
                "  {source: 'item_2', target: 'container_x', type: 'BELONGS_TO'}",
                "  {source: 'item_3', target: 'container_x', type: 'BELONGS_TO'}",
                "",
                "Example 3 - Sequential dependencies:",
                "If content indicates: 'Item B depends on Item A, Item C depends on Item B'",
                "Relationships:",
                "  {source: 'item_b', target: 'item_a', type: 'DEPENDS_ON'} (or appropriate relationship type)",
                "  {source: 'item_c', target: 'item_b', type: 'DEPENDS_ON'}",
                "",
                "Use the relationship types available in the schema that best describe each connection.",
                "Respect the allowed source and target node types for each relationship type.",
                "Only create relationships that are clearly stated or strongly implied in the content."
            ]

            prompt = "\n".join(node_prompt_parts)

            node_messages = [
                {
                    "role": "system", 
                    "content": """You are a graph node and relationship identifier that creates structured knowledge graphs.
                                Always include required properties for each node type.
                                Focus on extracting meaningful entities and their relationships from the content.
                                
                                CRITICAL RULE - CONTENT BOUNDARIES:
                                - The user message will have clearly marked sections: "MEMORY CONTENT" and "CONTEXT"
                                - ONLY extract nodes and relationships from the "MEMORY CONTENT" section
                                - NEVER create nodes or relationships from "CONTEXT" sections (use cases, goals, related memories)
                                - Context is provided ONLY to help you understand the user's workflow - it is NOT part of the memory
                                - If the memory says "Meeting notes from product planning" with no details, create NO nodes (just return empty)
                                
                                NODE CREATION:
                                - Extract ALL entities that are explicitly mentioned in the MEMORY CONTENT section ONLY
                                - Do not stop after creating one entity - extract every entity mentioned
                                - If 3 entities of the same type are mentioned, create 3 nodes of that type (not just 1)
                                - If 5 entities are listed, create 5 nodes (not just some)
                                - Use predefined node types from the schema
                                - Include all required properties for each node type
                                - Assign a unique llmGenNodeId to each node (e.g., 'entity_1', 'entity_2', 'container_main')
                                
                                RELATIONSHIP CREATION:
                                - Create ALL relationships that connect the nodes you extracted
                                - CRITICAL: Every node should connect to at least one other node - isolated nodes suggest missing relationships
                                - If you create a node with no relationships, either remove it or find its connection to other nodes
                                - Identify meaningful connections between the nodes you create based on the content
                                - Use relationship types from the schema that accurately reflect each connection
                                - Only create relationships that are explicitly stated or strongly implied in the MEMORY CONTENT
                                - Each relationship connects two nodes using their llmGenNodeId values as source and target
                                - The source and target must reference llmGenNodeId values from the nodes you created
                                - If you created multiple entities that connect to other entities, create a relationship for EACH connection
                                - Do not skip relationships - extract ALL connections mentioned in the content
                                - Final check: Verify each llmGenNodeId appears in at least one relationship (source or target)
                                
                                PRIORITIZATION RULES:
                                1. ALWAYS PREFER custom node types from the user schema over generic system types
                                2. PREFER domain-specific relationships over generic ones when available
                                
                                ANTI-HALLUCINATION RULES:
                                1. Only extract information that is explicitly present in the MEMORY CONTENT section
                                2. For any property where the information is not available, use null instead of guessing or fabricating values
                                3. It is better to have null properties than incorrect or hallucinated data
                                4. Do not infer, assume, or create information that is not directly stated in the MEMORY CONTENT
                                5. Only create relationships that are clearly evident from the MEMORY CONTENT - do not fabricate connections
                                6. Do NOT extract from use cases, goals, or related memories - these are context only
                                
                                The structured output schema defines all available node types, relationship types, properties, constraints, and enum values.
                                All properties accept null values when information is not available in the content.
                                Your response will be automatically validated against this schema."""
                },
                {"role": "user", "content": prompt}
            ]
                
            from memory.memory_graph import MemoryGraph  
            memory_graph = MemoryGraph()
            await memory_graph.ensure_async_connection()

            # Get user-specific schema or fall back to system schema
            user_id = memory_item.get('metadata', {}).get('user_id')
            workspace_id = workspace_id or memory_item.get('metadata', {}).get('workspace_id')
            
            logger.info(f"🔍 DEBUG: user_id={user_id}, workspace_id={workspace_id}")
            logger.info(f"🔍 DEBUG: memory_item keys: {list(memory_item.keys())}")
            
            if user_id:
                try:
                    # Pass memory content and metadata for LLM-powered schema selection
                    content = memory_item.get('content', '')
                    # Extract graph generation configuration
                    graph_generation = memory_item.get('graph_generation')
                    extracted_schema_ids = None
                    
                    # Handle GraphGeneration format
                    if graph_generation and isinstance(graph_generation, dict):
                        mode = graph_generation.get('mode', 'auto')
                        if mode == 'auto':
                            auto_config = graph_generation.get('auto', {})
                            extracted_schema_id = auto_config.get('schema_id')
                            if extracted_schema_id:
                                extracted_schema_ids = [extracted_schema_id]
                                logger.info(f"✅ Using schema_id from graph_generation.auto: {extracted_schema_id}")
                    
                    # Use extracted schema_ids if available, otherwise use method parameter
                    if extracted_schema_ids:
                        schema_ids = extracted_schema_ids
                    # schema_ids from method parameter is used as fallback
                    metadata = memory_item.get('metadata', {})
                    
                    logger.info(f"🚀 GRAPH STEP 1: Starting schema-aware graph generation for user_id={user_id}")
                    logger.info(f"🚀 GRAPH CONTENT: {content[:100]}...")
                    logger.info(f"🚀 GRAPH METADATA: {metadata}")
                    logger.info(f"🚀 SCHEMA ENFORCEMENT: schema_ids={schema_ids}")
                    
                    # Use LLMSchemaSelector to select the most appropriate UserGraphSchema
                    from services.llm_schema_selector import LLMSchemaSelector
                    from services.schema_service import SchemaService
                    
                    schema_service = SchemaService()
                    
                    # If schema_ids provided, use them directly (developer override)
                    if schema_ids and len(schema_ids) > 0:
                        logger.info(f"🔒 SCHEMA ENFORCEMENT: Using developer-specified schema_ids={schema_ids}")
                        selected_schema_id = schema_ids[0]  # Use first schema for now
                        confidence = 1.0  # Full confidence for explicit selection
                        
                        # Fetch the actual schema object for the selected schema_id
                        try:
                            schema_user_id = developer_user_id if developer_user_id else user_id
                            schema_workspace_id = developer_workspace_id if developer_workspace_id else workspace_id
                            
                            logger.info(f"🔒 SCHEMA LOOKUP: Fetching schema_id={selected_schema_id} with user_id={schema_user_id}, workspace_id={schema_workspace_id}")
                            logger.info(f"🔒 SCHEMA LOOKUP: developer_user_id={developer_user_id}, developer_workspace_id={developer_workspace_id}")
                            
                            # Use multi-tenant context passed from caller
                            logger.info(f"🔒 SCHEMA LOOKUP: Using multi-tenant context - org_id={organization_id}, namespace_id={namespace_id}")
                            
                            user_schemas = await schema_service.get_schemas_by_ids(
                                [selected_schema_id], 
                                schema_user_id, 
                                schema_workspace_id,
                                organization_id,
                                namespace_id
                            )
                            
                            logger.info(f"🔒 SCHEMA LOOKUP RESULT: Found {len(user_schemas) if user_schemas else 0} schemas")
                            if user_schemas:
                                selected_schema = user_schemas[0]
                                logger.info(f"🔒 SCHEMA ENFORCEMENT: Successfully loaded schema '{selected_schema.name}' (id={selected_schema.id})")
                                logger.info(f"🔒 SCHEMA DETAILS: node_types={len(selected_schema.node_types) if selected_schema.node_types else 0}, relationships={len(selected_schema.relationship_types) if selected_schema.relationship_types else 0}")
                            else:
                                logger.warning(f"🔒 SCHEMA ENFORCEMENT: Schema {selected_schema_id} not found, falling back to None")
                                selected_schema = None
                        except Exception as e:
                            logger.error(f"🔒 SCHEMA ENFORCEMENT: Failed to load schema {selected_schema_id}: {e}")
                            selected_schema = None
                    else:
                        # Otherwise, use LLM to select schema
                        schema_selector = LLMSchemaSelector(schema_service)
                        # Use developer_user_id for schema selection, fallback to user_id if not available
                        schema_user_id = developer_user_id if developer_user_id else user_id
                        # Use developer's workspace ID for schema selection, fallback to end user's workspace if not available
                        schema_workspace_id = developer_workspace_id if developer_workspace_id else workspace_id
                        logger.info(f"🔍 SCHEMA USER ID: Using {schema_user_id} (developer_user_id={developer_user_id}, user_id={user_id})")
                        logger.info(f"🔍 SCHEMA WORKSPACE ID: Using {schema_workspace_id} (developer_workspace_id={developer_workspace_id}, workspace_id={workspace_id})")
                        selected_schema_id, confidence, selected_schema = await schema_selector.select_schema_for_content(
                            content=content,
                            user_id=schema_user_id,
                            workspace_id=schema_workspace_id, 
                            metadata=metadata,
                            operation_type="add_memory",
                            organization_id=organization_id,
                            namespace_id=namespace_id
                        )
                    
                    logger.info(f"🚀 GRAPH STEP 2: LLM selected schema_id={selected_schema_id}, confidence={confidence}")
                    
                    # Use the selected schema object directly (no need to re-fetch from Parse Server)
                    if selected_schema_id and selected_schema:
                        logger.info(f"🚀 OPTIMIZED: Using schema object directly from LLM selection (no redundant Parse Server call)")
                    else:
                        logger.info("🚀 GRAPH STEP 3: No schema selected; using system schema")
                        selected_schema = None
                        
                    if selected_schema:
                        logger.info(f"🚀 GRAPH STEP 3: Using custom schema '{selected_schema.name}' with {len(selected_schema.node_types)} node types and {len(selected_schema.relationship_types)} relationship types")
                        
                        # Track schema usage for analytics
                        try:
                            await schema_service.update_schema_usage(selected_schema_id, user_id)
                        except Exception as e:
                            logger.warning(f"📊 Failed to track schema usage: {e}")
                        
                        # Convert UserGraphSchema to JSON Schema format for structured output
                        custom_node_labels = [nt.name for nt in selected_schema.node_types.values()]
                        custom_relationship_types = [rt.name for rt in selected_schema.relationship_types.values()]
                        
                        logger.info(f"🚀 CUSTOM NODES: {custom_node_labels}")
                        logger.info(f"🚀 CUSTOM RELATIONSHIPS: {custom_relationship_types}")
                        
                        # Register custom node and relationship types for validation
                        from models.shared_types import NodeLabel, RelationshipType
                        NodeLabel.register_custom_labels(custom_node_labels)
                        RelationshipType.register_custom_relationships(custom_relationship_types)
                        logger.info(f"🔧 REGISTERED: Custom node labels: {custom_node_labels}")
                        logger.info(f"🔧 REGISTERED: Custom relationship types: {custom_relationship_types}")
                        
                        # Generate JSON schema with custom node types including full property definitions
                        memory_graph_schema = memory_graph.get_custom_schema_for_structured_output(
                            custom_node_labels, 
                            custom_relationship_types,
                            [selected_schema]  # Pass the selected schema object for property definitions
                        )
                        
                        # Store schema for property indexing
                        memory_graph._last_memory_graph_schema = memory_graph_schema
                        logger.info("🔧 PROPERTY INDEXING: Stored custom schema for property indexing")
                    else:
                        logger.info(f"🚀 GRAPH STEP 3: No custom schema selected (confidence={confidence}), using system schema")
                        memory_graph_schema = None
                    
                    # Count node types and relationship types from JSON schema format
                    node_count = 0
                    rel_count = 0
                    # Ensure we have a valid schema dict before inspecting keys
                    if memory_graph_schema is None:
                        memory_graph_schema = memory_graph.get_memory_graph_schema()
                        # Store schema for property indexing
                        memory_graph._last_memory_graph_schema = memory_graph_schema
                        logger.info("🔧 PROPERTY INDEXING: Stored system schema for property indexing")
                    
                    if isinstance(memory_graph_schema, dict) and 'properties' in memory_graph_schema:
                        # JSON schema format
                        if 'nodes' in memory_graph_schema['properties'] and 'items' in memory_graph_schema['properties']['nodes']:
                            node_items = memory_graph_schema['properties']['nodes']['items']
                            if 'anyOf' in node_items:
                                node_count = len(node_items['anyOf'])
                        
                        if 'relationships' in memory_graph_schema['properties'] and 'items' in memory_graph_schema['properties']['relationships']:
                            rel_items = memory_graph_schema['properties']['relationships']['items']
                            # Check for anyOf structure (new constrained format)
                            if 'anyOf' in rel_items:
                                rel_count = len(rel_items['anyOf'])
                            # Check for enum in type property (old format)
                            elif 'properties' in rel_items and 'type' in rel_items['properties'] and 'enum' in rel_items['properties']['type']:
                                rel_count = len(rel_items['properties']['type']['enum'])
                    else:
                        # Legacy format
                        node_count = len(memory_graph_schema.get('node_types', {}))
                        rel_count = len(memory_graph_schema.get('relationship_types', {}))
                    
                    logger.info(f"🚀 GRAPH STEP 2: Got user schema with {node_count} node types and {rel_count} relationship types")
                    
                    if schema_ids:
                        logger.info(f"🚀 GRAPH: Using developer-specified schemas: {schema_ids}")
                    else:
                        logger.info(f"🚀 GRAPH: Using LLM-selected schema for user {user_id}")
                except Exception as e:
                    logger.warning(f"🚀 GRAPH FALLBACK: Failed to get user schema, falling back to system schema: {e}")
                    try:
                        # If schema selection failed due to NoneType, ensure defaults
                        memory_graph_schema = memory_graph.get_memory_graph_schema()
                        if not isinstance(memory_graph_schema, dict) or 'properties' not in memory_graph_schema:
                            # As a safeguard, build a minimal schema to avoid downstream issues
                            memory_graph_schema = memory_graph.get_memory_graph_schema()
                        # Store schema for property indexing
                        memory_graph._last_memory_graph_schema = memory_graph_schema
                        logger.info("🔧 PROPERTY INDEXING: Stored fallback schema for property indexing")
                    except Exception:
                        # Last-resort static minimal schema
                        logger.warning("🚀 GRAPH FALLBACK: Failed to get schema, using minimal schema fallback")
                        memory_graph_schema = {"title": "MemoryGraph", "type": "object", "properties": {"nodes": {"type": "array", "items": {"type": "object"}}, "relationships": {"type": "array", "items": {"type": "object"}}}}
                        # Store schema for property indexing
                        memory_graph._last_memory_graph_schema = memory_graph_schema
                        logger.info("🔧 PROPERTY INDEXING: Stored minimal fallback schema for property indexing")
            else:
                logger.info(f"🚀 GRAPH FALLBACK: No user_id, using system schema")
                memory_graph_schema = memory_graph.get_memory_graph_schema()
                # Store schema for property indexing
                memory_graph._last_memory_graph_schema = memory_graph_schema
                logger.info("🔧 PROPERTY INDEXING: Stored system schema (no user_id) for property indexing")
            
            # Use the custom user schema if available, otherwise fall back to system schema
            if user_id and isinstance(memory_graph_schema, dict) and 'properties' in memory_graph_schema:
                # Use the custom schema that was selected by the LLM
                node_schema = memory_graph_schema
                relationship_schema = memory_graph_schema  # Same schema contains both nodes and relationships
                logger.info(f"🚀 GRAPH STEP 3: Using custom schema for OpenAI structured output")
            else:
                # Fall back to system schema for backward compatibility
                node_schema = memory_graph.get_node_schema()
                relationship_schema = memory_graph.get_relationship_schema()
                logger.info(f"🚀 GRAPH STEP 3: Using system schema for OpenAI structured output")

            # Calculate initial token count for input
            node_input_token_count = self.count_tokens(json.dumps(node_messages))
            logger.info(f"node_input_token_count: {node_input_token_count}")
            # Log the messages before making the API call
            logger.debug(f"Messages for API call: {node_messages}")

            # First API call to generate nodes
            if self.model_location_cloud:
                # Log the request details for debugging
                logger.info(f"🔍 Making OpenAI request with model: {self.model_mini}")
                logger.info(f"🔍 Messages count: {len(node_messages)}")
                logger.info(f"🔍 Last message preview: {node_messages[-1]['content'][:300]}...")
                logger.info(f"🔍 Schema keys: {list(node_schema.keys()) if isinstance(node_schema, dict) else 'Not a dict'}")
                logger.info(f"🔍 Schema name: {node_schema.get('title', node_schema.get('name', 'Unknown'))}")
                logger.info(f"🔍 Request max_tokens: 16000")
                
                node_completion = await self._create_completion_async(
                    model=self.model_mini,
                    messages=node_messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Node",
                            "schema": node_schema,  
                            "strict": True
                        }
                    },
                    max_tokens=16000,  # Increased limit to handle larger responses
                    temperature=0.7
                )
                
                # Log the raw response for debugging
                logger.info(f"🔍 OpenAI Response Status: {node_completion}")
                logger.info(f"🔍 Response Choices Length: {len(node_completion.choices) if node_completion.choices else 'None'}")

                # Log token usage
                if hasattr(node_completion, 'usage') and node_completion.usage:
                    logger.info(f"🔍 Token Usage - Prompt: {node_completion.usage.prompt_tokens}, Completion: {node_completion.usage.completion_tokens}, Total: {node_completion.usage.total_tokens}")

                if node_completion.choices and len(node_completion.choices) > 0:
                    logger.info(f"🔍 First Choice Message: {node_completion.choices[0].message}")
                    logger.info(f"🔍 Finish Reason: {node_completion.choices[0].finish_reason}")

                    # Check for refusal (OpenAI safety filter)
                    if hasattr(node_completion.choices[0].message, 'refusal') and node_completion.choices[0].message.refusal:
                        logger.error(f"🔍 OpenAI Refusal: {node_completion.choices[0].message.refusal}")

                    logger.info(f"🔍 Message Content Type: {type(node_completion.choices[0].message.content)}")
                    logger.info(f"🔍 Message Content Length: {len(node_completion.choices[0].message.content) if node_completion.choices[0].message.content else 'None'}")
                    logger.info(f"🔍 Raw node completion response: '{node_completion.choices[0].message.content}'")
                else:
                    logger.error("🔍 No choices in OpenAI response!")
                
                # Initialize relationships_from_first_call early to avoid NameError in exception handlers
                relationships_from_first_call = []
                
                try:
                    # Check for refusal first
                    if hasattr(node_completion.choices[0].message, 'refusal') and node_completion.choices[0].message.refusal:
                        raise ValueError(f"OpenAI refused to respond: {node_completion.choices[0].message.refusal}")

                    # Validate the response format before parsing
                    content = node_completion.choices[0].message.content.strip() if node_completion.choices[0].message.content else ""
                    logger.info(f"🔍 Stripped content length: {len(content)}")
                    logger.info(f"🔍 Stripped content: '{content}...' (first 200 chars)")
                    if not content:
                        finish_reason = node_completion.choices[0].finish_reason
                        error_msg = f"Empty response from API (finish_reason: {finish_reason})"
                        if finish_reason == "length":
                            error_msg += " - Response truncated due to token limit. Consider reducing prompt size or increasing max_tokens."
                        elif finish_reason == "content_filter":
                            error_msg += " - Response blocked by content filter. Content may violate usage policies."
                        elif finish_reason == "stop":
                            error_msg += " - Model stopped naturally but produced no content. May be schema/prompt issue."
                        raise ValueError(error_msg)
                        
                    # Try to parse the JSON with multiple fallback strategies
                    nodes_result = None
                    
                    # Strategy 1: Try to find the end of valid JSON by looking for closing braces
                    json_end = content.rfind('}]}')
                    if json_end != -1:
                        json_content = content[:json_end + 3]
                        try:
                            nodes_result = json.loads(json_content)
                            logger.info("Successfully parsed JSON using closing braces strategy")
                        except json.JSONDecodeError:
                            pass
                    
                    # Strategy 2: If Strategy 1 failed, try the original content
                    if nodes_result is None:
                        try:
                            nodes_result = json.loads(content)
                            logger.info("Successfully parsed JSON using original content")
                        except json.JSONDecodeError:
                            pass
                    
                    # Strategy 3: If both failed, try to extract and fix malformed JSON
                    if nodes_result is None:
                        try:
                            import re
                            # Look for JSON-like structure and extract it
                            json_match = re.search(r'\{.*"nodes".*\}', content, re.DOTALL)
                            if json_match:
                                extracted_json = json_match.group(0)
                                # Try to fix common JSON issues
                                fixed_json = self._fix_malformed_json(extracted_json)
                                nodes_result = json.loads(fixed_json)
                                logger.info("Successfully extracted and fixed JSON using regex fallback")
                            else:
                                raise ValueError("Could not extract valid JSON from response")
                        except Exception as fallback_error:
                            logger.error(f"All JSON parsing strategies failed: {str(fallback_error)}")
                            raise
                    
                    # Validate the structure
                    if not isinstance(nodes_result, dict) or "nodes" not in nodes_result:
                        raise ValueError("Invalid response structure - missing 'nodes' key")
                    
                    # Log parsed nodes with details
                    nodes = nodes_result.get('nodes', [])
                    relationships_from_first_call = nodes_result.get('relationships', [])
                    logger.info(f"📊 STRUCTURED OUTPUT - NODES: Generated {len(nodes)} nodes, {len(relationships_from_first_call)} relationships from first call")
                    
                    # Log node summary
                    if nodes:
                        node_summary = {}
                        for node in nodes:
                            node_label = node.get('label', 'Unknown')
                            node_summary[node_label] = node_summary.get(node_label, 0) + 1
                        logger.info(f"📊 NODE SUMMARY: {dict(node_summary)}")
                        
                        # Log first few nodes as examples
                        for i, node in enumerate(nodes[:3]):
                            llm_gen_id = node.get('properties', {}).get('llmGenNodeId', 'N/A')
                            name = node.get('properties', {}).get('name', 'N/A')
                            logger.info(f"📊 NODE EXAMPLE {i+1}: {node.get('label')} [llmGenNodeId={llm_gen_id}, name={name}]")
                    
                    # Log relationship summary from nodes response (if present)
                    if relationships_from_first_call:
                        logger.info(f"📊 RELATIONSHIPS IN NODE RESPONSE: {len(relationships_from_first_call)} relationships")
                        for i, rel in enumerate(relationships_from_first_call[:3]):
                            logger.info(f"📊 REL EXAMPLE {i+1}: {rel.get('source')} -{rel.get('type')}-> {rel.get('target')}")
                    
                    # Validate relationship completeness: check for orphaned nodes
                    if nodes and relationships_from_first_call:
                        # Collect all llmGenNodeIds that appear in relationships
                        connected_node_ids = set()
                        for rel in relationships_from_first_call:
                            # Handle both old string format and new object format
                            source = rel.get('source')
                            target = rel.get('target')
                            
                            if isinstance(source, str):
                                connected_node_ids.add(source)
                            elif isinstance(source, dict):
                                # Try new format (llmGenNodeId) first, fallback to old format (id)
                                node_id = source.get('llmGenNodeId') or source.get('id')
                                if node_id:
                                    connected_node_ids.add(node_id)
                            
                            if isinstance(target, str):
                                connected_node_ids.add(target)
                            elif isinstance(target, dict):
                                # Try new format (llmGenNodeId) first, fallback to old format (id)
                                node_id = target.get('llmGenNodeId') or target.get('id')
                                if node_id:
                                    connected_node_ids.add(node_id)
                        
                        # Check which nodes are orphaned (not in any relationship)
                        orphaned_nodes = []
                        for node in nodes:
                            node_id = node.get('properties', {}).get('llmGenNodeId')
                            if node_id and node_id not in connected_node_ids:
                                orphaned_nodes.append({
                                    'llmGenNodeId': node_id,
                                    'label': node.get('label'),
                                    'name': node.get('properties', {}).get('name')
                                })
                        
                        if orphaned_nodes:
                            logger.warning(f"⚠️ ORPHANED NODES: {len(orphaned_nodes)} nodes have no relationships:")
                            for orphan in orphaned_nodes:
                                logger.warning(f"  - {orphan['label']} [{orphan['llmGenNodeId']}]: {orphan['name']}")
                            logger.warning("💡 Consider: These nodes may need relationships or should be removed")
                        else:
                            logger.info(f"✅ RELATIONSHIP VALIDATION: All {len(nodes)} nodes are connected")
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing error: {str(je)}")
                    logger.error(f"Error at position {je.pos} in content of length {len(content)}")
                    # Log a snippet around the error position
                    error_pos = getattr(je, 'pos', 0)
                    start_pos = max(0, error_pos - 50)
                    end_pos = min(len(content), error_pos + 50)
                    error_snippet = content[start_pos:end_pos]
                    logger.error(f"Content around error: '...{error_snippet}...'")
                    
                    # Try to recover by creating a fallback response
                    logger.warning("Attempting to create fallback response due to JSON parsing error")
                    nodes_result = {"nodes": []}
                    
                except Exception as e:
                    logger.error(f"Error processing node completion: {str(e)}")
                    # Create a fallback response
                    logger.warning("Creating fallback response due to processing error")
                    nodes_result = {"nodes": []}

                # Extract custom node labels and relationships from the schema for validation
                # Extract custom labels and relationships (will be used later for validation)
                custom_labels = []
                custom_relationships = []
                if user_id and 'properties' in memory_graph_schema:
                    # Extract node labels from JSON schema format
                    if 'nodes' in memory_graph_schema['properties'] and 'items' in memory_graph_schema['properties']['nodes']:
                        node_items = memory_graph_schema['properties']['nodes']['items']
                        if 'anyOf' in node_items:
                            for node_def in node_items['anyOf']:
                                if 'properties' in node_def and 'label' in node_def['properties'] and 'enum' in node_def['properties']['label']:
                                    custom_labels.extend(node_def['properties']['label']['enum'])
                    
                    # Extract relationship types from JSON schema format
                    if 'relationships' in memory_graph_schema['properties'] and 'items' in memory_graph_schema['properties']['relationships']:
                        rel_items = memory_graph_schema['properties']['relationships']['items']
                        if 'properties' in rel_items and 'type' in rel_items['properties'] and 'enum' in rel_items['properties']['type']:
                            custom_relationships = rel_items['properties']['type']['enum']
                    
                    logger.info(f"🔧 VALIDATION: Custom labels for validation: {custom_labels}")
                    logger.info(f"🔧 VALIDATION: Custom relationships for validation: {custom_relationships}")

                # Use generate_node_ids to handle the UUID generation and apply property overrides
                generated_nodes_with_ids, _ = await self.generate_node_ids(nodes_result["nodes"], [], custom_labels, custom_relationships, property_overrides)
                
                # Create the Memory node from the original memory_item
                memory_node = {
                    "label": "Memory",
                    "properties": {
                        "id": memory_item.get('id'),
                        "content": memory_item.get('content'),
                        "type": memory_item.get('type', 'TextMemoryItem'),
                        "createdAt": memory_item.get('createdAt'),
                        "topics": memory_item.get('topics', []),
                        "emotion_tags": memory_item.get('emotion_tags', []),
                        "steps": memory_item.get('steps', []),
                        "current_step": memory_item.get('current_step', '')
                    }
                }
                
                # Store memory node separately - it should not be passed to LLM for relationship generation
                # The LLM should only work with the generated nodes and use memory content as context
                logger.info(f"Generated nodes for LLM relationship generation: {generated_nodes_with_ids}")
                logger.info(f"Memory node (for context only): {memory_node}")

                # Extract custom relationships from the schema for dynamic prompting
                custom_relationships = []
                if user_id and 'properties' in memory_graph_schema:
                    # Extract relationship types from JSON schema format
                    if 'relationships' in memory_graph_schema['properties'] and 'items' in memory_graph_schema['properties']['relationships']:
                        rel_items = memory_graph_schema['properties']['relationships']['items']
                        if 'properties' in rel_items and 'type' in rel_items['properties'] and 'enum' in rel_items['properties']['type']:
                            all_relationships = rel_items['properties']['type']['enum']
                            # Filter out system relationships to get custom ones
                            # Import is at top of file (line 17), but ensure it's accessible
                            from models.shared_types import RelationshipType
                            system_relationships = RelationshipType.get_system_relationships()
                            custom_relationships = [rel for rel in all_relationships if rel not in system_relationships]

                # Step 2: Generate relationships between identified nodes (excluding Memory node)
                relationship_prompt_parts = [
                    f"Memory content for context: {memory_item.get('content')}",
                    f"Memory metadata: {json.dumps({k: v for k, v in memory_item.items() if k not in ['content']}, default=str)}",
                    f"Use case info: {json.dumps(usecase_memory_item)}",
                    f"Related memories: {json.dumps(related_memories)}",
                    f"Available nodes: {json.dumps(generated_nodes_with_ids)}",
                    "Create relationships between these specific nodes based on the memory content.",
                    "Use only the provided nodes - do not create new nodes.",
                    "Each relationship must use valid relationship types from the schema.",
                    "Note: The memory content is provided for context only - do not create relationships with Memory nodes.",
                ]
                
                # Add dynamic custom relationship prioritization
                if custom_relationships:
                    logger.info(f"🔧 CUSTOM RELATIONSHIPS DETECTED: {custom_relationships}")
                    prioritize_msg = f"PRIORITIZE these custom relationships when they match the content intent: {', '.join(custom_relationships)}"
                    generic_msg = "When custom relationships are available, do not use these generic relationships (REFERENCES, ASSOCIATED_WITH, RELATED_TO) when the content clearly indicates the specific relationship."
                    relationship_prompt_parts.append(prioritize_msg)
                    relationship_prompt_parts.append(generic_msg)
                    logger.info(f"🔧 ADDED TO PROMPT: {prioritize_msg}")
                    logger.info(f"🔧 ADDED TO PROMPT: {generic_msg}")
                else:
                    logger.info(f"🔧 NO CUSTOM RELATIONSHIPS DETECTED - using generic relationships only")
                
                logger.info(f"🔧 FINAL RELATIONSHIP PROMPT PARTS: {relationship_prompt_parts}")
                
                relationship_prompt_parts.extend([
                    "Ensure relationships accurately reflect the content's meaning.",
                    "Make sure all nodes are connected in a meaningful way."
                ])
                logger.info(f"🔧 FINAL - RELATIONSHIP PROMPT PARTS: {relationship_prompt_parts}")

                relationship_messages = [
                    {
                        "role": "system",
                        "content": """You are a graph relationship creator.
                                    Only create relationships between the provided nodes.
                                    Use valid relationship types from the schema.
                                    Ensure relationships accurately reflect the content.
                                    
                                    RELATIONSHIP PRIORITIZATION RULES:
                                    1. PREFER domain-specific relationships over generic ones when available
                                    
                                    CRITICAL JSON REQUIREMENTS:
                                    1. Return ONLY valid JSON - no explanatory text, comments, or additional content
                                    2. All quotes within string values must be properly escaped with backslashes
                                    3. All backslashes must be escaped as double backslashes
                                    4. Ensure all strings are properly terminated
                                    5. The response must be parseable JSON only
                                    6. If content contains quotes, escape them as \\"
                                    7. If content contains backslashes, escape them as \\\\
                                    8. Do not include any text after the closing brace of the JSON structure"""
                    },
                    {"role": "user", "content": "\n".join(relationship_prompt_parts)}
                ]

                # Use the SAME schema for relationships that was used for node generation
                # This ensures consistency between node and relationship schemas
                # If a custom schema was used for nodes, we should use it for relationships too
                # If system schema was used for nodes, we should use system relationships
                original_user_schema = None
                # Use developer_user_id for schema lookup, fallback to user_id if not available
                schema_user_id = developer_user_id if developer_user_id else user_id
                schema_workspace_id = developer_workspace_id if developer_workspace_id else workspace_id
                logger.info(f"🔒 SCHEMA LOOKUP DEBUG: schema_user_id={schema_user_id}, selected_schema_id={selected_schema_id}, schema_workspace_id={schema_workspace_id}")
                logger.info(f"🔒 SCHEMA LOOKUP DEBUG: developer_user_id={developer_user_id}, user_id={user_id}")
                
                if schema_user_id and selected_schema_id:
                    # A custom schema was selected, try to get it for relationship constraints
                    try:
                        from services.schema_service import SchemaService
                        schema_service = SchemaService()
                        logger.info(f"🔒 SCHEMA LOOKUP DEBUG: Calling get_schemas_by_ids with ids=[{selected_schema_id}], schema_user_id={schema_user_id}, schema_workspace_id={schema_workspace_id}")
                        # Use multi-tenant context passed from caller
                        logger.info(f"🔒 RELATIONSHIP SCHEMA LOOKUP: Using multi-tenant context - org_id={organization_id}, namespace_id={namespace_id}")
                        
                        user_schemas = await schema_service.get_schemas_by_ids(
                            [selected_schema_id], 
                            schema_user_id, 
                            schema_workspace_id,
                            organization_id,
                            namespace_id
                        )
                        logger.info(f"🔒 SCHEMA LOOKUP DEBUG: get_schemas_by_ids returned {len(user_schemas) if user_schemas else 0} schemas")
                        
                        if user_schemas:
                            original_user_schema = user_schemas[0]
                            logger.info(f"🔒 RELATIONSHIP SCHEMA: Using same custom schema as nodes - '{original_user_schema.name}' with {len(original_user_schema.relationship_types)} relationship types")
                        else:
                            logger.info(f"🔒 RELATIONSHIP SCHEMA: Custom schema {selected_schema_id} not found, using system relationships")
                    except Exception as e:
                        logger.warning(f"🔒 RELATIONSHIP SCHEMA: Failed to get custom schema for relationships: {e}")
                        import traceback
                        logger.warning(f"🔒 RELATIONSHIP SCHEMA: Exception traceback: {traceback.format_exc()}")
                else:
                    logger.info(f"🔒 RELATIONSHIP SCHEMA: Using system relationships (same as nodes)")

                # OPTIMIZATION: Skip second LLM call - use relationships from first call
                # The first structured output call already generated relationships, so reuse them
                logger.info(f"🚀 OPTIMIZATION: Skipping second LLM call, using {len(relationships_from_first_call)} relationships from first call")
                
                # Transform relationships from first call to match expected schema
                # First call returns: {'source': {'type': 'X', 'llmGenNodeId': 'Y'}, 'target': {...}, 'type': 'REL'}
                # Expected format: {'source': {'label': 'X', 'id': 'Y'}, 'target': {...}, 'type': 'REL', 'direction': '->'}
                transformed_relationships = []
                for rel in relationships_from_first_call:
                    transformed_rel = {
                        'type': rel.get('type'),
                        'direction': rel.get('direction', '->'),  # Default to outgoing if not specified
                        'source': {
                            'label': rel['source'].get('type') or rel['source'].get('label'),
                            'id': rel['source'].get('llmGenNodeId') or rel['source'].get('id')
                        },
                        'target': {
                            'label': rel['target'].get('type') or rel['target'].get('label'),
                            'id': rel['target'].get('llmGenNodeId') or rel['target'].get('id')
                        }
                    }
                    # Copy any relationship properties
                    if 'properties' in rel:
                        transformed_rel['properties'] = rel['properties']
                    transformed_relationships.append(transformed_rel)
                
                logger.info(f"🔄 TRANSFORMATION: Transformed {len(transformed_relationships)} relationships to expected schema")
                relationships_result = {"relationships": transformed_relationships}
                
                # COMMENTED OUT: Second API call for relationships (no longer needed)
                '''
                # Create restricted relationship schema based on generated nodes (excluding Memory)
                logger.info(f"🔒 SCHEMA DEBUG: memory_graph_schema type={type(memory_graph_schema)}")
                logger.info(f"🔒 SCHEMA DEBUG: original_user_schema type={type(original_user_schema)}")
                relationship_schema = self.create_restricted_relationship_schema(generated_nodes_with_ids, custom_relationships, original_user_schema)

                # Second API call to generate relationships
                # Use dedicated model for higher-quality relationship extraction
                relationship_completion_DISABLED = await self._create_completion_async(
                    model=self.model_mini,
                    messages=relationship_messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Relationship",
                            "schema": relationship_schema,
                            "strict": True
                        }
                    }
                )
                '''  # End of commented-out second LLM call block
                
                # Log relationships from first call
                parsed_relationships = relationships_result.get('relationships', [])
                logger.info(f"📊 RELATIONSHIPS FROM FIRST CALL: Using {len(parsed_relationships)} relationships")
                if parsed_relationships:
                    rel_summary = {}
                    for rel in parsed_relationships:
                        rel_type = rel.get('type', 'Unknown')
                        rel_summary[rel_type] = rel_summary.get(rel_type, 0) + 1
                    logger.info(f"📊 RELATIONSHIP SUMMARY: {dict(rel_summary)}")
                
                # Add automatic EXTRACTED relationships from Memory to all generated nodes
                memory_id = memory_item.get('id')
                automatic_relationships = []
                
                if memory_id and generated_nodes_with_ids:
                    logger.info(f"🔗 AUTO-CONNECT: Creating automatic EXTRACTED relationships from Memory {memory_id} to {len(generated_nodes_with_ids)} generated nodes")
                    for node in generated_nodes_with_ids:
                        # Use node's UUID id - the ID mapping in store_llm_generated_graph will handle the conversion
                        node_id = node.get('properties', {}).get('id')
                        if node_id:
                            # Create LLMGraphRelationship object for automatic connection
                            from models.structured_outputs import LLMGraphRelationship, NodeReference
                            automatic_relationships.append(
                                LLMGraphRelationship(
                                    type="EXTRACTED",
                                    direction="->",
                                    source=NodeReference(label="Memory", id=memory_id),
                                    target=NodeReference(label=node["label"], id=node_id)
                                )
                            )
                    logger.info(f"🔗 AUTO-CONNECT: Created {len(automatic_relationships)} automatic EXTRACTED relationships")
                
                # Combine results - Include Memory node + generated nodes + LLM relationships + automatic relationships
                nodes_with_memory = [memory_node] + generated_nodes_with_ids
                all_relationships = relationships_result["relationships"] + automatic_relationships
                final_result = MemoryGraphSchema(
                    nodes=nodes_with_memory,  # Include Memory node + generated nodes
                    relationships=all_relationships  # LLM relationships + automatic EXTRACTED relationships
                )

                # Calculate metrics
                total_input_tokens = (
                    self.count_tokens(json.dumps(node_messages)) +
                    self.count_tokens(json.dumps(relationship_messages))
                )
                total_output_tokens = (
                    self.count_tokens(json.dumps(nodes_result)) +
                    self.count_tokens(json.dumps(relationships_result))
                )
                
                total_cost = (
                    (total_input_tokens * self.cost_per_input_token) +
                    (total_output_tokens * self.cost_per_output_token)
                )

                metrics = {
                    "schema_token_count_input": total_input_tokens,
                    "schema_token_count_output": total_output_tokens,
                    "schema_total_cost": total_cost,
                    "schema_total_tokens": total_input_tokens + total_output_tokens
                }

                # Store in Neo4j using user schema if available
                neo4j_storage_success = False
                try:
                    logger.info(f"🏗️ STORAGE STEP 1: Attempting to store {len(final_result.nodes)} nodes and {len(final_result.relationships)} relationships in Neo4j")
                    
                    if user_id:
                        logger.info(f"🏗️ STORAGE STEP 2: Using user schema storage for user_id={user_id}")
                        
                        # If no custom schema was selected but we have a system schema, create a schema object for property indexing
                        schema_for_indexing = selected_schema
                        if not selected_schema and memory_graph_schema:
                            logger.info("🔧 PROPERTY INDEXING: Creating system schema object for property indexing")
                            # Create a minimal schema object that property indexing can use
                            from types import SimpleNamespace
                            schema_for_indexing = SimpleNamespace()
                            schema_for_indexing.node_types = {}
                            schema_for_indexing.relationship_types = {}
                            schema_for_indexing.name = "System Schema"
                            schema_for_indexing.id = "system_schema"
                            
                            # Define common indexable properties that exist in most nodes
                            def create_indexable_property(prop_type="string"):
                                prop = SimpleNamespace()
                                prop.type = prop_type
                                prop.required = True
                                prop.enum_values = None
                                return prop
                            
                            # Extract node types from memory_graph_schema if available
                            logger.info(f"🔧 PROPERTY INDEXING DEBUG: memory_graph_schema type={type(memory_graph_schema)}")
                            logger.info(f"🔧 PROPERTY INDEXING DEBUG: memory_graph_schema keys={list(memory_graph_schema.keys()) if isinstance(memory_graph_schema, dict) else 'not dict'}")
                            
                            if isinstance(memory_graph_schema, dict) and 'properties' in memory_graph_schema:
                                nodes_schema = memory_graph_schema.get('properties', {}).get('nodes', {})
                                logger.info(f"🔧 PROPERTY INDEXING DEBUG: nodes_schema keys={list(nodes_schema.keys()) if isinstance(nodes_schema, dict) else 'not dict'}")
                                
                                if 'items' in nodes_schema and 'properties' in nodes_schema['items']:
                                    node_props = nodes_schema['items']['properties']
                                    logger.info(f"🔧 PROPERTY INDEXING DEBUG: node_props keys={list(node_props.keys()) if isinstance(node_props, dict) else 'not dict'}")
                                    
                                    if 'label' in node_props and 'enum' in node_props['label']:
                                        logger.info(f"🔧 PROPERTY INDEXING DEBUG: Found node labels: {node_props['label']['enum']}")
                                        for label in node_props['label']['enum']:
                                            node_type = SimpleNamespace()
                                            node_type.name = label
                                            node_type.properties = {
                                                # Common properties that appear in most nodes and are good for indexing
                                                'name': create_indexable_property("string"),
                                                'description': create_indexable_property("string"),
                                                'id': create_indexable_property("string")
                                            }
                                            schema_for_indexing.node_types[label] = node_type
                                            logger.info(f"🔧 PROPERTY INDEXING: Created indexable properties for node type '{label}': name, description, id")
                                    else:
                                        logger.info(f"🔧 PROPERTY INDEXING DEBUG: No label enum found in node_props")
                                else:
                                    logger.info(f"🔧 PROPERTY INDEXING DEBUG: No items/properties found in nodes_schema")
                            else:
                                logger.info(f"🔧 PROPERTY INDEXING DEBUG: memory_graph_schema doesn't have expected structure")
                                
                            # Fallback: If we can't extract from schema, create properties for common node types we see in the system
                            if not schema_for_indexing.node_types:
                                logger.info("🔧 PROPERTY INDEXING: Using fallback - creating properties for common node types")
                                
                                # Define properties for each node type based on what they actually use
                                node_type_properties = {
                                    'Task': ['title', 'description', 'id'],  # Task uses 'title' not 'name'
                                    'Insight': ['title', 'description', 'id', 'source'],  # Insight uses 'title'
                                    'Meeting': ['title', 'description', 'id'],  # Meeting uses 'title'
                                    'Project': ['name', 'description', 'id'],  # Project uses 'name'
                                    'Company': ['name', 'description', 'id'],  # Company uses 'name'
                                    'Person': ['name', 'description', 'id'],  # Person uses 'name'
                                    'KnowledgeNote': ['name', 'description', 'id', 'content'],
                                    'Workflow': ['name', 'description', 'id'],
                                    'Step': ['name', 'description', 'id'],
                                    'Code': ['title', 'description', 'id'],  # Code uses 'title'
                                    'Opportunity': ['title', 'description', 'id']  # Opportunity uses 'title'
                                }
                                
                                for label, props in node_type_properties.items():
                                    node_type = SimpleNamespace()
                                    node_type.name = label
                                    node_type.properties = {
                                        prop: create_indexable_property("string") for prop in props
                                    }
                                    schema_for_indexing.node_types[label] = node_type
                                    logger.info(f"🔧 PROPERTY INDEXING: Created fallback indexable properties for node type '{label}': {', '.join(props)}")
                        
                        await memory_graph.store_llm_generated_graph(
                            final_result.nodes,  # Access nodes directly from MemoryGraphSchema
                            final_result.relationships,
                            memory_item,
                            neo_session,
                            workspace_id,
                            user_schema=schema_for_indexing
                        )
                        logger.info(f"🏗️ STORAGE STEP 3: ✅ User schema storage completed successfully")
                        neo4j_storage_success = True
                        
                    else:
                        logger.info(f"🏗️ STORAGE STEP 2: Using fallback storage (no user_id)")
                        # Fallback to original method
                        await memory_graph.store_llm_generated_graph(
                            final_result.nodes,  # Access nodes directly from MemoryGraphSchema
                            final_result.relationships,
                            memory_item,
                            neo_session,
                            workspace_id,
                            user_schema=None
                        )
                        logger.info(f"🏗️ STORAGE STEP 3: ✅ Fallback storage completed successfully")
                        neo4j_storage_success = True
                        
                        # Create automatic connections from Memory to all generated nodes AFTER successful storage
                        memory_id = memory_item.get('id')
                        if memory_id and generated_nodes_with_ids:
                            logger.info(f"🔗 AUTO-CONNECT: Creating automatic relationships from Memory {memory_id} to {len(generated_nodes_with_ids)} generated nodes")
                            automatic_relationships = []
                            
                            for node in generated_nodes_with_ids:
                                node_id = node.get('properties', {}).get('id')
                                if node_id:
                                    automatic_relationships.append({
                                        "source": memory_id,
                                        "target": node_id,
                                        "type": "EXTRACTED"  # Use EXTRACTED to indicate memory extracted/generated this node
                                    })
                            
                            if automatic_relationships:
                                logger.info(f"🔗 AUTO-CONNECT: Creating {len(automatic_relationships)} automatic EXTRACTED relationships")
                                # Convert to LLMGraphRelationship objects
                                from models.memory_models import LLMGraphRelationship
                                automatic_rel_objects = [
                                    LLMGraphRelationship(**rel) for rel in automatic_relationships
                                ]
                                
                                # Create the automatic relationships in Neo4j with extraction metadata
                                from datetime import datetime, timezone
                                extraction_metadata = {
                                    "extraction_method": "llm_schema_generation",
                                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                                    "schema_id": memory_item.get('schema_id'),
                                    "workspace_id": workspace_id,
                                    "user_id": user_id
                                }
                                
                                for rel in automatic_rel_objects:
                                    try:
                                        await memory_graph._create_relationship(
                                            neo_session=neo_session, 
                                            relationship=rel, 
                                            common_metadata=extraction_metadata
                                        )
                                        logger.info(f"🔗 AUTO-CONNECT: ✅ Created EXTRACTED relationship: {rel.source} -> {rel.target}")
                                    except Exception as rel_error:
                                        logger.error(f"🔗 AUTO-CONNECT: ❌ Failed to create relationship {rel.source} -> {rel.target}: {rel_error}")
                                
                                logger.info(f"🔗 AUTO-CONNECT: ✅ Completed automatic memory connections")
                            else:
                                logger.warning(f"🔗 AUTO-CONNECT: No valid node IDs found for automatic connections")
                        
                except Exception as neo4j_error:
                    logger.error(f"🏗️ STORAGE STEP 3: ❌ CRITICAL ERROR storing in Neo4j: {str(neo4j_error)}")
                    logger.error("🏗️ STORAGE ERROR: Full traceback:", exc_info=True)
                    logger.error(f"🏗️ STORAGE ERROR: This means Memory nodes and relationships will NOT be created!")
                    logger.error(f"🏗️ STORAGE ERROR: Nodes that failed to store: {[node.label for node in final_result.nodes]}")
                    logger.error(f"🏗️ STORAGE ERROR: Relationships that failed to store: {[rel.type for rel in final_result.relationships]}")
                    
                    # CRITICAL: Memory node creation failed - this will prevent cache population
                    logger.error(f"🏗️ FALLBACK: Memory node creation failed - ActiveNodeRel cache will not be populated")
                    
                    neo4j_storage_success = False

                return {
                    "data": final_result,
                    "metrics": metrics
                }

        except Exception as e:
            logger.error(f"Error in schema generation: {e}")
            logger.error("Full traceback:", exc_info=True)
            # Create empty MemoryGraphSchema for error case
            empty_schema = MemoryGraphSchema(nodes=[], relationships=[])
            return {
                "data": empty_schema,
                "metrics": {
                    "schema_token_count_input": total_input_tokens if 'total_input_tokens' in locals() else 0,
                    "schema_token_count_output": total_output_tokens if 'total_output_tokens' in locals() else 0,
                    "schema_total_cost": total_cost if 'total_cost' in locals() else 0,
                    "schema_total_tokens": (total_input_tokens + total_output_tokens) if 'total_input_tokens' in locals() and 'total_output_tokens' in locals() else 0
                },
                "error": str(e)
            }

    def _fix_malformed_json(self, json_str: str) -> str:
        """
        Fix common JSON malformation issues that LLMs produce.
        
        Args:
            json_str (str): The malformed JSON string
            
        Returns:
            str: The fixed JSON string
        """
        try:
            # First, try to parse it as-is to see if it's already valid
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        
        # Fix 1: Handle unescaped quotes in content fields
        # Look for content fields that contain unescaped quotes
        import re
        
        # Pattern to match content fields with potential unescaped quotes
        content_pattern = r'"content":\s*"([^"]*(?:\\"[^"]*)*)"'
        
        def fix_content_quotes(match):
            content = match.group(1)
            # Escape any unescaped quotes in the content
            # First, unescape any already escaped quotes
            content = content.replace('\\"', '"')
            # Then escape all quotes
            content = content.replace('"', '\\"')
            # Also escape backslashes
            content = content.replace('\\', '\\\\')
            return f'"content": "{content}"'
        
        fixed_json = re.sub(content_pattern, fix_content_quotes, json_str)
        
        # Fix 2: Handle unterminated strings by finding the last valid closing brace
        try:
            # Try to find a valid JSON structure
            brace_count = 0
            last_valid_pos = -1
            
            for i, char in enumerate(fixed_json):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i
                        break
            
            if last_valid_pos > 0:
                fixed_json = fixed_json[:last_valid_pos + 1]
            
            # Test if the fixed JSON is valid
            json.loads(fixed_json)
            return fixed_json
            
        except json.JSONDecodeError:
            # If still invalid, try a more aggressive approach
            # Remove any trailing content after the last valid JSON structure
            try:
                # Find the last occurrence of a complete JSON object
                last_brace = fixed_json.rfind('}')
                if last_brace > 0:
                    # Look for the matching opening brace
                    brace_count = 1
                    for i in range(last_brace - 1, -1, -1):
                        if fixed_json[i] == '}':
                            brace_count += 1
                        elif fixed_json[i] == '{':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found the complete JSON object
                                complete_json = fixed_json[i:last_brace + 1]
                                json.loads(complete_json)  # Test if valid
                                return complete_json
            except:
                pass
        
        # If all else fails, return the original string and let the caller handle the error
        return json_str

    def _get_all_source_properties(self, pattern_properties: Dict[str, Dict]) -> List[str]:
        """Extract all unique source properties from pattern properties"""
        all_props = set()
        for pattern_data in pattern_properties.values():
            source_props = pattern_data.get('source_properties', [])
            if isinstance(source_props, list):
                all_props.update(source_props)
        return sorted(list(all_props))
    
    def _get_all_target_properties(self, pattern_properties: Dict[str, Dict]) -> List[str]:
        """Extract all unique target properties from pattern properties"""
        all_props = set()
        for pattern_data in pattern_properties.values():
            target_props = pattern_data.get('target_properties', [])
            if isinstance(target_props, list):
                all_props.update(target_props)
        return sorted(list(all_props))

    def create_dynamic_cypher_schema(self, available_nodes: List[str], available_relationships: List[str] = None, relationship_patterns: List[Dict] = None) -> Dict[str, Any]:
        """Create an enhanced pattern-selection schema with discovered patterns as enums and property constraints"""
        logger.info("🔧 Creating ENHANCED pattern-selection schema with discovered patterns")
        
        # 🔍 DEBUG: Log input parameters
        logger.info(f"🔍 SCHEMA CREATION DEBUG:")
        logger.info(f"🔍   available_nodes: {available_nodes}")
        logger.info(f"🔍   available_relationships: {available_relationships}")
        logger.info(f"🔍   relationship_patterns count: {len(relationship_patterns) if relationship_patterns else 0}")
        if relationship_patterns:
            logger.info(f"🔍   First 3 patterns: {relationship_patterns[:3]}")
        
        if not available_relationships:
            available_relationships = []
        
        if not relationship_patterns:
            relationship_patterns = []
        
        # Extract valid combinations and build property mappings
        valid_pattern_strings = []
        pattern_properties = {}
        
        # 🔍 DEBUG: Log pattern processing
        logger.info(f"🔍 PATTERN PROCESSING DEBUG:")
        logger.info(f"🔍   Processing {len(relationship_patterns)} relationship patterns")
        
        # Extract valid node-relationship combinations from patterns
        for i, pattern in enumerate(relationship_patterns[:30]):  # Use top 30 patterns
            logger.info(f"🔍   Pattern {i+1}: {pattern}")
            
            # Handle both formats: Parse cache uses 'source'/'target'/'relationship', Neo4j discovery uses 'source_label'/'target_label'/'relationship_type'
            source_label = pattern.get('source') or pattern.get('source_label')
            relationship_type = pattern.get('relationship') or pattern.get('relationship_type')
            target_label = pattern.get('target') or pattern.get('target_label')
            count = pattern.get('count', 0)
            source_properties = pattern.get('source_properties', [])
            target_properties = pattern.get('target_properties', [])
            
            logger.info(f"🔍   Extracted: source={source_label}, rel={relationship_type}, target={target_label}")
            
            if source_label and relationship_type and target_label:
                # Create pattern string in format: "Source -> RELATIONSHIP -> Target"
                pattern_string = f"{source_label} -> {relationship_type} -> {target_label}"
                valid_pattern_strings.append(pattern_string)
                logger.info(f"🔍   ✅ Added pattern: {pattern_string}")
                
                # Store property information for this pattern
                pattern_properties[pattern_string] = {
                    "source_properties": source_properties if isinstance(source_properties, list) else [],
                    "target_properties": target_properties if isinstance(target_properties, list) else []
                }
            else:
                logger.warning(f"🔍   ❌ Skipped pattern {i+1}: missing required fields")
        
        logger.info(f"🔧 Enhanced schema: {len(valid_pattern_strings)} valid patterns discovered")
        logger.info(f"🔧 Top patterns: {valid_pattern_strings[:10]}")
        
        # Create enhanced schema with discovered patterns as enums and proper operator/value structure
        enhanced_schema = {
            "type": "object",
            "title": "GraphPatternSelection",
            "description": f"Select the most relevant graph pattern from actual database patterns. Available: {len(valid_pattern_strings)} patterns discovered from your data.",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original natural language query from the user"
                },
                "chosen_pattern": {
                        "type": "string",
                        "enum": valid_pattern_strings[:25],  # Top 25 most common patterns
                        "description": "Choose the most relevant graph pattern that matches the user's query intent"
                    },
                    "source_properties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "property": {
                                    "type": "string",
                                    "enum": self._get_all_source_properties(pattern_properties),
                                    "description": "The property of the source node to filter on"
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": ["=", "<>", ">", ">=", "<", "<=", "CONTAINS", "STARTS WITH", "ENDS WITH", "IN", "NOT IN", "IS NULL", "IS NOT NULL", "=~"],
                                    "default": "CONTAINS"
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Example value from the user query to filter against"
                                }
                            },
                            "required": ["property", "value"]
                        },
                        "uniqueItems": True,
                        "minItems": 0,
                        "maxItems": 3,
                        "description": "Select 0-3 properties from the source node, with example values"
                    },
                    "target_properties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "property": {
                                    "type": "string",
                                    "enum": self._get_all_target_properties(pattern_properties),
                                    "description": "The property of the target node to filter on"
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": ["=", "<>", ">", ">=", "<", "<=", "CONTAINS", "STARTS WITH", "ENDS WITH", "IN", "NOT IN", "IS NULL", "IS NOT NULL", "=~"],
                                    "default": "CONTAINS"
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Example value from the user query to filter against"
                                }
                            },
                            "required": ["property", "value"]
                        },
                        "uniqueItems": True,
                        "minItems": 0,
                        "maxItems": 3,
                        "description": "Select 0-3 properties from the target node, with example values"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this pattern is most relevant for the user's query"
                    }
            },
            "required": ["query", "chosen_pattern", "source_properties", "target_properties", "reasoning"],
            "additionalProperties": False
        }
        
        # Add pattern-specific property constraints as additional context
        if pattern_properties:
            enhanced_schema["pattern_properties"] = pattern_properties
            logger.info(f"🔧 Added property constraints for {len(pattern_properties)} patterns")
        
        logger.info(f"🔧 Created enhanced pattern selection schema with {len(valid_pattern_strings)} discovered patterns")
        return enhanced_schema
    
    def _get_all_source_properties(self, pattern_properties: Dict[str, Dict]) -> List[str]:
        """Extract all unique source properties from pattern_properties"""
        all_props = set()
        for pattern_info in pattern_properties.values():
            source_props = pattern_info.get("source_properties", [])
            if isinstance(source_props, list):
                all_props.update(source_props)
        return sorted(list(all_props))
    
    def _get_all_target_properties(self, pattern_properties: Dict[str, Dict]) -> List[str]:
        """Extract all unique target properties from pattern_properties"""
        all_props = set()
        for pattern_info in pattern_properties.values():
            target_props = pattern_info.get("target_properties", [])
            if isinstance(target_props, list):
                all_props.update(target_props)
        return sorted(list(all_props))

    def _group_filters_by_property(self, filters: List[Dict], node_prefix: str) -> List[str]:
        """Group filters by property and combine multiple values for the same property with OR logic"""
        if not filters:
            return []
        
        # Group filters by property
        property_groups = {}
        for filter_obj in filters:
            prop = filter_obj.get('property', '')
            operator = filter_obj.get('operator', 'CONTAINS')
            value = filter_obj.get('value', '')
            
            if not prop or not value:
                continue
            
            if prop not in property_groups:
                property_groups[prop] = []
            
            # Build the condition string
            if operator == 'CONTAINS':
                condition = f"{node_prefix}.{prop} CONTAINS '{value}'"
            elif operator == 'STARTS WITH':
                condition = f"{node_prefix}.{prop} STARTS WITH '{value}'"
            elif operator == '=':
                condition = f"{node_prefix}.{prop} = '{value}'"
            else:
                condition = f"{node_prefix}.{prop} {operator} '{value}'"
            
            property_groups[prop].append(condition)
        
        # Combine conditions for each property
        combined_conditions = []
        for prop, conditions in property_groups.items():
            if len(conditions) == 1:
                # Single condition for this property
                combined_conditions.append(conditions[0])
            else:
                # Multiple conditions for same property - combine with OR
                or_conditions = ' OR '.join(conditions)
                combined_conditions.append(f"({or_conditions})")
                logger.info(f"🔧 COMBINED PROPERTY: {prop} has {len(conditions)} conditions combined with OR")
        
        return combined_conditions

    def build_cypher_from_pattern(
        self, 
        selected_pattern: str, 
        source_filters: List[Dict] = None, 
        target_filters: List[Dict] = None,
        user_id: str = None
    ) -> str:
        """
        Build a Cypher query from a selected pattern and filters using templates.
        This replaces the complex LLM-generated Cypher with reliable templates.
        
        Args:
            selected_pattern: Pattern like "Function-CALLS->Function"
            source_filters: List of filters for source node
            target_filters: List of filters for target node
            user_id: User ID for ACL filtering
            
        Returns:
            Complete Cypher query string
        """
        logger.info(f"🔧 Building Cypher from pattern: {selected_pattern}")
        
        if not source_filters:
            source_filters = []
        if not target_filters:
            target_filters = []
        
        # Parse the pattern: "Source -> RELATIONSHIP -> Target" or Cypher format
        try:
            if ' -> ' in selected_pattern:
                # Arrow format: "Source -> RELATIONSHIP -> Target"
                parts = selected_pattern.split(' -> ')
                if len(parts) == 3:
                    source_label = parts[0].strip()
                    relationship_type = parts[1].strip()
                    target_label = parts[2].strip()
                else:
                    # Fallback parsing for malformed pattern
                    logger.warning(f"Malformed pattern '{selected_pattern}', expected 3 parts separated by ' -> '")
                    source_label = 'Memory'
                    target_label = 'Memory'
                    relationship_type = 'RELATED_TO'
            elif 'MATCH' in selected_pattern and '(' in selected_pattern:
                # Cypher format: "MATCH (m:Function)-[r:CREATED_BY]->(n:Person)"
                logger.info(f"🔧 Parsing Cypher pattern: {selected_pattern}")
                import re
                
                # Extract source label: (m:SourceLabel)
                source_match = re.search(r'\(m:(\w+)\)', selected_pattern)
                source_label = source_match.group(1) if source_match else 'Memory'
                
                # Extract relationship type: [r:RELATIONSHIP_TYPE]
                rel_match = re.search(r'\[r:(\w+)\]', selected_pattern)
                relationship_type = rel_match.group(1) if rel_match else 'RELATED_TO'
                
                # Extract target label: (n:TargetLabel)
                target_match = re.search(r'\(n:(\w+)\)', selected_pattern)
                target_label = target_match.group(1) if target_match else 'Memory'
                
                logger.info(f"🔧 Parsed Cypher - Source: {source_label}, Rel: {relationship_type}, Target: {target_label}")
            else:
                # Fallback for unexpected format
                logger.warning(f"Unexpected pattern format '{selected_pattern}', expected ' -> ' separators or Cypher format")
                source_label = 'Memory'
                target_label = 'Memory' 
                relationship_type = 'RELATED_TO'
                
        except Exception as e:
            logger.error(f"Error parsing pattern '{selected_pattern}': {e}")
            source_label = 'Memory'
            target_label = 'Memory'
            relationship_type = 'RELATED_TO'
        
        logger.info(f"🔧 Parsed pattern - Source: {source_label}, Relationship: {relationship_type}, Target: {target_label}")
        
        # Build the base MATCH clause using path matching for better performance
        base_match = f"MATCH path = (m:{source_label})-[:{relationship_type}*1..2]->(n:{target_label})"
        
        # Build WHERE conditions with intelligent property grouping
        conditions = []
        
        # Add source node filters (grouped by property with OR logic for duplicates)
        source_conditions = self._group_filters_by_property(source_filters or [], 'm')
        conditions.extend(source_conditions)
        
        # Add target node filters (grouped by property with OR logic for duplicates)
        target_conditions = self._group_filters_by_property(target_filters or [], 'n')
        conditions.extend(target_conditions)
        
        # Add ACL conditions for BOTH source and target nodes (always required for security)
        source_acl_conditions = [
            "m.user_id = $user_id",
            "any(x IN coalesce(m.user_read_access, []) WHERE x IN $user_read_access)",
            "any(x IN coalesce(m.workspace_read_access, []) WHERE x IN $workspace_read_access)",
            "any(x IN coalesce(m.role_read_access, []) WHERE x IN $role_read_access)",
            "any(x IN coalesce(m.organization_read_access, []) WHERE x IN $organization_read_access)",
            "any(x IN coalesce(m.namespace_read_access, []) WHERE x IN $namespace_read_access)"
        ]
        
        target_acl_conditions = [
            "n.user_id = $user_id",
            "any(x IN coalesce(n.user_read_access, []) WHERE x IN $user_read_access)",
            "any(x IN coalesce(n.workspace_read_access, []) WHERE x IN $workspace_read_access)",
            "any(x IN coalesce(n.role_read_access, []) WHERE x IN $role_read_access)",
            "any(x IN coalesce(n.organization_read_access, []) WHERE x IN $organization_read_access)",
            "any(x IN coalesce(n.namespace_read_access, []) WHERE x IN $namespace_read_access)"
        ]
        
        # Both source and target nodes must be accessible
        combined_acl_condition = f"({' OR '.join(source_acl_conditions)}) AND ({' OR '.join(target_acl_conditions)})"
        
        # Combine all conditions
        all_conditions = []
        # Always add ACL conditions for security
        all_conditions.append(combined_acl_condition)
        if conditions:
            all_conditions.extend(conditions)
        
        # Build the complete query
        query_parts = [base_match]
        
        if all_conditions:
            query_parts.append(f"WHERE {' AND '.join(all_conditions)}")
        
        # Add deduplication and return clause
        query_parts.extend([
            "WITH DISTINCT path",
            """RETURN {
                path: path,
                nodes: [n IN nodes(path) | { id: n.id, labels: labels(n), properties: properties(n) }],
                relationships: [r IN relationships(path) | {
                    type: type(r), properties: properties(r),
                    startNode: startNode(r).id, endNode: endNode(r).id
                }]
            } AS result"""
        ])
        
        # Join query parts
        final_query = "\n".join(query_parts)
        
        logger.info(f"🔧 Generated Cypher query:\n{final_query}")
        return final_query

    def create_restricted_relationship_schema(self, nodes_with_ids: List[Dict], custom_relationships: List[str] = None, user_schema: Dict = None) -> Dict[str, Any]:
        """
        Creates a relationship schema that only allows connections between the nodes we generated.
        
        Args:
            nodes_with_ids: List of nodes generated in step 1, each with UUID
        
        Returns:
            A modified relationship schema with restricted node options
        """
        # Create mappings of available nodes by label
        available_nodes_by_label = {}
        for node in nodes_with_ids:
            label = node["label"]
            node_id = node["properties"]["id"]
            if label not in available_nodes_by_label:
                available_nodes_by_label[label] = []
            available_nodes_by_label[label].append(node_id)

        logger.info(f"🔒 AVAILABLE NODES DEBUG: nodes_with_ids count={len(nodes_with_ids)}")
        logger.info(f"🔒 AVAILABLE NODES DEBUG: available_nodes_by_label={list(available_nodes_by_label.keys())}")
        for label, ids in available_nodes_by_label.items():
            logger.info(f"🔒 AVAILABLE NODES DEBUG: {label} -> {len(ids)} nodes")

        # Create the restricted schema with user relationship constraints
        if user_schema and self._has_relationship_constraints(user_schema):
            # Use user-defined relationship constraints
            restricted_schema = self._create_constrained_relationship_schema(nodes_with_ids, available_nodes_by_label, user_schema, custom_relationships)
        else:
            # Fallback to original unconstrained schema
            restricted_schema = {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": self._get_all_relationship_types(custom_relationships)
                            },
                            "direction": {"type": "string", "enum": ["->", "<-"]},
                            "source": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "enum": list(available_nodes_by_label.keys())
                                    },
                                    "id": {
                                        "type": "string",
                                        "enum": [node["properties"]["id"] for node in nodes_with_ids]
                                    }
                                },
                                "required": ["label", "id"],
                                "additionalProperties": False
                            },
                            "target": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "enum": list(available_nodes_by_label.keys())
                                    },
                                    "id": {
                                        "type": "string",
                                        "enum": [node["properties"]["id"] for node in nodes_with_ids]
                                    }
                                },
                                "required": ["label", "id"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["type", "direction", "source", "target"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["relationships"],
            "additionalProperties": False
        }

        return restricted_schema

    def _has_relationship_constraints(self, user_schema) -> bool:
        """Check if user schema has relationship constraints defined"""
        try:
            if user_schema is None:
                return False
                
            # Handle Pydantic objects
            if hasattr(user_schema, 'relationship_types'):
                relationship_types = user_schema.relationship_types
                for rel_type, rel_def in relationship_types.items():
                    if hasattr(rel_def, 'allowed_source_types') and hasattr(rel_def, 'allowed_target_types'):
                        if rel_def.allowed_source_types and rel_def.allowed_target_types:
                            return True
                return False
            
            # Handle dictionary format
            elif isinstance(user_schema, dict) and 'relationship_types' in user_schema:
                for rel_type, rel_def in user_schema['relationship_types'].items():
                    if 'allowed_source_types' in rel_def and 'allowed_target_types' in rel_def:
                        return True
                return False
            
            return False
        except Exception as e:
            logger.error(f"Error checking relationship constraints: {e}")
            return False

    def _create_constrained_relationship_schema(self, nodes_with_ids: List[Dict], available_nodes_by_label: Dict, user_schema, custom_relationships: List[str] = None) -> Dict[str, Any]:
        """Create relationship schema with user-defined constraints"""
        try:
            logger.info("🔒 Creating constrained relationship schema based on user-defined rules")
            
            # Extract relationship constraints from user schema
            relationship_constraints = {}
            
            # Handle Pydantic objects
            if hasattr(user_schema, 'relationship_types'):
                relationship_types = user_schema.relationship_types
                for rel_type, rel_def in relationship_types.items():
                    if hasattr(rel_def, 'allowed_source_types') and hasattr(rel_def, 'allowed_target_types'):
                        if rel_def.allowed_source_types and rel_def.allowed_target_types:
                            relationship_constraints[rel_type] = {
                                'allowed_source_types': rel_def.allowed_source_types,
                                'allowed_target_types': rel_def.allowed_target_types
                            }
            
            # Handle dictionary format
            elif isinstance(user_schema, dict) and 'relationship_types' in user_schema:
                for rel_type, rel_def in user_schema['relationship_types'].items():
                    if 'allowed_source_types' in rel_def and 'allowed_target_types' in rel_def:
                        relationship_constraints[rel_type] = {
                            'allowed_source_types': rel_def['allowed_source_types'],
                            'allowed_target_types': rel_def['allowed_target_types']
                        }
            
            logger.info(f"🔒 Found {len(relationship_constraints)} constrained relationship types: {list(relationship_constraints.keys())}")
            
            # Create conditional schema using anyOf for each relationship type
            relationship_items = []
            
            for rel_type, constraints in relationship_constraints.items():
                    # Only include this relationship if we have nodes that can use it
                    available_sources = [label for label in constraints['allowed_source_types'] if label in available_nodes_by_label]
                    available_targets = [label for label in constraints['allowed_target_types'] if label in available_nodes_by_label]

                    logger.info(f"🔒 CONSTRAINT DEBUG: {rel_type} - required sources: {constraints['allowed_source_types']}, available: {available_sources}")
                    logger.info(f"🔒 CONSTRAINT DEBUG: {rel_type} - required targets: {constraints['allowed_target_types']}, available: {available_targets}")

                    if available_sources and available_targets:
                        logger.info(f"🔒 Adding constrained relationship {rel_type}: {available_sources} -> {available_targets}")
                        
                        relationship_item = {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [rel_type]
                                },
                                "direction": {"type": "string", "enum": ["->", "<-"]},
                                "source": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "enum": available_sources
                                        },
                                        "id": {
                                            "type": "string",
                                            "enum": [node["properties"]["id"] for node in nodes_with_ids if node["label"] in available_sources]
                                        }
                                    },
                                    "required": ["label", "id"],
                                    "additionalProperties": False
                                },
                                "target": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "enum": available_targets
                                        },
                                        "id": {
                                            "type": "string",
                                            "enum": [node["properties"]["id"] for node in nodes_with_ids if node["label"] in available_targets]
                                        }
                                    },
                                    "required": ["label", "id"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["type", "direction", "source", "target"],
                            "additionalProperties": False
                        }
                        relationship_items.append(relationship_item)
                    else:
                        logger.info(f"🔒 Skipping relationship {rel_type}: no available nodes (sources: {available_sources}, targets: {available_targets})")
            
            if not relationship_items:
                logger.warning("🔒 No valid constrained relationships found, falling back to unconstrained schema")
                return self._create_unconstrained_schema(nodes_with_ids, available_nodes_by_label, custom_relationships)
            
            # Create the constrained schema
            constrained_schema = {
                "type": "object",
                "properties": {
                    "relationships": {
                        "type": "array",
                        "items": {
                            "anyOf": relationship_items
                        }
                    }
                },
                "required": ["relationships"],
                "additionalProperties": False
            }
            
            logger.info(f"🔒 Created constrained schema with {len(relationship_items)} relationship types")
            return constrained_schema
            
        except Exception as e:
            logger.error(f"Error creating constrained relationship schema: {e}")
            return self._create_unconstrained_schema(nodes_with_ids, available_nodes_by_label, custom_relationships)

    def _create_unconstrained_schema(self, nodes_with_ids: List[Dict], available_nodes_by_label: Dict, custom_relationships: List[str] = None) -> Dict[str, Any]:
        """Create unconstrained relationship schema (fallback)"""
        return {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": self._get_all_relationship_types(custom_relationships)
                            },
                            "direction": {"type": "string", "enum": ["->", "<-"]},
                            "source": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "enum": list(available_nodes_by_label.keys())
                                    },
                                    "id": {
                                        "type": "string",
                                        "enum": [node["properties"]["id"] for node in nodes_with_ids]
                                    }
                                },
                                "required": ["label", "id"],
                                "additionalProperties": False
                            },
                            "target": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "enum": list(available_nodes_by_label.keys())
                                    },
                                    "id": {
                                        "type": "string",
                                        "enum": [node["properties"]["id"] for node in nodes_with_ids]
                                    }
                                },
                                "required": ["label", "id"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["type", "direction", "source", "target"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["relationships"],
            "additionalProperties": False
        }

    def _get_all_relationship_types(self, custom_relationships: List[str] = None) -> List[str]:
        """
        Get available relationship types. When custom relationships are provided from a user schema,
        use ONLY those custom relationships to enforce schema constraints.
        
        Args:
            custom_relationships: List of custom relationship types from user schema
            
        Returns:
            Custom relationship types if provided, otherwise system relationship types
        """
        # Base system relationships (used only when no custom schema is provided)
        system_relationships = [
            "CREATED_BY", "WORKS_AT", "ASSOCIATED_WITH",
            "CONTAINS", "ASSIGNED_TO", "MANAGED_BY", 
            "RELATED_TO", "HAS", "IS_A", "PARTICIPATED_IN", 
            "BELONGS_TO", "REPORTED_BY", "REFERENCES"
        ]
        
        if custom_relationships:
            # Use ONLY custom relationships to enforce schema constraints
            logger.info(f"🔧 RELATIONSHIP SCHEMA INCLUDES (custom only): {custom_relationships}")
            return custom_relationships
        
        logger.info(f"🔧 RELATIONSHIP SCHEMA INCLUDES (system only): {system_relationships}")
        return system_relationships

    async def _enhance_query_with_property_suggestions(
        self,
        cypher_query: str,
        user_query: str,
        acl_filter: Dict[str, Any],
        enhanced_schema_cache: Dict[str, Any],
        memory_graph: "MemoryGraph",
        neo_session: Optional[AsyncSession] = None
    ) -> Optional[str]:
        """Enhance Cypher query by replacing LLM-suggested property values with Qdrant matches"""
        
        try:
            logger.info(f"🚀 PROPERTY ENHANCEMENT: Starting query enhancement")
            
            # Parse property filters from the Cypher query
            property_filters = self._parse_property_filters_from_cypher(cypher_query)
            
            logger.info(f"🚀 PROPERTY ENHANCEMENT: Parsed {len(property_filters) if property_filters else 0} property filters from query")
            if property_filters:
                for i, pf in enumerate(property_filters):
                    logger.info(f"🚀 PROPERTY ENHANCEMENT: Filter {i+1}: {pf.get('node_type', 'unknown')}.{pf.get('property_name', 'unknown')} = '{pf.get('value', 'unknown')}'")
            
            if not property_filters:
                logger.warning(f"🚀 PROPERTY ENHANCEMENT: No property filters found in query, returning original")
                return cypher_query, {}
                
            logger.info(f"🚀 PROPERTY ENHANCEMENT: Found {len(property_filters)} property filters to enhance")
            
            # Search for better property values in parallel
            enhancement_tasks = []
            for prop_filter in property_filters:
                node_type = prop_filter['node_type']
                property_name = prop_filter['property_name']
                llm_value = prop_filter['value']
                
                logger.info(f"🚀 PROPERTY ENHANCEMENT: Checking eligibility for {node_type}.{property_name} = '{llm_value}'")
                
                # Check if this node type has enough instances to warrant vector search
                if neo_session:
                    try:
                        node_count = await self._get_node_count(node_type, acl_filter, neo_session)
                        if node_count < 15:
                            logger.info(f"🚀 PROPERTY ENHANCEMENT: ❌ SKIPPED {node_type}.{property_name} - only {node_count} instances (need ≥15)")
                            continue
                        logger.info(f"🚀 PROPERTY ENHANCEMENT: ✅ Node count OK for {node_type}.{property_name} ({node_count} instances)")
                    except Exception as e:
                        logger.info(f"🚀 PROPERTY ENHANCEMENT: Node count check failed for {node_type}: {e}, proceeding anyway")
                else:
                    logger.info(f"🚀 PROPERTY ENHANCEMENT: ✅ No neo_session, skipping count check for {node_type}.{property_name}")
                    
                # Determine property schema for filtering
                logger.info(f"🚀 PROPERTY ENHANCEMENT: Checking schema context for {node_type}.{property_name}")
                schema_context = self._determine_property_schema(
                    node_type, property_name, enhanced_schema_cache
                )
                
                logger.info(f"🚀 PROPERTY ENHANCEMENT: Schema context result: {schema_context}")
                logger.info(f"🚀 PROPERTY ENHANCEMENT: Available indexable_properties keys: {list(enhanced_schema_cache.get('indexable_properties', {}).keys())}")
                
                if not schema_context:
                    logger.info(f"🚀 PROPERTY ENHANCEMENT: ❌ SKIPPED {node_type}.{property_name} - No schema context found")
                    logger.info(f"🚀 PROPERTY ENHANCEMENT: 🔧 BYPASSING schema context requirement for testing...")
                    # Create a minimal schema context for testing
                    schema_context = {
                        'schema_id': None,
                        'schema_name': 'system',
                        'node_type': node_type,
                        'property_name': property_name
                    }
                
                logger.info(f"🚀 PROPERTY ENHANCEMENT: ✅ Using schema context for {node_type}.{property_name}: {schema_context}")
                
                # Create search task
                search_task = self._search_property_values(
                    node_type=node_type,
                    property_name=property_name,
                    llm_suggested_value=llm_value,
                    acl_filter=acl_filter,
                    schema_context=schema_context,
                    memory_graph=memory_graph,
                    top_k=5
                )
                enhancement_tasks.append((prop_filter, search_task))
            
            if not enhancement_tasks:
                logger.info(f"🚀 PROPERTY ENHANCEMENT: No properties eligible for enhancement")
                return cypher_query, {}
            
            # BATCH OPTIMIZATION: Generate all embeddings in a single API call
            logger.info(f"🚀 PROPERTY ENHANCEMENT: 🚀 Generating {len(enhancement_tasks)} embeddings in batch")
            
            # Collect all property info for batch embedding
            property_infos = []
            for prop_filter, search_task in enhancement_tasks:
                # prop_filter is already a dict with node_type, property_name, value, etc.
                property_infos.append({
                    'node_type': prop_filter['node_type'],
                    'property_name': prop_filter['property_name'],
                    'llm_value': prop_filter['value'],
                    'prop_filter': prop_filter,
                    'search_task': search_task
                })
            
            # Generate batch embeddings
            search_contents = [f"Node: {p['node_type']}, Property: {p['property_name']}: {p['llm_value']}" for p in property_infos]
            batch_embeddings = await self._generate_batch_property_embeddings(search_contents, memory_graph)
            
            # Map embeddings back to property searches
            for i, prop_info in enumerate(property_infos):
                prop_info['embedding'] = batch_embeddings[i] if i < len(batch_embeddings) else None
            
            # Execute all searches with pre-generated embeddings
            logger.info(f"🚀 PROPERTY ENHANCEMENT: Running {len(enhancement_tasks)} property searches with batch embeddings")
            enhanced_query = cypher_query
            all_parameters = {}  # Collect all parameters for parameterized query
            param_counter = 0  # Global counter for unique parameter names
            
            for prop_info in property_infos:
                prop_filter = prop_info['prop_filter']
                embedding = prop_info['embedding']
                
                # Skip if embedding generation failed
                if embedding is None:
                    logger.warning(f"🚀 PROPERTY ENHANCEMENT: Skipping {prop_filter} - embedding generation failed")
                    continue
                
                try:
                    # Call search with pre-generated embedding
                    node_type = prop_info['node_type']
                    property_name = prop_info['property_name']
                    llm_value = prop_info['llm_value']
                    
                    # Find the original schema context and acl_filter
                    # We need to reconstruct the call parameters
                    matching_filter_idx = next((i for i, f in enumerate(property_filters) if f['node_type'] == node_type and f['property_name'] == property_name), None)
                    if matching_filter_idx is None:
                        continue
                    
                    # Get schema context from the original loop
                    schema_context = {'schema_id': None, 'schema_name': 'system', 'node_type': node_type, 'property_name': property_name}
                    
                    search_results = await self._search_property_values(
                        node_type=node_type,
                        property_name=property_name,
                        llm_suggested_value=llm_value,
                        acl_filter=acl_filter,
                        schema_context=schema_context,
                        memory_graph=memory_graph,
                        top_k=5,
                        pre_generated_embedding=embedding
                    )
                    
                    if search_results:
                        # Find the original filter dict
                        original_filter = next((f for f in property_filters if f['node_type'] == node_type and f['property_name'] == property_name), None)
                        if not original_filter:
                            continue
                        
                        # Build unique parameter base name
                        param_base = f"prop_value_{param_counter}"
                        param_counter += 1
                        
                        # Include BOTH original LLM value AND vector-matched values for best recall
                        # Original value comes first, followed by top vector matches
                        all_property_values = [llm_value] + [result['property_value'] for result in search_results]
                        
                        # Replace original condition with enhanced OR condition (now parameterized)
                        enhanced_condition, condition_params = self._build_multi_value_condition(
                            node_alias=original_filter['node_alias'],
                            property_name=original_filter['property_name'],
                            property_values=all_property_values,
                            original_operator=original_filter['operator'],
                            param_base_name=param_base
                        )
                        
                        # Merge parameters
                        all_parameters.update(condition_params)
                        
                        # Replace in query
                        original_condition = original_filter['original_condition']
                        enhanced_query = enhanced_query.replace(original_condition, enhanced_condition)
                        
                        logger.info(f"🚀 PROPERTY ENHANCEMENT: Enhanced {node_type}.{property_name} with {len(all_property_values)} values (1 original + {len(search_results)} vector-matched)")
                    
                except Exception as e:
                    logger.warning(f"🚀 PROPERTY ENHANCEMENT: Search failed for {node_type}.{property_name}: {e}")
            
            return enhanced_query, all_parameters
            
        except Exception as e:
            logger.error(f"🚀 PROPERTY ENHANCEMENT ERROR: {e}")
            return cypher_query, {}  # Return original query with empty params on error

    def _parse_property_filters_from_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Extract property conditions from Cypher WHERE clauses"""
        
        property_filters = []
        
        try:
            # Find WHERE clauses
            where_pattern = r'WHERE\s+(.+?)(?=\s+RETURN|\s+ORDER|\s+LIMIT|$)'
            where_matches = re.findall(where_pattern, cypher_query, re.IGNORECASE | re.DOTALL)
            
            for where_clause in where_matches:
                # Find property conditions like "n.name CONTAINS 'value'"
                prop_pattern = r'(\w+)\.(\w+)\s+(CONTAINS|=|STARTS WITH|ENDS WITH)\s+[\'"]([^\'"]+)[\'"]'
                prop_matches = re.findall(prop_pattern, where_clause, re.IGNORECASE)
                
                for match in prop_matches:
                    node_alias, property_name, operator, value = match
                    
                    # Try to determine node type from alias
                    node_type = self._extract_node_type_from_alias(cypher_query, node_alias)
                    
                    if node_type:
                        property_filters.append({
                            'node_alias': node_alias,
                            'node_type': node_type,
                            'property_name': property_name,
                            'operator': operator,
                            'value': value,
                            'original_condition': f"{node_alias}.{property_name} {operator} '{value}'"
                        })
            
            logger.info(f"🔧 CYPHER PARSER: Extracted {len(property_filters)} property filters")
            return property_filters
            
        except Exception as e:
            logger.error(f"🔧 CYPHER PARSER ERROR: {e}")
            return []

    def _extract_node_type_from_alias(self, cypher_query: str, node_alias: str) -> Optional[str]:
        """Extract node type from MATCH clause using alias"""
        
        try:
            # Find MATCH clauses with the alias
            match_pattern = rf'MATCH\s+.*\({node_alias}:(\w+)\)'
            match = re.search(match_pattern, cypher_query, re.IGNORECASE)
            
            if match:
                return match.group(1)
                
            return None
            
        except Exception as e:
            logger.error(f"🔧 NODE TYPE EXTRACTION ERROR: {e}")
            return None

    def _determine_property_schema(
        self, 
        node_type: str, 
        property_name: str, 
        enhanced_schema_cache: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Determine which schema a property belongs to for filtering"""
        
        try:
            indexable_properties = enhanced_schema_cache.get('indexable_properties', {})
            prop_key = f"{node_type}.{property_name}"
            
            if prop_key in indexable_properties:
                # Return first schema context (could be enhanced to handle multiple)
                schema_infos = indexable_properties[prop_key]
                if schema_infos:
                    return schema_infos[0]  # Use first schema for now
                    
            return None
            
        except Exception as e:
            logger.error(f"🔧 SCHEMA DETERMINATION ERROR: {e}")
            return None

    async def _generate_batch_property_embeddings(self, texts: List[str], memory_graph: "MemoryGraph") -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple property search texts using HuggingFace API batch processing.
        
        Args:
            texts: List of formatted property strings (e.g., "Node: Project, Property: name: Papr")
            memory_graph: MemoryGraph instance for accessing embedding model
            
        Returns:
            List of embeddings (or None for failed texts)
        """
        try:
            import httpx
            from os import environ as env
            
            api_url = env.get("HUGGING_FACE_API_URL_SENTENCE_BERT")
            access_token = env.get("HUGGING_FACE_ACCESS_TOKEN")
            
            if not api_url or not access_token:
                logger.error("HuggingFace API URL or token not configured for property search")
                return [None] * len(texts)
            
            headers = {"Authorization": f"Bearer {access_token}"}
            payload = {"inputs": texts}  # Send array of texts for batch processing
            
            async with httpx.AsyncClient(timeout=10.0) as client:  # Fast timeout for search
                response = await client.post(api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    embeddings = response.json()
                    logger.info(f"🚀 PROPERTY SEARCH: ✅ Generated {len(embeddings)} embeddings in batch")
                    return embeddings
                else:
                    logger.error(f"🚀 PROPERTY SEARCH: Batch embedding API error: {response.status_code} - {response.text}")
                    return [None] * len(texts)
                    
        except Exception as e:
            logger.error(f"🚀 PROPERTY SEARCH: Batch embedding error: {e}")
            return [None] * len(texts)
    
    async def _search_property_values(
        self,
        node_type: str,
        property_name: str,
        llm_suggested_value: str,
        acl_filter: Dict[str, Any],
        schema_context: Dict[str, Any],
        memory_graph: "MemoryGraph",
        top_k: int = 5,
        pre_generated_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Search Qdrant property collection for similar values using existing MemoryGraph instance"""
        
        try:
            from qdrant_client import models
            
            logger.warning(f"🚀 PROPERTY SEARCH: Starting search for {node_type}.{property_name} with value '{llm_suggested_value}'")
            logger.warning(f"🚀 PROPERTY SEARCH: Using existing MemoryGraph instance (reused from app state)")
            
            if not memory_graph.qdrant_client:
                logger.warning(f"🚀 PROPERTY SEARCH: Qdrant client not initialized")
                return []
                
            if not memory_graph.qdrant_property_collection:
                logger.warning(f"🚀 PROPERTY SEARCH: Property collection not initialized")
                return []
            
            logger.warning(f"🚀 PROPERTY SEARCH: Using collection '{memory_graph.qdrant_property_collection}'")
            
            # Use pre-generated embedding if provided (batch optimization), otherwise generate it
            if pre_generated_embedding is not None:
                embedding = pre_generated_embedding
                logger.warning(f"🚀 PROPERTY SEARCH: Using pre-generated embedding (batch optimization)")
            else:
                # Format search query to match indexed content format: "Node: X, Property: Y: Z"
                search_content = f"Node: {node_type}, Property: {property_name}: {llm_suggested_value}"
                logger.warning(f"🚀 PROPERTY SEARCH: Formatted search content: '{search_content}'")
                
                # Generate embedding using HuggingFace API (consistent with regular memories and property indexing)
                # If this fails (timeout, API error, etc.), we gracefully return [] and the search
                # continues with the original LLM-suggested property values without vector enhancement
                # Use max_retries=1 for search to avoid keeping users waiting (vs. 3 retries for indexing)
                try:
                    embeddings_result, _ = await memory_graph.embedding_model.get_sentence_embedding(
                        search_content, 
                        max_retries=1  # Fast failure for search - don't make users wait
                    )
                    if embeddings_result and len(embeddings_result) > 0:
                        embedding = embeddings_result[0]
                    else:
                        logger.warning(f"🚀 PROPERTY SEARCH: Failed to generate embedding, will use original LLM value without enhancement")
                        return []
                except Exception as e:
                    logger.warning(f"🚀 PROPERTY SEARCH: Embedding generation error (falling back to LLM value): {e}")
                    return []
            
            logger.warning(f"🚀 PROPERTY SEARCH: Generated/using embedding with dimension {len(embedding)}")
            
            # Build search filters using same metadata structure as property indexing
            # Use property_key which combines node_type.property_name (e.g., "Control.id")
            property_key = f"{node_type}.{property_name}"
            
            # Property key filter (must match)
            must_conditions = [
                models.FieldCondition(
                    key="property_key",
                    match=models.MatchValue(value=property_key)
                )
            ]
            
            # Add schema filter to must conditions if available
            if schema_context.get('schema_id'):
                must_conditions.append(
                    models.FieldCondition(
                        key="schema_id",
                        match=models.MatchValue(value=schema_context['schema_id'])
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added schema_id filter: {schema_context['schema_id']}")
            
            logger.warning(f"🚀 PROPERTY SEARCH: Using property_key filter: '{property_key}'")
            
            # ACL filters: user has access if ANY condition matches (OR logic)
            # For retrieval/search, user_id should be in ACL conditions (OR), not must (AND)
            user_id = acl_filter.get('user_id')
            user_read_access = acl_filter.get('user_read_access', [])
            workspace_read_access = acl_filter.get('workspace_read_access', [])
            organization_read_access = acl_filter.get('organization_read_access', [])
            namespace_read_access = acl_filter.get('namespace_read_access', [])
            role_read_access = acl_filter.get('role_read_access', [])
            
            acl_conditions = []
            
            # Add user_id as an ACL condition (OR with other access conditions)
            if user_id:
                acl_conditions.append(
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added user_id to ACL conditions: {user_id}")
            
            if user_read_access:
                acl_conditions.append(
                    models.FieldCondition(
                        key="user_read_access",
                        match=models.MatchAny(any=user_read_access)
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added user_read_access filter: {user_read_access}")
                
            if workspace_read_access:
                acl_conditions.append(
                    models.FieldCondition(
                        key="workspace_read_access",
                        match=models.MatchAny(any=workspace_read_access)
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added workspace_read_access filter: {workspace_read_access}")
                
            if organization_read_access:
                acl_conditions.append(
                    models.FieldCondition(
                        key="organization_read_access",
                        match=models.MatchAny(any=organization_read_access)
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added organization_read_access filter: {organization_read_access}")
                
            if namespace_read_access:
                acl_conditions.append(
                    models.FieldCondition(
                        key="namespace_read_access",
                        match=models.MatchAny(any=namespace_read_access)
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added namespace_read_access filter: {namespace_read_access}")
                
            if role_read_access:
                acl_conditions.append(
                    models.FieldCondition(
                        key="role_read_access",
                        match=models.MatchAny(any=role_read_access)
                    )
                )
                logger.warning(f"🚀 PROPERTY SEARCH: Added role_read_access filter: {role_read_access}")
            
            # Combine: property_key MUST match AND (user has access via ANY ACL condition)
            search_filter = models.Filter(
                must=must_conditions,
                should=acl_conditions  # OR: user has access if ANY ACL condition matches
            )
            
            logger.warning(f"🚀 PROPERTY SEARCH: Executing search with {len(must_conditions)} must conditions and {len(acl_conditions)} ACL conditions (OR)")
            
            # Handle embedding as list (HuggingFace API) or numpy array (local models)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
            # Search Qdrant property collection
            search_results = await memory_graph.qdrant_client.search(
                collection_name=memory_graph.qdrant_property_collection,
                query_vector=embedding_list,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=0.5,  # Lower threshold for partial matches (only top 5 results)
                with_payload=True,
                with_vectors=False
            )
            
            logger.warning(f"🚀 PROPERTY SEARCH: Found {len(search_results)} raw results")
            
            # Process results
            property_matches = []
            for i, result in enumerate(search_results):
                property_value = result.payload.get('property_value', '')
                similarity_score = result.score
                source_node_id = result.payload.get('source_node_id', '')
                schema_name = result.payload.get('schema_name', 'system')
                
                logger.warning(f"🚀 PROPERTY SEARCH: Result {i+1}: value='{property_value}', score={similarity_score:.3f}, node_id='{source_node_id}'")
                
                property_matches.append({
                    'property_value': property_value,
                    'similarity_score': similarity_score,
                    'source_node_id': source_node_id,
                    'schema_name': schema_name,
                    'original_llm_value': llm_suggested_value
                })
            
            logger.warning(f"🚀 PROPERTY SEARCH: Returning {len(property_matches)} processed matches for {node_type}.{property_name}")
            return property_matches
            
        except Exception as e:
            logger.error(f"🚀 PROPERTY SEARCH ERROR: {e}")
            import traceback
            logger.error(f"🚀 PROPERTY SEARCH TRACEBACK: {traceback.format_exc()}")
            return []

    async def _get_node_count(self, node_type: str, acl_filter: Dict[str, Any], neo_session: AsyncSession) -> int:
        """Get count of nodes of a specific type for threshold check"""
        
        try:
            # Simple count query with ACL
            count_query = f"""
            MATCH (n:{node_type})
            WHERE n.user_read_access IN $user_read_access
            RETURN count(n) as node_count
            """
            
            result = await neo_session.run(
                count_query,
                parameters={
                    'user_read_access': acl_filter.get('user_read_access', [])
                }
            )
            
            # Get the first record
            record = await result.single()
            if record:
                return record.get('node_count', 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"🚀 NODE COUNT ERROR: {e}")
            return 0

    def _build_multi_value_condition(
        self,
        node_alias: str,
        property_name: str,
        property_values: List[str],
        original_operator: str = "CONTAINS",
        param_base_name: str = "prop_value"
    ) -> tuple[str, Dict[str, str]]:
        """Build Cypher OR condition for multiple property values with parameterized queries
        
        Returns:
            tuple: (condition_string, parameters_dict)
        """
        
        if not property_values:
            return f"{node_alias}.{property_name} {original_operator} ''", {}
            
        def truncate_by_operator(text: str, operator: str, max_words: int = 5) -> str:
            """Truncate text based on the operator type"""
            words = text.strip().split()
            if not words:
                return text
                
            # For text-based operators that benefit from truncation
            if operator in ["CONTAINS", "STARTS WITH"]:
                # Take first N words
                return ' '.join(words[:max_words])
            elif operator == "ENDS WITH":
                # Take last N words
                return ' '.join(words[-max_words:])
            elif operator in ["=", "<>", "IN", "NOT IN"]:
                # For exact matching, don't truncate - use full value or reasonable limit
                if len(text) > 200:  # Prevent extremely long values
                    return text[:197] + "..."
                return text
            elif operator in [">", ">=", "<", "<=", "=~"]:
                # For comparison/regex operators, use full value
                return text
            elif operator in ["IS NULL", "IS NOT NULL"]:
                # These don't use values, but return something safe
                return ""
            else:
                # Default: use first N words for unknown operators
                return ' '.join(words[:max_words])
            
        # Collect parameters for parameterized query
        parameters = {}
        
        if len(property_values) == 1:
            truncated_value = truncate_by_operator(property_values[0], original_operator)
            if original_operator in ["IS NULL", "IS NOT NULL"]:
                # These operators don't use values
                return f"{node_alias}.{property_name} {original_operator}", {}
            # Use parameterized query
            param_name = f"{param_base_name}_0"
            parameters[param_name] = truncated_value
            return f"{node_alias}.{property_name} {original_operator} ${param_name}", parameters
            
        # Build OR condition for multiple values with parameters
        conditions = []
        for idx, value in enumerate(property_values[:5]):  # Limit to top 5 for performance
            if original_operator in ["IS NULL", "IS NOT NULL"]:
                # These operators don't use values, just add once
                conditions.append(f"{node_alias}.{property_name} {original_operator}")
                break  # Only need one condition for null checks
            else:
                truncated_value = truncate_by_operator(value, original_operator)
                if truncated_value:  # Only add non-empty values
                    param_name = f"{param_base_name}_{idx}"
                    parameters[param_name] = truncated_value
                    conditions.append(f"{node_alias}.{property_name} {original_operator} ${param_name}")
            
        result = f"({' OR '.join(conditions)})" if conditions else f"{node_alias}.{property_name} {original_operator} ''"
        return result, parameters
        return f"({' OR '.join(conditions)})" if conditions else f"{node_alias}.{property_name} {original_operator} ''"