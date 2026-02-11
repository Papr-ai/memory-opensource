#from transformers import BigBirdModel, BigBirdTokenizer
import requests
# Ensure SSL cert env vars are set BEFORE importing clients that create HTTPX/SSL contexts
import os
import certifi
from pathlib import Path

# Only set SSL_CERT_FILE if the file exists and is readable
# This prevents httpx from failing when the file doesn't exist
# httpx reads SSL_CERT_FILE directly from env and doesn't check if file exists
cert_path = certifi.where()
cert_file = Path(cert_path)
if cert_file.exists() and cert_file.is_file() and os.access(cert_path, os.R_OK):
    os.environ.setdefault('REQUESTS_CA_BUNDLE', cert_path)
    os.environ.setdefault('SSL_CERT_FILE', cert_path)
else:
    # If certifi path doesn't exist or isn't readable, unset SSL_CERT_FILE
    # This is critical - httpx will fail if SSL_CERT_FILE points to non-existent file
    # This is important for Docker containers where certifi path might differ
    if 'SSL_CERT_FILE' in os.environ:
        del os.environ['SSL_CERT_FILE']
    if 'REQUESTS_CA_BUNDLE' in os.environ:
        del os.environ['REQUESTS_CA_BUNDLE']

#from sentence_transformers import SentenceTransformer
from os import environ as env
from dotenv import find_dotenv, load_dotenv
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
import httpx
from langchain_community.embeddings import OllamaEmbeddings
import ollama 
from sentence_transformers import SentenceTransformer
from transformers import BigBirdModel, BigBirdTokenizer
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer
from transformers import AutoConfig
import asyncio
import time
from requests.exceptions import RequestException
from typing import Tuple, List, Optional
from services.logging_config import get_logger
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Import Vertex AI at module level to avoid slow import on first use
try:
    from google.cloud import aiplatform
    VERTEX_AI_AVAILABLE = True
except ImportError:
    aiplatform = None
    VERTEX_AI_AVAILABLE = False
# Create a logger instance for this module
logger = get_logger(__name__)  # Will use 'models.embedding_model' as the logger name


import  torch
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

def _set_ssl_cert_paths():
    """Safely set SSL certificate paths only if the file exists and is readable.
    This prevents httpx from failing when SSL_CERT_FILE points to a non-existent file.
    httpx reads SSL_CERT_FILE directly from env and doesn't check if file exists.
    Works for both cloud and open source editions."""
    cert_path = certifi.where()
    cert_file = Path(cert_path)
    if cert_file.exists() and cert_file.is_file() and os.access(cert_path, os.R_OK):
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['SSL_CERT_FILE'] = cert_path
    else:
        # Unset if file doesn't exist or isn't readable - let system/certifi handle it automatically
        # This is critical - httpx will fail if SSL_CERT_FILE points to non-existent file
        # This is important for Docker containers where certifi path might differ
        if 'SSL_CERT_FILE' in os.environ:
            del os.environ['SSL_CERT_FILE']
        if 'REQUESTS_CA_BUNDLE' in os.environ:
            del os.environ['REQUESTS_CA_BUNDLE']

# Retrieve environment variables
hugging_face_api_url_sentence_bert = env.get("HUGGING_FACE_API_URL_SENTENCE_BERT")
hugging_face_api_url_big_bird = env.get("HUGGING_FACE_API_URL_BIG_BIRD")
hugging_face_access_token = env.get("HUGGING_FACE_ACCESS_TOKEN")
hugging_face_api_url_qwen_4b = env.get("HUGGING_FACE_API_URL_QWEN_4B")
deepinfra_api_url = env.get("DEEPINFRA_API_URL")  
deepinfra_token = env.get("DEEPINFRA_TOKEN")
# Vertex AI configuration for custom trained model (Qwen4B embeddings)
vertex_ai_project = env.get("VERTEX_AI_PROJECT", "223473570766")
vertex_ai_endpoint_id = env.get("VERTEX_AI_ENDPOINT_ID", "8078932164944592896")
vertex_ai_location = env.get("VERTEX_AI_LOCATION", "us-west1")
use_vertex_ai = env.get("USE_VERTEX_AI", "false").lower() == "true"
# Local embedding configuration
use_local_embeddings = env.get("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
local_embedding_model = env.get("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
local_embedding_dimensions = int(env.get("LOCAL_EMBEDDING_DIMENSIONS", "1024"))
MAX_EMBEDDING_PAYLOAD_SIZE = int(env.get("MAX_EMBEDDING_PAYLOAD_SIZE", "2000000"))
MAX_EMBEDDING_BATCH_TOKENS = int(env.get("MAX_EMBEDDING_BATCH_TOKENS", "16384"))
MAX_EMBEDDING_CLIENT_BATCH_SIZE = int(env.get("MAX_EMBEDDING_CLIENT_BATCH_SIZE", "32"))

class EmbeddingModel:
    # Set certificate paths at class level before any HuggingFace operations
    # Only set if file exists and is readable to avoid httpx errors
    _set_ssl_cert_paths()
    cert_path = certifi.where()
    cert_file = Path(cert_path)
    if cert_file.exists() and cert_file.is_file() and os.access(cert_path, os.R_OK):
        logger.info(f"Set certificate paths to: {cert_path}")
    else:
        logger.info(f"SSL certificate file not found or not readable at {cert_path}, using system defaults")

    _sentence_model_instance = None
    _bigbird_model_instance = None
    _bigbird_tokenizer_instance = None
    _sentence_bert_tokenizer = None
    _sentence_bert_config = None
    _bigbird_tokenizer = None
    _bigbird_config = None
    _qwen4b_tokenizer = None
    _qwen4b_config = None
    _qwen4b_splitter = None
    _qwen4b_model_instance = None
    _qwen0pt6b_model_instance = None  # Local Qwen3-Embedding-0.6B model
    _vertex_ai_initialized = False  # Class-level flag to track Vertex AI initialization
    _vertex_ai_endpoint = None  # Class-level cached endpoint object for connection pooling
    def __init__(self):
        if env.get("LOCALPROCESSING"):
            logger.info("Applying local processing  ")  
            if  EmbeddingModel._bigbird_model_instance is None:     
                EmbeddingModel._bigbird_model_instance = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
                self.bigbird_model_instance =  EmbeddingModel._bigbird_model_instance
                EmbeddingModel._bigbird_tokenizer_instance = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
                self.bigbird_tokenizer_instance = EmbeddingModel._bigbird_tokenizer_instance
            if  EmbeddingModel._sentence_model_instance is None:
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                EmbeddingModel._sentence_model_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device = "cpu")
                self.sentence_model_instance = EmbeddingModel._sentence_model_instance
            # Qwen4B
            
            #self.client = Ollama(model=model)
            #self.embedding_model =env.get("EMBEDDING_MODEL_LOCAL") if env.get("EMBEDDING_MODEL_LOCAL") else "text-embedding-3-small"
           

        #if EmbeddingModel._bigbird_model_instance is None:
        #    EmbeddingModel._bigbird_model_instance = BigBirdModel.from_pretrained(
        #        bigbird_model_name
        #    )
        #    EmbeddingModel._bigbird_tokenizer_instance = BigBirdTokenizer.from_pretrained(bigbird_model_name)
        #self.bigbird_model = EmbeddingModel._bigbird_model_instance
        #self.bigbird_tokenizer = EmbeddingModel._bigbird_tokenizer_instance
        #pass

        # Initialize tokenizers if not already initialized
        if EmbeddingModel._sentence_bert_tokenizer is None:
            logger.info("Initializing BERT tokenizer...")
            # Set certificate path for HuggingFace
            _set_ssl_cert_paths()
            
            hugging_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            EmbeddingModel._sentence_bert_tokenizer = AutoTokenizer.from_pretrained(hugging_model_name)
            EmbeddingModel._sentence_bert_config = AutoConfig.from_pretrained(hugging_model_name)
            logger.info("BERT tokenizer initialized")

            # Initialize text splitter with the tokenizer
            num_special_tokens = EmbeddingModel._sentence_bert_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._sentence_bert_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._sentence_bert_splitter = TokenTextSplitter(
                chunk_size=max_token_limit, 
                chunk_overlap=0
            )
            logger.info("BERT tokenizer and text splitter initialized")

        if EmbeddingModel._bigbird_tokenizer is None:
            logger.info("Initializing BigBird tokenizer...")
            # Set certificate path for HuggingFace
            _set_ssl_cert_paths()
            
            model_bigbird = "google/bigbird-roberta-base"
            EmbeddingModel._bigbird_tokenizer = AutoTokenizer.from_pretrained(model_bigbird)
            EmbeddingModel._bigbird_config = AutoConfig.from_pretrained(model_bigbird)
            logger.info("BigBird tokenizer initialized")

            # Initialize BigBird text splitter
            num_special_tokens = EmbeddingModel._bigbird_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._bigbird_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._bigbird_splitter = TokenTextSplitter(
                chunk_size=max_token_limit, 
                chunk_overlap=0
            )
            logger.info("BigBird tokenizer and text splitter initialized")
        if EmbeddingModel._qwen4b_tokenizer is None:
            logger.info("Initializing Qwen3-Embedding-4B tokenizer...")
            _set_ssl_cert_paths()
            qwen_model_name = "Qwen/Qwen3-Embedding-4B"
            EmbeddingModel._qwen4b_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
            EmbeddingModel._qwen4b_config = AutoConfig.from_pretrained(qwen_model_name)
            
            # Initialize Qwen4B text splitter
            num_special_tokens = EmbeddingModel._qwen4b_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._qwen4b_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._qwen4b_splitter = TokenTextSplitter(
                chunk_size=max_token_limit,
                chunk_overlap=0
            )
            logger.info("Qwen3-Embedding-4B tokenizer, config and text splitter initialized")
        
        # Initialize local Qwen3-Embedding-0.6B model if USE_LOCAL_EMBEDDINGS is enabled
        if use_local_embeddings and EmbeddingModel._qwen0pt6b_model_instance is None:
            logger.info(f"Initializing local embedding model: {local_embedding_model}")
            logger.info("This may take a few minutes on first run (~1.2GB download)...")
            try:
                _set_ssl_cert_paths()
                # Use SentenceTransformer for easy local embedding generation
                EmbeddingModel._qwen0pt6b_model_instance = SentenceTransformer(
                    local_embedding_model,
                    device="cpu",  # Use CPU by default, GPU if available
                    trust_remote_code=True  # Required for Qwen models
                )
                # Enable GPU if available
                if torch.cuda.is_available():
                    EmbeddingModel._qwen0pt6b_model_instance = EmbeddingModel._qwen0pt6b_model_instance.to("cuda")
                    logger.info(f"Local embedding model loaded on GPU: {local_embedding_model}")
                else:
                    logger.info(f"Local embedding model loaded on CPU: {local_embedding_model}")
                self.qwen0pt6b_model_instance = EmbeddingModel._qwen0pt6b_model_instance
            except Exception as e:
                logger.error(f"Failed to initialize local embedding model {local_embedding_model}: {e}")
                logger.warning("Falling back to cloud embeddings. Set USE_LOCAL_EMBEDDINGS=false to suppress this warning.")
                # Don't raise - allow fallback to cloud embeddings
    #def get_sentence_embedding(self, text):
    #    embedding = self.sentence_model.encode([text])
    #    return embedding[0]
    def call_huggingface_api(input_ids):
    # Replace 'your-api-url' and 'your-api-token' with your actual API URL and token
        api_url = hugging_face_api_url_sentence_bert
        headers = {"Authorization": f"Bearer {hugging_face_access_token}"}
        payload = {"inputs": input_ids}

        response = requests.post(api_url, headers=headers, json=payload)
        return response.json()
    # Replace the local model call with a Hugging Face API call


    async def get_sentence_embedding(self, text: str, max_retries: int = 3, retry_delay: float = 1.0, semaphore: Optional[asyncio.Semaphore] = None, use_async: bool = True) -> Tuple[List[List[float]], List[str]]:
        """
        Get sentence embeddings for the given text asynchronously, with retry logic and concurrency control.
        
        Args:
            text (str): The text to embed
            max_retries (int): Maximum number of retries for each chunk (default: 3)
            retry_delay (float): Initial delay between retries in seconds (default: 1.0)
            semaphore (Optional[asyncio.Semaphore]): Semaphore to limit concurrent chunk requests (default: None, uses 10)
            use_async (bool): If False or only one chunk, process serially (no asyncio.gather). Default True.
        
        Returns:
            Tuple[List[List[float]], List[str]]: A tuple containing:
                - List of embeddings where each embedding is a list of floats
                - List of text chunks
        """
        overall_start_time = time.time()
        logger.warning("[TIMING] Starting get_sentence_embedding")

        if env.get("LOCALPROCESSING"):
            logger.info("Local processing is enabled for sentence embedding")
            loop = asyncio.get_event_loop()
            try:
                local_start = time.time()
                embeddinglocal = await loop.run_in_executor(
                    None,
                    lambda: EmbeddingModel._sentence_model_instance.encode(text)
                )
                local_end = time.time()
                logger.warning(f"[TIMING] Local embedding took: {local_end - local_start:.4f} seconds")
                logger.warning("[TIMING] Finished get_sentence_embedding (local)")
                return embeddinglocal.tolist(), [text]
            except Exception as e:
                logger.error(f"Error in local sentence embedding: {str(e)}")
                raise

        # Initialize tokenizers if not already initialized
        tokenizer_start = time.time()
        if EmbeddingModel._sentence_bert_tokenizer is None:
            logger.info("Initializing BERT tokenizer...")
            # Set certificate path for HuggingFace
            _set_ssl_cert_paths()
            
            hugging_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            EmbeddingModel._sentence_bert_tokenizer = AutoTokenizer.from_pretrained(hugging_model_name)
            EmbeddingModel._sentence_bert_config = AutoConfig.from_pretrained(hugging_model_name)
            
            # Initialize text splitter with the tokenizer
            num_special_tokens = EmbeddingModel._sentence_bert_tokenizer.num_special_tokens_to_add()
            max_token_limit = EmbeddingModel._sentence_bert_config.max_position_embeddings - num_special_tokens - 30
            EmbeddingModel._sentence_bert_splitter = TokenTextSplitter(
                chunk_size=max_token_limit, 
                chunk_overlap=0
            )
            logger.info("BERT tokenizer and text splitter initialized")
        tokenizer_end = time.time()
        logger.warning(f"[TIMING] Tokenizer init/check took: {tokenizer_end - tokenizer_start:.4f} seconds")

        # Use pre-initialized splitter
        chunking_start = time.time()
        chunks = EmbeddingModel._sentence_bert_splitter.split_text(text)
        logger.info(f"chunk size: {len(chunks)}")
        chunking_end = time.time()
        logger.warning(f"[TIMING] Split text into {len(chunks)} chunks in {chunking_end - chunking_start:.4f} seconds")

        # Use provided semaphore or create a default one
        if semaphore is None:
            semaphore = asyncio.Semaphore(10)

        def get_payload_size(payload):
            import json
            return len(json.dumps(payload).encode('utf-8'))

        async def process_chunk_with_retry(chunk: str, i: int) -> Optional[List[float]]:
            last_exception = None
            async with semaphore:
                for attempt in range(1, max_retries + 1):
                    attempt_start = time.time()
                    # Tokenization timing
                    token_start = time.time()
                    tokenized_chunk = EmbeddingModel._sentence_bert_tokenizer(
                        chunk,
                        truncation=True,
                        max_length=EmbeddingModel._sentence_bert_config.max_position_embeddings,
                        return_tensors="pt"
                    )
                    token_end = time.time()
                    logger.info(f"[TIMING] [Chunk {i}] Tokenization took: {token_end - token_start:.4f} seconds (attempt {attempt})")

                    num_tokens = len(tokenized_chunk['input_ids'][0])
                    if num_tokens > EmbeddingModel._sentence_bert_config.max_position_embeddings:
                        logger.warning(f"Chunk {i} exceeds max token size: {num_tokens} tokens")
                        return None
                    if num_tokens > MAX_EMBEDDING_BATCH_TOKENS:
                        logger.warning(f"Chunk {i} exceeds configured MAX_EMBEDDING_BATCH_TOKENS: {num_tokens} > {MAX_EMBEDDING_BATCH_TOKENS}")
                        return None

                    # Payload prep timing
                    payload_start = time.time()
                    payload = {
                        "inputs": chunk,
                        "parameters": {"truncation": True}
                    }
                    payload_size = get_payload_size(payload)
                    logger.info(f"[Chunk {i}] Payload size: {payload_size} bytes (limit: {MAX_EMBEDDING_PAYLOAD_SIZE})")
                    if payload_size > MAX_EMBEDDING_PAYLOAD_SIZE:
                        logger.error(f"[Chunk {i}] Payload size {payload_size} exceeds limit of {MAX_EMBEDDING_PAYLOAD_SIZE} bytes. Skipping.")
                        return None
                    payload_end = time.time()
                    logger.info(f"[TIMING] [Chunk {i}] Payload prep took: {payload_end - payload_start:.4f} seconds (attempt {attempt})")

                    # FIX: Define headers here
                    headers = {
                        "Authorization": f"Bearer {hugging_face_access_token}",
                        "Content-Type": "application/json"
                    }

                    try:
                        http_start = time.time()
                        async with httpx.AsyncClient(verify=False) as client:
                            response = await client.post(
                                hugging_face_api_url_sentence_bert,
                                headers=headers,
                                json=payload,
                                timeout=5.0
                            )
                        http_end = time.time()
                        logger.info(f"[TIMING] [Chunk {i}] HTTP request took: {http_end - http_start:.4f} seconds (attempt {attempt})")

                        if response.status_code == 400:
                            error_text = response.text
                            logger.error(f"Bad Request: {error_text}")
                            return None
                        if response.status_code == 413:
                            logger.error(f"[Chunk {i}] Received 413 Payload Too Large from embedding API. Payload size: {payload_size} bytes.")
                            return None
                        
                        parse_start = time.time()
                        response.raise_for_status()
                        response_json = response.json()
                        parse_end = time.time()
                        logger.info(f"[TIMING] [Chunk {i}] Response parsing took: {parse_end - parse_start:.4f} seconds (attempt {attempt})")

                        if isinstance(response_json, list) and isinstance(response_json[0], list):
                            embedding = response_json[0]
                            logger.info(f"Chunk {i} embedding dimensions: {len(embedding)}")
                            attempt_end = time.time()
                            logger.info(f"[TIMING] [Chunk {i}] Total attempt time: {attempt_end - attempt_start:.4f} seconds (attempt {attempt})")
                            return embedding
                        else:
                            logger.error(f"Unexpected API response format: {response_json}")
                            attempt_end = time.time()
                            logger.info(f"[TIMING] [Chunk {i}] Unexpected format, total attempt time: {attempt_end - attempt_start:.4f} seconds (attempt {attempt})")
                            return None
                    except httpx.HTTPError as e:
                        logger.error(f"HTTP error in async API request (attempt {attempt}/{max_retries}): {str(e)}")
                        last_exception = e
                    except Exception as e:
                        logger.error(f"Unexpected error in async API request (attempt {attempt}/{max_retries}): {str(e)}")
                        last_exception = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** (attempt - 1))
                        logger.warning(f"Retrying chunk {i} after {delay:.2f}s (attempt {attempt}/{max_retries})")
                        sleep_start = time.time()
                        await asyncio.sleep(delay)
                        sleep_end = time.time()
                        logger.info(f"[TIMING] [Chunk {i}] Slept for: {sleep_end - sleep_start:.4f} seconds before retry (attempt {attempt})")
                logger.error(f"All {max_retries} attempts failed for chunk {i}")
                if last_exception:
                    raise last_exception
                return None

        # Process all chunks concurrently with retry logic and concurrency control
        embedding_start = time.time()
        if not use_async or len(chunks) == 1:
            # Serial processing

            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                emb = await process_chunk_with_retry(chunk, i)
                chunk_embeddings.append(emb)
        else:
            chunk_embeddings = await asyncio.gather(
                *[process_chunk_with_retry(chunk, i) for i, chunk in enumerate(chunks)]
            )
        embedding_end = time.time()
        logger.info(f"[TIMING] All chunk embedding requests took: {embedding_end - embedding_start:.4f} seconds")

        # Filter out None values
        filter_start = time.time()
        embeddings = [emb for emb in chunk_embeddings if emb is not None]
        filter_end = time.time()
        logger.info(f"[TIMING] Filtering embeddings took: {filter_end - filter_start:.4f} seconds")

        if not embeddings:
            logger.warning("No embeddings were generated. Returning empty list.")
            logger.warning(f"[TIMING] Finished get_sentence_embedding (no embeddings)")
            return [], chunks

        total_time = time.time() - overall_start_time
        logger.info(f"[TIMING] Total async BERT embedding generation took: {total_time:.4f} seconds")
        logger.info("[TIMING] Finished get_sentence_embedding")

        return embeddings, chunks
    
    async def get_bigbird_embedding(self, text: str, max_retries: int = 3, retry_delay: float = 1.0, semaphore: Optional[asyncio.Semaphore] = None, use_async: bool = True) -> Tuple[List[List[float]], List[str]]:
        """
        Get BigBird embeddings for the given text asynchronously, with retry logic and concurrency control.
        
        Args:
            text (str): The text to embed
            max_retries (int): Maximum number of retries for each chunk (default: 3)
            retry_delay (float): Initial delay between retries in seconds (default: 1.0)
            semaphore (Optional[asyncio.Semaphore]): Semaphore to limit concurrent chunk requests (default: None, uses 10)
            use_async (bool): If False or only one chunk, process serially (no asyncio.gather). Default True.
        
        Returns:
            Tuple[List[List[float]], List[str]]: A tuple containing:
                - List of embeddings where each embedding is a list of floats
                - List of text chunks
        """
        overall_start_time = time.time()
        logger.info("[TIMING] Starting get_bigbird_embedding")

        if env.get("LOCALPROCESSING"):
            logger.info("Big bird Local processing is enabled")
            loop = asyncio.get_event_loop()
            local_start = time.time()
            def process_locally():
                inputs = EmbeddingModel._bigbird_tokenizer_instance(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = EmbeddingModel._bigbird_model_instance(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return embeddings
            result = await loop.run_in_executor(None, process_locally)
            local_end = time.time()
            logger.info(f"[TIMING] Local BigBird embedding took: {local_end - local_start:.4f} seconds")
            logger.info("[TIMING] Finished get_bigbird_embedding (local)")
            return result, [text]

        # Tokenizer/config check
        tokenizer_start = time.time()
        if not EmbeddingModel._bigbird_tokenizer or not EmbeddingModel._bigbird_config:
            logger.error("BigBird tokenizer or config not initialized")
            raise ValueError("BigBird tokenizer or config not initialized")
        tokenizer_end = time.time()
        logger.info(f"[TIMING] BigBird tokenizer/config check took: {tokenizer_end - tokenizer_start:.4f} seconds")

        # Use the singleton BigBird text splitter
        chunking_start = time.time()
        chunks = EmbeddingModel._bigbird_splitter.split_text(text)
        chunking_end = time.time()
        logger.info(f"[TIMING] Split text into {len(chunks)} chunks (BigBird) in {chunking_end - chunking_start:.4f} seconds")

        # Use provided semaphore or create a default one
        if semaphore is None:
            semaphore = asyncio.Semaphore(10)

        async def process_chunk_with_retry(chunk: str, i: int) -> Optional[List[float]]:
            last_exception = None
            async with semaphore:
                for attempt in range(1, max_retries + 1):
                    attempt_start = time.time()
                    # Tokenization timing
                    token_start = time.time()
                    tokenized_chunk = EmbeddingModel._bigbird_tokenizer(
                        chunk, 
                        truncation=True, 
                        max_length=EmbeddingModel._bigbird_config.max_position_embeddings,
                        return_tensors="pt"
                    )
                    token_end = time.time()
                    logger.info(f"[TIMING] [BigBird Chunk {i}] Tokenization took: {token_end - token_start:.4f} seconds (attempt {attempt})")

                    num_tokens = len(tokenized_chunk['input_ids'][0])
                    if num_tokens > EmbeddingModel._bigbird_config.max_position_embeddings:
                        logger.warning(f"Chunk {i} exceeds max token size: {num_tokens} tokens (BigBird)")
                        return None
                    if num_tokens > MAX_EMBEDDING_BATCH_TOKENS:
                        logger.warning(f"BigBird Chunk {i} exceeds configured MAX_EMBEDDING_BATCH_TOKENS: {num_tokens} > {MAX_EMBEDDING_BATCH_TOKENS}")
                        return None

                    # Payload prep timing
                    payload_start = time.time()
                    payload = {
                        "inputs": chunk,
                        "parameters": {"truncation": True}
                    }
                    def get_payload_size(payload):
                        return len(json.dumps(payload).encode('utf-8'))
                    payload_size = get_payload_size(payload)
                    logger.info(f"[BigBird Chunk {i}] Payload size: {payload_size} bytes (limit: {MAX_EMBEDDING_PAYLOAD_SIZE})")
                    if payload_size > MAX_EMBEDDING_PAYLOAD_SIZE:
                        logger.error(f"[BigBird Chunk {i}] Payload size {payload_size} exceeds limit of {MAX_EMBEDDING_PAYLOAD_SIZE} bytes. Skipping.")
                        return None
                    payload_end = time.time()
                    logger.info(f"[TIMING] [BigBird Chunk {i}] Payload prep took: {payload_end - payload_start:.4f} seconds (attempt {attempt})")

                    headers = {
                        "Authorization": f"Bearer {hugging_face_access_token}",
                        "Content-Type": "application/json"
                    }
                    payload_end = time.time()
                    logger.info(f"[TIMING] [BigBird Chunk {i}] Payload prep took: {payload_end - payload_start:.4f} seconds (attempt {attempt})")

                    try:
                        http_start = time.time()
                        async with httpx.AsyncClient(verify=False) as client:
                            response = await client.post(
                                hugging_face_api_url_big_bird,
                                headers=headers,
                                json=payload,
                                timeout=30.0
                            )
                        http_end = time.time()
                        logger.info(f"[TIMING] [BigBird Chunk {i}] HTTP request took: {http_end - http_start:.4f} seconds (attempt {attempt})")
                        if response.status_code == 400:
                            error_text = response.text
                            logger.error(f"Bad Request: {error_text}")
                            return None
                        # Response parsing timing
                        parse_start = time.time()
                        response.raise_for_status()
                        response_json = response.json()
                        parse_end = time.time()
                        logger.info(f"[TIMING] [BigBird Chunk {i}] Response parsing took: {parse_end - parse_start:.4f} seconds (attempt {attempt})")
                        # Expecting a list of lists of lists (tokens x dims)
                        if isinstance(response_json, list) and all(isinstance(elem, list) for elem in response_json) and all(isinstance(sub_elem, list) for elem in response_json for sub_elem in elem):
                            flattened_embeddings = [item for sublist in response_json for item in sublist]
                            embedding = np.mean(np.array(flattened_embeddings), axis=0).tolist()
                            logger.info(f"BigBird Chunk {i} embedding dimensions: {len(embedding)}")
                            attempt_end = time.time()
                            logger.info(f"[TIMING] [BigBird Chunk {i}] Total attempt time: {attempt_end - attempt_start:.4f} seconds (attempt {attempt})")
                            return embedding
                        else:
                            logger.error("Unexpected API response format (BigBird).")
                            attempt_end = time.time()
                            logger.info(f"[TIMING] [BigBird Chunk {i}] Unexpected format, total attempt time: {attempt_end - attempt_start:.4f} seconds (attempt {attempt})")
                            return None
                    except httpx.HTTPError as e:
                        logger.error(f"HTTP error occurred (BigBird, attempt {attempt}/{max_retries}): {str(e)}")
                        last_exception = e
                    except Exception as e:
                        logger.error(f"Unexpected error occurred (BigBird, attempt {attempt}/{max_retries}): {str(e)}")
                        last_exception = e
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** (attempt - 1))
                        logger.warning(f"Retrying BigBird chunk {i} after {delay:.2f}s (attempt {attempt}/{max_retries})")
                        sleep_start = time.time()
                        await asyncio.sleep(delay)
                        sleep_end = time.time()
                        logger.info(f"[TIMING] [BigBird Chunk {i}] Slept for: {sleep_end - sleep_start:.4f} seconds before retry (attempt {attempt})")
                logger.error(f"All {max_retries} attempts failed for BigBird chunk {i}")
                if last_exception:
                    raise last_exception
                return None

        # Process all chunks concurrently with retry logic and concurrency control
        embedding_start = time.time()
        if not use_async or len(chunks) == 1:
            # Serial processing
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                emb = await process_chunk_with_retry(chunk, i)
                chunk_embeddings.append(emb)
        else:
            chunk_embeddings = await asyncio.gather(
                *[process_chunk_with_retry(chunk, i) for i, chunk in enumerate(chunks)]
            )
        embedding_end = time.time()
        logger.warning(f"[TIMING] All BigBird chunk embedding requests took: {embedding_end - embedding_start:.4f} seconds")

        # Filter out None values and empty lists
        filter_start = time.time()
        embeddings = [emb for emb in chunk_embeddings if emb is not None]
        filter_end = time.time()
        logger.warning(f"[TIMING] Filtering BigBird embeddings took: {filter_end - filter_start:.4f} seconds")

        if not embeddings:
            logger.warning("No BigBird embeddings were generated. Returning empty list.")
            logger.warning(f"[TIMING] Finished get_bigbird_embedding (no embeddings)")
            return [], chunks

        total_time = time.time() - overall_start_time
        logger.warning(f"[TIMING] Total async BigBird embedding generation took: {total_time:.4f} seconds")
        logger.warning("[TIMING] Finished get_bigbird_embedding")

        return embeddings, chunks

    def get_bigbird_embedding_hugging_face(self, text):
        
        if env.get("LOCALPROCESSING"):
            logger.info("Big bird Local processing is enabled")
            inputs = EmbeddingModel._bigbird_tokenizer_instance(text, return_tensors="pt", padding=True, truncation=True)
            # Get the embeddings from BigBird model
            with torch.no_grad():
                outputs = EmbeddingModel._bigbird_model_instance(**inputs)
                # Extract the last layer hidden states (embeddings)
                last_hidden_states = outputs.last_hidden_state.numpy()
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                flattened_embeddings = last_hidden_states.reshape(-1, last_hidden_states.shape[-2], last_hidden_states.shape[-1])
                #embeddingsflat = last_hidden_states.tolist()
                #flattened_embeddings = last_hidden_states.view(last_hidden_states.size(0), -1)
                #embeddings = np.mean(np.array(flattened_embeddings), axis=0)
         
            #return last_hidden_states[0]
            return embeddings
        
        headers = {
        "Authorization": f"Bearer {hugging_face_access_token}"
        }
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True},  # Ensure the request waits for the model if it's currently loading
            # Add any other parameters here, such as truncation or padding
        }
        
        response = requests.post(hugging_face_api_url_big_bird, headers=headers, json=payload)

        if response.status_code == 200:
            response_json = response.json()

            # Log the raw response for debugging
            #logger.info(f"BigBird hugging_face raw response: {response_json}")

            # Assuming the API returns an array of token embeddings, we average them
            # Verify this assumption based on the actual structure of response_json
            if isinstance(response_json, list) and all(isinstance(elem, list) for elem in response_json) and all(isinstance(sub_elem, list) for elem in response_json for sub_elem in elem):
                # Flatten the list of lists of lists to a list of lists
                flattened_embeddings = [item for sublist in response_json for item in sublist]
                # Calculate the mean across the first dimension (tokens) to get a single embedding vector
                embeddings = np.mean(np.array(flattened_embeddings), axis=0)
                logger.info(f"BigBird hugging_face processed embedding dimensions: {embeddings.shape}")
                return embeddings
            else:
                logger.error("Unexpected API response format.")
                raise ValueError("Unexpected API response format.")

            
        else:
            logger.error("Failed to get embeddings from Hugging Face API")
            response.raise_for_status()
    
    def get_embeddinglocal(self,text):
        response = ollama.embeddings(model=self.embedding_model, prompt=text)            
            #response = self.embeddingclient.embed_documents(text)
        return response['embedding']

    async def get_qwen_embedding_4b(self, text: str, max_retries: int = 3, retry_delay: float = 1.0, semaphore: Optional[asyncio.Semaphore] = None, use_async: bool = True) -> Tuple[List[List[float]], List[str]]:
        """
        Get Qwen embeddings for the given text asynchronously, with retry logic and concurrency control.
        
        By default (USE_LOCAL_EMBEDDINGS=true), uses local Qwen3-Embedding-0.6B model.
        Falls back to cloud APIs (Qwen3-Embedding-4B via DeepInfra/Vertex AI) if local is disabled.
        
        Args:
            text (str): The text to embed
            max_retries (int): Maximum number of retries for each chunk (default: 3)
            retry_delay (float): Initial delay between retries in seconds (default: 1.0)
            semaphore (Optional[asyncio.Semaphore]): Semaphore to limit concurrent chunk requests (default: None, uses 10)
            use_async (bool): If False or only one chunk, process serially (no asyncio.gather). Default True.
        Returns:
            Tuple[List[List[float]], List[str]]: A tuple containing:
                - List of embeddings where each embedding is a list of floats
                - List of text chunks
        """
        
        overall_start_time = time.time()
        logger.info("[TIMING] Starting get_qwen_embedding_4b")
        
        # Check if using local embeddings first (default for open source)
        if use_local_embeddings and EmbeddingModel._qwen0pt6b_model_instance is not None:
            logger.info(f"Using local embedding model: {local_embedding_model}")
            loop = asyncio.get_event_loop()
            try:
                local_start = time.time()
                # Generate embeddings using local model
                embeddings_np = await loop.run_in_executor(
                    None,
                    lambda: EmbeddingModel._qwen0pt6b_model_instance.encode(
                        text,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True  # Normalize for better similarity search
                    )
                )
                local_end = time.time()
                logger.info(f"[TIMING] Local embedding generation took: {local_end - local_start:.4f} seconds")
                logger.info(f"Local embedding dimensions: {embeddings_np.shape}")
                
                # Return in expected format: list of embeddings (one per chunk), list of chunks
                # For single text input, we return one embedding
                embeddings_list = [embeddings_np.tolist()]
                chunks_list = [text]
                
                logger.info("[TIMING] Finished get_qwen_embedding_4b (local)")
                return embeddings_list, chunks_list
            except Exception as e:
                logger.error(f"Error in local embedding generation: {str(e)}")
                logger.warning("Falling back to cloud embeddings...")
                # Continue to cloud embedding logic below

        # Check if using Vertex AI or DeepInfra/HuggingFace for cloud embeddings
        if use_vertex_ai:
            logger.info(f"Using Vertex AI for Qwen4B embeddings: project={vertex_ai_project}, endpoint={vertex_ai_endpoint_id}, location={vertex_ai_location}")
            # Vertex AI will be used in process_chunk_with_retry
        else:
            # Validate that we have a valid API URL for DeepInfra/HuggingFace
            # Try deepinfra_api_url first, then fallback to hugging_face_api_url_qwen_4b
            api_url = deepinfra_api_url or hugging_face_api_url_qwen_4b
            api_token = deepinfra_token or hugging_face_access_token
            
            if not api_url:
                if use_local_embeddings:
                    error_msg = "Local embeddings failed and no cloud API configured. Please ensure Qwen3-Embedding-0.6B model loaded successfully, or set DEEPINFRA_API_URL/HUGGING_FACE_API_URL_QWEN_4B with USE_VERTEX_AI=true."
                else:
                    error_msg = "Qwen4B embedding API URL not configured. Please set DEEPINFRA_API_URL, HUGGING_FACE_API_URL_QWEN_4B, or USE_VERTEX_AI=true with VERTEX_AI_* variables, or enable local embeddings with USE_LOCAL_EMBEDDINGS=true."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not api_token:
                if use_local_embeddings:
                    error_msg = "Local embeddings failed and no cloud API token configured. Please ensure Qwen3-Embedding-0.6B model loaded successfully, or set DEEPINFRA_TOKEN/HUGGING_FACE_ACCESS_TOKEN."
                else:
                    error_msg = "Qwen4B embedding API token not configured. Please set DEEPINFRA_TOKEN, HUGGING_FACE_ACCESS_TOKEN, or USE_VERTEX_AI=true with Vertex AI credentials, or enable local embeddings with USE_LOCAL_EMBEDDINGS=true."
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Initialize Qwen tokenizer/config if not already (only needed for DeepInfra/HuggingFace, not Vertex AI)
        if not use_vertex_ai and not hasattr(self, '_qwen4b_splitter'):
            logger.info("Initializing Qwen3-Embedding-4B tokenizer and config...")
            # Set certificate path for HuggingFace
            _set_ssl_cert_paths()
            
            # Initialize tokenizer and config for token counting (only needed for DeepInfra path)
            model_name = "Qwen/Qwen3-Embedding-4B"
            self._qwen4b_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._qwen4b_config = AutoConfig.from_pretrained(model_name)
            
            # Use a simpler text splitter since we're using the API endpoint
            self._qwen4b_splitter = TokenTextSplitter(
                chunk_size=2048,  # Conservative token limit for API
                chunk_overlap=0
            )
            logger.info("Qwen3-Embedding-4B tokenizer, config and splitter initialized")
        elif use_vertex_ai and not hasattr(self, '_qwen4b_splitter'):
            # For Vertex AI, we only need the text splitter (no tokenizer needed)
            logger.info("Initializing Qwen3-Embedding-4B text splitter for Vertex AI...")
            self._qwen4b_splitter = TokenTextSplitter(
                chunk_size=2048,  # Conservative token limit
                chunk_overlap=0
            )
            logger.info("Qwen3-Embedding-4B text splitter initialized for Vertex AI")

        # Chunking
        chunking_start = time.time()
        chunks = self._qwen4b_splitter.split_text(text)
        logger.info(f"Qwen4B chunk size: {len(chunks)}")
        chunking_end = time.time()
        logger.info(f"[TIMING] Qwen4B split text into {len(chunks)} chunks in {chunking_end - chunking_start:.4f} seconds")

        # Use provided semaphore or create a default one
        if semaphore is None:
            semaphore = asyncio.Semaphore(10)

        def get_payload_size(payload):
            return len(json.dumps(payload).encode('utf-8'))

        async def process_chunk_with_retry(chunk: str, i: int) -> Optional[List[float]]:
            last_exception = None
            async with semaphore:
                for attempt in range(1, max_retries + 1):
                    attempt_start = time.time()
                    
                    if use_vertex_ai:
                        # Use Vertex AI prediction API
                        try:
                            if not VERTEX_AI_AVAILABLE:
                                raise ImportError("google-cloud-aiplatform not installed. Install with: poetry add google-cloud-aiplatform")
                            
                            # Initialize Vertex AI client if not already done (class-level check)
                            if not EmbeddingModel._vertex_ai_initialized:
                                init_start = time.time()
                                aiplatform.init(project=vertex_ai_project, location=vertex_ai_location)
                                EmbeddingModel._vertex_ai_initialized = True
                                init_end = time.time()
                                logger.info(f"Vertex AI initialized: project={vertex_ai_project}, location={vertex_ai_location}, endpoint={vertex_ai_endpoint_id} (took {init_end - init_start:.4f}s)")
                            
                            # Prepare instance for prediction
                            prep_start = time.time()
                            # Vertex AI expects instances as a list of dictionaries
                            # The model expects "inputs" field (plural) based on the model's input schema
                            instances = [{"inputs": chunk}]
                            prep_end = time.time()
                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Instance preparation took: {prep_end - prep_start:.4f} seconds (attempt {attempt})")
                            
                            # Cache endpoint object for connection pooling (reuse across all calls)
                            # This avoids creating a new endpoint object for each prediction, improving performance
                            if EmbeddingModel._vertex_ai_endpoint is None:
                                endpoint_init_start = time.time()
                                try:
                                    EmbeddingModel._vertex_ai_endpoint = aiplatform.Endpoint(vertex_ai_endpoint_id)
                                    endpoint_init_end = time.time()
                                    logger.info(f"Vertex AI endpoint cached for connection pooling (took {endpoint_init_end - endpoint_init_start:.4f}s)")
                                except Exception as endpoint_error:
                                    error_str = str(endpoint_error).lower()
                                    if "permission" in error_str or "iam_permission_denied" in error_str or "403" in error_str:
                                        logger.error(f"❌ VERTEX AI PERMISSION ERROR: Cannot access endpoint {vertex_ai_endpoint_id}")
                                        logger.error(f"❌ Required permission: 'aiplatform.endpoints.get' on resource 'projects/{vertex_ai_project}/locations/{vertex_ai_location}/endpoints/{vertex_ai_endpoint_id}'")
                                        logger.error(f"❌ Please grant the Vertex AI User role or 'aiplatform.endpoints.get' permission to your service account")
                                        logger.error(f"❌ Service account: Check GOOGLE_APPLICATION_CREDENTIALS or default credentials")
                                        # If DeepInfra/HuggingFace is available, suggest fallback
                                        if deepinfra_api_url or hugging_face_api_url_qwen_4b:
                                            logger.warning(f"🔄 FALLBACK AVAILABLE: DeepInfra/HuggingFace API is configured. Consider setting USE_VERTEX_AI=false to use fallback.")
                                        raise ValueError(f"Vertex AI permission denied. Please grant 'aiplatform.endpoints.get' permission or use DeepInfra/HuggingFace fallback by setting USE_VERTEX_AI=false")
                                    raise
                            
                            endpoint = EmbeddingModel._vertex_ai_endpoint
                            
                            # Vertex AI inference timing
                            # Run synchronous predict() in executor to avoid blocking event loop (async optimization)
                            # This allows other async operations to continue while waiting for the prediction
                            inference_start = time.time()
                            loop = asyncio.get_event_loop()
                            predictions = await loop.run_in_executor(
                                None,  # Use default ThreadPoolExecutor
                                endpoint.predict,
                                instances
                            )
                            inference_end = time.time()
                            inference_duration = inference_end - inference_start
                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Vertex AI inference took: {inference_duration:.4f} seconds (attempt {attempt})")
                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Vertex AI inference rate: {len(chunk)/inference_duration:.2f} chars/sec (attempt {attempt})")
                            
                            # Extract embedding from Vertex AI response
                            # Vertex AI returns predictions as a list, each prediction is a dict
                            if predictions and len(predictions.predictions) > 0:
                                prediction = predictions.predictions[0]
                                logger.debug(f"Vertex AI prediction type: {type(prediction)}, value: {prediction}")
                                
                                # The embedding might be in different fields depending on model output
                                # Common formats: 'embedding', 'output', 'predictions', or direct list
                                embedding = None
                                
                                if isinstance(prediction, list):
                                    # If prediction is directly a list, use it
                                    embedding = prediction
                                elif isinstance(prediction, dict):
                                    # Try common field names
                                    embedding = prediction.get('embedding') or prediction.get('output') or prediction.get('predictions')
                                    if embedding is None:
                                        # Try to get the first value if it's a dict with one key
                                        values = list(prediction.values())
                                        if values and isinstance(values[0], list):
                                            embedding = values[0]
                                else:
                                    embedding = None
                                
                                # Flatten nested lists if needed (handle list of lists)
                                if embedding is not None:
                                    if isinstance(embedding, list) and len(embedding) > 0:
                                        # Check if first element is a list (nested structure)
                                        if isinstance(embedding[0], list):
                                            # Flatten: take the first inner list or concatenate all
                                            embedding = embedding[0] if len(embedding) == 1 else [item for sublist in embedding for item in sublist]
                                            logger.info(f"Flattened nested embedding structure: {len(embedding)} dimensions")
                                        
                                        # Ensure all elements are floats (not strings or other types)
                                        try:
                                            parse_start = time.time()
                                            embedding = [float(x) for x in embedding]
                                            parse_end = time.time()
                                            logger.info(f"Qwen4B Chunk {i} embedding dimensions: {len(embedding)}")
                                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Embedding parsing took: {parse_end - parse_start:.4f} seconds (attempt {attempt})")
                                            
                                            attempt_end = time.time()
                                            total_attempt_time = attempt_end - attempt_start
                                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Total attempt time: {total_attempt_time:.4f} seconds (attempt {attempt})")
                                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Breakdown - Prep: {prep_end - prep_start:.4f}s, Inference: {inference_duration:.4f}s, Parse: {parse_end - parse_start:.4f}s")
                                            return embedding
                                        except (ValueError, TypeError) as e:
                                            logger.error(f"Qwen4B Vertex AI embedding contains non-numeric values: {e}, embedding: {embedding[:5] if len(embedding) > 5 else embedding}")
                                            return None
                                    else:
                                        logger.error(f"Qwen4B Vertex AI embedding is not a list: {type(embedding)}")
                                        return None
                                else:
                                    logger.error(f"Qwen4B Vertex AI unexpected response format: {type(prediction)}, value: {prediction}")
                                    return None
                            else:
                                logger.error(f"Qwen4B Vertex AI returned empty predictions")
                                return None
                                
                        except ImportError:
                            logger.error("google-cloud-aiplatform not installed. Install with: poetry add google-cloud-aiplatform")
                            raise
                        except Exception as e:
                            error_str = str(e).lower()
                            is_permission_error = "permission" in error_str or "iam_permission_denied" in error_str or "403" in error_str
                            
                            if is_permission_error:
                                logger.error(f"❌ Qwen4B Vertex AI PERMISSION ERROR (attempt {attempt}/{max_retries}): {str(e)}")
                                logger.error(f"❌ This is a permanent permission issue - retrying won't help")
                                # Don't retry permission errors - they won't resolve by retrying
                                raise ValueError(f"Vertex AI permission denied: {str(e)}. Please grant 'aiplatform.endpoints.get' permission or set USE_VERTEX_AI=false to use DeepInfra/HuggingFace fallback")
                            else:
                                logger.error(f"Qwen4B Vertex AI error (attempt {attempt}/{max_retries}): {str(e)}")
                                last_exception = e
                                if attempt < max_retries:
                                    delay = retry_delay * (2 ** (attempt - 1))
                                    logger.warning(f"Retrying Qwen4B chunk {i} after {delay:.2f}s (attempt {attempt}/{max_retries})")
                                    await asyncio.sleep(delay)
                                continue
                    else:
                        # Original DeepInfra/HuggingFace API code
                        # Tokenization timing
                        token_start = time.time()
                        tokenized_chunk = self._qwen4b_tokenizer(
                            chunk,
                            truncation=True,
                            max_length=self._qwen4b_config.max_position_embeddings,
                            return_tensors="pt"
                        )
                        token_end = time.time()
                        logger.info(f"[TIMING] [Qwen4B Chunk {i}] Tokenization took: {token_end - token_start:.4f} seconds (attempt {attempt})")

                        num_tokens = len(tokenized_chunk['input_ids'][0])
                        if num_tokens > self._qwen4b_config.max_position_embeddings:
                            logger.warning(f"Qwen4B Chunk {i} exceeds max token size: {num_tokens} tokens")
                            return None
                        if num_tokens > MAX_EMBEDDING_BATCH_TOKENS:
                            logger.warning(f"Qwen4B Chunk {i} exceeds configured MAX_EMBEDDING_BATCH_TOKENS: {num_tokens} > {MAX_EMBEDDING_BATCH_TOKENS}")
                            return None

                        # Payload prep timing
                        payload_start = time.time()
                        payload = {
                            "input": chunk,
                            "model": "Qwen/Qwen3-Embedding-4B",
                            "encoding_format": "float"
                        }
                        payload_size = get_payload_size(payload)
                        logger.info(f"[Qwen4B Chunk {i}] Payload size: {payload_size} bytes (limit: {MAX_EMBEDDING_PAYLOAD_SIZE})")
                        if payload_size > MAX_EMBEDDING_PAYLOAD_SIZE:
                            logger.error(f"[Qwen4B Chunk {i}] Payload size {payload_size} exceeds limit of {MAX_EMBEDDING_PAYLOAD_SIZE} bytes. Skipping.")
                            return None
                        payload_end = time.time()
                        logger.info(f"[TIMING] [Qwen4B Chunk {i}] Payload prep took: {payload_end - payload_start:.4f} seconds (attempt {attempt})")

                        headers = {
                            "Authorization": f"Bearer {api_token}",
                            "Content-Type": "application/json"
                        }

                        try:
                            
                            http_start = time.time()
                            # Use a shared client instance for better connection reuse
                            if not hasattr(self, '_http_client'):
                                self._http_client = httpx.AsyncClient(verify=False)
                            response = await self._http_client.post(
                                api_url,
                                headers=headers,
                                json=payload,
                                timeout=60.0  # Increased from 20.0 to handle slow API responses
                            )
                            http_end = time.time()
                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] HTTP request took: {http_end - http_start:.4f} seconds (attempt {attempt})")

                            if response.status_code == 400:
                                error_text = response.text
                                logger.error(f"Qwen4B Bad Request: {error_text}")
                                return None
                            if response.status_code == 413:
                                logger.error(f"[Qwen4B Chunk {i}] Received 413 Payload Too Large from embedding API. Payload size: {payload_size} bytes.")
                                return None

                            parse_start = time.time()
                            response.raise_for_status()
                            response_json = response.json()
                            parse_end = time.time()
                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Response parsing took: {parse_end - parse_start:.4f} seconds (attempt {attempt})")

                            # The DeepInfra API returns an OpenAI-compatible object.
                            # We need to extract the embedding from response_json['data'][0]['embedding']
                            if 'data' in response_json and response_json['data'] and 'embedding' in response_json['data'][0]:
                                embedding = response_json['data'][0]['embedding']
                                logger.info(f"Qwen4B Chunk {i} embedding dimensions: {len(embedding)}")
                                attempt_end = time.time()
                                logger.info(f"[TIMING] [Qwen4B Chunk {i}] Total attempt time: {attempt_end - attempt_start:.4f} seconds (attempt {attempt})")
                                return embedding
                            else:
                                logger.error(f"Qwen4B Unexpected API response format: {response_json}")
                                attempt_end = time.time()
                                logger.info(f"[TIMING] [Qwen4B Chunk {i}] Unexpected format, total attempt time: {attempt_end - attempt_start:.4f} seconds (attempt {attempt})")
                                return None
                        except httpx.HTTPError as e:
                            logger.error(f"Qwen4B HTTP error in async API request (attempt {attempt}/{max_retries}): {str(e)}")
                            last_exception = e
                        except Exception as e:
                            logger.error(f"Qwen4B Unexpected error in async API request (attempt {attempt}/{max_retries}): {str(e)}")
                            last_exception = e
                        if attempt < max_retries:
                            delay = retry_delay * (2 ** (attempt - 1))
                            logger.warning(f"Retrying Qwen4B chunk {i} after {delay:.2f}s (attempt {attempt}/{max_retries})")
                            sleep_start = time.time()
                            await asyncio.sleep(delay)
                            sleep_end = time.time()
                            logger.info(f"[TIMING] [Qwen4B Chunk {i}] Slept for: {sleep_end - sleep_start:.4f} seconds before retry (attempt {attempt})")
                
                logger.error(f"All {max_retries} attempts failed for Qwen4B chunk {i}")
                if last_exception:
                    raise last_exception
                return None

        # Process all chunks concurrently with retry logic and concurrency control
        embedding_start = time.time()
        if not use_async or len(chunks) == 1:
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                emb = await process_chunk_with_retry(chunk, i)
                chunk_embeddings.append(emb)
        else:
            chunk_embeddings = await asyncio.gather(
                *[process_chunk_with_retry(chunk, i) for i, chunk in enumerate(chunks)]
            )
        embedding_end = time.time()
        logger.info(f"[TIMING] All Qwen4B chunk embedding requests took: {embedding_end - embedding_start:.4f} seconds")

        # Filter out None values
        filter_start = time.time()
        embeddings = [emb for emb in chunk_embeddings if emb is not None]
        filter_end = time.time()
        logger.info(f"[TIMING] Filtering Qwen4B embeddings took: {filter_end - filter_start:.4f} seconds")

        if not embeddings:
            logger.warning("No Qwen4B embeddings were generated. Returning empty list.")
            logger.warning(f"[TIMING] Finished get_qwen_embedding_4b (no embeddings)")
            return [], chunks

        total_time = time.time() - overall_start_time
        logger.info(f"[TIMING] Total async Qwen4B embedding generation took: {total_time:.4f} seconds")
        logger.info("[TIMING] Finished get_qwen_embedding_4b")

        return embeddings, chunks