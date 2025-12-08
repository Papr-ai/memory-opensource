"""
Provider Type Parser - Reusable typed parsing for provider SDKs

This module provides utilities to parse provider responses using their native
SDK types (Pydantic models) when available, ensuring type safety and validation.
"""

from typing import Dict, Any, Optional, Type
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


def parse_with_provider_sdk(
    provider_name: str,
    provider_response: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parse provider response using provider SDK types (best-effort).
    
    This function attempts to validate and normalize provider responses using
    their native SDK Pydantic models when available. If typed parsing fails
    or is not available, it returns the original dict unchanged.
    
    Args:
        provider_name: Name of the provider (reducto, tensorlake, gemini, etc.)
        provider_response: Raw provider response dict
        
    Returns:
        Validated dict (from SDK model) or original dict if parsing unavailable
        
    Examples:
        >>> result = parse_with_provider_sdk("reducto", raw_response)
        >>> # Returns validated dict from reducto.types.shared.pipeline_response.PipelineResponse
    """
    name = (provider_name or "").lower()
    
    if name == "reducto":
        return _parse_reducto_response(provider_response)
    elif name == "tensorlake":
        return _parse_tensorlake_response(provider_response)
    elif name == "gemini":
        return _parse_gemini_response(provider_response)
    elif name == "deepseek-ocr":
        return _parse_deepseek_response(provider_response)
    elif name == "paddleocr":
        return _parse_paddleocr_response(provider_response)
    else:
        logger.info(f"No typed SDK parser available for provider: {provider_name}")
        return provider_response


def _parse_reducto_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Reducto response using Reducto SDK types."""
    try:
        from reducto.types.shared.pipeline_response import PipelineResponse  # type: ignore
        
        # Try to validate via SDK model
        typed_obj = PipelineResponse(**response)
        
        # Convert back to dict without lossy string casts
        if hasattr(typed_obj, 'model_dump'):
            validated_dict = typed_obj.model_dump(by_alias=True)
        elif hasattr(typed_obj, 'dict'):
            validated_dict = typed_obj.dict()
        else:
            validated_dict = response
            
        if isinstance(validated_dict, dict):
            logger.info("Reducto typed parse via PipelineResponse succeeded")
            return validated_dict
        else:
            logger.warning(f"Reducto SDK parse returned non-dict: {type(validated_dict)}")
            return response
            
    except ImportError:
        logger.info("Reducto SDK not available for typed parsing")
        return response
    except Exception as e:
        logger.info(f"Reducto typed parse failed, using raw response: {e}")
        return response


def _parse_tensorlake_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse TensorLake response using TensorLake SDK types."""
    try:
        # TensorLake SDK type parsing (if SDK becomes available)
        # For now, return as-is
        logger.info("TensorLake typed parsing not yet implemented")
        return response
    except Exception as e:
        logger.info(f"TensorLake typed parse failed: {e}")
        return response


def _parse_gemini_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Gemini response using Google AI SDK types."""
    try:
        # Gemini uses google-generativeai library
        # Response structure is typically already well-formed
        logger.info("Gemini responses are already typed via google-generativeai SDK")
        return response
    except Exception as e:
        logger.info(f"Gemini typed parse failed: {e}")
        return response


def _parse_deepseek_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse DeepSeek-OCR response using DeepSeek types."""
    try:
        # DeepSeek-OCR response parsing
        # The HuggingFace model returns structured JSON
        # We can add Pydantic models for validation
        from models.provider_types.deepseek import DeepSeekOCRResponse
        
        typed_obj = DeepSeekOCRResponse(**response)
        if hasattr(typed_obj, 'model_dump'):
            validated_dict = typed_obj.model_dump(by_alias=True)
            logger.info("DeepSeek-OCR typed parse succeeded")
            return validated_dict
        return response
    except ImportError:
        logger.info("DeepSeek-OCR types not yet defined")
        return response
    except Exception as e:
        logger.info(f"DeepSeek-OCR typed parse failed: {e}")
        return response


def _parse_paddleocr_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse PaddleOCR response using PaddleOCR SDK types."""
    try:
        # PaddleOCR returns list of detection/recognition results
        # We can add Pydantic models for validation
        from models.provider_types.paddleocr import PaddleOCRResponse
        
        typed_obj = PaddleOCRResponse(**response)
        if hasattr(typed_obj, 'model_dump'):
            validated_dict = typed_obj.model_dump(by_alias=True)
            logger.info("PaddleOCR typed parse succeeded")
            return validated_dict
        return response
    except ImportError:
        logger.info("PaddleOCR types not yet defined")
        return response
    except Exception as e:
        logger.info(f"PaddleOCR typed parse failed: {e}")
        return response


def validate_provider_response(
    provider_name: str,
    response: Dict[str, Any]
) -> bool:
    """
    Validate provider response structure without full parsing.
    
    Quick validation to check if response has expected top-level structure.
    
    Args:
        provider_name: Name of the provider
        response: Provider response to validate
        
    Returns:
        True if response appears valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
        
    name = provider_name.lower()
    
    if name == "reducto":
        # Reducto responses should have "result" or "status"
        return "result" in response or "status" in response
    elif name == "tensorlake":
        # TensorLake responses should have "pages" or "content"
        return "pages" in response or "content" in response
    elif name == "gemini":
        # Gemini responses vary but typically have "candidates" or "text"
        return "candidates" in response or "text" in response
    elif name == "deepseek-ocr":
        # DeepSeek-OCR responses should have OCR results
        return "ocr_results" in response or "text" in response
    elif name == "paddleocr":
        # PaddleOCR returns list of results
        return isinstance(response, (list, dict))
    else:
        # Unknown provider - assume valid if it's a dict
        return True

