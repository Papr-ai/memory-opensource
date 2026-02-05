"""
Temporal Plugin - Cloud Edition Only

Provides durable workflow execution for:
- Batch memory processing (> 100 items)
- Long-running document processing
- Guaranteed webhook delivery

Only available in cloud edition.
"""

from config import get_features

# Only export if Temporal is enabled
if get_features().is_enabled("temporal"):
    try:
        from .client import get_temporal_client
        from .workflows import process_batch_workflow
        
        __all__ = ["get_temporal_client", "process_batch_workflow"]
    except ImportError as e:
        import logging
        logging.warning(f"Temporal plugin enabled but import failed: {e}")
        __all__ = []
else:
    __all__ = []

