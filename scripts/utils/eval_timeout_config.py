"""
Eval script timeout and retry configuration for different network conditions.
Optimized for Lebanon→US→Qdrant network paths.
"""

import os
from typing import Optional

class EvalTimeoutConfig:
    """Timeout configuration for eval scripts based on network conditions"""
    
    # Base timeouts (in seconds)
    DEFAULT_REQUEST_TIMEOUT = 180  # 3 minutes for Lebanon→US
    FAST_REQUEST_TIMEOUT = 60      # 1 minute for US→US  
    BATCH_PROCESSING_TIMEOUT = 600 # 10 minutes for large batches
    
    # Retry configuration  
    MAX_RETRIES = 1               # Reduced to prevent duplicates
    RETRY_DELAY_BASE = 10         # Start with 10s delay
    RETRY_DELAY_MULTIPLIER = 1.5  # Slower exponential backoff
    
    # Batch size optimization
    LEBANON_BATCH_SIZE = 20       # Smaller batches for high latency
    US_BATCH_SIZE = 50           # Larger batches for low latency
    
    @classmethod
    def get_config_for_location(cls, location: str = "lebanon") -> dict:
        """Get optimized config based on network location"""
        
        if location.lower() == "lebanon":
            return {
                "request_timeout": cls.DEFAULT_REQUEST_TIMEOUT,
                "batch_size": cls.LEBANON_BATCH_SIZE,
                "max_retries": cls.MAX_RETRIES,
                "retry_delay": cls.RETRY_DELAY_BASE,
                "retry_multiplier": cls.RETRY_DELAY_MULTIPLIER,
                "use_production": True  # Recommend production for Lebanon
            }
        else:  # US or other low-latency locations
            return {
                "request_timeout": cls.FAST_REQUEST_TIMEOUT,
                "batch_size": cls.US_BATCH_SIZE, 
                "max_retries": 2,  # Can afford more retries
                "retry_delay": 5,   # Faster retries
                "retry_multiplier": 2.0,
                "use_production": False  # Can use local if desired
            }
    
    @classmethod 
    def get_papr_client_config(cls, location: str = "lebanon") -> dict:
        """Get Papr client configuration"""
        config = cls.get_config_for_location(location)
        
        base_url = "https://memory.papr.ai/api/v1" if config["use_production"] else None
        
        return {
            "base_url": base_url,
            "timeout": config["request_timeout"],
            "max_retries": config["max_retries"]
        }

# Auto-detect location from environment or default to Lebanon
NETWORK_LOCATION = os.getenv("EVAL_NETWORK_LOCATION", "lebanon")
EVAL_CONFIG = EvalTimeoutConfig.get_config_for_location(NETWORK_LOCATION)
PAPR_CONFIG = EvalTimeoutConfig.get_papr_client_config(NETWORK_LOCATION)

print(f"Eval config for {NETWORK_LOCATION}: {EVAL_CONFIG}")
print(f"Papr config: {PAPR_CONFIG}") 