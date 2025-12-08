import time
import threading
from typing import Optional, Any
import heapq  # For efficient min-heap eviction
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

# Simple in-memory cache with TTL
class TTLCache:
    def __init__(self, maxsize=1000, ttl_seconds=180, daily_flush=True):  # Increased to 3 minutes
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.eviction_heap = []  # Min-heap of (timestamp, key) for efficient oldest eviction
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Cleanup every 60 seconds
        self.daily_flush = daily_flush
        self.last_daily_flush = time.time()
        self.daily_flush_interval = 24 * 60 * 60  # 24 hours in seconds
    
    def _cleanup_expired(self):
        """Remove expired entries to prevent memory bloat"""
        current_time = time.time()
        
        # Check if daily flush is needed
        if self.daily_flush and current_time - self.last_daily_flush > self.daily_flush_interval:
            with self.lock:
                cache_size = len(self.cache)
                self.cache.clear()
                self.stats = {"hits": 0, "misses": 0, "sets": 0}
                self.last_daily_flush = current_time
                logger.info(f"Daily cache flush executed: cleared {cache_size} entries")
                return  # Skip regular cleanup since we just flushed everything
        
        # Regular cleanup of expired entries
        if current_time - self.last_cleanup > self.cleanup_interval:
            with self.lock:
                expired_keys = [k for k, (v, t) in self.cache.items() 
                              if current_time - t >= self.ttl_seconds]
                for k in expired_keys:
                    del self.cache[k]
                if expired_keys:
                    logger.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")
                self.last_cleanup = current_time
    
    def get(self, key: str) -> Optional[Any]:
        self._cleanup_expired()  # Cleanup expired entries
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self.stats["hits"] += 1
                    logger.info(f"Cache HIT for {key[:20]}... (age: {time.time() - timestamp:.1f}s)")
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
                    logger.info(f"Cache EXPIRED for {key[:20]}... (age: {time.time() - timestamp:.1f}s)")
            
            self.stats["misses"] += 1
            logger.info(f"Cache MISS for {key[:20]}...")
            return None
    
    def set(self, key: str, value: Any):
        self._cleanup_expired()  # Cleanup expired entries
        
        with self.lock:
            timestamp = time.time()
            self.cache[key] = (value, timestamp)
            heapq.heappush(self.eviction_heap, (timestamp, key))
            
            # Evict if over maxsize
            while len(self.cache) > self.maxsize:
                # Pop oldest from heap, skip if already removed/expired
                while self.eviction_heap:
                    old_ts, old_key = heapq.heappop(self.eviction_heap)
                    if old_key in self.cache and self.cache[old_key][1] == old_ts:
                        del self.cache[old_key]
                        logger.info(f"Cache EVICTED oldest entry: {old_key[:20]}...")
                        break
            
            self.stats["sets"] += 1
            logger.info(f"Cache SET for {key[:20]}... (size: {len(self.cache)}/{self.maxsize})")
    
    def get_stats(self):
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            hours_since_flush = (time.time() - self.last_daily_flush) / 3600
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "sets": self.stats["sets"],
                "hit_rate": f"{hit_rate:.1f}%",
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl_seconds,
                "daily_flush_enabled": self.daily_flush,
                "hours_since_last_flush": f"{hours_since_flush:.1f}h"
            }
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "sets": 0}
            logger.info("Cache cleared")
    
    def find_value(self, target_value: Any) -> Optional[str]:
        """Find a key by its value (reverse lookup)"""
        with self.lock:
            current_time = time.time()
            for key, (value, timestamp) in self.cache.items():
                # Check if entry is still valid
                if current_time - timestamp < self.ttl_seconds:
                    if value == target_value:
                        return key
            return None
    
    def __len__(self) -> int:
        """Return the number of valid (non-expired) entries in the cache"""
        with self.lock:
            current_time = time.time()
            valid_count = sum(1 for _, timestamp in self.cache.values() 
                            if current_time - timestamp < self.ttl_seconds)
            return valid_count

# Global cache instances following authentication security best practices
# Authentication caches: Short TTL (2-3 min) provides security - NO daily flush needed
api_key_cache = TTLCache(maxsize=1000, ttl_seconds=180, daily_flush=False)  # 3 minutes - TTL provides security
session_token_cache = TTLCache(maxsize=1000, ttl_seconds=180, daily_flush=False)  # 3 minutes - TTL provides security
access_token_cache = TTLCache(maxsize=1000, ttl_seconds=180, daily_flush=False)  # 3 minutes - TTL provides security
auth_optimized_cache = TTLCache(maxsize=1000, ttl_seconds=120, daily_flush=False)  # 2 minutes - optimal for auth security
api_key_to_user_id_cache = TTLCache(maxsize=1000, ttl_seconds=600, daily_flush=False)  # 10 minutes - stable mapping

# Business data caches: Longer TTL + daily flush for monthly billing cycles
workspace_subscription_cache = TTLCache(maxsize=1000, ttl_seconds=3600, daily_flush=True)  # 1 hour + daily flush - subscription changes monthly
customer_tier_cache = TTLCache(maxsize=1000, ttl_seconds=3600, daily_flush=True)  # 1 hour + daily flush - tier changes infrequently 

# Enhanced API key cache: caches resolved org/namespace and owner for keys
enhanced_api_key_cache = TTLCache(maxsize=1000, ttl_seconds=600, daily_flush=True)  # 10 minutes + daily flush 