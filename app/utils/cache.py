"""
Caching utilities for AI responses and data
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict
from cachetools import TTLCache
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """In-memory cache manager with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "errors": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if key in self.cache:
                self.stats["hits"] += 1
                value = self.cache[key]
                logger.debug(f"Cache hit: {key}")
                return value
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # For TTLCache, we can't set per-item TTL easily
            # We'll store with timestamp and check manually if needed
            cache_entry = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
            
            self.cache[key] = cache_entry
            self.stats["sets"] += 1
            logger.debug(f"Cache set: {key}")
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Cache delete: {key}")
                return True
            return False
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(total_requests, 1)) * 100
        
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

class TimedCache:
    """Simple time-based cache"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < entry["ttl"]:
                return entry["value"]
            else:
                # Expired, remove it
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL"""
        self.cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry["timestamp"] >= entry["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value and update access order"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value and manage size"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
