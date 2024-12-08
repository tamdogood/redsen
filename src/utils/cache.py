from typing import Any, Dict, Optional
from datetime import datetime, timedelta

class CacheManager:
    """Simple in-memory cache with expiration"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.expiry_times: Dict[str, datetime] = {}
        
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set a value in the cache with expiration
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache[key] = value
        self.expiry_times[key] = datetime.now() + timedelta(seconds=ttl)
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        if key in self.cache:
            if datetime.now() < self.expiry_times[key]:
                return self.cache[key]
            else:
                # Clean up expired entry
                del self.cache[key]
                del self.expiry_times[key]
        return None
        
    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.expiry_times.clear()
        
    def remove(self, key: str) -> None:
        """Remove specific key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.expiry_times[key]
            
    def clean_expired(self) -> None:
        """Clean up all expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, expiry in self.expiry_times.items()
            if current_time >= expiry
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.expiry_times[key]