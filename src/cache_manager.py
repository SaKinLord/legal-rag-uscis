# src/cache_manager.py

import os
import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import OrderedDict
import threading
import numpy as np
import redis
from dataclasses import dataclass, asdict
import sqlite3
import lz4.frame
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = None
    size_bytes: int = 0
    ttl: Optional[int] = None  # Time to live in seconds
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


class CacheManager:
    """
    Comprehensive caching system for RAG pipeline with multiple storage backends.
    Supports in-memory, disk-based, and Redis caching with various eviction policies.
    """
    
    def __init__(self,
                 cache_type: str = "hybrid",  # "memory", "disk", "redis", "hybrid"
                 max_memory_size_mb: int = 500,
                 max_disk_size_mb: int = 5000,
                 cache_dir: str = "cache",
                 redis_config: Optional[Dict] = None,
                 eviction_policy: str = "lru",  # "lru", "lfu", "ttl"
                 compression: bool = True,
                 default_ttl: Optional[int] = 3600):  # 1 hour default
        
        self.cache_type = cache_type
        self.max_memory_size = max_memory_size_mb * 1024 * 1024  # Convert to bytes
        self.max_disk_size = max_disk_size_mb * 1024 * 1024
        self.cache_dir = cache_dir
        self.eviction_policy = eviction_policy
        self.compression = compression
        self.default_ttl = default_ttl
        
        # Initialize caches
        self.memory_cache = OrderedDict() if cache_type in ["memory", "hybrid"] else None
        self.current_memory_size = 0
        
        # Disk cache setup
        if cache_type in ["disk", "hybrid"]:
            os.makedirs(cache_dir, exist_ok=True)
            self.db_path = os.path.join(cache_dir, "cache.db")
            self._init_disk_cache()
        
        # Redis setup
        self.redis_client = None
        if cache_type in ["redis", "hybrid"]:
            self._init_redis(redis_config)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
    def _init_disk_cache(self):
        """Initialize SQLite-based disk cache."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                timestamp REAL,
                access_count INTEGER,
                last_accessed REAL,
                size_bytes INTEGER,
                ttl INTEGER
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)')
        conn.commit()
        conn.close()
    
    def _init_redis(self, redis_config: Optional[Dict]):
        """Initialize Redis connection."""
        try:
            config = redis_config or {'host': 'localhost', 'port': 6379, 'db': 0}
            self.redis_client = redis.StrictRedis(**config, decode_responses=False)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to other cache types.")
            self.redis_client = None
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        serialized = pickle.dumps(value)
        if self.compression:
            serialized = lz4.frame.compress(serialized)
        return serialized
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.compression:
            data = lz4.frame.decompress(data)
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        with self.lock:
            self.stats['total_requests'] += 1
            
            # Check memory cache first
            if self.memory_cache is not None and key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_expired(entry):
                    self._remove_from_memory(key)
                else:
                    self.stats['hits'] += 1
                    self._update_access(key, 'memory')
                    return entry.value
            
            # Check Redis
            if self.redis_client:
                try:
                    data = self.redis_client.get(f"rag:{key}")
                    if data:
                        value = self._deserialize(data)
                        self.stats['hits'] += 1
                        self._update_access(key, 'redis')
                        # Promote to memory cache if hybrid
                        if self.cache_type == "hybrid" and self.memory_cache is not None:
                            self._add_to_memory(key, value)
                        return value
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
            
            # Check disk cache
            if self.cache_type in ["disk", "hybrid"]:
                value = self._get_from_disk(key)
                if value is not None:
                    self.stats['hits'] += 1
                    # Promote to memory/Redis if hybrid
                    if self.cache_type == "hybrid":
                        if self.memory_cache is not None:
                            self._add_to_memory(key, value)
                        if self.redis_client:
                            self._add_to_redis(key, value)
                    return value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache."""
        with self.lock:
            ttl = ttl or self.default_ttl
            
            # Add to appropriate caches based on type
            if self.cache_type == "memory" and self.memory_cache is not None:
                return self._add_to_memory(key, value, ttl)
            elif self.cache_type == "disk":
                return self._add_to_disk(key, value, ttl)
            elif self.cache_type == "redis" and self.redis_client:
                return self._add_to_redis(key, value, ttl)
            elif self.cache_type == "hybrid":
                # Add to all available caches
                success = False
                if self.memory_cache is not None:
                    success |= self._add_to_memory(key, value, ttl)
                if self.redis_client:
                    success |= self._add_to_redis(key, value, ttl)
                success |= self._add_to_disk(key, value, ttl)
                return success
            
            return False
    
    def _add_to_memory(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Add entry to memory cache."""
        try:
            serialized = self._serialize(value)
            size = len(serialized)
            
            # Check if we need to evict
            while self.current_memory_size + size > self.max_memory_size:
                if not self._evict_from_memory():
                    return False
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size,
                ttl=ttl
            )
            
            self.memory_cache[key] = entry
            self.current_memory_size += size
            
            # Move to end for LRU
            self.memory_cache.move_to_end(key)
            
            return True
        except Exception as e:
            logger.error(f"Error adding to memory cache: {e}")
            return False
    
    def _add_to_disk(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Add entry to disk cache."""
        try:
            serialized = self._serialize(value)
            size = len(serialized)
            
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO cache 
                (key, value, timestamp, access_count, last_accessed, size_bytes, ttl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                key, serialized, time.time(), 0, time.time(), size, ttl
            ))
            conn.commit()
            conn.close()
            
            # Check disk size and evict if necessary
            self._check_disk_size()
            
            return True
        except Exception as e:
            logger.error(f"Error adding to disk cache: {e}")
            return False
    
    def _add_to_redis(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Add entry to Redis cache."""
        try:
            serialized = self._serialize(value)
            redis_key = f"rag:{key}"
            
            if ttl:
                self.redis_client.setex(redis_key, ttl, serialized)
            else:
                self.redis_client.set(redis_key, serialized)
            
            # Store metadata
            meta_key = f"rag:meta:{key}"
            metadata = {
                'timestamp': time.time(),
                'access_count': 0,
                'last_accessed': time.time(),
                'size_bytes': len(serialized)
            }
            self.redis_client.hset(meta_key, mapping={
                k: str(v) for k, v in metadata.items()
            })
            
            if ttl:
                self.redis_client.expire(meta_key, ttl)
            
            return True
        except Exception as e:
            logger.error(f"Error adding to Redis cache: {e}")
            return False
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Retrieve entry from disk cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                'SELECT value, ttl, timestamp FROM cache WHERE key = ?',
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                value_data, ttl, timestamp = row
                # Check TTL
                if ttl and (time.time() - timestamp) > ttl:
                    conn.execute('DELETE FROM cache WHERE key = ?', (key,))
                    conn.commit()
                    conn.close()
                    return None
                
                # Update access statistics
                conn.execute('''
                    UPDATE cache 
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE key = ?
                ''', (time.time(), key))
                conn.commit()
                conn.close()
                
                return self._deserialize(value_data)
            
            conn.close()
            return None
        except Exception as e:
            logger.error(f"Error getting from disk cache: {e}")
            return None
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        return (time.time() - entry.timestamp) > entry.ttl
    
    def _update_access(self, key: str, cache_type: str):
        """Update access statistics for a cache entry."""
        if cache_type == 'memory' and self.memory_cache is not None:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                # Move to end for LRU
                self.memory_cache.move_to_end(key)
        elif cache_type == 'redis' and self.redis_client:
            try:
                meta_key = f"rag:meta:{key}"
                self.redis_client.hincrby(meta_key, 'access_count', 1)
                self.redis_client.hset(meta_key, 'last_accessed', str(time.time()))
            except Exception as e:
                logger.error(f"Error updating Redis access stats: {e}")
    
    def _evict_from_memory(self) -> bool:
        """Evict entry from memory cache based on policy."""
        if not self.memory_cache:
            return False
        
        if self.eviction_policy == "lru":
            # Remove least recently used (first item)
            key, entry = self.memory_cache.popitem(last=False)
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self.memory_cache.keys(), 
                     key=lambda k: self.memory_cache[k].access_count)
            entry = self.memory_cache.pop(key)
        elif self.eviction_policy == "ttl":
            # Remove expired or oldest
            expired_keys = [k for k, v in self.memory_cache.items() if self._is_expired(v)]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.memory_cache.keys(),
                         key=lambda k: self.memory_cache[k].timestamp)
            entry = self.memory_cache.pop(key)
        else:
            return False
        
        self.current_memory_size -= entry.size_bytes
        self.stats['evictions'] += 1
        logger.debug(f"Evicted {key} from memory cache")
        return True
    
    def _remove_from_memory(self, key: str):
        """Remove specific entry from memory cache."""
        if self.memory_cache is not None and key in self.memory_cache:
            entry = self.memory_cache.pop(key)
            self.current_memory_size -= entry.size_bytes
    
    def _check_disk_size(self):
        """Check and manage disk cache size."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('SELECT SUM(size_bytes) FROM cache')
            total_size = cursor.fetchone()[0] or 0
            
            if total_size > self.max_disk_size:
                # Evict based on policy
                if self.eviction_policy == "lru":
                    evict_query = '''
                        DELETE FROM cache 
                        WHERE key IN (
                            SELECT key FROM cache 
                            ORDER BY last_accessed ASC 
                            LIMIT ?
                        )
                    '''
                elif self.eviction_policy == "lfu":
                    evict_query = '''
                        DELETE FROM cache 
                        WHERE key IN (
                            SELECT key FROM cache 
                            ORDER BY access_count ASC 
                            LIMIT ?
                        )
                    '''
                else:  # ttl or default
                    evict_query = '''
                        DELETE FROM cache 
                        WHERE key IN (
                            SELECT key FROM cache 
                            ORDER BY timestamp ASC 
                            LIMIT ?
                        )
                    '''
                
                # Evict 10% of entries
                entries_to_evict = max(1, int(0.1 * conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0]))
                conn.execute(evict_query, (entries_to_evict,))
                conn.commit()
                self.stats['evictions'] += entries_to_evict
                logger.info(f"Evicted {entries_to_evict} entries from disk cache")
            
            conn.close()
        except Exception as e:
            logger.error(f"Error checking disk size: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from all caches."""
        with self.lock:
            success = False
            
            # Remove from memory
            if self.memory_cache is not None and key in self.memory_cache:
                self._remove_from_memory(key)
                success = True
            
            # Remove from Redis
            if self.redis_client:
                try:
                    self.redis_client.delete(f"rag:{key}", f"rag:meta:{key}")
                    success = True
                except Exception as e:
                    logger.error(f"Error deleting from Redis: {e}")
            
            # Remove from disk
            if self.cache_type in ["disk", "hybrid"]:
                try:
                    conn = sqlite3.connect(self.db_path)
                    conn.execute('DELETE FROM cache WHERE key = ?', (key,))
                    conn.commit()
                    conn.close()
                    success = True
                except Exception as e:
                    logger.error(f"Error deleting from disk: {e}")
            
            return success
    
    def clear(self) -> bool:
        """Clear all caches."""
        with self.lock:
            success = True
            
            # Clear memory cache
            if self.memory_cache is not None:
                self.memory_cache.clear()
                self.current_memory_size = 0
            
            # Clear Redis
            if self.redis_client:
                try:
                    keys = self.redis_client.keys("rag:*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Error clearing Redis: {e}")
                    success = False
            
            # Clear disk cache
            if self.cache_type in ["disk", "hybrid"]:
                try:
                    conn = sqlite3.connect(self.db_path)
                    conn.execute('DELETE FROM cache')
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error clearing disk cache: {e}")
                    success = False
            
            # Reset statistics
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_requests': 0
            }
            
            return success
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate hit rate
            if stats['total_requests'] > 0:
                stats['hit_rate'] = stats['hits'] / stats['total_requests']
            else:
                stats['hit_rate'] = 0.0
            
            # Memory cache stats
            if self.memory_cache is not None:
                stats['memory_entries'] = len(self.memory_cache)
                stats['memory_size_mb'] = self.current_memory_size / (1024 * 1024)
            
            # Disk cache stats
            if self.cache_type in ["disk", "hybrid"]:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.execute('SELECT COUNT(*), SUM(size_bytes) FROM cache')
                    count, size = cursor.fetchone()
                    stats['disk_entries'] = count or 0
                    stats['disk_size_mb'] = (size or 0) / (1024 * 1024)
                    conn.close()
                except Exception as e:
                    logger.error(f"Error getting disk stats: {e}")
            
            # Redis stats
            if self.redis_client:
                try:
                    stats['redis_entries'] = len(self.redis_client.keys("rag:*"))
                except Exception as e:
                    logger.error(f"Error getting Redis stats: {e}")
            
            return stats
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from all caches."""
        with self.lock:
            cleaned = 0
            
            # Clean memory cache
            if self.memory_cache is not None:
                expired_keys = [k for k, v in self.memory_cache.items() if self._is_expired(v)]
                for key in expired_keys:
                    self._remove_from_memory(key)
                    cleaned += 1
            
            # Clean disk cache
            if self.cache_type in ["disk", "hybrid"]:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.execute('''
                        DELETE FROM cache 
                        WHERE ttl IS NOT NULL 
                        AND (? - timestamp) > ttl
                    ''', (time.time(),))
                    cleaned += cursor.rowcount
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error cleaning disk cache: {e}")
            
            logger.info(f"Cleaned up {cleaned} expired entries")
            return cleaned


class CachedFunction:
    """Decorator for caching function results."""
    
    def __init__(self, cache_manager: CacheManager, ttl: Optional[int] = None):
        self.cache_manager = cache_manager
        self.ttl = ttl
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.cache_manager._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            self.cache_manager.set(cache_key, result, self.ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


class RAGCacheManager(CacheManager):
    """Specialized cache manager for RAG pipeline with domain-specific methods."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Specific cache namespaces
        self.namespaces = {
            'embeddings': 'emb',
            'queries': 'qry',
            'chunks': 'chk',
            'answers': 'ans',
            'documents': 'doc'
        }
    
    def cache_embedding(self, text: str, embedding: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache text embedding."""
        key = self._generate_key(self.namespaces['embeddings'], text)
        return self.set(key, embedding.tolist(), ttl or 86400)  # 24 hours default
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding."""
        key = self._generate_key(self.namespaces['embeddings'], text)
        result = self.get(key)
        return np.array(result) if result is not None else None
    
    def cache_query_results(self, 
                          query: str, 
                          results: List[Dict], 
                          ttl: Optional[int] = None) -> bool:
        """Cache query retrieval results."""
        key = self._generate_key(self.namespaces['queries'], query)
        return self.set(key, results, ttl or 3600)  # 1 hour default
    
    def get_query_results(self, query: str) -> Optional[List[Dict]]:
        """Retrieve cached query results."""
        key = self._generate_key(self.namespaces['queries'], query)
        return self.get(key)
    
    def cache_answer(self, 
                    query: str, 
                    context: List[str], 
                    answer: str, 
                    ttl: Optional[int] = None) -> bool:
        """Cache generated answer."""
        key = self._generate_key(self.namespaces['answers'], query, context)
        return self.set(key, answer, ttl or 1800)  # 30 minutes default
    
    def get_answer(self, query: str, context: List[str]) -> Optional[str]:
        """Retrieve cached answer."""
        key = self._generate_key(self.namespaces['answers'], query, context)
        return self.get(key)
    
    def cache_document_chunks(self, 
                            doc_id: str, 
                            chunks: List[Dict], 
                            ttl: Optional[int] = None) -> bool:
        """Cache document chunks."""
        key = self._generate_key(self.namespaces['chunks'], doc_id)
        return self.set(key, chunks, ttl or 86400)  # 24 hours default
    
    def get_document_chunks(self, doc_id: str) -> Optional[List[Dict]]:
        """Retrieve cached document chunks."""
        key = self._generate_key(self.namespaces['chunks'], doc_id)
        return self.get(key)
    
    def warm_cache(self, 
                  common_queries: List[str], 
                  collection,
                  embedding_function) -> Dict[str, int]:
        """Pre-warm cache with common queries."""
        warmed = {
            'embeddings': 0,
            'queries': 0
        }
        
        for query in common_queries:
            # Cache query embedding
            if self.get_embedding(query) is None:
                embedding = embedding_function([query])[0]
                if self.cache_embedding(query, np.array(embedding)):
                    warmed['embeddings'] += 1
            
            # Cache query results
            if self.get_query_results(query) is None:
                results = collection.query(
                    query_texts=[query],
                    n_results=10,
                    include=['documents', 'metadatas', 'distances']
                )
                if self.cache_query_results(query, results):
                    warmed['queries'] += 1
        
        logger.info(f"Warmed cache with {warmed['embeddings']} embeddings and {warmed['queries']} query results")
        return warmed
    
    def invalidate_document(self, doc_id: str) -> bool:
        """Invalidate all caches related to a document."""
        patterns = [
            self._generate_key(self.namespaces['chunks'], doc_id),
            # Add more patterns as needed
        ]
        
        success = True
        for pattern in patterns:
            success &= self.delete(pattern)
        
        return success
    
    def get_namespace_stats(self) -> Dict[str, Dict]:
        """Get statistics broken down by namespace."""
        namespace_stats = {}
        
        for name, prefix in self.namespaces.items():
            # This is simplified - in practice, you'd track this more granularly
            namespace_stats[name] = {
                'entries': 0,
                'size_mb': 0
            }
        
        return namespace_stats


class CacheMonitor:
    """Monitor cache performance and generate recommendations."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.metrics_history = []
        self.monitoring_interval = 60  # seconds
    
    def collect_metrics(self):
        """Collect current cache metrics."""
        metrics = {
            'timestamp': time.time(),
            'stats': self.cache_manager.get_stats()
        }
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of metrics
        cutoff_time = time.time() - 86400
        self.metrics_history = [m for m in self.metrics_history 
                               if m['timestamp'] > cutoff_time]
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and generate insights."""
        if len(self.metrics_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_stats = self.metrics_history[-1]['stats']
        
        # Calculate trends
        hit_rates = [m['stats'].get('hit_rate', 0) for m in self.metrics_history]
        avg_hit_rate = np.mean(hit_rates)
        hit_rate_trend = np.polyfit(range(len(hit_rates)), hit_rates, 1)[0] if len(hit_rates) > 1 else 0
        
        analysis = {
            'current_hit_rate': recent_stats.get('hit_rate', 0),
            'avg_hit_rate_24h': avg_hit_rate,
            'hit_rate_trend': 'improving' if hit_rate_trend > 0 else 'declining',
            'memory_utilization': recent_stats.get('memory_size_mb', 0) / (self.cache_manager.max_memory_size / 1024 / 1024),
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['current_hit_rate'] < 0.5:
            analysis['recommendations'].append(
                "Low hit rate detected. Consider increasing cache size or adjusting TTL values."
            )
        
        if analysis['memory_utilization'] > 0.9:
            analysis['recommendations'].append(
                "Memory cache nearly full. Consider increasing memory limit or more aggressive eviction."
            )
        
        if recent_stats.get('evictions', 0) > recent_stats.get('hits', 0):
            analysis['recommendations'].append(
                "High eviction rate. Cache may be undersized for workload."
            )
        
        return analysis
    
    def generate_report(self, output_file: str = "cache_report.json"):
        """Generate comprehensive cache performance report."""
        analysis = self.analyze_performance()
        stats = self.cache_manager.get_stats()
        
        report = {
            'report_time': datetime.now().isoformat(),
            'cache_configuration': {
                'type': self.cache_manager.cache_type,
                'max_memory_mb': self.cache_manager.max_memory_size / 1024 / 1024,
                'eviction_policy': self.cache_manager.eviction_policy,
                'compression': self.cache_manager.compression
            },
            'current_stats': stats,
            'performance_analysis': analysis,
            'metrics_history_summary': {
                'data_points': len(self.metrics_history),
                'time_span_hours': (self.metrics_history[-1]['timestamp'] - 
                                  self.metrics_history[0]['timestamp']) / 3600 if self.metrics_history else 0
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cache performance report saved to {output_file}")
        return report


# Example usage
if __name__ == "__main__":
    # Create a hybrid cache manager for RAG
    cache = RAGCacheManager(
        cache_type="hybrid",
        max_memory_size_mb=500,
        max_disk_size_mb=5000,
        eviction_policy="lru",
        compression=True,
        default_ttl=3600
    )
    
    # Example: Cache embedding
    sample_text = "What are the requirements for extraordinary ability?"
    sample_embedding = np.random.rand(384)  # Simulated embedding
    
    cache.cache_embedding(sample_text, sample_embedding)
    retrieved_embedding = cache.get_embedding(sample_text)
    print(f"Embedding cached and retrieved: {retrieved_embedding is not None}")
    
    # Example: Cache query results
    sample_results = [
        {'doc_id': 'doc1', 'score': 0.95},
        {'doc_id': 'doc2', 'score': 0.87}
    ]
    cache.cache_query_results(sample_text, sample_results)
    
    # Create cached function decorator
    @CachedFunction(cache, ttl=300)
    def expensive_computation(x, y):
        """Simulate expensive computation."""
        time.sleep(1)  # Simulate work
        return x * y + x ** y
    
    # First call - will be slow
    start = time.time()
    result1 = expensive_computation(5, 3)
    print(f"First call took: {time.time() - start:.2f}s, result: {result1}")
    
    # Second call - should be fast (cached)
    start = time.time()
    result2 = expensive_computation(5, 3)
    print(f"Second call took: {time.time() - start:.2f}s, result: {result2}")
    
    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Memory entries: {stats.get('memory_entries', 0)}")
    print(f"Memory size: {stats.get('memory_size_mb', 0):.2f} MB")
    
    # Create monitor and analyze
    monitor = CacheMonitor(cache)
    monitor.collect_metrics()
    analysis = monitor.analyze_performance()
    print(f"\nPerformance Analysis:")
    print(f"Current hit rate: {analysis.get('current_hit_rate', 0):.2%}")
    print(f"Recommendations: {analysis.get('recommendations', [])}")
    
    # Clean up
    cleaned = cache.cleanup_expired()
    print(f"\nCleaned up {cleaned} expired entries")
    
    # Clear cache
    cache.clear()
    print("Cache cleared")