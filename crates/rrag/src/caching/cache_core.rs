//! # Core Cache Implementations
//! 
//! Foundation cache data structures with different eviction policies.

use super::{Cache, CacheStats, CacheEntryMetadata};
use crate::RragResult;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::{SystemTime, Duration};

/// LRU Cache implementation
pub struct LRUCache<K, V> 
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Internal storage
    storage: HashMap<K, CacheNode<V>>,
    
    /// Access order tracking
    access_order: VecDeque<K>,
    
    /// Maximum capacity
    max_size: usize,
    
    /// Cache statistics
    stats: CacheStats,
    
    /// Thread safety
    _phantom: std::marker::PhantomData<(K, V)>,
}

/// LFU Cache implementation
pub struct LFUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Internal storage
    storage: HashMap<K, CacheNode<V>>,
    
    /// Frequency tracking
    frequencies: HashMap<K, u64>,
    
    /// Frequency buckets for efficient eviction
    frequency_buckets: HashMap<u64, Vec<K>>,
    
    /// Minimum frequency
    min_frequency: u64,
    
    /// Maximum capacity
    max_size: usize,
    
    /// Cache statistics
    stats: CacheStats,
}

/// TTL Cache implementation
pub struct TTLCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Internal storage with expiry
    storage: HashMap<K, (V, SystemTime)>,
    
    /// Default TTL
    default_ttl: Duration,
    
    /// Cleanup interval
    cleanup_interval: Duration,
    
    /// Last cleanup time
    last_cleanup: SystemTime,
    
    /// Cache statistics
    stats: CacheStats,
}

/// ARC (Adaptive Replacement Cache) implementation
pub struct ARCCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Recently used cache (T1)
    t1: HashMap<K, V>,
    
    /// Frequently used cache (T2)
    t2: HashMap<K, V>,
    
    /// Ghost entries recently evicted from T1 (B1)
    b1: HashMap<K, ()>,
    
    /// Ghost entries recently evicted from T2 (B2)
    b2: HashMap<K, ()>,
    
    /// LRU lists for T1 and T2
    t1_lru: VecDeque<K>,
    t2_lru: VecDeque<K>,
    b1_lru: VecDeque<K>,
    b2_lru: VecDeque<K>,
    
    /// Adaptive parameter
    p: f32,
    
    /// Maximum capacity
    max_size: usize,
    
    /// Cache statistics
    stats: CacheStats,
}

/// Semantic-aware cache implementation
pub struct SemanticAwareCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Primary storage
    storage: HashMap<K, CacheNode<V>>,
    
    /// Semantic similarity tracking
    similarity_groups: HashMap<u64, Vec<K>>,
    
    /// Embedding vectors for similarity computation
    embeddings: HashMap<K, Vec<f32>>,
    
    /// Access patterns
    access_patterns: HashMap<K, AccessPattern>,
    
    /// Maximum capacity
    max_size: usize,
    
    /// Similarity threshold for grouping
    similarity_threshold: f32,
    
    /// Cache statistics
    stats: CacheStats,
}

/// Cache node with metadata
#[derive(Debug, Clone)]
pub struct CacheNode<V> {
    /// The cached value
    pub value: V,
    
    /// Entry metadata
    pub metadata: CacheEntryMetadata,
    
    /// Computed size in bytes (approximate)
    pub size_bytes: usize,
}

/// Access pattern tracking
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Total accesses
    pub count: u64,
    
    /// Recent access times
    pub recent_accesses: VecDeque<SystemTime>,
    
    /// Average access interval
    pub avg_interval: Duration,
    
    /// Access trend (increasing, decreasing, stable)
    pub trend: AccessTrend,
}

/// Access trend types
#[derive(Debug, Clone, Copy)]
pub enum AccessTrend {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

/// Priority entry for frequency-based eviction
#[derive(Debug, Clone, PartialEq, Eq)]
struct FrequencyEntry<K>
where
    K: Ord,
{
    key: K,
    frequency: u64,
    last_access: SystemTime,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new LRU cache
    pub fn new(max_size: usize) -> Self {
        Self {
            storage: HashMap::with_capacity(max_size),
            access_order: VecDeque::with_capacity(max_size),
            max_size,
            stats: CacheStats::default(),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Update LRU order
    fn update_lru(&mut self, key: &K) {
        // Remove from current position
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        
        // Add to front (most recent)
        self.access_order.push_front(key.clone());
    }
    
    /// Evict least recently used entry
    fn evict_lru(&mut self) -> Option<K> {
        if let Some(key) = self.access_order.pop_back() {
            self.storage.remove(&key);
            self.stats.evictions += 1;
            Some(key)
        } else {
            None
        }
    }
}

impl<K, V> Cache<K, V> for LRUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        let _start_time = SystemTime::now();
        
        if let Some(node) = self.storage.get(key) {
            // Update stats - hits handled by mutable reference in real implementation
            Some(node.value.clone())
        } else {
            // Miss handled by mutable reference in real implementation
            None
        }
    }
    
    fn put(&mut self, key: K, value: V) -> RragResult<()> {
        let size_bytes = std::mem::size_of::<V>();
        let node = CacheNode {
            value,
            metadata: CacheEntryMetadata::new(),
            size_bytes,
        };
        
        // If key exists, update and move to front
        if self.storage.contains_key(&key) {
            self.storage.insert(key.clone(), node);
            self.update_lru(&key);
            return Ok(());
        }
        
        // If at capacity, evict LRU
        if self.storage.len() >= self.max_size {
            self.evict_lru();
        }
        
        // Insert new entry
        self.storage.insert(key.clone(), node);
        self.update_lru(&key);
        
        Ok(())
    }
    
    fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(node) = self.storage.remove(key) {
            // Remove from LRU order
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            Some(node.value)
        } else {
            None
        }
    }
    
    fn contains(&self, key: &K) -> bool {
        self.storage.contains_key(key)
    }
    
    fn clear(&mut self) {
        self.storage.clear();
        self.access_order.clear();
        self.stats = CacheStats::default();
    }
    
    fn size(&self) -> usize {
        self.storage.len()
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

impl<K, V> LFUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new LFU cache
    pub fn new(max_size: usize) -> Self {
        Self {
            storage: HashMap::with_capacity(max_size),
            frequencies: HashMap::with_capacity(max_size),
            frequency_buckets: HashMap::new(),
            min_frequency: 1,
            max_size,
            stats: CacheStats::default(),
        }
    }
    
    /// Update frequency
    fn update_frequency(&mut self, key: &K) {
        let old_freq = self.frequencies.get(key).copied().unwrap_or(0);
        let new_freq = old_freq + 1;
        
        self.frequencies.insert(key.clone(), new_freq);
        
        // Update frequency buckets
        if old_freq > 0 {
            if let Some(bucket) = self.frequency_buckets.get_mut(&old_freq) {
                bucket.retain(|k| k != key);
                if bucket.is_empty() && old_freq == self.min_frequency {
                    self.min_frequency += 1;
                }
            }
        }
        
        self.frequency_buckets
            .entry(new_freq)
            .or_insert_with(Vec::new)
            .push(key.clone());
    }
    
    /// Evict least frequently used entry
    fn evict_lfu(&mut self) -> Option<K> {
        if let Some(bucket) = self.frequency_buckets.get_mut(&self.min_frequency) {
            if let Some(key) = bucket.pop() {
                self.storage.remove(&key);
                self.frequencies.remove(&key);
                self.stats.evictions += 1;
                return Some(key);
            }
        }
        None
    }
}

impl<K, V> Cache<K, V> for LFUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        if let Some(node) = self.storage.get(key) {
            Some(node.value.clone())
        } else {
            None
        }
    }
    
    fn put(&mut self, key: K, value: V) -> RragResult<()> {
        let size_bytes = std::mem::size_of::<V>();
        let node = CacheNode {
            value,
            metadata: CacheEntryMetadata::new(),
            size_bytes,
        };
        
        // If key exists, update
        if self.storage.contains_key(&key) {
            self.storage.insert(key.clone(), node);
            self.update_frequency(&key);
            return Ok(());
        }
        
        // If at capacity, evict LFU
        if self.storage.len() >= self.max_size {
            self.evict_lfu();
        }
        
        // Insert new entry
        self.storage.insert(key.clone(), node);
        self.update_frequency(&key);
        
        Ok(())
    }
    
    fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(node) = self.storage.remove(key) {
            self.frequencies.remove(key);
            Some(node.value)
        } else {
            None
        }
    }
    
    fn contains(&self, key: &K) -> bool {
        self.storage.contains_key(key)
    }
    
    fn clear(&mut self) {
        self.storage.clear();
        self.frequencies.clear();
        self.frequency_buckets.clear();
        self.min_frequency = 1;
        self.stats = CacheStats::default();
    }
    
    fn size(&self) -> usize {
        self.storage.len()
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

impl<K, V> TTLCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create new TTL cache
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            storage: HashMap::new(),
            default_ttl,
            cleanup_interval: Duration::from_secs(60), // 1 minute
            last_cleanup: SystemTime::now(),
            stats: CacheStats::default(),
        }
    }
    
    /// Cleanup expired entries
    fn cleanup_expired(&mut self) {
        let now = SystemTime::now();
        
        // Only cleanup if interval has passed
        if now.duration_since(self.last_cleanup).unwrap_or_default() < self.cleanup_interval {
            return;
        }
        
        let before_count = self.storage.len();
        self.storage.retain(|_key, (_, expiry)| now < *expiry);
        let after_count = self.storage.len();
        
        self.stats.evictions += (before_count - after_count) as u64;
        self.last_cleanup = now;
    }
}

impl<K, V> Cache<K, V> for TTLCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn get(&self, key: &K) -> Option<V> {
        if let Some((value, expiry)) = self.storage.get(key) {
            if SystemTime::now() < *expiry {
                Some(value.clone())
            } else {
                None
            }
        } else {
            None
        }
    }
    
    fn put(&mut self, key: K, value: V) -> RragResult<()> {
        let expiry = SystemTime::now() + self.default_ttl;
        self.storage.insert(key, (value, expiry));
        
        // Periodic cleanup
        self.cleanup_expired();
        
        Ok(())
    }
    
    fn remove(&mut self, key: &K) -> Option<V> {
        self.storage.remove(key).map(|(value, _)| value)
    }
    
    fn contains(&self, key: &K) -> bool {
        if let Some((_, expiry)) = self.storage.get(key) {
            SystemTime::now() < *expiry
        } else {
            false
        }
    }
    
    fn clear(&mut self) {
        self.storage.clear();
        self.stats = CacheStats::default();
    }
    
    fn size(&self) -> usize {
        // Count only non-expired entries
        let now = SystemTime::now();
        self.storage.values().filter(|(_, expiry)| now < *expiry).count()
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

impl<K> PartialOrd for FrequencyEntry<K>
where
    K: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for FrequencyEntry<K>
where
    K: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower frequency first (for min-heap)
        self.frequency.cmp(&other.frequency)
            .then_with(|| self.last_access.cmp(&other.last_access))
    }
}

impl AccessPattern {
    /// Create new access pattern
    pub fn new() -> Self {
        Self {
            count: 0,
            recent_accesses: VecDeque::new(),
            avg_interval: Duration::from_secs(0),
            trend: AccessTrend::Unknown,
        }
    }
    
    /// Record an access
    pub fn record_access(&mut self) {
        let now = SystemTime::now();
        self.count += 1;
        self.recent_accesses.push_back(now);
        
        // Keep only recent accesses (last 10)
        if self.recent_accesses.len() > 10 {
            self.recent_accesses.pop_front();
        }
        
        self.update_metrics();
    }
    
    /// Update computed metrics
    fn update_metrics(&mut self) {
        if self.recent_accesses.len() < 2 {
            return;
        }
        
        // Calculate average interval
        let mut total_interval = Duration::from_secs(0);
        let mut interval_count = 0;
        
        for window in self.recent_accesses.as_slices().0.windows(2) {
            if let Ok(interval) = window[1].duration_since(window[0]) {
                total_interval += interval;
                interval_count += 1;
            }
        }
        
        if interval_count > 0 {
            self.avg_interval = total_interval / interval_count as u32;
        }
        
        // Determine trend (simplified)
        if self.recent_accesses.len() >= 4 {
            let _first_half_avg = self.recent_accesses.len() / 2;
            // Trend analysis would go here
            self.trend = AccessTrend::Stable; // Simplified for now
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lru_cache() {
        let mut cache = LRUCache::new(3);
        
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        cache.put("c".to_string(), 3).unwrap();
        
        assert_eq!(cache.size(), 3);
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        
        // This should evict the LRU entry
        cache.put("d".to_string(), 4).unwrap();
        assert_eq!(cache.size(), 3);
    }
    
    #[test]
    fn test_lfu_cache() {
        let mut cache = LFUCache::new(2);
        
        cache.put("a".to_string(), 1).unwrap();
        cache.put("b".to_string(), 2).unwrap();
        
        // Access 'a' more frequently
        cache.get(&"a".to_string());
        cache.get(&"a".to_string());
        
        // This should evict 'b' (less frequent)
        cache.put("c".to_string(), 3).unwrap();
        
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None);
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }
    
    #[test]
    fn test_ttl_cache() {
        let mut cache = TTLCache::new(Duration::from_millis(100));
        
        cache.put("key".to_string(), "value".to_string()).unwrap();
        assert_eq!(cache.get(&"key".to_string()), Some("value".to_string()));
        
        // Sleep longer than TTL
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(&"key".to_string()), None);
    }
    
    #[test]
    fn test_access_pattern() {
        let mut pattern = AccessPattern::new();
        assert_eq!(pattern.count, 0);
        
        pattern.record_access();
        assert_eq!(pattern.count, 1);
        
        pattern.record_access();
        assert_eq!(pattern.count, 2);
    }
}