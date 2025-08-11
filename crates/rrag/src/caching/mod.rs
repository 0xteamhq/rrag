//! # Intelligent Caching Layer
//! 
//! Multi-level caching system designed for RAG applications with semantic awareness,
//! intelligent eviction policies, and performance optimization features.

pub mod cache_core;
pub mod semantic_cache;
pub mod embedding_cache;
pub mod query_cache;
pub mod result_cache;
pub mod policies;
pub mod metrics;
pub mod persistence;

use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, SystemTime};

/// Main caching service orchestrating multiple cache layers
pub struct CacheService {
    /// Query result cache
    query_cache: Box<dyn Cache<String, QueryCacheEntry>>,
    
    /// Embedding cache for reusing computations
    embedding_cache: Box<dyn Cache<String, EmbeddingCacheEntry>>,
    
    /// Semantic similarity cache
    semantic_cache: Box<dyn Cache<String, SemanticCacheEntry>>,
    
    /// Document retrieval cache
    result_cache: Box<dyn Cache<String, ResultCacheEntry>>,
    
    /// Cache configuration
    config: CacheConfig,
    
    /// Performance metrics
    metrics: CacheMetrics,
}

/// Global cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Enable/disable caching globally
    pub enabled: bool,
    
    /// Query cache configuration
    pub query_cache: QueryCacheConfig,
    
    /// Embedding cache configuration
    pub embedding_cache: EmbeddingCacheConfig,
    
    /// Semantic cache configuration
    pub semantic_cache: SemanticCacheConfig,
    
    /// Result cache configuration
    pub result_cache: ResultCacheConfig,
    
    /// Persistence configuration
    pub persistence: PersistenceConfig,
    
    /// Performance tuning
    pub performance: PerformanceConfig,
}

/// Query cache configuration
#[derive(Debug, Clone)]
pub struct QueryCacheConfig {
    pub enabled: bool,
    pub max_size: usize,
    pub ttl: Duration,
    pub eviction_policy: EvictionPolicy,
    pub similarity_threshold: f32,
}

/// Embedding cache configuration
#[derive(Debug, Clone)]
pub struct EmbeddingCacheConfig {
    pub enabled: bool,
    pub max_size: usize,
    pub ttl: Duration,
    pub eviction_policy: EvictionPolicy,
    pub compression_enabled: bool,
}

/// Semantic cache configuration
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    pub enabled: bool,
    pub max_size: usize,
    pub ttl: Duration,
    pub similarity_threshold: f32,
    pub clustering_enabled: bool,
    pub max_clusters: usize,
}

/// Result cache configuration
#[derive(Debug, Clone)]
pub struct ResultCacheConfig {
    pub enabled: bool,
    pub max_size: usize,
    pub ttl: Duration,
    pub eviction_policy: EvictionPolicy,
    pub compress_large_results: bool,
}

/// Persistence configuration
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub storage_path: String,
    pub auto_save_interval: Duration,
    pub format: PersistenceFormat,
}

/// Performance configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub async_writes: bool,
    pub batch_operations: bool,
    pub background_cleanup: bool,
    pub memory_pressure_threshold: f32,
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-To-Live based
    TTL,
    /// Adaptive Replacement Cache
    ARC,
    /// Custom semantic-aware policy
    SemanticAware,
}

/// Persistence formats
#[derive(Debug, Clone)]
pub enum PersistenceFormat {
    Binary,
    Json,
    MessagePack,
}

/// Generic cache trait
pub trait Cache<K, V>: Send + Sync
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Get value from cache
    fn get(&self, key: &K) -> Option<V>;
    
    /// Put value into cache
    fn put(&mut self, key: K, value: V) -> RragResult<()>;
    
    /// Remove value from cache
    fn remove(&mut self, key: &K) -> Option<V>;
    
    /// Check if key exists
    fn contains(&self, key: &K) -> bool;
    
    /// Clear all entries
    fn clear(&mut self);
    
    /// Get cache size
    fn size(&self) -> usize;
    
    /// Get cache statistics
    fn stats(&self) -> CacheStats;
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntryMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last access timestamp
    pub last_accessed: SystemTime,
    
    /// Access count
    pub access_count: u64,
    
    /// Entry size in bytes
    pub size_bytes: usize,
    
    /// Time-to-live
    pub ttl: Option<Duration>,
    
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Query cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheEntry {
    /// Original query
    pub query: String,
    
    /// Query embedding hash
    pub embedding_hash: String,
    
    /// Cached results
    pub results: Vec<CachedSearchResult>,
    
    /// Generation result if any
    pub generated_answer: Option<String>,
    
    /// Metadata
    pub metadata: CacheEntryMetadata,
}

/// Embedding cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheEntry {
    /// Input text
    pub text: String,
    
    /// Text hash for verification
    pub text_hash: String,
    
    /// Computed embedding
    pub embedding: Vec<f32>,
    
    /// Model used for embedding
    pub model: String,
    
    /// Metadata
    pub metadata: CacheEntryMetadata,
}

/// Semantic cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheEntry {
    /// Representative query/text
    pub representative: String,
    
    /// Cluster ID if clustering enabled
    pub cluster_id: Option<usize>,
    
    /// Similar queries/texts
    pub similar_entries: Vec<SimilarEntry>,
    
    /// Cached semantic results
    pub results: Vec<CachedSearchResult>,
    
    /// Metadata
    pub metadata: CacheEntryMetadata,
}

/// Result cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultCacheEntry {
    /// Search parameters hash
    pub params_hash: String,
    
    /// Cached search results
    pub results: Vec<CachedSearchResult>,
    
    /// Result metadata
    pub result_metadata: HashMap<String, String>,
    
    /// Metadata
    pub metadata: CacheEntryMetadata,
}

/// Similar entry for semantic cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarEntry {
    /// Similar text
    pub text: String,
    
    /// Similarity score
    pub similarity: f32,
    
    /// When added
    pub added_at: SystemTime,
}

/// Cached search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSearchResult {
    /// Document ID
    pub document_id: String,
    
    /// Document content
    pub content: String,
    
    /// Relevance score
    pub score: f32,
    
    /// Result rank
    pub rank: usize,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    
    /// Total cache hits
    pub hits: u64,
    
    /// Total cache misses
    pub misses: u64,
    
    /// Hit rate percentage
    pub hit_rate: f32,
    
    /// Total memory usage in bytes
    pub memory_usage: usize,
    
    /// Average access time in microseconds
    pub avg_access_time_us: f32,
    
    /// Eviction count
    pub evictions: u64,
    
    /// Last cleanup time
    pub last_cleanup: SystemTime,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Query cache metrics
    pub query_cache: CacheStats,
    
    /// Embedding cache metrics
    pub embedding_cache: CacheStats,
    
    /// Semantic cache metrics
    pub semantic_cache: CacheStats,
    
    /// Result cache metrics
    pub result_cache: CacheStats,
    
    /// Overall performance metrics
    pub overall: OverallCacheMetrics,
}

/// Overall cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallCacheMetrics {
    /// Total memory saved (bytes)
    pub memory_saved: usize,
    
    /// Total time saved (milliseconds)
    pub time_saved_ms: f32,
    
    /// Cache efficiency score
    pub efficiency_score: f32,
    
    /// Memory pressure level (0.0 to 1.0)
    pub memory_pressure: f32,
    
    /// Total operations per second
    pub ops_per_second: f32,
}

impl CacheService {
    /// Create new cache service
    pub fn new(config: CacheConfig) -> RragResult<Self> {
        let query_cache = Box::new(
            query_cache::QueryCache::new(config.query_cache.clone())?
        );
        
        let embedding_cache = Box::new(
            embedding_cache::EmbeddingCache::new(config.embedding_cache.clone())?
        );
        
        let semantic_cache = Box::new(
            semantic_cache::SemanticCache::new(config.semantic_cache.clone())?
        );
        
        let result_cache = Box::new(
            result_cache::ResultCache::new(config.result_cache.clone())?
        );
        
        Ok(Self {
            query_cache,
            embedding_cache,
            semantic_cache,
            result_cache,
            config,
            metrics: CacheMetrics::default(),
        })
    }
    
    /// Get cached query results
    pub async fn get_query_results(&self, query: &str) -> Option<QueryCacheEntry> {
        if !self.config.enabled || !self.config.query_cache.enabled {
            return None;
        }
        
        self.query_cache.get(&query.to_string())
    }
    
    /// Cache query results
    pub async fn cache_query_results(
        &mut self, 
        query: String, 
        entry: QueryCacheEntry
    ) -> RragResult<()> {
        if !self.config.enabled || !self.config.query_cache.enabled {
            return Ok(());
        }
        
        self.query_cache.put(query, entry)
    }
    
    /// Get cached embedding
    pub async fn get_embedding(&self, text: &str, model: &str) -> Option<Vec<f32>> {
        if !self.config.enabled || !self.config.embedding_cache.enabled {
            return None;
        }
        
        let key = format!("{}:{}", model, text);
        self.embedding_cache.get(&key).map(|entry| entry.embedding)
    }
    
    /// Cache embedding
    pub async fn cache_embedding(
        &mut self,
        text: String,
        model: String,
        embedding: Vec<f32>
    ) -> RragResult<()> {
        if !self.config.enabled || !self.config.embedding_cache.enabled {
            return Ok(());
        }
        
        let key = format!("{}:{}", model, text);
        let entry = EmbeddingCacheEntry {
            text: text.clone(),
            text_hash: Self::hash_string(&text),
            embedding,
            model,
            metadata: CacheEntryMetadata::new(),
        };
        
        self.embedding_cache.put(key, entry)
    }
    
    /// Get semantically similar cached results
    pub async fn get_semantic_results(&self, query: &str) -> Option<SemanticCacheEntry> {
        if !self.config.enabled || !self.config.semantic_cache.enabled {
            return None;
        }
        
        self.semantic_cache.get(&query.to_string())
    }
    
    /// Cache semantic results
    pub async fn cache_semantic_results(
        &mut self,
        query: String,
        entry: SemanticCacheEntry
    ) -> RragResult<()> {
        if !self.config.enabled || !self.config.semantic_cache.enabled {
            return Ok(());
        }
        
        self.semantic_cache.put(query, entry)
    }
    
    /// Get cache metrics
    pub fn get_metrics(&self) -> &CacheMetrics {
        &self.metrics
    }
    
    /// Clear all caches
    pub fn clear_all(&mut self) {
        self.query_cache.clear();
        self.embedding_cache.clear();
        self.semantic_cache.clear();
        self.result_cache.clear();
    }
    
    /// Perform cache maintenance
    pub async fn maintenance(&mut self) -> RragResult<()> {
        // Background cleanup, eviction, persistence, etc.
        Ok(())
    }
    
    /// Hash string for cache keys
    fn hash_string(s: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl CacheEntryMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            last_accessed: now,
            access_count: 0,
            size_bytes: 0,
            ttl: None,
            custom: HashMap::new(),
        }
    }
    
    /// Update access info
    pub fn accessed(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }
    
    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            if let Ok(elapsed) = self.created_at.elapsed() {
                return elapsed > ttl;
            }
        }
        false
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            query_cache: QueryCacheConfig::default(),
            embedding_cache: EmbeddingCacheConfig::default(),
            semantic_cache: SemanticCacheConfig::default(),
            result_cache: ResultCacheConfig::default(),
            persistence: PersistenceConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1000,
            ttl: Duration::from_secs(3600), // 1 hour
            eviction_policy: EvictionPolicy::LRU,
            similarity_threshold: 0.95,
        }
    }
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 10000,
            ttl: Duration::from_secs(86400), // 24 hours
            eviction_policy: EvictionPolicy::LFU,
            compression_enabled: true,
        }
    }
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 5000,
            ttl: Duration::from_secs(7200), // 2 hours
            similarity_threshold: 0.85,
            clustering_enabled: true,
            max_clusters: 100,
        }
    }
}

impl Default for ResultCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 2000,
            ttl: Duration::from_secs(1800), // 30 minutes
            eviction_policy: EvictionPolicy::TTL,
            compress_large_results: true,
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            storage_path: "./cache".to_string(),
            auto_save_interval: Duration::from_secs(300), // 5 minutes
            format: PersistenceFormat::Binary,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            async_writes: true,
            batch_operations: true,
            background_cleanup: true,
            memory_pressure_threshold: 0.8,
        }
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            memory_usage: 0,
            avg_access_time_us: 0.0,
            evictions: 0,
            last_cleanup: SystemTime::now(),
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            query_cache: CacheStats::default(),
            embedding_cache: CacheStats::default(),
            semantic_cache: CacheStats::default(),
            result_cache: CacheStats::default(),
            overall: OverallCacheMetrics::default(),
        }
    }
}

impl Default for OverallCacheMetrics {
    fn default() -> Self {
        Self {
            memory_saved: 0,
            time_saved_ms: 0.0,
            efficiency_score: 0.0,
            memory_pressure: 0.0,
            ops_per_second: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cache_service_creation() {
        let config = CacheConfig::default();
        let cache_service = CacheService::new(config).unwrap();
        
        let metrics = cache_service.get_metrics();
        assert_eq!(metrics.overall.efficiency_score, 0.0);
    }
    
    #[test]
    fn test_cache_entry_metadata() {
        let mut metadata = CacheEntryMetadata::new();
        assert_eq!(metadata.access_count, 0);
        
        metadata.accessed();
        assert_eq!(metadata.access_count, 1);
    }
    
    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert!(config.enabled);
        assert!(config.query_cache.enabled);
        assert!(config.embedding_cache.enabled);
    }
}