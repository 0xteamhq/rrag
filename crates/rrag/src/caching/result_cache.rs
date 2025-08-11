//! # Result Cache Implementation
//!
//! High-performance caching for search results with compression.

use super::{
    Cache, CacheEntryMetadata, CacheStats, CachedSearchResult, ResultCacheConfig, ResultCacheEntry,
};
use crate::RragResult;
use std::collections::HashMap;

/// Result cache optimized for search results
pub struct ResultCache {
    /// Configuration
    config: ResultCacheConfig,

    /// Main storage
    storage: HashMap<String, ResultCacheEntry>,

    /// Compressed storage for large results
    compressed_storage: HashMap<String, CompressedResults>,

    /// Parameter hash index for fast lookups
    param_index: HashMap<String, Vec<String>>,

    /// Cache statistics
    stats: CacheStats,
}

/// Compressed search results
#[derive(Debug, Clone)]
pub struct CompressedResults {
    /// Compressed data
    pub data: Vec<u8>,

    /// Original size
    pub original_size: usize,

    /// Compression method used
    pub method: CompressionMethod,

    /// Number of results
    pub result_count: usize,
}

/// Compression methods for results
#[derive(Debug, Clone, Copy)]
pub enum CompressionMethod {
    None,
    Gzip,
    Snappy,
    Zstd,
}

impl ResultCache {
    /// Create new result cache
    pub fn new(config: ResultCacheConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            storage: HashMap::new(),
            compressed_storage: HashMap::new(),
            param_index: HashMap::new(),
            stats: CacheStats::default(),
        })
    }

    /// Get cached results by parameters
    pub fn get_by_params(&self, params_hash: &str) -> Option<Vec<CachedSearchResult>> {
        // Try direct lookup
        if let Some(entry) = self.storage.get(params_hash) {
            if !entry.metadata.is_expired() {
                return Some(entry.results.clone());
            }
        }

        // Try compressed storage
        if let Some(compressed) = self.compressed_storage.get(params_hash) {
            return Some(self.decompress_results(compressed));
        }

        None
    }

    /// Cache search results
    pub fn cache_results(
        &mut self,
        params_hash: String,
        results: Vec<CachedSearchResult>,
        metadata: HashMap<String, String>,
    ) -> RragResult<()> {
        // Check capacity
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }

        // Check if results should be compressed
        let should_compress = self.config.compress_large_results && results.len() > 100;

        if should_compress {
            let compressed = self.compress_results(&results);
            self.compressed_storage
                .insert(params_hash.clone(), compressed);
        } else {
            let mut entry_metadata = CacheEntryMetadata::new();
            entry_metadata.ttl = Some(self.config.ttl);

            let entry = ResultCacheEntry {
                params_hash: params_hash.clone(),
                results,
                result_metadata: metadata,
                metadata: entry_metadata,
            };

            self.storage.insert(params_hash.clone(), entry);
        }

        // Update parameter index
        self.update_param_index(&params_hash);

        Ok(())
    }

    /// Compress results
    fn compress_results(&self, results: &[CachedSearchResult]) -> CompressedResults {
        // Serialize results
        let serialized = bincode::serialize(results).unwrap_or_default();
        let original_size = serialized.len();

        // For now, just store serialized data (real implementation would use compression)
        CompressedResults {
            data: serialized,
            original_size,
            method: CompressionMethod::None,
            result_count: results.len(),
        }
    }

    /// Decompress results
    fn decompress_results(&self, compressed: &CompressedResults) -> Vec<CachedSearchResult> {
        // Deserialize results
        bincode::deserialize(&compressed.data).unwrap_or_default()
    }

    /// Update parameter index
    fn update_param_index(&mut self, params_hash: &str) {
        // Extract parameter components (simplified)
        let components = self.extract_param_components(params_hash);

        for component in components {
            self.param_index
                .entry(component)
                .or_insert_with(Vec::new)
                .push(params_hash.to_string());
        }
    }

    /// Extract parameter components for indexing
    fn extract_param_components(&self, params_hash: &str) -> Vec<String> {
        // Simplified: split hash into chunks for indexing
        let mut components = Vec::new();

        if params_hash.len() >= 8 {
            components.push(params_hash[0..4].to_string());
            components.push(params_hash[4..8].to_string());
        }

        components
    }

    /// Invalidate cache entries by pattern
    pub fn invalidate_pattern(&mut self, pattern: &str) {
        let keys_to_remove: Vec<String> = self
            .storage
            .keys()
            .filter(|k| k.contains(pattern))
            .cloned()
            .collect();

        for key in keys_to_remove {
            self.storage.remove(&key);
            self.compressed_storage.remove(&key);
        }
    }

    /// Evict entry based on policy
    fn evict_entry(&mut self) -> RragResult<()> {
        use super::EvictionPolicy;

        match self.config.eviction_policy {
            EvictionPolicy::TTL => self.evict_expired(),
            EvictionPolicy::LRU => self.evict_lru(),
            _ => self.evict_lru(),
        }
    }

    /// Evict expired entries
    fn evict_expired(&mut self) -> RragResult<()> {
        let before_count = self.storage.len();
        self.storage.retain(|_, entry| !entry.metadata.is_expired());

        let evicted = before_count - self.storage.len();
        self.stats.evictions += evicted as u64;

        // If still over capacity, evict oldest
        if self.storage.len() >= self.config.max_size {
            self.evict_lru()?;
        }

        Ok(())
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) -> RragResult<()> {
        if let Some((key, _)) = self
            .storage
            .iter()
            .min_by_key(|(_, entry)| entry.metadata.last_accessed)
        {
            let key = key.clone();
            self.storage.remove(&key);
            self.compressed_storage.remove(&key);
            self.stats.evictions += 1;
        }
        Ok(())
    }

    /// Get cache insights
    pub fn get_insights(&self) -> ResultCacheInsights {
        let total_entries = self.storage.len() + self.compressed_storage.len();
        let compressed_entries = self.compressed_storage.len();

        let avg_results_per_entry = if !self.storage.is_empty() {
            self.storage
                .values()
                .map(|e| e.results.len())
                .sum::<usize>() as f32
                / self.storage.len() as f32
        } else {
            0.0
        };

        let compression_ratio = if !self.compressed_storage.is_empty() {
            let total_original: usize = self
                .compressed_storage
                .values()
                .map(|c| c.original_size)
                .sum();
            let total_compressed: usize =
                self.compressed_storage.values().map(|c| c.data.len()).sum();

            if total_compressed > 0 {
                total_original as f32 / total_compressed as f32
            } else {
                1.0
            }
        } else {
            1.0
        };

        ResultCacheInsights {
            total_entries,
            compressed_entries,
            avg_results_per_entry,
            compression_ratio,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> usize {
        let storage_size: usize = self
            .storage
            .values()
            .map(|e| {
                std::mem::size_of::<ResultCacheEntry>()
                    + e.results.len() * std::mem::size_of::<CachedSearchResult>()
            })
            .sum();

        let compressed_size: usize = self
            .compressed_storage
            .values()
            .map(|c| std::mem::size_of::<CompressedResults>() + c.data.len())
            .sum();

        storage_size + compressed_size
    }
}

impl Cache<String, ResultCacheEntry> for ResultCache {
    fn get(&self, key: &String) -> Option<ResultCacheEntry> {
        self.storage.get(key).cloned()
    }

    fn put(&mut self, key: String, value: ResultCacheEntry) -> RragResult<()> {
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }

        self.storage.insert(key, value);
        Ok(())
    }

    fn remove(&mut self, key: &String) -> Option<ResultCacheEntry> {
        self.compressed_storage.remove(key);
        self.storage.remove(key)
    }

    fn contains(&self, key: &String) -> bool {
        self.storage.contains_key(key) || self.compressed_storage.contains_key(key)
    }

    fn clear(&mut self) {
        self.storage.clear();
        self.compressed_storage.clear();
        self.param_index.clear();
        self.stats = CacheStats::default();
    }

    fn size(&self) -> usize {
        self.storage.len() + self.compressed_storage.len()
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

/// Result cache insights
#[derive(Debug, Clone)]
pub struct ResultCacheInsights {
    /// Total cache entries
    pub total_entries: usize,

    /// Number of compressed entries
    pub compressed_entries: usize,

    /// Average results per entry
    pub avg_results_per_entry: f32,

    /// Compression ratio achieved
    pub compression_ratio: f32,

    /// Estimated memory usage in bytes
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_config() -> ResultCacheConfig {
        ResultCacheConfig {
            enabled: true,
            max_size: 100,
            ttl: Duration::from_secs(3600),
            eviction_policy: super::super::EvictionPolicy::TTL,
            compress_large_results: true,
        }
    }

    fn create_test_results(count: usize) -> Vec<CachedSearchResult> {
        (0..count)
            .map(|i| CachedSearchResult {
                document_id: format!("doc{}", i),
                content: format!("content {}", i),
                score: 0.9 - (i as f32 * 0.01),
                rank: i,
                metadata: HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_result_cache_creation() {
        let config = create_test_config();
        let cache = ResultCache::new(config).unwrap();

        assert_eq!(cache.size(), 0);
        assert_eq!(cache.param_index.len(), 0);
    }

    #[test]
    fn test_basic_caching() {
        let config = create_test_config();
        let mut cache = ResultCache::new(config).unwrap();

        let params_hash = "hash123".to_string();
        let results = create_test_results(5);
        let metadata = HashMap::new();

        cache
            .cache_results(params_hash.clone(), results.clone(), metadata)
            .unwrap();

        let cached = cache.get_by_params(&params_hash);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 5);
    }

    #[test]
    fn test_compression() {
        let config = create_test_config();
        let mut cache = ResultCache::new(config).unwrap();

        let params_hash = "hash_large".to_string();
        let results = create_test_results(150); // Should trigger compression
        let metadata = HashMap::new();

        cache
            .cache_results(params_hash.clone(), results.clone(), metadata)
            .unwrap();

        // Should be in compressed storage
        assert!(cache.compressed_storage.contains_key(&params_hash));
        assert!(!cache.storage.contains_key(&params_hash));

        // Should still be retrievable
        let cached = cache.get_by_params(&params_hash);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 150);
    }

    #[test]
    fn test_invalidation() {
        let config = create_test_config();
        let mut cache = ResultCache::new(config).unwrap();

        let results = create_test_results(5);
        let metadata = HashMap::new();

        cache
            .cache_results("user_123".to_string(), results.clone(), metadata.clone())
            .unwrap();
        cache
            .cache_results("user_456".to_string(), results.clone(), metadata.clone())
            .unwrap();
        cache
            .cache_results("product_789".to_string(), results.clone(), metadata)
            .unwrap();

        assert_eq!(cache.size(), 3);

        // Invalidate all user-related entries
        cache.invalidate_pattern("user_");

        assert_eq!(cache.size(), 1);
        assert!(cache.get_by_params("product_789").is_some());
        assert!(cache.get_by_params("user_123").is_none());
    }

    #[test]
    fn test_insights() {
        let config = create_test_config();
        let mut cache = ResultCache::new(config).unwrap();

        let results_small = create_test_results(10);
        let results_large = create_test_results(150);
        let metadata = HashMap::new();

        cache
            .cache_results("small".to_string(), results_small, metadata.clone())
            .unwrap();
        cache
            .cache_results("large".to_string(), results_large, metadata)
            .unwrap();

        let insights = cache.get_insights();
        assert_eq!(insights.total_entries, 2);
        assert_eq!(insights.compressed_entries, 1);
        assert!(insights.memory_usage > 0);
    }
}
