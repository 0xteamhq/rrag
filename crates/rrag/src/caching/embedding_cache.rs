//! # Embedding Cache Implementation
//!
//! High-performance caching for embedding computations with compression.

use super::{Cache, CacheEntryMetadata, CacheStats, EmbeddingCacheConfig, EmbeddingCacheEntry};
use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Embedding cache with compression and deduplication
pub struct EmbeddingCache {
    /// Configuration
    config: EmbeddingCacheConfig,

    /// Main storage
    storage: HashMap<String, EmbeddingCacheEntry>,

    /// Text hash to full key mapping for deduplication
    deduplication_map: HashMap<String, String>,

    /// Compressed embeddings storage
    compressed_storage: Option<HashMap<String, CompressedEmbedding>>,

    /// Cache statistics
    stats: CacheStats,
}

/// Compressed embedding representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedEmbedding {
    /// Quantized embedding values
    pub quantized_values: Vec<u8>,

    /// Quantization parameters
    pub scale: f32,
    pub offset: f32,

    /// Original dimension
    pub dimension: usize,

    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// Embedding compression methods
#[derive(Debug, Clone, Copy)]
pub enum CompressionMethod {
    /// No compression
    None,

    /// Simple quantization to 8-bit integers
    Quantization8Bit,

    /// Principal Component Analysis dimensionality reduction
    PCA,

    /// Product quantization
    ProductQuantization,

    /// Binary quantization
    BinaryQuantization,
}

/// Embedding deduplication statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationStats {
    /// Number of deduplicated entries
    pub deduplicated_count: usize,

    /// Memory saved through deduplication (bytes)
    pub memory_saved: usize,

    /// Deduplication ratio (0.0 to 1.0)
    pub deduplication_ratio: f32,
}

impl EmbeddingCache {
    /// Create new embedding cache
    pub fn new(config: EmbeddingCacheConfig) -> RragResult<Self> {
        let compressed_storage = if config.compression_enabled {
            Some(HashMap::new())
        } else {
            None
        };

        Ok(Self {
            config,
            storage: HashMap::new(),
            deduplication_map: HashMap::new(),
            compressed_storage,
            stats: CacheStats::default(),
        })
    }

    /// Get embedding with automatic decompression
    pub fn get_embedding(&self, text: &str, model: &str) -> Option<Vec<f32>> {
        let key = self.make_key(text, model);

        // Try direct lookup
        if let Some(entry) = self.storage.get(&key) {
            return Some(entry.embedding.clone());
        }

        // Try deduplication lookup
        let text_hash = self.hash_text(text);
        if let Some(canonical_key) = self.deduplication_map.get(&text_hash) {
            if let Some(entry) = self.storage.get(canonical_key) {
                return Some(entry.embedding.clone());
            }
        }

        // Try compressed storage
        if let Some(compressed_storage) = &self.compressed_storage {
            if let Some(compressed) = compressed_storage.get(&key) {
                return Some(self.decompress_embedding(compressed));
            }
        }

        None
    }

    /// Cache embedding with compression and deduplication
    pub fn cache_embedding(
        &mut self,
        text: String,
        model: String,
        embedding: Vec<f32>,
    ) -> RragResult<()> {
        let key = self.make_key(&text, &model);
        let text_hash = self.hash_text(&text);

        // Check for deduplication opportunity
        if let Some(existing_key) = self.deduplication_map.get(&text_hash) {
            // Text already cached, just add reference
            if !self.storage.contains_key(&key) {
                if let Some(existing_entry) = self.storage.get(existing_key).cloned() {
                    self.storage.insert(key, existing_entry);
                }
            }
            return Ok(());
        }

        // Check capacity
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }

        // Create entry
        let entry = EmbeddingCacheEntry {
            text: text.clone(),
            text_hash: text_hash.clone(),
            embedding: embedding.clone(),
            model: model.clone(),
            metadata: CacheEntryMetadata::new(),
        };

        // Store with or without compression
        if self.config.compression_enabled {
            let compressed = self.compress_embedding(&embedding);
            if let Some(compressed_storage) = &mut self.compressed_storage {
                compressed_storage.insert(key.clone(), compressed);
            }

            // Store metadata only in main storage
            let mut metadata_entry = entry;
            metadata_entry.embedding = Vec::new(); // Clear to save memory
            self.storage.insert(key.clone(), metadata_entry);
        } else {
            self.storage.insert(key.clone(), entry);
        }

        // Update deduplication map
        self.deduplication_map.insert(text_hash, key);

        Ok(())
    }

    /// Compress embedding using configured method
    fn compress_embedding(&self, embedding: &[f32]) -> CompressedEmbedding {
        // Simple 8-bit quantization for now
        let (min_val, max_val) = embedding
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let range = max_val - min_val;
        let scale = range / 255.0;
        let offset = min_val;

        let quantized_values: Vec<u8> = embedding
            .iter()
            .map(|&val| {
                let normalized = (val - offset) / scale;
                normalized.round().clamp(0.0, 255.0) as u8
            })
            .collect();

        let original_size = embedding.len() * std::mem::size_of::<f32>();
        let compressed_size =
            quantized_values.len() * std::mem::size_of::<u8>() + std::mem::size_of::<f32>() * 2; // scale + offset

        CompressedEmbedding {
            quantized_values,
            scale,
            offset,
            dimension: embedding.len(),
            compression_ratio: original_size as f32 / compressed_size as f32,
        }
    }

    /// Decompress embedding
    fn decompress_embedding(&self, compressed: &CompressedEmbedding) -> Vec<f32> {
        compressed
            .quantized_values
            .iter()
            .map(|&val| (val as f32) * compressed.scale + compressed.offset)
            .collect()
    }

    /// Make cache key
    fn make_key(&self, text: &str, model: &str) -> String {
        format!("{}:{}", model, text)
    }

    /// Hash text for deduplication
    fn hash_text(&self, text: &str) -> String {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Evict least frequently used entry
    fn evict_entry(&mut self) -> RragResult<()> {
        if self.storage.is_empty() {
            return Ok(());
        }

        // Find LFU entry
        let mut candidate_key: Option<String> = None;
        let mut min_access_count = u64::MAX;
        let mut oldest_time = std::time::SystemTime::now();

        for (key, entry) in &self.storage {
            if entry.metadata.access_count < min_access_count
                || (entry.metadata.access_count == min_access_count
                    && entry.metadata.last_accessed < oldest_time)
            {
                min_access_count = entry.metadata.access_count;
                oldest_time = entry.metadata.last_accessed;
                candidate_key = Some(key.clone());
            }
        }

        if let Some(key) = candidate_key {
            if let Some(entry) = self.storage.remove(&key) {
                // Remove from deduplication map
                self.deduplication_map.remove(&entry.text_hash);

                // Remove from compressed storage
                if let Some(compressed_storage) = &mut self.compressed_storage {
                    compressed_storage.remove(&key);
                }

                self.stats.evictions += 1;
            }
        }

        Ok(())
    }

    /// Get deduplication statistics
    pub fn get_deduplication_stats(&self) -> DeduplicationStats {
        let total_entries = self.storage.len();
        let unique_texts = self.deduplication_map.len();
        let deduplicated_count = if total_entries > unique_texts {
            total_entries - unique_texts
        } else {
            0
        };

        let embedding_size = 1536 * std::mem::size_of::<f32>(); // Assume typical size
        let memory_saved = deduplicated_count * embedding_size;

        let deduplication_ratio = if total_entries > 0 {
            deduplicated_count as f32 / total_entries as f32
        } else {
            0.0
        };

        DeduplicationStats {
            deduplicated_count,
            memory_saved,
            deduplication_ratio,
        }
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> Option<CompressionStats> {
        if !self.config.compression_enabled {
            return None;
        }

        let compressed_storage = self.compressed_storage.as_ref()?;

        let mut total_original_size = 0;
        let mut total_compressed_size = 0;
        let mut compression_ratios = Vec::new();

        for compressed in compressed_storage.values() {
            let original_size = compressed.dimension * std::mem::size_of::<f32>();
            let compressed_size =
                compressed.quantized_values.len() + std::mem::size_of::<f32>() * 2; // scale + offset

            total_original_size += original_size;
            total_compressed_size += compressed_size;
            compression_ratios.push(compressed.compression_ratio);
        }

        let overall_ratio = if total_compressed_size > 0 {
            total_original_size as f32 / total_compressed_size as f32
        } else {
            1.0
        };

        let avg_ratio = if !compression_ratios.is_empty() {
            compression_ratios.iter().sum::<f32>() / compression_ratios.len() as f32
        } else {
            1.0
        };

        Some(CompressionStats {
            total_entries: compressed_storage.len(),
            total_original_size,
            total_compressed_size,
            overall_compression_ratio: overall_ratio,
            average_compression_ratio: avg_ratio,
            memory_saved: total_original_size - total_compressed_size,
        })
    }
}

impl Cache<String, EmbeddingCacheEntry> for EmbeddingCache {
    fn get(&self, key: &String) -> Option<EmbeddingCacheEntry> {
        self.storage.get(key).cloned()
    }

    fn put(&mut self, key: String, value: EmbeddingCacheEntry) -> RragResult<()> {
        // Check capacity
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }

        self.storage.insert(key, value);
        Ok(())
    }

    fn remove(&mut self, key: &String) -> Option<EmbeddingCacheEntry> {
        let entry = self.storage.remove(key);

        if let Some(ref entry_val) = entry {
            self.deduplication_map.remove(&entry_val.text_hash);

            if let Some(compressed_storage) = &mut self.compressed_storage {
                compressed_storage.remove(key);
            }
        }

        entry
    }

    fn contains(&self, key: &String) -> bool {
        self.storage.contains_key(key)
            || (self
                .compressed_storage
                .as_ref()
                .map_or(false, |cs| cs.contains_key(key)))
    }

    fn clear(&mut self) {
        self.storage.clear();
        self.deduplication_map.clear();
        if let Some(compressed_storage) = &mut self.compressed_storage {
            compressed_storage.clear();
        }
        self.stats = CacheStats::default();
    }

    fn size(&self) -> usize {
        self.storage.len()
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Number of compressed entries
    pub total_entries: usize,

    /// Total original size in bytes
    pub total_original_size: usize,

    /// Total compressed size in bytes
    pub total_compressed_size: usize,

    /// Overall compression ratio
    pub overall_compression_ratio: f32,

    /// Average compression ratio
    pub average_compression_ratio: f32,

    /// Memory saved through compression
    pub memory_saved: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> EmbeddingCacheConfig {
        EmbeddingCacheConfig {
            enabled: true,
            max_size: 100,
            ttl: std::time::Duration::from_secs(3600),
            eviction_policy: super::super::EvictionPolicy::LFU,
            compression_enabled: true,
        }
    }

    #[test]
    fn test_embedding_cache_creation() {
        let config = create_test_config();
        let cache = EmbeddingCache::new(config).unwrap();

        assert_eq!(cache.size(), 0);
        assert!(cache.compressed_storage.is_some());
    }

    #[test]
    fn test_basic_operations() {
        let config = create_test_config();
        let mut cache = EmbeddingCache::new(config).unwrap();

        let text = "test text".to_string();
        let model = "test-model".to_string();
        let embedding = vec![1.0, 2.0, 3.0];

        // Cache embedding
        cache
            .cache_embedding(text.clone(), model.clone(), embedding.clone())
            .unwrap();
        assert_eq!(cache.size(), 1);

        // Retrieve embedding
        let retrieved = cache.get_embedding(&text, &model);
        assert!(retrieved.is_some());

        // Should be approximately equal (due to compression)
        let retrieved_embedding = retrieved.unwrap();
        assert_eq!(retrieved_embedding.len(), embedding.len());
    }

    #[test]
    fn test_compression() {
        let config = create_test_config();
        let cache = EmbeddingCache::new(config).unwrap();

        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let compressed = cache.compress_embedding(&embedding);

        assert_eq!(compressed.dimension, 5);
        assert_eq!(compressed.quantized_values.len(), 5);
        assert!(compressed.compression_ratio > 1.0);

        let decompressed = cache.decompress_embedding(&compressed);
        assert_eq!(decompressed.len(), embedding.len());
    }

    #[test]
    fn test_deduplication() {
        let config = create_test_config();
        let mut cache = EmbeddingCache::new(config).unwrap();

        let text = "same text".to_string();
        let embedding = vec![1.0, 2.0, 3.0];

        // Cache same text with different models
        cache
            .cache_embedding(text.clone(), "model1".to_string(), embedding.clone())
            .unwrap();
        cache
            .cache_embedding(text.clone(), "model2".to_string(), embedding.clone())
            .unwrap();

        let stats = cache.get_deduplication_stats();
        assert_eq!(stats.deduplicated_count, 1);
        assert!(stats.deduplication_ratio > 0.0);
    }

    #[test]
    fn test_hash_text() {
        let config = create_test_config();
        let cache = EmbeddingCache::new(config).unwrap();

        let text1 = "hello world";
        let text2 = "hello world";
        let text3 = "goodbye world";

        let hash1 = cache.hash_text(text1);
        let hash2 = cache.hash_text(text2);
        let hash3 = cache.hash_text(text3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
