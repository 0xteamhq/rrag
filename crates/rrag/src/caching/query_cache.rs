//! # Query Cache Implementation
//! 
//! Caching for query results with intelligent similarity matching.

use super::{
    Cache, CacheStats, QueryCacheConfig, QueryCacheEntry, CacheEntryMetadata,
    CachedSearchResult
};
use crate::{RragResult, RragError};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};

/// Query cache with similarity-based retrieval
pub struct QueryCache {
    /// Configuration
    config: QueryCacheConfig,
    
    /// Main storage
    storage: HashMap<String, QueryCacheEntry>,
    
    /// Query normalization cache
    normalized_queries: HashMap<String, String>,
    
    /// Query patterns for template matching
    query_patterns: Vec<QueryPattern>,
    
    /// Access statistics for adaptive caching
    access_stats: HashMap<String, QueryAccessStats>,
    
    /// Cache statistics
    stats: CacheStats,
}

/// Query pattern for template-based caching
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Pattern ID
    pub id: String,
    
    /// Pattern template (with placeholders)
    pub template: String,
    
    /// Pattern match count
    pub match_count: u64,
    
    /// Average result similarity
    pub avg_similarity: f32,
    
    /// Pattern effectiveness score
    pub effectiveness: f32,
}

/// Query access statistics
#[derive(Debug, Clone)]
pub struct QueryAccessStats {
    /// Total accesses
    pub access_count: u64,
    
    /// Last access time
    pub last_access: SystemTime,
    
    /// Average response time
    pub avg_response_time_ms: f32,
    
    /// Cache hit rate for similar queries
    pub similarity_hit_rate: f32,
    
    /// Query variations seen
    pub variations: Vec<String>,
}

impl QueryCache {
    /// Create new query cache
    pub fn new(config: QueryCacheConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            storage: HashMap::new(),
            normalized_queries: HashMap::new(),
            query_patterns: Vec::new(),
            access_stats: HashMap::new(),
            stats: CacheStats::default(),
        })
    }
    
    /// Get cached results for query
    pub fn get_results(&self, query: &str) -> Option<QueryCacheEntry> {
        // Direct lookup
        if let Some(entry) = self.storage.get(query) {
            if !entry.metadata.is_expired() {
                return Some(entry.clone());
            }
        }
        
        // Try normalized query
        let normalized = self.normalize_query(query);
        if let Some(canonical) = self.normalized_queries.get(&normalized) {
            if let Some(entry) = self.storage.get(canonical) {
                if !entry.metadata.is_expired() {
                    return Some(entry.clone());
                }
            }
        }
        
        // Try similarity matching if threshold is set
        if self.config.similarity_threshold > 0.0 {
            return self.find_similar_query(query);
        }
        
        None
    }
    
    /// Cache query results with intelligent deduplication
    pub fn cache_results(
        &mut self,
        query: String,
        results: Vec<CachedSearchResult>,
        generated_answer: Option<String>,
        embedding_hash: String,
    ) -> RragResult<()> {
        // Check capacity
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }
        
        // Create cache entry
        let mut metadata = CacheEntryMetadata::new();
        metadata.ttl = Some(self.config.ttl);
        
        let entry = QueryCacheEntry {
            query: query.clone(),
            embedding_hash,
            results,
            generated_answer,
            metadata,
        };
        
        // Store with normalization
        let normalized = self.normalize_query(&query);
        self.normalized_queries.insert(normalized, query.clone());
        self.storage.insert(query.clone(), entry);
        
        // Update patterns
        self.update_patterns(&query);
        
        // Update access stats
        self.update_access_stats(&query);
        
        Ok(())
    }
    
    /// Normalize query for better cache hits
    fn normalize_query(&self, query: &str) -> String {
        query
            .to_lowercase()
            .trim()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Find similar cached query
    fn find_similar_query(&self, query: &str) -> Option<QueryCacheEntry> {
        let normalized = self.normalize_query(query);
        let query_tokens: Vec<&str> = normalized.split_whitespace().collect();
        
        let mut best_match: Option<(&String, &QueryCacheEntry, f32)> = None;
        
        for (cached_query, entry) in &self.storage {
            if entry.metadata.is_expired() {
                continue;
            }
            
            let cached_normalized = self.normalize_query(cached_query);
            let cached_tokens: Vec<&str> = cached_normalized.split_whitespace().collect();
            
            // Calculate Jaccard similarity
            let intersection = query_tokens.iter()
                .filter(|t| cached_tokens.contains(t))
                .count();
            let union = (query_tokens.len() + cached_tokens.len() - intersection).max(1);
            let similarity = intersection as f32 / union as f32;
            
            if similarity >= self.config.similarity_threshold {
                if best_match.is_none() || similarity > best_match.as_ref().unwrap().2 {
                    best_match = Some((cached_query, entry, similarity));
                }
            }
        }
        
        best_match.map(|(_, entry, _)| entry.clone())
    }
    
    /// Update query patterns
    fn update_patterns(&mut self, query: &str) {
        // Extract potential pattern from query
        let pattern = self.extract_pattern(query);
        
        // Check if pattern exists
        if let Some(existing) = self.query_patterns.iter_mut()
            .find(|p| p.template == pattern) {
            existing.match_count += 1;
        } else if self.query_patterns.len() < 100 { // Limit patterns
            self.query_patterns.push(QueryPattern {
                id: format!("pattern_{}", self.query_patterns.len()),
                template: pattern,
                match_count: 1,
                avg_similarity: 0.0,
                effectiveness: 0.0,
            });
        }
    }
    
    /// Extract pattern from query
    fn extract_pattern(&self, query: &str) -> String {
        // Simple pattern extraction - replace numbers and quoted strings
        let mut pattern = query.to_string();
        
        // Replace numbers with placeholder
        pattern = regex::Regex::new(r"\b\d+\b")
            .unwrap_or_else(|_| regex::Regex::new("").unwrap())
            .replace_all(&pattern, "{NUM}")
            .to_string();
        
        // Replace quoted strings with placeholder
        pattern = regex::Regex::new(r#""[^"]*""#)
            .unwrap_or_else(|_| regex::Regex::new("").unwrap())
            .replace_all(&pattern, "{STR}")
            .to_string();
        
        pattern
    }
    
    /// Update access statistics
    fn update_access_stats(&mut self, query: &str) {
        let stats = self.access_stats.entry(query.to_string())
            .or_insert_with(|| QueryAccessStats {
                access_count: 0,
                last_access: SystemTime::now(),
                avg_response_time_ms: 0.0,
                similarity_hit_rate: 0.0,
                variations: Vec::new(),
            });
        
        stats.access_count += 1;
        stats.last_access = SystemTime::now();
    }
    
    /// Evict entry based on policy
    fn evict_entry(&mut self) -> RragResult<()> {
        use super::EvictionPolicy;
        
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::LFU => self.evict_lfu(),
            EvictionPolicy::TTL => self.evict_expired(),
            _ => self.evict_lru(), // Default to LRU
        }
    }
    
    /// Evict least recently used entry
    fn evict_lru(&mut self) -> RragResult<()> {
        if let Some((key, _)) = self.storage.iter()
            .min_by_key(|(_, entry)| entry.metadata.last_accessed) {
            let key = key.clone();
            self.storage.remove(&key);
            self.stats.evictions += 1;
        }
        Ok(())
    }
    
    /// Evict least frequently used entry
    fn evict_lfu(&mut self) -> RragResult<()> {
        if let Some((key, _)) = self.storage.iter()
            .min_by_key(|(_, entry)| entry.metadata.access_count) {
            let key = key.clone();
            self.storage.remove(&key);
            self.stats.evictions += 1;
        }
        Ok(())
    }
    
    /// Evict expired entries
    fn evict_expired(&mut self) -> RragResult<()> {
        let now = SystemTime::now();
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
    
    /// Get cache insights
    pub fn get_insights(&self) -> QueryCacheInsights {
        let total_queries = self.storage.len();
        let expired_queries = self.storage.values()
            .filter(|e| e.metadata.is_expired())
            .count();
        
        let avg_results_per_query = if total_queries > 0 {
            self.storage.values()
                .map(|e| e.results.len())
                .sum::<usize>() as f32 / total_queries as f32
        } else {
            0.0
        };
        
        let top_patterns: Vec<String> = self.query_patterns.iter()
            .filter(|p| p.match_count > 1)
            .take(5)
            .map(|p| p.template.clone())
            .collect();
        
        QueryCacheInsights {
            total_queries,
            expired_queries,
            avg_results_per_query,
            top_patterns,
            similarity_threshold: self.config.similarity_threshold,
        }
    }
}

impl Cache<String, QueryCacheEntry> for QueryCache {
    fn get(&self, key: &String) -> Option<QueryCacheEntry> {
        self.get_results(key)
    }
    
    fn put(&mut self, key: String, value: QueryCacheEntry) -> RragResult<()> {
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }
        
        let normalized = self.normalize_query(&key);
        self.normalized_queries.insert(normalized, key.clone());
        self.storage.insert(key, value);
        Ok(())
    }
    
    fn remove(&mut self, key: &String) -> Option<QueryCacheEntry> {
        let entry = self.storage.remove(key);
        
        // Remove from normalized queries
        let normalized = self.normalize_query(key);
        self.normalized_queries.remove(&normalized);
        
        // Remove from access stats
        self.access_stats.remove(key);
        
        entry
    }
    
    fn contains(&self, key: &String) -> bool {
        self.storage.contains_key(key) && 
        !self.storage.get(key).map_or(true, |e| e.metadata.is_expired())
    }
    
    fn clear(&mut self) {
        self.storage.clear();
        self.normalized_queries.clear();
        self.query_patterns.clear();
        self.access_stats.clear();
        self.stats = CacheStats::default();
    }
    
    fn size(&self) -> usize {
        self.storage.values()
            .filter(|e| !e.metadata.is_expired())
            .count()
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

/// Query cache insights
#[derive(Debug, Clone)]
pub struct QueryCacheInsights {
    /// Total cached queries
    pub total_queries: usize,
    
    /// Number of expired queries
    pub expired_queries: usize,
    
    /// Average results per query
    pub avg_results_per_query: f32,
    
    /// Top query patterns
    pub top_patterns: Vec<String>,
    
    /// Configured similarity threshold
    pub similarity_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_config() -> QueryCacheConfig {
        QueryCacheConfig {
            enabled: true,
            max_size: 100,
            ttl: Duration::from_secs(3600),
            eviction_policy: super::super::EvictionPolicy::LRU,
            similarity_threshold: 0.8,
        }
    }
    
    fn create_test_results() -> Vec<CachedSearchResult> {
        vec![
            CachedSearchResult {
                document_id: "doc1".to_string(),
                content: "test content".to_string(),
                score: 0.9,
                rank: 0,
                metadata: HashMap::new(),
            }
        ]
    }
    
    #[test]
    fn test_query_cache_creation() {
        let config = create_test_config();
        let cache = QueryCache::new(config).unwrap();
        
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.query_patterns.len(), 0);
    }
    
    #[test]
    fn test_basic_caching() {
        let config = create_test_config();
        let mut cache = QueryCache::new(config).unwrap();
        
        let query = "test query".to_string();
        let results = create_test_results();
        
        cache.cache_results(
            query.clone(),
            results.clone(),
            None,
            "hash123".to_string()
        ).unwrap();
        
        assert_eq!(cache.size(), 1);
        
        let cached = cache.get_results(&query);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().results.len(), 1);
    }
    
    #[test]
    fn test_query_normalization() {
        let config = create_test_config();
        let cache = QueryCache::new(config).unwrap();
        
        let query1 = "  What is   Rust?  ";
        let query2 = "what is rust";
        let query3 = "What is Rust???";
        
        let norm1 = cache.normalize_query(query1);
        let norm2 = cache.normalize_query(query2);
        let norm3 = cache.normalize_query(query3);
        
        assert_eq!(norm1, norm2);
        assert_eq!(norm2, norm3);
    }
    
    #[test]
    fn test_similarity_matching() {
        let config = create_test_config();
        let mut cache = QueryCache::new(config).unwrap();
        
        let query1 = "how to learn rust programming".to_string();
        let results = create_test_results();
        
        cache.cache_results(
            query1.clone(),
            results.clone(),
            None,
            "hash1".to_string()
        ).unwrap();
        
        // Similar query should find cached results
        let query2 = "learn rust programming how to";
        let cached = cache.get_results(query2);
        assert!(cached.is_some());
    }
    
    #[test]
    fn test_pattern_extraction() {
        let config = create_test_config();
        let cache = QueryCache::new(config).unwrap();
        
        let query1 = "get user 123 details";
        let query2 = "get user 456 details";
        
        let pattern1 = cache.extract_pattern(query1);
        let pattern2 = cache.extract_pattern(query2);
        
        assert_eq!(pattern1, pattern2);
        assert!(pattern1.contains("{NUM}"));
    }
    
    #[test]
    fn test_eviction() {
        let mut config = create_test_config();
        config.max_size = 2;
        let mut cache = QueryCache::new(config).unwrap();
        
        let results = create_test_results();
        
        cache.cache_results("query1".to_string(), results.clone(), None, "h1".to_string()).unwrap();
        cache.cache_results("query2".to_string(), results.clone(), None, "h2".to_string()).unwrap();
        
        assert_eq!(cache.size(), 2);
        
        // This should trigger eviction
        cache.cache_results("query3".to_string(), results.clone(), None, "h3".to_string()).unwrap();
        
        assert_eq!(cache.size(), 2);
        assert_eq!(cache.stats.evictions, 1);
    }
}