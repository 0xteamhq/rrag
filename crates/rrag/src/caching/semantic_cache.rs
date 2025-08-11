//! # Semantic Cache Implementation
//! 
//! Intelligent caching based on semantic similarity for RAG applications.

use super::{
    Cache, CacheStats, SemanticCacheConfig, SemanticCacheEntry
};
use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Semantic cache with similarity-based retrieval
pub struct SemanticCache {
    /// Configuration
    config: SemanticCacheConfig,
    
    /// Main storage indexed by query hash
    storage: HashMap<String, SemanticCacheEntry>,
    
    /// Embedding vectors for similarity computation
    embeddings: HashMap<String, Vec<f32>>,
    
    /// Semantic clusters for efficient search
    clusters: Vec<SemanticCluster>,
    
    /// Query to cluster mapping
    query_clusters: HashMap<String, usize>,
    
    /// Cache statistics
    stats: CacheStats,
}

/// Semantic cluster for grouping similar queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCluster {
    /// Cluster ID
    pub id: usize,
    
    /// Cluster centroid (average embedding)
    pub centroid: Vec<f32>,
    
    /// Queries in this cluster
    pub queries: Vec<String>,
    
    /// Representative query (closest to centroid)
    pub representative: String,
    
    /// Cluster quality metrics
    pub cohesion: f32,
    
    /// Last updated
    pub last_updated: SystemTime,
}

/// Similarity search result
#[derive(Debug, Clone)]
pub struct SimilaritySearchResult {
    /// Query text
    pub query: String,
    
    /// Similarity score
    pub similarity: f32,
    
    /// Cached entry
    pub entry: SemanticCacheEntry,
}

/// Clustering algorithm types
#[derive(Debug, Clone)]
pub enum ClusteringAlgorithm {
    KMeans,
    HierarchicalClustering,
    DBSCAN,
    OnlineKMeans,
}

impl SemanticCache {
    /// Create new semantic cache
    pub fn new(config: SemanticCacheConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            storage: HashMap::new(),
            embeddings: HashMap::new(),
            clusters: Vec::new(),
            query_clusters: HashMap::new(),
            stats: CacheStats::default(),
        })
    }
    
    /// Find semantically similar cached entries
    pub fn find_similar(&self, _query: &str, embedding: &[f32]) -> Vec<SimilaritySearchResult> {
        let mut results = Vec::new();
        
        for (cached_query, cached_embedding) in &self.embeddings {
            let similarity = self.compute_similarity(embedding, cached_embedding);
            
            if similarity >= self.config.similarity_threshold {
                if let Some(entry) = self.storage.get(cached_query) {
                    results.push(SimilaritySearchResult {
                        query: cached_query.clone(),
                        similarity,
                        entry: entry.clone(),
                    });
                }
            }
        }
        
        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        
        // Limit results
        results.truncate(10);
        results
    }
    
    /// Get or find semantically similar entry
    pub fn get_or_similar(&self, query: &str, embedding: Option<&[f32]>) -> Option<SemanticCacheEntry> {
        // Direct hit first
        if let Some(entry) = self.storage.get(query) {
            return Some(entry.clone());
        }
        
        // Semantic similarity search
        if let Some(emb) = embedding {
            let similar = self.find_similar(query, emb);
            if let Some(best_match) = similar.first() {
                return Some(best_match.entry.clone());
            }
        }
        
        None
    }
    
    /// Cache entry with semantic clustering
    pub fn cache_with_clustering(&mut self, query: String, embedding: Vec<f32>, entry: SemanticCacheEntry) -> RragResult<()> {
        // Store embedding
        self.embeddings.insert(query.clone(), embedding.clone());
        
        // Find best cluster or create new one
        if self.config.clustering_enabled {
            let cluster_id = self.assign_to_cluster(&query, &embedding)?;
            self.query_clusters.insert(query.clone(), cluster_id);
        }
        
        // Store entry
        self.storage.insert(query, entry);
        
        // Update clusters if needed
        if self.config.clustering_enabled && self.storage.len() % 10 == 0 {
            self.update_clusters()?;
        }
        
        Ok(())
    }
    
    /// Assign query to best cluster
    fn assign_to_cluster(&mut self, query: &str, embedding: &[f32]) -> RragResult<usize> {
        if self.clusters.is_empty() {
            // Create first cluster
            let cluster = SemanticCluster {
                id: 0,
                centroid: embedding.to_vec(),
                queries: vec![query.to_string()],
                representative: query.to_string(),
                cohesion: 1.0,
                last_updated: SystemTime::now(),
            };
            self.clusters.push(cluster);
            return Ok(0);
        }
        
        // Find best cluster by centroid similarity
        let mut best_cluster = 0;
        let mut best_similarity = 0.0;
        
        for (i, cluster) in self.clusters.iter().enumerate() {
            let similarity = self.compute_similarity(embedding, &cluster.centroid);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_cluster = i;
            }
        }
        
        // Create new cluster if similarity is too low
        if best_similarity < self.config.similarity_threshold {
            if self.clusters.len() < self.config.max_clusters {
                let cluster_id = self.clusters.len();
                let cluster = SemanticCluster {
                    id: cluster_id,
                    centroid: embedding.to_vec(),
                    queries: vec![query.to_string()],
                    representative: query.to_string(),
                    cohesion: 1.0,
                    last_updated: SystemTime::now(),
                };
                self.clusters.push(cluster);
                return Ok(cluster_id);
            }
        }
        
        // Add to best cluster
        if let Some(cluster) = self.clusters.get_mut(best_cluster) {
            cluster.queries.push(query.to_string());
            cluster.last_updated = SystemTime::now();
        }
        
        Ok(best_cluster)
    }
    
    /// Update cluster centroids and representatives
    fn update_clusters(&mut self) -> RragResult<()> {
        for cluster in &mut self.clusters {
            if cluster.queries.is_empty() {
                continue;
            }
            
            // Compute new centroid
            let mut centroid = vec![0.0; cluster.centroid.len()];
            let mut count = 0;
            
            for query in &cluster.queries {
                if let Some(embedding) = self.embeddings.get(query) {
                    for (i, &val) in embedding.iter().enumerate() {
                        if i < centroid.len() {
                            centroid[i] += val;
                        }
                    }
                    count += 1;
                }
            }
            
            if count > 0 {
                for val in &mut centroid {
                    *val /= count as f32;
                }
                cluster.centroid = centroid;
            }
            
            // Find new representative (closest to centroid)
            let mut best_query = cluster.representative.clone();
            let mut best_similarity = 0.0;
            
            for query in &cluster.queries {
                if let Some(embedding) = self.embeddings.get(query) {
                    // Inline cosine similarity calculation to avoid borrowing self
                    let dot_product: f32 = cluster.centroid.iter().zip(embedding.iter()).map(|(x, y)| x * y).sum();
                    let norm_a: f32 = cluster.centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_b: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let similarity = if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) };
                    
                    if similarity > best_similarity {
                        best_similarity = similarity;
                        best_query = query.clone();
                    }
                }
            }
            
            cluster.representative = best_query;
            cluster.cohesion = best_similarity;
        }
        
        Ok(())
    }
    
    /// Compute cosine similarity between two embeddings
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    /// Get cluster information
    pub fn get_clusters(&self) -> &[SemanticCluster] {
        &self.clusters
    }
    
    /// Get cache insights
    pub fn get_insights(&self) -> SemanticCacheInsights {
        let total_queries = self.storage.len();
        let total_clusters = self.clusters.len();
        let avg_cluster_size = if total_clusters > 0 {
            total_queries as f32 / total_clusters as f32
        } else {
            0.0
        };
        
        let cluster_cohesions: Vec<f32> = self.clusters.iter().map(|c| c.cohesion).collect();
        let avg_cohesion = if !cluster_cohesions.is_empty() {
            cluster_cohesions.iter().sum::<f32>() / cluster_cohesions.len() as f32
        } else {
            0.0
        };
        
        SemanticCacheInsights {
            total_queries,
            total_clusters,
            avg_cluster_size,
            avg_cohesion,
            similarity_threshold: self.config.similarity_threshold,
            clustering_enabled: self.config.clustering_enabled,
        }
    }
}

impl Cache<String, SemanticCacheEntry> for SemanticCache {
    fn get(&self, key: &String) -> Option<SemanticCacheEntry> {
        self.storage.get(key).cloned()
    }
    
    fn put(&mut self, key: String, value: SemanticCacheEntry) -> RragResult<()> {
        // Check capacity
        if self.storage.len() >= self.config.max_size {
            self.evict_entry()?;
        }
        
        self.storage.insert(key, value);
        Ok(())
    }
    
    fn remove(&mut self, key: &String) -> Option<SemanticCacheEntry> {
        let entry = self.storage.remove(key);
        self.embeddings.remove(key);
        
        // Remove from cluster
        if let Some(cluster_id) = self.query_clusters.remove(key) {
            if let Some(cluster) = self.clusters.get_mut(cluster_id) {
                cluster.queries.retain(|q| q != key);
            }
        }
        
        entry
    }
    
    fn contains(&self, key: &String) -> bool {
        self.storage.contains_key(key)
    }
    
    fn clear(&mut self) {
        self.storage.clear();
        self.embeddings.clear();
        self.clusters.clear();
        self.query_clusters.clear();
        self.stats = CacheStats::default();
    }
    
    fn size(&self) -> usize {
        self.storage.len()
    }
    
    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
}

impl SemanticCache {
    /// Evict entry using semantic-aware policy
    fn evict_entry(&mut self) -> RragResult<()> {
        if self.storage.is_empty() {
            return Ok(());
        }
        
        // Find entry with lowest access frequency in largest cluster
        let mut candidate_key: Option<String> = None;
        let mut min_score = f32::INFINITY;
        
        for (key, entry) in &self.storage {
            // Calculate eviction score based on access patterns and cluster size
            let access_score = entry.metadata.access_count as f32;
            let time_score = entry.metadata.last_accessed
                .elapsed()
                .unwrap_or_default()
                .as_secs() as f32;
            
            // Prefer evicting from larger clusters
            let cluster_score = if let Some(&cluster_id) = self.query_clusters.get(key) {
                if let Some(cluster) = self.clusters.get(cluster_id) {
                    cluster.queries.len() as f32
                } else {
                    1.0
                }
            } else {
                1.0
            };
            
            // Combined score (lower is better for eviction)
            let eviction_score = access_score / (time_score + 1.0) / cluster_score;
            
            if eviction_score < min_score {
                min_score = eviction_score;
                candidate_key = Some(key.clone());
            }
        }
        
        if let Some(key) = candidate_key {
            self.remove(&key);
            self.stats.evictions += 1;
        }
        
        Ok(())
    }
}

/// Semantic cache insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheInsights {
    /// Total cached queries
    pub total_queries: usize,
    
    /// Total clusters
    pub total_clusters: usize,
    
    /// Average cluster size
    pub avg_cluster_size: f32,
    
    /// Average cluster cohesion
    pub avg_cohesion: f32,
    
    /// Configured similarity threshold
    pub similarity_threshold: f32,
    
    /// Whether clustering is enabled
    pub clustering_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    fn create_test_config() -> SemanticCacheConfig {
        SemanticCacheConfig {
            enabled: true,
            max_size: 100,
            ttl: std::time::Duration::from_secs(3600),
            similarity_threshold: 0.8,
            clustering_enabled: true,
            max_clusters: 10,
        }
    }
    
    fn create_test_entry() -> SemanticCacheEntry {
        SemanticCacheEntry {
            representative: "test query".to_string(),
            cluster_id: None,
            similar_entries: vec![],
            results: vec![
                CachedSearchResult {
                    document_id: "doc1".to_string(),
                    content: "test content".to_string(),
                    score: 0.9,
                    rank: 0,
                    metadata: HashMap::new(),
                }
            ],
            metadata: CacheEntryMetadata::new(),
        }
    }
    
    #[test]
    fn test_semantic_cache_creation() {
        let config = create_test_config();
        let cache = SemanticCache::new(config).unwrap();
        
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.clusters.len(), 0);
    }
    
    #[test]
    fn test_basic_cache_operations() {
        let config = create_test_config();
        let mut cache = SemanticCache::new(config).unwrap();
        
        let entry = create_test_entry();
        let key = "test_query".to_string();
        
        // Test put and get
        cache.put(key.clone(), entry.clone()).unwrap();
        assert_eq!(cache.size(), 1);
        
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().representative, entry.representative);
        
        // Test remove
        let removed = cache.remove(&key);
        assert!(removed.is_some());
        assert_eq!(cache.size(), 0);
    }
    
    #[test]
    fn test_similarity_computation() {
        let config = create_test_config();
        let cache = SemanticCache::new(config).unwrap();
        
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0];
        let vec_c = vec![0.0, 1.0, 0.0];
        
        // Test identical vectors
        let similarity = cache.compute_similarity(&vec_a, &vec_b);
        assert!((similarity - 1.0).abs() < 0.001);
        
        // Test orthogonal vectors
        let similarity = cache.compute_similarity(&vec_a, &vec_c);
        assert!((similarity - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_clustering() {
        let config = create_test_config();
        let mut cache = SemanticCache::new(config).unwrap();
        
        let entry = create_test_entry();
        let embedding = vec![1.0, 0.0, 0.0];
        
        cache.cache_with_clustering(
            "test query".to_string(), 
            embedding,
            entry
        ).unwrap();
        
        assert_eq!(cache.clusters.len(), 1);
        assert_eq!(cache.clusters[0].queries.len(), 1);
    }
    
    #[test]
    fn test_cache_insights() {
        let config = create_test_config();
        let cache = SemanticCache::new(config).unwrap();
        
        let insights = cache.get_insights();
        assert_eq!(insights.total_queries, 0);
        assert_eq!(insights.total_clusters, 0);
        assert!(insights.clustering_enabled);
    }
}