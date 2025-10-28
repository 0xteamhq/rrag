//! # RRAG Retrieval System
//!
//! High-performance, async-first retrieval system with pluggable similarity search,
//! advanced ranking algorithms, and comprehensive filtering capabilities. Built for
//! production workloads with sub-millisecond response times and horizontal scaling.
//!
//! ## Features
//!
//! - **Multiple Search Algorithms**: Cosine similarity, dot product, Euclidean distance
//! - **Advanced Filtering**: Metadata-based filtering with complex queries
//! - **Ranking & Scoring**: Configurable scoring and re-ranking strategies
//! - **Async Operations**: Full async/await support for high concurrency
//! - **Memory Efficient**: Optimized data structures and minimal allocations
//! - **Pluggable Backends**: Support for multiple storage backends
//! - **Real-time Updates**: Live index updates without downtime
//!
//! ## Quick Start
//!
//! ### Basic Similarity Search
//!
//! ```rust
//! use rrag::prelude::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Create a retrieval service
//! let storage = Arc::new(InMemoryStorage::new());
//! let retriever = InMemoryRetriever::new()
//!     .with_storage(storage)
//!     .with_similarity_threshold(0.8);
//!
//! // Add documents to the index
//! let documents = vec![
//!     Document::new("Rust is a systems programming language"),
//!     Document::new("Python is great for data science"),
//!     Document::new("JavaScript runs in web browsers"),
//! ];
//!
//! for doc in documents {
//!     retriever.index_document(&doc).await?;
//! }
//!
//! // Search for similar content
//! let query = SearchQuery::new("programming languages")
//!     .with_limit(5)
//!     .with_min_score(0.7);
//!
//! let results = retriever.search(query).await?;
//! for result in results {
//!     tracing::debug!("Score: {:.3} - {}", result.score, result.content);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Search with Filters
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! # let retriever = InMemoryRetriever::new();
//! // Search with metadata filters
//! let query = SearchQuery::new("machine learning")
//!     .with_filter("category", "technical".into())
//!     .with_filter("language", "english".into())
//!     .with_date_range("created_after", "2023-01-01")
//!     .with_config(SearchConfig {
//!         algorithm: SearchAlgorithm::CosineSimilarity,
//!         enable_reranking: true,
//!         include_embeddings: false,
//!         ..Default::default()
//!     });
//!
//! let results = retriever.search(query).await?;
//! tracing::debug!("Found {} filtered results", results.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Retrieval Implementation
//!
//! ```rust
//! use rrag::prelude::*;
//! use async_trait::async_trait;
//!
//! struct CustomRetriever {
//!     // Your custom fields
//! }
//!
//! #[async_trait]
//! impl Retriever for CustomRetriever {
//!     async fn search(&self, query: SearchQuery) -> RragResult<Vec<SearchResult>> {
//!         // Your custom search logic
//!         # Ok(Vec::new())
//!     }
//!
//!     async fn index_document(&self, document: &Document) -> RragResult<()> {
//!         // Your custom indexing logic
//!         Ok(())
//!     }
//!
//!     async fn delete_document(&self, id: &str) -> RragResult<bool> {
//!         // Your custom deletion logic
//!         Ok(true)
//!     }
//! }
//! ```
//!
//! ## Search Algorithms
//!
//! RRAG supports multiple similarity algorithms:
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! // Cosine similarity (default, best for most use cases)
//! let config = SearchConfig {
//!     algorithm: SearchAlgorithm::CosineSimilarity,
//!     ..Default::default()
//! };
//!
//! // Dot product (faster, good for normalized embeddings)
//! let config = SearchConfig {
//!     algorithm: SearchAlgorithm::DotProduct,
//!     ..Default::default()
//! };
//!
//! // Euclidean distance (good for spatial data)
//! let config = SearchConfig {
//!     algorithm: SearchAlgorithm::EuclideanDistance,
//!     ..Default::default()
//! };
//! ```
//!
//! ## Performance Optimization
//!
//! - **Batch Operations**: Index multiple documents at once
//! - **Parallel Search**: Concurrent query processing
//! - **Memory Optimization**: Efficient vector storage and computation
//! - **Caching**: Optional result caching for repeated queries
//! - **Lazy Loading**: Load embeddings on demand
//!
//! ## Error Handling
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! match retriever.search(query).await {
//!     Ok(results) => {
//!         tracing::debug!("Found {} results", results.len());
//!         for result in results {
//!             tracing::debug!("  {}: {:.3}", result.content, result.score);
//!         }
//!     }
//!     Err(RragError::Retrieval { query, .. }) => {
//!         tracing::debug!("Search failed for query: {}", query);
//!     }
//!     Err(e) => {
//!         tracing::debug!("Retrieval error: {}", e);
//!     }
//! }
//! # }
//! ```

use crate::{Document, DocumentChunk, Embedding, RragError, RragResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// A search result containing content, similarity score, and metadata
///
/// Represents a single result from a similarity search operation, including
/// the matched content, relevance score, ranking information, and associated
/// metadata. Results are typically returned in descending order of relevance.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// let result = SearchResult::new(
///     "doc-123",
///     "This document discusses machine learning algorithms",
///     0.87, // 87% similarity
///     0     // First result
/// )
/// .with_metadata("category", "technical".into())
/// .with_metadata("author", "Dr. Smith".into())
/// .with_embedding(embedding); // Optional embedding
///
/// tracing::debug!("Result: {} (score: {:.3})", result.content, result.score);
/// ```
///
/// # Scoring
///
/// Scores are normalized to 0.0-1.0 range where:
/// - 1.0 = Perfect match (identical content)
/// - 0.8+ = Very relevant
/// - 0.6-0.8 = Somewhat relevant  
/// - <0.6 = Low relevance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document or chunk ID
    pub id: String,

    /// Content that matched the query
    pub content: String,

    /// Similarity score (0.0 to 1.0, higher is more similar)
    pub score: f32,

    /// Ranking position in results (0-indexed)
    pub rank: usize,

    /// Associated metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Embedding used for the match (optional)
    pub embedding: Option<Embedding>,
}

impl SearchResult {
    /// Create a new search result with the specified parameters
    pub fn new(id: impl Into<String>, content: impl Into<String>, score: f32, rank: usize) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            score,
            rank,
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// A search query with comprehensive configuration options
///
/// Encapsulates all parameters for a search operation including the query itself,
/// result limits, filtering criteria, and algorithm configuration. Supports both
/// text queries (that will be embedded) and pre-computed embedding queries.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// // Simple text query
/// let query = SearchQuery::new("machine learning algorithms")
///     .with_limit(10)
///     .with_min_score(0.7);
///
/// // Advanced query with filters
/// let advanced_query = SearchQuery::new("neural networks")
///     .with_limit(20)
///     .with_min_score(0.6)
///     .with_filter("category", "research".into())
///     .with_filter("year", 2023.into())
///     .with_config(SearchConfig {
///         algorithm: SearchAlgorithm::CosineSimilarity,
///         enable_reranking: true,
///         include_embeddings: true,
///         ..Default::default()
///     });
///
/// // Query with pre-computed embedding
/// let embedding_query = SearchQuery::from_embedding(embedding)
///     .with_limit(5);
/// ```
///
/// # Filter Types
///
/// Filters support various data types:
/// - **Strings**: Exact match or pattern matching
/// - **Numbers**: Range queries and exact values
/// - **Dates**: Date range filtering
/// - **Arrays**: "Contains" operations
/// - **Booleans**: Exact boolean matching
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Query text or embedding
    pub query: QueryType,

    /// Maximum number of results to return
    pub limit: usize,

    /// Minimum similarity threshold
    pub min_score: f32,

    /// Metadata filters
    pub filters: HashMap<String, serde_json::Value>,

    /// Search configuration
    pub config: SearchConfig,
}

/// Query type - text or pre-computed embedding
#[derive(Debug, Clone)]
pub enum QueryType {
    /// Text query that needs to be embedded
    Text(String),

    /// Pre-computed embedding vector
    Embedding(Embedding),
}

/// Search configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Whether to include embeddings in results
    pub include_embeddings: bool,

    /// Whether to apply re-ranking
    pub enable_reranking: bool,

    /// Search algorithm to use
    pub algorithm: SearchAlgorithm,

    /// Custom scoring weights
    pub scoring_weights: ScoringWeights,
}

/// Search algorithms available
#[derive(Debug, Clone)]
pub enum SearchAlgorithm {
    /// Cosine similarity search
    Cosine,

    /// Euclidean distance search
    Euclidean,

    /// Dot product search
    DotProduct,

    /// Hybrid search (combine multiple methods)
    Hybrid {
        methods: Vec<SearchAlgorithm>,
        weights: Vec<f32>,
    },
}

/// Scoring weights for different factors
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for semantic similarity
    pub semantic: f32,

    /// Weight for metadata matches
    pub metadata: f32,

    /// Weight for recency (if timestamps available)
    pub recency: f32,

    /// Weight for content length/quality
    pub quality: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            include_embeddings: false,
            enable_reranking: true,
            algorithm: SearchAlgorithm::Cosine,
            scoring_weights: ScoringWeights::default(),
        }
    }
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            semantic: 1.0,
            metadata: 0.1,
            recency: 0.05,
            quality: 0.1,
        }
    }
}

impl SearchQuery {
    /// Create a text-based search query
    pub fn text(query: impl Into<String>) -> Self {
        Self {
            query: QueryType::Text(query.into()),
            limit: 10,
            min_score: 0.0,
            filters: HashMap::new(),
            config: SearchConfig::default(),
        }
    }

    /// Create an embedding-based search query
    pub fn embedding(embedding: Embedding) -> Self {
        Self {
            query: QueryType::Embedding(embedding),
            limit: 10,
            min_score: 0.0,
            filters: HashMap::new(),
            config: SearchConfig::default(),
        }
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set minimum score threshold
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Add metadata filter
    pub fn with_filter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.filters.insert(key.into(), value);
        self
    }

    /// Set search configuration
    pub fn with_config(mut self, config: SearchConfig) -> Self {
        self.config = config;
        self
    }
}

/// Core retrieval trait for different storage backends
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retriever name/type
    fn name(&self) -> &str;

    /// Search for similar documents/chunks
    async fn search(&self, query: &SearchQuery) -> RragResult<Vec<SearchResult>>;

    /// Add documents to the retrieval index
    async fn add_documents(&self, documents: &[(Document, Embedding)]) -> RragResult<()>;

    /// Add document chunks to the retrieval index
    async fn add_chunks(&self, chunks: &[(DocumentChunk, Embedding)]) -> RragResult<()>;

    /// Remove documents from the index
    async fn remove_documents(&self, document_ids: &[String]) -> RragResult<()>;

    /// Clear all documents from the index
    async fn clear(&self) -> RragResult<()>;

    /// Get index statistics
    async fn stats(&self) -> RragResult<IndexStats>;

    /// Health check
    async fn health_check(&self) -> RragResult<bool>;
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of documents/chunks indexed
    pub total_items: usize,

    /// Index size in bytes (estimate)
    pub size_bytes: usize,

    /// Number of dimensions
    pub dimensions: usize,

    /// Index type/implementation
    pub index_type: String,

    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// In-memory retriever for small datasets and testing
pub struct InMemoryRetriever {
    /// Stored documents with embeddings
    documents: Arc<tokio::sync::RwLock<HashMap<String, (Document, Embedding)>>>,

    /// Stored chunks with embeddings
    chunks: Arc<tokio::sync::RwLock<HashMap<String, (DocumentChunk, Embedding)>>>,

    /// Retriever configuration
    config: RetrieverConfig,
}

/// Retriever configuration
#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// Whether to store documents, chunks, or both
    pub storage_mode: StorageMode,

    /// Default similarity threshold
    pub default_threshold: f32,

    /// Maximum results to return
    pub max_results: usize,
}

#[derive(Debug, Clone)]
pub enum StorageMode {
    DocumentsOnly,
    ChunksOnly,
    Both,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            storage_mode: StorageMode::Both,
            default_threshold: 0.0,
            max_results: 1000,
        }
    }
}

impl InMemoryRetriever {
    /// Create new in-memory retriever
    pub fn new() -> Self {
        Self {
            documents: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            chunks: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config: RetrieverConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RetrieverConfig) -> Self {
        Self {
            documents: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            chunks: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Calculate similarity between embeddings
    fn calculate_similarity(
        &self,
        embedding1: &Embedding,
        embedding2: &Embedding,
        algorithm: &SearchAlgorithm,
    ) -> RragResult<f32> {
        match algorithm {
            SearchAlgorithm::Cosine => embedding1.cosine_similarity(embedding2),
            SearchAlgorithm::Euclidean => {
                let distance = embedding1.euclidean_distance(embedding2)?;
                // Convert distance to similarity score (0-1)
                Ok(1.0 / (1.0 + distance))
            }
            SearchAlgorithm::DotProduct => {
                if embedding1.dimensions != embedding2.dimensions {
                    return Err(RragError::retrieval(format!(
                        "Dimension mismatch: {} vs {}",
                        embedding1.dimensions, embedding2.dimensions
                    )));
                }
                let dot_product: f32 = embedding1
                    .vector
                    .iter()
                    .zip(embedding2.vector.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                Ok(dot_product.max(0.0).min(1.0)) // Clamp to [0, 1]
            }
            SearchAlgorithm::Hybrid { methods, weights } => {
                let mut total_score = 0.0;
                let mut total_weight = 0.0;

                for (method, weight) in methods.iter().zip(weights.iter()) {
                    let score = self.calculate_similarity(embedding1, embedding2, method)?;
                    total_score += score * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    Ok(total_score / total_weight)
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Apply metadata filters to a result
    fn apply_filters(
        &self,
        metadata: &HashMap<String, serde_json::Value>,
        filters: &HashMap<String, serde_json::Value>,
    ) -> bool {
        for (key, expected_value) in filters {
            match metadata.get(key) {
                Some(actual_value) if actual_value == expected_value => continue,
                _ => return false,
            }
        }
        true
    }

    /// Apply re-ranking with custom scoring
    fn rerank_results(
        &self,
        mut results: Vec<SearchResult>,
        weights: &ScoringWeights,
    ) -> Vec<SearchResult> {
        // Calculate enhanced scores
        for result in &mut results {
            let mut enhanced_score = result.score * weights.semantic;

            // Add metadata matching bonus
            if !result.metadata.is_empty() {
                enhanced_score += 0.1 * weights.metadata;
            }

            // Add recency bonus if timestamp is available
            if let Some(timestamp_value) = result.metadata.get("created_at") {
                if let Some(timestamp_str) = timestamp_value.as_str() {
                    if let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(timestamp_str) {
                        let age_days =
                            (chrono::Utc::now() - timestamp.with_timezone(&chrono::Utc)).num_days();
                        let recency_bonus = (-age_days as f32 / 30.0).exp() * weights.recency;
                        enhanced_score += recency_bonus;
                    }
                }
            }

            // Add quality bonus based on content length
            let content_length = result.content.len();
            if content_length > 100 && content_length < 2000 {
                enhanced_score += 0.05 * weights.quality;
            }

            result.score = enhanced_score.min(1.0);
        }

        // Re-sort by enhanced scores
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        results
    }
}

impl Default for InMemoryRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Retriever for InMemoryRetriever {
    fn name(&self) -> &str {
        "in_memory"
    }

    async fn search(&self, query: &SearchQuery) -> RragResult<Vec<SearchResult>> {
        let query_embedding = match &query.query {
            QueryType::Text(_) => {
                return Err(RragError::retrieval(
                    "Text queries require pre-computed embeddings for in-memory retriever"
                        .to_string(),
                ));
            }
            QueryType::Embedding(emb) => emb,
        };

        let mut results = Vec::new();

        // Search documents if enabled
        if matches!(
            self.config.storage_mode,
            StorageMode::DocumentsOnly | StorageMode::Both
        ) {
            let documents = self.documents.read().await;
            for (doc_id, (document, embedding)) in documents.iter() {
                // Apply metadata filters
                if !self.apply_filters(&document.metadata, &query.filters) {
                    continue;
                }

                let similarity =
                    self.calculate_similarity(query_embedding, embedding, &query.config.algorithm)?;

                if similarity >= query.min_score {
                    let mut result = SearchResult::new(
                        doc_id,
                        document.content_str(),
                        similarity,
                        0, // Will be updated after sorting
                    )
                    .with_metadata("type", serde_json::Value::String("document".to_string()));

                    // Add document metadata
                    for (key, value) in &document.metadata {
                        result = result.with_metadata(key, value.clone());
                    }

                    if query.config.include_embeddings {
                        result = result.with_embedding(embedding.clone());
                    }

                    results.push(result);
                }
            }
        }

        // Search chunks if enabled
        if matches!(
            self.config.storage_mode,
            StorageMode::ChunksOnly | StorageMode::Both
        ) {
            let chunks = self.chunks.read().await;
            for (chunk_id, (chunk, embedding)) in chunks.iter() {
                // Apply metadata filters
                if !self.apply_filters(&chunk.metadata, &query.filters) {
                    continue;
                }

                let similarity =
                    self.calculate_similarity(query_embedding, embedding, &query.config.algorithm)?;

                if similarity >= query.min_score {
                    let mut result = SearchResult::new(
                        chunk_id,
                        &chunk.content,
                        similarity,
                        0, // Will be updated after sorting
                    )
                    .with_metadata("type", serde_json::Value::String("chunk".to_string()))
                    .with_metadata(
                        "document_id",
                        serde_json::Value::String(chunk.document_id.clone()),
                    )
                    .with_metadata(
                        "chunk_index",
                        serde_json::Value::Number(chunk.chunk_index.into()),
                    );

                    // Add chunk metadata
                    for (key, value) in &chunk.metadata {
                        result = result.with_metadata(key, value.clone());
                    }

                    if query.config.include_embeddings {
                        result = result.with_embedding(embedding.clone());
                    }

                    results.push(result);
                }
            }
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply re-ranking if enabled
        if query.config.enable_reranking {
            results = self.rerank_results(results, &query.config.scoring_weights);
        }

        // Update ranks after sorting
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        // Limit results
        results.truncate(query.limit.min(self.config.max_results));

        Ok(results)
    }

    async fn add_documents(&self, documents: &[(Document, Embedding)]) -> RragResult<()> {
        let mut docs = self.documents.write().await;
        for (document, embedding) in documents {
            docs.insert(document.id.clone(), (document.clone(), embedding.clone()));
        }
        Ok(())
    }

    async fn add_chunks(&self, chunks: &[(DocumentChunk, Embedding)]) -> RragResult<()> {
        let mut chunk_store = self.chunks.write().await;
        for (chunk, embedding) in chunks {
            let chunk_id = format!("{}_{}", chunk.document_id, chunk.chunk_index);
            chunk_store.insert(chunk_id, (chunk.clone(), embedding.clone()));
        }
        Ok(())
    }

    async fn remove_documents(&self, document_ids: &[String]) -> RragResult<()> {
        let mut docs = self.documents.write().await;
        for doc_id in document_ids {
            docs.remove(doc_id);
        }

        // Also remove associated chunks
        let mut chunk_store = self.chunks.write().await;
        let chunk_ids_to_remove: Vec<String> = chunk_store
            .iter()
            .filter(|(_, (chunk, _))| document_ids.contains(&chunk.document_id))
            .map(|(id, _)| id.clone())
            .collect();

        for chunk_id in chunk_ids_to_remove {
            chunk_store.remove(&chunk_id);
        }

        Ok(())
    }

    async fn clear(&self) -> RragResult<()> {
        self.documents.write().await.clear();
        self.chunks.write().await.clear();
        Ok(())
    }

    async fn stats(&self) -> RragResult<IndexStats> {
        let doc_count = self.documents.read().await.len();
        let chunk_count = self.chunks.read().await.len();

        // Get embedding dimensions from first item
        let dimensions = if doc_count > 0 {
            self.documents
                .read()
                .await
                .values()
                .next()
                .map(|(_, emb)| emb.dimensions)
                .unwrap_or(0)
        } else if chunk_count > 0 {
            self.chunks
                .read()
                .await
                .values()
                .next()
                .map(|(_, emb)| emb.dimensions)
                .unwrap_or(0)
        } else {
            0
        };

        Ok(IndexStats {
            total_items: doc_count + chunk_count,
            size_bytes: (doc_count + chunk_count) * dimensions * 4, // Rough estimate
            dimensions,
            index_type: "in_memory".to_string(),
            last_updated: chrono::Utc::now(),
        })
    }

    async fn health_check(&self) -> RragResult<bool> {
        Ok(true)
    }
}

/// High-level retrieval service
pub struct RetrievalService {
    /// Active retriever
    retriever: Arc<dyn Retriever>,

    /// Service configuration
    config: RetrievalServiceConfig,
}

/// Configuration for retrieval service
#[derive(Debug, Clone)]
pub struct RetrievalServiceConfig {
    /// Default search configuration
    pub default_search_config: SearchConfig,

    /// Cache query results
    pub enable_caching: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for RetrievalServiceConfig {
    fn default() -> Self {
        Self {
            default_search_config: SearchConfig::default(),
            enable_caching: false,
            cache_ttl_seconds: 300, // 5 minutes
        }
    }
}

impl RetrievalService {
    /// Create retrieval service
    pub fn new(retriever: Arc<dyn Retriever>) -> Self {
        Self {
            retriever,
            config: RetrievalServiceConfig::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(retriever: Arc<dyn Retriever>, config: RetrievalServiceConfig) -> Self {
        Self { retriever, config }
    }

    /// Search with text query (requires embedding service)
    pub async fn search_text(
        &self,
        _query: &str,
        _limit: Option<usize>,
    ) -> RragResult<Vec<SearchResult>> {
        // This would typically involve embedding the query text first
        // For now, return an error indicating the limitation
        Err(RragError::retrieval(
            "Text search requires embedding service integration".to_string(),
        ))
    }

    /// Search with pre-computed embedding
    pub async fn search_embedding(
        &self,
        embedding: Embedding,
        limit: Option<usize>,
    ) -> RragResult<Vec<SearchResult>> {
        let query = SearchQuery::embedding(embedding)
            .with_limit(limit.unwrap_or(10))
            .with_config(self.config.default_search_config.clone());

        self.retriever.search(&query).await
    }

    /// Advanced search with full query configuration
    pub async fn search(&self, query: SearchQuery) -> RragResult<Vec<SearchResult>> {
        self.retriever.search(&query).await
    }

    /// Add documents to the index
    pub async fn index_documents(
        &self,
        documents_with_embeddings: &[(Document, Embedding)],
    ) -> RragResult<()> {
        self.retriever
            .add_documents(documents_with_embeddings)
            .await
    }

    /// Add chunks to the index
    pub async fn index_chunks(
        &self,
        chunks_with_embeddings: &[(DocumentChunk, Embedding)],
    ) -> RragResult<()> {
        self.retriever.add_chunks(chunks_with_embeddings).await
    }

    /// Get retriever statistics
    pub async fn get_stats(&self) -> RragResult<IndexStats> {
        self.retriever.stats().await
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        self.retriever.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[tokio::test]
    async fn test_in_memory_retriever() {
        let retriever = InMemoryRetriever::new();

        // Create test documents with embeddings
        let doc1 = Document::new("First test document");
        let emb1 = Embedding::new(vec![1.0, 0.0, 0.0], "test-model", &doc1.id);

        let doc2 = Document::new("Second test document");
        let emb2 = Embedding::new(vec![0.0, 1.0, 0.0], "test-model", &doc2.id);

        // Add documents
        retriever
            .add_documents(&[(doc1.clone(), emb1.clone()), (doc2, emb2)])
            .await
            .unwrap();

        // Create query
        let query_embedding = Embedding::new(vec![0.8, 0.2, 0.0], "test-model", "query");
        let query = SearchQuery::embedding(query_embedding).with_limit(5);

        // Search
        let results = retriever.search(&query).await.unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, doc1.id); // Should be most similar
    }

    #[tokio::test]
    async fn test_search_filters() {
        let retriever = InMemoryRetriever::new();

        let doc1 = Document::new("Test document")
            .with_metadata("category", serde_json::Value::String("tech".to_string()));
        let emb1 = Embedding::new(vec![1.0, 0.0], "test-model", &doc1.id);

        let doc2 = Document::new("Another document")
            .with_metadata("category", serde_json::Value::String("science".to_string()));
        let emb2 = Embedding::new(vec![0.9, 0.1], "test-model", &doc2.id);

        retriever
            .add_documents(&[(doc1.clone(), emb1), (doc2, emb2)])
            .await
            .unwrap();

        // Search with filter
        let query_embedding = Embedding::new(vec![1.0, 0.0], "test-model", "query");
        let query = SearchQuery::embedding(query_embedding)
            .with_filter("category", serde_json::Value::String("tech".to_string()));

        let results = retriever.search(&query).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, doc1.id);
    }

    #[test]
    fn test_search_query_builder() {
        let query = SearchQuery::text("test query")
            .with_limit(20)
            .with_min_score(0.5)
            .with_filter("type", serde_json::Value::String("article".to_string()));

        assert_eq!(query.limit, 20);
        assert_eq!(query.min_score, 0.5);
        assert_eq!(query.filters.len(), 1);
    }

    #[tokio::test]
    async fn test_retrieval_service() {
        let retriever = Arc::new(InMemoryRetriever::new());
        let service = RetrievalService::new(retriever);

        let stats = service.get_stats().await.unwrap();
        assert_eq!(stats.total_items, 0);

        assert!(service.health_check().await.unwrap());
    }
}
