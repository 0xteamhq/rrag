//! # Semantic Vector Search
//!
//! High-performance semantic search using vector embeddings and similarity metrics.
//! Supports multiple similarity algorithms and optimization techniques.

use crate::{Document, Embedding, EmbeddingProvider, RragResult, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Semantic retriever configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Similarity metric to use
    pub similarity_metric: SimilarityMetric,

    /// Embedding dimension
    pub embedding_dimension: usize,

    /// Whether to normalize embeddings
    pub normalize_embeddings: bool,

    /// Index type for efficient search
    pub index_type: IndexType,

    /// Number of clusters for IVF index
    pub num_clusters: Option<usize>,

    /// Number of probes for IVF search
    pub num_probes: Option<usize>,

    /// Enable GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            similarity_metric: SimilarityMetric::Cosine,
            embedding_dimension: 768,
            normalize_embeddings: true,
            index_type: IndexType::Flat,
            num_clusters: None,
            num_probes: None,
            use_gpu: false,
        }
    }
}

/// Similarity metrics for vector comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity (angle between vectors)
    Cosine,
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Dot product (inner product)
    DotProduct,
    /// Manhattan distance (L1 norm)
    Manhattan,
}

/// Index types for efficient search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Flat index (brute force search)
    Flat,
    /// Inverted File Index (clustering-based)
    IVF,
    /// Hierarchical Navigable Small World
    HNSW,
    /// Locality Sensitive Hashing
    LSH,
}

/// Vector document for semantic search
#[derive(Debug, Clone)]
struct VectorDocument {
    /// Document ID
    id: String,

    /// Original content
    content: String,

    /// Document embedding
    embedding: Embedding,

    /// Normalized embedding (if applicable)
    normalized_embedding: Option<Vec<f32>>,

    /// Metadata
    metadata: HashMap<String, serde_json::Value>,
}

/// Semantic retriever implementation
pub struct SemanticRetriever {
    /// Configuration
    config: SemanticConfig,

    /// Document storage
    documents: Arc<RwLock<HashMap<String, VectorDocument>>>,

    /// Embedding service
    embedding_service: Arc<dyn EmbeddingProvider>,

    /// Index for efficient search (simplified for this example)
    index: Arc<RwLock<VectorIndex>>,
}

/// Simplified vector index
struct VectorIndex {
    /// Document IDs in order
    doc_ids: Vec<String>,

    /// Embeddings matrix (row-major)
    embeddings: Vec<Vec<f32>>,

    /// Index type
    index_type: IndexType,
}

impl SemanticRetriever {
    /// Create a new semantic retriever
    pub fn new(config: SemanticConfig, embedding_service: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            config,
            documents: Arc::new(RwLock::new(HashMap::new())),
            embedding_service,
            index: Arc::new(RwLock::new(VectorIndex {
                doc_ids: Vec::new(),
                embeddings: Vec::new(),
                index_type: IndexType::Flat,
            })),
        }
    }

    /// Index a document with semantic embedding
    pub async fn index_document(&self, doc: &Document) -> RragResult<()> {
        // Generate embedding for the document
        let embedding = self.embedding_service.embed_text(&doc.content).await?;

        // Normalize if configured
        let normalized = if self.config.normalize_embeddings {
            Some(Self::normalize_vector(&embedding.vector))
        } else {
            None
        };

        let vector_doc = VectorDocument {
            id: doc.id.clone(),
            content: doc.content.to_string(),
            embedding: embedding.clone(),
            normalized_embedding: normalized,
            metadata: doc.metadata.clone(),
        };

        // Store document
        let mut documents = self.documents.write().await;
        documents.insert(doc.id.clone(), vector_doc);

        // Update index
        let mut index = self.index.write().await;
        index.doc_ids.push(doc.id.clone());
        index.embeddings.push(if self.config.normalize_embeddings {
            Self::normalize_vector(&embedding.vector)
        } else {
            embedding.vector
        });

        Ok(())
    }

    /// Search for similar documents
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        min_score: Option<f32>,
    ) -> RragResult<Vec<SearchResult>> {
        // Generate query embedding
        let query_embedding = self.embedding_service.embed_text(query).await?;

        let query_vector = if self.config.normalize_embeddings {
            Self::normalize_vector(&query_embedding.vector)
        } else {
            query_embedding.vector
        };

        // Perform search
        let index = self.index.read().await;
        let documents = self.documents.read().await;

        let mut scores: Vec<(String, f32)> = Vec::new();

        // Calculate similarities
        for (i, doc_embedding) in index.embeddings.iter().enumerate() {
            let similarity = self.calculate_similarity(&query_vector, doc_embedding);

            if let Some(threshold) = min_score {
                if similarity < threshold {
                    continue;
                }
            }

            scores.push((index.doc_ids[i].clone(), similarity));
        }

        // Sort by similarity
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(limit);

        // Build results
        let results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (doc_id, score))| {
                documents.get(&doc_id).map(|doc| SearchResult {
                    id: doc_id,
                    content: doc.content.clone(),
                    score,
                    rank,
                    metadata: doc.metadata.clone(),
                    embedding: Some(doc.embedding.clone()),
                })
            })
            .collect();

        Ok(results)
    }

    /// Search with pre-computed embedding
    pub async fn search_by_embedding(
        &self,
        embedding: &Embedding,
        limit: usize,
        min_score: Option<f32>,
    ) -> RragResult<Vec<SearchResult>> {
        let query_vector = if self.config.normalize_embeddings {
            Self::normalize_vector(&embedding.vector)
        } else {
            embedding.vector.clone()
        };

        let index = self.index.read().await;
        let documents = self.documents.read().await;

        let mut scores: Vec<(String, f32)> = Vec::new();

        for (i, doc_embedding) in index.embeddings.iter().enumerate() {
            let similarity = self.calculate_similarity(&query_vector, doc_embedding);

            if let Some(threshold) = min_score {
                if similarity < threshold {
                    continue;
                }
            }

            scores.push((index.doc_ids[i].clone(), similarity));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(limit);

        let results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (doc_id, score))| {
                documents.get(&doc_id).map(|doc| SearchResult {
                    id: doc_id,
                    content: doc.content.clone(),
                    score,
                    rank,
                    metadata: doc.metadata.clone(),
                    embedding: Some(doc.embedding.clone()),
                })
            })
            .collect();

        Ok(results)
    }

    /// Calculate similarity between two vectors
    fn calculate_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        match self.config.similarity_metric {
            SimilarityMetric::Cosine => Self::cosine_similarity(vec1, vec2),
            SimilarityMetric::Euclidean => {
                let distance = Self::euclidean_distance(vec1, vec2);
                1.0 / (1.0 + distance) // Convert distance to similarity
            }
            SimilarityMetric::DotProduct => Self::dot_product(vec1, vec2),
            SimilarityMetric::Manhattan => {
                let distance = Self::manhattan_distance(vec1, vec2);
                1.0 / (1.0 + distance) // Convert distance to similarity
            }
        }
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        let dot = Self::dot_product(vec1, vec2);
        let norm1 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }

    /// Dot product of two vectors
    fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
    }

    /// Euclidean distance between two vectors
    fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Manhattan distance between two vectors
    fn manhattan_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    /// Normalize a vector to unit length
    fn normalize_vector(vec: &[f32]) -> Vec<f32> {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            vec.to_vec()
        } else {
            vec.iter().map(|x| x / norm).collect()
        }
    }

    /// Batch index multiple documents
    pub async fn index_batch(&self, documents: Vec<Document>) -> RragResult<()> {
        // Generate embedding requests
        let requests: Vec<crate::EmbeddingRequest> = documents
            .iter()
            .map(|doc| crate::EmbeddingRequest::new(&doc.id, doc.content.as_ref()))
            .collect();

        let embedding_batch = self.embedding_service.embed_batch(requests).await?;

        let mut docs_map = self.documents.write().await;
        let mut index = self.index.write().await;

        for doc in documents.iter() {
            if let Some(embedding) = embedding_batch.embeddings.get(&doc.id) {
                let normalized = if self.config.normalize_embeddings {
                    Some(Self::normalize_vector(&embedding.vector))
                } else {
                    None
                };

                let vector_doc = VectorDocument {
                    id: doc.id.clone(),
                    content: doc.content.to_string(),
                    embedding: embedding.clone(),
                    normalized_embedding: normalized.clone(),
                    metadata: doc.metadata.clone(),
                };

                docs_map.insert(doc.id.clone(), vector_doc);
                index.doc_ids.push(doc.id.clone());
                index
                    .embeddings
                    .push(normalized.unwrap_or_else(|| embedding.vector.clone()));
            }
        }

        Ok(())
    }

    /// Clear the index
    pub async fn clear(&self) -> RragResult<()> {
        let mut documents = self.documents.write().await;
        let mut index = self.index.write().await;

        documents.clear();
        index.doc_ids.clear();
        index.embeddings.clear();

        Ok(())
    }

    /// Get index statistics
    pub async fn stats(&self) -> HashMap<String, serde_json::Value> {
        let documents = self.documents.read().await;
        let _index = self.index.read().await;

        let mut stats = HashMap::new();
        stats.insert("total_documents".to_string(), documents.len().into());
        stats.insert(
            "embedding_dimension".to_string(),
            self.config.embedding_dimension.into(),
        );
        stats.insert(
            "index_type".to_string(),
            format!("{:?}", self.config.index_type).into(),
        );
        stats.insert(
            "similarity_metric".to_string(),
            format!("{:?}", self.config.similarity_metric).into(),
        );

        let memory_size = documents.len() * self.config.embedding_dimension * 4; // 4 bytes per f32
        stats.insert("index_memory_bytes".to_string(), memory_size.into());

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingService;

    #[tokio::test]
    async fn test_semantic_search() {
        let mock_service = Arc::new(MockEmbeddingService::new());
        let retriever = SemanticRetriever::new(SemanticConfig::default(), mock_service);

        let docs = vec![
            Document::with_id(
                "1",
                "Machine learning is a subset of artificial intelligence",
            ),
            Document::with_id("2", "Deep learning uses neural networks"),
            Document::with_id(
                "3",
                "Natural language processing enables computers to understand text",
            ),
        ];

        retriever.index_batch(docs).await.unwrap();

        let results = retriever
            .search("AI and machine learning", 2, Some(0.5))
            .await
            .unwrap();
        assert!(!results.is_empty());
    }
}
