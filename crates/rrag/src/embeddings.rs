//! # RRAG Embeddings System
//! 
//! Async-first embedding generation with pluggable providers and efficient batching.
//! Designed for Rust's zero-cost abstractions and async ecosystem.

use crate::{RragError, RragResult, Document, DocumentChunk};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Embedding vector type - optimized for common dimensions
pub type EmbeddingVector = Vec<f32>;

/// Embedding with associated metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The actual embedding vector
    pub vector: EmbeddingVector,
    
    /// Dimensions of the embedding
    pub dimensions: usize,
    
    /// Model used to generate this embedding
    pub model: String,
    
    /// Source content identifier
    pub source_id: String,
    
    /// Embedding metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Generation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Embedding {
    /// Create a new embedding
    pub fn new(
        vector: EmbeddingVector,
        model: impl Into<String>,
        source_id: impl Into<String>,
    ) -> Self {
        let dimensions = vector.len();
        Self {
            vector,
            dimensions,
            model: model.into(),
            source_id: source_id.into(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Add metadata using builder pattern
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Calculate cosine similarity with another embedding
    pub fn cosine_similarity(&self, other: &Embedding) -> RragResult<f32> {
        if self.dimensions != other.dimensions {
            return Err(RragError::embedding(
                "similarity_calculation",
                format!(
                    "Dimension mismatch: {} vs {}",
                    self.dimensions, other.dimensions
                ),
            ));
        }

        let dot_product: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Calculate Euclidean distance with another embedding
    pub fn euclidean_distance(&self, other: &Embedding) -> RragResult<f32> {
        if self.dimensions != other.dimensions {
            return Err(RragError::embedding(
                "distance_calculation",
                format!(
                    "Dimension mismatch: {} vs {}",
                    self.dimensions, other.dimensions
                ),
            ));
        }

        let distance: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        Ok(distance)
    }
}

/// Embedding request for batch processing
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Unique identifier for the request
    pub id: String,
    
    /// Text content to embed
    pub content: String,
    
    /// Optional metadata to attach to the embedding
    pub metadata: HashMap<String, serde_json::Value>,
}

impl EmbeddingRequest {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Batch embedding response
#[derive(Debug)]
pub struct EmbeddingBatch {
    /// Generated embeddings indexed by request ID
    pub embeddings: HashMap<String, Embedding>,
    
    /// Processing metadata
    pub metadata: BatchMetadata,
}

/// Metadata for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMetadata {
    /// Total items processed
    pub total_items: usize,
    
    /// Successfully processed items
    pub successful_items: usize,
    
    /// Failed items with error messages
    pub failed_items: HashMap<String, String>,
    
    /// Processing duration in milliseconds
    pub duration_ms: u64,
    
    /// Model used for embedding
    pub model: String,
    
    /// Provider used
    pub provider: String,
}

/// Core embedding provider trait
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Provider name (e.g., "openai", "huggingface")
    fn name(&self) -> &str;
    
    /// Supported models for this provider
    fn supported_models(&self) -> Vec<&str>;
    
    /// Maximum batch size supported
    fn max_batch_size(&self) -> usize;
    
    /// Embedding dimensions for the current model
    fn embedding_dimensions(&self) -> usize;
    
    /// Generate embedding for a single text
    async fn embed_text(&self, text: &str) -> RragResult<Embedding>;
    
    /// Generate embeddings for multiple texts (more efficient)
    async fn embed_batch(&self, requests: Vec<EmbeddingRequest>) -> RragResult<EmbeddingBatch>;
    
    /// Health check for the provider
    async fn health_check(&self) -> RragResult<bool>;
}

/// OpenAI embedding provider
#[allow(dead_code)]
pub struct OpenAIEmbeddingProvider {
    /// API client (placeholder - would use actual HTTP client)
    client: String, // In production: reqwest::Client or rsllm client
    
    /// Model to use for embeddings
    model: String,
    
    /// API key
    api_key: String,
    
    /// Request timeout
    timeout: std::time::Duration,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: "openai_client".to_string(), // Placeholder
            model: "text-embedding-ada-002".to_string(),
            api_key: api_key.into(),
            timeout: std::time::Duration::from_secs(30),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn supported_models(&self) -> Vec<&str> {
        vec![
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
    }

    fn max_batch_size(&self) -> usize {
        100 // OpenAI's current limit
    }

    fn embedding_dimensions(&self) -> usize {
        match self.model.as_str() {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => 1536, // Default fallback
        }
    }

    async fn embed_text(&self, text: &str) -> RragResult<Embedding> {
        // Mock implementation - in production, this would make actual API calls
        let start = std::time::Instant::now();
        
        // Simulate API delay
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        // Generate mock embedding vector
        let dimensions = self.embedding_dimensions();
        let vector: Vec<f32> = (0..dimensions)
            .map(|i| (text.len() as f32 + i as f32) / 1000.0)
            .collect();
        
        let embedding = Embedding::new(vector, &self.model, text)
            .with_metadata(
                "processing_time_ms",
                serde_json::Value::Number((start.elapsed().as_millis() as u64).into()),
            )
            .with_metadata("provider", serde_json::Value::String(self.name().to_string()));

        Ok(embedding)
    }

    async fn embed_batch(&self, requests: Vec<EmbeddingRequest>) -> RragResult<EmbeddingBatch> {
        let start = std::time::Instant::now();
        
        if requests.len() > self.max_batch_size() {
            return Err(RragError::embedding(
                "batch_processing",
                format!(
                    "Batch size {} exceeds maximum {}",
                    requests.len(),
                    self.max_batch_size()
                ),
            ));
        }

        let mut embeddings = HashMap::new();
        let mut failed_items = HashMap::new();
        let mut successful_count = 0;

        for request in requests.iter() {
            match self.embed_text(&request.content).await {
                Ok(mut embedding) => {
                    // Merge request metadata
                    embedding.metadata.extend(request.metadata.clone());
                    embedding.source_id = request.id.clone();
                    
                    embeddings.insert(request.id.clone(), embedding);
                    successful_count += 1;
                }
                Err(e) => {
                    failed_items.insert(request.id.clone(), e.to_string());
                }
            }
        }

        let batch = EmbeddingBatch {
            embeddings,
            metadata: BatchMetadata {
                total_items: requests.len(),
                successful_items: successful_count,
                failed_items,
                duration_ms: start.elapsed().as_millis() as u64,
                model: self.model.clone(),
                provider: self.name().to_string(),
            },
        };

        Ok(batch)
    }

    async fn health_check(&self) -> RragResult<bool> {
        // Mock health check - in production, this would ping the API
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        Ok(true)
    }
}

/// Local/Hugging Face embedding provider
pub struct LocalEmbeddingProvider {
    /// Model name or path
    model_path: String,
    
    /// Embedding dimensions
    dimensions: usize,
}

impl LocalEmbeddingProvider {
    pub fn new(model_path: impl Into<String>, dimensions: usize) -> Self {
        Self {
            model_path: model_path.into(),
            dimensions,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn supported_models(&self) -> Vec<&str> {
        vec!["sentence-transformers/all-MiniLM-L6-v2", "custom-local-model"]
    }

    fn max_batch_size(&self) -> usize {
        32 // Smaller batches for local processing
    }

    fn embedding_dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed_text(&self, text: &str) -> RragResult<Embedding> {
        // Mock local model inference
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        
        let vector: Vec<f32> = (0..self.dimensions)
            .map(|i| ((text.len() * 31 + i * 17) % 1000) as f32 / 1000.0)
            .collect();
        
        Ok(Embedding::new(vector, &self.model_path, text)
            .with_metadata("provider", serde_json::Value::String(self.name().to_string())))
    }

    async fn embed_batch(&self, requests: Vec<EmbeddingRequest>) -> RragResult<EmbeddingBatch> {
        let start = std::time::Instant::now();
        
        let mut embeddings = HashMap::new();
        let failed_items = HashMap::new();

        // Process in parallel for local models
        let futures: Vec<_> = requests
            .iter()
            .map(|req| async move {
                let embedding = self.embed_text(&req.content).await?;
                Ok::<_, RragError>((req.id.clone(), embedding))
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        
        for result in results {
            match result {
                Ok((id, embedding)) => {
                    embeddings.insert(id, embedding);
                }
                Err(_) => {
                    // Error handling would be more sophisticated in production
                }
            }
        }

        let successful_items = embeddings.len();
        let batch = EmbeddingBatch {
            embeddings,
            metadata: BatchMetadata {
                total_items: requests.len(),
                successful_items,
                failed_items,
                duration_ms: start.elapsed().as_millis() as u64,
                model: self.model_path.clone(),
                provider: self.name().to_string(),
            },
        };

        Ok(batch)
    }

    async fn health_check(&self) -> RragResult<bool> {
        // Check if model is loaded/accessible
        Ok(true)
    }
}

/// High-level embedding service with provider management
pub struct EmbeddingService {
    /// Active embedding provider
    provider: Arc<dyn EmbeddingProvider>,
    
    /// Service configuration
    config: EmbeddingConfig,
}

/// Configuration for embedding service
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Batch size for processing documents
    pub batch_size: usize,
    
    /// Whether to enable parallel processing
    pub parallel_processing: bool,
    
    /// Maximum retries for failed embeddings
    pub max_retries: usize,
    
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            parallel_processing: true,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}

impl EmbeddingService {
    /// Create embedding service with provider
    pub fn new(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            provider,
            config: EmbeddingConfig::default(),
        }
    }

    /// Create service with configuration
    pub fn with_config(provider: Arc<dyn EmbeddingProvider>, config: EmbeddingConfig) -> Self {
        Self { provider, config }
    }

    /// Embed a single document
    pub async fn embed_document(&self, document: &Document) -> RragResult<Embedding> {
        self.provider.embed_text(document.content_str()).await
    }

    /// Embed multiple documents with batching
    pub async fn embed_documents(&self, documents: &[Document]) -> RragResult<Vec<Embedding>> {
        let requests: Vec<EmbeddingRequest> = documents
            .iter()
            .map(|doc| EmbeddingRequest::new(&doc.id, doc.content_str()))
            .collect();

        let batches = self.create_batches(requests);
        let mut all_embeddings = Vec::new();

        for batch in batches {
            let batch_result = self.process_batch_with_retry(batch).await?;
            
            // Collect embeddings in original order
            for request_id in batch_result.embeddings.keys() {
                if let Some(embedding) = batch_result.embeddings.get(request_id) {
                    all_embeddings.push(embedding.clone());
                }
            }
        }

        Ok(all_embeddings)
    }

    /// Embed document chunks efficiently
    pub async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> RragResult<Vec<Embedding>> {
        let requests: Vec<EmbeddingRequest> = chunks
            .iter()
            .map(|chunk| {
                EmbeddingRequest::new(
                    format!("{}_{}", chunk.document_id, chunk.chunk_index),
                    &chunk.content,
                )
                .with_metadata("chunk_index", serde_json::Value::Number(chunk.chunk_index.into()))
                .with_metadata("document_id", serde_json::Value::String(chunk.document_id.clone()))
            })
            .collect();

        let batches = self.create_batches(requests);
        let mut all_embeddings = Vec::new();

        for batch in batches {
            let batch_result = self.process_batch_with_retry(batch).await?;
            
            for embedding in batch_result.embeddings.into_values() {
                all_embeddings.push(embedding);
            }
        }

        Ok(all_embeddings)
    }

    /// Create batches from requests
    fn create_batches(&self, requests: Vec<EmbeddingRequest>) -> Vec<Vec<EmbeddingRequest>> {
        requests
            .chunks(self.config.batch_size.min(self.provider.max_batch_size()))
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Process a batch with retry logic
    async fn process_batch_with_retry(
        &self,
        batch: Vec<EmbeddingRequest>,
    ) -> RragResult<EmbeddingBatch> {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.max_retries {
            match self.provider.embed_batch(batch.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;
                    
                    if attempts < self.config.max_retries {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            self.config.retry_delay_ms * attempts as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            RragError::embedding("batch_processing", "Max retries exceeded")
        }))
    }

    /// Get provider information
    pub fn provider_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: self.provider.name().to_string(),
            supported_models: self.provider.supported_models().iter().map(|s| s.to_string()).collect(),
            max_batch_size: self.provider.max_batch_size(),
            embedding_dimensions: self.provider.embedding_dimensions(),
        }
    }
}

/// Provider information for introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub name: String,
    pub supported_models: Vec<String>,
    pub max_batch_size: usize,
    pub embedding_dimensions: usize,
}

/// Mock embedding provider for testing
pub struct MockEmbeddingProvider {
    model: String,
    dimensions: usize,
}

impl MockEmbeddingProvider {
    pub fn new() -> Self {
        Self {
            model: "mock-model".to_string(),
            dimensions: 384,
        }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn name(&self) -> &str {
        "mock"
    }
    
    fn supported_models(&self) -> Vec<&str> {
        vec!["mock-model"]
    }
    
    fn max_batch_size(&self) -> usize {
        100
    }
    
    fn embedding_dimensions(&self) -> usize {
        self.dimensions
    }
    
    async fn embed_text(&self, text: &str) -> RragResult<Embedding> {
        // Generate a simple mock embedding based on text hash
        let hash = text.len() as f32;
        let mut vector = vec![0.0; self.dimensions];
        for i in 0..self.dimensions {
            vector[i] = (hash + i as f32).sin() / (i + 1) as f32;
        }
        
        Ok(Embedding::new(vector, &self.model, "mock"))
    }
    
    async fn embed_batch(&self, requests: Vec<EmbeddingRequest>) -> RragResult<EmbeddingBatch> {
        let mut embeddings = HashMap::new();
        
        for request in &requests {
            let embedding = self.embed_text(&request.content).await?;
            embeddings.insert(request.id.clone(), embedding);
        }
        
        Ok(EmbeddingBatch {
            embeddings,
            metadata: BatchMetadata {
                total_items: requests.len(),
                successful_items: requests.len(),
                failed_items: HashMap::new(),
                duration_ms: 10,
                model: self.model.clone(),
                provider: self.name().to_string(),
            },
        })
    }
    
    async fn health_check(&self) -> RragResult<bool> {
        Ok(true)
    }
}

// For backward compatibility
pub type MockEmbeddingService = MockEmbeddingProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_openai_provider() {
        let provider = OpenAIEmbeddingProvider::new("test-key");
        
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.embedding_dimensions(), 1536);
        assert!(provider.health_check().await.unwrap());
        
        let embedding = provider.embed_text("Hello, world!").await.unwrap();
        assert_eq!(embedding.dimensions, 1536);
        assert_eq!(embedding.model, "text-embedding-ada-002");
    }

    #[tokio::test]
    async fn test_local_provider() {
        let provider = LocalEmbeddingProvider::new("test-model", 384);
        
        assert_eq!(provider.name(), "local");
        assert_eq!(provider.embedding_dimensions(), 384);
        
        let embedding = provider.embed_text("Test content").await.unwrap();
        assert_eq!(embedding.dimensions, 384);
    }

    #[tokio::test]
    async fn test_embedding_service() {
        let provider = Arc::new(LocalEmbeddingProvider::new("test-model", 384));
        let service = EmbeddingService::new(provider);
        
        let doc = Document::new("Test document content");
        let embedding = service.embed_document(&doc).await.unwrap();
        
        assert_eq!(embedding.dimensions, 384);
        assert!(!embedding.vector.is_empty());
    }

    #[test]
    fn test_cosine_similarity() {
        let vector1 = vec![1.0, 0.0, 0.0];
        let vector2 = vec![0.0, 1.0, 0.0];
        let vector3 = vec![1.0, 0.0, 0.0];
        
        let emb1 = Embedding::new(vector1, "test", "1");
        let emb2 = Embedding::new(vector2, "test", "2");
        let emb3 = Embedding::new(vector3, "test", "3");
        
        let similarity_12 = emb1.cosine_similarity(&emb2).unwrap();
        let similarity_13 = emb1.cosine_similarity(&emb3).unwrap();
        
        assert!((similarity_12 - 0.0).abs() < 1e-6); // Orthogonal vectors
        assert!((similarity_13 - 1.0).abs() < 1e-6); // Identical vectors
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let provider = Arc::new(LocalEmbeddingProvider::new("test-model", 128));
        
        let requests = vec![
            EmbeddingRequest::new("1", "First text"),
            EmbeddingRequest::new("2", "Second text"),
            EmbeddingRequest::new("3", "Third text"),
        ];
        
        let batch_result = provider.embed_batch(requests).await.unwrap();
        
        assert_eq!(batch_result.metadata.total_items, 3);
        assert_eq!(batch_result.metadata.successful_items, 3);
        assert_eq!(batch_result.embeddings.len(), 3);
    }
}