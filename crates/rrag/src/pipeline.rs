//! # RRAG Pipeline System
//! 
//! Composable processing pipelines with Rust's async patterns and zero-cost abstractions.
//! Designed for building complex RAG workflows from simple, reusable components.

use crate::{
    RragError, RragResult, Document, DocumentChunk, Embedding,
    EmbeddingService, RetrievalService, StorageService, DocumentChunker,
    SearchResult
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Pipeline execution context carrying data and metadata
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Execution ID for tracking
    pub execution_id: String,
    
    /// Input data for the pipeline
    pub data: PipelineData,
    
    /// Execution metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Step execution history
    pub execution_history: Vec<StepExecution>,
    
    /// Pipeline configuration
    pub config: PipelineConfig,
}

/// Data flowing through the pipeline
#[derive(Debug, Clone)]
pub enum PipelineData {
    /// Raw text input
    Text(String),
    
    /// Document input
    Document(Document),
    
    /// Multiple documents
    Documents(Vec<Document>),
    
    /// Document chunks
    Chunks(Vec<DocumentChunk>),
    
    /// Embeddings
    Embeddings(Vec<Embedding>),
    
    /// Search results
    SearchResults(Vec<SearchResult>),
    
    /// JSON data
    Json(serde_json::Value),
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum execution time in seconds
    pub max_execution_time: u64,
    
    /// Whether to continue on step errors
    pub continue_on_error: bool,
    
    /// Parallel execution where possible
    pub enable_parallelism: bool,
    
    /// Maximum parallel steps
    pub max_parallel_steps: usize,
    
    /// Enable step caching
    pub enable_caching: bool,
    
    /// Custom configuration
    pub custom_config: HashMap<String, serde_json::Value>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_execution_time: 300, // 5 minutes
            continue_on_error: false,
            enable_parallelism: true,
            max_parallel_steps: 4,
            enable_caching: false,
            custom_config: HashMap::new(),
        }
    }
}

/// Step execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecution {
    /// Step name/ID
    pub step_id: String,
    
    /// Execution start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    
    /// Whether step succeeded
    pub success: bool,
    
    /// Error message if failed
    pub error_message: Option<String>,
    
    /// Step metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PipelineContext {
    /// Create new pipeline context
    pub fn new(data: PipelineData) -> Self {
        Self {
            execution_id: uuid::Uuid::new_v4().to_string(),
            data,
            metadata: HashMap::new(),
            execution_history: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(data: PipelineData, config: PipelineConfig) -> Self {
        Self {
            execution_id: uuid::Uuid::new_v4().to_string(),
            data,
            metadata: HashMap::new(),
            execution_history: Vec::new(),
            config,
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Record step execution
    pub fn record_step(&mut self, step_execution: StepExecution) {
        self.execution_history.push(step_execution);
    }

    /// Get total execution time
    pub fn total_execution_time(&self) -> u64 {
        self.execution_history.iter().map(|step| step.duration_ms).sum()
    }

    /// Check if any step failed
    pub fn has_failures(&self) -> bool {
        self.execution_history.iter().any(|step| !step.success)
    }
}

/// Core pipeline step trait
#[async_trait]
pub trait PipelineStep: Send + Sync {
    /// Step name/identifier
    fn name(&self) -> &str;
    
    /// Step description
    fn description(&self) -> &str;
    
    /// Input data types this step accepts
    fn input_types(&self) -> Vec<&'static str>;
    
    /// Output data type this step produces
    fn output_type(&self) -> &'static str;
    
    /// Execute the step
    async fn execute(&self, context: PipelineContext) -> RragResult<PipelineContext>;
    
    /// Validate input data
    fn validate_input(&self, _data: &PipelineData) -> RragResult<()> {
        // Default implementation - override for custom validation
        Ok(())
    }
    
    /// Whether this step can run in parallel with others
    fn is_parallelizable(&self) -> bool {
        true
    }
    
    /// Dependencies on other steps (step names)
    fn dependencies(&self) -> Vec<&str> {
        Vec::new()
    }
}

/// Text preprocessing step
pub struct TextPreprocessingStep {
    /// Preprocessing operations to apply
    operations: Vec<TextOperation>,
}

#[derive(Debug, Clone)]
pub enum TextOperation {
    /// Convert to lowercase
    ToLowercase,
    
    /// Remove extra whitespace
    NormalizeWhitespace,
    
    /// Remove special characters
    RemoveSpecialChars,
    
    /// Custom regex replacement
    RegexReplace { pattern: String, replacement: String },
}

impl TextPreprocessingStep {
    pub fn new(operations: Vec<TextOperation>) -> Self {
        Self { operations }
    }

    fn process_text(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        for operation in &self.operations {
            result = match operation {
                TextOperation::ToLowercase => result.to_lowercase(),
                TextOperation::NormalizeWhitespace => {
                    result.split_whitespace().collect::<Vec<_>>().join(" ")
                }
                TextOperation::RemoveSpecialChars => {
                    result.chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect()
                }
                TextOperation::RegexReplace { pattern, replacement } => {
                    // Simple implementation - in production would use regex crate
                    result.replace(pattern, replacement)
                }
            };
        }
        
        result
    }
}

#[async_trait]
impl PipelineStep for TextPreprocessingStep {
    fn name(&self) -> &str {
        "text_preprocessing"
    }

    fn description(&self) -> &str {
        "Preprocesses text data with various normalization operations"
    }

    fn input_types(&self) -> Vec<&'static str> {
        vec!["Text", "Document", "Documents"]
    }

    fn output_type(&self) -> &'static str {
        "Text|Document|Documents"
    }

    async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
        let start_time = Instant::now();
        let step_start = chrono::Utc::now();
        
        let processed_data = match &context.data {
            PipelineData::Text(text) => {
                PipelineData::Text(self.process_text(text))
            }
            PipelineData::Document(doc) => {
                // Create a new document with processed content
                let processed_content = self.process_text(doc.content_str());
                let mut new_doc = Document::new(processed_content);
                new_doc.id = doc.id.clone();
                new_doc.metadata = doc.metadata.clone();
                new_doc.content_hash = doc.content_hash.clone();
                new_doc.created_at = doc.created_at;
                PipelineData::Document(new_doc)
            }
            PipelineData::Documents(docs) => {
                let processed_docs: Vec<Document> = docs.iter().map(|doc| {
                    let processed_content = self.process_text(doc.content_str());
                    let mut new_doc = Document::new(processed_content);
                    new_doc.id = doc.id.clone();
                    new_doc.metadata = doc.metadata.clone();
                    new_doc.content_hash = doc.content_hash.clone();
                    new_doc.created_at = doc.created_at;
                    new_doc
                }).collect();
                PipelineData::Documents(processed_docs)
            }
            _ => {
                let error = "Input must be Text, Document, or Documents";
                context.record_step(StepExecution {
                    step_id: self.name().to_string(),
                    start_time: step_start,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    success: false,
                    error_message: Some(error.to_string()),
                    metadata: HashMap::new(),
                });
                return Err(RragError::document_processing(error));
            }
        };
        
        context.data = processed_data;
        
        context.record_step(StepExecution {
            step_id: self.name().to_string(),
            start_time: step_start,
            duration_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        });
        
        Ok(context)
    }
}

/// Document chunking step
pub struct DocumentChunkingStep {
    /// Document chunker instance
    chunker: DocumentChunker,
}

impl DocumentChunkingStep {
    pub fn new(chunker: DocumentChunker) -> Self {
        Self { chunker }
    }
}

#[async_trait]
impl PipelineStep for DocumentChunkingStep {
    fn name(&self) -> &str {
        "document_chunking"
    }

    fn description(&self) -> &str {
        "Splits documents into smaller chunks for processing"
    }

    fn input_types(&self) -> Vec<&'static str> {
        vec!["Document", "Documents"]
    }

    fn output_type(&self) -> &'static str {
        "Chunks"
    }

    async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
        let start_time = Instant::now();
        let step_start = chrono::Utc::now();
        
        let chunks = match &context.data {
            PipelineData::Document(doc) => {
                self.chunker.chunk_document(doc)?
            }
            PipelineData::Documents(docs) => {
                let mut all_chunks = Vec::new();
                for doc in docs {
                    all_chunks.extend(self.chunker.chunk_document(doc)?);
                }
                all_chunks
            }
            _ => {
                let error = "Input must be Document or Documents";
                context.record_step(StepExecution {
                    step_id: self.name().to_string(),
                    start_time: step_start,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    success: false,
                    error_message: Some(error.to_string()),
                    metadata: HashMap::new(),
                });
                return Err(RragError::document_processing(error));
            }
        };
        
        context.data = PipelineData::Chunks(chunks);
        
        context.record_step(StepExecution {
            step_id: self.name().to_string(),
            start_time: step_start,
            duration_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        });
        
        Ok(context)
    }
}

/// Embedding generation step
pub struct EmbeddingStep {
    /// Embedding service
    embedding_service: Arc<EmbeddingService>,
}

impl EmbeddingStep {
    pub fn new(embedding_service: Arc<EmbeddingService>) -> Self {
        Self { embedding_service }
    }
}

#[async_trait]
impl PipelineStep for EmbeddingStep {
    fn name(&self) -> &str {
        "embedding_generation"
    }

    fn description(&self) -> &str {
        "Generates embeddings for documents or chunks"
    }

    fn input_types(&self) -> Vec<&'static str> {
        vec!["Document", "Documents", "Chunks"]
    }

    fn output_type(&self) -> &'static str {
        "Embeddings"
    }

    async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
        let start_time = Instant::now();
        let step_start = chrono::Utc::now();
        
        let embeddings = match &context.data {
            PipelineData::Document(doc) => {
                vec![self.embedding_service.embed_document(doc).await?]
            }
            PipelineData::Documents(docs) => {
                self.embedding_service.embed_documents(docs).await?
            }
            PipelineData::Chunks(chunks) => {
                self.embedding_service.embed_chunks(chunks).await?
            }
            _ => {
                let error = "Input must be Document, Documents, or Chunks";
                context.record_step(StepExecution {
                    step_id: self.name().to_string(),
                    start_time: step_start,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    success: false,
                    error_message: Some(error.to_string()),
                    metadata: HashMap::new(),
                });
                return Err(RragError::embedding("pipeline", error));
            }
        };
        
        context.data = PipelineData::Embeddings(embeddings);
        
        context.record_step(StepExecution {
            step_id: self.name().to_string(),
            start_time: step_start,
            duration_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        });
        
        Ok(context)
    }
}

/// Retrieval step for similarity search
pub struct RetrievalStep {
    /// Retrieval service
    retrieval_service: Arc<RetrievalService>,
    
    /// Search configuration
    search_config: SearchStepConfig,
}

#[derive(Debug, Clone)]
pub struct SearchStepConfig {
    /// Number of results to retrieve
    pub limit: usize,
    
    /// Minimum similarity threshold
    pub min_score: f32,
    
    /// Search query text (if not using embeddings)
    pub query_text: Option<String>,
}

impl Default for SearchStepConfig {
    fn default() -> Self {
        Self {
            limit: 10,
            min_score: 0.0,
            query_text: None,
        }
    }
}

impl RetrievalStep {
    pub fn new(retrieval_service: Arc<RetrievalService>) -> Self {
        Self {
            retrieval_service,
            search_config: SearchStepConfig::default(),
        }
    }

    pub fn with_config(retrieval_service: Arc<RetrievalService>, config: SearchStepConfig) -> Self {
        Self {
            retrieval_service,
            search_config: config,
        }
    }
}

#[async_trait]
impl PipelineStep for RetrievalStep {
    fn name(&self) -> &str {
        "similarity_retrieval"
    }

    fn description(&self) -> &str {
        "Performs similarity search using embeddings"
    }

    fn input_types(&self) -> Vec<&'static str> {
        vec!["Embeddings"]
    }

    fn output_type(&self) -> &'static str {
        "SearchResults"
    }

    async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
        let start_time = Instant::now();
        let step_start = chrono::Utc::now();
        
        let search_results = match &context.data {
            PipelineData::Embeddings(embeddings) => {
                if embeddings.is_empty() {
                    Vec::new()
                } else {
                    // Use the first embedding as query (could be enhanced)
                    let query_embedding = embeddings[0].clone();
                    self.retrieval_service
                        .search_embedding(query_embedding, Some(self.search_config.limit))
                        .await?
                }
            }
            _ => {
                let error = "Input must be Embeddings";
                context.record_step(StepExecution {
                    step_id: self.name().to_string(),
                    start_time: step_start,
                    duration_ms: start_time.elapsed().as_millis() as u64,
                    success: false,
                    error_message: Some(error.to_string()),
                    metadata: HashMap::new(),
                });
                return Err(RragError::retrieval(error));
            }
        };
        
        context.data = PipelineData::SearchResults(search_results);
        
        context.record_step(StepExecution {
            step_id: self.name().to_string(),
            start_time: step_start,
            duration_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        });
        
        Ok(context)
    }
}

/// Pipeline for composing multiple steps
pub struct Pipeline {
    /// Pipeline steps in execution order
    steps: Vec<Arc<dyn PipelineStep>>,
    
    /// Pipeline configuration
    config: PipelineConfig,
    
    /// Pipeline metadata
    metadata: HashMap<String, serde_json::Value>,
}

impl Pipeline {
    /// Create new pipeline
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            config: PipelineConfig::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            steps: Vec::new(),
            config,
            metadata: HashMap::new(),
        }
    }

    /// Add a step to the pipeline
    pub fn add_step(mut self, step: Arc<dyn PipelineStep>) -> Self {
        self.steps.push(step);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Execute the pipeline
    pub async fn execute(&self, initial_data: PipelineData) -> RragResult<PipelineContext> {
        let mut context = PipelineContext::with_config(initial_data, self.config.clone());
        
        // Add pipeline metadata to context
        context.metadata.extend(self.metadata.clone());
        
        let start_time = Instant::now();
        
        for step in &self.steps {
            // Check timeout
            if start_time.elapsed().as_secs() > self.config.max_execution_time {
                return Err(RragError::timeout("pipeline_execution", self.config.max_execution_time * 1000));
            }
            
            // Validate input
            if let Err(e) = step.validate_input(&context.data) {
                if !self.config.continue_on_error {
                    return Err(e);
                }
                // Record error and continue
                context.record_step(StepExecution {
                    step_id: step.name().to_string(),
                    start_time: chrono::Utc::now(),
                    duration_ms: 0,
                    success: false,
                    error_message: Some(e.to_string()),
                    metadata: HashMap::new(),
                });
                continue;
            }
            
            // Execute step (clone context to satisfy borrow checker)
            let context_clone = PipelineContext {
                execution_id: context.execution_id.clone(),
                data: context.data.clone(),
                metadata: context.metadata.clone(),
                execution_history: context.execution_history.clone(),
                config: context.config.clone(),
            };
            
            match step.execute(context_clone).await {
                Ok(new_context) => {
                    context = new_context;
                }
                Err(e) => {
                    if !self.config.continue_on_error {
                        return Err(e);
                    }
                    // Record error and continue with unchanged context
                    context.record_step(StepExecution {
                        step_id: step.name().to_string(),
                        start_time: chrono::Utc::now(),
                        duration_ms: 0,
                        success: false,
                        error_message: Some(e.to_string()),
                        metadata: HashMap::new(),
                    });
                }
            }
        }
        
        Ok(context)
    }

    /// Get pipeline step information
    pub fn get_step_info(&self) -> Vec<PipelineStepInfo> {
        self.steps
            .iter()
            .map(|step| PipelineStepInfo {
                name: step.name().to_string(),
                description: step.description().to_string(),
                input_types: step.input_types().iter().map(|s| s.to_string()).collect(),
                output_type: step.output_type().to_string(),
                is_parallelizable: step.is_parallelizable(),
                dependencies: step.dependencies().iter().map(|s| s.to_string()).collect(),
            })
            .collect()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline step information for introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStepInfo {
    pub name: String,
    pub description: String,
    pub input_types: Vec<String>,
    pub output_type: String,
    pub is_parallelizable: bool,
    pub dependencies: Vec<String>,
}

/// Pre-built pipeline builder for common RAG workflows
pub struct RagPipelineBuilder {
    /// Embedding service
    embedding_service: Option<Arc<EmbeddingService>>,
    
    /// Retrieval service
    retrieval_service: Option<Arc<RetrievalService>>,
    
    /// Storage service
    storage_service: Option<Arc<StorageService>>,
    
    /// Pipeline configuration
    config: PipelineConfig,
}

impl RagPipelineBuilder {
    pub fn new() -> Self {
        Self {
            embedding_service: None,
            retrieval_service: None,
            storage_service: None,
            config: PipelineConfig::default(),
        }
    }

    pub fn with_embedding_service(mut self, service: Arc<EmbeddingService>) -> Self {
        self.embedding_service = Some(service);
        self
    }

    pub fn with_retrieval_service(mut self, service: Arc<RetrievalService>) -> Self {
        self.retrieval_service = Some(service);
        self
    }

    pub fn with_storage_service(mut self, service: Arc<StorageService>) -> Self {
        self.storage_service = Some(service);
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Build document ingestion pipeline
    pub fn build_ingestion_pipeline(&self) -> RragResult<Pipeline> {
        let embedding_service = self.embedding_service
            .as_ref()
            .ok_or_else(|| RragError::config("embedding_service", "required", "missing"))?;

        let pipeline = Pipeline::with_config(self.config.clone())
            .add_step(Arc::new(TextPreprocessingStep::new(vec![
                TextOperation::NormalizeWhitespace,
                TextOperation::ToLowercase,
            ])))
            .add_step(Arc::new(DocumentChunkingStep::new(DocumentChunker::new())))
            .add_step(Arc::new(EmbeddingStep::new(embedding_service.clone())));

        Ok(pipeline)
    }

    /// Build query pipeline for search
    pub fn build_query_pipeline(&self) -> RragResult<Pipeline> {
        let embedding_service = self.embedding_service
            .as_ref()
            .ok_or_else(|| RragError::config("embedding_service", "required", "missing"))?;
        
        let retrieval_service = self.retrieval_service
            .as_ref()
            .ok_or_else(|| RragError::config("retrieval_service", "required", "missing"))?;

        let pipeline = Pipeline::with_config(self.config.clone())
            .add_step(Arc::new(TextPreprocessingStep::new(vec![
                TextOperation::NormalizeWhitespace,
            ])))
            .add_step(Arc::new(EmbeddingStep::new(embedding_service.clone())))
            .add_step(Arc::new(RetrievalStep::new(retrieval_service.clone())));

        Ok(pipeline)
    }
}

impl Default for RagPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Document, InMemoryRetriever, LocalEmbeddingProvider, EmbeddingService};

    #[tokio::test]
    async fn test_text_preprocessing_step() {
        let step = TextPreprocessingStep::new(vec![
            TextOperation::ToLowercase,
            TextOperation::NormalizeWhitespace,
        ]);
        
        let context = PipelineContext::new(PipelineData::Text("  HELLO    WORLD  ".to_string()));
        let result = step.execute(context).await.unwrap();
        
        if let PipelineData::Text(processed) = result.data {
            assert_eq!(processed, "hello world");
        } else {
            panic!("Expected Text output");
        }
        
        assert!(result.execution_history[0].success);
    }

    #[tokio::test]
    async fn test_document_chunking_step() {
        let step = DocumentChunkingStep::new(DocumentChunker::new());
        
        let doc = Document::new("This is a test document with some content that should be chunked appropriately.");
        let context = PipelineContext::new(PipelineData::Document(doc));
        
        let result = step.execute(context).await.unwrap();
        
        if let PipelineData::Chunks(chunks) = result.data {
            assert!(!chunks.is_empty());
        } else {
            panic!("Expected Chunks output");
        }
    }

    #[tokio::test]
    async fn test_embedding_step() {
        let provider = Arc::new(LocalEmbeddingProvider::new("test-model", 128));
        let embedding_service = Arc::new(EmbeddingService::new(provider));
        let step = EmbeddingStep::new(embedding_service);
        
        let doc = Document::new("Test document for embedding");
        let context = PipelineContext::new(PipelineData::Document(doc));
        
        let result = step.execute(context).await.unwrap();
        
        if let PipelineData::Embeddings(embeddings) = result.data {
            assert_eq!(embeddings.len(), 1);
            assert_eq!(embeddings[0].dimensions, 128);
        } else {
            panic!("Expected Embeddings output");
        }
    }

    #[tokio::test]
    async fn test_pipeline_execution() {
        let provider = Arc::new(LocalEmbeddingProvider::new("test-model", 128));
        let embedding_service = Arc::new(EmbeddingService::new(provider));
        
        let pipeline = Pipeline::new()
            .add_step(Arc::new(TextPreprocessingStep::new(vec![
                TextOperation::ToLowercase,
            ])))
            .add_step(Arc::new(EmbeddingStep::new(embedding_service)));
        
        let doc = Document::new("TEST DOCUMENT");
        let result = pipeline.execute(PipelineData::Document(doc)).await.unwrap();
        
        // Should have executed 2 steps
        assert_eq!(result.execution_history.len(), 2);
        assert!(result.execution_history.iter().all(|step| step.success));
        
        // Final output should be embeddings
        if let PipelineData::Embeddings(embeddings) = result.data {
            assert_eq!(embeddings.len(), 1);
        } else {
            panic!("Expected Embeddings output");
        }
    }

    #[tokio::test]
    async fn test_rag_pipeline_builder() {
        let provider = Arc::new(LocalEmbeddingProvider::new("test-model", 128));
        let embedding_service = Arc::new(EmbeddingService::new(provider));
        
        let builder = RagPipelineBuilder::new()
            .with_embedding_service(embedding_service);
        
        let pipeline = builder.build_ingestion_pipeline().unwrap();
        let step_info = pipeline.get_step_info();
        
        assert_eq!(step_info.len(), 3); // preprocessing, chunking, embedding
        assert_eq!(step_info[0].name, "text_preprocessing");
        assert_eq!(step_info[1].name, "document_chunking");
        assert_eq!(step_info[2].name, "embedding_generation");
    }

    #[test]
    fn test_pipeline_context() {
        let mut context = PipelineContext::new(PipelineData::Text("test".to_string()))
            .with_metadata("test_key", serde_json::Value::String("test_value".to_string()));
        
        assert_eq!(context.metadata.get("test_key").unwrap().as_str().unwrap(), "test_value");
        
        let step_execution = StepExecution {
            step_id: "test_step".to_string(),
            start_time: chrono::Utc::now(),
            duration_ms: 100,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        };
        
        context.record_step(step_execution);
        
        assert_eq!(context.execution_history.len(), 1);
        assert_eq!(context.total_execution_time(), 100);
        assert!(!context.has_failures());
    }
}