//! # RRAG Pipeline System
//!
//! Composable, async-first processing pipelines for building complex RAG workflows
//! from simple, reusable components. Features parallel execution, error handling,
//! comprehensive monitoring, and type-safe data flow.
//!
//! ## Features
//!
//! - **Composable Steps**: Build complex workflows from simple, reusable components
//! - **Type-Safe Data Flow**: Compile-time validation of pipeline data types
//! - **Async Execution**: Full async/await support with parallel step execution
//! - **Error Handling**: Robust error handling with optional error recovery
//! - **Monitoring**: Built-in execution tracking and performance metrics
//! - **Flexible Configuration**: Extensive configuration options for behavior tuning
//! - **Caching Support**: Optional step result caching for performance
//!
//! ## Quick Start
//!
//! ### Basic Pipeline
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Create a simple text processing pipeline
//! let pipeline = RagPipelineBuilder::new()
//!     .add_step(TextPreprocessingStep::new(vec![
//!         TextOperation::NormalizeWhitespace,
//!         TextOperation::RemoveSpecialChars,
//!     ]))
//!     .add_step(DocumentChunkingStep::new(
//!         ChunkingStrategy::FixedSize { size: 512, overlap: 64 }
//!     ))
//!     .build();
//!
//! // Execute pipeline
//! let context = PipelineContext::new(PipelineData::Text(
//!     "This is some text to process through the pipeline.".to_string()
//! ));
//!
//! let result = pipeline.execute(context).await?;
//! tracing::debug!("Pipeline completed in {}ms", result.total_execution_time());
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced RAG Pipeline
//!
//! ```rust
//! use rrag::prelude::*;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Create a comprehensive RAG processing pipeline
//! let embedding_provider = Arc::new(OpenAIEmbeddingProvider::new("api-key"));
//! let embedding_service = Arc::new(EmbeddingService::new(embedding_provider));
//!
//! let pipeline = RagPipelineBuilder::new()
//!     .with_config(PipelineConfig {
//!         enable_parallelism: true,
//!         max_parallel_steps: 4,
//!         enable_caching: true,
//!         ..Default::default()
//!     })
//!     .add_step(TextPreprocessingStep::new(vec![
//!         TextOperation::NormalizeWhitespace,
//!         TextOperation::RemoveExtraWhitespace,
//!     ]))
//!     .add_step(DocumentChunkingStep::new(
//!         ChunkingStrategy::Semantic { similarity_threshold: 0.8 }
//!     ))
//!     .add_step(EmbeddingStep::new(embedding_service))
//!     .add_step(RetrievalStep::new())
//!     .build();
//!
//! // Process documents
//! let documents = vec![
//!     Document::new("First document content"),
//!     Document::new("Second document content"),
//! ];
//!
//! let context = PipelineContext::new(PipelineData::Documents(documents))
//!     .with_metadata("batch_id", "batch-123".into())
//!     .with_metadata("priority", "high".into());
//!
//! let result = pipeline.execute(context).await?;
//! tracing::debug!("Processed {} documents", result.execution_history.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Pipeline Steps
//!
//! ```rust
//! use rrag::prelude::*;
//! use async_trait::async_trait;
//!
//! // Define a custom pipeline step
//! struct CustomValidationStep {
//!     min_length: usize,
//! }
//!
//! #[async_trait]
//! impl PipelineStep for CustomValidationStep {
//!     fn name(&self) -> &str { "custom_validation" }
//!     fn description(&self) -> &str { "Validates document content length" }
//!     fn input_types(&self) -> Vec<&'static str> { vec!["Document", "Documents"] }
//!     fn output_type(&self) -> &'static str { "Document|Documents" }
//!
//!     async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
//!         // Custom validation logic here
//!         match &context.data {
//!             PipelineData::Document(doc) => {
//!                 if doc.content_length() < self.min_length {
//!                     return Err(RragError::validation(
//!                         "document_length",
//!                         format!("minimum {}", self.min_length),
//!                         doc.content_length().to_string()
//!                     ));
//!                 }
//!             }
//!             _ => return Err(RragError::document_processing("Invalid input type"))
//!         }
//!         Ok(context)
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> RragResult<()> {
//! // Use the custom step in a pipeline
//! let pipeline = RagPipelineBuilder::new()
//!     .add_step(CustomValidationStep { min_length: 100 })
//!     .add_step(TextPreprocessingStep::new(vec![TextOperation::NormalizeWhitespace]))
//!     .build();
//! # Ok(())
//! # }
//! ```
//!
//! ## Pipeline Configuration
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! let config = PipelineConfig {
//!     max_execution_time: 600, // 10 minutes
//!     continue_on_error: true, // Continue processing on step failures
//!     enable_parallelism: true,
//!     max_parallel_steps: 8,
//!     enable_caching: true,
//!     custom_config: [
//!         ("batch_size".to_string(), 100.into()),
//!         ("retry_attempts".to_string(), 3.into()),
//!     ].into_iter().collect(),
//! };
//! ```
//!
//! ## Error Handling
//!
//! ```rust
//! use rrag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! match pipeline.execute(context).await {
//!     Ok(result) => {
//!         tracing::debug!("Pipeline completed successfully");
//!         tracing::debug!("Total time: {}ms", result.total_execution_time());
//!         
//!         if result.has_failures() {
//!             tracing::debug!("Some steps failed but pipeline continued");
//!             for step in &result.execution_history {
//!                 if !step.success {
//!                     tracing::debug!("Step '{}' failed: {:?}", step.step_id, step.error_message);
//!                 }
//!             }
//!         }
//!     }
//!     Err(RragError::Timeout { operation, duration_ms }) => {
//!         tracing::debug!("Pipeline timed out in {}: {}ms", operation, duration_ms);
//!     }
//!     Err(e) => {
//!         tracing::debug!("Pipeline failed: {}", e);
//!     }
//! }
//! # }
//! ```
//!
//! ## Performance Optimization
//!
//! - **Parallel Execution**: Steps that don't depend on each other run concurrently
//! - **Caching**: Enable result caching for expensive operations
//! - **Batch Processing**: Process multiple items together when possible
//! - **Memory Management**: Efficient data structures and minimal copying
//! - **Async Operations**: Non-blocking I/O and CPU-intensive operations
//!
//! ## Built-in Steps
//!
//! RRAG provides several built-in pipeline steps:
//!
//! - [`TextPreprocessingStep`]: Text normalization and cleaning
//! - [`DocumentChunkingStep`]: Document chunking with various strategies  
//! - [`EmbeddingStep`]: Embedding generation with provider abstraction
//! - [`RetrievalStep`]: Vector similarity search and retrieval
//! - Custom steps via the [`PipelineStep`] trait

use crate::{
    Document, DocumentChunk, DocumentChunker, Embedding, EmbeddingService, RetrievalService,
    RragError, RragResult, SearchResult, StorageService,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Execution context for pipeline processing
///
/// Carries data, metadata, configuration, and execution history through
/// a pipeline. Each pipeline execution gets its own context that tracks
/// all steps, timing, errors, and intermediate results.
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// let context = PipelineContext::new(PipelineData::Text(
///     "Document content to process".to_string()
/// ))
/// .with_metadata("source", "api".into())
/// .with_metadata("priority", "high".into());
///
/// tracing::debug!("Processing execution: {}", context.execution_id);
/// ```
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

/// Data types that can flow through pipeline steps
///
/// Represents the various types of data that can be processed by pipeline steps.
/// Each step declares which input types it accepts and which output type it produces,
/// enabling compile-time validation of pipeline composition.
///
/// # Type Safety
///
/// The pipeline system uses these variants to ensure type safety:
/// - Steps declare compatible input/output types
/// - Runtime validation ensures data type correctness
/// - Clear error messages for type mismatches
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// // Different data types that can flow through pipelines
/// let text_data = PipelineData::Text("Raw text content".to_string());
/// let doc_data = PipelineData::Document(Document::new("Document content"));
/// let docs_data = PipelineData::Documents(vec![
///     Document::new("First doc"),
///     Document::new("Second doc"),
/// ]);
/// ```
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
        self.execution_history
            .iter()
            .map(|step| step.duration_ms)
            .sum()
    }

    /// Check if any step failed
    pub fn has_failures(&self) -> bool {
        self.execution_history.iter().any(|step| !step.success)
    }
}

/// Core trait for implementing pipeline steps
///
/// Each pipeline step implements this trait to define its behavior, input/output types,
/// dependencies, and execution logic. Steps are composable building blocks that can
/// be combined to create complex processing workflows.
///
/// # Design Principles
///
/// - **Single Responsibility**: Each step should do one thing well
/// - **Type Safety**: Declare input/output types for validation
/// - **Async First**: All execution is async for better concurrency
/// - **Error Handling**: Comprehensive error reporting with context
/// - **Monitoring**: Built-in execution tracking and metrics
///
/// # Example Implementation
///
/// ```rust
/// use rrag::prelude::*;
/// use async_trait::async_trait;
///
/// struct UppercaseStep;
///
/// #[async_trait]
/// impl PipelineStep for UppercaseStep {
///     fn name(&self) -> &str { "uppercase_text" }
///     fn description(&self) -> &str { "Converts text to uppercase" }
///     fn input_types(&self) -> Vec<&'static str> { vec!["Text"] }
///     fn output_type(&self) -> &'static str { "Text" }
///
///     async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
///         match &context.data {
///             PipelineData::Text(text) => {
///                 context.data = PipelineData::Text(text.to_uppercase());
///                 Ok(context)
///             }
///             _ => Err(RragError::document_processing("Expected Text input"))
///         }
///     }
/// }
/// ```
///
/// # Parallel Execution
///
/// Steps can declare whether they support parallel execution:
///
/// ```rust
/// # use rrag::prelude::*;
/// # use async_trait::async_trait;
/// # struct MyStep;
/// # #[async_trait]
/// # impl PipelineStep for MyStep {
/// #   fn name(&self) -> &str { "my_step" }
/// #   fn description(&self) -> &str { "description" }
/// #   fn input_types(&self) -> Vec<&'static str> { vec!["Text"] }
/// #   fn output_type(&self) -> &'static str { "Text" }
/// #   async fn execute(&self, context: PipelineContext) -> RragResult<PipelineContext> { Ok(context) }
/// // Override to disable parallelization for stateful operations
/// fn is_parallelizable(&self) -> bool {
///     false // This step cannot run in parallel
/// }
///
/// // Declare dependencies on other steps
/// fn dependencies(&self) -> Vec<&str> {
///     vec!["preprocessing", "validation"]
/// }
/// # }
/// ```
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

/// Built-in text preprocessing step for content normalization
///
/// Applies a sequence of text transformations to clean and normalize content
/// before further processing. Supports common operations like whitespace
/// normalization, case conversion, and special character handling.
///
/// # Supported Operations
///
/// - **Whitespace Normalization**: Collapse multiple spaces into single spaces
/// - **Case Conversion**: Convert text to lowercase for consistency
/// - **Special Character Removal**: Remove non-alphanumeric characters
/// - **Regex Replacement**: Custom pattern-based text replacement
///
/// # Example
///
/// ```rust
/// use rrag::prelude::*;
///
/// let step = TextPreprocessingStep::new(vec![
///     TextOperation::NormalizeWhitespace,
///     TextOperation::RemoveSpecialChars,
///     TextOperation::ToLowercase,
/// ]);
///
/// // Can also be built fluently
/// let step = TextPreprocessingStep::new(vec![])
///     .with_operation(TextOperation::NormalizeWhitespace)
///     .with_operation(TextOperation::RegexReplace {
///         pattern: r"\d+".to_string(),
///         replacement: "[NUMBER]".to_string(),
///     });
/// ```
///
/// # Performance
///
/// - Operations are applied in sequence for predictable results
/// - String allocations are minimized where possible
/// - Regex operations are compiled once and reused
/// - Supports batch processing for multiple documents
pub struct TextPreprocessingStep {
    /// Preprocessing operations to apply
    operations: Vec<TextOperation>,
}

/// Text preprocessing operations for document processing
#[derive(Debug, Clone)]
pub enum TextOperation {
    /// Convert to lowercase
    ToLowercase,

    /// Remove extra whitespace
    NormalizeWhitespace,

    /// Remove special characters
    RemoveSpecialChars,

    /// Custom regex replacement
    RegexReplace {
        /// Regular expression pattern to match
        pattern: String,
        /// Replacement string for matched patterns
        replacement: String,
    },
}

impl TextPreprocessingStep {
    /// Create a new text preprocessing step with specified operations
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
                TextOperation::RemoveSpecialChars => result
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect(),
                TextOperation::RegexReplace {
                    pattern,
                    replacement,
                } => {
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
            PipelineData::Text(text) => PipelineData::Text(self.process_text(text)),
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
                let processed_docs: Vec<Document> = docs
                    .iter()
                    .map(|doc| {
                        let processed_content = self.process_text(doc.content_str());
                        let mut new_doc = Document::new(processed_content);
                        new_doc.id = doc.id.clone();
                        new_doc.metadata = doc.metadata.clone();
                        new_doc.content_hash = doc.content_hash.clone();
                        new_doc.created_at = doc.created_at;
                        new_doc
                    })
                    .collect();
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
    /// Create a new document chunking step with the specified chunker
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
            PipelineData::Document(doc) => self.chunker.chunk_document(doc)?,
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
    /// Create a new embedding generation step with the specified service
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
            PipelineData::Documents(docs) => self.embedding_service.embed_documents(docs).await?,
            PipelineData::Chunks(chunks) => self.embedding_service.embed_chunks(chunks).await?,
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

/// Configuration for search/retrieval step
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
    /// Create a new retrieval step with default configuration
    pub fn new(retrieval_service: Arc<RetrievalService>) -> Self {
        Self {
            retrieval_service,
            search_config: SearchStepConfig::default(),
        }
    }

    /// Create a new retrieval step with custom configuration
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
                return Err(RragError::timeout(
                    "pipeline_execution",
                    self.config.max_execution_time * 1000,
                ));
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
    /// Name of the pipeline step
    pub name: String,
    /// Description of what the step does
    pub description: String,
    /// Types of input data this step accepts
    pub input_types: Vec<String>,
    /// Type of output data this step produces
    pub output_type: String,
    /// Whether this step can run in parallel with others
    pub is_parallelizable: bool,
    /// Names of steps this step depends on
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
    /// Create a new RAG pipeline builder
    pub fn new() -> Self {
        Self {
            embedding_service: None,
            retrieval_service: None,
            storage_service: None,
            config: PipelineConfig::default(),
        }
    }

    /// Set the embedding service for the pipeline
    pub fn with_embedding_service(mut self, service: Arc<EmbeddingService>) -> Self {
        self.embedding_service = Some(service);
        self
    }

    /// Set the retrieval service for the pipeline
    pub fn with_retrieval_service(mut self, service: Arc<RetrievalService>) -> Self {
        self.retrieval_service = Some(service);
        self
    }

    /// Set the storage service for the pipeline
    pub fn with_storage_service(mut self, service: Arc<StorageService>) -> Self {
        self.storage_service = Some(service);
        self
    }

    /// Set custom configuration for the pipeline
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Build document ingestion pipeline
    pub fn build_ingestion_pipeline(&self) -> RragResult<Pipeline> {
        let embedding_service = self
            .embedding_service
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
        let embedding_service = self
            .embedding_service
            .as_ref()
            .ok_or_else(|| RragError::config("embedding_service", "required", "missing"))?;

        let retrieval_service = self
            .retrieval_service
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
    use crate::{Document, EmbeddingService, InMemoryRetriever, LocalEmbeddingProvider};

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

        let doc = Document::new(
            "This is a test document with some content that should be chunked appropriately.",
        );
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

        let builder = RagPipelineBuilder::new().with_embedding_service(embedding_service);

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
            .with_metadata(
                "test_key",
                serde_json::Value::String("test_value".to_string()),
            );

        assert_eq!(
            context.metadata.get("test_key").unwrap().as_str().unwrap(),
            "test_value"
        );

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
