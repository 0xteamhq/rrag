//! # RRAG - Rust RAG Framework
//! 
//! **RRAG** (Rust RAG) is a native Rust framework for building Retrieval-Augmented Generation 
//! applications with a focus on performance, type safety, and Rust ecosystem integration.
//! 
//! ## Design Philosophy
//! 
//! RRAG embraces Rust's core principles:
//! - **Zero-cost abstractions**: Compile-time optimizations with runtime efficiency
//! - **Memory safety**: Ownership system prevents data races and memory leaks  
//! - **Type safety**: Compile-time guarantees for correctness
//! - **Fearless concurrency**: Safe parallel processing with async/await
//! - **Ecosystem integration**: Works seamlessly with rsllm and other Rust crates
//! 
//! ## Architecture
//! 
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Documents     │───▶│   Processing    │───▶│   Vector Store  │
//! │   (Input)       │    │   Pipeline      │    │   (Storage)     │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!                                 │
//!                                 ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Responses     │◀───│     Agent       │◀───│    Retriever    │
//! │   (Output)      │    │   (rsllm)       │    │   (Search)      │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//! ```
//! 
//! ## Quick Start
//! 
//! ```rust
//! use rrag::prelude::*;
//! 
//! #[tokio::main]
//! async fn main() -> RragResult<()> {
//!     // Create a RAG system with rsllm integration
//!     let rag = RragSystem::builder()
//!         .with_rsllm_client("http://localhost:8080")
//!         .with_vector_store(InMemoryStore::new())
//!         .with_chunk_size(512)
//!         .build()
//!         .await?;
//!     
//!     // Add documents to the knowledge base
//!     rag.ingest_documents(vec![
//!         Document::new("Rust is a systems programming language..."),
//!         Document::new("RAG combines retrieval with generation..."),
//!     ]).await?;
//!     
//!     // Query the system
//!     let response = rag.query("What is Rust?").await?;
//!     println!("Response: {}", response.text);
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ## Features
//! 
//! - **rsllm Integration**: Native Rust LLM client support
//! - **Streaming**: Real-time token streaming with async iterators
//! - **Vector Storage**: Pluggable storage backends (in-memory, persistent)
//! - **Document Processing**: Efficient chunking and embedding pipelines
//! - **Agent Tools**: Type-safe tool calling with automatic schema generation
//! - **Memory Management**: Conversation state and context handling
//! - **Error Handling**: Comprehensive error types with context

// Core modules
pub mod error;
pub mod document;
pub mod embeddings;
pub mod retrieval_core;
pub mod storage;
pub mod memory;
pub mod tools;
pub mod streaming;
pub mod agent;
pub mod pipeline;
pub mod system;

// Retrieval submodules
#[path = "retrieval/mod.rs"]
pub mod retrieval_enhanced;

// Query processing module
pub mod query;

// Re-exports for convenience
pub use error::{RragError, RragResult, ErrorSeverity};
pub use document::{Document, DocumentChunk, DocumentChunker, ChunkingStrategy, Metadata};
pub use embeddings::{
    Embedding, EmbeddingService, EmbeddingProvider, 
    OpenAIEmbeddingProvider, LocalEmbeddingProvider,
    EmbeddingRequest, EmbeddingBatch, MockEmbeddingService
};
pub use retrieval_core::{
    Retriever, RetrievalService, InMemoryRetriever,
    SearchResult, SearchQuery, SearchConfig, SearchAlgorithm
};
pub use retrieval_enhanced::{
    HybridRetriever, HybridConfig, FusionStrategy,
    BM25Retriever, BM25Config, TokenizerType,
    SemanticRetriever, SemanticConfig,
    RankFusion, ReciprocalRankFusion, WeightedFusion
};
pub use storage::{
    Storage, StorageService, InMemoryStorage, FileStorage,
    StorageKey, StorageEntry, StorageQuery
};
pub use memory::{
    Memory, MemoryService, ConversationMessage, MessageRole,
    ConversationBufferMemory, ConversationTokenBufferMemory, ConversationSummaryMemory
};
pub use tools::{
    Tool, ToolRegistry, ToolResult, Calculator, EchoTool
};
#[cfg(feature = "http")]
pub use tools::HttpTool;
pub use streaming::{
    StreamingResponse, StreamToken, TokenType, TokenStreamBuilder
};
pub use agent::{
    RragAgent, AgentBuilder, AgentConfig, ModelConfig, AgentResponse, ToolCall
};
pub use pipeline::{
    Pipeline, PipelineStep, PipelineContext, PipelineData, PipelineConfig,
    RagPipelineBuilder, TextPreprocessingStep, DocumentChunkingStep, EmbeddingStep, RetrievalStep,
    TextOperation
};
pub use system::{
    RragSystem, RragSystemBuilder, RragSystemConfig,
    ProcessingResult, SearchResponse, ChatResponse, HealthCheckResult, SystemMetrics
};

// rsllm re-exports when feature is enabled
#[cfg(feature = "rsllm-client")]
pub use rsllm;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        // System
        RragSystem, RragSystemBuilder, RragSystemConfig,
        
        // Core types
        RragError, RragResult, ErrorSeverity,
        Document, DocumentChunk, DocumentChunker, ChunkingStrategy,
        Embedding, EmbeddingService,
        
        // Services
        RetrievalService, StorageService, MemoryService,
        
        // Agents and tools
        RragAgent, AgentBuilder, AgentConfig,
        Tool, ToolRegistry, ToolResult, Calculator,
        
        // Memory and conversations
        Memory, ConversationMessage, MessageRole,
        ConversationBufferMemory,
        
        // Streaming
        StreamingResponse, StreamToken,
        
        // Pipeline
        Pipeline, PipelineStep, PipelineContext, PipelineData,
        RagPipelineBuilder,
        
        // Search
        SearchResult, SearchQuery,
    };
    
    // External dependencies commonly used
    pub use async_trait::async_trait;
    pub use futures::{Stream, StreamExt};
    pub use serde::{Deserialize, Serialize};
    pub use tokio;
    
    // rsllm integration when feature is enabled
    #[cfg(feature = "rsllm-client")]
    pub use rsllm;
}

/// Framework version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework name
pub const NAME: &str = "RRAG";

/// Framework description
pub const DESCRIPTION: &str = "Rust RAG Framework";