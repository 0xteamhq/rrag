#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]

//! # RRAG - Rust RAG Framework (Minimal Release)
//! 
//! [![Crates.io](https://img.shields.io/crates/v/rrag.svg)](https://crates.io/crates/rrag)
//! [![Documentation](https://docs.rs/rrag/badge.svg)](https://docs.rs/rrag)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//! 
//! **RRAG** (Rust RAG) is a high-performance, native Rust framework for building 
//! Retrieval-Augmented Generation applications. This is the minimal release containing
//! the core, stable functionality.
//! 
//! ## 🚀 Quick Start
//! 
//! ```rust,no_run
//! use rrag::prelude::*;
//! 
//! #[tokio::main]
//! async fn main() -> RragResult<()> {
//!     // Create a basic RAG system
//!     let rag = RragSystemBuilder::new()
//!         .with_name("My RAG System")
//!         .with_environment("development")
//!         .build()
//!         .await?;
//!     
//!     // Add documents
//!     let doc = Document::new("Rust is a systems programming language...")
//!         .with_metadata("source", "documentation".into())
//!         .with_content_hash();
//!     
//!     rag.process_document(doc).await?;
//!     
//!     // Search for relevant content
//!     let results = rag.search("What is Rust?".to_string(), Some(5)).await?;
//!     println!("Found {} results", results.total_results);
//!     
//!     Ok(())
//! }
//! ```

// Core modules - foundational components
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

// Enhanced retrieval capabilities
#[path = "retrieval/mod.rs"]
pub mod retrieval_enhanced;

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
    //! Convenient re-exports for common RRAG functionality (minimal version)
    
    // System components
    pub use crate::{
        RragSystem, RragSystemBuilder, RragSystemConfig,
        ProcessingResult, SearchResponse, ChatResponse, 
        HealthCheckResult, SystemMetrics,
    };
    
    // Core types and error handling
    pub use crate::{
        RragError, RragResult, ErrorSeverity,
        Document, DocumentChunk, DocumentChunker, ChunkingStrategy, Metadata,
        Embedding, EmbeddingService, EmbeddingProvider,
    };
    
    // Service interfaces
    pub use crate::{
        RetrievalService, StorageService, MemoryService,
        InMemoryRetriever, SearchResult, SearchQuery, SearchConfig,
    };
    
    // Agents and tools
    pub use crate::{
        RragAgent, AgentBuilder, AgentConfig, AgentResponse,
        Tool, ToolRegistry, ToolResult, Calculator,
    };
    
    // HTTP tools when feature is enabled
    #[cfg(feature = "http")]
    pub use crate::HttpTool;
    
    // Memory and conversations
    pub use crate::{
        Memory, ConversationMessage, MessageRole,
        ConversationBufferMemory, ConversationTokenBufferMemory,
        ConversationSummaryMemory,
    };
    
    // Streaming support
    pub use crate::{
        StreamingResponse, StreamToken, TokenType, TokenStreamBuilder,
    };
    
    // Pipeline processing
    pub use crate::{
        Pipeline, PipelineStep, PipelineContext, PipelineData,
        RagPipelineBuilder, TextPreprocessingStep, DocumentChunkingStep,
        EmbeddingStep, RetrievalStep, TextOperation,
    };
    
    // Enhanced retrieval
    pub use crate::{
        HybridRetriever, HybridConfig, FusionStrategy,
        BM25Retriever, SemanticRetriever,
        RankFusion, ReciprocalRankFusion,
    };
    
    // External dependencies commonly used with RRAG
    pub use async_trait::async_trait;
    pub use futures::{Stream, StreamExt};
    pub use serde::{Deserialize, Serialize};
    pub use tokio;
    
    // rsllm integration when feature is enabled
    #[cfg(feature = "rsllm-client")]
    pub use rsllm;
}

/// Current version of the RRAG framework
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework name identifier
pub const NAME: &str = "RRAG";

/// Framework description
pub const DESCRIPTION: &str = "Rust RAG Framework - High-performance Retrieval-Augmented Generation";

/// Framework repository URL
pub const REPOSITORY: &str = "https://github.com/leval-ai/rrag";

/// Framework license
pub const LICENSE: &str = "MIT";

/// Minimum supported Rust version (MSRV)
pub const MSRV: &str = "1.70.0";