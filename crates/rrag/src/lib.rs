#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]

//! # RRAG - Rust RAG Framework
//! 
//! [![Crates.io](https://img.shields.io/crates/v/rrag.svg)](https://crates.io/crates/rrag)
//! [![Documentation](https://docs.rs/rrag/badge.svg)](https://docs.rs/rrag)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//! 
//! **RRAG** (Rust RAG) is a high-performance, native Rust framework for building 
//! Retrieval-Augmented Generation applications. Built from the ground up with Rust's 
//! principles of safety, performance, and concurrency.
//! 
//! ## 🚀 Key Features
//! 
//! - **🔥 Native Performance**: Zero-cost abstractions with compile-time optimizations
//! - **🛡️ Memory Safe**: Leverage Rust's ownership system for bulletproof memory management
//! - **⚡ Async First**: Built on Tokio for high-concurrency workloads
//! - **🎯 Type Safe**: Compile-time guarantees prevent runtime errors
//! - **🔌 Pluggable**: Modular architecture with swappable components
//! - **🌊 Streaming**: Real-time token streaming with async iterators
//! - **📊 Observable**: Built-in metrics, tracing, and health checks
//! 
//! ## 🏗️ Architecture Overview
//! 
//! RRAG follows a modular, pipeline-based architecture that maximizes performance
//! while maintaining flexibility:
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
//! ## 🚀 Quick Start
//! 
//! ### Basic RAG System
//! 
//! ```rust
//! use rrag::prelude::*;
//! 
//! #[tokio::main]
//! async fn main() -> RragResult<()> {
//!     // Create a RAG system
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
//! 
//! ### Advanced Agent with Tools
//! 
//! ```rust
//! use rrag::prelude::*;
//! 
//! #[tokio::main]
//! async fn main() -> RragResult<()> {
//!     // Create an agent with tools
//!     let mut agent = AgentBuilder::new()
//!         .with_model("gpt-4")
//!         .with_temperature(0.7)
//!         .with_streaming(true)
//!         .build()
//!         .await?;
//!     
//!     // Register tools
//!     agent.register_tool(Calculator::new())?;
//!     # #[cfg(feature = "http")]
//!     agent.register_tool(HttpTool::new())?;
//!     
//!     // Chat with memory
//!     let memory = ConversationBufferMemory::new(100);
//!     let response = agent.chat_with_memory(
//!         "Calculate 15 * 23",
//!         &memory
//!     ).await?;
//!     
//!     println!("Agent: {}", response.text);
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ## 📦 Feature Flags
//! 
//! RRAG supports multiple feature flags for flexible deployments:
//! 
//! - `default`: Core functionality with HTTP and concurrency support
//! - `rsllm-client`: Integration with rsllm for LLM operations
//! - `http`: HTTP client support for external services
//! - `concurrent`: Advanced concurrency features with DashMap
//! - `observability`: Metrics, monitoring, and alerting
//! - `security`: Authentication, authorization, and security features
//! - `security-full`: Complete security suite with 2FA and WebAuthn
//! 
//! ```toml
//! [dependencies]
//! rrag = { version = "0.1", features = ["rsllm-client", "observability", "security"] }
//! ```

//! 
//! ## 🏗️ Module Organization
//! 
//! RRAG is organized into focused modules, each handling specific aspects of RAG functionality:
//! 
//! ### Core Modules
//! 
//! - [`error`]: Comprehensive error handling with structured error types
//! - [`document`]: Document processing, chunking, and metadata management
//! - [`embeddings`]: Multi-provider embedding generation and management
//! - [`storage`]: Pluggable storage backends for documents and vectors
//! - [`memory`]: Conversation memory and context management
//! - [`agent`]: LLM agents with tool calling and streaming support
//! - [`pipeline`]: Composable processing pipelines
//! - [`system`]: High-level system orchestration and lifecycle management
//! 
//! ### Advanced Modules
//! 
//! - [`retrieval_core`]: Core retrieval interfaces and implementations
//! - [`retrieval_enhanced`]: Advanced hybrid retrieval with BM25 and semantic search
//! - [`reranking`]: Result reranking and relevance scoring
//! - [`evaluation`]: Framework for evaluating RAG system performance
//! - [`caching`]: Intelligent caching strategies for performance optimization
//! - [`graph_retrieval`]: Knowledge graph-based retrieval and reasoning
//! - [`incremental`]: Incremental indexing for large-scale document updates
//! - [`observability`]: Comprehensive monitoring, metrics, and alerting
//! - [`tools`]: Built-in and extensible tool implementations
//! - [`streaming`]: Real-time streaming response handling
//! - [`query`]: Query processing and optimization
//! - [`multimodal`]: Multi-modal content processing support

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

// Query processing and optimization
pub mod query;

// Advanced reranking algorithms
pub mod reranking;

// Evaluation and benchmarking framework
pub mod evaluation;

// Intelligent caching layer
pub mod caching;

// Multi-modal content support
pub mod multimodal;

// Graph-based knowledge retrieval
pub mod graph_retrieval;

// Incremental indexing system
pub mod incremental;

// Observability and monitoring
pub mod observability;

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

// Graph retrieval re-exports
pub use graph_retrieval::{
    GraphRetriever, GraphRetrievalBuilder, GraphRetrievalConfig, GraphBuildConfig,
    KnowledgeGraph, GraphNode, GraphEdge, NodeType, EdgeType, GraphMetrics,
    EntityExtractor, Entity, Relationship, EntityType, RelationType,
    GraphAlgorithms, PageRankConfig, TraversalConfig, PathFindingConfig,
    GraphStorage, GraphIndex, GraphQuery, GraphQueryResult,
    QueryExpander, ExpansionStrategy, ExpansionResult,
    GraphConfig, GraphConfigBuilder,
};

// Incremental indexing re-exports
pub use incremental::{
    IncrementalIndexingService, IncrementalServiceConfig, IncrementalServiceBuilder,
    ChangeDetector, ChangeResult, ChangeType, DocumentChange, ContentDelta, ChangeDetectionConfig,
    IncrementalIndexManager, IndexOperation, IndexUpdate, UpdateResult, ConflictResolution, IndexManagerConfig,
    BatchProcessor, BatchOperation, BatchConfig, BatchResult, BatchProcessingStats, BatchExecutor, QueueManager,
    DocumentVersion, VersionManager, VersionConflict, VersionResolution, VersioningConfig, VersionHistory,
    RollbackManager, RollbackOperation, RollbackPoint, RecoveryResult, RollbackConfig, OperationLog,
    IntegrityChecker, ConsistencyReport, IntegrityError, ValidationResult, IntegrityConfig, HealthMetrics,
    VectorUpdateManager, VectorOperation, EmbeddingUpdate, VectorBatch, VectorUpdateConfig, IndexUpdateStrategy,
    IncrementalMetrics, PerformanceTracker, IndexingStats, MonitoringConfig, AlertConfig, MetricsCollector,
};

// Observability re-exports  
pub use observability::{
    ObservabilitySystem, ObservabilityConfig, ObservabilityBuilder,
    MetricsCollector as ObsMetricsCollector, MetricsRegistry, Metric, MetricType, MetricValue,
    SystemMonitor, PerformanceMonitor, SearchAnalyzer, UserActivityTracker,
    AlertManager, AlertRule, AlertSeverity, AlertCondition, AlertNotification,
    DashboardServer, DashboardConfig, WebSocketManager, RealtimeMetrics,
    LogAggregator, LogLevel, LogEntry, LogQuery, LogFilter,
    HealthChecker, HealthReport, ComponentStatus,
    Profiler, ProfileData, BottleneckAnalysis, PerformanceReport,
    ExportManager, ExportFormat, ReportGenerator, MetricsExporter,
    DataRetention, RetentionPolicy, HistoricalAnalyzer,
};

/// Prelude module for convenient imports
///
/// This module re-exports the most commonly used types and traits from RRAG,
/// making it easy to get started with a single import:
///
/// ```rust
/// use rrag::prelude::*;
/// ```
///
/// The prelude includes:
/// - Core system components ([`RragSystem`], [`RragSystemBuilder`])
/// - Document processing ([`Document`], [`DocumentChunk`], [`ChunkingStrategy`])
/// - Error handling ([`RragError`], [`RragResult`])
/// - Agents and tools ([`RragAgent`], [`Tool`], [`Calculator`])
/// - Memory management ([`Memory`], [`ConversationBufferMemory`])
/// - Streaming support ([`StreamingResponse`], [`StreamToken`])
/// - Pipeline processing ([`Pipeline`], [`RagPipelineBuilder`])
/// - Search functionality ([`SearchResult`], [`SearchQuery`])
/// - Advanced features (incremental indexing, observability)
///
/// ## Feature-Gated Exports
///
/// Some exports are only available with specific features:
/// - `rsllm-client`: [`rsllm`] integration
/// - `http`: HTTP-based tools like [`HttpTool`]
///
/// ## External Dependencies
///
/// Common external crates are also re-exported for convenience:
/// - [`tokio`]: Async runtime
/// - [`serde`]: Serialization traits
/// - [`futures`]: Stream processing
/// - [`async_trait`]: Async trait support
pub mod prelude {
    //! Convenient re-exports for common RRAG functionality
    //!
    //! Import everything you need with:
    //! ```rust
    //! use rrag::prelude::*;
    //! ```
    
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
    
    // Incremental indexing
    pub use crate::{
        IncrementalIndexingService, IncrementalServiceBuilder,
        ChangeDetector, ChangeResult, ChangeType, IncrementalIndexManager,
        BatchProcessor, VersionManager, RollbackManager,
        IntegrityChecker, VectorUpdateManager, MetricsCollector,
    };
    
    // Graph retrieval
    pub use crate::{
        GraphRetriever, GraphRetrievalBuilder, KnowledgeGraph,
        GraphNode, GraphEdge, EntityExtractor, QueryExpander,
    };
    
    // Observability
    pub use crate::{
        ObservabilitySystem, MetricsRegistry, AlertManager,
        HealthChecker, Profiler,
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

/// Framework constants and metadata
///
/// These constants provide version and identification information for the RRAG framework.

/// Current version of the RRAG framework
///
/// This version is automatically synchronized with the version in `Cargo.toml`.
/// Use this for version checking, compatibility testing, or reporting.
///
/// # Example
/// ```rust
/// use rrag::VERSION;
/// println!("RRAG Framework version: {}", VERSION);
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework name identifier
///
/// The canonical name of the framework, used for logging, metrics,
/// and system identification.
pub const NAME: &str = "RRAG";

/// Framework description
///
/// A brief description of the framework's purpose and capabilities.
pub const DESCRIPTION: &str = "Rust RAG Framework - High-performance Retrieval-Augmented Generation";

/// Framework repository URL
///
/// The primary repository location for source code, issues, and documentation.
pub const REPOSITORY: &str = "https://github.com/leval-ai/rrag";

/// Framework license
///
/// The software license under which RRAG is distributed.
pub const LICENSE: &str = "MIT";

/// Minimum supported Rust version (MSRV)
///
/// The minimum version of Rust required to compile and use RRAG.
pub const MSRV: &str = "1.70.0";