#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]
#![allow(dead_code)]

//! # RRAG - Enterprise-Grade Rust RAG Framework
//!
//! [![Crates.io](https://img.shields.io/crates/v/rrag.svg)](https://crates.io/crates/rrag)
//! [![Documentation](https://docs.rs/rrag/badge.svg)](https://docs.rs/rrag)
//! [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
//! [![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/rrag/ci.yml?branch=main)](https://github.com/yourusername/rrag/actions)
//!
//! **RRAG** (Rust RAG) is a comprehensive, high-performance framework for building
//! production-ready Retrieval-Augmented Generation (RAG) applications in Rust.
//!
//! Designed for enterprise deployments requiring extreme performance, reliability, and
//! observability, RRAG provides everything needed to build, deploy, and maintain
//! sophisticated RAG systems at scale.
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
//!     tracing::debug!("Found {} results", results.total_results);
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
//!     tracing::debug!("Agent: {}", response.text);
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
pub mod agent;
pub mod document;
pub mod embeddings;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod retrieval_core;
pub mod storage;
pub mod streaming;
pub mod system;
pub mod tools;

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
pub use agent::{Agent, AgentBuilder, AgentConfig, ConversationMemory, ConversationMode, ToolExecutor};
pub use document::{ChunkingStrategy, Document, DocumentChunk, DocumentChunker, Metadata};
pub use embeddings::{
    Embedding, EmbeddingBatch, EmbeddingProvider, EmbeddingRequest, EmbeddingService,
    LocalEmbeddingProvider, MockEmbeddingService, OpenAIEmbeddingProvider,
};
pub use error::{ErrorSeverity, RragError, RragResult};
pub use memory::{
    ConversationBufferMemory, ConversationMessage, ConversationSummaryMemory,
    ConversationTokenBufferMemory, Memory, MemoryService, MessageRole,
};
pub use pipeline::{
    DocumentChunkingStep, EmbeddingStep, Pipeline, PipelineConfig, PipelineContext, PipelineData,
    PipelineStep, RagPipelineBuilder, RetrievalStep, TextOperation, TextPreprocessingStep,
};
pub use retrieval_core::{
    InMemoryRetriever, RetrievalService, Retriever, SearchAlgorithm, SearchConfig, SearchQuery,
    SearchResult,
};
pub use retrieval_enhanced::{
    BM25Config, BM25Retriever, FusionStrategy, HybridConfig, HybridRetriever, RankFusion,
    ReciprocalRankFusion, SemanticConfig, SemanticRetriever, TokenizerType, WeightedFusion,
};
pub use storage::{
    FileStorage, InMemoryStorage, Storage, StorageEntry, StorageKey, StorageQuery, StorageService,
};
pub use streaming::{StreamToken, StreamingResponse, TokenStreamBuilder, TokenType};
pub use system::{
    ChatResponse, HealthCheckResult, ProcessingResult, RragSystem, RragSystemBuilder,
    RragSystemConfig, SearchResponse, SystemMetrics,
};
#[cfg(feature = "http")]
pub use tools::HttpTool;
// Tools module is kept for backward compatibility but agents use rsllm::tools now
pub use tools::{Calculator, EchoTool, Tool, ToolResult};

// rsllm re-exports when feature is enabled
#[cfg(feature = "rsllm-client")]
pub use rsllm;

// Graph retrieval re-exports
pub use graph_retrieval::{
    EdgeType, Entity, EntityExtractor, EntityType, ExpansionResult, ExpansionStrategy,
    GraphAlgorithms, GraphBuildConfig, GraphConfig, GraphConfigBuilder, GraphEdge, GraphIndex,
    GraphMetrics, GraphNode, GraphQuery, GraphQueryResult, GraphRetrievalBuilder,
    GraphRetrievalConfig, GraphRetriever, GraphStorage, KnowledgeGraph, NodeType, PageRankConfig,
    PathFindingConfig, QueryExpander, RelationType, Relationship, TraversalConfig,
};

// Incremental indexing re-exports
pub use incremental::{
    AlertConfig, BatchConfig, BatchExecutor, BatchOperation, BatchProcessingStats, BatchProcessor,
    BatchResult, ChangeDetectionConfig, ChangeDetector, ChangeResult, ChangeType,
    ConflictResolution, ConsistencyReport, ContentDelta, DocumentChange, DocumentVersion,
    EmbeddingUpdate, HealthMetrics, IncrementalIndexManager, IncrementalIndexingService,
    IncrementalMetrics, IncrementalServiceBuilder, IncrementalServiceConfig, IndexManagerConfig,
    IndexOperation, IndexUpdate, IndexUpdateStrategy, IndexingStats, IntegrityChecker,
    IntegrityConfig, IntegrityError, MetricsCollector, MonitoringConfig, OperationLog,
    PerformanceTracker, QueueManager, RecoveryResult, RollbackConfig, RollbackManager,
    RollbackOperation, RollbackPoint, UpdateResult, ValidationResult, VectorBatch, VectorOperation,
    VectorUpdateConfig, VectorUpdateManager, VersionConflict, VersionHistory, VersionManager,
    VersionResolution, VersioningConfig,
};

// Observability re-exports
pub use observability::{
    AlertCondition, AlertManager, AlertNotification, AlertRule, AlertSeverity, BottleneckAnalysis,
    ComponentStatus, DashboardConfig, DashboardServer, DataRetention, ExportFormat, ExportManager,
    HealthChecker, HealthReport, HistoricalAnalyzer, LogAggregator, LogEntry, LogFilter, LogLevel,
    LogQuery, Metric, MetricType, MetricValue, MetricsCollector as ObsMetricsCollector,
    MetricsExporter, MetricsRegistry, ObservabilityBuilder, ObservabilityConfig,
    ObservabilitySystem, PerformanceMonitor, PerformanceReport, ProfileData, Profiler,
    RealtimeMetrics, ReportGenerator, RetentionPolicy, SearchAnalyzer, SystemMonitor,
    UserActivityTracker, WebSocketManager,
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
/// - Agents and tools ([`Agent`], [`Tool`], [`Calculator`])
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
        ChatResponse, HealthCheckResult, ProcessingResult, RragSystem, RragSystemBuilder,
        RragSystemConfig, SearchResponse, SystemMetrics,
    };

    // Core types and error handling
    pub use crate::{
        ChunkingStrategy, Document, DocumentChunk, DocumentChunker, Embedding, EmbeddingProvider,
        EmbeddingService, ErrorSeverity, Metadata, RragError, RragResult,
    };

    // Service interfaces
    pub use crate::{
        InMemoryRetriever, MemoryService, RetrievalService, SearchConfig, SearchQuery,
        SearchResult, StorageService,
    };

    // Agents and tools
    pub use crate::{
        Agent, AgentBuilder, AgentConfig, ConversationMemory, ConversationMode, ToolExecutor,
    };

    // HTTP tools when feature is enabled
    #[cfg(feature = "http")]
    pub use crate::HttpTool;

    // Memory and conversations
    pub use crate::{
        ConversationBufferMemory, ConversationMessage, ConversationSummaryMemory,
        ConversationTokenBufferMemory, Memory, MessageRole,
    };

    // Streaming support
    pub use crate::{StreamToken, StreamingResponse, TokenStreamBuilder, TokenType};

    // Pipeline processing
    pub use crate::{
        DocumentChunkingStep, EmbeddingStep, Pipeline, PipelineContext, PipelineData, PipelineStep,
        RagPipelineBuilder, RetrievalStep, TextOperation, TextPreprocessingStep,
    };

    // Enhanced retrieval
    pub use crate::{
        BM25Retriever, FusionStrategy, HybridConfig, HybridRetriever, RankFusion,
        ReciprocalRankFusion, SemanticRetriever,
    };

    // Incremental indexing
    pub use crate::{
        BatchProcessor, ChangeDetector, ChangeResult, ChangeType, IncrementalIndexManager,
        IncrementalIndexingService, IncrementalServiceBuilder, IntegrityChecker, MetricsCollector,
        RollbackManager, VectorUpdateManager, VersionManager,
    };

    // Graph retrieval
    pub use crate::{
        EntityExtractor, GraphEdge, GraphNode, GraphRetrievalBuilder, GraphRetriever,
        KnowledgeGraph, QueryExpander,
    };

    // Observability
    pub use crate::{AlertManager, HealthChecker, MetricsRegistry, ObservabilitySystem, Profiler};

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
/// tracing::debug!("RRAG Framework version: {}", VERSION);
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
pub const DESCRIPTION: &str =
    "Rust RAG Framework - High-performance Retrieval-Augmented Generation";

/// Framework repository URL
///
/// The primary repository location for source code, issues, and documentation.
pub const REPOSITORY: &str = "https://github.com/levalhq/rrag";

/// Framework license
///
/// The software license under which RRAG is distributed.
pub const LICENSE: &str = "MIT";

/// Minimum supported Rust version (MSRV)
///
/// The minimum version of Rust required to compile and use RRAG.
pub const MSRV: &str = "1.70.0";
