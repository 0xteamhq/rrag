//! # Graph-Based Retrieval Module
//! 
//! Provides knowledge graph construction, entity/relationship extraction,
//! graph-based retrieval algorithms, and integration with existing retrieval systems.
//! 
//! ## Features
//! 
//! - **Knowledge Graph Construction**: Build graphs from documents with automatic entity/relationship extraction
//! - **Graph Algorithms**: PageRank, graph traversal, semantic path finding
//! - **Entity Recognition**: NER with custom entity types and relationship detection
//! - **Graph Storage**: Efficient graph indexing and querying
//! - **Query Expansion**: Leverage graph structure for enhanced query understanding
//! - **Integration**: Seamless integration with existing RRAG retrieval systems

pub mod graph;
pub mod entity;
pub mod algorithms;
pub mod storage;
pub mod query_expansion;
pub mod retriever;
pub mod builder;
pub mod config;

// Re-exports
pub use graph::{KnowledgeGraph, GraphNode, GraphEdge, NodeType, EdgeType, GraphMetrics};
pub use entity::{EntityExtractor, Entity, Relationship, EntityType, RelationType};
pub use algorithms::{GraphAlgorithms, PageRankConfig, TraversalConfig, PathFindingConfig};
pub use storage::{GraphStorage, GraphIndex, GraphQuery, GraphQueryResult};
pub use query_expansion::{QueryExpander, ExpansionStrategy, ExpansionResult};
pub use retriever::{GraphRetriever, GraphRetrievalConfig, GraphSearchResult};
pub use builder::{GraphRetrievalBuilder, GraphBuildConfig};
pub use config::{GraphConfig, GraphConfigBuilder, AlgorithmConfig};
pub use storage::GraphStorageConfig;

use crate::RragError;

/// Graph-based retrieval error types
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Entity extraction failed: {message}")]
    EntityExtraction { message: String },
    
    #[error("Graph construction failed: {message}")]
    GraphConstruction { message: String },
    
    #[error("Graph algorithm error: {algorithm} - {message}")]
    Algorithm { algorithm: String, message: String },
    
    #[error("Graph storage error: {operation} - {message}")]
    Storage { operation: String, message: String },
    
    #[error("Query expansion failed: {strategy} - {message}")]
    QueryExpansion { strategy: String, message: String },
    
    #[error("Graph index error: {message}")]
    Index { message: String },
}

impl From<GraphError> for RragError {
    fn from(err: GraphError) -> Self {
        RragError::retrieval(err.to_string())
    }
}