//! # Graph-Based Retrieval Module
//! 
//! Advanced knowledge graph construction and graph-based retrieval for RAG systems.
//! 
//! This module enables sophisticated reasoning over structured knowledge by building
//! knowledge graphs from documents and leveraging graph traversal algorithms for
//! enhanced retrieval. It provides both automatic graph construction and manual
//! graph management capabilities.
//! 
//! ## Features
//! 
//! - **Knowledge Graph Construction**: Automatic entity and relationship extraction
//! - **Graph Algorithms**: PageRank, community detection, path finding
//! - **Entity Recognition**: Multi-type entity extraction with confidence scoring
//! - **Relationship Extraction**: Semantic relationship detection between entities
//! - **Graph Storage**: Efficient storage and indexing for large graphs
//! - **Query Expansion**: Graph-based query enhancement and expansion
//! - **Hybrid Retrieval**: Combine graph and vector retrieval
//! - **Graph Analytics**: Centrality measures, clustering, and graph statistics
//! 
//! ## Use Cases
//! 
//! - **Knowledge Base Construction**: Build structured knowledge from documents
//! - **Question Answering**: Multi-hop reasoning across connected entities
//! - **Recommendation Systems**: Find related entities and concepts
//! - **Fact Verification**: Verify claims using graph-based evidence
//! - **Research Discovery**: Find connections between research topics
//! 
//! ## Examples
//! 
//! ### Building a Knowledge Graph
//! ```rust
//! use rrag::graph_retrieval::{GraphBuilder, EntityExtractor, RelationshipExtractor};
//! 
//! # async fn example() -> rrag::RragResult<()> {
//! let mut builder = GraphBuilder::new()
//!     .with_entity_extraction(true)
//!     .with_relationship_detection(true)
//!     .build();
//! 
//! // Add documents to build the graph
//! let documents = vec![
//!     "Albert Einstein was born in Germany in 1879.",
//!     "Einstein developed the theory of relativity.",
//!     "The theory of relativity revolutionized physics."
//! ];
//! 
//! for doc in documents {
//!     builder.add_document(doc).await?;
//! }
//! 
//! let graph = builder.build().await?;
//! println!("Built graph with {} nodes and {} edges", 
//!          graph.node_count(), 
//!          graph.edge_count());
//! # Ok(())
//! # }
//! ```
//! 
//! ### Graph-Based Query Expansion
//! ```rust
//! use rrag::graph_retrieval::{GraphQueryExpander, ExpansionStrategy};
//! 
//! # async fn example() -> rrag::RragResult<()> {
//! let expander = GraphQueryExpander::new(graph)
//!     .with_strategy(ExpansionStrategy::SemanticPath)
//!     .with_max_hops(2);
//! 
//! let original_query = "Einstein's theories";
//! let expanded = expander.expand_query(original_query).await?;
//! 
//! println!("Original: {}", original_query);
//! println!("Expanded: {:?}", expanded.expanded_terms);
//! // Output might include: ["theory of relativity", "special relativity", 
//! //                       "general relativity", "physics", "German physicist"]
//! # Ok(())
//! # }
//! ```
//! 
//! ### Multi-Hop Reasoning
//! ```rust
//! use rrag::graph_retrieval::{GraphRetriever, TraversalStrategy};
//! 
//! # async fn example() -> rrag::RragResult<()> {
//! let retriever = GraphRetriever::new(graph)
//!     .with_traversal_strategy(TraversalStrategy::BreadthFirst)
//!     .with_max_depth(3);
//! 
//! // Find connections between entities
//! let connections = retriever.find_path_between(
//!     "Einstein", 
//!     "quantum mechanics"
//! ).await?;
//! 
//! for connection in connections {
//!     println!("Path: {}", connection.format_path());
//! }
//! # Ok(())
//! # }
//! ```
//! 
//! ### Entity-Centric Retrieval
//! ```rust
//! use rrag::graph_retrieval::{EntityCentricRetriever, RetrievalOptions};
//! 
//! # async fn example() -> rrag::RragResult<()> {
//! let retriever = EntityCentricRetriever::new(graph);
//! 
//! let query = "What did Einstein contribute to physics?";
//! let results = retriever.retrieve_with_entities(
//!     query,
//!     RetrievalOptions::new()
//!         .with_entity_expansion(true)
//!         .with_relationship_traversal(true)
//! ).await?;
//! 
//! for result in results {
//!     println!("Document: {}", result.content);
//!     println!("Related entities: {:?}", result.entities);
//!     println!("Relationship path: {:?}", result.path);
//! }
//! # Ok(())
//! # }
//! ```
//! 
//! ### Graph Analytics
//! ```rust
//! use rrag::graph_retrieval::{GraphAnalyzer, CentralityMetric};
//! 
//! # async fn example() -> rrag::RragResult<()> {
//! let analyzer = GraphAnalyzer::new(graph);
//! 
//! // Find most important entities
//! let pagerank_scores = analyzer.compute_centrality(
//!     CentralityMetric::PageRank
//! ).await?;
//! 
//! let top_entities = pagerank_scores.top_k(10);
//! for (entity, score) in top_entities {
//!     println!("Entity: {}, Importance: {:.3}", entity, score);
//! }
//! 
//! // Detect communities
//! let communities = analyzer.detect_communities().await?;
//! for (idx, community) in communities.iter().enumerate() {
//!     println!("Community {}: {:?}", idx, community.entities);
//! }
//! # Ok(())
//! # }
//! ```
//! 
//! ## Performance Optimization
//! 
//! - **Parallel Processing**: Multi-threaded entity extraction
//! - **Batch Operations**: Process multiple documents together
//! - **Graph Indexing**: Pre-built indexes for fast traversal
//! - **Caching**: Cache frequently accessed graph patterns
//! - **Memory Mapping**: Efficient storage for large graphs
//! - **Incremental Updates**: Add nodes/edges without rebuilding

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