//! # Graph-Based Retriever
//! 
//! Main retriever implementation that integrates graph-based algorithms with traditional retrieval methods.

use super::{
    KnowledgeGraph, GraphNode, algorithms::GraphAlgorithms, 
    query_expansion::{QueryExpander, GraphQueryExpander, ExpansionOptions, ExpansionStrategy},
    storage::GraphStorage,
};
use crate::{
    RragResult, SearchResult, SearchQuery, Retriever, Embedding, 
    Document, DocumentChunk, retrieval_core::{IndexStats, QueryType}
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use async_trait::async_trait;

/// Graph-based retriever that combines traditional and graph-based search
pub struct GraphRetriever {
    /// Knowledge graph (using RwLock for interior mutability)
    graph: tokio::sync::RwLock<KnowledgeGraph>,
    
    /// Graph storage backend
    storage: tokio::sync::RwLock<Box<dyn GraphStorage>>,
    
    /// Query expander (using RwLock for interior mutability)
    query_expander: tokio::sync::RwLock<GraphQueryExpander>,
    
    /// Configuration
    config: GraphRetrievalConfig,
    
    /// PageRank scores cache
    pagerank_cache: tokio::sync::RwLock<Option<HashMap<String, f32>>>,
    
    /// Entity to document mapping (using RwLock for interior mutability)
    entity_document_map: tokio::sync::RwLock<HashMap<String, HashSet<String>>>,
}

/// Graph retrieval configuration
#[derive(Debug, Clone)]
pub struct GraphRetrievalConfig {
    /// Enable query expansion
    pub enable_query_expansion: bool,
    
    /// Enable PageRank scoring
    pub enable_pagerank_scoring: bool,
    
    /// Enable path-based retrieval
    pub enable_path_based_retrieval: bool,
    
    /// Weight for graph-based scores vs traditional similarity
    pub graph_weight: f32,
    
    /// Weight for traditional similarity scores
    pub similarity_weight: f32,
    
    /// Maximum number of graph hops for retrieval
    pub max_graph_hops: usize,
    
    /// Minimum graph score threshold
    pub min_graph_score: f32,
    
    /// Query expansion configuration
    pub expansion_options: ExpansionOptions,
    
    /// PageRank configuration
    pub pagerank_config: super::algorithms::PageRankConfig,
    
    /// Enable result diversification
    pub enable_diversification: bool,
    
    /// Diversification factor (0.0 to 1.0)
    pub diversification_factor: f32,
}

impl Default for GraphRetrievalConfig {
    fn default() -> Self {
        Self {
            enable_query_expansion: true,
            enable_pagerank_scoring: true,
            enable_path_based_retrieval: true,
            graph_weight: 0.4,
            similarity_weight: 0.6,
            max_graph_hops: 3,
            min_graph_score: 0.1,
            expansion_options: ExpansionOptions {
                strategies: vec![
                    ExpansionStrategy::Semantic,
                    ExpansionStrategy::Similarity,
                    ExpansionStrategy::CoOccurrence,
                ],
                max_terms: Some(10),
                min_confidence: 0.3,
                ..Default::default()
            },
            pagerank_config: super::algorithms::PageRankConfig::default(),
            enable_diversification: true,
            diversification_factor: 0.3,
        }
    }
}

/// Graph search result with additional graph-specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    /// Base search result
    pub search_result: SearchResult,
    
    /// Graph-based score
    pub graph_score: f32,
    
    /// PageRank score of associated entities
    pub pagerank_score: f32,
    
    /// Related entities found in the content
    pub related_entities: Vec<String>,
    
    /// Graph paths that led to this result
    pub graph_paths: Vec<GraphPath>,
    
    /// Expanded query terms that matched
    pub matched_expansions: Vec<String>,
}

/// Graph path information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Node IDs in the path
    pub nodes: Vec<String>,
    
    /// Path score
    pub score: f32,
    
    /// Path type/description
    pub path_type: String,
    
    /// Path length
    pub length: usize,
}

impl GraphRetriever {
    /// Create a new graph retriever
    pub fn new(
        graph: KnowledgeGraph,
        storage: Box<dyn GraphStorage>,
        config: GraphRetrievalConfig,
    ) -> RragResult<Self> {
        let query_expander = GraphQueryExpander::new(
            graph.clone(),
            super::query_expansion::ExpansionConfig::default(),
        );
        
        let mut entity_document_map = HashMap::new();
        
        // Build entity-document mapping
        for (_, node) in &graph.nodes {
            for doc_id in &node.source_documents {
                entity_document_map
                    .entry(node.id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(doc_id.clone());
            }
        }
        
        let retriever = Self {
            graph: tokio::sync::RwLock::new(graph),
            storage: tokio::sync::RwLock::new(storage),
            query_expander: tokio::sync::RwLock::new(query_expander),
            config,
            pagerank_cache: tokio::sync::RwLock::new(None),
            entity_document_map: tokio::sync::RwLock::new(entity_document_map),
        };
        
        Ok(retriever)
    }

    /// Update the knowledge graph
    pub async fn update_graph(&self, graph: KnowledgeGraph) -> RragResult<()> {
        *self.graph.write().await = graph.clone();
        self.query_expander.write().await.update_graph(graph.clone()).await;
        
        // Rebuild entity-document mapping
        let mut entity_map = self.entity_document_map.write().await;
        entity_map.clear();
        for (_, node) in &graph.nodes {
            for doc_id in &node.source_documents {
                entity_map
                    .entry(node.id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(doc_id.clone());
            }
        }
        
        // Clear PageRank cache
        *self.pagerank_cache.write().await = None;
        
        // Update storage
        self.storage.write().await.store_graph(&graph).await?;
        
        Ok(())
    }

    /// Get or compute PageRank scores
    async fn get_pagerank_scores(&self) -> RragResult<HashMap<String, f32>> {
        let mut cache = self.pagerank_cache.write().await;
        
        if cache.is_none() {
            let graph = self.graph.read().await;
            let scores = GraphAlgorithms::pagerank(&*graph, &self.config.pagerank_config)?;
            *cache = Some(scores);
        }
        
        Ok(cache.clone().unwrap())
    }

    /// Expand query using graph structure
    async fn expand_query(&self, query: &str) -> RragResult<Vec<String>> {
        if !self.config.enable_query_expansion {
            return Ok(vec![query.to_string()]);
        }
        
        let expansion_result = self.query_expander.read().await
            .expand_query(query, &self.config.expansion_options)
            .await?;
        
        let mut terms = vec![query.to_string()];
        terms.extend(expansion_result.expanded_terms.into_iter().map(|t| t.term));
        
        Ok(terms)
    }

    /// Find entities related to query
    async fn find_query_entities(&self, query: &str) -> Vec<String> {
        let query_lower = query.to_lowercase();
        let mut entities = Vec::new();
        
        let graph = self.graph.read().await;
        
        // Find entities that match query terms
        for (entity_id, node) in &graph.nodes {
            let label_lower = node.label.to_lowercase();
            if query_lower.contains(&label_lower) || label_lower.contains(&query_lower) {
                entities.push(entity_id.clone());
            }
        }
        
        entities
    }

    /// Add documents and their entities to the graph
    pub async fn add_document_with_entities(
        &self,
        document: &Document,
        entities: Vec<GraphNode>,
        relationships: Vec<super::GraphEdge>,
    ) -> RragResult<()> {
        let mut graph = self.graph.write().await;
        
        // Add document node
        let doc_node = GraphNode::new(
            format!("doc_{}", document.id),
            super::NodeType::Document,
        )
        .with_source_document(document.id.clone())
        .with_attribute("title", serde_json::Value::String(
            document.metadata.get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(&document.id)
                .to_string()
        ));
        
        graph.add_node(doc_node.clone())?;
        
        // Add entities and connect them to the document
        for entity in entities {
            let entity_id = entity.id.clone();
            graph.add_node(entity)?;
            
            // Create containment edge from document to entity
            let containment_edge = super::GraphEdge::new(
                doc_node.id.clone(),
                entity_id.clone(),
                "contains",
                super::EdgeType::Contains,
            );
            graph.add_edge(containment_edge)?;
            
            // Update entity-document mapping
            self.entity_document_map.write().await
                .entry(entity_id)
                .or_insert_with(HashSet::new)
                .insert(document.id.clone());
        }
        
        // Add relationships
        for relationship in relationships {
            graph.add_edge(relationship)?;
        }
        
        // Clear PageRank cache
        *self.pagerank_cache.write().await = None;
        
        // Update storage
        self.storage.write().await.store_graph(&*graph).await?;
        
        Ok(())
    }
}

#[async_trait]
impl Retriever for GraphRetriever {
    fn name(&self) -> &str {
        "graph_retriever"
    }

    async fn search(&self, query: &SearchQuery) -> RragResult<Vec<SearchResult>> {
        let query_text = match &query.query {
            QueryType::Text(text) => text,
            QueryType::Embedding(_) => {
                // For embedding queries, we can't do text-based entity extraction
                // Fall back to basic similarity search (would need embedding-based entity matching)
                return Ok(Vec::new());
            }
        };
        
        // Expand query if enabled
        let expanded_terms = self.expand_query(query_text).await?;
        let expanded_query = expanded_terms.join(" ");
        
        // Find entities in the (expanded) query
        let query_entities = self.find_query_entities(&expanded_query).await;
        
        // For simplicity, return basic results based on entity matching
        let mut results = Vec::new();
        
        let entity_map = self.entity_document_map.read().await;
        let pagerank_scores = if self.config.enable_pagerank_scoring {
            self.get_pagerank_scores().await?
        } else {
            HashMap::new()
        };
        
        // Find documents connected to query entities
        let mut candidate_docs = HashSet::new();
        for entity_id in &query_entities {
            if let Some(doc_ids) = entity_map.get(entity_id) {
                candidate_docs.extend(doc_ids.clone());
            }
        }
        
        // Create search results for candidate documents
        for (rank, doc_id) in candidate_docs.iter().enumerate() {
            // Calculate graph-based score
            let mut graph_score = 0.5; // Base score
            
            // Add PageRank contribution
            for entity_id in &query_entities {
                if let Some(doc_ids) = entity_map.get(entity_id) {
                    if doc_ids.contains(doc_id) {
                        let pagerank_score = pagerank_scores.get(entity_id).copied().unwrap_or(0.0);
                        graph_score += pagerank_score * 0.3;
                    }
                }
            }
            
            if graph_score >= self.config.min_graph_score {
                let result = SearchResult {
                    id: doc_id.clone(),
                    content: format!("Document {}", doc_id), // Placeholder
                    score: graph_score,
                    rank,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("graph_score".to_string(), 
                                       serde_json::json!(graph_score));
                        metadata
                    },
                    embedding: None,
                };
                
                results.push(result);
            }
        }
        
        // Sort by score and apply limits
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.retain(|result| result.score >= query.min_score);
        results.truncate(query.limit);
        
        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
        
        Ok(results)
    }

    async fn add_documents(&self, documents: &[(Document, Embedding)]) -> RragResult<()> {
        // This would typically involve:
        // 1. Extracting entities and relationships from documents
        // 2. Adding them to the graph
        // 3. Updating storage
        // For now, just add document nodes
        
        let mut graph = self.graph.write().await;
        let mut nodes = Vec::new();
        
        for (document, _embedding) in documents {
            let doc_node = GraphNode::new(
                format!("doc_{}", document.id),
                super::NodeType::Document,
            )
            .with_source_document(document.id.clone());
            
            nodes.push(doc_node.clone());
            graph.add_node(doc_node)?;
        }
        
        self.storage.write().await.store_nodes(&nodes).await?;
        
        Ok(())
    }

    async fn add_chunks(&self, chunks: &[(DocumentChunk, Embedding)]) -> RragResult<()> {
        // Similar to add_documents but for chunks
        let mut graph = self.graph.write().await;
        let mut nodes = Vec::new();
        
        for (chunk, _embedding) in chunks {
            let chunk_node = GraphNode::new(
                format!("chunk_{}_{}", chunk.document_id, chunk.chunk_index),
                super::NodeType::DocumentChunk,
            )
            .with_source_document(chunk.document_id.clone());
            
            nodes.push(chunk_node.clone());
            graph.add_node(chunk_node)?;
        }
        
        self.storage.write().await.store_nodes(&nodes).await?;
        
        Ok(())
    }

    async fn remove_documents(&self, document_ids: &[String]) -> RragResult<()> {
        let mut graph = self.graph.write().await;
        
        // Remove document nodes and update entity mappings
        let doc_node_ids: Vec<_> = document_ids.iter()
            .map(|doc_id| format!("doc_{}", doc_id))
            .collect();
        
        for node_id in &doc_node_ids {
            graph.remove_node(node_id)?;
        }
        
        // Update entity-document mapping
        let mut entity_map = self.entity_document_map.write().await;
        for doc_id in document_ids {
            for entity_docs in entity_map.values_mut() {
                entity_docs.remove(doc_id);
            }
        }
        
        self.storage.write().await.delete_nodes(&doc_node_ids).await?;
        
        Ok(())
    }

    async fn clear(&self) -> RragResult<()> {
        *self.graph.write().await = KnowledgeGraph::new();
        self.entity_document_map.write().await.clear();
        *self.pagerank_cache.write().await = None;
        self.storage.write().await.clear().await?;
        Ok(())
    }

    async fn stats(&self) -> RragResult<IndexStats> {
        let storage_stats = self.storage.read().await.get_stats().await?;
        let graph = self.graph.read().await;
        let _graph_metrics = graph.calculate_metrics();
        
        Ok(IndexStats {
            total_items: storage_stats.total_nodes,
            size_bytes: storage_stats.storage_size_bytes,
            dimensions: 0, // Graph doesn't have fixed dimensions
            index_type: "graph_based".to_string(),
            last_updated: storage_stats.last_updated,
        })
    }

    async fn health_check(&self) -> RragResult<bool> {
        // Check graph consistency and storage health
        let graph = self.graph.read().await;
        Ok(!graph.nodes.is_empty() || self.storage.read().await.get_stats().await.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_retrieval::{storage::InMemoryGraphStorage, GraphEdge, NodeType, EdgeType};

    #[tokio::test]
    async fn test_graph_retriever_creation() {
        let graph = KnowledgeGraph::new();
        let storage = Box::new(InMemoryGraphStorage::new());
        let config = GraphRetrievalConfig::default();
        
        let retriever = GraphRetriever::new(graph, storage, config).unwrap();
        assert_eq!(retriever.name(), "graph_retriever");
    }

    #[tokio::test]
    async fn test_query_expansion() {
        let mut graph = KnowledgeGraph::new();
        
        // Create test graph with related entities
        let node1 = GraphNode::new("machine learning", NodeType::Concept);
        let node2 = GraphNode::new("artificial intelligence", NodeType::Concept);
        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        
        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();
        
        graph.add_edge(GraphEdge::new(
            node1_id.clone(),
            node2_id.clone(),
            "part_of",
            EdgeType::Semantic("part_of".to_string())
        ).with_confidence(0.8)).unwrap();
        
        let storage = Box::new(InMemoryGraphStorage::new());
        let config = GraphRetrievalConfig::default();
        
        let retriever = GraphRetriever::new(graph, storage, config).unwrap();
        
        // Test query expansion
        let expanded = retriever.expand_query("machine learning").await.unwrap();
        assert!(!expanded.is_empty());
        assert!(expanded.contains(&"machine learning".to_string()));
    }

    #[tokio::test]
    async fn test_find_query_entities() {
        let mut graph = KnowledgeGraph::new();
        
        let node = GraphNode::new("neural networks", NodeType::Concept);
        let node_id = node.id.clone();
        graph.add_node(node).unwrap();
        
        let storage = Box::new(InMemoryGraphStorage::new());
        let config = GraphRetrievalConfig::default();
        
        let retriever = GraphRetriever::new(graph, storage, config).unwrap();
        
        let entities = retriever.find_query_entities("neural networks deep learning").await;
        assert!(!entities.is_empty());
        assert!(entities.contains(&node_id));
    }

    #[tokio::test]
    async fn test_health_check() {
        let graph = KnowledgeGraph::new();
        let storage = Box::new(InMemoryGraphStorage::new());
        let config = GraphRetrievalConfig::default();
        
        let retriever = GraphRetriever::new(graph, storage, config).unwrap();
        let is_healthy = retriever.health_check().await.unwrap();
        assert!(is_healthy);
    }
}