//! # Graph Storage and Indexing
//! 
//! Efficient storage and indexing system for knowledge graphs with support for
//! various storage backends and optimized query operations.

use super::{KnowledgeGraph, GraphNode, GraphEdge, NodeType, EdgeType, GraphError};
use crate::{RragResult, Embedding};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use async_trait::async_trait;

/// Graph storage trait for different storage backends
#[async_trait]
pub trait GraphStorage: Send + Sync {
    /// Store a complete knowledge graph
    async fn store_graph(&self, graph: &KnowledgeGraph) -> RragResult<()>;
    
    /// Load a complete knowledge graph
    async fn load_graph(&self, graph_id: &str) -> RragResult<KnowledgeGraph>;
    
    /// Store individual nodes
    async fn store_nodes(&self, nodes: &[GraphNode]) -> RragResult<()>;
    
    /// Store individual edges
    async fn store_edges(&self, edges: &[GraphEdge]) -> RragResult<()>;
    
    /// Query nodes by criteria
    async fn query_nodes(&self, query: &NodeQuery) -> RragResult<Vec<GraphNode>>;
    
    /// Query edges by criteria
    async fn query_edges(&self, query: &EdgeQuery) -> RragResult<Vec<GraphEdge>>;
    
    /// Get node by ID
    async fn get_node(&self, node_id: &str) -> RragResult<Option<GraphNode>>;
    
    /// Get edge by ID
    async fn get_edge(&self, edge_id: &str) -> RragResult<Option<GraphEdge>>;
    
    /// Get neighbors of a node
    async fn get_neighbors(&self, node_id: &str, direction: EdgeDirection) -> RragResult<Vec<GraphNode>>;
    
    /// Delete nodes
    async fn delete_nodes(&self, node_ids: &[String]) -> RragResult<()>;
    
    /// Delete edges
    async fn delete_edges(&self, edge_ids: &[String]) -> RragResult<()>;
    
    /// Clear all data
    async fn clear(&self) -> RragResult<()>;
    
    /// Get storage statistics
    async fn get_stats(&self) -> RragResult<StorageStats>;
}

/// In-memory graph storage implementation
pub struct InMemoryGraphStorage {
    /// Stored graphs
    graphs: tokio::sync::RwLock<HashMap<String, KnowledgeGraph>>,
    
    /// Global node index
    node_index: tokio::sync::RwLock<GraphIndex<GraphNode>>,
    
    /// Global edge index
    edge_index: tokio::sync::RwLock<GraphIndex<GraphEdge>>,
    
    /// Configuration
    config: GraphStorageConfig,
}

/// Graph indexing system for fast queries
#[derive(Debug, Clone)]
pub struct GraphIndex<T> {
    /// Primary index by ID
    by_id: HashMap<String, T>,
    
    /// Secondary indices
    indices: HashMap<String, BTreeMap<String, HashSet<String>>>,
    
    /// Full-text search index (simple implementation)
    text_index: HashMap<String, HashSet<String>>,
}

/// Node query parameters
#[derive(Debug, Clone)]
pub struct NodeQuery {
    /// Node IDs to match
    pub node_ids: Option<Vec<String>>,
    
    /// Node types to match
    pub node_types: Option<Vec<NodeType>>,
    
    /// Text search in labels
    pub text_search: Option<String>,
    
    /// Attribute filters
    pub attribute_filters: HashMap<String, serde_json::Value>,
    
    /// Source document filters
    pub source_document_filters: Option<Vec<String>>,
    
    /// Confidence threshold
    pub min_confidence: Option<f32>,
    
    /// Limit number of results
    pub limit: Option<usize>,
    
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Edge query parameters
#[derive(Debug, Clone)]
pub struct EdgeQuery {
    /// Edge IDs to match
    pub edge_ids: Option<Vec<String>>,
    
    /// Source node IDs
    pub source_node_ids: Option<Vec<String>>,
    
    /// Target node IDs
    pub target_node_ids: Option<Vec<String>>,
    
    /// Edge types to match
    pub edge_types: Option<Vec<EdgeType>>,
    
    /// Text search in labels
    pub text_search: Option<String>,
    
    /// Attribute filters
    pub attribute_filters: HashMap<String, serde_json::Value>,
    
    /// Weight range
    pub weight_range: Option<(f32, f32)>,
    
    /// Confidence threshold
    pub min_confidence: Option<f32>,
    
    /// Limit number of results
    pub limit: Option<usize>,
    
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Edge direction for neighbor queries
#[derive(Debug, Clone, Copy)]
pub enum EdgeDirection {
    /// Outgoing edges from the node
    Outgoing,
    
    /// Incoming edges to the node
    Incoming,
    
    /// Both directions
    Both,
}

/// Graph query for complex graph operations
#[derive(Debug, Clone)]
pub struct GraphQuery {
    /// Starting nodes for the query
    pub start_nodes: Vec<String>,
    
    /// Query pattern (simplified graph pattern matching)
    pub pattern: GraphPattern,
    
    /// Maximum traversal depth
    pub max_depth: usize,
    
    /// Result limit
    pub limit: Option<usize>,
}

/// Graph pattern for pattern matching
#[derive(Debug, Clone)]
pub struct GraphPattern {
    /// Node patterns
    pub node_patterns: Vec<NodePattern>,
    
    /// Edge patterns
    pub edge_patterns: Vec<EdgePattern>,
    
    /// Pattern constraints
    pub constraints: Vec<PatternConstraint>,
}

/// Node pattern for pattern matching
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// Pattern variable name
    pub variable: String,
    
    /// Node type constraint
    pub node_type: Option<NodeType>,
    
    /// Label constraint
    pub label_pattern: Option<String>,
    
    /// Attribute constraints
    pub attribute_constraints: HashMap<String, serde_json::Value>,
}

/// Edge pattern for pattern matching
#[derive(Debug, Clone)]
pub struct EdgePattern {
    /// Source node variable
    pub source_variable: String,
    
    /// Target node variable
    pub target_variable: String,
    
    /// Edge type constraint
    pub edge_type: Option<EdgeType>,
    
    /// Label constraint
    pub label_pattern: Option<String>,
    
    /// Attribute constraints
    pub attribute_constraints: HashMap<String, serde_json::Value>,
}

/// Pattern constraint
#[derive(Debug, Clone)]
pub enum PatternConstraint {
    /// Distance constraint between two nodes
    Distance { var1: String, var2: String, max_distance: usize },
    
    /// Path constraint
    Path { start_var: String, end_var: String, path_type: PathType },
    
    /// Count constraint
    Count { variable: String, min_count: usize, max_count: Option<usize> },
}

/// Path type for path constraints
#[derive(Debug, Clone)]
pub enum PathType {
    /// Any path
    Any,
    
    /// Shortest path
    Shortest,
    
    /// Path with specific edge types
    EdgeTypes(Vec<EdgeType>),
}

/// Query result for graph queries
#[derive(Debug, Clone)]
pub struct GraphQueryResult {
    /// Matched variable bindings
    pub bindings: Vec<HashMap<String, String>>,
    
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    
    /// Number of nodes examined
    pub nodes_examined: usize,
    
    /// Number of edges examined
    pub edges_examined: usize,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Number of stored graphs
    pub graph_count: usize,
    
    /// Total number of nodes
    pub total_nodes: usize,
    
    /// Total number of edges
    pub total_edges: usize,
    
    /// Storage size in bytes (estimate)
    pub storage_size_bytes: usize,
    
    /// Index size in bytes (estimate)
    pub index_size_bytes: usize,
    
    /// Node type distribution
    pub node_type_distribution: HashMap<String, usize>,
    
    /// Edge type distribution
    pub edge_type_distribution: HashMap<String, usize>,
    
    /// Last update timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Graph storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStorageConfig {
    /// Enable full-text indexing
    pub enable_text_indexing: bool,
    
    /// Enable attribute indexing
    pub enable_attribute_indexing: bool,
    
    /// Maximum cache size for frequently accessed items
    pub max_cache_size: usize,
    
    /// Batch size for bulk operations
    pub batch_size: usize,
    
    /// Enable compression for storage
    pub enable_compression: bool,
}

impl Default for GraphStorageConfig {
    fn default() -> Self {
        Self {
            enable_text_indexing: true,
            enable_attribute_indexing: true,
            max_cache_size: 10_000,
            batch_size: 1_000,
            enable_compression: false,
        }
    }
}

impl<T> GraphIndex<T>
where
    T: Clone + Send + Sync,
{
    /// Create a new graph index
    pub fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            indices: HashMap::new(),
            text_index: HashMap::new(),
        }
    }

    /// Add an item to the index
    pub fn add_item(&mut self, id: String, item: T, indexable_fields: &HashMap<String, String>) {
        // Add to primary index
        self.by_id.insert(id.clone(), item);
        
        // Add to secondary indices
        for (field_name, field_value) in indexable_fields {
            self.indices
                .entry(field_name.clone())
                .or_insert_with(BTreeMap::new)
                .entry(field_value.clone())
                .or_insert_with(HashSet::new)
                .insert(id.clone());
        }
        
        // Add to text index (simple tokenization)
        for (_, field_value) in indexable_fields {
            let tokens = Self::tokenize(field_value);
            for token in tokens {
                self.text_index
                    .entry(token.to_lowercase())
                    .or_insert_with(HashSet::new)
                    .insert(id.clone());
            }
        }
    }

    /// Remove an item from the index
    pub fn remove_item(&mut self, id: &str) {
        self.by_id.remove(id);
        
        // Remove from secondary indices
        for index in self.indices.values_mut() {
            for ids in index.values_mut() {
                ids.remove(id);
            }
        }
        
        // Remove from text index
        for ids in self.text_index.values_mut() {
            ids.remove(id);
        }
    }

    /// Get item by ID
    pub fn get_by_id(&self, id: &str) -> Option<&T> {
        self.by_id.get(id)
    }

    /// Find items by field value
    pub fn find_by_field(&self, field_name: &str, field_value: &str) -> Vec<&T> {
        if let Some(index) = self.indices.get(field_name) {
            if let Some(ids) = index.get(field_value) {
                return ids.iter()
                    .filter_map(|id| self.by_id.get(id))
                    .collect();
            }
        }
        Vec::new()
    }

    /// Text search
    pub fn text_search(&self, query: &str) -> Vec<&T> {
        let tokens = Self::tokenize(query);
        let mut matching_ids = HashSet::new();
        
        for (i, token) in tokens.iter().enumerate() {
            if let Some(ids) = self.text_index.get(&token.to_lowercase()) {
                if i == 0 {
                    matching_ids.extend(ids.clone());
                } else {
                    matching_ids.retain(|id| ids.contains(id));
                }
            }
        }
        
        matching_ids.iter()
            .filter_map(|id| self.by_id.get(id))
            .collect()
    }

    /// Get all items
    pub fn get_all(&self) -> Vec<&T> {
        self.by_id.values().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_items".to_string(), self.by_id.len());
        stats.insert("indices_count".to_string(), self.indices.len());
        stats.insert("text_terms".to_string(), self.text_index.len());
        stats
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.by_id.clear();
        self.indices.clear();
        self.text_index.clear();
    }

    /// Simple tokenization
    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

impl Default for NodeQuery {
    fn default() -> Self {
        Self {
            node_ids: None,
            node_types: None,
            text_search: None,
            attribute_filters: HashMap::new(),
            source_document_filters: None,
            min_confidence: None,
            limit: None,
            offset: None,
        }
    }
}

impl Default for EdgeQuery {
    fn default() -> Self {
        Self {
            edge_ids: None,
            source_node_ids: None,
            target_node_ids: None,
            edge_types: None,
            text_search: None,
            attribute_filters: HashMap::new(),
            weight_range: None,
            min_confidence: None,
            limit: None,
            offset: None,
        }
    }
}

impl InMemoryGraphStorage {
    /// Create a new in-memory graph storage
    pub fn new() -> Self {
        Self::with_config(GraphStorageConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GraphStorageConfig) -> Self {
        Self {
            graphs: tokio::sync::RwLock::new(HashMap::new()),
            node_index: tokio::sync::RwLock::new(GraphIndex::new()),
            edge_index: tokio::sync::RwLock::new(GraphIndex::new()),
            config,
        }
    }

    /// Create indexable fields for a node
    fn create_node_indexable_fields(node: &GraphNode) -> HashMap<String, String> {
        let mut fields = HashMap::new();
        
        fields.insert("label".to_string(), node.label.clone());
        fields.insert("node_type".to_string(), Self::node_type_string(&node.node_type));
        fields.insert("confidence".to_string(), node.confidence.to_string());
        
        // Add attribute fields
        for (key, value) in &node.attributes {
            if let Some(string_value) = value.as_str() {
                fields.insert(format!("attr_{}", key), string_value.to_string());
            }
        }
        
        fields
    }

    /// Create indexable fields for an edge
    fn create_edge_indexable_fields(edge: &GraphEdge) -> HashMap<String, String> {
        let mut fields = HashMap::new();
        
        fields.insert("label".to_string(), edge.label.clone());
        fields.insert("edge_type".to_string(), Self::edge_type_string(&edge.edge_type));
        fields.insert("source_id".to_string(), edge.source_id.clone());
        fields.insert("target_id".to_string(), edge.target_id.clone());
        fields.insert("weight".to_string(), edge.weight.to_string());
        fields.insert("confidence".to_string(), edge.confidence.to_string());
        
        // Add attribute fields
        for (key, value) in &edge.attributes {
            if let Some(string_value) = value.as_str() {
                fields.insert(format!("attr_{}", key), string_value.to_string());
            }
        }
        
        fields
    }

    /// Convert node type to string for indexing
    fn node_type_string(node_type: &NodeType) -> String {
        match node_type {
            NodeType::Entity(entity_type) => format!("Entity({})", entity_type),
            NodeType::Concept => "Concept".to_string(),
            NodeType::Document => "Document".to_string(),
            NodeType::DocumentChunk => "DocumentChunk".to_string(),
            NodeType::Keyword => "Keyword".to_string(),
            NodeType::Custom(custom_type) => format!("Custom({})", custom_type),
        }
    }

    /// Convert edge type to string for indexing
    fn edge_type_string(edge_type: &EdgeType) -> String {
        match edge_type {
            EdgeType::Semantic(relation) => format!("Semantic({})", relation),
            EdgeType::Hierarchical => "Hierarchical".to_string(),
            EdgeType::Contains => "Contains".to_string(),
            EdgeType::References => "References".to_string(),
            EdgeType::CoOccurs => "CoOccurs".to_string(),
            EdgeType::Similar => "Similar".to_string(),
            EdgeType::Custom(custom_type) => format!("Custom({})", custom_type),
        }
    }

    /// Apply filters to node query results
    fn apply_node_filters(&self, nodes: Vec<&GraphNode>, query: &NodeQuery) -> Vec<GraphNode> {
        let mut result: Vec<_> = nodes.into_iter().cloned().collect();
        
        // Apply node type filter
        if let Some(node_types) = &query.node_types {
            result.retain(|node| node_types.contains(&node.node_type));
        }
        
        // Apply confidence filter
        if let Some(min_confidence) = query.min_confidence {
            result.retain(|node| node.confidence >= min_confidence);
        }
        
        // Apply attribute filters
        for (attr_key, attr_value) in &query.attribute_filters {
            result.retain(|node| {
                node.attributes.get(attr_key).map_or(false, |v| v == attr_value)
            });
        }
        
        // Apply source document filter
        if let Some(source_docs) = &query.source_document_filters {
            result.retain(|node| {
                node.source_documents.iter().any(|doc| source_docs.contains(doc))
            });
        }
        
        // Apply pagination
        if let Some(offset) = query.offset {
            if offset < result.len() {
                result.drain(0..offset);
            } else {
                result.clear();
            }
        }
        
        if let Some(limit) = query.limit {
            result.truncate(limit);
        }
        
        result
    }

    /// Apply filters to edge query results
    fn apply_edge_filters(&self, edges: Vec<&GraphEdge>, query: &EdgeQuery) -> Vec<GraphEdge> {
        let mut result: Vec<_> = edges.into_iter().cloned().collect();
        
        // Apply source/target node filters
        if let Some(source_ids) = &query.source_node_ids {
            result.retain(|edge| source_ids.contains(&edge.source_id));
        }
        
        if let Some(target_ids) = &query.target_node_ids {
            result.retain(|edge| target_ids.contains(&edge.target_id));
        }
        
        // Apply edge type filter
        if let Some(edge_types) = &query.edge_types {
            result.retain(|edge| edge_types.contains(&edge.edge_type));
        }
        
        // Apply weight range filter
        if let Some((min_weight, max_weight)) = query.weight_range {
            result.retain(|edge| edge.weight >= min_weight && edge.weight <= max_weight);
        }
        
        // Apply confidence filter
        if let Some(min_confidence) = query.min_confidence {
            result.retain(|edge| edge.confidence >= min_confidence);
        }
        
        // Apply attribute filters
        for (attr_key, attr_value) in &query.attribute_filters {
            result.retain(|edge| {
                edge.attributes.get(attr_key).map_or(false, |v| v == attr_value)
            });
        }
        
        // Apply pagination
        if let Some(offset) = query.offset {
            if offset < result.len() {
                result.drain(0..offset);
            } else {
                result.clear();
            }
        }
        
        if let Some(limit) = query.limit {
            result.truncate(limit);
        }
        
        result
    }
}

#[async_trait]
impl GraphStorage for InMemoryGraphStorage {
    async fn store_graph(&self, graph: &KnowledgeGraph) -> RragResult<()> {
        // Store graph
        let graph_id = uuid::Uuid::new_v4().to_string();
        self.graphs.write().await.insert(graph_id, graph.clone());
        
        // Update indices
        self.store_nodes(&graph.nodes.values().cloned().collect::<Vec<_>>()).await?;
        self.store_edges(&graph.edges.values().cloned().collect::<Vec<_>>()).await?;
        
        Ok(())
    }

    async fn load_graph(&self, graph_id: &str) -> RragResult<KnowledgeGraph> {
        self.graphs.read().await
            .get(graph_id)
            .cloned()
            .ok_or_else(|| GraphError::Storage {
                operation: "load_graph".to_string(),
                message: format!("Graph '{}' not found", graph_id),
            }.into())
    }

    async fn store_nodes(&self, nodes: &[GraphNode]) -> RragResult<()> {
        let mut node_index = self.node_index.write().await;
        
        for node in nodes {
            let indexable_fields = Self::create_node_indexable_fields(node);
            node_index.add_item(node.id.clone(), node.clone(), &indexable_fields);
        }
        
        Ok(())
    }

    async fn store_edges(&self, edges: &[GraphEdge]) -> RragResult<()> {
        let mut edge_index = self.edge_index.write().await;
        
        for edge in edges {
            let indexable_fields = Self::create_edge_indexable_fields(edge);
            edge_index.add_item(edge.id.clone(), edge.clone(), &indexable_fields);
        }
        
        Ok(())
    }

    async fn query_nodes(&self, query: &NodeQuery) -> RragResult<Vec<GraphNode>> {
        let node_index = self.node_index.read().await;
        let mut candidates = Vec::new();
        
        if let Some(node_ids) = &query.node_ids {
            // Query by specific IDs
            for node_id in node_ids {
                if let Some(node) = node_index.get_by_id(node_id) {
                    candidates.push(node);
                }
            }
        } else if let Some(text_query) = &query.text_search {
            // Text search
            candidates = node_index.text_search(text_query);
        } else {
            // Get all nodes
            candidates = node_index.get_all();
        }
        
        Ok(self.apply_node_filters(candidates, query))
    }

    async fn query_edges(&self, query: &EdgeQuery) -> RragResult<Vec<GraphEdge>> {
        let edge_index = self.edge_index.read().await;
        let mut candidates = Vec::new();
        
        if let Some(edge_ids) = &query.edge_ids {
            // Query by specific IDs
            for edge_id in edge_ids {
                if let Some(edge) = edge_index.get_by_id(edge_id) {
                    candidates.push(edge);
                }
            }
        } else if let Some(text_query) = &query.text_search {
            // Text search
            candidates = edge_index.text_search(text_query);
        } else {
            // Get all edges
            candidates = edge_index.get_all();
        }
        
        Ok(self.apply_edge_filters(candidates, query))
    }

    async fn get_node(&self, node_id: &str) -> RragResult<Option<GraphNode>> {
        let node_index = self.node_index.read().await;
        Ok(node_index.get_by_id(node_id).cloned())
    }

    async fn get_edge(&self, edge_id: &str) -> RragResult<Option<GraphEdge>> {
        let edge_index = self.edge_index.read().await;
        Ok(edge_index.get_by_id(edge_id).cloned())
    }

    async fn get_neighbors(
        &self, 
        node_id: &str, 
        direction: EdgeDirection
    ) -> RragResult<Vec<GraphNode>> {
        let edge_index = self.edge_index.read().await;
        let node_index = self.node_index.read().await;
        let mut neighbor_ids = HashSet::new();
        
        match direction {
            EdgeDirection::Outgoing => {
                let outgoing_edges = edge_index.find_by_field("source_id", node_id);
                for edge in outgoing_edges {
                    neighbor_ids.insert(&edge.target_id);
                }
            }
            EdgeDirection::Incoming => {
                let incoming_edges = edge_index.find_by_field("target_id", node_id);
                for edge in incoming_edges {
                    neighbor_ids.insert(&edge.source_id);
                }
            }
            EdgeDirection::Both => {
                let outgoing_edges = edge_index.find_by_field("source_id", node_id);
                for edge in outgoing_edges {
                    neighbor_ids.insert(&edge.target_id);
                }
                let incoming_edges = edge_index.find_by_field("target_id", node_id);
                for edge in incoming_edges {
                    neighbor_ids.insert(&edge.source_id);
                }
            }
        }
        
        let neighbors = neighbor_ids
            .into_iter()
            .filter_map(|id| node_index.get_by_id(id))
            .cloned()
            .collect();
        
        Ok(neighbors)
    }

    async fn delete_nodes(&self, node_ids: &[String]) -> RragResult<()> {
        let mut node_index = self.node_index.write().await;
        
        for node_id in node_ids {
            node_index.remove_item(node_id);
        }
        
        Ok(())
    }

    async fn delete_edges(&self, edge_ids: &[String]) -> RragResult<()> {
        let mut edge_index = self.edge_index.write().await;
        
        for edge_id in edge_ids {
            edge_index.remove_item(edge_id);
        }
        
        Ok(())
    }

    async fn clear(&self) -> RragResult<()> {
        self.graphs.write().await.clear();
        self.node_index.write().await.clear();
        self.edge_index.write().await.clear();
        Ok(())
    }

    async fn get_stats(&self) -> RragResult<StorageStats> {
        let graphs = self.graphs.read().await;
        let node_index = self.node_index.read().await;
        let edge_index = self.edge_index.read().await;
        
        let graph_count = graphs.len();
        let total_nodes = node_index.by_id.len();
        let total_edges = edge_index.by_id.len();
        
        // Calculate node type distribution
        let mut node_type_distribution = HashMap::new();
        for node in node_index.by_id.values() {
            let type_key = Self::node_type_string(&node.node_type);
            *node_type_distribution.entry(type_key).or_insert(0) += 1;
        }
        
        // Calculate edge type distribution
        let mut edge_type_distribution = HashMap::new();
        for edge in edge_index.by_id.values() {
            let type_key = Self::edge_type_string(&edge.edge_type);
            *edge_type_distribution.entry(type_key).or_insert(0) += 1;
        }
        
        // Rough size estimates
        let storage_size_bytes = (total_nodes + total_edges) * 1000; // Rough estimate
        let index_size_bytes = (node_index.indices.len() + edge_index.indices.len()) * 100;
        
        Ok(StorageStats {
            graph_count,
            total_nodes,
            total_edges,
            storage_size_bytes,
            index_size_bytes,
            node_type_distribution,
            edge_type_distribution,
            last_updated: chrono::Utc::now(),
        })
    }
}

impl Default for InMemoryGraphStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_retrieval::{NodeType, EdgeType};

    #[tokio::test]
    async fn test_in_memory_graph_storage() {
        let storage = InMemoryGraphStorage::new();
        
        // Create test nodes
        let node1 = GraphNode::new("Test Node 1", NodeType::Concept);
        let node2 = GraphNode::new("Test Node 2", NodeType::Entity("Person".to_string()));
        
        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        
        // Store nodes
        storage.store_nodes(&[node1.clone(), node2.clone()]).await.unwrap();
        
        // Query nodes
        let mut query = NodeQuery::default();
        query.text_search = Some("Test".to_string());
        
        let results = storage.query_nodes(&query).await.unwrap();
        assert_eq!(results.len(), 2);
        
        // Get specific node
        let retrieved_node = storage.get_node(&node1_id).await.unwrap();
        assert!(retrieved_node.is_some());
        assert_eq!(retrieved_node.unwrap().label, "Test Node 1");
    }

    #[tokio::test]
    async fn test_edge_storage_and_queries() {
        let storage = InMemoryGraphStorage::new();
        
        // Create test nodes
        let node1 = GraphNode::new("Node 1", NodeType::Concept);
        let node2 = GraphNode::new("Node 2", NodeType::Concept);
        
        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        
        storage.store_nodes(&[node1, node2]).await.unwrap();
        
        // Create test edge
        let edge = GraphEdge::new(
            node1_id.clone(),
            node2_id.clone(),
            "test_relation",
            EdgeType::Similar
        );
        let edge_id = edge.id.clone();
        
        // Store edge
        storage.store_edges(&[edge]).await.unwrap();
        
        // Query edges
        let mut query = EdgeQuery::default();
        query.source_node_ids = Some(vec![node1_id.clone()]);
        
        let results = storage.query_edges(&query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_id, node1_id);
        assert_eq!(results[0].target_id, node2_id);
        
        // Get neighbors
        let neighbors = storage.get_neighbors(&node1_id, EdgeDirection::Outgoing).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, node2_id);
    }

    #[tokio::test]
    async fn test_storage_stats() {
        let storage = InMemoryGraphStorage::new();
        
        // Add some test data
        let nodes = vec![
            GraphNode::new("Node 1", NodeType::Concept),
            GraphNode::new("Node 2", NodeType::Entity("Person".to_string())),
            GraphNode::new("Node 3", NodeType::Document),
        ];
        
        storage.store_nodes(&nodes).await.unwrap();
        
        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 0);
        assert!(stats.node_type_distribution.contains_key("Concept"));
        assert!(stats.node_type_distribution.contains_key("Entity(Person)"));
    }
}