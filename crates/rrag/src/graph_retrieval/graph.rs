//! # Knowledge Graph Core
//!
//! Core graph structures and operations for knowledge graph construction and management.

use crate::{Embedding, RragResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Core knowledge graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// Graph nodes indexed by ID
    pub nodes: HashMap<String, GraphNode>,

    /// Graph edges indexed by ID
    pub edges: HashMap<String, GraphEdge>,

    /// Node adjacency list for efficient traversal
    pub adjacency_list: HashMap<String, HashSet<String>>,

    /// Reverse adjacency list for incoming edges
    pub reverse_adjacency_list: HashMap<String, HashSet<String>>,

    /// Graph metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Graph creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Graph node representing entities and concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: String,

    /// Node label/name
    pub label: String,

    /// Node type
    pub node_type: NodeType,

    /// Node attributes
    pub attributes: HashMap<String, serde_json::Value>,

    /// Associated embedding for semantic operations
    pub embedding: Option<Embedding>,

    /// Source document references
    pub source_documents: HashSet<String>,

    /// Node confidence score
    pub confidence: f32,

    /// PageRank score
    pub pagerank_score: Option<f32>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Graph edge representing relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique edge identifier
    pub id: String,

    /// Source node ID
    pub source_id: String,

    /// Target node ID
    pub target_id: String,

    /// Edge label/relationship type
    pub label: String,

    /// Edge type
    pub edge_type: EdgeType,

    /// Edge attributes
    pub attributes: HashMap<String, serde_json::Value>,

    /// Edge weight/strength
    pub weight: f32,

    /// Edge confidence score
    pub confidence: f32,

    /// Source document references
    pub source_documents: HashSet<String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Node types in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Named entity (person, organization, location, etc.)
    Entity(String),

    /// Concept or topic
    Concept,

    /// Document node
    Document,

    /// Document chunk/segment
    DocumentChunk,

    /// Keyword or term
    Keyword,

    /// Custom node type
    Custom(String),
}

/// Edge types in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Semantic relationship
    Semantic(String),

    /// Hierarchical relationship
    Hierarchical,

    /// Document containment relationship
    Contains,

    /// Reference/citation relationship
    References,

    /// Co-occurrence relationship
    CoOccurs,

    /// Similarity relationship
    Similar,

    /// Custom edge type
    Custom(String),
}

/// Graph metrics and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Number of nodes
    pub node_count: usize,

    /// Number of edges
    pub edge_count: usize,

    /// Number of connected components
    pub connected_components: usize,

    /// Graph density
    pub density: f32,

    /// Average degree
    pub average_degree: f32,

    /// Maximum degree
    pub max_degree: usize,

    /// Clustering coefficient
    pub clustering_coefficient: f32,

    /// Diameter (longest shortest path)
    pub diameter: Option<usize>,

    /// Node type distribution
    pub node_type_distribution: HashMap<String, usize>,

    /// Edge type distribution
    pub edge_type_distribution: HashMap<String, usize>,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency_list: HashMap::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> RragResult<()> {
        let node_id = node.id.clone();

        // Initialize adjacency lists
        self.adjacency_list
            .entry(node_id.clone())
            .or_insert_with(HashSet::new);
        self.reverse_adjacency_list
            .entry(node_id.clone())
            .or_insert_with(HashSet::new);

        // Insert node
        self.nodes.insert(node_id, node);
        self.updated_at = chrono::Utc::now();

        Ok(())
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) -> RragResult<()> {
        let edge_id = edge.id.clone();
        let source_id = edge.source_id.clone();
        let target_id = edge.target_id.clone();

        // Verify nodes exist
        if !self.nodes.contains_key(&source_id) {
            return Err(crate::RragError::retrieval(format!(
                "Source node {} not found",
                source_id
            )));
        }

        if !self.nodes.contains_key(&target_id) {
            return Err(crate::RragError::retrieval(format!(
                "Target node {} not found",
                target_id
            )));
        }

        // Update adjacency lists
        self.adjacency_list
            .entry(source_id.clone())
            .or_insert_with(HashSet::new)
            .insert(target_id.clone());

        self.reverse_adjacency_list
            .entry(target_id.clone())
            .or_insert_with(HashSet::new)
            .insert(source_id.clone());

        // Insert edge
        self.edges.insert(edge_id, edge);
        self.updated_at = chrono::Utc::now();

        Ok(())
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&GraphNode> {
        self.nodes.get(node_id)
    }

    /// Get edge by ID
    pub fn get_edge(&self, edge_id: &str) -> Option<&GraphEdge> {
        self.edges.get(edge_id)
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<&GraphNode> {
        self.adjacency_list
            .get(node_id)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter_map(|neighbor_id| self.nodes.get(neighbor_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: &str) -> Vec<&GraphNode> {
        self.reverse_adjacency_list
            .get(node_id)
            .map(|neighbors| {
                neighbors
                    .iter()
                    .filter_map(|neighbor_id| self.nodes.get(neighbor_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get edges connected to a node
    pub fn get_node_edges(&self, node_id: &str) -> Vec<&GraphEdge> {
        self.edges
            .values()
            .filter(|edge| edge.source_id == node_id || edge.target_id == node_id)
            .collect()
    }

    /// Remove a node and its connected edges
    pub fn remove_node(&mut self, node_id: &str) -> RragResult<()> {
        // Remove from adjacency lists
        if let Some(neighbors) = self.adjacency_list.remove(node_id) {
            for neighbor in neighbors {
                if let Some(reverse_neighbors) = self.reverse_adjacency_list.get_mut(&neighbor) {
                    reverse_neighbors.remove(node_id);
                }
            }
        }

        if let Some(incoming_neighbors) = self.reverse_adjacency_list.remove(node_id) {
            for neighbor in incoming_neighbors {
                if let Some(outgoing_neighbors) = self.adjacency_list.get_mut(&neighbor) {
                    outgoing_neighbors.remove(node_id);
                }
            }
        }

        // Remove connected edges
        let edges_to_remove: Vec<String> = self
            .edges
            .iter()
            .filter(|(_, edge)| edge.source_id == node_id || edge.target_id == node_id)
            .map(|(edge_id, _)| edge_id.clone())
            .collect();

        for edge_id in edges_to_remove {
            self.edges.remove(&edge_id);
        }

        // Remove node
        self.nodes.remove(node_id);
        self.updated_at = chrono::Utc::now();

        Ok(())
    }

    /// Find nodes by type
    pub fn find_nodes_by_type(&self, node_type: &NodeType) -> Vec<&GraphNode> {
        self.nodes
            .values()
            .filter(|node| &node.node_type == node_type)
            .collect()
    }

    /// Find edges by type
    pub fn find_edges_by_type(&self, edge_type: &EdgeType) -> Vec<&GraphEdge> {
        self.edges
            .values()
            .filter(|edge| &edge.edge_type == edge_type)
            .collect()
    }

    /// Calculate graph metrics
    pub fn calculate_metrics(&self) -> GraphMetrics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        // Calculate density
        let max_edges = if node_count > 1 {
            node_count * (node_count - 1)
        } else {
            0
        };
        let density = if max_edges > 0 {
            edge_count as f32 / max_edges as f32
        } else {
            0.0
        };

        // Calculate degree statistics
        let degrees: Vec<usize> = self
            .adjacency_list
            .values()
            .map(|neighbors| neighbors.len())
            .collect();

        let average_degree = if !degrees.is_empty() {
            degrees.iter().sum::<usize>() as f32 / degrees.len() as f32
        } else {
            0.0
        };

        let max_degree = degrees.iter().max().copied().unwrap_or(0);

        // Calculate connected components
        let connected_components = self.count_connected_components();

        // Calculate clustering coefficient
        let clustering_coefficient = self.calculate_clustering_coefficient();

        // Node type distribution
        let mut node_type_distribution = HashMap::new();
        for node in self.nodes.values() {
            let type_key = self.node_type_key(&node.node_type);
            *node_type_distribution.entry(type_key).or_insert(0) += 1;
        }

        // Edge type distribution
        let mut edge_type_distribution = HashMap::new();
        for edge in self.edges.values() {
            let type_key = self.edge_type_key(&edge.edge_type);
            *edge_type_distribution.entry(type_key).or_insert(0) += 1;
        }

        GraphMetrics {
            node_count,
            edge_count,
            connected_components,
            density,
            average_degree,
            max_degree,
            clustering_coefficient,
            diameter: None, // Expensive to calculate, could be computed on demand
            node_type_distribution,
            edge_type_distribution,
        }
    }

    /// Count connected components using DFS
    fn count_connected_components(&self) -> usize {
        let mut visited = HashSet::new();
        let mut components = 0;

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id) {
                self.dfs_component(node_id, &mut visited);
                components += 1;
            }
        }

        components
    }

    /// DFS helper for connected components
    fn dfs_component(&self, node_id: &str, visited: &mut HashSet<String>) {
        visited.insert(node_id.to_string());

        if let Some(neighbors) = self.adjacency_list.get(node_id) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    self.dfs_component(neighbor, visited);
                }
            }
        }

        if let Some(incoming_neighbors) = self.reverse_adjacency_list.get(node_id) {
            for neighbor in incoming_neighbors {
                if !visited.contains(neighbor) {
                    self.dfs_component(neighbor, visited);
                }
            }
        }
    }

    /// Calculate clustering coefficient
    fn calculate_clustering_coefficient(&self) -> f32 {
        let mut total_coefficient = 0.0;
        let mut nodes_with_neighbors = 0;

        for (_node_id, neighbors) in &self.adjacency_list {
            if neighbors.len() < 2 {
                continue;
            }

            let neighbor_count = neighbors.len();
            let possible_edges = neighbor_count * (neighbor_count - 1) / 2;

            // Count actual edges between neighbors
            let mut actual_edges = 0;
            let neighbor_vec: Vec<_> = neighbors.iter().collect();

            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    let neighbor1 = neighbor_vec[i];
                    let neighbor2 = neighbor_vec[j];

                    if let Some(neighbor1_neighbors) = self.adjacency_list.get(neighbor1) {
                        if neighbor1_neighbors.contains(neighbor2) {
                            actual_edges += 1;
                        }
                    }
                }
            }

            if possible_edges > 0 {
                total_coefficient += actual_edges as f32 / possible_edges as f32;
                nodes_with_neighbors += 1;
            }
        }

        if nodes_with_neighbors > 0 {
            total_coefficient / nodes_with_neighbors as f32
        } else {
            0.0
        }
    }

    /// Get string representation of node type
    fn node_type_key(&self, node_type: &NodeType) -> String {
        match node_type {
            NodeType::Entity(entity_type) => format!("Entity({})", entity_type),
            NodeType::Concept => "Concept".to_string(),
            NodeType::Document => "Document".to_string(),
            NodeType::DocumentChunk => "DocumentChunk".to_string(),
            NodeType::Keyword => "Keyword".to_string(),
            NodeType::Custom(custom_type) => format!("Custom({})", custom_type),
        }
    }

    /// Get string representation of edge type
    fn edge_type_key(&self, edge_type: &EdgeType) -> String {
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

    /// Merge another graph into this one
    pub fn merge(&mut self, other: &KnowledgeGraph) -> RragResult<()> {
        // Add nodes
        for (_, node) in &other.nodes {
            if !self.nodes.contains_key(&node.id) {
                self.add_node(node.clone())?;
            }
        }

        // Add edges
        for (_, edge) in &other.edges {
            if !self.edges.contains_key(&edge.id) {
                self.add_edge(edge.clone())?;
            }
        }

        Ok(())
    }

    /// Clear the entire graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.adjacency_list.clear();
        self.reverse_adjacency_list.clear();
        self.updated_at = chrono::Utc::now();
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(label: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            label: label.into(),
            node_type,
            attributes: HashMap::new(),
            embedding: None,
            source_documents: HashSet::new(),
            confidence: 1.0,
            pagerank_score: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create node with specific ID
    pub fn with_id(id: impl Into<String>, label: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            node_type,
            attributes: HashMap::new(),
            embedding: None,
            source_documents: HashSet::new(),
            confidence: 1.0,
            pagerank_score: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Add attribute using builder pattern
    pub fn with_attribute(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add source document
    pub fn with_source_document(mut self, document_id: impl Into<String>) -> Self {
        self.source_documents.insert(document_id.into());
        self
    }
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        label: impl Into<String>,
        edge_type: EdgeType,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            label: label.into(),
            edge_type,
            attributes: HashMap::new(),
            weight: 1.0,
            confidence: 1.0,
            source_documents: HashSet::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Create edge with specific ID
    pub fn with_id(
        id: impl Into<String>,
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        label: impl Into<String>,
        edge_type: EdgeType,
    ) -> Self {
        Self {
            id: id.into(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            label: label.into(),
            edge_type,
            attributes: HashMap::new(),
            weight: 1.0,
            confidence: 1.0,
            source_documents: HashSet::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Add attribute using builder pattern
    pub fn with_attribute(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    /// Set edge weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add source document
    pub fn with_source_document(mut self, document_id: impl Into<String>) -> Self {
        self.source_documents.insert(document_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_creation() {
        let graph = KnowledgeGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_add_node() {
        let mut graph = KnowledgeGraph::new();
        let node = GraphNode::new("test_entity", NodeType::Entity("Person".to_string()));
        let node_id = node.id.clone();

        graph.add_node(node).unwrap();
        assert!(graph.nodes.contains_key(&node_id));
        assert!(graph.adjacency_list.contains_key(&node_id));
    }

    #[test]
    fn test_add_edge() {
        let mut graph = KnowledgeGraph::new();

        let node1 = GraphNode::new("person1", NodeType::Entity("Person".to_string()));
        let node2 = GraphNode::new("person2", NodeType::Entity("Person".to_string()));
        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();

        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();

        let edge = GraphEdge::new(
            node1_id.clone(),
            node2_id.clone(),
            "knows",
            EdgeType::Semantic("knows".to_string()),
        );

        graph.add_edge(edge).unwrap();

        assert_eq!(graph.edges.len(), 1);
        assert!(graph.adjacency_list[&node1_id].contains(&node2_id));
        assert!(graph.reverse_adjacency_list[&node2_id].contains(&node1_id));
    }

    #[test]
    fn test_get_neighbors() {
        let mut graph = KnowledgeGraph::new();

        let node1 = GraphNode::new("node1", NodeType::Concept);
        let node2 = GraphNode::new("node2", NodeType::Concept);
        let node3 = GraphNode::new("node3", NodeType::Concept);

        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        let node3_id = node3.id.clone();

        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();
        graph.add_node(node3).unwrap();

        graph
            .add_edge(GraphEdge::new(
                node1_id.clone(),
                node2_id.clone(),
                "connected",
                EdgeType::Similar,
            ))
            .unwrap();

        graph
            .add_edge(GraphEdge::new(
                node1_id.clone(),
                node3_id.clone(),
                "connected",
                EdgeType::Similar,
            ))
            .unwrap();

        let neighbors = graph.get_neighbors(&node1_id);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_graph_metrics() {
        let mut graph = KnowledgeGraph::new();

        // Create a simple graph with 3 nodes and 2 edges
        let node1 = GraphNode::new("node1", NodeType::Concept);
        let node2 = GraphNode::new("node2", NodeType::Concept);
        let node3 = GraphNode::new("node3", NodeType::Concept);

        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        let node3_id = node3.id.clone();

        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();
        graph.add_node(node3).unwrap();

        graph
            .add_edge(GraphEdge::new(
                node1_id.clone(),
                node2_id.clone(),
                "edge1",
                EdgeType::Similar,
            ))
            .unwrap();

        graph
            .add_edge(GraphEdge::new(
                node2_id.clone(),
                node3_id.clone(),
                "edge2",
                EdgeType::Similar,
            ))
            .unwrap();

        let metrics = graph.calculate_metrics();
        assert_eq!(metrics.node_count, 3);
        assert_eq!(metrics.edge_count, 2);
        assert_eq!(metrics.connected_components, 1);
    }
}
