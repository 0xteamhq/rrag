//! # Core Graph Abstractions
//! 
//! This module contains the fundamental types and traits that form the foundation
//! of the RGraph system, including the workflow graph, nodes, edges, and execution context.

use crate::{RGraphError, RGraphResult};
use crate::state::GraphState;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use petgraph::{Graph, Directed};
type NodeIndex = petgraph::graph::NodeIndex;
#[allow(dead_code)]
type EdgeIndex = petgraph::graph::EdgeIndex;
use parking_lot::RwLock;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for a node in the workflow graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeId(pub String);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    
    /// Generate a random node ID
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }
    
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for NodeId {
    fn from(id: String) -> Self {
        NodeId(id)
    }
}

impl From<&str> for NodeId {
    fn from(id: &str) -> Self {
        NodeId(id.to_string())
    }
}

/// Unique identifier for an edge in the workflow graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EdgeId(pub String);

impl EdgeId {
    /// Create a new edge ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    
    /// Generate a random edge ID
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

/// Core trait for all executable nodes in the workflow graph
#[async_trait]
pub trait Node: Send + Sync {
    /// Execute the node with the given state and context
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult>;
    
    /// Get the node's unique identifier
    fn id(&self) -> &NodeId;
    
    /// Get the node's display name
    fn name(&self) -> &str;
    
    /// Get the node's description
    fn description(&self) -> Option<&str> {
        None
    }
    
    /// Get the expected input keys from the state
    fn input_keys(&self) -> Vec<&str> {
        vec![]
    }
    
    /// Get the output keys that this node will write to the state
    fn output_keys(&self) -> Vec<&str> {
        vec![]
    }
    
    /// Validate that the node can execute with the current state
    fn validate(&self, _state: &GraphState) -> RGraphResult<()> {
        Ok(())
    }
    
    /// Get node metadata for observability
    fn metadata(&self) -> NodeMetadata {
        NodeMetadata {
            id: self.id().clone(),
            name: self.name().to_string(),
            description: self.description().map(|s| s.to_string()),
            input_keys: self.input_keys().iter().map(|s| s.to_string()).collect(),
            output_keys: self.output_keys().iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Metadata about a node for introspection and observability
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeMetadata {
    pub id: NodeId,
    pub name: String,
    pub description: Option<String>,
    pub input_keys: Vec<String>,
    pub output_keys: Vec<String>,
}

/// Represents an edge in the workflow graph
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Edge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub condition: Option<EdgeCondition>,
}

/// Condition that must be met for an edge to be traversed
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EdgeCondition {
    /// Always traverse the edge
    Always,
    /// Traverse if the condition function returns true
    Conditional(String), // Serialized condition function
    /// Traverse if the state contains a specific value
    StateCondition {
        key: String,
        expected_value: serde_json::Value,
    },
}

/// Result of executing a node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExecutionResult {
    /// Continue to the next node
    Continue,
    /// Stop execution and return the current state
    Stop,
    /// Jump to a specific node
    JumpTo(NodeId),
    /// Conditional routing based on state
    Route(String), // Next node ID based on routing logic
}

/// Context information available during node execution
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub graph_id: String,
    pub execution_id: String,
    pub current_node: NodeId,
    pub execution_path: Vec<NodeId>,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionContext {
    pub fn new(graph_id: String, current_node: NodeId) -> Self {
        Self {
            graph_id,
            execution_id: Uuid::new_v4().to_string(),
            current_node,
            execution_path: Vec::new(),
            start_time: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// The main workflow graph that orchestrates node execution
pub struct WorkflowGraph {
    id: String,
    name: String,
    description: Option<String>,
    graph: Arc<RwLock<Graph<Arc<dyn Node>, Edge, Directed>>>,
    node_lookup: Arc<RwLock<HashMap<NodeId, NodeIndex>>>,
    entry_points: Arc<RwLock<Vec<NodeId>>>,
    exit_points: Arc<RwLock<Vec<NodeId>>>,
}

impl WorkflowGraph {
    /// Create a new workflow graph
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            description: None,
            graph: Arc::new(RwLock::new(Graph::new())),
            node_lookup: Arc::new(RwLock::new(HashMap::new())),
            entry_points: Arc::new(RwLock::new(Vec::new())),
            exit_points: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Set the graph description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Add a node to the graph
    pub async fn add_node(
        &mut self,
        node_id: impl Into<NodeId>,
        node: Arc<dyn Node>,
    ) -> RGraphResult<()> {
        let node_id = node_id.into();
        
        // Validate the node
        let dummy_state = GraphState::new();
        node.validate(&dummy_state)?;
        
        let mut graph = self.graph.write();
        let mut lookup = self.node_lookup.write();
        
        // Check if node already exists
        if lookup.contains_key(&node_id) {
            return Err(RGraphError::validation(
                format!("Node '{}' already exists", node_id.as_str())
            ));
        }
        
        // Add node to the graph
        let node_index = graph.add_node(node);
        lookup.insert(node_id.clone(), node_index);
        
        // If this is the first node, make it an entry point
        if lookup.len() == 1 {
            self.entry_points.write().push(node_id);
        }
        
        Ok(())
    }
    
    /// Add an edge between two nodes
    pub fn add_edge(
        &mut self,
        from: impl Into<NodeId>,
        to: impl Into<NodeId>,
    ) -> RGraphResult<EdgeId> {
        self.add_edge_with_condition(from, to, EdgeCondition::Always)
    }
    
    /// Add an edge with a condition
    pub fn add_edge_with_condition(
        &mut self,
        from: impl Into<NodeId>,
        to: impl Into<NodeId>,
        condition: EdgeCondition,
    ) -> RGraphResult<EdgeId> {
        let from_id = from.into();
        let to_id = to.into();
        let edge_id = EdgeId::generate();
        
        let graph_lock = self.graph.clone();
        let lookup_lock = self.node_lookup.clone();
        
        let mut graph = graph_lock.write();
        let lookup = lookup_lock.read();
        
        // Get node indices
        let from_index = lookup.get(&from_id)
            .ok_or_else(|| RGraphError::validation(
                format!("Node '{}' not found", from_id.as_str())
            ))?;
        let to_index = lookup.get(&to_id)
            .ok_or_else(|| RGraphError::validation(
                format!("Node '{}' not found", to_id.as_str())
            ))?;
        
        // Create edge
        let edge = Edge {
            id: edge_id.clone(),
            from: from_id,
            to: to_id,
            condition: Some(condition),
        };
        
        // Add edge to graph
        graph.add_edge(*from_index, *to_index, edge);
        
        Ok(edge_id)
    }
    
    /// Add a conditional edge with a custom routing function
    pub fn add_conditional_edge<F>(
        &mut self,
        from: impl Into<NodeId>,
        _condition_fn: F,
    ) -> RGraphResult<EdgeId>
    where
        F: Fn(&GraphState) -> RGraphResult<String> + Send + Sync + 'static,
    {
        // In a real implementation, we'd store the condition function
        // For now, we'll create a placeholder edge
        let _from_id = from.into();
        let edge_id = EdgeId::generate();
        
        // This is a simplified implementation - in reality, we'd need to handle
        // the conditional routing during execution
        Ok(edge_id)
    }
    
    /// Set entry points for the graph
    pub fn set_entry_points(&mut self, entry_points: Vec<NodeId>) {
        *self.entry_points.write() = entry_points;
    }
    
    /// Set exit points for the graph
    pub fn set_exit_points(&mut self, exit_points: Vec<NodeId>) {
        *self.exit_points.write() = exit_points;
    }
    
    /// Get the graph ID
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Get the graph name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the graph description
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    
    /// Get all node IDs in the graph
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.node_lookup.read().keys().cloned().collect()
    }
    
    /// Get entry points (returns owned values to avoid lifetime issues)
    pub fn entry_points(&self) -> Vec<NodeId> {
        self.entry_points.read().clone()
    }
    
    /// Get entry points as owned values
    pub fn entry_points_owned(&self) -> Vec<NodeId> {
        self.entry_points.read().clone()
    }
    
    /// Get a node by ID
    pub fn get_node(&self, node_id: &NodeId) -> Option<Arc<dyn Node>> {
        let lookup = self.node_lookup.read();
        let graph = self.graph.read();
        
        if let Some(&node_index) = lookup.get(node_id) {
            if let Some(node_weight) = graph.node_weight(node_index) {
                return Some(node_weight.clone());
            }
        }
        None
    }
    
    /// Validate the graph structure
    pub fn validate(&self) -> RGraphResult<()> {
        let lookup = self.node_lookup.read();
        let entry_points = self.entry_points.read();
        
        // Check that we have nodes
        if lookup.is_empty() {
            return Err(RGraphError::validation("Graph has no nodes"));
        }
        
        // Check that we have entry points
        if entry_points.is_empty() {
            return Err(RGraphError::validation("Graph has no entry points"));
        }
        
        // Validate that all entry points exist
        for entry_point in entry_points.iter() {
            if !lookup.contains_key(entry_point) {
                return Err(RGraphError::validation(
                    format!("Entry point '{}' does not exist", entry_point.as_str())
                ));
            }
        }
        
        Ok(())
    }
}

/// Builder for creating workflow graphs with a fluent API
pub struct GraphBuilder {
    graph: WorkflowGraph,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph: WorkflowGraph::new(name),
        }
    }
    
    /// Set the graph description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.graph = self.graph.with_description(description);
        self
    }
    
    /// Add a node to the graph
    pub async fn add_node(
        mut self,
        node_id: impl Into<NodeId>,
        node: Arc<dyn Node>,
    ) -> RGraphResult<Self> {
        self.graph.add_node(node_id, node).await?;
        Ok(self)
    }
    
    /// Add an edge between two nodes
    pub fn add_edge(
        mut self,
        from: impl Into<NodeId>,
        to: impl Into<NodeId>,
    ) -> RGraphResult<Self> {
        self.graph.add_edge(from, to)?;
        Ok(self)
    }
    
    /// Set entry points
    pub fn entry_points(mut self, entry_points: Vec<NodeId>) -> Self {
        self.graph.set_entry_points(entry_points);
        self
    }
    
    /// Build the workflow graph
    pub fn build(self) -> RGraphResult<WorkflowGraph> {
        self.graph.validate()?;
        Ok(self.graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateValue;
    
    // Mock node for testing
    struct TestNode {
        id: NodeId,
        name: String,
    }
    
    impl TestNode {
        fn new(id: impl Into<NodeId>, name: impl Into<String>) -> Arc<Self> {
            Arc::new(Self {
                id: id.into(),
                name: name.into(),
            })
        }
    }
    
    #[async_trait]
    impl Node for TestNode {
        async fn execute(
            &self,
            state: &mut GraphState,
            _context: &ExecutionContext,
        ) -> RGraphResult<ExecutionResult> {
            state.set("executed_nodes", StateValue::Array(vec![
                StateValue::String(self.name.clone())
            ]));
            Ok(ExecutionResult::Continue)
        }
        
        fn id(&self) -> &NodeId {
            &self.id
        }
        
        fn name(&self) -> &str {
            &self.name
        }
    }
    
    #[tokio::test]
    async fn test_graph_creation() {
        let mut graph = WorkflowGraph::new("test_graph");
        assert_eq!(graph.name(), "test_graph");
        
        let node = TestNode::new("test_node", "Test Node");
        graph.add_node("test_node", node).await.unwrap();
        
        assert_eq!(graph.node_ids().len(), 1);
        assert!(graph.node_ids().contains(&NodeId::new("test_node")));
    }
    
    #[tokio::test]
    async fn test_graph_builder() {
        let node1 = TestNode::new("node1", "Node 1");
        let node2 = TestNode::new("node2", "Node 2");
        
        let graph = GraphBuilder::new("test_graph")
            .description("A test graph")
            .add_node("node1", node1).await.unwrap()
            .add_node("node2", node2).await.unwrap()
            .add_edge("node1", "node2").unwrap()
            .build().unwrap();
        
        assert_eq!(graph.name(), "test_graph");
        assert_eq!(graph.description(), Some("A test graph"));
        assert_eq!(graph.node_ids().len(), 2);
    }
    
    #[test]
    fn test_node_id() {
        let id1 = NodeId::new("test");
        let id2 = NodeId::from("test");
        let id3: NodeId = "test".into();
        
        assert_eq!(id1, id2);
        assert_eq!(id2, id3);
        assert_eq!(id1.as_str(), "test");
    }
    
    #[test]
    fn test_execution_context() {
        let context = ExecutionContext::new("graph1".to_string(), NodeId::new("node1"))
            .with_metadata("key".to_string(), serde_json::json!("value"));
        
        assert_eq!(context.graph_id, "graph1");
        assert_eq!(context.current_node, NodeId::new("node1"));
        assert!(context.metadata.contains_key("key"));
    }
}