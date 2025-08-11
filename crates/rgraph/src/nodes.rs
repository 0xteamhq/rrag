//! # Graph Node Implementations
//! 
//! This module provides various types of nodes that can be used in workflow graphs,
//! including agent nodes, tool nodes, condition nodes, and transform nodes.

pub mod agent;
pub mod tool;
pub mod condition;
pub mod transform;

// Re-export node types
pub use agent::{AgentNode, AgentNodeConfig};
pub use tool::{ToolNode, ToolNodeConfig};
pub use condition::{ConditionNode, ConditionNodeConfig};
pub use transform::{TransformNode, TransformNodeConfig};

use crate::core::NodeId;
use crate::{RGraphError, RGraphResult};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Base configuration for all node types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeConfig {
    /// Node ID
    pub id: NodeId,
    
    /// Display name
    pub name: String,
    
    /// Optional description
    pub description: Option<String>,
    
    /// Input key mappings (state_key -> node_input_key)
    pub input_mappings: HashMap<String, String>,
    
    /// Output key mappings (node_output_key -> state_key)
    pub output_mappings: HashMap<String, String>,
    
    /// Node-specific configuration
    pub config: serde_json::Value,
    
    /// Whether this node can be retried on failure
    pub retryable: bool,
    
    /// Maximum number of retry attempts
    pub max_retries: usize,
    
    /// Tags for organizing and filtering nodes
    pub tags: Vec<String>,
}

impl NodeConfig {
    /// Create a new node configuration
    pub fn new(id: impl Into<NodeId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            input_mappings: HashMap::new(),
            output_mappings: HashMap::new(),
            config: serde_json::Value::Null,
            retryable: false,
            max_retries: 0,
            tags: Vec::new(),
        }
    }
    
    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    
    /// Add an input mapping
    pub fn with_input_mapping(
        mut self,
        state_key: impl Into<String>,
        node_input_key: impl Into<String>,
    ) -> Self {
        self.input_mappings.insert(state_key.into(), node_input_key.into());
        self
    }
    
    /// Add an output mapping
    pub fn with_output_mapping(
        mut self,
        node_output_key: impl Into<String>,
        state_key: impl Into<String>,
    ) -> Self {
        self.output_mappings.insert(node_output_key.into(), state_key.into());
        self
    }
    
    /// Set the configuration
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }
    
    /// Make the node retryable
    pub fn with_retries(mut self, max_retries: usize) -> Self {
        self.retryable = true;
        self.max_retries = max_retries;
        self
    }
    
    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Add a single tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// Metadata about a node implementation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeMetadata {
    /// Node ID
    pub id: NodeId,
    
    /// Node name
    pub name: String,
    
    /// Node description
    pub description: Option<String>,
    
    /// Expected input keys
    pub input_keys: Vec<String>,
    
    /// Output keys
    pub output_keys: Vec<String>,
    
    /// Node type
    pub node_type: String,
    
    /// Node version
    pub version: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Trait for node builders that can create nodes from configuration
pub trait NodeBuilder: Send + Sync {
    /// The type of node this builder creates
    type Node: crate::core::Node;
    
    /// Build a node from configuration
    fn build(&self, config: NodeConfig) -> RGraphResult<Self::Node>;
    
    /// Get the node type this builder creates
    fn node_type(&self) -> &str;
    
    /// Validate the configuration
    fn validate_config(&self, config: &NodeConfig) -> RGraphResult<()> {
        // Default implementation - can be overridden
        if config.name.is_empty() {
            return Err(RGraphError::validation("Node name cannot be empty"));
        }
        Ok(())
    }
}

/// Registry for node builders (placeholder implementation)
pub struct NodeBuilderRegistry {
    _placeholder: bool,
}

impl NodeBuilderRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            _placeholder: true,
        }
    }
    
    /// Register a node builder (placeholder)
    pub fn register<B>(&mut self, _node_type: String, _builder: B)
    where
        B: NodeBuilder + 'static,
        B::Node: crate::core::Node + 'static,
    {
        // This would need proper type erasure in a real implementation
        // For now, this is a placeholder
    }
    
    /// Get available node types
    pub fn node_types(&self) -> Vec<String> {
        vec![]
    }
}

impl Default for NodeBuilderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a simple pass-through node for testing
#[cfg(test)]
pub mod test_utils {
    use super::*;
    use crate::core::{Node, ExecutionContext, ExecutionResult};
    use crate::state::GraphState;
    use async_trait::async_trait;
    use std::sync::Arc;
    
    pub struct PassThroughNode {
        id: NodeId,
        name: String,
        output_key: String,
        output_value: String,
    }
    
    impl PassThroughNode {
        pub fn new(
            id: impl Into<NodeId>,
            name: impl Into<String>,
            output_key: impl Into<String>,
            output_value: impl Into<String>,
        ) -> Arc<Self> {
            Arc::new(Self {
                id: id.into(),
                name: name.into(),
                output_key: output_key.into(),
                output_value: output_value.into(),
            })
        }
    }
    
    #[async_trait]
    impl Node for PassThroughNode {
        async fn execute(
            &self,
            state: &mut GraphState,
            _context: &ExecutionContext,
        ) -> crate::RGraphResult<ExecutionResult> {
            state.set(&self.output_key, &self.output_value);
            Ok(ExecutionResult::Continue)
        }
        
        fn id(&self) -> &NodeId {
            &self.id
        }
        
        fn name(&self) -> &str {
            &self.name
        }
        
        fn output_keys(&self) -> Vec<&str> {
            vec![&self.output_key]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_node_config_creation() {
        let config = NodeConfig::new("test_node", "Test Node")
            .with_description("A test node")
            .with_input_mapping("user_input", "prompt")
            .with_output_mapping("result", "node_output")
            .with_config(json!({"temperature": 0.7}))
            .with_retries(3)
            .with_tag("test");
        
        assert_eq!(config.id.as_str(), "test_node");
        assert_eq!(config.name, "Test Node");
        assert_eq!(config.description, Some("A test node".to_string()));
        assert_eq!(config.input_mappings.get("user_input"), Some(&"prompt".to_string()));
        assert_eq!(config.output_mappings.get("result"), Some(&"node_output".to_string()));
        assert!(config.retryable);
        assert_eq!(config.max_retries, 3);
        assert!(config.tags.contains(&"test".to_string()));
    }
    
    #[test]
    fn test_node_metadata() {
        let metadata = NodeMetadata {
            id: NodeId::new("test_node"),
            name: "Test Node".to_string(),
            description: Some("A test node".to_string()),
            input_keys: vec!["input".to_string()],
            output_keys: vec!["output".to_string()],
            node_type: "test".to_string(),
            version: "1.0.0".to_string(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(metadata.id.as_str(), "test_node");
        assert_eq!(metadata.name, "Test Node");
        assert_eq!(metadata.node_type, "test");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.input_keys.len(), 1);
        assert_eq!(metadata.output_keys.len(), 1);
    }
    
    #[test]
    fn test_node_builder_registry() {
        let mut registry = NodeBuilderRegistry::new();
        assert_eq!(registry.node_types().len(), 0);
        
        // In a real implementation, we'd register actual builders here
        // For now, we just test the basic structure
        assert!(registry.node_types().is_empty());
    }
    
    #[cfg(test)]
    #[tokio::test]
    async fn test_pass_through_node() {
        use crate::state::{GraphState, StateValue};
        use crate::core::ExecutionContext;
        use test_utils::PassThroughNode;
        
        let node = PassThroughNode::new("test", "Test", "output", "test_value");
        let mut state = GraphState::new();
        let context = ExecutionContext::new("graph1".to_string(), NodeId::new("test"));
        
        let result = node.execute(&mut state, &context).await.unwrap();
        
        assert!(matches!(result, ExecutionResult::Continue));
        assert_eq!(state.get("output").unwrap(), StateValue::String("test_value".to_string()));
    }
}