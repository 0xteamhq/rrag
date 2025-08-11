//! # Tool Node Implementation
//! 
//! Tool nodes directly execute tools without agent reasoning.

use crate::core::{Node, NodeId, ExecutionContext, ExecutionResult};
use crate::state::{GraphState, StateValue};
use crate::tools::Tool;
use crate::{RGraphError, RGraphResult};
use async_trait::async_trait;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for tool nodes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ToolNodeConfig {
    pub tool_name: String,
    pub argument_mappings: std::collections::HashMap<String, String>,
    pub output_key: String,
}

/// A node that executes a specific tool
pub struct ToolNode {
    id: NodeId,
    name: String,
    tool: Arc<dyn Tool>,
    config: ToolNodeConfig,
}

impl ToolNode {
    pub fn new(
        id: impl Into<NodeId>,
        name: impl Into<String>,
        tool: Arc<dyn Tool>,
        config: ToolNodeConfig,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            tool,
            config,
        }
    }
}

#[async_trait]
impl Node for ToolNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Build arguments from state using mappings
        let mut arguments = serde_json::Map::new();
        
        for (state_key, arg_key) in &self.config.argument_mappings {
            if let Ok(value) = state.get(state_key) {
                let json_value: serde_json::Value = value.into();
                arguments.insert(arg_key.clone(), json_value);
            }
        }
        
        let arguments_json = serde_json::Value::Object(arguments);
        
        // Execute the tool
        match self.tool.execute(&arguments_json, state).await {
            Ok(result) => {
                // Store result in state
                state.set_with_context(
                    context.current_node.as_str(),
                    &self.config.output_key,
                    StateValue::from(result.output),
                );
                Ok(ExecutionResult::Continue)
            }
            Err(e) => Err(RGraphError::tool(e.to_string())),
        }
    }
    
    fn id(&self) -> &NodeId {
        &self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn input_keys(&self) -> Vec<&str> {
        self.config.argument_mappings.keys().map(|s| s.as_str()).collect()
    }
    
    fn output_keys(&self) -> Vec<&str> {
        vec![&self.config.output_key]
    }
}