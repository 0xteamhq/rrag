//! # Condition Node Implementation
//! 
//! Condition nodes make routing decisions based on state.

use crate::core::{Node, NodeId, ExecutionContext, ExecutionResult};
use crate::state::GraphState;
use crate::RGraphResult;
use async_trait::async_trait;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for condition nodes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConditionNodeConfig {
    pub condition_key: String,
    pub condition_value: serde_json::Value,
    pub true_route: String,
    pub false_route: String,
}

/// A node that routes based on conditions
pub struct ConditionNode {
    id: NodeId,
    name: String,
    config: ConditionNodeConfig,
}

impl ConditionNode {
    pub fn new(
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: ConditionNodeConfig,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            config,
        }
    }
}

#[async_trait]
impl Node for ConditionNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        _context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Check condition
        let state_value = state.get(&self.config.condition_key)?;
        let state_json: serde_json::Value = state_value.into();
        
        let route = if state_json == self.config.condition_value {
            &self.config.true_route
        } else {
            &self.config.false_route
        };
        
        Ok(ExecutionResult::Route(route.clone()))
    }
    
    fn id(&self) -> &NodeId {
        &self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.config.condition_key]
    }
    
    fn output_keys(&self) -> Vec<&str> {
        vec![]
    }
}