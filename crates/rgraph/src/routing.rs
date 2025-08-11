//! # Routing System
//! 
//! Conditional routing and decision-making for graph execution.

use crate::state::GraphState;
use crate::core::NodeId;
use crate::RGraphResult;
use async_trait::async_trait;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Routing decision result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RoutingDecision {
    /// Route to specific node
    Route(NodeId),
    /// Stop execution
    Stop,
    /// Continue with default routing
    Continue,
}

/// Trait for routing conditions
#[async_trait]
pub trait RoutingCondition: Send + Sync {
    /// Evaluate the condition and return routing decision
    async fn evaluate(&self, state: &GraphState) -> RGraphResult<RoutingDecision>;
}

/// Simple state-based routing condition
pub struct StateCondition {
    key: String,
    expected_value: serde_json::Value,
    target_node: NodeId,
}

impl StateCondition {
    pub fn new(
        key: impl Into<String>,
        expected_value: serde_json::Value,
        target_node: impl Into<NodeId>,
    ) -> Self {
        Self {
            key: key.into(),
            expected_value,
            target_node: target_node.into(),
        }
    }
}

#[async_trait]
impl RoutingCondition for StateCondition {
    async fn evaluate(&self, state: &GraphState) -> RGraphResult<RoutingDecision> {
        match state.get(&self.key) {
            Ok(value) => {
                let state_json: serde_json::Value = value.into();
                if state_json == self.expected_value {
                    Ok(RoutingDecision::Route(self.target_node.clone()))
                } else {
                    Ok(RoutingDecision::Continue)
                }
            }
            Err(_) => Ok(RoutingDecision::Continue),
        }
    }
}

/// Conditional edge with routing logic
pub struct ConditionalEdge {
    condition: Box<dyn RoutingCondition>,
    source: NodeId,
}

impl ConditionalEdge {
    pub fn new(
        source: impl Into<NodeId>,
        condition: Box<dyn RoutingCondition>,
    ) -> Self {
        Self {
            condition,
            source: source.into(),
        }
    }
    
    pub async fn evaluate(&self, state: &GraphState) -> RGraphResult<RoutingDecision> {
        self.condition.evaluate(state).await
    }
}

/// Router for managing conditional routing
pub struct Router {
    conditions: Vec<ConditionalEdge>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }
    
    pub fn add_condition(&mut self, condition: ConditionalEdge) {
        self.conditions.push(condition);
    }
    
    pub async fn route(&self, current_node: &NodeId, state: &GraphState) -> RGraphResult<RoutingDecision> {
        for condition in &self.conditions {
            if &condition.source == current_node {
                let decision = condition.evaluate(state).await?;
                match decision {
                    RoutingDecision::Continue => continue,
                    _ => return Ok(decision),
                }
            }
        }
        
        Ok(RoutingDecision::Continue)
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}