//! # Transform Node Implementation
//! 
//! Transform nodes modify and process data in the state.

use crate::core::{Node, NodeId, ExecutionContext, ExecutionResult};
use crate::state::{GraphState, StateValue};
use crate::{RGraphError, RGraphResult};
use async_trait::async_trait;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for transform nodes
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransformNodeConfig {
    pub input_key: String,
    pub output_key: String,
    pub transform_type: TransformType,
}

/// Types of transformations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TransformType {
    /// Convert to uppercase
    ToUpperCase,
    /// Convert to lowercase
    ToLowerCase,
    /// Extract substring
    Substring { start: usize, length: Option<usize> },
    /// Replace text
    Replace { from: String, to: String },
    /// JSON parse
    JsonParse,
    /// JSON stringify
    JsonStringify,
}

/// A node that transforms data
pub struct TransformNode {
    id: NodeId,
    name: String,
    config: TransformNodeConfig,
}

impl TransformNode {
    pub fn new(
        id: impl Into<NodeId>,
        name: impl Into<String>,
        config: TransformNodeConfig,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            config,
        }
    }
    
    fn apply_transform(&self, input: &StateValue) -> RGraphResult<StateValue> {
        match &self.config.transform_type {
            TransformType::ToUpperCase => {
                if let Some(s) = input.as_string() {
                    Ok(StateValue::String(s.to_uppercase()))
                } else {
                    Err(RGraphError::node(
                        self.id.as_str(),
                        "ToUpperCase requires string input"
                    ))
                }
            }
            TransformType::ToLowerCase => {
                if let Some(s) = input.as_string() {
                    Ok(StateValue::String(s.to_lowercase()))
                } else {
                    Err(RGraphError::node(
                        self.id.as_str(),
                        "ToLowerCase requires string input"
                    ))
                }
            }
            TransformType::Substring { start, length } => {
                if let Some(s) = input.as_string() {
                    let end = length.map(|l| start + l).unwrap_or(s.len());
                    let substring = s.chars().skip(*start).take(end - start).collect::<String>();
                    Ok(StateValue::String(substring))
                } else {
                    Err(RGraphError::node(
                        self.id.as_str(),
                        "Substring requires string input"
                    ))
                }
            }
            TransformType::Replace { from, to } => {
                if let Some(s) = input.as_string() {
                    Ok(StateValue::String(s.replace(from, to)))
                } else {
                    Err(RGraphError::node(
                        self.id.as_str(),
                        "Replace requires string input"
                    ))
                }
            }
            TransformType::JsonParse => {
                if let Some(s) = input.as_string() {
                    match serde_json::from_str::<serde_json::Value>(s) {
                        Ok(json) => Ok(StateValue::from(json)),
                        Err(e) => Err(RGraphError::node(
                            self.id.as_str(),
                            format!("JSON parse error: {}", e)
                        ))
                    }
                } else {
                    Err(RGraphError::node(
                        self.id.as_str(),
                        "JsonParse requires string input"
                    ))
                }
            }
            TransformType::JsonStringify => {
                let json_value: serde_json::Value = input.clone().into();
                match serde_json::to_string(&json_value) {
                    Ok(json_str) => Ok(StateValue::String(json_str)),
                    Err(e) => Err(RGraphError::node(
                        self.id.as_str(),
                        format!("JSON stringify error: {}", e)
                    ))
                }
            }
        }
    }
}

#[async_trait]
impl Node for TransformNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Get input value
        let input_value = state.get(&self.config.input_key)?;
        
        // Apply transformation
        let output_value = self.apply_transform(&input_value)?;
        
        // Store output
        state.set_with_context(
            context.current_node.as_str(),
            &self.config.output_key,
            output_value,
        );
        
        Ok(ExecutionResult::Continue)
    }
    
    fn id(&self) -> &NodeId {
        &self.id
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn input_keys(&self) -> Vec<&str> {
        vec![&self.config.input_key]
    }
    
    fn output_keys(&self) -> Vec<&str> {
        vec![&self.config.output_key]
    }
}