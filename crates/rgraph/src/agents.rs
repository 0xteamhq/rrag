//! # Agent System
//! 
//! High-level agent abstractions and configurations.

use crate::tools::Tool;
use std::sync::Arc;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Agent configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgentConfig {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub tools: Vec<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            description: "A helpful AI assistant".to_string(),
            system_prompt: "You are a helpful AI assistant.".to_string(),
            temperature: 0.7,
            max_tokens: Some(1000),
            tools: Vec::new(),
        }
    }
}

/// Agent builder for fluent configuration
pub struct AgentBuilder {
    config: AgentConfig,
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl AgentBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: AgentConfig {
                name: name.into(),
                ..Default::default()
            },
            tools: HashMap::new(),
        }
    }
    
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.config.description = description.into();
        self
    }
    
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = prompt.into();
        self
    }
    
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature.clamp(0.0, 2.0);
        self
    }
    
    pub fn with_tool(mut self, name: String, tool: Arc<dyn Tool>) -> Self {
        self.config.tools.push(name.clone());
        self.tools.insert(name, tool);
        self
    }
}

/// High-level agent trait
pub trait Agent: Send + Sync {
    /// Get agent name
    fn name(&self) -> &str;
    
    /// Get agent description
    fn description(&self) -> &str;
    
    /// Get agent configuration
    fn config(&self) -> &AgentConfig;
}