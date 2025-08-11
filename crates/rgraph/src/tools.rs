//! # Tool System for RGraph Agents
//!
//! This module provides the tool system that allows agents to interact with
//! external systems, perform computations, and access data.

use crate::state::GraphState;
// Future use for tool implementations
use async_trait::async_trait;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result of tool execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ToolResult {
    /// Tool output
    pub output: serde_json::Value,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Error that can occur during tool execution
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool execution error: {message}")]
    Execution { message: String },

    #[error("Invalid arguments: {message}")]
    InvalidArguments { message: String },

    #[error("Tool not found: {name}")]
    NotFound { name: String },

    #[error("Permission denied for tool: {name}")]
    PermissionDenied { name: String },

    #[error("Tool timeout: {name}")]
    Timeout { name: String },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Configuration for a tool
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ToolConfig {
    /// Tool name
    pub name: String,

    /// Tool description
    pub description: String,

    /// Tool version
    pub version: String,

    /// Whether the tool requires authentication
    pub requires_auth: bool,

    /// Maximum execution time in milliseconds
    pub timeout_ms: Option<u64>,

    /// Tool-specific configuration
    pub config: serde_json::Value,
}

/// Core trait for all tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Execute the tool with given arguments
    async fn execute(
        &self,
        arguments: &serde_json::Value,
        state: &GraphState,
    ) -> Result<ToolResult, ToolError>;

    /// Get the tool name
    fn name(&self) -> &str;

    /// Get the tool description
    fn description(&self) -> &str;

    /// Get the tool's argument schema (JSON Schema)
    fn argument_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true
        })
    }

    /// Validate tool arguments
    fn validate_arguments(&self, _arguments: &serde_json::Value) -> Result<(), ToolError> {
        Ok(())
    }

    /// Check if the tool requires authentication
    fn requires_auth(&self) -> bool {
        false
    }

    /// Get tool metadata
    fn metadata(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }
}

/// Simple echo tool for testing
pub struct EchoTool {
    name: String,
}

impl EchoTool {
    pub fn new() -> Self {
        Self {
            name: "echo".to_string(),
        }
    }
}

impl Default for EchoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for EchoTool {
    async fn execute(
        &self,
        arguments: &serde_json::Value,
        _state: &GraphState,
    ) -> Result<ToolResult, ToolError> {
        let message = arguments
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello from EchoTool!");

        Ok(ToolResult {
            output: serde_json::json!({
                "echo": message,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "A simple tool that echoes back the input message"
    }

    fn argument_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back"
                }
            },
            "required": ["message"]
        })
    }
}

/// Calculator tool for basic arithmetic
pub struct CalculatorTool {
    name: String,
}

impl CalculatorTool {
    pub fn new() -> Self {
        Self {
            name: "calculator".to_string(),
        }
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    async fn execute(
        &self,
        arguments: &serde_json::Value,
        _state: &GraphState,
    ) -> Result<ToolResult, ToolError> {
        let operation = arguments
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArguments {
                message: "Missing 'operation' field".to_string(),
            })?;

        let a = arguments.get("a").and_then(|v| v.as_f64()).ok_or_else(|| {
            ToolError::InvalidArguments {
                message: "Missing or invalid 'a' field".to_string(),
            }
        })?;

        let b = arguments.get("b").and_then(|v| v.as_f64()).ok_or_else(|| {
            ToolError::InvalidArguments {
                message: "Missing or invalid 'b' field".to_string(),
            }
        })?;

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err(ToolError::Execution {
                        message: "Division by zero".to_string(),
                    });
                }
                a / b
            }
            _ => {
                return Err(ToolError::InvalidArguments {
                    message: format!("Unknown operation: {}", operation),
                })
            }
        };

        Ok(ToolResult {
            output: serde_json::json!({
                "operation": operation,
                "operands": [a, b],
                "result": result
            }),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "A calculator tool for basic arithmetic operations"
    }

    fn argument_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            "required": ["operation", "a", "b"]
        })
    }

    fn validate_arguments(&self, arguments: &serde_json::Value) -> Result<(), ToolError> {
        if !arguments.is_object() {
            return Err(ToolError::InvalidArguments {
                message: "Arguments must be an object".to_string(),
            });
        }

        // Check required fields
        let required_fields = ["operation", "a", "b"];
        for field in &required_fields {
            if !arguments.get(field).is_some() {
                return Err(ToolError::InvalidArguments {
                    message: format!("Missing required field: {}", field),
                });
            }
        }

        // Validate operation
        if let Some(op) = arguments.get("operation").and_then(|v| v.as_str()) {
            if !["add", "subtract", "multiply", "divide"].contains(&op) {
                return Err(ToolError::InvalidArguments {
                    message: format!("Invalid operation: {}", op),
                });
            }
        }

        Ok(())
    }
}

/// Tool registry for managing available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Get all available tool names
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Execute a tool
    pub async fn execute(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
        state: &GraphState,
    ) -> Result<ToolResult, ToolError> {
        let tool = self.get(tool_name).ok_or_else(|| ToolError::NotFound {
            name: tool_name.to_string(),
        })?;

        // Validate arguments
        tool.validate_arguments(arguments)?;

        // Execute tool
        tool.execute(arguments, state).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        // Register default tools
        registry.register(Box::new(EchoTool::new()));
        registry.register(Box::new(CalculatorTool::new()));

        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_tool() {
        let tool = EchoTool::new();
        let state = GraphState::new();
        let arguments = serde_json::json!({
            "message": "Hello, World!"
        });

        let result = tool.execute(&arguments, &state).await.unwrap();

        assert_eq!(result.output["echo"], "Hello, World!");
        assert!(result.output.get("timestamp").is_some());
    }

    #[tokio::test]
    async fn test_calculator_tool() {
        let tool = CalculatorTool::new();
        let state = GraphState::new();

        // Test addition
        let arguments = serde_json::json!({
            "operation": "add",
            "a": 5.0,
            "b": 3.0
        });

        let result = tool.execute(&arguments, &state).await.unwrap();
        assert_eq!(result.output["result"], 8.0);

        // Test division by zero
        let arguments = serde_json::json!({
            "operation": "divide",
            "a": 5.0,
            "b": 0.0
        });

        let result = tool.execute(&arguments, &state).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool::new()));

        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());

        let tool_names = registry.tool_names();
        assert!(tool_names.contains(&"echo".to_string()));

        // Test execution through registry
        let arguments = serde_json::json!({
            "message": "Test"
        });
        let state = GraphState::new();

        let result = registry.execute("echo", &arguments, &state).await.unwrap();
        assert_eq!(result.output["echo"], "Test");
    }

    #[test]
    fn test_calculator_validation() {
        let tool = CalculatorTool::new();

        // Valid arguments
        let valid_args = serde_json::json!({
            "operation": "add",
            "a": 1.0,
            "b": 2.0
        });
        assert!(tool.validate_arguments(&valid_args).is_ok());

        // Invalid operation
        let invalid_args = serde_json::json!({
            "operation": "invalid",
            "a": 1.0,
            "b": 2.0
        });
        assert!(tool.validate_arguments(&invalid_args).is_err());

        // Missing field
        let missing_field = serde_json::json!({
            "operation": "add",
            "a": 1.0
        });
        assert!(tool.validate_arguments(&missing_field).is_err());
    }

    #[test]
    fn test_tool_schemas() {
        let echo_tool = EchoTool::new();
        let calc_tool = CalculatorTool::new();

        let echo_schema = echo_tool.argument_schema();
        assert_eq!(echo_schema["type"], "object");
        assert!(echo_schema["properties"].get("message").is_some());

        let calc_schema = calc_tool.argument_schema();
        assert_eq!(calc_schema["type"], "object");
        assert!(calc_schema["properties"].get("operation").is_some());
        assert!(calc_schema["properties"].get("a").is_some());
        assert!(calc_schema["properties"].get("b").is_some());
    }
}
