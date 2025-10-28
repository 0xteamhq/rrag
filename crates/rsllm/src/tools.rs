//! # Tool Calling Support
//!
//! Simple, ergonomic tool/function calling for LLMs.
//! Inspired by LangChain but with Rust-native patterns.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rsllm::tools::{Tool, ToolRegistry, ToolCall};
//!
//! // Define a tool
//! struct Calculator;
//!
//! impl Tool for Calculator {
//!     fn name(&self) -> &str { "calculator" }
//!     fn description(&self) -> &str { "Performs basic arithmetic" }
//!
//!     fn parameters_schema(&self) -> serde_json::Value {
//!         json!({
//!             "type": "object",
//!             "properties": {
//!                 "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
//!                 "a": {"type": "number"},
//!                 "b": {"type": "number"}
//!             },
//!             "required": ["operation", "a", "b"]
//!         })
//!     }
//!
//!     fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, Box<dyn Error>> {
//!         // Implementation
//!     }
//! }
//!
//! // Use the tool
//! let mut registry = ToolRegistry::new();
//! registry.register(Box::new(Calculator));
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[cfg(feature = "json-schema")]
use schemars::{schema_for, JsonSchema};

/// A tool that can be called by an LLM
pub trait Tool: Send + Sync {
    /// The name of the tool (must be unique)
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters
    fn parameters_schema(&self) -> JsonValue;

    /// Execute the tool with the given arguments
    fn execute(&self, args: JsonValue) -> Result<JsonValue, Box<dyn Error + Send + Sync>>;

    /// Optional: Validate arguments before execution
    fn validate(&self, _args: &JsonValue) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

/// Schema-based tool with automatic JSON Schema generation
///
/// This trait provides automatic schema generation using `schemars`.
/// Your parameter types just need to derive `JsonSchema`:
///
/// ```rust,ignore
/// #[derive(JsonSchema, Serialize, Deserialize)]
/// struct MyParams {
///     name: String,
///     age: u32,
/// }
///
/// impl SchemaBasedTool for MyTool {
///     type Params = MyParams;
///
///     fn name(&self) -> &str { "my_tool" }
///     fn description(&self) -> &str { "Does something" }
///
///     fn execute_typed(&self, params: Self::Params) -> Result<JsonValue, Box<dyn Error + Send + Sync>> {
///         // Work with strongly-typed params!
///         Ok(json!({"result": params.name}))
///     }
/// }
/// ```
#[cfg(feature = "json-schema")]
pub trait SchemaBasedTool: Send + Sync {
    /// The parameter type (must derive JsonSchema, Serialize, Deserialize)
    type Params: JsonSchema + for<'de> Deserialize<'de>;

    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Execute with strongly-typed parameters
    fn execute_typed(&self, params: Self::Params) -> Result<JsonValue, Box<dyn Error + Send + Sync>>;

    /// Optional: Validate typed parameters before execution
    fn validate_typed(&self, _params: &Self::Params) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

/// Blanket implementation: SchemaBasedTool automatically implements Tool
#[cfg(feature = "json-schema")]
impl<T: SchemaBasedTool> Tool for T {
    fn name(&self) -> &str {
        SchemaBasedTool::name(self)
    }

    fn description(&self) -> &str {
        SchemaBasedTool::description(self)
    }

    fn parameters_schema(&self) -> JsonValue {
        // Automatically generate schema from the Params type!
        // Uses JSON Schema Draft 7 by default (OpenAI compatible)
        let schema = schema_for!(T::Params);
        serde_json::to_value(&schema).unwrap_or_else(|_| JsonValue::Null)
    }

    fn execute(&self, args: JsonValue) -> Result<JsonValue, Box<dyn Error + Send + Sync>> {
        // Deserialize to strongly-typed params
        let params: T::Params = serde_json::from_value(args)
            .map_err(|e| format!("Failed to deserialize parameters: {}", e))?;

        // Validate typed params
        self.validate_typed(&params)?;

        // Execute with typed params
        self.execute_typed(params)
    }

    fn validate(&self, args: &JsonValue) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Deserialize and validate
        let params: T::Params = serde_json::from_value(args.clone())
            .map_err(|e| format!("Invalid parameters: {}", e))?;
        self.validate_typed(&params)
    }
}

/// Tool definition for serialization to LLM API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,

    /// Tool description
    pub description: String,

    /// Parameters schema (JSON Schema)
    pub parameters: JsonValue,

    /// Tool type (default: "function")
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
}

fn default_tool_type() -> String {
    "function".to_string()
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: JsonValue) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            tool_type: "function".to_string(),
        }
    }

    /// Create from a Tool trait object
    pub fn from_tool(tool: &dyn Tool) -> Self {
        Self::new(tool.name(), tool.description(), tool.parameters_schema())
    }
}

/// A tool call request from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,

    /// Tool name to call
    pub name: String,

    /// Arguments as JSON
    pub arguments: JsonValue,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: JsonValue) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// Result of executing a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool call ID this result is for
    pub tool_call_id: String,

    /// Tool name
    pub tool_name: String,

    /// Result content as JSON
    pub content: JsonValue,

    /// Whether the tool execution was successful
    pub success: bool,

    /// Error message if execution failed
    pub error: Option<String>,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(tool_call_id: impl Into<String>, tool_name: impl Into<String>, content: JsonValue) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            success: true,
            error: None,
        }
    }

    /// Create a failed tool result
    pub fn error(tool_call_id: impl Into<String>, tool_name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: JsonValue::Null,
            success: false,
            error: Some(error.into()),
        }
    }
}

/// Registry for managing tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register(&mut self, tool: Box<dyn Tool>) -> Result<(), ToolRegistryError> {
        let name = tool.name().to_string();

        if self.tools.contains_key(&name) {
            return Err(ToolRegistryError::DuplicateTool(name));
        }

        self.tools.insert(name, tool);
        Ok(())
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Check if a tool exists
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get all registered tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get all tool definitions for LLM API
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| ToolDefinition::from_tool(tool.as_ref()))
            .collect()
    }

    /// Execute a tool call
    pub fn execute(&self, tool_call: &ToolCall) -> ToolResult {
        match self.get(&tool_call.name) {
            Some(tool) => {
                // Validate arguments
                if let Err(e) = tool.validate(&tool_call.arguments) {
                    return ToolResult::error(
                        &tool_call.id,
                        &tool_call.name,
                        format!("Validation failed: {}", e),
                    );
                }

                // Execute the tool
                match tool.execute(tool_call.arguments.clone()) {
                    Ok(result) => ToolResult::success(&tool_call.id, &tool_call.name, result),
                    Err(e) => ToolResult::error(&tool_call.id, &tool_call.name, e.to_string()),
                }
            }
            None => ToolResult::error(
                &tool_call.id,
                &tool_call.name,
                format!("Tool '{}' not found", tool_call.name),
            ),
        }
    }

    /// Execute multiple tool calls
    pub fn execute_batch(&self, tool_calls: &[ToolCall]) -> Vec<ToolResult> {
        tool_calls.iter().map(|tc| self.execute(tc)).collect()
    }

    /// Number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tool_names())
            .finish()
    }
}

/// Errors that can occur with tool registry
#[derive(Debug, Clone)]
pub enum ToolRegistryError {
    /// Tool with this name already exists
    DuplicateTool(String),
}

impl fmt::Display for ToolRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolRegistryError::DuplicateTool(name) => {
                write!(f, "Tool '{}' is already registered", name)
            }
        }
    }
}

impl Error for ToolRegistryError {}

// Helper macro for creating simple tools
#[macro_export]
macro_rules! simple_tool {
    (
        name: $name:expr,
        description: $desc:expr,
        parameters: $params:expr,
        execute: |$args:ident| $body:expr
    ) => {{
        struct SimpleTool;
        impl $crate::tools::Tool for SimpleTool {
            fn name(&self) -> &str {
                $name
            }
            fn description(&self) -> &str {
                $desc
            }
            fn parameters_schema(&self) -> serde_json::Value {
                $params
            }
            fn execute(
                &self,
                $args: serde_json::Value,
            ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
                Ok($body)
            }
        }
        Box::new(SimpleTool)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct TestTool;
    impl Tool for TestTool {
        fn name(&self) -> &str {
            "test_tool"
        }
        fn description(&self) -> &str {
            "A test tool"
        }
        fn parameters_schema(&self) -> JsonValue {
            json!({"type": "object", "properties": {"input": {"type": "string"}}})
        }
        fn execute(&self, args: JsonValue) -> Result<JsonValue, Box<dyn Error + Send + Sync>> {
            Ok(json!({"result": format!("Processed: {}", args["input"])}))
        }
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);

        registry.register(Box::new(TestTool)).unwrap();
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test_tool"));
    }

    #[test]
    fn test_tool_execution() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(TestTool)).unwrap();

        let call = ToolCall::new("call-1", "test_tool", json!({"input": "hello"}));
        let result = registry.execute(&call);

        assert!(result.success);
        assert_eq!(result.tool_name, "test_tool");
    }

    #[test]
    fn test_simple_tool_macro() {
        let tool = simple_tool!(
            name: "echo",
            description: "Echoes input",
            parameters: json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            execute: |args| {
                json!({"echo": args["text"]})
            }
        );

        assert_eq!(tool.name(), "echo");
        let result = tool.execute(json!({"text": "hello"})).unwrap();
        assert_eq!(result["echo"], "hello");
    }
}
