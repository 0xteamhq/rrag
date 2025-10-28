//! Tool execution for agents

use rsllm::tools::{ToolCall as ToolExec, ToolRegistry, ToolResult};
use rsllm::{ChatMessage, ToolCall};

/// Handles tool execution for the agent
pub struct ToolExecutor {
    registry: ToolRegistry,
}

impl ToolExecutor {
    /// Create a new tool executor
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    /// Execute a tool call and return the result message
    pub fn execute_tool_call(&self, tool_call: &ToolCall) -> ChatMessage {
        // Convert to ToolExec format
        let tool_exec = ToolExec::new(
            &tool_call.id,
            &tool_call.function.name,
            tool_call.function.arguments.clone(),
        );

        // Execute the tool
        let result = self.registry.execute(&tool_exec);

        // Convert result to chat message
        let result_content = if result.success {
            serde_json::to_string(&result.content).unwrap_or_else(|_| "{}".to_string())
        } else {
            format!("Error: {}", result.error.unwrap_or_default())
        };

        ChatMessage::tool(&tool_call.id, result_content)
    }

    /// Execute multiple tool calls
    pub fn execute_tool_calls(&self, tool_calls: &[ToolCall]) -> Vec<ChatMessage> {
        tool_calls
            .iter()
            .map(|call| self.execute_tool_call(call))
            .collect()
    }

    /// Get the tool registry
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }
}
