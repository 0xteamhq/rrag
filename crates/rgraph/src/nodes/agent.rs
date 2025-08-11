//! # Agent Node Implementation
//!
//! Agent nodes represent autonomous AI agents that can reason, make decisions,
//! and use tools to accomplish tasks.

use crate::core::{ExecutionContext, ExecutionResult, Node, NodeId};
use crate::state::{GraphState, StateValue};
use crate::tools::Tool;
use crate::{RGraphError, RGraphResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for an agent node
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgentNodeConfig {
    /// Agent name
    pub name: String,

    /// System prompt for the agent
    pub system_prompt: String,

    /// Available tools
    pub tools: Vec<String>,

    /// Maximum number of reasoning steps
    pub max_steps: usize,

    /// Temperature for generation
    pub temperature: f32,

    /// Maximum tokens for generation
    pub max_tokens: Option<usize>,

    /// Whether to use structured output
    pub structured_output: bool,

    /// Custom instructions
    pub instructions: Vec<String>,
}

impl Default for AgentNodeConfig {
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            system_prompt: "You are a helpful AI assistant.".to_string(),
            tools: Vec::new(),
            max_steps: 10,
            temperature: 0.7,
            max_tokens: Some(1000),
            structured_output: false,
            instructions: Vec::new(),
        }
    }
}

/// An agent node that can reason and use tools
pub struct AgentNode {
    id: NodeId,
    config: AgentNodeConfig,
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl AgentNode {
    /// Create a new agent node
    pub fn new(id: impl Into<NodeId>, config: AgentNodeConfig) -> Self {
        Self {
            id: id.into(),
            config,
            tools: HashMap::new(),
        }
    }

    /// Add a tool to the agent
    pub fn with_tool(mut self, name: String, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(name, tool);
        self
    }

    /// Set the system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = prompt.into();
        self
    }

    /// Add tools by name
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.config.tools = tools;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature.clamp(0.0, 2.0);
        self
    }

    /// Execute the agent's reasoning loop
    async fn reasoning_loop(
        &self,
        state: &mut GraphState,
        _context: &ExecutionContext,
        initial_input: &str,
    ) -> RGraphResult<String> {
        let mut conversation_history = Vec::new();
        let mut step_count = 0;

        // Add system prompt
        conversation_history.push(AgentMessage {
            role: MessageRole::System,
            content: self.config.system_prompt.clone(),
            tool_calls: None,
        });

        // Add user input
        conversation_history.push(AgentMessage {
            role: MessageRole::User,
            content: initial_input.to_string(),
            tool_calls: None,
        });

        loop {
            if step_count >= self.config.max_steps {
                break;
            }

            step_count += 1;

            // Generate agent response
            let agent_response = self.generate_response(&conversation_history, state).await?;

            // Check if agent wants to use tools
            if let Some(tool_calls) = &agent_response.tool_calls {
                // Execute tool calls
                let mut tool_results = Vec::new();

                for tool_call in tool_calls {
                    if let Some(tool) = self.tools.get(&tool_call.name) {
                        match tool.execute(&tool_call.arguments, state).await {
                            Ok(result) => {
                                tool_results.push(ToolCallResult {
                                    call_id: tool_call.id.clone(),
                                    name: tool_call.name.clone(),
                                    result: result.output,
                                    success: true,
                                    error: None,
                                });
                            }
                            Err(e) => {
                                tool_results.push(ToolCallResult {
                                    call_id: tool_call.id.clone(),
                                    name: tool_call.name.clone(),
                                    result: serde_json::Value::Null,
                                    success: false,
                                    error: Some(e.to_string()),
                                });
                            }
                        }
                    } else {
                        tool_results.push(ToolCallResult {
                            call_id: tool_call.id.clone(),
                            name: tool_call.name.clone(),
                            result: serde_json::Value::Null,
                            success: false,
                            error: Some(format!("Tool '{}' not found", tool_call.name)),
                        });
                    }
                }

                // Add assistant message with tool calls
                conversation_history.push(agent_response);

                // Add tool results
                for tool_result in tool_results {
                    conversation_history.push(AgentMessage {
                        role: MessageRole::Tool,
                        content: if tool_result.success {
                            serde_json::to_string_pretty(&tool_result.result)
                                .unwrap_or_else(|_| "Tool execution completed".to_string())
                        } else {
                            format!(
                                "Error: {}",
                                tool_result
                                    .error
                                    .unwrap_or_else(|| "Unknown error".to_string())
                            )
                        },
                        tool_calls: None,
                    });
                }
            } else {
                // Agent provided final response
                conversation_history.push(agent_response.clone());
                return Ok(agent_response.content);
            }
        }

        // If we exit the loop without a final response, return the last agent message
        conversation_history
            .iter()
            .filter(|msg| msg.role == MessageRole::Assistant)
            .last()
            .map(|msg| msg.content.clone())
            .unwrap_or_else(|| "Maximum reasoning steps reached without conclusion".to_string())
            .pipe(Ok)
    }

    /// Generate a response from the agent
    async fn generate_response(
        &self,
        conversation: &[AgentMessage],
        _state: &GraphState,
    ) -> RGraphResult<AgentMessage> {
        // In a real implementation, this would call an LLM API
        // For now, we'll simulate an agent response

        let empty_string = String::new();
        let last_user_message = conversation
            .iter()
            .filter(|msg| msg.role == MessageRole::User)
            .last()
            .map(|msg| &msg.content)
            .unwrap_or(&empty_string);

        // Check if we should use tools based on the input
        if self.should_use_tools(last_user_message) && !self.tools.is_empty() {
            // Simulate tool usage decision
            let tool_name = self.tools.keys().next().unwrap().clone();

            Ok(AgentMessage {
                role: MessageRole::Assistant,
                content: format!(
                    "I'll help you with that. Let me use the {} tool.",
                    tool_name
                ),
                tool_calls: Some(vec![ToolCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: tool_name,
                    arguments: serde_json::json!({
                        "query": last_user_message
                    }),
                }]),
            })
        } else {
            // Generate a direct response
            Ok(AgentMessage {
                role: MessageRole::Assistant,
                content: format!(
                    "Based on your request '{}', I can provide assistance. This is a simulated response from the {} agent.",
                    last_user_message,
                    self.config.name
                ),
                tool_calls: None,
            })
        }
    }

    /// Determine if the agent should use tools for this input
    fn should_use_tools(&self, input: &str) -> bool {
        // Simple heuristic - in a real implementation this would be more sophisticated
        let tool_keywords = ["search", "calculate", "analyze", "find", "lookup", "query"];
        let input_lower = input.to_lowercase();

        tool_keywords
            .iter()
            .any(|keyword| input_lower.contains(keyword))
    }
}

#[async_trait]
impl Node for AgentNode {
    async fn execute(
        &self,
        state: &mut GraphState,
        context: &ExecutionContext,
    ) -> RGraphResult<ExecutionResult> {
        // Get input from state
        let input = state
            .get("user_input")
            .or_else(|_| state.get("query"))
            .or_else(|_| state.get("prompt"))
            .map_err(|_| {
                RGraphError::node(
                    self.id.as_str(),
                    "No input found in state (expected 'user_input', 'query', or 'prompt')",
                )
            })?;

        let input_text = match input {
            StateValue::String(s) => s,
            _ => {
                return Err(RGraphError::node(
                    self.id.as_str(),
                    "Input must be a string",
                ))
            }
        };

        // Execute reasoning loop
        let response = self.reasoning_loop(state, context, &input_text).await?;

        // Store the response in state
        state.set_with_context(
            context.current_node.as_str(),
            "agent_response",
            response.clone(),
        );

        // Also store in a generic output key
        state.set_with_context(context.current_node.as_str(), "output", response);

        Ok(ExecutionResult::Continue)
    }

    fn id(&self) -> &NodeId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn input_keys(&self) -> Vec<&str> {
        vec!["user_input", "query", "prompt"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec!["agent_response", "output"]
    }

    fn validate(&self, state: &GraphState) -> RGraphResult<()> {
        // Check that we have input
        if !state.contains_key("user_input")
            && !state.contains_key("query")
            && !state.contains_key("prompt")
        {
            return Err(RGraphError::validation(
                "Agent node requires 'user_input', 'query', or 'prompt' in state",
            ));
        }

        Ok(())
    }
}

/// Message in agent conversation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgentMessage {
    pub role: MessageRole,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Role of message in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Tool call from agent
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of a tool call
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ToolCallResult {
    pub call_id: String,
    pub name: String,
    pub result: serde_json::Value,
    pub success: bool,
    pub error: Option<String>,
}

// Helper trait for pipe operations
trait Pipe<T> {
    fn pipe<U, F>(self, f: F) -> U
    where
        F: FnOnce(T) -> U;
}

impl<T> Pipe<T> for T {
    fn pipe<U, F>(self, f: F) -> U
    where
        F: FnOnce(T) -> U,
    {
        f(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ExecutionContext;
    use crate::tools::{Tool, ToolError, ToolResult};

    // Mock tool for testing
    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        async fn execute(
            &self,
            _arguments: &serde_json::Value,
            _state: &GraphState,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                output: serde_json::json!({
                    "tool": self.name,
                    "result": "mock result"
                }),
                metadata: HashMap::new(),
            })
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Mock tool for testing"
        }
    }

    #[tokio::test]
    async fn test_agent_node_creation() {
        let config = AgentNodeConfig::default();
        let agent = AgentNode::new("test_agent", config);

        assert_eq!(agent.id().as_str(), "test_agent");
        assert_eq!(agent.name(), "assistant");
    }

    #[tokio::test]
    async fn test_agent_node_with_tools() {
        let config = AgentNodeConfig::default();
        let tool = Arc::new(MockTool {
            name: "search".to_string(),
        });

        let agent = AgentNode::new("test_agent", config).with_tool("search".to_string(), tool);

        assert!(agent.tools.contains_key("search"));
    }

    #[tokio::test]
    async fn test_agent_execution() {
        let config = AgentNodeConfig::default();
        let agent = AgentNode::new("test_agent", config);

        let mut state = GraphState::new();
        state.set("user_input", "Hello, how can you help me?");

        let context = ExecutionContext::new("test_graph".to_string(), NodeId::new("test_agent"));
        let result = agent.execute(&mut state, &context).await.unwrap();

        assert!(matches!(result, ExecutionResult::Continue));
        assert!(state.contains_key("agent_response"));
    }

    #[test]
    fn test_should_use_tools() {
        let config = AgentNodeConfig::default();
        let agent = AgentNode::new("test_agent", config);

        assert!(agent.should_use_tools("Please search for information"));
        assert!(agent.should_use_tools("Can you calculate this?"));
        assert!(!agent.should_use_tools("Hello there"));
    }

    #[test]
    fn test_agent_message() {
        let message = AgentMessage {
            role: MessageRole::User,
            content: "Test message".to_string(),
            tool_calls: None,
        };

        assert_eq!(message.role, MessageRole::User);
        assert_eq!(message.content, "Test message");
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn test_tool_call() {
        let tool_call = ToolCall {
            id: "test-123".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };

        assert_eq!(tool_call.id, "test-123");
        assert_eq!(tool_call.name, "search");
        assert_eq!(tool_call.arguments["query"], "test");
    }
}
