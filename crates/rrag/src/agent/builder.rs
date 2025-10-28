//! Agent builder pattern

use super::{Agent, AgentConfig, ConversationMode, ToolExecutor};
use crate::error::RragResult;
use rsllm::tools::{Tool, ToolRegistry};
use rsllm::Client;

/// Builder for creating agents
pub struct AgentBuilder {
    llm_client: Option<Client>,
    tools: Vec<Box<dyn Tool>>,
    config: AgentConfig,
}

impl AgentBuilder {
    /// Create a new agent builder
    pub fn new() -> Self {
        Self {
            llm_client: None,
            tools: Vec::new(),
            config: AgentConfig::default(),
        }
    }

    /// Set the LLM client
    pub fn with_llm(mut self, client: Client) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Add a single tool
    pub fn with_tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools
    pub fn with_tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = prompt.into();
        self
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set conversation mode
    pub fn with_conversation_mode(mut self, mode: ConversationMode) -> Self {
        self.config.conversation_mode = mode;
        self
    }

    /// Enable stateful mode (maintains conversation)
    pub fn stateful(mut self) -> Self {
        self.config.conversation_mode = ConversationMode::Stateful;
        self
    }

    /// Enable stateless mode (fresh each call)
    pub fn stateless(mut self) -> Self {
        self.config.conversation_mode = ConversationMode::Stateless;
        self
    }

    /// Set max conversation length
    pub fn with_max_conversation_length(mut self, length: usize) -> Self {
        self.config.max_conversation_length = length;
        self
    }

    /// Build the agent
    pub fn build(self) -> RragResult<Agent> {
        let llm_client = self.llm_client.ok_or_else(|| crate::error::RragError::Agent {
            agent_id: "builder".to_string(),
            message: "LLM client is required. Use with_llm()".to_string(),
            source: None,
        })?;

        // Create tool registry
        let mut registry = ToolRegistry::new();
        for tool in self.tools {
            registry.register(tool).map_err(|e| crate::error::RragError::Agent {
                agent_id: "builder".to_string(),
                message: format!("Failed to register tool: {}", e),
                source: Some(Box::new(e)),
            })?;
        }

        let tool_executor = ToolExecutor::new(registry);

        Agent::new(llm_client, tool_executor, self.config)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
