//! Core Agent implementation

use super::{AgentConfig, ConversationMemory, ConversationMode, ToolExecutor};
use crate::error::RragResult;
use rsllm::{ChatMessage, ChatResponse, Client};
use tracing::{debug, error, info};

/// Agent that can use tools and maintain conversation
pub struct Agent {
    /// LLM client
    llm_client: Client,

    /// Tool executor
    tool_executor: ToolExecutor,

    /// Conversation memory (for stateful mode)
    memory: ConversationMemory,

    /// Agent configuration
    config: AgentConfig,
}

impl Agent {
    /// Create a new agent
    pub fn new(
        llm_client: Client,
        tool_executor: ToolExecutor,
        config: AgentConfig,
    ) -> RragResult<Self> {
        let mut memory = ConversationMemory::with_max_length(config.max_conversation_length);

        // Add system prompt
        memory.add_message(ChatMessage::system(config.system_prompt.clone()));

        Ok(Self {
            llm_client,
            tool_executor,
            memory,
            config,
        })
    }

    /// Run the agent with a user query
    ///
    /// In stateless mode: Creates fresh conversation for each call
    /// In stateful mode: Continues previous conversation
    pub async fn run(&mut self, user_input: impl Into<String>) -> RragResult<String> {
        let input = user_input.into();

        info!(user_input = %input, "Agent received user input");

        if self.config.verbose {
            debug!(input = %input, "Processing user query");
        }

        // Prepare conversation based on mode
        let mut conversation = match self.config.conversation_mode {
            ConversationMode::Stateless => {
                // Fresh conversation: system prompt + user message
                vec![
                    ChatMessage::system(self.config.system_prompt.clone()),
                    ChatMessage::user(input.clone()),
                ]
            }
            ConversationMode::Stateful => {
                // Continue existing conversation
                self.memory.add_message(ChatMessage::user(input.clone()));
                self.memory.to_messages()
            }
        };

        // Agent loop: iterate until we get a final answer
        for iteration in 1..=self.config.max_iterations {
            debug!(
                iteration,
                max_iterations = self.config.max_iterations,
                "Agent iteration"
            );

            // Call LLM with tools
            let response = self.llm_step(&conversation).await?;

            // Check for tool calls
            if let Some(tool_calls) = &response.tool_calls {
                if !tool_calls.is_empty() {
                    info!(
                        tool_count = tool_calls.len(),
                        tools = ?tool_calls.iter().map(|t| &t.function.name).collect::<Vec<_>>(),
                        "Agent requesting tool calls"
                    );

                    // Add assistant message with tool calls to conversation
                    let mut assistant_msg = ChatMessage::assistant(response.content.clone());
                    assistant_msg.tool_calls = Some(tool_calls.clone());
                    conversation.push(assistant_msg);

                    // Execute all tool calls
                    let tool_results = self.tool_executor.execute_tool_calls(tool_calls);

                    // Add tool results to conversation
                    for result in tool_results {
                        if let rsllm::MessageContent::Text(ref content) = result.content {
                            debug!(tool_result = %content, "Tool execution completed");
                        }
                        conversation.push(result);
                    }

                    // Continue loop to let LLM process results
                    continue;
                }
            }

            // No tool calls - this is the final answer
            info!(
                response = %response.content,
                iterations = iteration,
                "Agent generated final answer"
            );

            // Update memory in stateful mode
            if self.config.conversation_mode == ConversationMode::Stateful {
                self.memory
                    .add_message(ChatMessage::assistant(response.content.clone()));
            }

            return Ok(response.content);
        }

        // Exceeded max iterations
        error!(
            max_iterations = self.config.max_iterations,
            "Agent exceeded maximum iterations without reaching final answer"
        );

        Err(crate::error::RragError::Agent {
            agent_id: "default".to_string(),
            message: format!(
                "Agent exceeded maximum iterations ({})",
                self.config.max_iterations
            ),
            source: None,
        })
    }

    /// Single LLM call with tools
    async fn llm_step(&self, conversation: &[ChatMessage]) -> RragResult<ChatResponse> {
        // Get tool definitions
        let tools = self.tool_executor.registry().tool_definitions();

        debug!(
            tool_count = tools.len(),
            message_count = conversation.len(),
            "Calling LLM with tools"
        );

        // Call LLM
        let response = self
            .llm_client
            .chat_completion_with_tools(conversation.to_vec(), tools)
            .await?;

        debug!(
            content_length = response.content.len(),
            has_tool_calls = response.tool_calls.is_some(),
            tool_call_count = response.tool_calls.as_ref().map(|t| t.len()).unwrap_or(0),
            "LLM response received"
        );

        Ok(response)
    }

    /// Reset conversation (clears history, keeps system prompt)
    pub fn reset(&mut self) {
        self.memory.clear();
    }

    /// Get conversation history
    pub fn get_conversation(&self) -> &[ChatMessage] {
        self.memory.get_messages()
    }

    /// Get agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut AgentConfig {
        &mut self.config
    }
}
