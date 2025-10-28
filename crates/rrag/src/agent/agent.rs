//! Core Agent implementation

use super::{AgentConfig, ConversationMemory, ConversationMode, ToolExecutor};
use crate::error::RragResult;
use rsllm::{ChatMessage, ChatResponse, Client};

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

        if self.config.verbose {
            println!("\n🤔 User: {}", input);
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
            if self.config.verbose {
                println!("\n🔄 Agent iteration {}", iteration);
            }

            // Call LLM with tools
            let response = self.llm_step(&conversation).await?;

            // Check for tool calls
            if let Some(tool_calls) = &response.tool_calls {
                if !tool_calls.is_empty() {
                    if self.config.verbose {
                        println!("🛠️  Agent wants to use {} tool(s)", tool_calls.len());
                    }

                    // Add assistant message with tool calls to conversation
                    let mut assistant_msg = ChatMessage::assistant(response.content.clone());
                    assistant_msg.tool_calls = Some(tool_calls.clone());
                    conversation.push(assistant_msg);

                    // Execute all tool calls
                    let tool_results = self.tool_executor.execute_tool_calls(tool_calls);

                    // Add tool results to conversation
                    for result in tool_results {
                        if self.config.verbose {
                            if let rsllm::MessageContent::Text(ref content) = result.content {
                                println!("   ✅ Tool result: {}", content);
                            }
                        }
                        conversation.push(result);
                    }

                    // Continue loop to let LLM process results
                    continue;
                }
            }

            // No tool calls - this is the final answer
            if self.config.verbose {
                println!("✅ Agent: {}", response.content);
            }

            // Update memory in stateful mode
            if self.config.conversation_mode == ConversationMode::Stateful {
                self.memory.add_message(ChatMessage::assistant(response.content.clone()));
            }

            return Ok(response.content);
        }

        // Exceeded max iterations
        Err(crate::error::RragError::Agent {
            agent_id: "default".to_string(),
            message: format!("Agent exceeded maximum iterations ({})", self.config.max_iterations),
            source: None,
        })
    }

    /// Single LLM call with tools
    async fn llm_step(&self, conversation: &[ChatMessage]) -> RragResult<ChatResponse> {
        // Get tool definitions
        let tools = self.tool_executor.registry().tool_definitions();

        if self.config.verbose {
            println!("   🔧 Calling LLM with {} tools", tools.len());
        }

        // Call LLM
        let response = self
            .llm_client
            .chat_completion_with_tools(conversation.to_vec(), tools)
            .await?;

        if self.config.verbose {
            println!(
                "   📥 LLM Response: content='{}', tool_calls={:?}",
                response.content,
                response.tool_calls.as_ref().map(|t| t.len())
            );
        }

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
