//! Agent configuration

use serde::{Deserialize, Serialize};

/// Agent conversation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationMode {
    /// Stateless: Each call is independent
    Stateless,
    /// Stateful: Maintains conversation across calls
    Stateful,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// System prompt that defines agent behavior
    pub system_prompt: String,

    /// Maximum iterations before stopping (prevents infinite loops)
    pub max_iterations: usize,

    /// Whether to show verbose output
    pub verbose: bool,

    /// Conversation mode
    pub conversation_mode: ConversationMode,

    /// Maximum conversation history length (for stateful mode)
    pub max_conversation_length: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful assistant with access to tools. Use tools when needed to provide accurate information.".to_string(),
            max_iterations: 10,
            verbose: false,
            conversation_mode: ConversationMode::Stateless,
            max_conversation_length: 50,
        }
    }
}

impl AgentConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set conversation mode
    pub fn with_conversation_mode(mut self, mode: ConversationMode) -> Self {
        self.conversation_mode = mode;
        self
    }

    /// Set max conversation length
    pub fn with_max_conversation_length(mut self, length: usize) -> Self {
        self.max_conversation_length = length;
        self
    }
}
