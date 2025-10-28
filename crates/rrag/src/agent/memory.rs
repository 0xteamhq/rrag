//! Conversation memory management

use rsllm::ChatMessage;

/// Manages conversation history for the agent
#[derive(Debug, Clone)]
pub struct ConversationMemory {
    messages: Vec<ChatMessage>,
    max_length: usize,
}

impl ConversationMemory {
    /// Create a new conversation memory
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_length: 50,
        }
    }

    /// Create with custom max length
    pub fn with_max_length(max_length: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_length,
        }
    }

    /// Add a message to history
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);

        // Trim if exceeds max length (keep system message)
        if self.messages.len() > self.max_length {
            let system_msg = self.messages.first().cloned();
            self.messages
                .drain(1..self.messages.len() - self.max_length + 1);
            if let Some(sys_msg) = system_msg {
                if matches!(sys_msg.role, rsllm::MessageRole::System) {
                    // System message was removed, restore it
                    if !matches!(
                        self.messages.first().unwrap().role,
                        rsllm::MessageRole::System
                    ) {
                        self.messages.insert(0, sys_msg);
                    }
                }
            }
        }
    }

    /// Get all messages
    pub fn get_messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Get messages as owned Vec
    pub fn to_messages(&self) -> Vec<ChatMessage> {
        self.messages.clone()
    }

    /// Clear all messages except system
    pub fn clear(&mut self) {
        let system_msg = self
            .messages
            .iter()
            .find(|m| matches!(m.role, rsllm::MessageRole::System))
            .cloned();

        self.messages.clear();

        if let Some(msg) = system_msg {
            self.messages.push(msg);
        }
    }

    /// Get number of messages
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}
