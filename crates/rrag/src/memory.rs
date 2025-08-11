//! # RRAG Memory System
//!
//! Conversation memory and context management with Rust-native async patterns.
//! Designed for efficient state management and persistence.

use crate::{RragError, RragResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Conversation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    /// Message ID
    pub id: String,

    /// Role (user, assistant, system, tool)
    pub role: MessageRole,

    /// Message content
    pub content: String,

    /// Message metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Token count (if available)
    pub token_count: Option<usize>,
}

/// Message roles in conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

impl ConversationMessage {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role,
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            token_count: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Tool, content)
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }

    /// Get estimated token count (simple heuristic if not set)
    pub fn estimated_tokens(&self) -> usize {
        self.token_count.unwrap_or_else(|| {
            // Simple estimation: ~4 characters per token
            self.content.len() / 4
        })
    }
}

/// Memory summary for efficient storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    /// Summary text
    pub summary: String,

    /// Number of messages summarized
    pub message_count: usize,

    /// Token count of original messages
    pub original_tokens: usize,

    /// Token count of summary
    pub summary_tokens: usize,

    /// Time range covered
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,

    /// Summary metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Core memory trait for conversation management
#[async_trait]
pub trait Memory: Send + Sync {
    /// Add a message to the conversation
    async fn add_message(&self, conversation_id: &str, role: &str, content: &str)
        -> RragResult<()>;

    /// Add a structured message
    async fn add_structured_message(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> RragResult<()>;

    /// Get conversation history
    async fn get_conversation_history(&self, conversation_id: &str) -> RragResult<Vec<String>>;

    /// Get structured conversation history
    async fn get_messages(&self, conversation_id: &str) -> RragResult<Vec<ConversationMessage>>;

    /// Get recent messages with limit
    async fn get_recent_messages(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> RragResult<Vec<ConversationMessage>>;

    /// Clear conversation history
    async fn clear_conversation(&self, conversation_id: &str) -> RragResult<()>;

    /// Get memory variables for prompt injection
    async fn get_memory_variables(
        &self,
        conversation_id: &str,
    ) -> RragResult<HashMap<String, String>>;

    /// Save arbitrary context
    async fn save_context(
        &self,
        conversation_id: &str,
        context: HashMap<String, String>,
    ) -> RragResult<()>;

    /// Health check
    async fn health_check(&self) -> RragResult<bool>;
}

/// Buffer memory - keeps recent messages in memory
pub struct ConversationBufferMemory {
    /// Stored conversations
    conversations: Arc<RwLock<HashMap<String, VecDeque<ConversationMessage>>>>,

    /// Configuration
    config: BufferMemoryConfig,
}

#[derive(Debug, Clone)]
pub struct BufferMemoryConfig {
    /// Maximum messages to keep per conversation
    pub max_messages: Option<usize>,

    /// Maximum age of messages in seconds
    pub max_age_seconds: Option<u64>,

    /// Memory key for prompt variables
    pub memory_key: String,
}

impl Default for BufferMemoryConfig {
    fn default() -> Self {
        Self {
            max_messages: Some(100),
            max_age_seconds: Some(3600 * 24), // 24 hours
            memory_key: "history".to_string(),
        }
    }
}

impl ConversationBufferMemory {
    pub fn new() -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
            config: BufferMemoryConfig::default(),
        }
    }

    pub fn with_config(config: BufferMemoryConfig) -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Clean up old messages based on configuration
    async fn cleanup_old_messages(&self, conversation_id: &str) {
        let mut conversations = self.conversations.write().await;

        if let Some(messages) = conversations.get_mut(conversation_id) {
            // Remove old messages by age
            if let Some(max_age) = self.config.max_age_seconds {
                let cutoff_time = chrono::Utc::now() - chrono::Duration::seconds(max_age as i64);
                while let Some(front) = messages.front() {
                    if front.timestamp < cutoff_time {
                        messages.pop_front();
                    } else {
                        break;
                    }
                }
            }

            // Limit by count
            if let Some(max_messages) = self.config.max_messages {
                while messages.len() > max_messages {
                    messages.pop_front();
                }
            }
        }
    }
}

impl Default for ConversationBufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for ConversationBufferMemory {
    async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> RragResult<()> {
        let role = match role.to_lowercase().as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User, // Default fallback
        };

        let message = ConversationMessage::new(role, content);
        self.add_structured_message(conversation_id, message).await
    }

    async fn add_structured_message(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> RragResult<()> {
        let mut conversations = self.conversations.write().await;

        let messages = conversations
            .entry(conversation_id.to_string())
            .or_insert_with(VecDeque::new);

        messages.push_back(message);

        // Release the lock before cleanup
        drop(conversations);

        // Clean up old messages
        self.cleanup_old_messages(conversation_id).await;

        Ok(())
    }

    async fn get_conversation_history(&self, conversation_id: &str) -> RragResult<Vec<String>> {
        let conversations = self.conversations.read().await;

        if let Some(messages) = conversations.get(conversation_id) {
            let history = messages
                .iter()
                .map(|msg| format!("{:?}: {}", msg.role, msg.content))
                .collect();
            Ok(history)
        } else {
            Ok(Vec::new())
        }
    }

    async fn get_messages(&self, conversation_id: &str) -> RragResult<Vec<ConversationMessage>> {
        let conversations = self.conversations.read().await;

        if let Some(messages) = conversations.get(conversation_id) {
            Ok(messages.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    async fn get_recent_messages(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> RragResult<Vec<ConversationMessage>> {
        let conversations = self.conversations.read().await;

        if let Some(messages) = conversations.get(conversation_id) {
            let recent: Vec<ConversationMessage> =
                messages.iter().rev().take(limit).rev().cloned().collect();
            Ok(recent)
        } else {
            Ok(Vec::new())
        }
    }

    async fn clear_conversation(&self, conversation_id: &str) -> RragResult<()> {
        let mut conversations = self.conversations.write().await;
        conversations.remove(conversation_id);
        Ok(())
    }

    async fn get_memory_variables(
        &self,
        conversation_id: &str,
    ) -> RragResult<HashMap<String, String>> {
        let history = self.get_conversation_history(conversation_id).await?;
        let mut variables = HashMap::new();

        variables.insert(self.config.memory_key.clone(), history.join("\n"));

        Ok(variables)
    }

    async fn save_context(
        &self,
        _conversation_id: &str,
        _context: HashMap<String, String>,
    ) -> RragResult<()> {
        // Simple buffer memory doesn't persist additional context
        // This could be extended to store context in message metadata
        Ok(())
    }

    async fn health_check(&self) -> RragResult<bool> {
        Ok(true)
    }
}

/// Token-aware buffer memory that respects token limits
pub struct ConversationTokenBufferMemory {
    /// Base buffer memory
    buffer: ConversationBufferMemory,

    /// Token-specific configuration
    token_config: TokenBufferConfig,
}

#[derive(Debug, Clone)]
pub struct TokenBufferConfig {
    /// Maximum tokens to keep in memory
    pub max_tokens: usize,

    /// Buffer size to keep below max (for safety)
    pub buffer_tokens: usize,

    /// How to handle token overflow
    pub overflow_strategy: TokenOverflowStrategy,
}

#[derive(Debug, Clone)]
pub enum TokenOverflowStrategy {
    /// Remove oldest messages
    RemoveOldest,

    /// Summarize old messages
    Summarize,

    /// Truncate message content
    Truncate,
}

impl Default for TokenBufferConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4000,
            buffer_tokens: 500,
            overflow_strategy: TokenOverflowStrategy::RemoveOldest,
        }
    }
}

impl ConversationTokenBufferMemory {
    pub fn new() -> Self {
        Self {
            buffer: ConversationBufferMemory::new(),
            token_config: TokenBufferConfig::default(),
        }
    }

    pub fn with_config(buffer_config: BufferMemoryConfig, token_config: TokenBufferConfig) -> Self {
        Self {
            buffer: ConversationBufferMemory::with_config(buffer_config),
            token_config,
        }
    }

    /// Calculate total tokens in conversation
    async fn calculate_total_tokens(&self, conversation_id: &str) -> RragResult<usize> {
        let messages = self.buffer.get_messages(conversation_id).await?;
        let total = messages.iter().map(|msg| msg.estimated_tokens()).sum();
        Ok(total)
    }

    /// Handle token overflow
    async fn handle_token_overflow(&self, conversation_id: &str) -> RragResult<()> {
        let current_tokens = self.calculate_total_tokens(conversation_id).await?;

        if current_tokens <= self.token_config.max_tokens {
            return Ok(());
        }

        match self.token_config.overflow_strategy {
            TokenOverflowStrategy::RemoveOldest => {
                let mut conversations = self.buffer.conversations.write().await;

                if let Some(messages) = conversations.get_mut(conversation_id) {
                    while !messages.is_empty() {
                        let total: usize = messages.iter().map(|msg| msg.estimated_tokens()).sum();
                        if total <= self.token_config.max_tokens - self.token_config.buffer_tokens {
                            break;
                        }
                        messages.pop_front();
                    }
                }
            }
            TokenOverflowStrategy::Summarize => {
                // This would require integration with an LLM for summarization
                // For now, fall back to removing oldest
                let mut conversations = self.buffer.conversations.write().await;

                if let Some(messages) = conversations.get_mut(conversation_id) {
                    // Remove half the messages as a simple strategy
                    let remove_count = messages.len() / 2;
                    for _ in 0..remove_count {
                        messages.pop_front();
                    }
                }
            }
            TokenOverflowStrategy::Truncate => {
                // Truncate message content (not implemented in this example)
                return Err(RragError::memory(
                    "token_overflow",
                    "Truncate strategy not implemented",
                ));
            }
        }

        Ok(())
    }
}

impl Default for ConversationTokenBufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for ConversationTokenBufferMemory {
    async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> RragResult<()> {
        self.buffer
            .add_message(conversation_id, role, content)
            .await?;
        self.handle_token_overflow(conversation_id).await?;
        Ok(())
    }

    async fn add_structured_message(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> RragResult<()> {
        self.buffer
            .add_structured_message(conversation_id, message)
            .await?;
        self.handle_token_overflow(conversation_id).await?;
        Ok(())
    }

    async fn get_conversation_history(&self, conversation_id: &str) -> RragResult<Vec<String>> {
        self.buffer.get_conversation_history(conversation_id).await
    }

    async fn get_messages(&self, conversation_id: &str) -> RragResult<Vec<ConversationMessage>> {
        self.buffer.get_messages(conversation_id).await
    }

    async fn get_recent_messages(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> RragResult<Vec<ConversationMessage>> {
        self.buffer
            .get_recent_messages(conversation_id, limit)
            .await
    }

    async fn clear_conversation(&self, conversation_id: &str) -> RragResult<()> {
        self.buffer.clear_conversation(conversation_id).await
    }

    async fn get_memory_variables(
        &self,
        conversation_id: &str,
    ) -> RragResult<HashMap<String, String>> {
        let mut variables = self.buffer.get_memory_variables(conversation_id).await?;

        // Add token information
        let token_count = self.calculate_total_tokens(conversation_id).await?;
        variables.insert("token_count".to_string(), token_count.to_string());
        variables.insert(
            "max_tokens".to_string(),
            self.token_config.max_tokens.to_string(),
        );

        Ok(variables)
    }

    async fn save_context(
        &self,
        conversation_id: &str,
        context: HashMap<String, String>,
    ) -> RragResult<()> {
        self.buffer.save_context(conversation_id, context).await
    }

    async fn health_check(&self) -> RragResult<bool> {
        self.buffer.health_check().await
    }
}

/// Summary memory that automatically summarizes old conversations
pub struct ConversationSummaryMemory {
    /// Current conversation buffer
    current_messages: Arc<RwLock<HashMap<String, VecDeque<ConversationMessage>>>>,

    /// Stored summaries
    summaries: Arc<RwLock<HashMap<String, Vec<MemorySummary>>>>,

    /// Configuration
    config: SummaryMemoryConfig,
}

#[derive(Debug, Clone)]
pub struct SummaryMemoryConfig {
    /// Maximum messages before summarization
    pub max_messages_before_summary: usize,

    /// Maximum total tokens before summarization
    pub max_tokens_before_summary: usize,

    /// Number of recent messages to keep after summarization
    pub keep_recent_messages: usize,

    /// Memory key for variables
    pub memory_key: String,

    /// Summary key for variables
    pub summary_key: String,
}

impl Default for SummaryMemoryConfig {
    fn default() -> Self {
        Self {
            max_messages_before_summary: 20,
            max_tokens_before_summary: 2000,
            keep_recent_messages: 5,
            memory_key: "history".to_string(),
            summary_key: "summary".to_string(),
        }
    }
}

impl ConversationSummaryMemory {
    pub fn new() -> Self {
        Self {
            current_messages: Arc::new(RwLock::new(HashMap::new())),
            summaries: Arc::new(RwLock::new(HashMap::new())),
            config: SummaryMemoryConfig::default(),
        }
    }

    pub fn with_config(config: SummaryMemoryConfig) -> Self {
        Self {
            current_messages: Arc::new(RwLock::new(HashMap::new())),
            summaries: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if summarization is needed
    async fn should_summarize(&self, conversation_id: &str) -> RragResult<bool> {
        let messages = self.current_messages.read().await;

        if let Some(msg_deque) = messages.get(conversation_id) {
            // Check message count
            if msg_deque.len() > self.config.max_messages_before_summary {
                return Ok(true);
            }

            // Check token count
            let total_tokens: usize = msg_deque.iter().map(|msg| msg.estimated_tokens()).sum();
            if total_tokens > self.config.max_tokens_before_summary {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Perform summarization (mock implementation)
    async fn summarize_conversation(&self, conversation_id: &str) -> RragResult<()> {
        let mut messages = self.current_messages.write().await;
        let mut summaries = self.summaries.write().await;

        if let Some(msg_deque) = messages.get_mut(conversation_id) {
            if msg_deque.len() <= self.config.keep_recent_messages {
                return Ok(());
            }

            // Calculate how many messages to summarize
            let to_summarize_count = msg_deque.len() - self.config.keep_recent_messages;

            // Extract messages to summarize
            let mut to_summarize = Vec::new();
            for _ in 0..to_summarize_count {
                if let Some(msg) = msg_deque.pop_front() {
                    to_summarize.push(msg);
                }
            }

            if !to_summarize.is_empty() {
                // Create a simple summary (in production, would use LLM)
                let summary_text = format!(
                    "Summary of {} messages from {} to {}",
                    to_summarize.len(),
                    to_summarize
                        .first()
                        .unwrap()
                        .timestamp
                        .format("%Y-%m-%d %H:%M:%S"),
                    to_summarize
                        .last()
                        .unwrap()
                        .timestamp
                        .format("%Y-%m-%d %H:%M:%S")
                );

                let original_tokens = to_summarize.iter().map(|msg| msg.estimated_tokens()).sum();

                let summary = MemorySummary {
                    summary: summary_text,
                    message_count: to_summarize.len(),
                    original_tokens,
                    summary_tokens: 50, // Estimated
                    start_time: to_summarize.first().unwrap().timestamp,
                    end_time: to_summarize.last().unwrap().timestamp,
                    metadata: HashMap::new(),
                };

                // Store the summary
                summaries
                    .entry(conversation_id.to_string())
                    .or_insert_with(Vec::new)
                    .push(summary);
            }
        }

        Ok(())
    }
}

impl Default for ConversationSummaryMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for ConversationSummaryMemory {
    async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> RragResult<()> {
        let role = match role.to_lowercase().as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };

        let message = ConversationMessage::new(role, content);
        self.add_structured_message(conversation_id, message).await
    }

    async fn add_structured_message(
        &self,
        conversation_id: &str,
        message: ConversationMessage,
    ) -> RragResult<()> {
        // Add the message
        {
            let mut messages = self.current_messages.write().await;
            let msg_deque = messages
                .entry(conversation_id.to_string())
                .or_insert_with(VecDeque::new);
            msg_deque.push_back(message);
        }

        // Check if summarization is needed
        if self.should_summarize(conversation_id).await? {
            self.summarize_conversation(conversation_id).await?;
        }

        Ok(())
    }

    async fn get_conversation_history(&self, conversation_id: &str) -> RragResult<Vec<String>> {
        let messages = self.current_messages.read().await;
        let summaries = self.summaries.read().await;

        let mut history = Vec::new();

        // Add summaries first
        if let Some(summary_list) = summaries.get(conversation_id) {
            for summary in summary_list {
                history.push(format!("Summary: {}", summary.summary));
            }
        }

        // Add current messages
        if let Some(msg_deque) = messages.get(conversation_id) {
            for msg in msg_deque {
                history.push(format!("{:?}: {}", msg.role, msg.content));
            }
        }

        Ok(history)
    }

    async fn get_messages(&self, conversation_id: &str) -> RragResult<Vec<ConversationMessage>> {
        let messages = self.current_messages.read().await;

        if let Some(msg_deque) = messages.get(conversation_id) {
            Ok(msg_deque.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    async fn get_recent_messages(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> RragResult<Vec<ConversationMessage>> {
        let messages = self.current_messages.read().await;

        if let Some(msg_deque) = messages.get(conversation_id) {
            let recent: Vec<ConversationMessage> =
                msg_deque.iter().rev().take(limit).rev().cloned().collect();
            Ok(recent)
        } else {
            Ok(Vec::new())
        }
    }

    async fn clear_conversation(&self, conversation_id: &str) -> RragResult<()> {
        let mut messages = self.current_messages.write().await;
        let mut summaries = self.summaries.write().await;

        messages.remove(conversation_id);
        summaries.remove(conversation_id);

        Ok(())
    }

    async fn get_memory_variables(
        &self,
        conversation_id: &str,
    ) -> RragResult<HashMap<String, String>> {
        let mut variables = HashMap::new();

        // Get current conversation
        let history = self.get_conversation_history(conversation_id).await?;
        variables.insert(self.config.memory_key.clone(), history.join("\n"));

        // Get summary
        let summaries = self.summaries.read().await;
        if let Some(summary_list) = summaries.get(conversation_id) {
            let summary_text = summary_list
                .iter()
                .map(|s| s.summary.clone())
                .collect::<Vec<_>>()
                .join("\n");
            variables.insert(self.config.summary_key.clone(), summary_text);
        }

        Ok(variables)
    }

    async fn save_context(
        &self,
        _conversation_id: &str,
        _context: HashMap<String, String>,
    ) -> RragResult<()> {
        // Could store context in message metadata or separate storage
        Ok(())
    }

    async fn health_check(&self) -> RragResult<bool> {
        Ok(true)
    }
}

/// High-level memory service that can switch between different memory types
pub struct MemoryService {
    /// Active memory implementation
    memory: Arc<dyn Memory>,

    /// Service configuration
    config: MemoryServiceConfig,
}

#[derive(Debug, Clone)]
pub struct MemoryServiceConfig {
    /// Default conversation settings
    pub default_conversation_settings: ConversationSettings,

    /// Enable memory persistence
    pub enable_persistence: bool,

    /// Persistence interval in seconds
    pub persistence_interval_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct ConversationSettings {
    /// Maximum messages per conversation
    pub max_messages: Option<usize>,

    /// Maximum age for messages
    pub max_age_hours: Option<u64>,

    /// Auto-summarization threshold
    pub auto_summarize_threshold: Option<usize>,
}

impl Default for MemoryServiceConfig {
    fn default() -> Self {
        Self {
            default_conversation_settings: ConversationSettings::default(),
            enable_persistence: false,
            persistence_interval_seconds: 300, // 5 minutes
        }
    }
}

impl Default for ConversationSettings {
    fn default() -> Self {
        Self {
            max_messages: Some(100),
            max_age_hours: Some(24),
            auto_summarize_threshold: Some(50),
        }
    }
}

impl MemoryService {
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self {
            memory,
            config: MemoryServiceConfig::default(),
        }
    }

    pub fn with_config(memory: Arc<dyn Memory>, config: MemoryServiceConfig) -> Self {
        Self { memory, config }
    }

    /// Add a user message
    pub async fn add_user_message(&self, conversation_id: &str, content: &str) -> RragResult<()> {
        self.memory
            .add_message(conversation_id, "user", content)
            .await
    }

    /// Add an assistant message
    pub async fn add_assistant_message(
        &self,
        conversation_id: &str,
        content: &str,
    ) -> RragResult<()> {
        self.memory
            .add_message(conversation_id, "assistant", content)
            .await
    }

    /// Get formatted conversation for prompts
    pub async fn get_conversation_context(&self, conversation_id: &str) -> RragResult<String> {
        let variables = self.memory.get_memory_variables(conversation_id).await?;

        // Return the main history
        Ok(variables.get("history").unwrap_or(&String::new()).clone())
    }

    /// Get memory variables for prompt templates
    pub async fn get_prompt_variables(
        &self,
        conversation_id: &str,
    ) -> RragResult<HashMap<String, String>> {
        self.memory.get_memory_variables(conversation_id).await
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        self.memory.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conversation_message() {
        let msg = ConversationMessage::user("Hello world")
            .with_metadata("source", serde_json::Value::String("test".to_string()))
            .with_token_count(10);

        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "Hello world");
        assert_eq!(msg.estimated_tokens(), 10);
        assert_eq!(
            msg.metadata.get("source").unwrap().as_str().unwrap(),
            "test"
        );
    }

    #[tokio::test]
    async fn test_buffer_memory() {
        let memory = ConversationBufferMemory::new();
        let conv_id = "test_conversation";

        // Add messages
        memory.add_message(conv_id, "user", "Hello").await.unwrap();
        memory
            .add_message(conv_id, "assistant", "Hi there!")
            .await
            .unwrap();

        // Get history
        let history = memory.get_conversation_history(conv_id).await.unwrap();
        assert_eq!(history.len(), 2);
        assert!(history[0].contains("Hello"));
        assert!(history[1].contains("Hi there!"));

        // Get messages
        let messages = memory.get_messages(conv_id).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);

        // Test recent messages
        let recent = memory.get_recent_messages(conv_id, 1).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].content, "Hi there!");
    }

    #[tokio::test]
    async fn test_token_buffer_memory() {
        let config = TokenBufferConfig {
            max_tokens: 100,
            buffer_tokens: 10,
            overflow_strategy: TokenOverflowStrategy::RemoveOldest,
        };

        let memory =
            ConversationTokenBufferMemory::with_config(BufferMemoryConfig::default(), config);

        let conv_id = "test_token_conversation";

        // Add many messages to trigger overflow
        for i in 0..20 {
            memory
                .add_message(
                    conv_id,
                    "user",
                    &format!("Message number {} with some content", i),
                )
                .await
                .unwrap();
        }

        let total_tokens = memory.calculate_total_tokens(conv_id).await.unwrap();
        assert!(
            total_tokens <= 100,
            "Total tokens {} should be <= 100",
            total_tokens
        );

        let messages = memory.get_messages(conv_id).await.unwrap();
        assert!(
            messages.len() < 20,
            "Should have removed some messages due to token limit"
        );
    }

    #[tokio::test]
    async fn test_memory_service() {
        let memory = Arc::new(ConversationBufferMemory::new());
        let service = MemoryService::new(memory);

        let conv_id = "service_test";

        service
            .add_user_message(conv_id, "How are you?")
            .await
            .unwrap();
        service
            .add_assistant_message(conv_id, "I'm doing well, thank you!")
            .await
            .unwrap();

        let context = service.get_conversation_context(conv_id).await.unwrap();
        assert!(context.contains("How are you?"));
        assert!(context.contains("I'm doing well"));

        let variables = service.get_prompt_variables(conv_id).await.unwrap();
        assert!(variables.contains_key("history"));

        assert!(service.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_summary_memory() {
        let config = SummaryMemoryConfig {
            max_messages_before_summary: 3,
            max_tokens_before_summary: 1000,
            keep_recent_messages: 1,
            memory_key: "history".to_string(),
            summary_key: "summary".to_string(),
        };

        let memory = ConversationSummaryMemory::with_config(config);
        let conv_id = "summary_test";

        // Add enough messages to trigger summarization
        memory
            .add_message(conv_id, "user", "First message")
            .await
            .unwrap();
        memory
            .add_message(conv_id, "assistant", "First response")
            .await
            .unwrap();
        memory
            .add_message(conv_id, "user", "Second message")
            .await
            .unwrap();
        memory
            .add_message(conv_id, "assistant", "Second response")
            .await
            .unwrap();

        // Should have triggered summarization
        let messages = memory.get_messages(conv_id).await.unwrap();
        assert!(messages.len() <= 1, "Should have summarized old messages");

        let variables = memory.get_memory_variables(conv_id).await.unwrap();
        assert!(
            variables.contains_key("summary"),
            "Should have summary in variables"
        );
    }
}
