//! # RSLLM Response Types
//!
//! Response types for chat completions, embeddings, and other LLM operations.
//! Supports both streaming and non-streaming responses with usage tracking.

use crate::{MessageRole, ToolCall};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Response from a chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Generated content
    pub content: String,

    /// Model used for generation
    pub model: String,

    /// Usage statistics
    pub usage: Option<Usage>,

    /// Finish reason
    pub finish_reason: Option<String>,

    /// Tool calls made by the assistant
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Response metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Response timestamp
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,

    /// Response ID (if provided by provider)
    pub id: Option<String>,
}

impl ChatResponse {
    /// Create a new chat response
    pub fn new(content: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            model: model.into(),
            usage: None,
            finish_reason: None,
            tool_calls: None,
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
            id: None,
        }
    }

    /// Set usage statistics
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Set tool calls
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set response ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Check if the response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .map_or(false, |calls| !calls.is_empty())
    }

    /// Check if the response finished successfully
    pub fn is_finished(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("stop") | Some("end_turn") | Some("tool_calls")
        )
    }

    /// Check if the response was truncated due to length
    pub fn is_truncated(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("length") | Some("max_tokens")
        )
    }

    /// Get the content length
    pub fn content_length(&self) -> usize {
        self.content.len()
    }
}

/// Response from a completion request (non-chat)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated text
    pub text: String,

    /// Model used for generation
    pub model: String,

    /// Usage statistics
    pub usage: Option<Usage>,

    /// Finish reason
    pub finish_reason: Option<String>,

    /// Log probabilities (if requested)
    pub logprobs: Option<LogProbs>,

    /// Response metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Response timestamp
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,

    /// Response ID (if provided by provider)
    pub id: Option<String>,
}

impl CompletionResponse {
    /// Create a new completion response
    pub fn new(text: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: model.into(),
            usage: None,
            finish_reason: None,
            logprobs: None,
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
            id: None,
        }
    }

    /// Set usage statistics
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Set log probabilities
    pub fn with_logprobs(mut self, logprobs: LogProbs) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set response ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// A single chunk in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Content delta for this chunk
    pub content: String,

    /// Whether this is a delta (partial) or complete content
    pub is_delta: bool,

    /// Whether this is the final chunk
    pub is_done: bool,

    /// Model name
    pub model: String,

    /// Role of the message (if applicable)
    pub role: Option<MessageRole>,

    /// Tool calls delta (if applicable)
    pub tool_calls_delta: Option<Vec<ToolCallDelta>>,

    /// Finish reason (if this is the final chunk)
    pub finish_reason: Option<String>,

    /// Usage statistics (typically only in final chunk)
    pub usage: Option<Usage>,

    /// Chunk metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Chunk timestamp
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl StreamChunk {
    /// Create a new stream chunk
    pub fn new(
        content: impl Into<String>,
        model: impl Into<String>,
        is_delta: bool,
        is_done: bool,
    ) -> Self {
        Self {
            content: content.into(),
            is_delta,
            is_done,
            model: model.into(),
            role: None,
            tool_calls_delta: None,
            finish_reason: None,
            usage: None,
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Create a delta chunk
    pub fn delta(content: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(content, model, true, false)
    }

    /// Create a final chunk
    pub fn done(model: impl Into<String>) -> Self {
        Self::new("", model, false, true)
    }

    /// Set the role
    pub fn with_role(mut self, role: MessageRole) -> Self {
        self.role = Some(role);
        self
    }

    /// Set tool calls delta
    pub fn with_tool_calls_delta(mut self, delta: Vec<ToolCallDelta>) -> Self {
        self.tool_calls_delta = Some(delta);
        self
    }

    /// Set finish reason
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Set usage statistics
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Check if this chunk has content
    pub fn has_content(&self) -> bool {
        !self.content.is_empty()
    }

    /// Check if this chunk has tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls_delta
            .as_ref()
            .map_or(false, |calls| !calls.is_empty())
    }
}

/// Tool call delta for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Tool call index
    pub index: u32,

    /// Tool call ID (if starting a new call)
    pub id: Option<String>,

    /// Tool call type (if starting a new call)
    #[serde(rename = "type")]
    pub call_type: Option<String>,

    /// Function delta
    pub function: Option<ToolFunctionDelta>,
}

/// Tool function delta for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionDelta {
    /// Function name (if starting a new call)
    pub name: Option<String>,

    /// Arguments delta (partial JSON string)
    pub arguments: Option<String>,
}

/// Usage statistics for API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Number of tokens in the completion
    pub completion_tokens: u32,

    /// Total number of tokens used
    pub total_tokens: u32,

    /// Number of cached tokens (if applicable)
    pub cached_tokens: Option<u32>,

    /// Reasoning tokens (for models with reasoning capabilities)
    pub reasoning_tokens: Option<u32>,
}

impl Usage {
    /// Create new usage statistics
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: None,
            reasoning_tokens: None,
        }
    }

    /// Set cached tokens
    pub fn with_cached_tokens(mut self, cached_tokens: u32) -> Self {
        self.cached_tokens = Some(cached_tokens);
        self
    }

    /// Set reasoning tokens
    pub fn with_reasoning_tokens(mut self, reasoning_tokens: u32) -> Self {
        self.reasoning_tokens = Some(reasoning_tokens);
        self
    }

    /// Get effective prompt tokens (excluding cached)
    pub fn effective_prompt_tokens(&self) -> u32 {
        self.prompt_tokens - self.cached_tokens.unwrap_or(0)
    }

    /// Get total cost in tokens
    pub fn total_cost(&self) -> u32 {
        self.effective_prompt_tokens() + self.completion_tokens
    }
}

/// Log probabilities for completion responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    /// Top log probabilities for each token
    pub token_logprobs: Vec<Option<f64>>,

    /// Top alternative tokens and their log probabilities
    pub top_logprobs: Vec<Option<HashMap<String, f64>>>,

    /// Text offset for each token
    pub text_offset: Vec<usize>,
}

/// Embedding response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Embedding vectors
    pub embeddings: Vec<Vec<f32>>,

    /// Model used for embeddings
    pub model: String,

    /// Usage statistics
    pub usage: Option<Usage>,

    /// Response metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Response timestamp
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl EmbeddingResponse {
    /// Create a new embedding response
    pub fn new(embeddings: Vec<Vec<f32>>, model: impl Into<String>) -> Self {
        Self {
            embeddings,
            model: model.into(),
            usage: None,
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Set usage statistics
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get the number of embeddings
    pub fn count(&self) -> usize {
        self.embeddings.len()
    }

    /// Get the embedding dimension (if any embeddings exist)
    pub fn dimension(&self) -> Option<usize> {
        self.embeddings.first().map(|emb| emb.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_response_creation() {
        let response = ChatResponse::new("Hello!", "gpt-4")
            .with_finish_reason("stop")
            .with_usage(Usage::new(10, 5));

        assert_eq!(response.content, "Hello!");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.finish_reason, Some("stop".to_string()));
        assert!(response.usage.is_some());
        assert!(response.is_finished());
    }

    #[test]
    fn test_stream_chunk() {
        let chunk = StreamChunk::delta("Hello", "gpt-4").with_role(MessageRole::Assistant);

        assert_eq!(chunk.content, "Hello");
        assert!(chunk.is_delta);
        assert!(!chunk.is_done);
        assert_eq!(chunk.role, Some(MessageRole::Assistant));
        assert!(chunk.has_content());
    }

    #[test]
    fn test_usage_calculation() {
        let usage = Usage::new(100, 50).with_cached_tokens(20);

        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.effective_prompt_tokens(), 80);
        assert_eq!(usage.total_cost(), 130);
    }
}
