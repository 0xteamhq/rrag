//! # RSLLM Message Types
//!
//! Chat message types and content handling for RSLLM.
//! Supports text, multi-modal content, and role-based messaging.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Role of a message participant in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions, context)
    System,
    /// User message (human input)
    User,
    /// Assistant message (AI response)
    Assistant,
    /// Function/tool call message
    Tool,
}

impl MessageRole {
    /// Check if this role can initiate a conversation
    pub fn can_initiate(&self) -> bool {
        matches!(self, MessageRole::System | MessageRole::User)
    }

    /// Check if this role can respond to messages
    pub fn can_respond(&self) -> bool {
        matches!(self, MessageRole::Assistant | MessageRole::Tool)
    }
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
        }
    }
}

/// Content of a chat message - supports text and multi-modal content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),

    /// Multi-modal content (text + images/attachments)
    MultiModal {
        text: Option<String>,
        attachments: Vec<ContentAttachment>,
    },
}

impl MessageContent {
    /// Create text content
    pub fn text(content: impl Into<String>) -> Self {
        Self::Text(content.into())
    }

    /// Create multi-modal content with text
    pub fn multi_modal(text: impl Into<String>) -> Self {
        Self::MultiModal {
            text: Some(text.into()),
            attachments: Vec::new(),
        }
    }

    /// Add an attachment to multi-modal content
    pub fn with_attachment(mut self, attachment: ContentAttachment) -> Self {
        match &mut self {
            Self::MultiModal { attachments, .. } => {
                attachments.push(attachment);
            }
            Self::Text(text) => {
                let text = text.clone();
                self = Self::MultiModal {
                    text: Some(text),
                    attachments: vec![attachment],
                };
            }
        }
        self
    }

    /// Get the text content, if any
    pub fn text_content(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::MultiModal { text, .. } => text.as_deref(),
        }
    }

    /// Get attachments, if any
    pub fn attachments(&self) -> &[ContentAttachment] {
        match self {
            Self::Text(_) => &[],
            Self::MultiModal { attachments, .. } => attachments,
        }
    }

    /// Check if content is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::MultiModal { text, attachments } => {
                text.as_ref().map_or(true, |t| t.is_empty()) && attachments.is_empty()
            }
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

/// Attachment within message content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAttachment {
    /// Type of attachment
    pub attachment_type: AttachmentType,

    /// Content of the attachment
    pub content: AttachmentContent,

    /// Optional metadata
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl ContentAttachment {
    /// Create an image attachment from base64 data
    pub fn image_base64(mime_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            attachment_type: AttachmentType::Image,
            content: AttachmentContent::Base64 {
                mime_type: mime_type.into(),
                data: data.into(),
            },
            metadata: None,
        }
    }

    /// Create an image attachment from URL
    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            attachment_type: AttachmentType::Image,
            content: AttachmentContent::Url { url: url.into() },
            metadata: None,
        }
    }

    /// Add metadata to the attachment
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }
}

/// Type of content attachment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttachmentType {
    /// Image attachment
    Image,
    /// Audio attachment
    Audio,
    /// Video attachment
    Video,
    /// Document attachment
    Document,
    /// Other/custom attachment type
    Other,
}

/// Content of an attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AttachmentContent {
    /// Base64-encoded content
    Base64 { mime_type: String, data: String },

    /// URL reference
    Url { url: String },

    /// Raw bytes (for internal use)
    #[serde(skip)]
    Bytes { mime_type: String, data: Vec<u8> },
}

/// A chat message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: MessageRole,

    /// Content of the message
    pub content: MessageContent,

    /// Optional name of the sender (for user/assistant disambiguation)
    pub name: Option<String>,

    /// Tool call information (for assistant messages)
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID (for tool response messages)
    pub tool_call_id: Option<String>,

    /// Message metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Message timestamp
    #[serde(with = "chrono::serde::ts_seconds_option")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl ChatMessage {
    /// Create a new chat message
    pub fn new(role: MessageRole, content: impl Into<MessageContent>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<MessageContent>) -> Self {
        Self::new(MessageRole::System, content)
    }

    /// Create a user message
    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create a tool response message
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<MessageContent>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            metadata: HashMap::new(),
            timestamp: Some(chrono::Utc::now()),
        }
    }

    /// Set the sender name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add tool calls to the message
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Add metadata to the message
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get the text content of the message
    pub fn text(&self) -> Option<&str> {
        self.content.text_content()
    }

    /// Check if message is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get message length in characters
    pub fn len(&self) -> usize {
        self.text().map_or(0, |t| t.len())
    }
}

/// Tool call information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,

    /// Type of tool call
    #[serde(rename = "type")]
    pub call_type: ToolCallType,

    /// Tool function details
    pub function: ToolFunction,
}

impl ToolCall {
    /// Create a new function tool call
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            call_type: ToolCallType::Function,
            function: ToolFunction {
                name: name.into(),
                arguments,
            },
        }
    }
}

/// Type of tool call
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallType {
    /// Function call
    Function,
}

/// Tool function call details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Name of the function to call
    pub name: String,

    /// Arguments to pass to the function (as JSON)
    pub arguments: serde_json::Value,
}

/// Message builder for fluent message construction
pub struct MessageBuilder {
    message: ChatMessage,
}

impl MessageBuilder {
    /// Start building a new message
    pub fn new(role: MessageRole) -> Self {
        Self {
            message: ChatMessage {
                role,
                content: MessageContent::Text(String::new()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                metadata: HashMap::new(),
                timestamp: Some(chrono::Utc::now()),
            },
        }
    }

    /// Set the content
    pub fn content(mut self, content: impl Into<MessageContent>) -> Self {
        self.message.content = content.into();
        self
    }

    /// Set the sender name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.message.name = Some(name.into());
        self
    }

    /// Add tool calls
    pub fn tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.message.tool_calls = Some(tool_calls);
        self
    }

    /// Set tool call ID
    pub fn tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.message.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.message.metadata.insert(key.into(), value);
        self
    }

    /// Build the message
    pub fn build(self) -> ChatMessage {
        self.message
    }
}

impl Default for MessageBuilder {
    fn default() -> Self {
        Self::new(MessageRole::User)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = ChatMessage::user("Hello, world!");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.text(), Some("Hello, world!"));
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_message_builder() {
        let msg = MessageBuilder::new(MessageRole::Assistant)
            .content("Hello there!")
            .name("Assistant")
            .metadata("source", serde_json::Value::String("test".to_string()))
            .build();

        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.text(), Some("Hello there!"));
        assert_eq!(msg.name, Some("Assistant".to_string()));
        assert!(msg.metadata.contains_key("source"));
    }

    #[test]
    fn test_multi_modal_content() {
        let content = MessageContent::multi_modal("Check this image").with_attachment(
            ContentAttachment::image_url("https://example.com/image.jpg"),
        );

        assert_eq!(content.text_content(), Some("Check this image"));
        assert_eq!(content.attachments().len(), 1);
    }
}
