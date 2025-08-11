//! # RRAG Error Types
//! 
//! Comprehensive error handling designed for the Rust ecosystem.
//! Focuses on providing detailed context while maintaining performance.

use thiserror::Error;

/// Main error type for RRAG operations
/// 
/// Designed with Rust's error handling best practices:
/// - Uses `thiserror` for automatic trait implementations
/// - Provides structured error data for programmatic handling
/// - Includes source chain for debugging
/// - Categorizes errors for metrics and logging
#[derive(Error, Debug)]
pub enum RragError {
    /// Document processing errors
    #[error("Document processing failed: {message}")]
    DocumentProcessing { 
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Embedding generation errors
    #[error("Embedding generation failed for {content_type}: {message}")]
    Embedding {
        content_type: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Vector storage errors
    #[error("Vector storage operation failed: {operation}")]
    Storage {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// rsllm client errors
    #[error("rsllm client error: {operation}")]
    RsllmClient {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Retrieval/search errors
    #[error("Retrieval failed: {query}")]
    Retrieval {
        query: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Tool execution errors
    #[error("Tool '{tool}' execution failed: {message}")]
    ToolExecution {
        tool: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Configuration errors
    #[error("Configuration error: {field}")]
    Configuration {
        field: String,
        expected: String,
        actual: String,
    },

    /// Network/IO errors
    #[error("Network operation failed: {operation}")]
    Network {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error: {data_type}")]
    Serialization {
        data_type: String,
        #[source]
        source: serde_json::Error,
    },

    /// Timeout errors
    #[error("Operation timed out after {duration_ms}ms: {operation}")]
    Timeout {
        operation: String,
        duration_ms: u64,
    },

    /// Memory/conversation errors
    #[error("Memory operation failed: {operation}")]
    Memory {
        operation: String,
        message: String,
    },

    /// Streaming errors
    #[error("Stream error in {context}: {message}")]
    Stream {
        context: String,
        message: String,
    },

    /// Agent execution errors
    #[error("Agent execution failed: {agent_id}")]
    Agent {
        agent_id: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Validation errors
    #[error("Validation failed: {field}")]
    Validation {
        field: String,
        constraint: String,
        value: String,
    },
}

impl RragError {
    /// Create a document processing error
    pub fn document_processing(message: impl Into<String>) -> Self {
        Self::DocumentProcessing {
            message: message.into(),
            source: None,
        }
    }

    /// Create a document processing error with source
    pub fn document_processing_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::DocumentProcessing {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create an embedding error
    pub fn embedding(content_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Embedding {
            content_type: content_type.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a storage error
    pub fn storage(
        operation: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Storage {
            operation: operation.into(),
            source: Box::new(source),
        }
    }

    /// Create an rsllm client error
    pub fn rsllm_client(
        operation: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::RsllmClient {
            operation: operation.into(),
            source: Box::new(source),
        }
    }

    /// Create a retrieval error
    pub fn retrieval(query: impl Into<String>) -> Self {
        Self::Retrieval {
            query: query.into(),
            source: None,
        }
    }

    /// Create an evaluation error
    pub fn evaluation(message: impl Into<String>) -> Self {
        Self::Agent {
            agent_id: "evaluation".to_string(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a tool execution error
    pub fn tool_execution(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolExecution {
            tool: tool.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a configuration error
    pub fn config(
        field: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::Configuration {
            field: field.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, duration_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration_ms,
        }
    }

    /// Create a memory error
    pub fn memory(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Memory {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Create a stream error
    pub fn stream(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Stream {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Create an agent error
    pub fn agent(agent_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Agent {
            agent_id: agent_id.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a validation error
    pub fn validation(
        field: impl Into<String>,
        constraint: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::Validation {
            field: field.into(),
            constraint: constraint.into(),
            value: value.into(),
        }
    }

    /// Check if this error suggests a retry might succeed
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network { .. } 
            | Self::Timeout { .. } 
            | Self::RsllmClient { .. }
            | Self::Stream { .. }
        )
    }

    /// Get error category for metrics and logging
    pub fn category(&self) -> &'static str {
        match self {
            Self::DocumentProcessing { .. } => "document_processing",
            Self::Embedding { .. } => "embedding",
            Self::Storage { .. } => "storage",
            Self::RsllmClient { .. } => "rsllm_client",
            Self::Retrieval { .. } => "retrieval",
            Self::ToolExecution { .. } => "tool_execution",
            Self::Configuration { .. } => "configuration",
            Self::Network { .. } => "network",
            Self::Serialization { .. } => "serialization",
            Self::Timeout { .. } => "timeout",
            Self::Memory { .. } => "memory",
            Self::Stream { .. } => "stream",
            Self::Agent { agent_id, .. } => {
                if agent_id == "evaluation" {
                    "evaluation"
                } else {
                    "agent"
                }
            },
            Self::Validation { .. } => "validation",
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Configuration { .. } | Self::Validation { .. } => ErrorSeverity::Critical,
            Self::Storage { .. } | Self::RsllmClient { .. } => ErrorSeverity::High,
            Self::DocumentProcessing { .. } | Self::Embedding { .. } | Self::Retrieval { .. } => {
                ErrorSeverity::Medium
            }
            Self::ToolExecution { .. } | Self::Agent { .. } => ErrorSeverity::Medium,
            Self::Network { .. } | Self::Timeout { .. } | Self::Stream { .. } => ErrorSeverity::Low,
            Self::Serialization { .. } | Self::Memory { .. } => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Convenience type alias following Rust conventions
pub type RragResult<T> = std::result::Result<T, RragError>;

/// Extension trait for adding RRAG context to any Result
pub trait RragResultExt<T> {
    /// Add context to an error
    fn with_rrag_context(self, context: &str) -> RragResult<T>;
    
    /// Map to a specific RRAG error type
    fn map_to_rrag_error<F>(self, f: F) -> RragResult<T>
    where
        F: FnOnce() -> RragError;
}

impl<T, E> RragResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_rrag_context(self, context: &str) -> RragResult<T> {
        self.map_err(|e| {
            RragError::Agent {
                agent_id: context.to_string(),
                message: e.to_string(),
                source: Some(Box::new(e)),
            }
        })
    }

    fn map_to_rrag_error<F>(self, f: F) -> RragResult<T>
    where
        F: FnOnce() -> RragError,
    {
        self.map_err(|_| f())
    }
}

// Automatic conversions from common error types
impl From<serde_json::Error> for RragError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            data_type: "json".to_string(),
            source: err,
        }
    }
}

impl From<reqwest::Error> for RragError {
    fn from(err: reqwest::Error) -> Self {
        Self::Network {
            operation: "http_request".to_string(),
            source: Box::new(err),
        }
    }
}

impl From<tokio::time::error::Elapsed> for RragError {
    fn from(_err: tokio::time::error::Elapsed) -> Self {
        Self::Timeout {
            operation: "async_operation".to_string(),
            duration_ms: 0, // Unknown duration
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_categories() {
        assert_eq!(RragError::document_processing("test").category(), "document_processing");
        assert_eq!(RragError::timeout("op", 1000).category(), "timeout");
        assert_eq!(RragError::config("field", "expected", "actual").category(), "configuration");
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(RragError::config("field", "expected", "actual").severity(), ErrorSeverity::Critical);
        assert_eq!(RragError::timeout("op", 1000).severity(), ErrorSeverity::Low);
        assert_eq!(RragError::storage("op", std::io::Error::new(std::io::ErrorKind::Other, "test")).severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_retryable() {
        assert!(RragError::timeout("op", 1000).is_retryable());
        assert!(!RragError::config("field", "expected", "actual").is_retryable());
    }

    #[test]
    fn test_error_construction() {
        let err = RragError::tool_execution("calculator", "invalid input");
        if let RragError::ToolExecution { tool, message, .. } = err {
            assert_eq!(tool, "calculator");
            assert_eq!(message, "invalid input");
        } else {
            panic!("Wrong error type");
        }
    }
}