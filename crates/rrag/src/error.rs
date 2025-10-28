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
        /// Error message describing what went wrong
        message: String,
        #[source]
        /// Optional source error that caused this failure
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Embedding generation errors
    #[error("Embedding generation failed for {content_type}: {message}")]
    Embedding {
        /// Type of content that failed to embed (text, image, etc.)
        content_type: String,
        /// Error message describing the failure
        message: String,
        #[source]
        /// Optional source error that caused this failure
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Vector storage errors
    #[error("Vector storage operation failed: {operation}")]
    Storage {
        /// Storage operation that failed (insert, query, delete, etc.)
        operation: String,
        #[source]
        /// Source error from the storage backend
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// rsllm client errors
    #[error("rsllm client error: {operation}")]
    RsllmClient {
        /// Client operation that failed (request, auth, etc.)
        operation: String,
        #[source]
        /// Source error from the rsllm client
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Retrieval/search errors
    #[error("Retrieval failed: {query}")]
    Retrieval {
        /// Query that failed to execute
        query: String,
        #[source]
        /// Optional source error that caused this failure
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Tool execution errors
    #[error("Tool '{tool}' execution failed: {message}")]
    ToolExecution {
        /// Name of the tool that failed
        tool: String,
        /// Error message from the tool execution
        message: String,
        #[source]
        /// Optional source error that caused this failure
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Configuration errors
    #[error("Configuration error: {field}")]
    Configuration {
        /// Configuration field that has an invalid value
        field: String,
        /// Expected value or format for the field
        expected: String,
        /// Actual value that was provided
        actual: String,
    },

    /// Network/IO errors
    #[error("Network operation failed: {operation}")]
    Network {
        /// Network operation that failed (request, response, etc.)
        operation: String,
        #[source]
        /// Source error from the network operation
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Serialization/deserialization errors
    #[error("Serialization error: {data_type}")]
    Serialization {
        /// Type of data that failed to serialize/deserialize
        data_type: String,
        #[source]
        /// Source error from the serialization library
        source: serde_json::Error,
    },

    /// Timeout errors
    #[error("Operation timed out after {duration_ms}ms: {operation}")]
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Duration in milliseconds before timeout
        duration_ms: u64,
    },

    /// Memory/conversation errors
    #[error("Memory operation failed: {operation}")]
    Memory {
        /// Memory operation that failed
        operation: String,
        /// Error message describing the failure
        message: String,
    },

    /// Streaming errors
    #[error("Stream error in {context}: {message}")]
    Stream {
        /// Context where the stream error occurred
        context: String,
        /// Error message describing the stream failure
        message: String,
    },

    /// Agent execution errors
    #[error("Agent execution failed: {agent_id}")]
    Agent {
        /// ID of the agent that encountered the error
        agent_id: String,
        /// Error message from the agent
        message: String,
        #[source]
        /// Optional source error that caused this failure
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Validation errors
    #[error("Validation failed: {field}")]
    Validation {
        /// Field that failed validation
        field: String,
        /// Validation constraint that was violated
        constraint: String,
        /// Value that failed validation
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

    /// Create a network error
    pub fn network(
        operation: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Network {
            operation: operation.into(),
            source: Box::new(source),
        }
    }

    /// Create a configuration error (shorthand for config method)
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            field: "configuration".to_string(),
            expected: "valid configuration".to_string(),
            actual: message.into(),
        }
    }

    /// Create a serialization error with message
    pub fn serialization_with_message(
        data_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Agent {
            agent_id: "serialization".to_string(),
            message: format!("{}: {}", data_type.into(), message.into()),
            source: None,
        }
    }

    /// Create an I/O error
    pub fn io_error(message: impl Into<String>) -> Self {
        Self::Network {
            operation: "io_operation".to_string(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                message.into(),
            )),
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
            }
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
    /// Low priority error that doesn't affect core functionality
    Low = 1,
    /// Medium priority error that may cause minor issues
    Medium = 2,
    /// High priority error that affects core functionality
    High = 3,
    /// Critical error that causes system failure
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
        self.map_err(|e| RragError::Agent {
            agent_id: context.to_string(),
            message: e.to_string(),
            source: Some(Box::new(e)),
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

#[cfg(feature = "http")]
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
        assert_eq!(
            RragError::document_processing("test").category(),
            "document_processing"
        );
        assert_eq!(RragError::timeout("op", 1000).category(), "timeout");
        assert_eq!(
            RragError::config("field", "expected", "actual").category(),
            "configuration"
        );
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(
            RragError::config("field", "expected", "actual").severity(),
            ErrorSeverity::Critical
        );
        assert_eq!(
            RragError::timeout("op", 1000).severity(),
            ErrorSeverity::Low
        );
        assert_eq!(
            RragError::storage("op", std::io::Error::new(std::io::ErrorKind::Other, "test"))
                .severity(),
            ErrorSeverity::High
        );
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

// Implement From<RsllmError> for RragError
#[cfg(feature = "rsllm-client")]
impl From<rsllm::RsllmError> for RragError {
    fn from(err: rsllm::RsllmError) -> Self {
        RragError::RsllmClient {
            operation: "LLM operation".to_string(),
            source: Box::new(err),
        }
    }
}
