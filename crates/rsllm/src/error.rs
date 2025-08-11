//! # RSLLM Error Handling
//!
//! Comprehensive error types for the RSLLM client library.
//! Designed for precise error categorization and helpful debugging.

use thiserror::Error;

/// Result type for RSLLM operations
pub type RsllmResult<T> = Result<T, RsllmError>;

/// Comprehensive error types for RSLLM operations
#[derive(Error, Debug)]
pub enum RsllmError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Provider-specific errors
    #[error("Provider error ({provider}): {message}")]
    Provider {
        provider: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// HTTP/Network errors
    #[error("Network error: {message}")]
    Network {
        message: String,
        status_code: Option<u16>,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Authentication errors
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        message: String,
        retry_after: Option<std::time::Duration>,
    },

    /// API errors from providers
    #[error("API error ({provider}): {message} (code: {code})")]
    Api {
        provider: String,
        message: String,
        code: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Serialization/Deserialization errors
    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Streaming errors
    #[error("Streaming error: {message}")]
    Streaming {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Timeout errors
    #[error("Operation timed out after {timeout_ms}ms: {operation}")]
    Timeout { operation: String, timeout_ms: u64 },

    /// Validation errors
    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },

    /// Resource not found errors
    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    /// Invalid state errors
    #[error("Invalid state: {message}")]
    InvalidState { message: String },
}

impl RsllmError {
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    /// Create a configuration error with source
    pub fn configuration_with_source(
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(source.into()),
        }
    }

    /// Create a provider error
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a provider error with source
    pub fn provider_with_source(
        provider: impl Into<String>,
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            source: Some(source.into()),
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
            status_code: None,
            source: None,
        }
    }

    /// Create a network error with status code
    pub fn network_with_status(message: impl Into<String>, status_code: u16) -> Self {
        Self::Network {
            message: message.into(),
            status_code: Some(status_code),
            source: None,
        }
    }

    /// Create an authentication error
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a rate limit error
    pub fn rate_limit(
        message: impl Into<String>,
        retry_after: Option<std::time::Duration>,
    ) -> Self {
        Self::RateLimit {
            message: message.into(),
            retry_after,
        }
    }

    /// Create an API error
    pub fn api(
        provider: impl Into<String>,
        message: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self::Api {
            provider: provider.into(),
            message: message.into(),
            code: code.into(),
            source: None,
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
            source: None,
        }
    }

    /// Create a streaming error
    pub fn streaming(message: impl Into<String>) -> Self {
        Self::Streaming {
            message: message.into(),
            source: None,
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }

    /// Create a validation error
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Create a not found error
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Create an invalid state error
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState {
            message: message.into(),
        }
    }

    /// Get error category for metrics/logging
    pub fn category(&self) -> &'static str {
        match self {
            Self::Configuration { .. } => "configuration",
            Self::Provider { .. } => "provider",
            Self::Network { .. } => "network",
            Self::Authentication { .. } => "authentication",
            Self::RateLimit { .. } => "rate_limit",
            Self::Api { .. } => "api",
            Self::Serialization { .. } => "serialization",
            Self::Streaming { .. } => "streaming",
            Self::Timeout { .. } => "timeout",
            Self::Validation { .. } => "validation",
            Self::NotFound { .. } => "not_found",
            Self::InvalidState { .. } => "invalid_state",
        }
    }

    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::RateLimit { .. } => true,
            Self::Timeout { .. } => true,
            Self::Provider { .. } => false, // Depends on specific provider error
            Self::Api { .. } => false,      // Depends on specific API error
            _ => false,
        }
    }

    /// Get retry delay if applicable
    pub fn retry_delay(&self) -> Option<std::time::Duration> {
        match self {
            Self::RateLimit { retry_after, .. } => *retry_after,
            Self::Network { .. } => Some(std::time::Duration::from_secs(1)),
            Self::Timeout { .. } => Some(std::time::Duration::from_secs(2)),
            _ => None,
        }
    }
}

// Implement conversions from common error types
impl From<serde_json::Error> for RsllmError {
    fn from(err: serde_json::Error) -> Self {
        Self::serialization(format!("JSON error: {}", err))
    }
}

impl From<url::ParseError> for RsllmError {
    fn from(err: url::ParseError) -> Self {
        Self::configuration(format!("Invalid URL: {}", err))
    }
}

#[cfg(feature = "openai")]
impl From<reqwest::Error> for RsllmError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::timeout("HTTP request", 30000) // Default 30s timeout
        } else if err.is_connect() {
            Self::network(format!("Connection error: {}", err))
        } else if let Some(status) = err.status() {
            Self::network_with_status(format!("HTTP error: {}", err), status.as_u16())
        } else {
            Self::network(format!("Request error: {}", err))
        }
    }
}

impl From<tokio::time::error::Elapsed> for RsllmError {
    fn from(_err: tokio::time::error::Elapsed) -> Self {
        Self::timeout("operation", 0)
    }
}
