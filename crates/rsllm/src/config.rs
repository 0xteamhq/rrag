//! # RSLLM Configuration
//!
//! Configuration types and utilities for the RSLLM client library.
//! Supports environment variables, config files, and programmatic configuration.

use crate::{Provider, RsllmError, RsllmResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use url::Url;

/// Main client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Provider configuration
    pub provider: ProviderConfig,

    /// Model configuration
    pub model: ModelConfig,

    /// HTTP configuration
    pub http: HttpConfig,

    /// Retry configuration
    pub retry: RetryConfig,

    /// Custom headers
    pub headers: HashMap<String, String>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            provider: ProviderConfig::default(),
            model: ModelConfig::default(),
            http: HttpConfig::default(),
            retry: RetryConfig::default(),
            headers: HashMap::new(),
        }
    }
}

impl ClientConfig {
    /// Create a new configuration builder
    pub fn builder() -> ClientConfigBuilder {
        ClientConfigBuilder::new()
    }

    /// Load configuration from environment variables
    pub fn from_env() -> RsllmResult<Self> {
        dotenv::dotenv().ok(); // Load .env file if present

        let mut config = Self::default();

        // Provider configuration
        if let Ok(provider_str) = std::env::var("RSLLM_PROVIDER") {
            config.provider.provider = provider_str.parse()?;
        }

        if let Ok(api_key) = std::env::var("RSLLM_API_KEY") {
            config.provider.api_key = Some(api_key);
        }

        if let Ok(base_url) = std::env::var("RSLLM_BASE_URL") {
            config.provider.base_url = Some(base_url.parse()?);
        }

        // Model configuration
        if let Ok(model) = std::env::var("RSLLM_MODEL") {
            config.model.model = model;
        }

        if let Ok(temp_str) = std::env::var("RSLLM_TEMPERATURE") {
            config.model.temperature = Some(
                temp_str
                    .parse()
                    .map_err(|_| RsllmError::configuration("Invalid temperature value"))?,
            );
        }

        if let Ok(max_tokens_str) = std::env::var("RSLLM_MAX_TOKENS") {
            config.model.max_tokens = Some(
                max_tokens_str
                    .parse()
                    .map_err(|_| RsllmError::configuration("Invalid max_tokens value"))?,
            );
        }

        // HTTP configuration
        if let Ok(timeout_str) = std::env::var("RSLLM_TIMEOUT") {
            let timeout_secs: u64 = timeout_str
                .parse()
                .map_err(|_| RsllmError::configuration("Invalid timeout value"))?;
            config.http.timeout = Duration::from_secs(timeout_secs);
        }

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> RsllmResult<()> {
        // Validate provider
        self.provider.validate()?;

        // Validate model
        self.model.validate()?;

        // Validate HTTP config
        self.http.validate()?;

        Ok(())
    }
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// LLM provider type
    pub provider: Provider,

    /// API key for the provider
    pub api_key: Option<String>,

    /// Base URL for the provider (if custom)
    pub base_url: Option<Url>,

    /// Organization ID (for providers that support it)
    pub organization_id: Option<String>,

    /// Custom provider-specific settings
    pub custom_settings: HashMap<String, serde_json::Value>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider: Provider::OpenAI,
            api_key: None,
            base_url: None,
            organization_id: None,
            custom_settings: HashMap::new(),
        }
    }
}

impl ProviderConfig {
    /// Validate provider configuration
    pub fn validate(&self) -> RsllmResult<()> {
        // Check if API key is required and present
        match self.provider {
            Provider::OpenAI | Provider::Claude => {
                if self.api_key.is_none() {
                    return Err(RsllmError::configuration(format!(
                        "API key required for provider: {:?}",
                        self.provider
                    )));
                }
            }
            Provider::Ollama => {
                // Ollama typically doesn't require an API key for local instances
            }
        }

        // Validate base URL if provided
        if let Some(url) = &self.base_url {
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err(RsllmError::configuration(
                    "Base URL must use HTTP or HTTPS scheme",
                ));
            }
        }

        Ok(())
    }

    /// Get the effective base URL for the provider
    pub fn effective_base_url(&self) -> RsllmResult<Url> {
        if let Some(url) = &self.base_url {
            Ok(url.clone())
        } else {
            Ok(self.provider.default_base_url())
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier
    pub model: String,

    /// Temperature for sampling (0.0 to 2.0)
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Top-p sampling parameter
    pub top_p: Option<f32>,

    /// Frequency penalty
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    pub presence_penalty: Option<f32>,

    /// Stop sequences
    pub stop: Option<Vec<String>>,

    /// Whether to stream responses
    pub stream: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: false,
        }
    }
}

impl ModelConfig {
    /// Validate model configuration
    pub fn validate(&self) -> RsllmResult<()> {
        if self.model.is_empty() {
            return Err(RsllmError::validation(
                "model",
                "Model name cannot be empty",
            ));
        }

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(RsllmError::validation(
                    "temperature",
                    "Temperature must be between 0.0 and 2.0",
                ));
            }
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(RsllmError::validation(
                    "top_p",
                    "Top-p must be between 0.0 and 1.0",
                ));
            }
        }

        if let Some(freq_penalty) = self.frequency_penalty {
            if !(-2.0..=2.0).contains(&freq_penalty) {
                return Err(RsllmError::validation(
                    "frequency_penalty",
                    "Frequency penalty must be between -2.0 and 2.0",
                ));
            }
        }

        if let Some(pres_penalty) = self.presence_penalty {
            if !(-2.0..=2.0).contains(&pres_penalty) {
                return Err(RsllmError::validation(
                    "presence_penalty",
                    "Presence penalty must be between -2.0 and 2.0",
                ));
            }
        }

        Ok(())
    }
}

/// HTTP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Request timeout
    pub timeout: Duration,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Maximum number of redirects to follow
    pub max_redirects: u32,

    /// User agent string
    pub user_agent: String,

    /// Whether to use TLS verification
    pub verify_tls: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            max_redirects: 5,
            user_agent: format!("rsllm/{}", crate::VERSION),
            verify_tls: true,
        }
    }
}

impl HttpConfig {
    /// Validate HTTP configuration
    pub fn validate(&self) -> RsllmResult<()> {
        if self.timeout.as_secs() == 0 {
            return Err(RsllmError::validation(
                "timeout",
                "Timeout must be greater than 0",
            ));
        }

        if self.connect_timeout.as_secs() == 0 {
            return Err(RsllmError::validation(
                "connect_timeout",
                "Connect timeout must be greater than 0",
            ));
        }

        Ok(())
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,

    /// Base delay between retries
    pub base_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Backoff multiplier
    pub backoff_multiplier: f32,

    /// Whether to add jitter to retry delays
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

/// Builder for client configuration
pub struct ClientConfigBuilder {
    config: ClientConfig,
}

impl ClientConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ClientConfig::default(),
        }
    }

    /// Set the provider
    pub fn provider(mut self, provider: Provider) -> Self {
        self.config.provider.provider = provider;
        self
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config.provider.api_key = Some(api_key.into());
        self
    }

    /// Set the base URL
    pub fn base_url(mut self, base_url: impl AsRef<str>) -> RsllmResult<Self> {
        self.config.provider.base_url = Some(base_url.as_ref().parse()?);
        Ok(self)
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model.model = model.into();
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.model.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.config.model.max_tokens = Some(max_tokens);
        self
    }

    /// Enable streaming
    pub fn stream(mut self, stream: bool) -> Self {
        self.config.model.stream = stream;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.http.timeout = timeout;
        self
    }

    /// Add a custom header
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.insert(key.into(), value.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> RsllmResult<ClientConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ClientConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
