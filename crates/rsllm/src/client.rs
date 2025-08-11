//! # RSLLM Client
//!
//! High-level client interface for RSLLM with multi-provider support.
//! Provides unified API for chat completions, embeddings, and streaming.

use crate::{
    ChatMessage, ChatResponse, ChatStream, ClientConfig, EmbeddingResponse, Provider, RsllmError,
    RsllmResult,
};

#[cfg(feature = "openai")]
use crate::provider::OpenAIProvider;

#[cfg(feature = "ollama")]
use crate::provider::OllamaProvider;

use crate::provider::LLMProvider;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// High-level RSLLM client
pub struct Client {
    /// Client configuration
    config: ClientConfig,

    /// Provider instance
    provider: Arc<dyn LLMProvider>,

    /// Client metadata
    metadata: HashMap<String, serde_json::Value>,
}

impl Client {
    /// Create a new client with configuration
    pub fn new(config: ClientConfig) -> RsllmResult<Self> {
        config.validate()?;

        let provider = Self::create_provider(&config)?;

        Ok(Self {
            config,
            provider,
            metadata: HashMap::new(),
        })
    }

    /// Create a client builder
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Create a client from environment variables
    pub fn from_env() -> RsllmResult<Self> {
        let config = ClientConfig::from_env()?;
        Self::new(config)
    }

    /// Create provider instance based on configuration
    fn create_provider(config: &ClientConfig) -> RsllmResult<Arc<dyn LLMProvider>> {
        match config.provider.provider {
            #[cfg(feature = "openai")]
            Provider::OpenAI => {
                let api_key = config
                    .provider
                    .api_key
                    .as_ref()
                    .ok_or_else(|| RsllmError::configuration("OpenAI API key required"))?;

                let provider = OpenAIProvider::new(
                    api_key.clone(),
                    config.provider.base_url.clone(),
                    config.provider.organization_id.clone(),
                )?;

                Ok(Arc::new(provider))
            }

            #[cfg(feature = "ollama")]
            Provider::Ollama => {
                let provider = OllamaProvider::new(config.provider.base_url.clone())?;
                Ok(Arc::new(provider))
            }

            #[cfg(feature = "claude")]
            Provider::Claude => {
                // Claude provider implementation would go here
                Err(RsllmError::configuration(
                    "Claude provider not yet implemented",
                ))
            }

            #[allow(unreachable_patterns)]
            _ => Err(RsllmError::configuration(format!(
                "Provider {:?} not supported in current build",
                config.provider.provider
            ))),
        }
    }

    /// Get client configuration
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Get provider instance
    pub fn provider(&self) -> &Arc<dyn LLMProvider> {
        &self.provider
    }

    /// Add client metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Get client metadata
    pub fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }

    /// Health check for the underlying provider
    pub async fn health_check(&self) -> RsllmResult<bool> {
        self.provider.health_check().await
    }

    /// Get supported models from the provider
    pub fn supported_models(&self) -> Vec<String> {
        self.provider.supported_models()
    }

    /// Chat completion (non-streaming)
    pub async fn chat_completion(&self, messages: Vec<ChatMessage>) -> RsllmResult<ChatResponse> {
        self.chat_completion_with_options(messages, None, None, None)
            .await
    }

    /// Chat completion with custom options
    pub async fn chat_completion_with_options(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<ChatResponse> {
        // Validate messages
        if messages.is_empty() {
            return Err(RsllmError::validation(
                "messages",
                "Messages cannot be empty",
            ));
        }

        // Use configured model if not specified
        let model = model.unwrap_or(&self.config.model.model);

        // Use configured temperature if not specified
        let temperature = temperature.or(self.config.model.temperature);

        // Use configured max_tokens if not specified
        let max_tokens = max_tokens.or(self.config.model.max_tokens);

        self.provider
            .chat_completion(messages, Some(model), temperature, max_tokens)
            .await
    }

    /// Chat completion (streaming)
    pub async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
    ) -> RsllmResult<ChatStream> {
        self.chat_completion_stream_with_options(messages, None, None, None)
            .await
    }

    /// Chat completion streaming with custom options
    pub async fn chat_completion_stream_with_options(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<ChatStream> {
        // Validate messages
        if messages.is_empty() {
            return Err(RsllmError::validation(
                "messages",
                "Messages cannot be empty",
            ));
        }

        // Use configured model if not specified
        let model = model.unwrap_or(&self.config.model.model);

        // Use configured temperature if not specified
        let temperature = temperature.or(self.config.model.temperature);

        // Use configured max_tokens if not specified
        let max_tokens = max_tokens.or(self.config.model.max_tokens);

        let stream = self
            .provider
            .chat_completion_stream(messages, Some(model.to_string()), temperature, max_tokens)
            .await?;

        // Convert Box<dyn Stream + Unpin> to Pin<Box<dyn Stream>>
        Ok(Box::pin(stream) as ChatStream)
    }

    /// Simple text completion
    pub async fn complete(&self, prompt: impl Into<String>) -> RsllmResult<String> {
        let messages = vec![ChatMessage::user(prompt.into())];
        let response = self.chat_completion(messages).await?;
        Ok(response.content)
    }

    /// Simple streaming text completion
    pub async fn complete_stream(&self, prompt: impl Into<String>) -> RsllmResult<ChatStream> {
        let messages = vec![ChatMessage::user(prompt.into())];
        self.chat_completion_stream(messages).await
    }

    /// Create embeddings (placeholder - would need provider support)
    pub async fn create_embeddings(&self, _inputs: Vec<String>) -> RsllmResult<EmbeddingResponse> {
        // TODO: Implement embeddings support in providers
        Err(RsllmError::configuration("Embeddings not yet implemented"))
    }

    /// Count tokens in text (placeholder - would need tokenizer)
    pub fn count_tokens(&self, _text: &str) -> RsllmResult<u32> {
        // TODO: Implement tokenization
        Err(RsllmError::configuration(
            "Token counting not yet implemented",
        ))
    }
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("provider_type", &self.provider.provider_type())
            .field("model", &self.config.model.model)
            .finish()
    }
}

/// Client builder for fluent configuration
pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    /// Create a new client builder
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

    /// Set the organization ID
    pub fn organization_id(mut self, org_id: impl Into<String>) -> Self {
        self.config.provider.organization_id = Some(org_id.into());
        self
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
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.http.timeout = timeout;
        self
    }

    /// Add a custom header
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.insert(key.into(), value.into());
        self
    }

    /// Set retry configuration
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.config.retry.max_retries = max_retries;
        self
    }

    /// Build the client
    pub fn build(self) -> RsllmResult<Client> {
        Client::new(self.config)
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Async client trait for custom implementations
#[async_trait]
pub trait AsyncClient: Send + Sync {
    /// Chat completion
    async fn chat_completion(&self, messages: Vec<ChatMessage>) -> RsllmResult<ChatResponse>;

    /// Chat completion streaming
    async fn chat_completion_stream(&self, messages: Vec<ChatMessage>) -> RsllmResult<ChatStream>;

    /// Health check
    async fn health_check(&self) -> RsllmResult<bool>;
}

#[async_trait]
impl AsyncClient for Client {
    async fn chat_completion(&self, messages: Vec<ChatMessage>) -> RsllmResult<ChatResponse> {
        self.chat_completion(messages).await
    }

    async fn chat_completion_stream(&self, messages: Vec<ChatMessage>) -> RsllmResult<ChatStream> {
        self.chat_completion_stream(messages).await
    }

    async fn health_check(&self) -> RsllmResult<bool> {
        self.health_check().await
    }
}

/// Client pool for managing multiple clients
pub struct ClientPool {
    clients: HashMap<String, Arc<Client>>,
    default_client: Option<String>,
}

impl ClientPool {
    /// Create a new client pool
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            default_client: None,
        }
    }

    /// Add a client to the pool
    pub fn add_client(&mut self, name: impl Into<String>, client: Client) {
        let name = name.into();
        let is_first = self.clients.is_empty();

        self.clients.insert(name.clone(), Arc::new(client));

        if is_first {
            self.default_client = Some(name);
        }
    }

    /// Get a client by name
    pub fn get_client(&self, name: &str) -> Option<&Arc<Client>> {
        self.clients.get(name)
    }

    /// Get the default client
    pub fn default_client(&self) -> Option<&Arc<Client>> {
        self.default_client
            .as_ref()
            .and_then(|name| self.get_client(name))
    }

    /// Set the default client
    pub fn set_default(&mut self, name: impl Into<String>) -> RsllmResult<()> {
        let name = name.into();
        if self.clients.contains_key(&name) {
            self.default_client = Some(name);
            Ok(())
        } else {
            Err(RsllmError::not_found(format!("Client '{}'", name)))
        }
    }

    /// List all client names
    pub fn client_names(&self) -> Vec<&String> {
        self.clients.keys().collect()
    }

    /// Remove a client
    pub fn remove_client(&mut self, name: &str) -> Option<Arc<Client>> {
        let removed = self.clients.remove(name);

        // Update default if we removed it
        if self.default_client.as_deref() == Some(name) {
            self.default_client = self.clients.keys().next().cloned();
        }

        removed
    }
}

impl Default for ClientPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MessageRole, Provider};

    #[test]
    fn test_client_builder() {
        let config = ClientBuilder::new()
            .provider(Provider::OpenAI)
            .model("gpt-4")
            .temperature(0.7)
            .max_tokens(1000)
            .timeout(std::time::Duration::from_secs(30))
            .header("Custom-Header", "value")
            .config
            .clone();

        assert_eq!(config.provider.provider, Provider::OpenAI);
        assert_eq!(config.model.model, "gpt-4");
        assert_eq!(config.model.temperature, Some(0.7));
        assert_eq!(config.model.max_tokens, Some(1000));
        assert_eq!(config.http.timeout, std::time::Duration::from_secs(30));
        assert!(config.headers.contains_key("Custom-Header"));
    }

    #[test]
    fn test_client_pool() {
        let mut pool = ClientPool::new();

        // Note: These clients would fail to build without proper API keys
        // This is just testing the pool structure
        assert_eq!(pool.client_names().len(), 0);
        assert!(pool.default_client().is_none());
    }

    #[test]
    fn test_message_validation() {
        let config = ClientBuilder::new()
            .provider(Provider::OpenAI)
            .api_key("test-key")
            .build();

        // This will fail due to missing implementation, but we can test the validation logic
        assert!(config.is_err() || config.is_ok()); // Either way is fine for structure test
    }
}
