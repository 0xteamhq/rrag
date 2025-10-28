//! # RSLLM Provider Abstraction
//!
//! Multi-provider support for different LLM APIs with unified interface.
//! Supports OpenAI, Claude (Anthropic), Ollama, and custom providers.

use crate::{ChatMessage, ChatResponse, RsllmError, RsllmResult, StreamChunk};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use url::Url;

/// Normalize URL to ensure it has a trailing slash for proper path joining
/// This allows users to provide URLs with or without trailing slashes
fn normalize_base_url(url: &Url) -> Url {
    let url_str = url.as_str();
    if url_str.ends_with('/') {
        url.clone()
    } else {
        // Add trailing slash
        format!("{}/", url_str)
            .parse()
            .unwrap_or_else(|_| url.clone())
    }
}

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    /// OpenAI (GPT models)
    OpenAI,
    /// Anthropic Claude
    Claude,
    /// Ollama (local models)
    Ollama,
}

impl Provider {
    /// Get the default base URL for this provider
    pub fn default_base_url(&self) -> Url {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1/".parse().unwrap(),
            Provider::Claude => "https://api.anthropic.com/v1/".parse().unwrap(),
            Provider::Ollama => "http://localhost:11434/api/".parse().unwrap(),
        }
    }

    /// Get the default models for this provider
    pub fn default_models(&self) -> Vec<&'static str> {
        match self {
            Provider::OpenAI => vec![
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-instruct",
            ],
            Provider::Claude => vec![
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            Provider::Ollama => vec![
                "llama3.1",
                "llama3.1:70b",
                "llama3.1:405b",
                "mistral",
                "codellama",
                "vicuna",
            ],
        }
    }

    /// Get the recommended model for this provider
    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::OpenAI => "gpt-4o-mini",
            Provider::Claude => "claude-3-5-haiku-20241022",
            Provider::Ollama => "llama3.1",
        }
    }

    /// Check if this provider supports streaming
    pub fn supports_streaming(&self) -> bool {
        match self {
            Provider::OpenAI => true,
            Provider::Claude => true,
            Provider::Ollama => true,
        }
    }

    /// Check if this provider requires authentication
    pub fn requires_auth(&self) -> bool {
        match self {
            Provider::OpenAI => true,
            Provider::Claude => true,
            Provider::Ollama => false, // Local deployment typically doesn't need auth
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provider::OpenAI => write!(f, "openai"),
            Provider::Claude => write!(f, "claude"),
            Provider::Ollama => write!(f, "ollama"),
        }
    }
}

impl FromStr for Provider {
    type Err = RsllmError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" | "gpt" => Ok(Provider::OpenAI),
            "claude" | "anthropic" => Ok(Provider::Claude),
            "ollama" => Ok(Provider::Ollama),
            _ => Err(RsllmError::configuration(format!(
                "Unknown provider: {}",
                s
            ))),
        }
    }
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type
    pub provider: Provider,

    /// API key (if required)
    pub api_key: Option<String>,

    /// Base URL (if custom)
    pub base_url: Option<Url>,

    /// Organization ID (for providers that support it)
    pub organization_id: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider: Provider::OpenAI,
            api_key: None,
            base_url: None,
            organization_id: None,
        }
    }
}

/// Core provider trait for LLM interactions
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Provider name/identifier
    fn name(&self) -> &str;

    /// Provider type
    fn provider_type(&self) -> Provider;

    /// Supported models
    fn supported_models(&self) -> Vec<String>;

    /// Health check
    async fn health_check(&self) -> RsllmResult<bool>;

    /// Chat completion (non-streaming)
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<ChatResponse>;

    /// Chat completion (streaming)
    async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<Box<dyn futures_util::Stream<Item = RsllmResult<StreamChunk>> + Send + Unpin>>;

    /// Chat completion with tool calling support
    async fn chat_completion_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<crate::tools::ToolDefinition>,
        model: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<ChatResponse> {
        // Default implementation: call without tools (fallback for providers without tool support)
        let _ = tools; // Suppress unused warning
        self.chat_completion(messages, model, temperature, max_tokens)
            .await
    }
}

/// OpenAI provider implementation
#[cfg(feature = "openai")]
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: Url,
    organization_id: Option<String>,
}

#[cfg(feature = "openai")]
impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(
        api_key: String,
        base_url: Option<Url>,
        organization_id: Option<String>,
    ) -> RsllmResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                RsllmError::configuration_with_source("Failed to create HTTP client", e)
            })?;

        let base = base_url.unwrap_or_else(|| Provider::OpenAI.default_base_url());
        let normalized_base_url = normalize_base_url(&base);

        Ok(Self {
            client,
            api_key,
            base_url: normalized_base_url,
            organization_id,
        })
    }

    /// Build request headers
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();

        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        if let Some(org_id) = &self.organization_id {
            headers.insert("OpenAI-Organization", org_id.parse().unwrap());
        }

        headers
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "OpenAI"
    }

    fn provider_type(&self) -> Provider {
        Provider::OpenAI
    }

    fn supported_models(&self) -> Vec<String> {
        Provider::OpenAI
            .default_models()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    async fn health_check(&self) -> RsllmResult<bool> {
        let url = self.base_url.join("models")?;
        let response = self
            .client
            .get(url)
            .headers(self.build_headers())
            .send()
            .await?;

        Ok(response.status().is_success())
    }

    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<ChatResponse> {
        let url = self.base_url.join("chat/completions")?;

        let mut request_body = serde_json::json!({
            "model": model.unwrap_or(Provider::OpenAI.default_model()),
            "messages": messages,
        });

        if let Some(temp) = temperature {
            request_body["temperature"] = temp.into();
        }

        if let Some(max_tokens) = max_tokens {
            request_body["max_tokens"] = max_tokens.into();
        }

        let response = self
            .client
            .post(url)
            .headers(self.build_headers())
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RsllmError::api(
                "OpenAI",
                format!("API request failed: {}", error_text),
                status.as_str(),
            ));
        }

        let response_data: serde_json::Value = response.json().await?;

        // Extract the response content
        let content = response_data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(
            ChatResponse::new(content, model.unwrap_or(Provider::OpenAI.default_model()))
                .with_finish_reason("stop"),
        )
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<String>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> RsllmResult<Box<dyn futures_util::Stream<Item = RsllmResult<StreamChunk>> + Send + Unpin>>
    {
        use futures_util::stream;

        // For now, implement a simple mock stream
        // In production, this would handle Server-Sent Events (SSE) from OpenAI
        let _url = self.base_url.join("chat/completions")?;

        let model_name = model.unwrap_or_else(|| Provider::OpenAI.default_model().to_string());
        let mut _request_body = serde_json::json!({
            "model": &model_name,
            "messages": messages,
            "stream": true,
        });

        if let Some(temp) = temperature {
            _request_body["temperature"] = temp.into();
        }

        if let Some(max_tokens) = max_tokens {
            _request_body["max_tokens"] = max_tokens.into();
        }

        // Mock streaming response
        let chunks = vec![
            "Hello",
            " there!",
            " This",
            " is",
            " a",
            " streaming",
            " response",
            " from",
            " OpenAI.",
        ];

        let stream = stream::iter(chunks.into_iter().enumerate().map(move |(i, chunk)| {
            let _ = tokio::time::sleep(std::time::Duration::from_millis(100));

            if i == 8 {
                // Last chunk
                Ok(StreamChunk::done(&model_name).with_finish_reason("stop"))
            } else {
                Ok(StreamChunk::delta(chunk, &model_name))
            }
        }));

        Ok(Box::new(stream))
    }
}

/// Ollama provider implementation  
#[cfg(feature = "ollama")]
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: Url,
}

#[cfg(feature = "ollama")]
impl OllamaProvider {
    /// Create a new Ollama provider
    pub fn new(base_url: Option<Url>) -> RsllmResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60)) // Ollama can be slower
            .build()
            .map_err(|e| {
                RsllmError::configuration_with_source("Failed to create HTTP client", e)
            })?;

        let base = base_url.unwrap_or_else(|| Provider::Ollama.default_base_url());
        let normalized_base_url = normalize_base_url(&base);

        Ok(Self {
            client,
            base_url: normalized_base_url,
        })
    }
}

#[cfg(feature = "ollama")]
#[async_trait]
impl LLMProvider for OllamaProvider {
    fn name(&self) -> &str {
        "Ollama"
    }

    fn provider_type(&self) -> Provider {
        Provider::Ollama
    }

    fn supported_models(&self) -> Vec<String> {
        Provider::Ollama
            .default_models()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    async fn health_check(&self) -> RsllmResult<bool> {
        let url = self.base_url.join("tags")?;
        let response = self.client.get(url).send().await?;
        Ok(response.status().is_success())
    }

    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
        temperature: Option<f32>,
        _max_tokens: Option<u32>,
    ) -> RsllmResult<ChatResponse> {
        let url = self.base_url.join("chat")?;

        let mut request_body = serde_json::json!({
            "model": model.unwrap_or(Provider::Ollama.default_model()),
            "messages": messages,
            "stream": false,
        });

        if let Some(temp) = temperature {
            request_body["options"] = serde_json::json!({
                "temperature": temp
            });
        }

        let response = self.client.post(url).json(&request_body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RsllmError::api(
                "Ollama",
                format!("API request failed: {}", error_text),
                status.as_str(),
            ));
        }

        let response_data: serde_json::Value = response.json().await?;

        let content = response_data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(
            ChatResponse::new(content, model.unwrap_or(Provider::Ollama.default_model()))
                .with_finish_reason("stop"),
        )
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<String>,
        temperature: Option<f32>,
        _max_tokens: Option<u32>,
    ) -> RsllmResult<Box<dyn futures_util::Stream<Item = RsllmResult<StreamChunk>> + Send + Unpin>>
    {
        use futures_util::stream;

        // Mock streaming response for Ollama
        let _url = self.base_url.join("chat")?;

        let model_name = model.unwrap_or_else(|| Provider::Ollama.default_model().to_string());
        let mut _request_body = serde_json::json!({
            "model": &model_name,
            "messages": messages,
            "stream": true,
        });

        if let Some(temp) = temperature {
            _request_body["options"] = serde_json::json!({
                "temperature": temp
            });
        }

        // Mock streaming response
        let chunks = vec![
            "This",
            " is",
            " a",
            " response",
            " from",
            " Ollama",
            " running",
            " locally.",
        ];

        let stream = stream::iter(chunks.into_iter().enumerate().map(move |(i, chunk)| {
            let _ = tokio::time::sleep(std::time::Duration::from_millis(150));

            if i == 7 {
                // Last chunk
                Ok(StreamChunk::done(&model_name).with_finish_reason("stop"))
            } else {
                Ok(StreamChunk::delta(chunk, &model_name))
            }
        }));

        Ok(Box::new(stream))
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<crate::tools::ToolDefinition>,
        model: Option<&str>,
        temperature: Option<f32>,
        _max_tokens: Option<u32>,
    ) -> RsllmResult<ChatResponse> {
        let url = self.base_url.join("chat")?;

        // Build tools in Ollama/OpenAI format
        let tools_json: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                })
            })
            .collect();

        let mut request_body = serde_json::json!({
            "model": model.unwrap_or(Provider::Ollama.default_model()),
            "messages": messages,
            "stream": false,
            "tools": tools_json,
        });

        if let Some(temp) = temperature {
            request_body["options"] = serde_json::json!({
                "temperature": temp
            });
        }

        let response = self.client.post(url).json(&request_body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(RsllmError::api(
                "Ollama",
                format!("API request failed: {}", error_text),
                status.as_str(),
            ));
        }

        let response_data: serde_json::Value = response.json().await?;

        let content = response_data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Parse tool calls if present
        let tool_calls = if let Some(calls_array) = response_data["message"]["tool_calls"].as_array() {
            let parsed_calls: Vec<crate::message::ToolCall> = calls_array
                .iter()
                .enumerate()
                .filter_map(|(idx, call)| {
                    let function_name = call["function"]["name"].as_str()?;

                    // Ollama returns arguments as an object, sometimes with string values
                    // Convert string numbers to actual numbers for compatibility
                    let mut arguments = call["function"]["arguments"].clone();
                    if let serde_json::Value::Object(ref mut args_obj) = arguments {
                        for (_key, value) in args_obj.iter_mut() {
                            if let serde_json::Value::String(s) = value {
                                // Try to parse as number
                                if let Ok(num) = s.parse::<f64>() {
                                    *value = serde_json::json!(num);
                                } else if let Ok(int_num) = s.parse::<i64>() {
                                    *value = serde_json::json!(int_num);
                                }
                            }
                        }
                    }

                    // Ollama doesn't provide an ID, so generate one
                    let id = call["id"]
                        .as_str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("call_{}", idx));

                    Some(crate::message::ToolCall {
                        id,
                        call_type: crate::message::ToolCallType::Function,
                        function: crate::message::ToolFunction {
                            name: function_name.to_string(),
                            arguments,
                        },
                    })
                })
                .collect();

            if parsed_calls.is_empty() {
                None
            } else {
                Some(parsed_calls)
            }
        } else {
            None
        };

        let mut response = ChatResponse::new(content, model.unwrap_or(Provider::Ollama.default_model()))
            .with_finish_reason("stop");

        if let Some(calls) = tool_calls {
            response = response.with_tool_calls(calls);
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_base_url_without_trailing_slash() {
        let url = Url::parse("http://localhost:11434/api").unwrap();
        let normalized = normalize_base_url(&url);
        assert_eq!(normalized.as_str(), "http://localhost:11434/api/");
    }

    #[test]
    fn test_normalize_base_url_with_trailing_slash() {
        let url = Url::parse("http://localhost:11434/api/").unwrap();
        let normalized = normalize_base_url(&url);
        assert_eq!(normalized.as_str(), "http://localhost:11434/api/");
    }

    #[test]
    fn test_normalize_base_url_complex() {
        let url = Url::parse("https://api.openai.com/v1").unwrap();
        let normalized = normalize_base_url(&url);
        assert_eq!(normalized.as_str(), "https://api.openai.com/v1/");
    }

    #[test]
    fn test_url_join_after_normalization() {
        // Test that after normalization, joining works correctly
        let url_without_slash = Url::parse("http://localhost:11434/api").unwrap();
        let normalized = normalize_base_url(&url_without_slash);
        let joined = normalized.join("chat").unwrap();
        assert_eq!(joined.as_str(), "http://localhost:11434/api/chat");

        let url_with_slash = Url::parse("http://localhost:11434/api/").unwrap();
        let normalized2 = normalize_base_url(&url_with_slash);
        let joined2 = normalized2.join("chat").unwrap();
        assert_eq!(joined2.as_str(), "http://localhost:11434/api/chat");
    }
}
