# RSLLM - Rust LLM Client Library

[![Crates.io](https://img.shields.io/crates/v/rsllm.svg)](https://crates.io/crates/rsllm)
[![Documentation](https://docs.rs/rsllm/badge.svg)](https://docs.rs/rsllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RSLLM** is a Rust-native client library for Large Language Models with multi-provider support, streaming capabilities, and type-safe interfaces.

## 🚀 Features

- **🤖 Multi-Provider Support**: OpenAI, Anthropic Claude, Ollama, and more
- **⚡ Streaming Responses**: Real-time token streaming with async iterators
- **🛡️ Type Safety**: Compile-time guarantees for API contracts
- **📊 Memory Efficient**: Zero-copy operations where possible
- **🔌 Easy Integration**: Seamless integration with RAG frameworks like RRAG
- **⚙️ Configurable**: Flexible configuration with builder patterns
- **🌊 Async-First**: Built around async/await from the ground up

## 🏗️ Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│    RSLLM        │───▶│   LLM Provider  │
│   (RRAG, etc)   │    │    Client       │    │  (OpenAI/etc)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streaming     │◀───│   Provider      │◀───│    HTTP/API     │
│   Response      │    │   Abstraction   │    │    Transport    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

Add RSLLM to your `Cargo.toml`:

```toml
[dependencies]
rsllm = "0.1"
tokio = { version = "1.0", features = ["full"] }
```

### Basic Chat Completion

```rust
use rsllm::{Client, Provider, ChatMessage, MessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with OpenAI provider
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("your-api-key")
        .model("gpt-4")
        .build()?;

    // Simple chat completion
    let messages = vec![
        ChatMessage::new(MessageRole::User, "What is Rust?")
    ];

    let response = client.chat_completion(messages).await?;
    tracing::debug!("Response: {}", response.content);

    Ok(())
}
```

### Streaming Responses

```rust
use rsllm::{Client, Provider, ChatMessage, MessageRole};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("your-api-key")
        .model("gpt-4")
        .build()?;

    let messages = vec![
        ChatMessage::new(MessageRole::User, "Tell me a story")
    ];

    let mut stream = client.chat_completion_stream(messages).await?;

    while let Some(chunk) = stream.next().await {
        tracing::debug!("{}", chunk?.content);
    }

    Ok(())
}
```

### Multiple Providers

```rust
use rsllm::{Client, Provider};

// OpenAI
let openai_client = Client::builder()
    .provider(Provider::OpenAI)
    .api_key("openai-api-key")
    .model("gpt-4")
    .build()?;

// Anthropic Claude
let claude_client = Client::builder()
    .provider(Provider::Claude)
    .api_key("claude-api-key")
    .model("claude-3-sonnet")
    .build()?;

// Local Ollama
let ollama_client = Client::builder()
    .provider(Provider::Ollama)
    .base_url("http://localhost:11434")
    .model("llama3.1")
    .build()?;
```

## 🔧 Configuration

RSLLM supports extensive configuration options:

```rust
use rsllm::{Client, Provider, ClientConfig};
use std::time::Duration;

let client = Client::builder()
    .provider(Provider::OpenAI)
    .api_key("your-api-key")
    .model("gpt-4")
    .base_url("https://api.openai.com/v1")
    .timeout(Duration::from_secs(60))
    .max_tokens(4096)
    .temperature(0.7)
    .build()?;
```

### Environment Variables

RSLLM supports configuration through environment variables, perfect for CI/CD pipelines, different deployment environments, and custom/self-hosted endpoints:

```rust
use rsllm::Client;

// Load configuration from environment variables
let client = Client::from_env()?;
```

#### Supported Environment Variables

**Provider Configuration:**

- `RSLLM_PROVIDER` - Provider name (openai, claude, ollama)
- `RSLLM_API_KEY` - API key for the provider

**Base URL Configuration (supports custom/self-hosted endpoints):**

- `RSLLM_BASE_URL` - Generic base URL (works for any provider)
- `RSLLM_OPENAI_BASE_URL` - OpenAI-specific base URL (overrides generic)
- `RSLLM_OLLAMA_BASE_URL` - Ollama-specific base URL (overrides generic)
- `RSLLM_CLAUDE_BASE_URL` - Claude-specific base URL (overrides generic)

**Model Configuration (supports custom models and fine-tuned models):**

- `RSLLM_MODEL` - Generic model name
- `RSLLM_OPENAI_MODEL` - OpenAI-specific model (overrides generic)
- `RSLLM_OLLAMA_MODEL` - Ollama-specific model (overrides generic)
- `RSLLM_CLAUDE_MODEL` - Claude-specific model (overrides generic)

**Other Settings:**

- `RSLLM_TEMPERATURE` - Temperature setting (0.0 to 2.0)
- `RSLLM_MAX_TOKENS` - Maximum tokens to generate
- `RSLLM_TIMEOUT` - Request timeout in seconds

**Example `.env` file:**

```bash
# Using Ollama with custom model
RSLLM_PROVIDER=ollama
RSLLM_OLLAMA_BASE_URL=http://localhost:11434/api
RSLLM_OLLAMA_MODEL=llama3.2:3b
RSLLM_TEMPERATURE=0.7

# Or using a self-hosted OpenAI-compatible endpoint
RSLLM_PROVIDER=openai
RSLLM_OPENAI_BASE_URL=https://my-custom-llm.example.com/v1
RSLLM_OPENAI_MODEL=my-fine-tuned-gpt-4
RSLLM_API_KEY=my-custom-api-key
```

**Key Features:**

- ✅ **Custom Models Supported**: Use any model name, not limited to predefined lists
- ✅ **Custom Endpoints**: Point to self-hosted or custom LLM endpoints
- ✅ **URL Flexibility**: Base URLs work with or without trailing slashes
- ✅ **Provider-Specific Priority**: Provider-specific env vars override generic ones

## 🌟 Supported Providers

| Provider         | Status | Models                         | Streaming |
| ---------------- | ------ | ------------------------------ | --------- |
| OpenAI           | ✅     | GPT-4, GPT-3.5                 | ✅        |
| Anthropic Claude | ✅     | Claude-3 (Sonnet, Opus, Haiku) | ✅        |
| Ollama           | ✅     | Llama, Mistral, CodeLlama      | ✅        |
| Azure OpenAI     | 🚧     | GPT-4, GPT-3.5                 | 🚧        |
| Cohere           | 📝     | Command                        | 📝        |
| Google Gemini    | 📝     | Gemini Pro                     | 📝        |

Legend: ✅ Supported | 🚧 In Progress | 📝 Planned

## 📖 Documentation

- [API Documentation](https://docs.rs/rsllm) - Complete API reference
- [Examples](examples/) - Working code examples
- [RRAG Integration](https://github.com/0xteamhq/rrag) - RAG framework integration

## 🔧 Feature Flags

```toml
[dependencies.rsllm]
version = "0.1"
features = [
    "openai",        # OpenAI provider support
    "claude",        # Anthropic Claude support
    "ollama",        # Ollama local model support
    "streaming",     # Streaming response support
    "json-schema",   # JSON schema support for structured outputs
]
```

## 🤝 Integration with RRAG

RSLLM is designed to work seamlessly with the [RRAG framework](https://github.com/0xteamhq/rrag):

```rust
use rrag::prelude::*;
use rsllm::Client;

let llm_client = Client::builder()
    .provider(Provider::OpenAI)
    .api_key("your-api-key")
    .build()?;

let rag_system = RragSystemBuilder::new()
    .with_llm_client(llm_client)
    .build()
    .await?;
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](../../CONTRIBUTING.md) for details.

---

**Part of the [RRAG](https://github.com/0xteamhq/rrag) ecosystem - Build powerful RAG applications with Rust.**
