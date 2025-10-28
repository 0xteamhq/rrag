//! # RSLLM - Rust LLM Client Library
//!
//! **RSLLM** is a Rust-native client library for Large Language Models with multi-provider
//! support, streaming capabilities, and type-safe interfaces.
//!
//! ## Design Philosophy
//!
//! RSLLM embraces Rust's core principles:
//! - **Type Safety**: Compile-time guarantees for API contracts
//! - **Memory Safety**: Zero-copy operations where possible  
//! - **Async-First**: Built around async/await and streaming
//! - **Multi-Provider**: Unified interface for OpenAI, Claude, Ollama, etc.
//! - **Composable**: Easy integration with frameworks like RRAG
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Application   │───▶│    RSLLM        │───▶│   LLM Provider  │
//! │   (RRAG, etc)   │    │    Client       │    │  (OpenAI/etc)   │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!                                 │
//!                                 ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   Streaming     │◀───│   Provider      │◀───│    HTTP/API     │
//! │   Response      │    │   Abstraction   │    │    Transport    │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rsllm::{Client, Provider, ChatMessage, MessageRole};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client with OpenAI provider
//!     let client = Client::builder()
//!         .provider(Provider::OpenAI)
//!         .api_key("your-api-key")
//!         .model("gpt-4")
//!         .build()?;
//!     
//!     // Simple chat completion
//!     let messages = vec![
//!         ChatMessage::new(MessageRole::User, "What is Rust?")
//!     ];
//!     
//!     let response = client.chat_completion(messages).await?;
//!     println!("Response: {}", response.content);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Streaming Example
//!
//! ```rust,no_run
//! use rsllm::{Client, Provider, ChatMessage, MessageRole};
//! use futures_util::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::builder()
//!         .provider(Provider::OpenAI)
//!         .api_key("your-api-key")
//!         .build()?;
//!     
//!     let messages = vec![
//!         ChatMessage::new(MessageRole::User, "Tell me a story")
//!     ];
//!     
//!     let mut stream = client.chat_completion_stream(messages).await?;
//!     
//!     while let Some(chunk) = stream.next().await {
//!         match chunk? {
//!             chunk if chunk.is_delta() => {
//!                 print!("{}", chunk.content);
//!             }
//!             chunk if chunk.is_done() => {
//!                 println!("\n[DONE]");
//!                 break;
//!             }
//!             _ => {}
//!         }
//!     }
//!     
//!     Ok(())
//! }
//! ```

// Core modules
pub mod client;
pub mod config;
pub mod error;
pub mod message;
pub mod provider;
pub mod response;
pub mod streaming;
pub mod tools;

// Re-export proc macros
#[cfg(feature = "macros")]
pub use rsllm_macros::{arg, context, tool};

// Re-exports for convenience
pub use client::{Client, ClientBuilder};
pub use config::{ClientConfig, ModelConfig};
pub use error::{RsllmError, RsllmResult};
pub use message::{ChatMessage, MessageContent, MessageRole, ToolCall};
pub use provider::{LLMProvider, Provider, ProviderConfig};
pub use response::{ChatResponse, CompletionResponse, EmbeddingResponse, StreamChunk, Usage};
pub use streaming::{ChatStream, CompletionStream};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Framework name
pub const NAME: &str = "RSLLM";

/// Framework description
pub const DESCRIPTION: &str = "Rust LLM Client Library";

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        ChatMessage, ChatResponse, ChatStream, Client, ClientBuilder, ClientConfig,
        CompletionResponse, CompletionStream, EmbeddingResponse, LLMProvider, MessageContent,
        MessageRole, ModelConfig, Provider, ProviderConfig, RsllmError, RsllmResult, StreamChunk,
        ToolCall, Usage,
    };

    // External dependencies commonly used
    pub use async_trait::async_trait;
    pub use futures_util::{Stream, StreamExt};
    pub use serde::{Deserialize, Serialize};
    pub use tokio;
}
