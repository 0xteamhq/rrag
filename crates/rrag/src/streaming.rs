//! # RRAG Streaming System
//! 
//! Real-time streaming responses using Rust's async ecosystem.
//! Leverages tokio-stream and futures for efficient token streaming.

use crate::{RragError, RragResult};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Streaming response token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamToken {
    /// Token content
    pub content: String,
    
    /// Token type (text, tool_call, metadata, etc.)
    pub token_type: TokenType,
    
    /// Position in the stream
    pub position: usize,
    
    /// Whether this is the final token
    pub is_final: bool,
    
    /// Token metadata
    pub metadata: Option<serde_json::Value>,
}

impl StreamToken {
    pub fn text(content: impl Into<String>, position: usize) -> Self {
        Self {
            content: content.into(),
            token_type: TokenType::Text,
            position,
            is_final: false,
            metadata: None,
        }
    }

    pub fn tool_call(content: impl Into<String>, position: usize) -> Self {
        Self {
            content: content.into(),
            token_type: TokenType::ToolCall,
            position,
            is_final: false,
            metadata: None,
        }
    }

    pub fn final_token(position: usize) -> Self {
        Self {
            content: String::new(),
            token_type: TokenType::End,
            position,
            is_final: true,
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Token types for different streaming content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TokenType {
    /// Regular text content
    Text,
    
    /// Tool call information
    ToolCall,
    
    /// Tool result
    ToolResult,
    
    /// Metadata/system information
    Metadata,
    
    /// Stream end marker
    End,
    
    /// Error token
    Error,
}

/// Streaming response wrapper
pub struct StreamingResponse {
    stream: Pin<Box<dyn Stream<Item = RragResult<StreamToken>> + Send>>,
}

impl StreamingResponse {
    /// Create from a text string by splitting into tokens
    pub fn from_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let tokens: Vec<_> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| {
                Ok(StreamToken::text(format!("{} ", word), i))
            })
            .collect();

        // Add final token
        let mut tokens = tokens;
        let final_pos = tokens.len();
        tokens.push(Ok(StreamToken::final_token(final_pos)));

        let stream = futures::stream::iter(tokens);
        
        Self {
            stream: Box::pin(stream),
        }
    }

    /// Create from a token stream
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = RragResult<StreamToken>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
        }
    }

    /// Create from an async channel
    pub fn from_channel(receiver: mpsc::UnboundedReceiver<RragResult<StreamToken>>) -> Self {
        let stream = UnboundedReceiverStream::new(receiver);
        Self::from_stream(stream)
    }

    /// Collect all tokens into a single string
    pub async fn collect_text(mut self) -> RragResult<String> {
        let mut result = String::new();
        
        while let Some(token_result) = self.stream.next().await {
            match token_result? {
                token if token.token_type == TokenType::Text => {
                    result.push_str(&token.content);
                }
                token if token.is_final => break,
                _ => {} // Skip non-text tokens
            }
        }
        
        Ok(result.trim().to_string())
    }

    /// Filter tokens by type
    pub fn filter_by_type(self, token_type: TokenType) -> FilteredStream {
        FilteredStream {
            stream: self.stream,
            filter_type: token_type,
        }
    }

    /// Map tokens to a different type
    pub fn map_tokens<F, T>(self, f: F) -> MappedStream<T>
    where
        F: Fn(StreamToken) -> T + Send + 'static,
        T: Send + 'static,
    {
        let mapped_stream = self.stream.map(move |result| {
            result.map(&f)
        });

        MappedStream {
            stream: Box::pin(mapped_stream),
        }
    }
}

impl Stream for StreamingResponse {
    type Item = RragResult<StreamToken>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

/// Filtered stream that only yields specific token types
pub struct FilteredStream {
    stream: Pin<Box<dyn Stream<Item = RragResult<StreamToken>> + Send>>,
    filter_type: TokenType,
}

impl Stream for FilteredStream {
    type Item = RragResult<StreamToken>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(token))) => {
                    if token.token_type == self.filter_type || token.is_final {
                        return Poll::Ready(Some(Ok(token)));
                    }
                    // Continue polling for matching tokens
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Mapped stream that transforms tokens
pub struct MappedStream<T> {
    stream: Pin<Box<dyn Stream<Item = RragResult<T>> + Send>>,
}

impl<T> Stream for MappedStream<T> {
    type Item = RragResult<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

/// Token stream builder for creating custom streams
pub struct TokenStreamBuilder {
    sender: mpsc::UnboundedSender<RragResult<StreamToken>>,
    position: usize,
}

impl TokenStreamBuilder {
    /// Create a new token stream builder
    pub fn new() -> (Self, mpsc::UnboundedReceiver<RragResult<StreamToken>>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let builder = Self {
            sender,
            position: 0,
        };
        
        (builder, receiver)
    }

    /// Send a text token
    pub fn send_text(&mut self, content: impl Into<String>) -> RragResult<()> {
        let token = StreamToken::text(content, self.position);
        self.position += 1;
        
        self.sender
            .send(Ok(token))
            .map_err(|_| RragError::stream("token_builder", "Channel closed"))?;
            
        Ok(())
    }

    /// Send a tool call token
    pub fn send_tool_call(&mut self, content: impl Into<String>) -> RragResult<()> {
        let token = StreamToken::tool_call(content, self.position);
        self.position += 1;
        
        self.sender
            .send(Ok(token))
            .map_err(|_| RragError::stream("token_builder", "Channel closed"))?;
            
        Ok(())
    }

    /// Send an error token
    pub fn send_error(&mut self, error: RragError) -> RragResult<()> {
        self.sender
            .send(Err(error))
            .map_err(|_| RragError::stream("token_builder", "Channel closed"))?;
            
        Ok(())
    }

    /// Finalize the stream
    pub fn finish(self) -> RragResult<()> {
        let final_token = StreamToken::final_token(self.position);
        
        self.sender
            .send(Ok(final_token))
            .map_err(|_| RragError::stream("token_builder", "Channel closed"))?;
            
        // Close the channel
        drop(self.sender);
        
        Ok(())
    }
}

impl Default for TokenStreamBuilder {
    fn default() -> Self {
        let (builder, _) = Self::new();
        builder
    }
}

/// Convenience type alias for token streams
pub type TokenStream = StreamingResponse;

/// Utility functions for working with streams
pub mod stream_utils {
    use super::*;
    use std::time::Duration;

    /// Create a stream that emits tokens with a delay (for demo purposes)
    pub fn create_delayed_stream(
        text: impl Into<String>,
        delay: Duration,
    ) -> StreamingResponse {
        let text = text.into();
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        let stream = async_stream::stream! {
            for (i, word) in words.iter().enumerate() {
                tokio::time::sleep(delay).await;
                yield Ok(StreamToken::text(format!("{} ", word), i));
            }
            yield Ok(StreamToken::final_token(words.len()));
        };

        StreamingResponse::from_stream(stream)
    }

    /// Create a stream from multiple text chunks
    pub fn create_chunked_stream(chunks: Vec<String>) -> StreamingResponse {
        let stream = async_stream::stream! {
            for (i, chunk) in chunks.iter().enumerate() {
                yield Ok(StreamToken::text(chunk.clone(), i));
            }
            yield Ok(StreamToken::final_token(chunks.len()));
        };

        StreamingResponse::from_stream(stream)
    }

    /// Merge multiple streams into one
    pub async fn merge_streams(
        streams: Vec<StreamingResponse>,
    ) -> RragResult<StreamingResponse> {
        let (mut builder, receiver) = TokenStreamBuilder::new();

        tokio::spawn(async move {
            let mut position = 0;
            
            for mut stream in streams {
                while let Some(token_result) = stream.next().await {
                    match token_result {
                        Ok(mut token) => {
                            if !token.is_final {
                                token.position = position;
                                position += 1;
                                
                                if let Err(_) = builder.sender.send(Ok(token)) {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = builder.send_error(e);
                            break;
                        }
                    }
                }
            }
            
            let _ = builder.finish();
        });

        Ok(StreamingResponse::from_channel(receiver))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tokio_test;

    #[tokio::test]
    async fn test_streaming_response_from_text() {
        let response = StreamingResponse::from_text("Hello world test");
        let text = response.collect_text().await.unwrap();
        
        assert_eq!(text, "Hello world test");
    }

    #[tokio::test]
    async fn test_token_stream_builder() {
        let (mut builder, receiver) = TokenStreamBuilder::new();
        
        tokio::spawn(async move {
            builder.send_text("Hello").unwrap();
            builder.send_text("world").unwrap();
            builder.finish().unwrap();
        });
        
        let response = StreamingResponse::from_channel(receiver);
        let text = response.collect_text().await.unwrap();
        
        assert_eq!(text, "Hello world");
    }

    #[tokio::test]
    async fn test_filtered_stream() {
        let (mut builder, receiver) = TokenStreamBuilder::new();
        
        tokio::spawn(async move {
            builder.send_text("Hello").unwrap();
            builder.send_tool_call("tool_call").unwrap();
            builder.send_text("world").unwrap();
            builder.finish().unwrap();
        });
        
        let response = StreamingResponse::from_channel(receiver);
        let mut text_stream = response.filter_by_type(TokenType::Text);
        
        let mut text_tokens = Vec::new();
        while let Some(token_result) = text_stream.next().await {
            match token_result.unwrap() {
                token if token.token_type == TokenType::Text => {
                    text_tokens.push(token.content);
                }
                token if token.is_final => break,
                _ => {}
            }
        }
        
        assert_eq!(text_tokens, vec!["Hello ", "world "]);
    }

    #[tokio::test]
    async fn test_stream_utils_delayed() {
        use std::time::Duration;
        
        let start = std::time::Instant::now();
        let response = stream_utils::create_delayed_stream(
            "one two", 
            Duration::from_millis(10)
        );
        let text = response.collect_text().await.unwrap();
        let elapsed = start.elapsed();
        
        assert_eq!(text, "one two");
        assert!(elapsed >= Duration::from_millis(20)); // At least 2 delays
    }

    #[test]
    fn test_stream_token_creation() {
        let token = StreamToken::text("hello", 0);
        assert_eq!(token.content, "hello");
        assert_eq!(token.token_type, TokenType::Text);
        assert_eq!(token.position, 0);
        assert!(!token.is_final);
        
        let final_token = StreamToken::final_token(10);
        assert!(final_token.is_final);
        assert_eq!(final_token.token_type, TokenType::End);
    }
}