//! # RSLLM Streaming Support
//! 
//! Streaming response handling with proper async Stream traits.
//! Supports real-time token streaming with backpressure and error handling.

use crate::{RsllmError, RsllmResult, StreamChunk, ChatResponse, CompletionResponse};
use futures_util::Stream;
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use futures_util::Future;

/// Type alias for chat streaming responses
pub type ChatStream = Pin<Box<dyn Stream<Item = RsllmResult<StreamChunk>> + Send>>;

/// Type alias for completion streaming responses  
pub type CompletionStream = Pin<Box<dyn Stream<Item = RsllmResult<StreamChunk>> + Send>>;

/// Stream collector for assembling complete responses from chunks
pin_project! {
    pub struct StreamCollector<S> {
        #[pin]
        stream: S,
        accumulated_content: String,
        model: Option<String>,
        finish_reason: Option<String>,
        usage: Option<crate::Usage>,
        metadata: std::collections::HashMap<String, serde_json::Value>,
        tool_calls: Vec<crate::ToolCall>,
        is_done: bool,
    }
}

impl<S> StreamCollector<S>
where
    S: Stream<Item = RsllmResult<StreamChunk>>,
{
    /// Create a new stream collector
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            accumulated_content: String::new(),
            model: None,
            finish_reason: None,
            usage: None,
            metadata: std::collections::HashMap::new(),
            tool_calls: Vec::new(),
            is_done: false,
        }
    }
    
    /// Collect all chunks into a complete chat response
    pub async fn collect_chat_response(mut self) -> RsllmResult<ChatResponse>
    where
        S: Unpin,
    {
        use futures_util::StreamExt;
        while let Some(chunk_result) = self.next().await {
            let _chunk = chunk_result?;
            // Process chunk - this updates internal state
        }
        
        let model = self.model.unwrap_or_else(|| "unknown".to_string());
        
        let mut response = ChatResponse::new(self.accumulated_content, model);
        
        if let Some(reason) = self.finish_reason {
            response = response.with_finish_reason(reason);
        }
        
        if let Some(usage) = self.usage {
            response = response.with_usage(usage);
        }
        
        if !self.tool_calls.is_empty() {
            response = response.with_tool_calls(self.tool_calls);
        }
        
        for (key, value) in self.metadata {
            response = response.with_metadata(key, value);
        }
        
        Ok(response)
    }
    
    /// Collect all chunks into a complete completion response
    pub async fn collect_completion_response(mut self) -> RsllmResult<CompletionResponse>
    where
        S: Unpin,
    {
        use futures_util::StreamExt;
        while let Some(chunk_result) = self.next().await {
            let _chunk = chunk_result?;
            // Process chunk - this updates internal state
        }
        
        let model = self.model.unwrap_or_else(|| "unknown".to_string());
        
        let mut response = CompletionResponse::new(self.accumulated_content, model);
        
        if let Some(reason) = self.finish_reason {
            response = response.with_finish_reason(reason);
        }
        
        if let Some(usage) = self.usage {
            response = response.with_usage(usage);
        }
        
        for (key, value) in self.metadata {
            response = response.with_metadata(key, value);
        }
        
        Ok(response)
    }
}

impl<S> Stream for StreamCollector<S>
where
    S: Stream<Item = RsllmResult<StreamChunk>>,
{
    type Item = RsllmResult<StreamChunk>;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        
        if *this.is_done {
            return Poll::Ready(None);
        }
        
        match this.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                // Update accumulated state
                if chunk.has_content() {
                    this.accumulated_content.push_str(&chunk.content);
                }
                
                if this.model.is_none() && !chunk.model.is_empty() {
                    *this.model = Some(chunk.model.clone());
                }
                
                if let Some(reason) = &chunk.finish_reason {
                    *this.finish_reason = Some(reason.clone());
                }
                
                if let Some(usage) = &chunk.usage {
                    *this.usage = Some(usage.clone());
                }
                
                // Merge metadata
                for (key, value) in &chunk.metadata {
                    this.metadata.insert(key.clone(), value.clone());
                }
                
                // Handle tool calls delta (simplified - would need proper delta merging)
                if let Some(_tool_calls_delta) = &chunk.tool_calls_delta {
                    // TODO: Implement proper tool call delta merging
                }
                
                if chunk.is_done {
                    *this.is_done = true;
                }
                
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                *this.is_done = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Stream adapter for rate limiting
pin_project! {
    pub struct RateLimitedStream<S> {
        #[pin]
        stream: S,
        delay: std::time::Duration,
        last_emit: Option<std::time::Instant>,
    }
}

impl<S> RateLimitedStream<S> {
    /// Create a new rate-limited stream
    pub fn new(stream: S, max_chunks_per_second: f64) -> Self {
        let delay = std::time::Duration::from_secs_f64(1.0 / max_chunks_per_second);
        Self {
            stream,
            delay,
            last_emit: None,
        }
    }
}

impl<S> Stream for RateLimitedStream<S>
where
    S: Stream<Item = RsllmResult<StreamChunk>>,
{
    type Item = S::Item;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        
        // Check if we need to delay
        if let Some(last) = this.last_emit {
            let elapsed = last.elapsed();
            if elapsed < *this.delay {
                let remaining = *this.delay - elapsed;
                
                // Set up a timer for the remaining delay
                let sleep = tokio::time::sleep(remaining);
                tokio::pin!(sleep);
                
                if sleep.as_mut().poll(cx).is_pending() {
                    return Poll::Pending;
                }
            }
        }
        
        match this.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                *this.last_emit = Some(std::time::Instant::now());
                Poll::Ready(Some(item))
            }
            other => other,
        }
    }
}

/// Stream adapter for filtering chunks
pin_project! {
    pub struct FilteredStream<S, F> {
        #[pin]
        stream: S,
        filter: F,
    }
}

impl<S, F> FilteredStream<S, F>
where
    F: Fn(&StreamChunk) -> bool,
{
    /// Create a new filtered stream
    pub fn new(stream: S, filter: F) -> Self {
        Self { stream, filter }
    }
}

impl<S, F> Stream for FilteredStream<S, F>
where
    S: Stream<Item = RsllmResult<StreamChunk>>,
    F: Fn(&StreamChunk) -> bool,
{
    type Item = S::Item;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        
        loop {
            match this.stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    if (this.filter)(&chunk) {
                        return Poll::Ready(Some(Ok(chunk)));
                    }
                    // Continue polling if chunk was filtered out
                }
                other => return other,
            }
        }
    }
}

/// Stream adapter for mapping chunks
pin_project! {
    pub struct MappedStream<S, F> {
        #[pin]
        stream: S,
        mapper: F,
    }
}

impl<S, F> MappedStream<S, F>
where
    F: Fn(StreamChunk) -> StreamChunk,
{
    /// Create a new mapped stream
    pub fn new(stream: S, mapper: F) -> Self {
        Self { stream, mapper }
    }
}

impl<S, F> Stream for MappedStream<S, F>
where
    S: Stream<Item = RsllmResult<StreamChunk>>,
    F: Fn(StreamChunk) -> StreamChunk,
{
    type Item = S::Item;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        
        match this.stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let mapped = (this.mapper)(chunk);
                Poll::Ready(Some(Ok(mapped)))
            }
            other => other,
        }
    }
}

/// Stream utilities
pub struct StreamUtils;

impl StreamUtils {
    /// Convert a vector of chunks into a stream
    pub fn from_chunks(chunks: Vec<StreamChunk>) -> ChatStream {
        let stream = tokio_stream::iter(chunks.into_iter().map(Ok));
        Box::pin(stream)
    }
    
    /// Create an empty stream
    pub fn empty() -> ChatStream {
        let stream = tokio_stream::empty();
        Box::pin(stream)
    }
    
    /// Create a stream that immediately returns an error
    pub fn error(error: RsllmError) -> ChatStream {
        use futures_util::stream;
        let stream = stream::once(async move { Err(error) });
        Box::pin(stream)
    }
    
    /// Collect stream into a vector of chunks
    pub async fn collect_chunks<S>(stream: S) -> RsllmResult<Vec<StreamChunk>>
    where
        S: Stream<Item = RsllmResult<StreamChunk>>,
    {
        tokio_stream::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .collect::<RsllmResult<Vec<_>>>()
    }
    
    /// Take only the first N chunks from a stream
    pub fn take<S>(stream: S, n: usize) -> impl Stream<Item = RsllmResult<StreamChunk>>
    where
        S: Stream<Item = RsllmResult<StreamChunk>>,
    {
        tokio_stream::StreamExt::take(stream, n)
    }
    
    /// Skip the first N chunks from a stream
    pub fn skip<S>(stream: S, n: usize) -> impl Stream<Item = RsllmResult<StreamChunk>>
    where
        S: Stream<Item = RsllmResult<StreamChunk>>,
    {
        tokio_stream::StreamExt::skip(stream, n)
    }
    
    /// Filter chunks based on a predicate
    pub fn filter<S, F>(stream: S, filter: F) -> FilteredStream<S, F>
    where
        S: Stream<Item = RsllmResult<StreamChunk>>,
        F: Fn(&StreamChunk) -> bool,
    {
        FilteredStream::new(stream, filter)
    }
    
    /// Map chunks with a function
    pub fn map<S, F>(stream: S, mapper: F) -> MappedStream<S, F>
    where
        S: Stream<Item = RsllmResult<StreamChunk>>,
        F: Fn(StreamChunk) -> StreamChunk,
    {
        MappedStream::new(stream, mapper)
    }
    
    /// Rate limit a stream
    pub fn rate_limit<S>(stream: S, max_chunks_per_second: f64) -> RateLimitedStream<S>
    where
        S: Stream<Item = RsllmResult<StreamChunk>>,
    {
        RateLimitedStream::new(stream, max_chunks_per_second)
    }
    
    /// Buffer chunks to reduce API calls (simplified implementation)
    pub async fn buffer<S>(
        mut stream: S,
        max_size: usize,
    ) -> RsllmResult<Vec<StreamChunk>>
    where
        S: Stream<Item = RsllmResult<StreamChunk>> + Unpin,
    {
        let mut chunks = Vec::new();
        let mut count = 0;
        
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk?);
            count += 1;
            
            if count >= max_size {
                break;
            }
        }
        
        Ok(chunks)
    }
}

/// Stream extension traits for additional functionality
pub trait RsllmStreamExt: Stream<Item = RsllmResult<StreamChunk>> + Sized {
    /// Collect stream into a complete chat response
    fn collect_chat_response(self) -> impl std::future::Future<Output = RsllmResult<ChatResponse>> + Send
    where
        Self: Send + Unpin,
    {
        StreamCollector::new(self).collect_chat_response()
    }
    
    /// Collect stream into a complete completion response
    fn collect_completion_response(self) -> impl std::future::Future<Output = RsllmResult<CompletionResponse>> + Send
    where
        Self: Send + Unpin,
    {
        StreamCollector::new(self).collect_completion_response()
    }
    
    /// Filter chunks that have content
    fn content_only(self) -> FilteredStream<Self, fn(&StreamChunk) -> bool> {
        FilteredStream::new(self, |chunk| chunk.has_content())
    }
    
    /// Filter out done chunks
    fn exclude_done(self) -> FilteredStream<Self, fn(&StreamChunk) -> bool> {
        FilteredStream::new(self, |chunk| !chunk.is_done)
    }
    
    /// Rate limit the stream
    fn rate_limit(self, max_chunks_per_second: f64) -> RateLimitedStream<Self> {
        RateLimitedStream::new(self, max_chunks_per_second)
    }
}

impl<S> RsllmStreamExt for S where S: Stream<Item = RsllmResult<StreamChunk>> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MessageRole, StreamChunk};
    
    #[tokio::test]
    async fn test_stream_collector() {
        let chunks = vec![
            StreamChunk::delta("Hello", "gpt-4").with_role(MessageRole::Assistant),
            StreamChunk::delta(" world", "gpt-4"),
            StreamChunk::done("gpt-4").with_finish_reason("stop"),
        ];
        
        let stream = StreamUtils::from_chunks(chunks);
        let response = stream.collect_chat_response().await.unwrap();
        
        assert_eq!(response.content, "Hello world");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.finish_reason, Some("stop".to_string()));
    }
    
    #[tokio::test]
    async fn test_filter_stream() {
        let chunks = vec![
            StreamChunk::delta("Hello", "gpt-4"),
            StreamChunk::new("", "gpt-4", false, false), // Empty chunk
            StreamChunk::delta(" world", "gpt-4"),
        ];
        
        let stream = StreamUtils::from_chunks(chunks);
        use futures_util::StreamExt;
        let mut filtered_stream = stream.content_only();
        let mut filtered_chunks = Vec::new();
        while let Some(chunk) = filtered_stream.next().await {
            filtered_chunks.push(chunk.unwrap());
        }
        
        assert_eq!(filtered_chunks.len(), 2);
        assert_eq!(filtered_chunks[0].content, "Hello");
        assert_eq!(filtered_chunks[1].content, " world");
    }
    
    #[tokio::test]
    async fn test_map_stream() {
        let chunks = vec![
            StreamChunk::delta("hello", "gpt-4"),
            StreamChunk::delta(" world", "gpt-4"),
        ];
        
        let stream = StreamUtils::from_chunks(chunks);
        let mapped_stream = StreamUtils::map(stream, |mut chunk| {
            chunk.content = chunk.content.to_uppercase();
            chunk
        });
        
        let collected = StreamUtils::collect_chunks(mapped_stream).await.unwrap();
        
        assert_eq!(collected[0].content, "HELLO");
        assert_eq!(collected[1].content, " WORLD");
    }
}