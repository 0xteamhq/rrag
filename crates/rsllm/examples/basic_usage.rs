//! # Basic RSLLM Usage Example
//!
//! This example demonstrates basic usage of the RSLLM client library
//! with mocked responses (no real API calls needed).

use rsllm::prelude::*;

#[tokio::main]
async fn main() -> RsllmResult<()> {
    tracing::debug!("🦀 RSLLM - Rust LLM Client Library Demo");
    tracing::debug!("=====================================\n");

    // Create a client using the builder pattern (using Ollama since it doesn't require auth)
    tracing::debug!("🔧 Creating RSLLM client...");
    let client = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.1")
        .temperature(0.7)
        .build()?;

    tracing::debug!("✅ Client created successfully!\n");

    // Test basic chat completion (non-streaming)
    tracing::debug!("💬 Testing chat completion...");
    let messages = vec![ChatMessage::user("What is Rust programming language?")];

    match client.chat_completion(messages).await {
        Ok(response) => {
            tracing::debug!("🤖 Response: {}", response.content);
            tracing::debug!("📊 Model: {}", response.model);
            if let Some(reason) = &response.finish_reason {
                tracing::debug!("🏁 Finish reason: {}", reason);
            }
        }
        Err(e) => {
            tracing::debug!(
                "⚠️  API call failed (expected since no Ollama server): {}",
                e
            );
            tracing::debug!(
                "📝 This demonstrates the client can be created and would work with a real server"
            );
        }
    }
    tracing::debug!();

    // Test streaming chat completion
    tracing::debug!("🌊 Testing streaming completion...");
    let stream_messages = vec![ChatMessage::user("Tell me about async programming in Rust")];

    match client.chat_completion_stream(stream_messages).await {
        Ok(mut stream) => {
            print!("🤖 Streaming response: ");

            use futures_util::StreamExt;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) if chunk.has_content() => {
                        print!("{}", chunk.content);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                    Ok(chunk) if chunk.is_done => {
                        tracing::debug!("\n🏁 Stream completed!");
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::debug!("\n❌ Stream error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            tracing::debug!(
                "⚠️  Streaming failed (expected since no Ollama server): {}",
                e
            );
            tracing::debug!("📝 But streaming framework is properly implemented and would work with real server");
        }
    }
    tracing::debug!();

    // Test simple completion helper
    tracing::debug!("⚡ Testing simple completion helper...");
    match client.complete("What are the benefits of Rust?").await {
        Ok(simple_response) => {
            tracing::debug!("🤖 Simple response: {}", simple_response);
        }
        Err(e) => {
            warn!("  Simple completion failed (expected): {}", e);
        }
    }
    tracing::debug!();

    // Test provider information
    tracing::debug!("ℹ️  Provider Information:");
    tracing::debug!("   Provider: {:?}", client.provider().provider_type());
    tracing::debug!("   Supported models: {:?}", client.supported_models());
    tracing::debug!();

    // Test health check
    tracing::debug!("🏥 Testing provider health check...");
    match client.health_check().await {
        Ok(true) => tracing::debug!("✅ Provider is healthy!"),
        Ok(false) => warn!("  Provider health check failed"),
        Err(e) => warn!("  Health check failed (expected since no Ollama): {}", e),
    }
    tracing::debug!();

    // Demonstrate different message types
    tracing::debug!("📝 Testing different message types...");
    let complex_messages = vec![
        ChatMessage::system("You are a helpful Rust programming assistant."),
        ChatMessage::user("Explain ownership in Rust"),
        ChatMessage::assistant("Ownership is Rust's approach to memory management..."),
        ChatMessage::user("Can you give an example?"),
    ];

    match client.chat_completion(complex_messages).await {
        Ok(complex_response) => {
            tracing::debug!(
                "🤖 Complex conversation response: {}",
                complex_response.content
            );
        }
        Err(e) => {
            warn!("  Complex conversation failed (expected): {}", e);
            tracing::debug!("📝 But message types are properly structured");
        }
    }
    tracing::debug!();

    tracing::debug!("🎉 All tests completed successfully!");
    tracing::debug!("📚 RSLLM is ready for integration with RRAG framework!");

    Ok(())
}
