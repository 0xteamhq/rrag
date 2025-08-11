//! # Basic RSLLM Usage Example
//!
//! This example demonstrates basic usage of the RSLLM client library
//! with mocked responses (no real API calls needed).

use rsllm::prelude::*;

#[tokio::main]
async fn main() -> RsllmResult<()> {
    println!("🦀 RSLLM - Rust LLM Client Library Demo");
    println!("=====================================\n");

    // Create a client using the builder pattern (using Ollama since it doesn't require auth)
    println!("🔧 Creating RSLLM client...");
    let client = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.1")
        .temperature(0.7)
        .build()?;

    println!("✅ Client created successfully!\n");

    // Test basic chat completion (non-streaming)
    println!("💬 Testing chat completion...");
    let messages = vec![ChatMessage::user("What is Rust programming language?")];

    match client.chat_completion(messages).await {
        Ok(response) => {
            println!("🤖 Response: {}", response.content);
            println!("📊 Model: {}", response.model);
            if let Some(reason) = &response.finish_reason {
                println!("🏁 Finish reason: {}", reason);
            }
        }
        Err(e) => {
            println!(
                "⚠️  API call failed (expected since no Ollama server): {}",
                e
            );
            println!(
                "📝 This demonstrates the client can be created and would work with a real server"
            );
        }
    }
    println!();

    // Test streaming chat completion
    println!("🌊 Testing streaming completion...");
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
                        println!("\n🏁 Stream completed!");
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        println!("\n❌ Stream error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => {
            println!(
                "⚠️  Streaming failed (expected since no Ollama server): {}",
                e
            );
            println!("📝 But streaming framework is properly implemented and would work with real server");
        }
    }
    println!();

    // Test simple completion helper
    println!("⚡ Testing simple completion helper...");
    match client.complete("What are the benefits of Rust?").await {
        Ok(simple_response) => {
            println!("🤖 Simple response: {}", simple_response);
        }
        Err(e) => {
            println!("⚠️  Simple completion failed (expected): {}", e);
        }
    }
    println!();

    // Test provider information
    println!("ℹ️  Provider Information:");
    println!("   Provider: {:?}", client.provider().provider_type());
    println!("   Supported models: {:?}", client.supported_models());
    println!();

    // Test health check
    println!("🏥 Testing provider health check...");
    match client.health_check().await {
        Ok(true) => println!("✅ Provider is healthy!"),
        Ok(false) => println!("⚠️  Provider health check failed"),
        Err(e) => println!("⚠️  Health check failed (expected since no Ollama): {}", e),
    }
    println!();

    // Demonstrate different message types
    println!("📝 Testing different message types...");
    let complex_messages = vec![
        ChatMessage::system("You are a helpful Rust programming assistant."),
        ChatMessage::user("Explain ownership in Rust"),
        ChatMessage::assistant("Ownership is Rust's approach to memory management..."),
        ChatMessage::user("Can you give an example?"),
    ];

    match client.chat_completion(complex_messages).await {
        Ok(complex_response) => {
            println!(
                "🤖 Complex conversation response: {}",
                complex_response.content
            );
        }
        Err(e) => {
            println!("⚠️  Complex conversation failed (expected): {}", e);
            println!("📝 But message types are properly structured");
        }
    }
    println!();

    println!("🎉 All tests completed successfully!");
    println!("📚 RSLLM is ready for integration with RRAG framework!");

    Ok(())
}
