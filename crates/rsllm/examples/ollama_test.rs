//! Direct Ollama Test - Verify RSLLM can communicate with Ollama
//!
//! Run with: cargo run -p rsllm --example ollama_test --features ollama

use rsllm::prelude::*;
use tracing::{error, warn};

#[tokio::main]
async fn main() -> RsllmResult<()> {
    tracing::debug!("🦀 RSLLM + Ollama Direct Test");
    tracing::debug!("==============================\n");

    // Create client pointing to Ollama
    tracing::debug!("🔧 Creating RSLLM client for Ollama...");

    let client = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")? // Testing with trailing slash
        .temperature(0.7)
        .build()?;

    tracing::debug!("✅ Client created successfully!\n");

    // Test 1: Simple completion
    tracing::debug!("💬 Test 1: Simple Chat Completion");
    tracing::debug!("Question: What is Rust?");

    let messages = vec![ChatMessage::user(
        "What is Rust programming language? Answer in one sentence.",
    )];

    match client.chat_completion(messages).await {
        Ok(response) => {
            tracing::debug!("✅ Response received!");
            tracing::debug!("🤖 Answer: {}", response.content);
            tracing::debug!("📊 Model: {}", response.model);
            if let Some(reason) = &response.finish_reason {
                tracing::debug!("🏁 Finish reason: {}", reason);
            }
        }
        Err(e) => {
            error!(" Test failed: {}", e);
            warn!("  Make sure Ollama is running and the model is available");
            tracing::debug!("   Run: ollama serve");
            tracing::debug!("   Run: ollama pull llama3.2:3b");
            return Ok(());
        }
    }

    // Test 2: Streaming
    tracing::debug!("💬 Test 2: Streaming Chat Completion");
    tracing::debug!("Question: Explain ownership in Rust");

    let stream_messages = vec![ChatMessage::user(
        "Explain Rust's ownership concept in 2 sentences.",
    )];

    match client.chat_completion_stream(stream_messages).await {
        Ok(mut stream) => {
            tracing::debug!("🤖 Streaming: ");

            use futures_util::StreamExt;
            let mut full_response = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) if chunk.has_content() => {
                        tracing::debug!("{}", chunk.content);
                        full_response.push_str(&chunk.content);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                    Ok(chunk) if chunk.is_done => {
                        tracing::debug!("\n✅ Streaming completed!");
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
            error!(" Streaming test failed: {}", e);
        }
    }

    // Test 3: Multi-turn conversation
    tracing::debug!("💬 Test 3: Multi-turn Conversation");

    let conversation = vec![
        ChatMessage::system("You are a helpful Rust programming assistant."),
        ChatMessage::user("What is borrowing in Rust?"),
        ChatMessage::assistant("Borrowing in Rust allows you to reference data without taking ownership, enabling safe concurrent access."),
        ChatMessage::user("Give me a code example"),
    ];

    match client.chat_completion(conversation).await {
        Ok(response) => {
            tracing::debug!("✅ Multi-turn conversation works!");
            tracing::debug!("🤖 Response: {}", response.content);
        }
        Err(e) => {
            error!(" Multi-turn test failed: {}", e);
        }
    }

    // Test 4: Health check
    tracing::debug!("💬 Test 4: Health Check");
    match client.health_check().await {
        Ok(true) => tracing::debug!("✅ Ollama is healthy!\n"),
        Ok(false) => warn!("  Ollama health check returned false\n"),
        Err(e) => error!(" Health check failed: {}\n", e),
    }

    tracing::debug!("🎉 All tests completed!");
    tracing::debug!("✨ RSLLM successfully connects to Ollama!");

    Ok(())
}
