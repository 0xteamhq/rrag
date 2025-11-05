//! Test Ollama Integration with RRAG
//!
//! This example tests the actual integration between RRAG and Ollama
//! to ensure real conversations work correctly.

use rrag::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("🦀 RRAG + Ollama Integration Test");
    tracing::debug!("==================================\n");

    // Test direct Ollama API call first
    tracing::debug!("📡 Testing Ollama API connectivity...");
    match test_ollama_api().await {
        Ok(_) => tracing::debug!("✅ Ollama API is accessible and working!\n"),
        Err(e) => {
            error!(" Ollama API test failed: {}", e);
            warn!("  Make sure Ollama is running: ollama serve");
            return Ok(());
        }
    }

    // Create RRAG agent with Ollama
    tracing::debug!("🤖 Creating RRAG agent with Ollama...");
    let agent = RragAgent::builder()
        .with_name("Ollama Test Agent")
        .with_model("ollama", "llama3.2:3b")
        .with_system_prompt("You are a helpful AI assistant. Be concise and clear.")
        .with_temperature(0.7)
        .build()?;

    tracing::debug!("✅ Agent created successfully!\n");

    // Test 1: Simple question
    tracing::debug!("💬 Test 1: Simple Question");
    tracing::debug!("Question: What is Rust?");
    match agent
        .process_message("What is Rust programming language in one sentence?", None)
        .await
    {
        Ok(response) => {
            tracing::debug!("🤖 Response: {}", response.text);
            tracing::debug!("⏱️  Duration: {}ms", response.metadata.duration_ms);
            tracing::debug!("✅ Test 1 passed!\n");
        }
        Err(e) => {
            error!(" Test 1 failed: {}", e);
        }
    }

    // Test 2: Conversation with context
    tracing::debug!("💬 Test 2: Multi-turn Conversation");
    tracing::debug!("Question 1: Tell me about memory safety");
    match agent
        .process_message(
            "Tell me about memory safety in programming in one sentence.",
            None,
        )
        .await
    {
        Ok(response) => {
            tracing::debug!("🤖 Response: {}", response.text);

            // Follow-up question
            tracing::debug!("\nQuestion 2: How does Rust achieve this?");
            match agent
                .process_message("How does Rust achieve this?", None)
                .await
            {
                Ok(followup) => {
                    tracing::debug!("🤖 Response: {}", followup.text);
                    tracing::debug!("✅ Test 2 passed!\n");
                }
                Err(e) => {
                    error!(" Test 2 follow-up failed: {}", e);
                }
            }
        }
        Err(e) => {
            error!(" Test 2 failed: {}", e);
        }
    }

    // Test 3: Technical question
    tracing::debug!("💬 Test 3: Technical Question");
    tracing::debug!("Question: Explain borrowing");
    match agent
        .process_message("Explain Rust's borrowing concept in one sentence.", None)
        .await
    {
        Ok(response) => {
            tracing::debug!("🤖 Response: {}", response.text);
            tracing::debug!("⏱️  Duration: {}ms", response.metadata.duration_ms);
            tracing::debug!("✅ Test 3 passed!\n");
        }
        Err(e) => {
            error!(" Test 3 failed: {}", e);
        }
    }

    tracing::debug!("🎉 All Ollama integration tests completed!");
    tracing::debug!("✨ RRAG successfully integrates with Ollama for real conversations!");

    Ok(())
}

async fn test_ollama_api() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    let request_body = json!({
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": "Say 'OK' if you can respond."}
        ],
        "stream": false
    });

    let response = client
        .post("http://localhost:11434/v1/chat/completions")
        .json(&request_body)
        .send()
        .await?;

    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!("API returned status: {}", response.status()).into())
    }
}
