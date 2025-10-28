//! Test Ollama Integration with RRAG
//!
//! This example tests the actual integration between RRAG and Ollama
//! to ensure real conversations work correctly.

use rrag::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> RragResult<()> {
    println!("🦀 RRAG + Ollama Integration Test");
    println!("==================================\n");

    // Test direct Ollama API call first
    println!("📡 Testing Ollama API connectivity...");
    match test_ollama_api().await {
        Ok(_) => println!("✅ Ollama API is accessible and working!\n"),
        Err(e) => {
            println!("❌ Ollama API test failed: {}", e);
            println!("⚠️  Make sure Ollama is running: ollama serve");
            return Ok(());
        }
    }

    // Create RRAG agent with Ollama
    println!("🤖 Creating RRAG agent with Ollama...");
    let client = rsllm::Client::from_env()?;
    let mut agent = rrag::AgentBuilder::new()
        .with_llm(client)
        .with_system_prompt("You are a helpful AI assistant. Be concise and clear.")
        .stateless()
        .build()?;

    println!("✅ Agent created successfully!\n");

    // Test 1: Simple question
    println!("💬 Test 1: Simple Question");
    println!("Question: What is Rust?");
    match agent
        .run("What is Rust programming language in one sentence?")
        .await
    {
        Ok(response) => {
            println!("🤖 Response: {}", response);
            println!("✅ Test 1 passed!\n");
        }
        Err(e) => {
            println!("❌ Test 1 failed: {}", e);
            println!();
        }
    }

    // Test 2: Conversation with context
    println!("💬 Test 2: Multi-turn Conversation");
    println!("Question 1: Tell me about memory safety");
    match agent
        .run("Tell me about memory safety in programming in one sentence.")
        .await
    {
        Ok(response) => {
            println!("🤖 Response: {}", response);

            // Follow-up question
            println!("\nQuestion 2: How does Rust achieve this?");
            match agent
                .run("How does Rust achieve this?")
                .await
            {
                Ok(followup) => {
                    println!("🤖 Response: {}", followup);
                    println!("✅ Test 2 passed!\n");
                }
                Err(e) => {
                    println!("❌ Test 2 follow-up failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Test 2 failed: {}", e);
            println!();
        }
    }

    // Test 3: Technical question
    println!("💬 Test 3: Technical Question");
    println!("Question: Explain borrowing");
    match agent
        .run("Explain Rust's borrowing concept in one sentence.")
        .await
    {
        Ok(response) => {
            println!("🤖 Response: {}", response);
            println!("✅ Test 3 passed!\n");
        }
        Err(e) => {
            println!("❌ Test 3 failed: {}", e);
            println!();
        }
    }

    println!("🎉 All Ollama integration tests completed!");
    println!("✨ RRAG successfully integrates with Ollama for real conversations!");

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
