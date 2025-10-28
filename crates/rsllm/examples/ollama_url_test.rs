//! Test URL flexibility - with and without trailing slashes
//!
//! Run with: cargo run -p rsllm --example ollama_url_test --features ollama

use rsllm::prelude::*;

#[tokio::main]
async fn main() -> RsllmResult<()> {
    println!("🦀 RSLLM URL Flexibility Test");
    println!("================================\n");

    // Test 1: URL WITHOUT trailing slash
    println!("📝 Test 1: Base URL WITHOUT trailing slash");
    println!("   URL: http://localhost:11434/api");

    let client1 = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api")? // NO trailing slash
        .temperature(0.7)
        .build()?;

    match client1.health_check().await {
        Ok(true) => println!("   ✅ Health check passed!"),
        Ok(false) => println!("   ⚠️  Health check returned false"),
        Err(e) => {
            println!("   ❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    let messages1 = vec![ChatMessage::user(
        "Say 'Hello from test 1' in one sentence.",
    )];
    match client1.chat_completion(messages1).await {
        Ok(response) => {
            println!("   ✅ Chat completion succeeded!");
            println!("   📤 Response: {}\n", response.content);
        }
        Err(e) => {
            println!("   ❌ Chat completion failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 2: URL WITH trailing slash
    println!("📝 Test 2: Base URL WITH trailing slash");
    println!("   URL: http://localhost:11434/api/");

    let client2 = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")? // WITH trailing slash
        .temperature(0.7)
        .build()?;

    match client2.health_check().await {
        Ok(true) => println!("   ✅ Health check passed!"),
        Ok(false) => println!("   ⚠️  Health check returned false"),
        Err(e) => {
            println!("   ❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    let messages2 = vec![ChatMessage::user(
        "Say 'Hello from test 2' in one sentence.",
    )];
    match client2.chat_completion(messages2).await {
        Ok(response) => {
            println!("   ✅ Chat completion succeeded!");
            println!("   📤 Response: {}\n", response.content);
        }
        Err(e) => {
            println!("   ❌ Chat completion failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 3: Using default URL (which has trailing slash)
    println!("📝 Test 3: Using default provider URL");
    println!("   (Provider::Ollama.default_base_url())");

    let client3 = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .temperature(0.7)
        .build()?;

    match client3.health_check().await {
        Ok(true) => println!("   ✅ Health check passed!"),
        Ok(false) => println!("   ⚠️  Health check returned false"),
        Err(e) => {
            println!("   ❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    let messages3 = vec![ChatMessage::user(
        "Say 'Hello from test 3' in one sentence.",
    )];
    match client3.chat_completion(messages3).await {
        Ok(response) => {
            println!("   ✅ Chat completion succeeded!");
            println!("   📤 Response: {}\n", response.content);
        }
        Err(e) => {
            println!("   ❌ Chat completion failed: {}\n", e);
            return Ok(());
        }
    }

    println!("🎉 All URL format tests passed!");
    println!("✨ RSLLM handles both URL formats correctly!");
    println!("\n💡 Key Insight:");
    println!("   Customers can provide base URLs with or without trailing slashes,");
    println!("   and the library will normalize them automatically for consistent behavior.");

    Ok(())
}
