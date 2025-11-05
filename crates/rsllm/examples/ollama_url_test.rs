//! Test URL flexibility - with and without trailing slashes
//!
//! Run with: cargo run -p rsllm --example ollama_url_test --features ollama

use rsllm::prelude::*;

#[tokio::main]
async fn main() -> RsllmResult<()> {
    tracing::debug!("🦀 RSLLM URL Flexibility Test");
    tracing::debug!("================================\n");

    // Test 1: URL WITHOUT trailing slash
    tracing::debug!("📝 Test 1: Base URL WITHOUT trailing slash");
    tracing::debug!("   URL: http://localhost:11434/api");

    let client1 = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api")? // NO trailing slash
        .temperature(0.7)
        .build()?;

    match client1.health_check().await {
        Ok(true) => tracing::debug!("   ✅ Health check passed!"),
        Ok(false) => tracing::debug!("   ⚠️  Health check returned false"),
        Err(e) => {
            tracing::debug!("   ❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    let messages1 = vec![ChatMessage::user(
        "Say 'Hello from test 1' in one sentence.",
    )];
    match client1.chat_completion(messages1).await {
        Ok(response) => {
            tracing::debug!("   ✅ Chat completion succeeded!");
            tracing::debug!("   📤 Response: {}\n", response.content);
        }
        Err(e) => {
            tracing::debug!("   ❌ Chat completion failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 2: URL WITH trailing slash
    tracing::debug!("📝 Test 2: Base URL WITH trailing slash");
    tracing::debug!("   URL: http://localhost:11434/api/");

    let client2 = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")? // WITH trailing slash
        .temperature(0.7)
        .build()?;

    match client2.health_check().await {
        Ok(true) => tracing::debug!("   ✅ Health check passed!"),
        Ok(false) => tracing::debug!("   ⚠️  Health check returned false"),
        Err(e) => {
            tracing::debug!("   ❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    let messages2 = vec![ChatMessage::user(
        "Say 'Hello from test 2' in one sentence.",
    )];
    match client2.chat_completion(messages2).await {
        Ok(response) => {
            tracing::debug!("   ✅ Chat completion succeeded!");
            tracing::debug!("   📤 Response: {}\n", response.content);
        }
        Err(e) => {
            tracing::debug!("   ❌ Chat completion failed: {}\n", e);
            return Ok(());
        }
    }

    // Test 3: Using default URL (which has trailing slash)
    tracing::debug!("📝 Test 3: Using default provider URL");
    tracing::debug!("   (Provider::Ollama.default_base_url())");

    let client3 = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .temperature(0.7)
        .build()?;

    match client3.health_check().await {
        Ok(true) => tracing::debug!("   ✅ Health check passed!"),
        Ok(false) => tracing::debug!("   ⚠️  Health check returned false"),
        Err(e) => {
            tracing::debug!("   ❌ Health check failed: {}", e);
            return Ok(());
        }
    }

    let messages3 = vec![ChatMessage::user(
        "Say 'Hello from test 3' in one sentence.",
    )];
    match client3.chat_completion(messages3).await {
        Ok(response) => {
            tracing::debug!("   ✅ Chat completion succeeded!");
            tracing::debug!("   📤 Response: {}\n", response.content);
        }
        Err(e) => {
            tracing::debug!("   ❌ Chat completion failed: {}\n", e);
            return Ok(());
        }
    }

    tracing::debug!("🎉 All URL format tests passed!");
    tracing::debug!("✨ RSLLM handles both URL formats correctly!");
    tracing::debug!("\n💡 Key Insight:");
    tracing::debug!("   Customers can provide base URLs with or without trailing slashes,");
    tracing::debug!("   and the library will normalize them automatically for consistent behavior.");

    Ok(())
}
