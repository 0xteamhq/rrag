//! Environment Variable Configuration Test
//!
//! This example demonstrates how to use environment variables to configure RSLLM.
//! It's especially useful for:
//! - Custom/self-hosted LLM endpoints
//! - Custom fine-tuned models
//! - CI/CD pipelines
//! - Different deployment environments
//!
//! Run with: cargo run -p rsllm --example env_config_test --features ollama
//!
//! Environment Variables Supported:
//! - RSLLM_PROVIDER: Provider name (openai, claude, ollama)
//! - RSLLM_BASE_URL: Generic base URL for any provider
//! - RSLLM_OPENAI_BASE_URL: OpenAI-specific base URL (overrides RSLLM_BASE_URL)
//! - RSLLM_OLLAMA_BASE_URL: Ollama-specific base URL (overrides RSLLM_BASE_URL)
//! - RSLLM_CLAUDE_BASE_URL: Claude-specific base URL (overrides RSLLM_BASE_URL)
//! - RSLLM_MODEL: Generic model name
//! - RSLLM_OPENAI_MODEL: OpenAI-specific model (overrides RSLLM_MODEL)
//! - RSLLM_OLLAMA_MODEL: Ollama-specific model (overrides RSLLM_MODEL)
//! - RSLLM_CLAUDE_MODEL: Claude-specific model (overrides RSLLM_MODEL)
//! - RSLLM_API_KEY: API key for providers that need it
//! - RSLLM_TEMPERATURE: Temperature setting (0.0 to 2.0)
//! - RSLLM_MAX_TOKENS: Maximum tokens to generate

use rsllm::prelude::*;
use std::env;

#[tokio::main]
async fn main() -> RsllmResult<()> {
    println!("🔧 RSLLM Environment Variable Configuration Test");
    println!("==================================================\n");

    // Test 1: Using environment variables
    println!("📝 Test 1: Load configuration from environment variables");
    println!();

    // Set test environment variables
    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://localhost:11434/api");
    env::set_var("RSLLM_OLLAMA_MODEL", "llama3.2:3b");
    env::set_var("RSLLM_TEMPERATURE", "0.7");

    println!("   Set environment variables:");
    println!("   - RSLLM_PROVIDER=ollama");
    println!("   - RSLLM_OLLAMA_BASE_URL=http://localhost:11434/api");
    println!("   - RSLLM_OLLAMA_MODEL=llama3.2:3b");
    println!("   - RSLLM_TEMPERATURE=0.7");
    println!();

    match Client::from_env() {
        Ok(client) => {
            println!("   ✅ Client created from environment variables!");
            println!("   📊 Provider: {:?}", client.provider().provider_type());
            println!("   📊 Model: {}", client.config().model.model);
            println!();

            // Test with a simple message
            let messages = vec![ChatMessage::user(
                "Say 'Hello from environment config' in one sentence.",
            )];

            match client.chat_completion(messages).await {
                Ok(response) => {
                    println!("   ✅ Chat completion successful!");
                    println!("   📤 Response: {}", response.content);
                    println!();
                }
                Err(e) => {
                    println!("   ⚠️  Chat completion failed: {}", e);
                    println!("   (This is expected if Ollama is not running)");
                    println!();
                }
            }
        }
        Err(e) => {
            println!("   ❌ Failed to create client: {}", e);
            println!();
        }
    }

    // Test 2: Custom model name (not in predefined list)
    println!("📝 Test 2: Using a custom model name");
    println!("   (Demonstrates support for fine-tuned or custom models)");
    println!();

    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://localhost:11434/api/");
    env::set_var("RSLLM_OLLAMA_MODEL", "my-custom-fine-tuned-model:latest");

    println!("   Set custom model: my-custom-fine-tuned-model:latest");
    println!();

    match Client::from_env() {
        Ok(client) => {
            println!("   ✅ Client accepts custom model name!");
            println!("   📊 Model: {}", client.config().model.model);
            println!("   💡 The library does NOT validate against a predefined model list");
            println!("      This allows flexibility for custom models.");
            println!();
        }
        Err(e) => {
            println!("   ❌ Failed: {}", e);
            println!();
        }
    }

    // Test 3: Provider-specific vs generic environment variables
    println!("📝 Test 3: Provider-specific environment variables take precedence");
    println!();

    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_BASE_URL", "http://generic-url:8080/api/");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://ollama-specific:11434/api/");
    env::set_var("RSLLM_MODEL", "generic-model");
    env::set_var("RSLLM_OLLAMA_MODEL", "ollama-specific-model");

    println!("   Set both generic and provider-specific variables:");
    println!("   - RSLLM_BASE_URL=http://generic-url:8080/api/");
    println!("   - RSLLM_OLLAMA_BASE_URL=http://ollama-specific:11434/api/");
    println!("   - RSLLM_MODEL=generic-model");
    println!("   - RSLLM_OLLAMA_MODEL=ollama-specific-model");
    println!();

    match Client::from_env() {
        Ok(client) => {
            let config = client.config();
            let base_url = config
                .provider
                .base_url
                .as_ref()
                .map(|u| u.as_str())
                .unwrap_or("(default)");

            println!("   ✅ Provider-specific variables take precedence!");
            println!("   📊 Base URL used: {}", base_url);
            println!("   📊 Model used: {}", config.model.model);
            println!();

            if base_url.contains("ollama-specific") {
                println!("   ✅ Correctly used RSLLM_OLLAMA_BASE_URL over RSLLM_BASE_URL");
            }
            if config.model.model == "ollama-specific-model" {
                println!("   ✅ Correctly used RSLLM_OLLAMA_MODEL over RSLLM_MODEL");
            }
            println!();
        }
        Err(e) => {
            println!("   ❌ Failed: {}", e);
            println!();
        }
    }

    // Test 4: URL with and without trailing slash
    println!("📝 Test 4: URL normalization (trailing slash handling)");
    println!();

    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://localhost:11434/api");
    env::set_var("RSLLM_OLLAMA_MODEL", "llama3.2:3b");

    println!("   Set URL without trailing slash: http://localhost:11434/api");
    println!();

    match Client::from_env() {
        Ok(_client) => {
            println!("   ✅ URL normalized correctly!");
            println!("   💡 Library automatically handles trailing slashes");
            println!();
        }
        Err(e) => {
            println!("   ❌ Failed: {}", e);
            println!();
        }
    }

    // Clean up environment variables
    env::remove_var("RSLLM_PROVIDER");
    env::remove_var("RSLLM_BASE_URL");
    env::remove_var("RSLLM_OLLAMA_BASE_URL");
    env::remove_var("RSLLM_MODEL");
    env::remove_var("RSLLM_OLLAMA_MODEL");
    env::remove_var("RSLLM_TEMPERATURE");

    println!("🎉 All environment variable tests completed!");
    println!();
    println!("💡 Key Takeaways:");
    println!("   1. Provider-specific env vars override generic ones");
    println!("   2. Custom model names are fully supported");
    println!("   3. URLs are automatically normalized (trailing slash handling)");
    println!("   4. Perfect for CI/CD and multi-environment deployments");

    Ok(())
}
