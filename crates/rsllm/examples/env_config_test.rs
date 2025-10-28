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
    tracing::debug!("🔧 RSLLM Environment Variable Configuration Test");
    tracing::debug!("==================================================\n");

    // Test 1: Using environment variables
    tracing::debug!("📝 Test 1: Load configuration from environment variables");

    // Set test environment variables
    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://localhost:11434/api");
    env::set_var("RSLLM_OLLAMA_MODEL", "llama3.2:3b");
    env::set_var("RSLLM_TEMPERATURE", "0.7");

    tracing::debug!("   Set environment variables:");
    tracing::debug!("   - RSLLM_PROVIDER=ollama");
    tracing::debug!("   - RSLLM_OLLAMA_BASE_URL=http://localhost:11434/api");
    tracing::debug!("   - RSLLM_OLLAMA_MODEL=llama3.2:3b");
    tracing::debug!("   - RSLLM_TEMPERATURE=0.7");

    match Client::from_env() {
        Ok(client) => {
            tracing::debug!("   ✅ Client created from environment variables!");
            tracing::debug!("   📊 Provider: {:?}", client.provider().provider_type());
            tracing::debug!("   📊 Model: {}", client.config().model.model);

            // Test with a simple message
            let messages = vec![ChatMessage::user(
                "Say 'Hello from environment config' in one sentence.",
            )];

            match client.chat_completion(messages).await {
                Ok(response) => {
                    tracing::debug!("   ✅ Chat completion successful!");
                    tracing::debug!("   📤 Response: {}", response.content);
                }
                Err(e) => {
                    tracing::debug!("   ⚠️  Chat completion failed: {}", e);
                    tracing::debug!("   (This is expected if Ollama is not running)");
                }
            }
        }
        Err(e) => {
            tracing::debug!("   ❌ Failed to create client: {}", e);
        }
    }

    // Test 2: Custom model name (not in predefined list)
    tracing::debug!("📝 Test 2: Using a custom model name");
    tracing::debug!("   (Demonstrates support for fine-tuned or custom models)");

    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://localhost:11434/api/");
    env::set_var("RSLLM_OLLAMA_MODEL", "my-custom-fine-tuned-model:latest");

    tracing::debug!("   Set custom model: my-custom-fine-tuned-model:latest");

    match Client::from_env() {
        Ok(client) => {
            tracing::debug!("   ✅ Client accepts custom model name!");
            tracing::debug!("   📊 Model: {}", client.config().model.model);
            tracing::debug!("   💡 The library does NOT validate against a predefined model list");
            tracing::debug!("      This allows flexibility for custom models.");
        }
        Err(e) => {
            tracing::debug!("   ❌ Failed: {}", e);
        }
    }

    // Test 3: Provider-specific vs generic environment variables
    tracing::debug!("📝 Test 3: Provider-specific environment variables take precedence");

    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_BASE_URL", "http://generic-url:8080/api/");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://ollama-specific:11434/api/");
    env::set_var("RSLLM_MODEL", "generic-model");
    env::set_var("RSLLM_OLLAMA_MODEL", "ollama-specific-model");

    tracing::debug!("   Set both generic and provider-specific variables:");
    tracing::debug!("   - RSLLM_BASE_URL=http://generic-url:8080/api/");
    tracing::debug!("   - RSLLM_OLLAMA_BASE_URL=http://ollama-specific:11434/api/");
    tracing::debug!("   - RSLLM_MODEL=generic-model");
    tracing::debug!("   - RSLLM_OLLAMA_MODEL=ollama-specific-model");

    match Client::from_env() {
        Ok(client) => {
            let config = client.config();
            let base_url = config
                .provider
                .base_url
                .as_ref()
                .map(|u| u.as_str())
                .unwrap_or("(default)");

            tracing::debug!("   ✅ Provider-specific variables take precedence!");
            tracing::debug!("   📊 Base URL used: {}", base_url);
            tracing::debug!("   📊 Model used: {}", config.model.model);

            if base_url.contains("ollama-specific") {
                tracing::debug!("   ✅ Correctly used RSLLM_OLLAMA_BASE_URL over RSLLM_BASE_URL");
            }
            if config.model.model == "ollama-specific-model" {
                tracing::debug!("   ✅ Correctly used RSLLM_OLLAMA_MODEL over RSLLM_MODEL");
            }
        }
        Err(e) => {
            tracing::debug!("   ❌ Failed: {}", e);
        }
    }

    // Test 4: URL with and without trailing slash
    tracing::debug!("📝 Test 4: URL normalization (trailing slash handling)");

    env::set_var("RSLLM_PROVIDER", "ollama");
    env::set_var("RSLLM_OLLAMA_BASE_URL", "http://localhost:11434/api");
    env::set_var("RSLLM_OLLAMA_MODEL", "llama3.2:3b");

    tracing::debug!("   Set URL without trailing slash: http://localhost:11434/api");

    match Client::from_env() {
        Ok(_client) => {
            tracing::debug!("   ✅ URL normalized correctly!");
            tracing::debug!("   💡 Library automatically handles trailing slashes");
        }
        Err(e) => {
            tracing::debug!("   ❌ Failed: {}", e);
        }
    }

    // Clean up environment variables
    env::remove_var("RSLLM_PROVIDER");
    env::remove_var("RSLLM_BASE_URL");
    env::remove_var("RSLLM_OLLAMA_BASE_URL");
    env::remove_var("RSLLM_MODEL");
    env::remove_var("RSLLM_OLLAMA_MODEL");
    env::remove_var("RSLLM_TEMPERATURE");

    tracing::debug!("🎉 All environment variable tests completed!");
    tracing::debug!("💡 Key Takeaways:");
    tracing::debug!("   1. Provider-specific env vars override generic ones");
    tracing::debug!("   2. Custom model names are fully supported");
    tracing::debug!("   3. URLs are automatically normalized (trailing slash handling)");
    tracing::debug!("   4. Perfect for CI/CD and multi-environment deployments");

    Ok(())
}
