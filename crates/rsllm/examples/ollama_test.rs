//! Direct Ollama Test - Verify RSLLM can communicate with Ollama
//!
//! Run with: cargo run -p rsllm --example ollama_test --features ollama

use rsllm::prelude::*;

#[tokio::main]
async fn main() -> RsllmResult<()> {
    println!("🦀 RSLLM + Ollama Direct Test");
    println!("==============================\n");

    // Create client pointing to Ollama
    println!("🔧 Creating RSLLM client for Ollama...");

    let client = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")? // Testing with trailing slash
        .temperature(0.7)
        .build()?;

    println!("✅ Client created successfully!\n");

    // Test 1: Simple completion
    println!("💬 Test 1: Simple Chat Completion");
    println!("Question: What is Rust?");

    let messages = vec![ChatMessage::user(
        "What is Rust programming language? Answer in one sentence.",
    )];

    match client.chat_completion(messages).await {
        Ok(response) => {
            println!("✅ Response received!");
            println!("🤖 Answer: {}", response.content);
            println!("📊 Model: {}", response.model);
            if let Some(reason) = &response.finish_reason {
                println!("🏁 Finish reason: {}", reason);
            }
            println!();
        }
        Err(e) => {
            println!("❌ Test failed: {}", e);
            println!("⚠️  Make sure Ollama is running and the model is available");
            println!("   Run: ollama serve");
            println!("   Run: ollama pull llama3.2:3b");
            return Ok(());
        }
    }

    // Test 2: Streaming
    println!("💬 Test 2: Streaming Chat Completion");
    println!("Question: Explain ownership in Rust");

    let stream_messages = vec![ChatMessage::user(
        "Explain Rust's ownership concept in 2 sentences.",
    )];

    match client.chat_completion_stream(stream_messages).await {
        Ok(mut stream) => {
            print!("🤖 Streaming: ");

            use futures_util::StreamExt;
            let mut full_response = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) if chunk.has_content() => {
                        print!("{}", chunk.content);
                        full_response.push_str(&chunk.content);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                    Ok(chunk) if chunk.is_done => {
                        println!("\n✅ Streaming completed!");
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        println!("\n❌ Stream error: {}", e);
                        break;
                    }
                }
            }
            println!();
        }
        Err(e) => {
            println!("❌ Streaming test failed: {}", e);
            println!();
        }
    }

    // Test 3: Multi-turn conversation
    println!("💬 Test 3: Multi-turn Conversation");

    let conversation = vec![
        ChatMessage::system("You are a helpful Rust programming assistant."),
        ChatMessage::user("What is borrowing in Rust?"),
        ChatMessage::assistant("Borrowing in Rust allows you to reference data without taking ownership, enabling safe concurrent access."),
        ChatMessage::user("Give me a code example"),
    ];

    match client.chat_completion(conversation).await {
        Ok(response) => {
            println!("✅ Multi-turn conversation works!");
            println!("🤖 Response: {}", response.content);
            println!();
        }
        Err(e) => {
            println!("❌ Multi-turn test failed: {}", e);
            println!();
        }
    }

    // Test 4: Health check
    println!("💬 Test 4: Health Check");
    match client.health_check().await {
        Ok(true) => println!("✅ Ollama is healthy!\n"),
        Ok(false) => println!("⚠️  Ollama health check returned false\n"),
        Err(e) => println!("❌ Health check failed: {}\n", e),
    }

    println!("🎉 All tests completed!");
    println!("✨ RSLLM successfully connects to Ollama!");

    Ok(())
}
