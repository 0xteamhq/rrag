//! # RRAG Comprehensive Example
//!
//! This example demonstrates the RRAG framework capabilities:
//! - Agent-based interactions with tool calling
//! - LLM integration with multiple providers
//! - Conversation memory management
//! - Production-ready error handling
//!
//! Run with: `cargo run --bin rrag_comprehensive`

use rrag::prelude::*;
use rsllm::tool;
use rsllm::tools::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error;
use tokio;

// ============================================================================
// TOOL DEFINITIONS (using rsllm tool system)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalculatorParams {
    /// Operation: add, subtract, multiply, divide
    pub operation: String,
    pub a: f64,
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalculatorResult {
    pub result: f64,
}

#[tool(description = "Performs arithmetic calculations: add, subtract, multiply, divide")]
fn calculator(params: CalculatorParams) -> Result<CalculatorResult, Box<dyn Error + Send + Sync>> {
    let result = match params.operation.as_str() {
        "add" => params.a + params.b,
        "subtract" => params.a - params.b,
        "multiply" => params.a * params.b,
        "divide" if params.b != 0.0 => params.a / params.b,
        "divide" => return Err("Cannot divide by zero".into()),
        _ => {
            return Err(format!(
                "Unknown operation: '{}'. Valid operations: add, subtract, multiply, divide",
                params.operation
            )
            .into())
        }
    };
    Ok(CalculatorResult { result })
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct EchoParams {
    /// Message to echo back
    pub message: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct EchoResult {
    pub echoed: String,
}

#[tool(description = "Echo back the provided message")]
fn echo_tool(params: EchoParams) -> Result<EchoResult, Box<dyn Error + Send + Sync>> {
    Ok(EchoResult {
        echoed: format!("Echo: {}", params.message),
    })
}

#[tokio::main]
async fn main() -> RragResult<()> {
    println!("🦀 RRAG - Rust RAG Framework Comprehensive Demo");
    println!("==============================================\n");

    // Demo 1: Agent interactions with tool calling  
    println!("🤖 Demo 1: Agent Tool Calling");
    println!("------------------------------");
    demo_agent_interactions().await?;
    println!();

    // Demo 2: Memory management
    println!("🧠 Demo 2: Conversation Memory");
    println!("------------------------------");
    demo_memory_systems().await?;
    println!();

    // Demo 3: Multi-provider LLM support
    println!("🔌 Demo 3: LLM Provider Support");
    println!("--------------------------------");
    demo_llm_providers().await?;
    println!();

    println!("🎉 All RRAG comprehensive demos completed successfully!");
    Ok(())
}

/// Demonstrate agent interactions with tool calling
async fn demo_agent_interactions() -> RragResult<()> {
    println!("   🤖 Setting up intelligent agent...");

    // Create agent with tools
    let llm_client = rsllm::Client::from_env()?;
    let mut agent = rrag::AgentBuilder::new()
        .with_llm(llm_client)
        .with_tool(Box::new(CalculatorTool) as Box<dyn Tool>)
        .with_tool(Box::new(EchoToolTool) as Box<dyn Tool>)
        .verbose(true)
        .stateless()
        .build()?;

    // Test calculations
    println!("     - Testing calculator tool...");
    let calc_response = agent.run("Calculate 15 * 8 + 42").await?;
    println!("       Agent: {}", calc_response);

    // Test echo tool
    println!("     - Testing echo tool...");
    let echo_response = agent
        .run("Echo: Hello from RRAG!")
        .await?;
    println!("       Agent: {}", echo_response);

    // Test response (simulated since agent doesn't have streaming)
    println!("     - Testing response...");
    let response = agent
        .run("Tell me about Rust in a few sentences")
        .await?;
    println!("       Response: {}", response.chars().take(200).collect::<String>() + "...");

    Ok(())
}

/// Demonstrate different memory systems
async fn demo_memory_systems() -> RragResult<()> {
    println!("   🧠 Testing memory systems...");

    // Test stateless mode (no memory)
    println!("     - Testing stateless mode...");
    let client1 = rsllm::Client::from_env()?;
    let mut stateless_agent = rrag::AgentBuilder::new()
        .with_llm(client1)
        .stateless()
        .build()?;

    let _response1 = stateless_agent.run("My favorite color is blue").await?;
    let _response2 = stateless_agent.run("What is my favorite color?").await?;
    println!("       Stateless: Each query is independent");

    // Test stateful mode (with memory)
    println!("     - Testing stateful mode...");
    let client2 = rsllm::Client::from_env()?;
    let mut stateful_agent = rrag::AgentBuilder::new()
        .with_llm(client2)
        .stateful()
        .build()?;

    let _response3 = stateful_agent.run("My favorite color is blue").await?;
    let _response4 = stateful_agent.run("What is my favorite color?").await?;
    println!("       Stateful: Maintains conversation history");

    Ok(())
}

/// Demonstrate multi-provider LLM support
async fn demo_llm_providers() -> RragResult<()> {
    println!("   🔌 Testing LLM provider support...");

    println!("     - Available providers:");
    println!("       • OpenAI (GPT-3.5, GPT-4)");
    println!("       • Ollama (Local models)");
    println!("       • Claude (via API)");

    println!("     - Provider configuration via environment:");
    println!("       RSLLM_PROVIDER=ollama RSLLM_MODEL=llama3.2:3b");
    println!("       RSLLM_PROVIDER=openai RSLLM_MODEL=gpt-4");

    // Test current provider
    match rsllm::Client::from_env() {
        Ok(_client) => {
            let provider = std::env::var("RSLLM_PROVIDER").unwrap_or_else(|_| "default".to_string());
            let model = std::env::var("RSLLM_MODEL").unwrap_or_else(|_| "default".to_string());
            println!("       ✅ Connected to provider: {} with model: {}", provider, model);
        }
        Err(e) => {
            println!("       ⚠️  Client creation failed: {}", e);
            println!("       This is expected if no LLM provider is configured");
        }
    }

    Ok(())
}