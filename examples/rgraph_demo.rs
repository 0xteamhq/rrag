//! # RRAG Agent Demo - Agent System with Tools
//!
//! This example demonstrates RRAG's agent capabilities:
//! - Building agents with tools
//! - Agent workflow execution
//! - Multi-step reasoning and decision making
//! - Tool usage patterns
//!
//! Run with: `cargo run --bin rgraph_demo`

use rrag::{AgentBuilder, RragResult};
use rsllm::tool;
use rsllm::tools::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error;

// ============================================================================
// DEFINE TOOLS
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

use tokio;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🤖 RRAG Agent System Demo");
    println!("==========================\n");

    // Demo 1: Simple Agent Workflow
    println!("📋 Demo 1: Simple Agent with Tools");
    println!("----------------------------------");
    simple_agent_workflow().await?;
    println!();

    // Demo 2: Calculator Integration
    println!("🔢 Demo 2: Calculator Tool Integration");
    println!("--------------------------------------");
    calculator_integration_demo().await?;
    println!();

    // Demo 3: Multi-step Reasoning
    println!("🧠 Demo 3: Multi-step Reasoning");
    println!("--------------------------------");
    multi_step_reasoning_demo().await?;
    println!();

    println!("🎉 All RRAG agent demos completed successfully!");
    Ok(())
}

async fn simple_agent_workflow() -> RragResult<()> {
    // Create a simple agent with basic configuration
    let llm_client = rsllm::Client::from_env()?;
    let mut agent = AgentBuilder::new()
        .with_llm(llm_client)
        .with_system_prompt(
            "You are a helpful AI assistant that provides clear and concise answers.",
        )
        .stateless()
        .build()?;

    // Execute a simple query
    let response = agent
        .run("What are the benefits of using Rust for RAG systems?")
        .await?;

    println!(
        "✅ Agent Response: {}",
        response.chars().take(100).collect::<String>() + "..."
    );

    Ok(())
}

async fn calculator_integration_demo() -> RragResult<()> {
    // Create an agent with calculator tool
    let mut agent = AgentBuilder::new()
        .with_llm(rsllm::Client::from_env()?)
        .with_system_prompt(
            "You are a mathematics assistant. Use the calculator tool for computations.",
        )
        .with_tool(Box::new(CalculatorTool) as Box<dyn Tool>)
        .stateless()
        .build()?;

    // Test mathematical query
    let response = agent
        .run("Please calculate 15 * 23 + 45 and explain the result")
        .await?;

    println!(
        "🔢 Calculator Response: {}",
        response.chars().take(150).collect::<String>() + "..."
    );

    Ok(())
}

async fn multi_step_reasoning_demo() -> RragResult<()> {
    // Create an agent for complex reasoning
    let mut agent = AgentBuilder::new()
        .with_llm(rsllm::Client::from_env()?)
        .with_system_prompt("You are an expert at breaking down complex problems into steps.")
        .stateless()
        .build()?;

    // Complex reasoning task
    let response = agent
        .run(
            "Plan a strategy for implementing a RAG system for a legal document database. \
         Consider performance, accuracy, and compliance requirements.",
        )
        .await?;

    println!(
        "🧠 Reasoning Response: {}",
        response.chars().take(200).collect::<String>() + "..."
    );

    Ok(())
}
