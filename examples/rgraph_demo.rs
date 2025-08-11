//! # RRAG Agent Demo - Agent System with Tools
//!
//! This example demonstrates RRAG's agent capabilities:
//! - Building agents with tools
//! - Agent workflow execution
//! - Multi-step reasoning and decision making
//! - Tool usage patterns
//!
//! Run with: `cargo run --bin rgraph_demo`

use rrag::agent::AgentBuilder;
use rrag::prelude::*;
use rrag::tools::Calculator;
use std::sync::Arc;
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
    let agent = AgentBuilder::new()
        .with_name("helpful_assistant")
        .with_system_prompt(
            "You are a helpful AI assistant that provides clear and concise answers.",
        )
        .build()?;

    // Execute a simple query
    let response = agent
        .process_message("What are the benefits of using Rust for RAG systems?", None)
        .await?;

    println!(
        "✅ Agent Response: {}",
        response.text.chars().take(100).collect::<String>() + "..."
    );
    println!("📊 Response completed: {}", response.is_final);

    Ok(())
}

async fn calculator_integration_demo() -> RragResult<()> {
    // Create an agent with calculator tool
    let calculator = Calculator;

    let agent = AgentBuilder::new()
        .with_name("math_assistant")
        .with_system_prompt(
            "You are a mathematics assistant. Use the calculator tool for computations.",
        )
        .with_tool(Arc::new(calculator))
        .build()?;

    // Test mathematical query
    let response = agent
        .process_message("Please calculate 15 * 23 + 45 and explain the result", None)
        .await?;

    println!(
        "🔢 Calculator Response: {}",
        response.text.chars().take(150).collect::<String>() + "..."
    );

    if !response.tool_calls.is_empty() {
        println!("🔧 Tools used: {} tool calls", response.tool_calls.len());
    }

    Ok(())
}

async fn multi_step_reasoning_demo() -> RragResult<()> {
    // Create an agent for complex reasoning
    let agent = AgentBuilder::new()
        .with_name("reasoning_agent")
        .with_system_prompt("You are an expert at breaking down complex problems into steps.")
        .build()?;

    // Complex reasoning task
    let response = agent
        .process_message(
            "Plan a strategy for implementing a RAG system for a legal document database. \
         Consider performance, accuracy, and compliance requirements.",
            None,
        )
        .await?;

    println!(
        "🧠 Reasoning Response: {}",
        response.text.chars().take(200).collect::<String>() + "..."
    );
    println!("📊 Response is final: {}", response.is_final);
    println!("⏱️  Tool calls: {} calls", response.tool_calls.len());

    Ok(())
}
