//! RRAG Agent Demo - Using the New Agent Framework
//!
//! This demonstrates the production-ready agent from crates/rrag/src/agent/
//!
//! Features:
//! - LangChain-style agent with tool calling
//! - Stateless and stateful modes
//! - Real LLM integration (Ollama, OpenAI, Claude)
//! - Type-safe tool execution
//!
//! Run: cargo run --bin agent_demo --features rsllm-client

use rrag::prelude::*;
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
        _ => return Err(format!("Unknown operation: '{}'. Valid operations: add, subtract, multiply, divide", params.operation).into()),
    };
    Ok(CalculatorResult { result })
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TempConvertParams {
    /// Temperature value to convert
    pub temperature: f64,
    /// Source unit: "celsius" or "fahrenheit"
    pub from_unit: String,
    /// Target unit: "celsius" or "fahrenheit"
    pub to_unit: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TempConvertResult {
    /// Converted temperature value
    pub temperature: f64,
    /// Unit of the result
    pub unit: String,
}

#[tool(description = "Convert temperature between Celsius and Fahrenheit")]
fn convert_temperature(params: TempConvertParams) -> Result<TempConvertResult, Box<dyn Error + Send + Sync>> {
    let result = match (params.from_unit.as_str(), params.to_unit.as_str()) {
        ("celsius", "fahrenheit") => (params.temperature * 9.0 / 5.0) + 32.0,
        ("fahrenheit", "celsius") => (params.temperature - 32.0) * 5.0 / 9.0,
        ("celsius", "celsius") | ("fahrenheit", "fahrenheit") => params.temperature,
        _ => return Err(format!("Invalid units: {} to {}. Use 'celsius' or 'fahrenheit'", params.from_unit, params.to_unit).into()),
    };

    Ok(TempConvertResult {
        temperature: result,
        unit: params.to_unit,
    })
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct WeatherParams {
    /// City name
    pub city: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct WeatherResult {
    pub temperature: i32,
    pub condition: String,
}

#[tool(description = "Get weather for a city")]
fn get_weather(_params: WeatherParams) -> Result<WeatherResult, Box<dyn Error + Send + Sync>> {
    Ok(WeatherResult {
        temperature: 22,
        condition: "Sunny".to_string(),
    })
}

// ============================================================================
// MAIN - Demonstrate Both Modes
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  🤖 RRAG Agent Demo - Production Framework          ║");
    println!("╚══════════════════════════════════════════════════════╝\n");
    println!("💡 Set RUST_LOG=debug for detailed logs\n");

    println!("✅ Initializing LLM clients...\n");

    // ═══════════════════════════════════════════════════════════════════════
    // MODE 1: STATELESS (Each call is independent)
    // ═══════════════════════════════════════════════════════════════════════

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  MODE 1: Stateless Agent                             ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let llm_client1 = rsllm::Client::builder()
        .provider(rsllm::Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")?
        .temperature(0.7)
        .build()?;

    let mut stateless_agent = AgentBuilder::new()
        .with_llm(llm_client1)
        .with_tools(vec![
            Box::new(CalculatorTool) as Box<dyn Tool>,
            Box::new(GetWeatherTool) as Box<dyn Tool>,
        ])
        .stateless()
        .verbose(true)
        .build()?;

    println!("🧪 Query 1: What is 50 * 12?");
    let response1 = stateless_agent.run("What is 50 * 12?").await?;
    println!("📤 Final: {}\n", response1);

    println!("🧪 Query 2: What was my previous question?");
    let response2 = stateless_agent.run("What was my previous question?").await?;
    println!("📤 Final: {}", response2);
    println!("   💡 Stateless mode: Agent doesn't remember previous question!\n");

    // ═══════════════════════════════════════════════════════════════════════
    // MODE 2: STATEFUL (Maintains conversation across calls)
    // ═══════════════════════════════════════════════════════════════════════

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  MODE 2: Stateful Agent                              ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let llm_client2 = rsllm::Client::builder()
        .provider(rsllm::Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")?
        .temperature(0.7)
        .build()?;

    let mut stateful_agent = AgentBuilder::new()
        .with_llm(llm_client2)
        .with_tools(vec![
            Box::new(CalculatorTool) as Box<dyn Tool>,
            Box::new(GetWeatherTool) as Box<dyn Tool>,
            Box::new(ConvertTemperatureTool) as Box<dyn Tool>,
        ])
        .stateful() // ← Key difference!
        .verbose(true)
        .build()?;

    println!("🧪 Query 1: Calculate 25 + 75");
    let response3 = stateful_agent.run("Calculate 25 + 75").await?;
    println!("📤 Final: {}\n", response3);

    println!("🧪 Query 2: Now multiply that by 2");
    let response4 = stateful_agent.run("Now multiply that by 2").await?;
    println!("📤 Final: {}", response4);
    println!("   💡 Stateful mode: Agent remembers the result was 100!\n");

    println!("🧪 Query 3: What's the weather in Tokyo?");
    let response5 = stateful_agent.run("What's the weather in Tokyo?").await?;
    println!("📤 Final: {}\n", response5);

    println!("🧪 Query 4: How warm is that in Fahrenheit?");
    let response6 = stateful_agent.run("How warm is that in Fahrenheit?").await?;
    println!("📤 Final: {}", response6);
    println!("   💡 Stateful mode: Agent remembers the temperature!\n");

    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  Summary                                             ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    println!("✅ Stateless Mode:");
    println!("   • Each call is independent");
    println!("   • No memory between calls");
    println!("   • Good for one-off queries\n");

    println!("✅ Stateful Mode:");
    println!("   • Maintains conversation history");
    println!("   • Remembers context between calls");
    println!("   • Perfect for chat applications\n");

    println!("🎉 RRAG Agent framework is production-ready!");

    Ok(())
}
