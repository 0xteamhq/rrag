//! Simple Tool Calling Example
//!
//! This example demonstrates the easiest way to use tool calling with RSLLM.
//! Perfect for getting started with function calling.
//!
//! Run with: cargo run -p rsllm --example tool_calling_simple --features ollama

use rsllm::prelude::*;
use rsllm::tools::{Tool, ToolRegistry, ToolCall as ToolCallExec};
use rsllm::simple_tool;
use serde_json::json;
use std::error::Error;

// ============================================================================
// EXAMPLE 1: Calculator Tool
// ============================================================================

struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Performs basic arithmetic operations: add, subtract, multiply, divide"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let operation = args["operation"].as_str().ok_or("Missing operation")?;
        let a = args["a"].as_f64().ok_or("Invalid number for 'a'")?;
        let b = args["b"].as_f64().ok_or("Invalid number for 'b'")?;

        let result = match operation {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => {
                if b == 0.0 {
                    return Err("Division by zero".into());
                }
                a / b
            }
            _ => return Err(format!("Unknown operation: {}", operation).into()),
        };

        Ok(json!({
            "result": result,
            "operation": operation,
            "a": a,
            "b": b
        }))
    }
}

// ============================================================================
// EXAMPLE 2: Weather Tool
// ============================================================================

struct WeatherTool;

impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get the current weather for a city"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                    "default": "celsius"
                }
            },
            "required": ["city"]
        })
    }

    fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let city = args["city"].as_str().ok_or("Missing city")?;
        let unit = args["unit"].as_str().unwrap_or("celsius");

        // Mock weather data
        let temperature = match unit {
            "celsius" => 22,
            "fahrenheit" => 72,
            _ => 22,
        };

        Ok(json!({
            "city": city,
            "temperature": temperature,
            "unit": unit,
            "condition": "Sunny",
            "humidity": 65
        }))
    }
}

// ============================================================================
// EXAMPLE 3: Using the macro for quick tools
// ============================================================================

fn create_echo_tool() -> Box<dyn Tool> {
    simple_tool!(
        name: "echo",
        description: "Echoes back the input text",
        parameters: json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back"
                }
            },
            "required": ["text"]
        }),
        execute: |args| {
            json!({
                "echoed": args["text"],
                "length": args["text"].as_str().unwrap_or("").len()
            })
        }
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🛠️  RSLLM Tool Calling - Simple Example");
    println!("=========================================\n");

    // Step 1: Create a tool registry
    println!("📦 Step 1: Creating tool registry...");
    let mut registry = ToolRegistry::new();

    // Step 2: Register tools
    println!("🔧 Step 2: Registering tools...");
    registry.register(Box::new(Calculator))?;
    registry.register(Box::new(WeatherTool))?;
    registry.register(create_echo_tool())?;

    println!("   ✅ Registered {} tools:", registry.len());
    for tool_name in registry.tool_names() {
        println!("      - {}", tool_name);
    }
    println!();

    // Step 3: Show tool definitions (what gets sent to LLM)
    println!("📋 Step 3: Tool definitions for LLM:");
    let tool_defs = registry.tool_definitions();
    for def in &tool_defs {
        println!("   📝 Tool: {}", def.name);
        println!("      Description: {}", def.description);
    }
    println!();

    // Step 4: Simulate tool calling (what the LLM would request)
    println!("🤖 Step 4: Simulating LLM tool calls...\n");

    // Example 1: Calculator
    println!("   Example 1: Calculator");
    let calc_call = ToolCallExec::new(
        "call-1",
        "calculator",
        json!({"operation": "add", "a": 15, "b": 27})
    );
    let calc_result = registry.execute(&calc_call);
    println!("   Input: Add 15 + 27");
    println!("   Result: {}", serde_json::to_string_pretty(&calc_result.content)?);
    println!();

    // Example 2: Weather
    println!("   Example 2: Weather");
    let weather_call = ToolCallExec::new(
        "call-2",
        "get_weather",
        json!({"city": "San Francisco", "unit": "celsius"})
    );
    let weather_result = registry.execute(&weather_call);
    println!("   Input: Get weather for San Francisco");
    println!("   Result: {}", serde_json::to_string_pretty(&weather_result.content)?);
    println!();

    // Example 3: Echo
    println!("   Example 3: Echo");
    let echo_call = ToolCallExec::new(
        "call-3",
        "echo",
        json!({"text": "Hello, tool calling!"})
    );
    let echo_result = registry.execute(&echo_call);
    println!("   Input: Echo 'Hello, tool calling!'");
    println!("   Result: {}", serde_json::to_string_pretty(&echo_result.content)?);
    println!();

    // Step 5: Batch execution
    println!("🔄 Step 5: Batch tool execution");
    let batch_calls = vec![
        ToolCallExec::new("batch-1", "calculator", json!({"operation": "multiply", "a": 6, "b": 7})),
        ToolCallExec::new("batch-2", "echo", json!({"text": "Batch processing works!"})),
    ];

    let batch_results = registry.execute_batch(&batch_calls);
    println!("   Executed {} tools in batch:", batch_results.len());
    for result in batch_results {
        println!("      ✅ {}: {:?}", result.tool_name, result.content);
    }
    println!();

    // Step 6: Error handling
    println!("⚠️  Step 6: Error handling");
    let error_call = ToolCallExec::new(
        "error-1",
        "calculator",
        json!({"operation": "divide", "a": 10, "b": 0})
    );
    let error_result = registry.execute(&error_call);
    if !error_result.success {
        println!("   ❌ Tool error: {}", error_result.error.unwrap());
    }
    println!();

    println!("🎉 Tool calling example complete!\n");
    println!("💡 Next steps:");
    println!("   1. Integrate with LLM client (see tool_calling_with_llm.rs)");
    println!("   2. Create your own custom tools");
    println!("   3. Use the simple_tool! macro for quick tools");

    Ok(())
}
