//! Automatic JSON Schema Generation Example
//!
//! This example demonstrates how to use `SchemaBasedTool` for automatic
//! JSON schema generation from Rust types. NO MORE MANUAL JSON!
//!
//! Run with: cargo run -p rsllm --example tool_calling_auto_schema --features ollama

use rsllm::tools::{SchemaBasedTool, ToolRegistry, ToolCall as ToolCallExec};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// EXAMPLE 1: Calculator with Automatic Schema
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
#[schemars(description = "Parameters for arithmetic operations")]
struct CalculatorParams {
    /// The arithmetic operation to perform
    #[schemars(description = "Operation: add, subtract, multiply, or divide")]
    operation: Operation,

    /// First number in the operation
    a: f64,

    /// Second number in the operation
    b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

struct Calculator;

impl SchemaBasedTool for Calculator {
    type Params = CalculatorParams;

    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Performs basic arithmetic operations with automatic schema generation"
    }

    fn execute_typed(
        &self,
        params: Self::Params,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        println!("   📥 Received strongly-typed params: {:?}", params);

        let result = match params.operation {
            Operation::Add => params.a + params.b,
            Operation::Subtract => params.a - params.b,
            Operation::Multiply => params.a * params.b,
            Operation::Divide => {
                if params.b == 0.0 {
                    return Err("Division by zero".into());
                }
                params.a / params.b
            }
        };

        Ok(json!({
            "result": result,
            "operation": format!("{:?}", params.operation),
            "a": params.a,
            "b": params.b
        }))
    }

    fn validate_typed(&self, params: &Self::Params) -> Result<(), Box<dyn Error + Send + Sync>> {
        if params.a.is_nan() || params.b.is_nan() {
            return Err("Parameters cannot be NaN".into());
        }
        if params.a.is_infinite() || params.b.is_infinite() {
            return Err("Parameters cannot be infinite".into());
        }
        Ok(())
    }
}

// ============================================================================
// EXAMPLE 2: Weather Tool with Complex Types
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
struct WeatherParams {
    /// City name to get weather for
    #[schemars(description = "Name of the city")]
    city: String,

    /// Temperature unit preference
    #[serde(default = "default_unit")]
    #[schemars(description = "Temperature unit", default = "default_unit")]
    unit: TemperatureUnit,

    /// Include forecast data
    #[serde(default)]
    #[schemars(description = "Whether to include 7-day forecast")]
    include_forecast: bool,
}

fn default_unit() -> TemperatureUnit {
    TemperatureUnit::Celsius
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
enum TemperatureUnit {
    Celsius,
    Fahrenheit,
    Kelvin,
}

struct WeatherTool;

impl SchemaBasedTool for WeatherTool {
    type Params = WeatherParams;

    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a city with optional forecast"
    }

    fn execute_typed(
        &self,
        params: Self::Params,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        println!("   📥 Weather request for: {:?}", params);

        let temperature = match params.unit {
            TemperatureUnit::Celsius => 22,
            TemperatureUnit::Fahrenheit => 72,
            TemperatureUnit::Kelvin => 295,
        };

        let mut response = json!({
            "city": params.city,
            "temperature": temperature,
            "unit": format!("{:?}", params.unit),
            "condition": "Sunny",
            "humidity": 65
        });

        if params.include_forecast {
            response["forecast"] = json!([
                {"day": "Tomorrow", "temp": temperature + 2, "condition": "Partly Cloudy"},
                {"day": "Day 2", "temp": temperature - 1, "condition": "Rainy"},
                {"day": "Day 3", "temp": temperature + 3, "condition": "Sunny"},
            ]);
        }

        Ok(response)
    }

    fn validate_typed(&self, params: &Self::Params) -> Result<(), Box<dyn Error + Send + Sync>> {
        if params.city.is_empty() {
            return Err("City name cannot be empty".into());
        }
        if params.city.len() > 100 {
            return Err("City name too long".into());
        }
        Ok(())
    }
}

// ============================================================================
// EXAMPLE 3: Search Tool with Optional Parameters
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
struct SearchParams {
    /// Search query
    query: String,

    /// Maximum number of results
    #[serde(default = "default_limit")]
    #[schemars(range(min = 1, max = 100), default = "default_limit")]
    limit: u32,

    /// Search category (optional)
    #[schemars(description = "Filter by category")]
    category: Option<String>,
}

fn default_limit() -> u32 {
    10
}

struct SearchTool;

impl SchemaBasedTool for SearchTool {
    type Params = SearchParams;

    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for information with optional filters"
    }

    fn execute_typed(
        &self,
        params: Self::Params,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        Ok(json!({
            "query": params.query,
            "results": [
                json!({"title": "Result 1", "url": "https://example.com/1"}),
                json!({"title": "Result 2", "url": "https://example.com/2"}),
            ],
            "count": 2,
            "limit": params.limit,
            "category": params.category
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("✨ RSLLM Automatic Schema Generation Example");
    println!("==============================================\n");

    // Step 1: Create registry and register tools
    println!("📦 Step 1: Creating tool registry with auto-schema tools...");
    let mut registry = ToolRegistry::new();

    registry.register(Box::new(Calculator))?;
    registry.register(Box::new(WeatherTool))?;
    registry.register(Box::new(SearchTool))?;

    println!("   ✅ Registered {} tools with AUTOMATIC schemas!\n", registry.len());

    // Step 2: Show auto-generated schemas
    println!("🔍 Step 2: Auto-generated JSON schemas:");
    for def in registry.tool_definitions() {
        println!("\n   📝 Tool: {}", def.name);
        println!("      Description: {}", def.description);
        println!("      Schema: {}", serde_json::to_string_pretty(&def.parameters)?);
    }
    println!();

    // Step 3: Execute tools with type safety
    println!("🚀 Step 3: Executing tools with type-safe parameters\n");

    // Example 1: Calculator
    println!("   Example 1: Calculator (add 42 + 58)");
    let calc_call = ToolCallExec::new(
        "call-1",
        "calculator",
        json!({"operation": "add", "a": 42, "b": 58}),
    );
    let calc_result = registry.execute(&calc_call);
    if calc_result.success {
        println!("   ✅ Result: {}", serde_json::to_string_pretty(&calc_result.content)?);
    }
    println!();

    // Example 2: Weather with options
    println!("   Example 2: Weather (Tokyo with forecast)");
    let weather_call = ToolCallExec::new(
        "call-2",
        "get_weather",
        json!({
            "city": "Tokyo",
            "unit": "celsius",
            "include_forecast": true
        }),
    );
    let weather_result = registry.execute(&weather_call);
    if weather_result.success {
        println!("   ✅ Result: {}", serde_json::to_string_pretty(&weather_result.content)?);
    }
    println!();

    // Example 3: Search with defaults
    println!("   Example 3: Search (using default limit)");
    let search_call = ToolCallExec::new(
        "call-3",
        "search",
        json!({"query": "Rust programming"}),
    );
    let search_result = registry.execute(&search_call);
    if search_result.success {
        println!("   ✅ Result: {}", serde_json::to_string_pretty(&search_result.content)?);
    }
    println!();

    // Step 4: Show validation in action
    println!("⚠️  Step 4: Automatic validation");
    println!("   Attempting invalid operation (divide by zero)...");
    let invalid_call = ToolCallExec::new(
        "call-4",
        "calculator",
        json!({"operation": "divide", "a": 10, "b": 0}),
    );
    let invalid_result = registry.execute(&invalid_call);
    if !invalid_result.success {
        println!("   ❌ Caught error: {}", invalid_result.error.unwrap());
    }
    println!();

    println!("🎉 Automatic Schema Generation Complete!\n");
    println!("💡 Key Benefits:");
    println!("   ✅ NO manual JSON schema writing");
    println!("   ✅ Type-safe parameter handling");
    println!("   ✅ Automatic validation");
    println!("   ✅ Schema always in sync with types");
    println!("   ✅ Serde integration for defaults");
    println!("   ✅ Schemars attributes for descriptions");

    Ok(())
}
