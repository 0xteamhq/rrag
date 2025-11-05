//! Individual Parameters Mode Example
//!
//! This demonstrates using the #[tool] macro with parameter structs.
//! Each tool function takes a params struct that defines its schema.
//!
//! This example shows:
//! - Defining params structs with JSON schema annotations
//! - Using the #[tool] macro to generate tool implementations
//! - Registering multiple tools in a registry
//! - Executing tools with different parameter types
//!
//! Run with: cargo run -p rsllm --example tool_individual_params --all-features

use rsllm::tool;
use rsllm::tools::{ToolCall as ToolCallExec, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// For now, we still use structs (individual params coming soon!)
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct AddParams {
    /// First number
    pub a: f64,
    /// Second number
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct MultiplyParams {
    /// First number to multiply
    pub x: f64,
    /// Second number to multiply
    pub y: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct PowerParams {
    /// Base number
    pub base: f64,
    /// Exponent
    pub exp: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct MathResult {
    pub result: f64,
}

// Current working approach
#[tool(description = "Adds two numbers")]
fn add_simple(params: AddParams) -> Result<MathResult, Box<dyn Error + Send + Sync>> {
    tracing::debug!("   🔢 Adding {} + {}", params.a, params.b);
    Ok(MathResult {
        result: params.a + params.b,
    })
}

#[tool(description = "Multiplies two numbers")]
fn multiply(params: MultiplyParams) -> Result<MathResult, Box<dyn Error + Send + Sync>> {
    tracing::debug!("   ✖️  Multiplying {} × {}", params.x, params.y);
    Ok(MathResult {
        result: params.x * params.y,
    })
}

#[tool(description = "Raises base to the power of exponent")]
fn power(params: PowerParams) -> Result<MathResult, Box<dyn Error + Send + Sync>> {
    tracing::debug!("   ⚡ Calculating {}^{}", params.base, params.exp);
    Ok(MathResult {
        result: params.base.powf(params.exp),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing::debug!("🎯 Tool Calling with Parameter Structs");
    tracing::debug!("======================================\n");

    let mut registry = ToolRegistry::new();

    // Register tools
    tracing::debug!("📦 Registering tools...");
    registry.register(Box::new(AddSimpleTool))?;
    registry.register(Box::new(MultiplyTool))?;
    registry.register(Box::new(PowerTool))?;

    tracing::debug!("   ✅ Registered {} tools\n", registry.len());

    tracing::debug!("💡 Key Point: Each tool uses a params struct for type safety!");
    tracing::debug!("   AddParams, MultiplyParams, PowerParams define the schemas\n");

    // Show generated schemas
    tracing::debug!("🔍 Auto-generated schemas:");
    for def in registry.tool_definitions() {
        tracing::debug!("\n   📝 {}", def.name);
        tracing::debug!("      {}", serde_json::to_string_pretty(&def.parameters)?);
    }

    // Execute tools
    tracing::debug!("🚀 Executing tools:\n");

    let add_result = registry.execute(&ToolCallExec::new(
        "1",
        "add_simple",
        json!({"a": 15, "b": 27}),
    ));
    tracing::debug!("   ✅ add_simple(15, 27) = {}", add_result.content);

    let mul_result = registry.execute(&ToolCallExec::new("2", "multiply", json!({"x": 6, "y": 7})));
    tracing::debug!("   ✅ multiply(6, 7) = {}", mul_result.content);

    let pow_result = registry.execute(&ToolCallExec::new(
        "3",
        "power",
        json!({"base": 2, "exp": 10}),
    ));
    tracing::debug!("   ✅ power(2, 10) = {}", pow_result.content);

    tracing::debug!("\n🎉 Tool calling with parameter structs complete!");
    tracing::debug!("\n💡 Benefits:");
    tracing::debug!("   ✅ Type-safe parameter definitions");
    tracing::debug!("   ✅ Automatic JSON schema generation from Rust types");
    tracing::debug!("   ✅ Clear documentation via doc comments");
    tracing::debug!("   ✅ Perfect for tools with multiple parameters");

    Ok(())
}
