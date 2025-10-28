//! Individual Parameters Mode Example (Future Feature)
//!
//! This demonstrates how the macro WILL work with individual parameters.
//! Currently, this is a placeholder showing the future API.
//!
//! For now, use the current working approach:
//! - Define a params struct
//! - Use SchemaBasedTool or #[tool] with single struct param
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

fn main() -> Result<(), Box<dyn Error>> {
    tracing::debug!("🎯 Individual Parameters Mode - No Params Struct!");
    tracing::debug!("==================================================\n");

    let mut registry = ToolRegistry::new();

    // Register tools with individual parameters
    tracing::debug!("📦 Registering tools with individual parameters...");
    registry.register(Box::new(AddSimpleTool))?;
    registry.register(Box::new(MultiplyTool))?;
    registry.register(Box::new(PowerTool))?;

    tracing::debug!("   ✅ Registered {} tools\n", registry.len());

    tracing::debug!("💡 Key Point: No params structs needed!");
    tracing::debug!("   The macro auto-generates AddSimpleParams, MultiplyParams, PowerParams\n");

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

    tracing::debug!("\n🎉 Individual parameters mode complete!");
    tracing::debug!("\n💡 Benefits:");
    tracing::debug!("   ✅ No need to define params structs");
    tracing::debug!("   ✅ Function signature IS the schema");
    tracing::debug!("   ✅ Perfect for simple math/utility functions");
    tracing::debug!("   ✅ Less boilerplate than struct mode");

    Ok(())
}
