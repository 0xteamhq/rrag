//! Comparing All Three Tool Calling Approaches
//!
//! This example demonstrates all three ways to create tools in RSLLM:
//! 1. Manual JSON (most control)
//! 2. SchemaBasedTool (automatic schema)
//! 3. #[tool] macro (easiest)
//!
//! Run with: cargo run -p rsllm --example tool_calling_all_approaches --features ollama

use rsllm::simple_tool;
use rsllm::tool;
use rsllm::tools::{SchemaBasedTool, Tool, ToolCall as ToolCallExec, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// APPROACH 1: Manual JSON (Traditional)
// ============================================================================

struct ManualCalculator;

impl Tool for ManualCalculator {
    fn name(&self) -> &str {
        "manual_calc"
    }

    fn description(&self) -> &str {
        "Manual JSON approach - full control"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        })
    }

    fn execute(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let a = args["a"].as_f64().ok_or("Missing 'a'")?;
        let b = args["b"].as_f64().ok_or("Missing 'b'")?;
        Ok(json!({"result": a + b}))
    }
}

// ============================================================================
// APPROACH 2: SchemaBasedTool (Automatic Schema)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct AutoCalcParams {
    /// First number
    pub a: f64,
    /// Second number
    pub b: f64,
}

struct AutoCalculator;

impl SchemaBasedTool for AutoCalculator {
    type Params = AutoCalcParams;

    fn name(&self) -> &str {
        "auto_calc"
    }

    fn description(&self) -> &str {
        "SchemaBasedTool approach - automatic schema generation"
    }

    fn execute_typed(
        &self,
        params: Self::Params,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        Ok(json!({"result": params.a + params.b}))
    }
}

// ============================================================================
// APPROACH 3: #[tool] Macro (Easiest!)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct MacroCalcParams {
    /// First number
    pub a: f64,
    /// Second number
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct MacroCalcResult {
    /// The result of the addition
    pub result: f64,
}

#[tool(description = "Macro approach - easiest to use")]
fn macro_calc(params: MacroCalcParams) -> Result<MacroCalcResult, Box<dyn Error + Send + Sync>> {
    Ok(MacroCalcResult {
        result: params.a + params.b,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing::debug!("🔬 Comparing All Three Tool Calling Approaches");
    tracing::debug!("================================================\n");

    // Create registry
    let mut registry = ToolRegistry::new();

    // Register all three approaches
    tracing::debug!("📦 Registering tools from all three approaches...\n");

    // Approach 1: Manual
    registry.register(Box::new(ManualCalculator))?;
    tracing::debug!("   ✅ Approach 1 (Manual JSON): manual_calc");

    // Approach 2: SchemaBasedTool
    registry.register(Box::new(AutoCalculator))?;
    tracing::debug!("   ✅ Approach 2 (SchemaBasedTool): auto_calc");

    // Approach 3: Macro
    registry.register(Box::new(MacroCalcTool))?;
    tracing::debug!("   ✅ Approach 3 (#[tool] macro): macro_calc");

    // Approach 4: simple_tool! macro
    let quick_tool = simple_tool!(
        name: "quick_calc",
        description: "Quick inline tool with simple_tool! macro",
        parameters: json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        }),
        execute: |args| {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            json!({"result": a + b})
        }
    );
    registry.register(quick_tool)?;
    tracing::debug!("   ✅ Approach 4 (simple_tool! macro): quick_calc");

    tracing::debug!("\n   Total: {} tools registered\n", registry.len());

    // Execute all tools with the same input
    tracing::debug!("🚀 Executing all tools with same input: a=10, b=20\n");

    for tool_name in &["manual_calc", "auto_calc", "macro_calc", "quick_calc"] {
        let call = ToolCallExec::new(
            format!("call-{}", tool_name),
            *tool_name,
            json!({"a": 10, "b": 20}),
        );

        let result = registry.execute(&call);
        if result.success {
            tracing::debug!(
                "   ✅ {}: {}",
                tool_name,
                serde_json::to_string(&result.content)?
            );
        } else {
            tracing::debug!("   ❌ {}: {:?}", tool_name, result.error);
        }
    }

    tracing::debug!("\n📊 Code Comparison:");
    tracing::debug!("┌─────────────────────┬──────────────┬─────────────┬────────────┐");
    tracing::debug!("│ Approach            │ Lines of Code│ Type Safety │ Auto-Schema│");
    tracing::debug!("├─────────────────────┼──────────────┼─────────────┼────────────┤");
    tracing::debug!("│ Manual JSON         │ ~50 lines    │ ❌ No       │ ❌ No      │");
    tracing::debug!("│ simple_tool! macro  │ ~20 lines    │ ❌ No       │ ❌ No      │");
    tracing::debug!("│ SchemaBasedTool     │ ~30 lines    │ ✅ Yes      │ ✅ Yes     │");
    tracing::debug!("│ #[tool] macro       │ ~15 lines    │ ✅ Yes      │ ✅ Yes     │");
    tracing::debug!("└─────────────────────┴──────────────┴─────────────┴────────────┘");

    tracing::debug!("\n✨ Recommendations:");
    tracing::debug!("   🎯 Use #[tool] macro:      For new tools (easiest!)");
    tracing::debug!("   🔧 Use SchemaBasedTool:    For complex tools with custom logic");
    tracing::debug!("   ⚡ Use simple_tool! macro:  For quick inline tools");
    tracing::debug!("   📝 Use Manual JSON:        When you need full schema control");

    tracing::debug!("\n🎉 All four approaches work together perfectly!");

    Ok(())
}
