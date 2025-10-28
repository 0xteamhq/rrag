//! Tool Calling with #[tool] Macro - The Easiest Way!
//!
//! This example shows how to use the #[tool] attribute macro for
//! the absolute easiest tool definition. Just write a function!
//!
//! Run with: cargo run -p rsllm --example tool_calling_with_macro --features ollama

use rsllm::tool;
use rsllm::tools::{ToolCall as ToolCallExec, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// EXAMPLE 1: Simple Function-based Tool with Macro
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct AddParams {
    /// First number to add
    pub a: f64,
    /// Second number to add
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct AddResult {
    /// The sum of the two numbers
    pub sum: f64,
}

#[tool(description = "Adds two numbers together")]
fn add_numbers(params: AddParams) -> Result<AddResult, Box<dyn Error + Send + Sync>> {
    println!("   🔢 Adding {} + {}", params.a, params.b);
    Ok(AddResult {
        sum: params.a + params.b,
    })
}

// ============================================================================
// EXAMPLE 2: String Manipulation Tool with Custom Name
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct StringOpParams {
    /// The input text
    pub text: String,
    /// Operation to perform
    pub operation: StringOperation,
}

#[derive(JsonSchema, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StringOperation {
    Uppercase,
    Lowercase,
    Reverse,
    Length,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct StringOpResult {
    /// The result of the operation
    pub result: String,
    /// Original length
    pub original_length: usize,
}

#[tool(
    name = "string_ops",
    description = "Performs various string operations"
)]
fn string_operations(
    params: StringOpParams,
) -> Result<StringOpResult, Box<dyn Error + Send + Sync>> {
    let result = match params.operation {
        StringOperation::Uppercase => params.text.to_uppercase(),
        StringOperation::Lowercase => params.text.to_lowercase(),
        StringOperation::Reverse => params.text.chars().rev().collect(),
        StringOperation::Length => params.text.len().to_string(),
    };

    Ok(StringOpResult {
        result,
        original_length: params.text.len(),
    })
}

// ============================================================================
// EXAMPLE 3: Data Processing Tool
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct FilterParams {
    /// List of numbers to filter
    pub numbers: Vec<i32>,
    /// Minimum value (inclusive)
    #[serde(default)]
    pub min: Option<i32>,
    /// Maximum value (inclusive)
    #[serde(default)]
    pub max: Option<i32>,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct FilterResult {
    /// Filtered numbers
    pub filtered: Vec<i32>,
    /// Count of numbers that passed the filter
    pub count: usize,
    /// Count of numbers that were filtered out
    pub removed: usize,
}

#[tool(description = "Filters a list of numbers by min/max values")]
fn filter_numbers(params: FilterParams) -> Result<FilterResult, Box<dyn Error + Send + Sync>> {
    let original_count = params.numbers.len();

    let filtered: Vec<i32> = params
        .numbers
        .into_iter()
        .filter(|&n| {
            if let Some(min) = params.min {
                if n < min {
                    return false;
                }
            }
            if let Some(max) = params.max {
                if n > max {
                    return false;
                }
            }
            true
        })
        .collect();

    let count = filtered.len();
    let removed = original_count - count;

    Ok(FilterResult {
        filtered,
        count,
        removed,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🎯 RSLLM #[tool] Macro - The Easiest Way!");
    println!("==========================================\n");

    // Step 1: Create registry
    println!("📦 Step 1: Creating tool registry...");
    let mut registry = ToolRegistry::new();

    println!("   💡 Notice: Tools defined with just #[tool] attribute!\n");

    // Step 2: Register the macro-generated tools
    println!("🔧 Step 2: Registering macro-generated tools...");
    registry.register(Box::new(AddNumbersTool))?;
    registry.register(Box::new(StringOperationsTool))?;
    registry.register(Box::new(FilterNumbersTool))?;

    println!("   ✅ Registered {} tools\n", registry.len());

    // Step 3: Show what the macro generated
    println!("🔍 Step 3: What the #[tool] macro generated:");
    for def in registry.tool_definitions() {
        println!("   📝 {}: {}", def.name, def.description);
    }
    println!();

    // Step 4: Execute tools
    println!("🚀 Step 4: Executing macro-defined tools\n");

    // Example 1: Add numbers
    println!("   Example 1: add_numbers");
    let add_call = ToolCallExec::new("call-1", "add_numbers", json!({"a": 123, "b": 456}));
    let add_result = registry.execute(&add_call);
    if add_result.success {
        println!(
            "   ✅ {}",
            serde_json::to_string_pretty(&add_result.content)?
        );
    }
    println!();

    // Example 2: String operations
    println!("   Example 2: string_ops");
    let string_call = ToolCallExec::new(
        "call-2",
        "string_ops",
        json!({"text": "Hello Rust!", "operation": "uppercase"}),
    );
    let string_result = registry.execute(&string_call);
    if string_result.success {
        println!(
            "   ✅ {}",
            serde_json::to_string_pretty(&string_result.content)?
        );
    }
    println!();

    // Example 3: Filter numbers
    println!("   Example 3: filter_numbers");
    let filter_call = ToolCallExec::new(
        "call-3",
        "filter_numbers",
        json!({"numbers": [1, 5, 10, 15, 20, 25, 30], "min": 10, "max": 25}),
    );
    let filter_result = registry.execute(&filter_call);
    if filter_result.success {
        println!(
            "   ✅ {}",
            serde_json::to_string_pretty(&filter_result.content)?
        );
    }
    println!();

    println!("🎉 Macro-based tool calling complete!\n");
    println!("💡 Compare the approaches:");
    println!("   📝 Manual JSON:          ~50 lines per tool");
    println!("   ✨ SchemaBasedTool:      ~30 lines per tool");
    println!("   🎯 #[tool] macro:        ~15 lines per tool!");
    println!();
    println!("   ⚡ The #[tool] macro is 3x less code!");
    println!();
    println!("📖 Code breakdown:");
    println!("   1. Define your params struct with #[derive(JsonSchema)]");
    println!("   2. Define your result struct with #[derive(JsonSchema)]");
    println!("   3. Add #[tool(description = \"...\")] to your function");
    println!("   4. Write your function logic - that's it!");

    Ok(())
}
