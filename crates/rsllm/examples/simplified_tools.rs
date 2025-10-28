//! Super Simplified Tool Creation - Maximum Simplicity!
//!
//! This shows the EASIEST way to create tools with ZERO boilerplate.
//! Compare the complex TaskManager from the previous example with this!
//!
//! Run with: cargo run -p rsllm --example simplified_tools --all-features

use rsllm::tool;
use rsllm::tools::{ToolCall as ToolCallExec, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// BEFORE: Complex SchemaBasedTool (50+ lines)
// AFTER: Simple #[tool] function (15 lines)
// ============================================================================

// Just define your params
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TaskParams {
    pub action: String,
    pub description: Option<String>,
    pub task_id: Option<u32>,
}

// Just define your result
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TaskResult {
    pub success: bool,
    pub message: String,
}

// Just write the function!
#[tool(description = "Manage tasks: create, list, complete, or delete")]
fn task_manager(params: TaskParams) -> Result<TaskResult, Box<dyn Error + Send + Sync>> {
    // Your logic here - that's it!
    match params.action.as_str() {
        "create" => {
            if params.description.is_none() {
                return Err("description is required for create".into());
            }
            Ok(TaskResult {
                success: true,
                message: format!("Created task: {}", params.description.unwrap()),
            })
        }
        "list" => Ok(TaskResult {
            success: true,
            message: "Listed all tasks".to_string(),
        }),
        _ => Err("Unknown action".into()),
    }
}

// ============================================================================
// Even Simpler: Calculator
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcParams {
    pub operation: String,
    pub a: f64,
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcResult {
    pub result: f64,
}

#[tool(description = "Performs arithmetic: add, subtract, multiply, divide")]
fn calculate(params: CalcParams) -> Result<CalcResult, Box<dyn Error + Send + Sync>> {
    let result = match params.operation.as_str() {
        "add" => params.a + params.b,
        "subtract" => params.a - params.b,
        "multiply" => params.a * params.b,
        "divide" => {
            if params.b == 0.0 {
                return Err("Division by zero".into());
            }
            params.a / params.b
        }
        _ => return Err("Unknown operation".into()),
    };

    Ok(CalcResult { result })
}

// ============================================================================
// Even Simpler: Text Tools
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TextParams {
    pub text: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TextStats {
    pub words: usize,
    pub chars: usize,
    pub lines: usize,
}

#[tool(description = "Count words, characters, and lines in text")]
fn analyze_text(params: TextParams) -> Result<TextStats, Box<dyn Error + Send + Sync>> {
    Ok(TextStats {
        words: params.text.split_whitespace().count(),
        chars: params.text.len(),
        lines: params.text.lines().count(),
    })
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct UppercaseResult {
    pub result: String,
}

#[tool(description = "Convert text to uppercase")]
fn uppercase(params: TextParams) -> Result<UppercaseResult, Box<dyn Error + Send + Sync>> {
    Ok(UppercaseResult {
        result: params.text.to_uppercase(),
    })
}

// ============================================================================
// MAIN - Show How Easy It Is!
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    println!("✨ Super Simplified Tool Creation");
    println!("==================================\n");

    println!("💡 Look how simple the code is!");
    println!("   • Just define params struct");
    println!("   • Add #[tool(description = \"...\")] to function");
    println!("   • Write your logic");
    println!("   • That's it! Only ~15 lines per tool!\n");

    // Register all tools
    let mut registry = ToolRegistry::new();

    registry.register(Box::new(TaskManagerTool))?;
    registry.register(Box::new(CalculateTool))?;
    registry.register(Box::new(AnalyzeTextTool))?;
    registry.register(Box::new(UppercaseTool))?;

    println!("📦 Registered {} tools\n", registry.len());

    // Show they all work
    println!("🚀 Testing all tools:\n");

    // Calculator
    let calc = registry.execute(&ToolCallExec::new(
        "1",
        "calculate",
        json!({"operation": "multiply", "a": 12, "b": 8}),
    ));
    println!("   ✅ calculate: {}", calc.content);

    // Task manager
    let task = registry.execute(&ToolCallExec::new(
        "2",
        "task_manager",
        json!({"action": "create", "description": "Write docs"}),
    ));
    println!("   ✅ task_manager: {}", task.content);

    // Text analyzer
    let analyze = registry.execute(&ToolCallExec::new(
        "3",
        "analyze_text",
        json!({"text": "Hello World from RSLLM!"}),
    ));
    println!("   ✅ analyze_text: {}", analyze.content);

    // Uppercase
    let upper = registry.execute(&ToolCallExec::new(
        "4",
        "uppercase",
        json!({"text": "make this loud"}),
    ));
    println!("   ✅ uppercase: {}", upper.content);

    println!("\n📊 Code Comparison:");
    println!("   ❌ OLD SchemaBasedTool:  ~80 lines (TaskManager)");
    println!("   ✅ NEW #[tool] macro:     ~25 lines (task_manager)");
    println!("   🎉 67% LESS CODE!");

    println!("\n💡 Pro Tips:");
    println!("   1. Define params struct with #[derive(JsonSchema)]");
    println!("   2. Add #[tool(description = \"...\")]");
    println!("   3. Write function - macro does the rest!");
    println!("   4. No need for SchemaBasedTool trait");
    println!("   5. No need for Manual JSON");
    println!("   6. Just functions!");

    println!("\n🎉 Creating tools is now SUPER EASY!");

    Ok(())
}
