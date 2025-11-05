//! Simple Complete Application - All Features, Minimal Code!
//!
//! This is the SIMPLIFIED version of complete_tool_application.rs
//! Same features, 70% less code!
//!
//! Run with: cargo run -p rsllm --example simple_complete_app --all-features

use rsllm::tool;
use rsllm::tools::{ToolCall as ToolCallExec, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// TOOL 1: Calculator - Simple arithmetic
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcParams {
    pub op: String, // "add", "sub", "mul", "div"
    pub a: f64,
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcResult {
    pub result: f64,
}

#[tool(description = "Calculator: add, sub, mul, div")]
fn calc(p: CalcParams) -> Result<CalcResult, Box<dyn Error + Send + Sync>> {
    let r = match p.op.as_str() {
        "add" => p.a + p.b,
        "sub" => p.a - p.b,
        "mul" => p.a * p.b,
        "div" if p.b != 0.0 => p.a / p.b,
        "div" => return Err("Division by zero".into()),
        _ => return Err("Unknown operation".into()),
    };
    Ok(CalcResult { result: r })
}

// ============================================================================
// TOOL 2: Task Manager - CRUD operations
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TaskParams {
    pub action: String, // "create", "list", "done", "delete"
    pub desc: Option<String>,
    pub id: Option<u32>,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TaskResult {
    pub ok: bool,
    pub msg: String,
    pub data: Option<serde_json::Value>,
}

#[tool(description = "Manage tasks: create, list, done, delete")]
fn tasks(p: TaskParams) -> Result<TaskResult, Box<dyn Error + Send + Sync>> {
    match p.action.as_str() {
        "create" => Ok(TaskResult {
            ok: true,
            msg: format!("Created: {}", p.desc.unwrap_or_default()),
            data: Some(json!({"id": 1})),
        }),
        "list" => Ok(TaskResult {
            ok: true,
            msg: "All tasks".into(),
            data: Some(json!({"tasks": [], "count": 0})),
        }),
        "done" => Ok(TaskResult {
            ok: true,
            msg: format!("Completed task {}", p.id.unwrap_or(0)),
            data: None,
        }),
        "delete" => Ok(TaskResult {
            ok: true,
            msg: format!("Deleted task {}", p.id.unwrap_or(0)),
            data: None,
        }),
        _ => Err("Unknown action".into()),
    }
}

// ============================================================================
// TOOL 3: Text Analyzer - Text statistics
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TextParams {
    pub text: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TextResult {
    pub words: usize,
    pub chars: usize,
    pub lines: usize,
}

#[tool(description = "Analyze text statistics")]
fn analyze(p: TextParams) -> Result<TextResult, Box<dyn Error + Send + Sync>> {
    Ok(TextResult {
        words: p.text.split_whitespace().count(),
        chars: p.text.len(),
        lines: p.text.lines().count(),
    })
}

// ============================================================================
// TOOL 4: Data Converter - Format conversion
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct ConvertParams {
    pub data: String,
    pub from: String,
    pub to: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct ConvertResult {
    pub output: String,
}

#[tool(description = "Convert data between formats")]
fn convert(p: ConvertParams) -> Result<ConvertResult, Box<dyn Error + Send + Sync>> {
    Ok(ConvertResult {
        output: format!("Converted {} from {} to {}", p.data, p.from, p.to),
    })
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    tracing::debug!("🚀 Simple Complete Application");
    tracing::debug!("===============================\n");

    // Initialize registry
    let mut registry = ToolRegistry::new();

    // Register all tools (notice the pattern: XxxTool auto-generated!)
    registry.register(Box::new(CalcTool))?;
    registry.register(Box::new(TasksTool))?;
    registry.register(Box::new(AnalyzeTool))?;
    registry.register(Box::new(ConvertTool))?;

    tracing::debug!("✅ Registered {} tools\n", registry.len());

    // Execute tools
    tracing::debug!("🔧 Tool Executions:\n");

    let r1 = registry.execute(&ToolCallExec::new(
        "1",
        "calc",
        json!({"op": "mul", "a": 15, "b": 8}),
    ));
    tracing::debug!("   Calculator: {}", r1.content);

    let r2 = registry.execute(&ToolCallExec::new(
        "2",
        "tasks",
        json!({"action": "create", "desc": "Review PR"}),
    ));
    tracing::debug!("   Tasks: {}", r2.content);

    let r3 = registry.execute(&ToolCallExec::new(
        "3",
        "analyze",
        json!({"text": "Hello World!"}),
    ));
    tracing::debug!("   Analyze: {}", r3.content);

    let r4 = registry.execute(&ToolCallExec::new(
        "4",
        "convert",
        json!({"data": "test", "from": "json", "to": "csv"}),
    ));
    tracing::debug!("   Convert: {}", r4.content);

    // Batch execution
    tracing::debug!("\n🔄 Batch Execution:\n");
    let batch = vec![
        ToolCallExec::new("b1", "calc", json!({"op": "add", "a": 100, "b": 50})),
        ToolCallExec::new("b2", "analyze", json!({"text": "Batch test!"})),
    ];

    for r in registry.execute_batch(&batch) {
        tracing::debug!(
            "   {} → {}",
            r.tool_name,
            if r.success { "✅" } else { "❌" }
        );
    }

    // Error handling
    tracing::debug!("\n⚠️  Error Handling:\n");
    let err = registry.execute(&ToolCallExec::new(
        "e1",
        "calc",
        json!({"op": "div", "a": 10, "b": 0}),
    ));
    tracing::debug!("   Division by zero: {}", err.error.unwrap_or_default());

    tracing::debug!("\n📊 Summary:");
    tracing::debug!("   Total code per tool: ~15-20 lines");
    tracing::debug!("   Total code for 4 tools: ~80 lines");
    tracing::debug!("   vs Old approach: ~300 lines");
    tracing::debug!("   Savings: 75% less code!");

    tracing::debug!("\n🎉 Simple, type-safe, automatic schema generation!");

    Ok(())
}
