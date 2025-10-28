//! Complete Tool Calling Application Example
//!
//! This example demonstrates ALL tool calling features in a real-world scenario:
//! - Multiple tools using different approaches
//! - Tool registry management
//! - Batch execution
//! - Error handling
//! - Validation
//! - Type safety
//! - Automatic schema generation
//!
//! Scenario: A productivity assistant with various tools
//!
//! Run with: cargo run -p rsllm --example complete_tool_application --features ollama

use rsllm::simple_tool;
use rsllm::tool;
use rsllm::tools::{SchemaBasedTool, Tool, ToolCall as ToolCallExec, ToolRegistry, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// TOOL 1: Calculator (Using #[tool] Macro - EASIEST!)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct CalculatorParams {
    /// The arithmetic operation to perform
    operation: Operation,
    /// First number
    a: f64,
    /// Second number
    b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalculatorResult {
    /// The result of the calculation
    pub result: f64,
    /// The operation performed
    pub operation: String,
}

#[tool(description = "Performs basic arithmetic operations")]
fn calculator(params: CalculatorParams) -> Result<CalculatorResult, Box<dyn Error + Send + Sync>> {
    let result = match params.operation {
        Operation::Add => params.a + params.b,
        Operation::Subtract => params.a - params.b,
        Operation::Multiply => params.a * params.b,
        Operation::Divide => {
            if params.b == 0.0 {
                return Err("Cannot divide by zero".into());
            }
            params.a / params.b
        }
    };

    Ok(CalculatorResult {
        result,
        operation: format!("{:?}", params.operation),
    })
}

// ============================================================================
// TOOL 2: Task Manager (Using SchemaBasedTool - More Control)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
pub struct TaskParams {
    /// Action to perform on tasks
    action: TaskAction,
    /// Task description (for create/update)
    #[serde(default)]
    description: Option<String>,
    /// Task ID (for update/complete/delete)
    #[serde(default)]
    task_id: Option<u32>,
    /// Priority level
    #[serde(default = "default_priority")]
    priority: Priority,
}

fn default_priority() -> Priority {
    Priority::Medium
}

#[derive(JsonSchema, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum TaskAction {
    Create,
    List,
    Complete,
    Delete,
}

#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low,
    Medium,
    High,
}

pub struct TaskManager {
    tasks: std::sync::Mutex<Vec<Task>>,
}

#[derive(Debug, Clone, Serialize)]
struct Task {
    id: u32,
    description: String,
    priority: Priority,
    completed: bool,
}

impl TaskManager {
    fn new() -> Self {
        Self {
            tasks: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl SchemaBasedTool for TaskManager {
    type Params = TaskParams;

    fn name(&self) -> &str {
        "task_manager"
    }

    fn description(&self) -> &str {
        "Manage tasks: create, list, complete, or delete tasks"
    }

    fn validate_typed(&self, params: &Self::Params) -> Result<(), Box<dyn Error + Send + Sync>> {
        match params.action {
            TaskAction::Create => {
                if params.description.is_none() {
                    return Err("description is required for create action".into());
                }
            }
            TaskAction::Complete | TaskAction::Delete => {
                if params.task_id.is_none() {
                    return Err("task_id is required for this action".into());
                }
            }
            TaskAction::List => {}
        }
        Ok(())
    }

    fn execute_typed(
        &self,
        params: Self::Params,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let mut tasks = self.tasks.lock().unwrap();

        match params.action {
            TaskAction::Create => {
                let id = tasks.len() as u32 + 1;
                let task = Task {
                    id,
                    description: params.description.unwrap(),
                    priority: params.priority,
                    completed: false,
                };
                tasks.push(task.clone());
                Ok(json!({
                    "success": true,
                    "message": "Task created",
                    "task": task
                }))
            }
            TaskAction::List => Ok(json!({
                "tasks": tasks.clone(),
                "count": tasks.len()
            })),
            TaskAction::Complete => {
                let task_id = params.task_id.unwrap();
                if let Some(task) = tasks.iter_mut().find(|t| t.id == task_id) {
                    task.completed = true;
                    Ok(json!({"success": true, "message": "Task completed"}))
                } else {
                    Err(format!("Task {} not found", task_id).into())
                }
            }
            TaskAction::Delete => {
                let task_id = params.task_id.unwrap();
                if let Some(pos) = tasks.iter().position(|t| t.id == task_id) {
                    tasks.remove(pos);
                    Ok(json!({"success": true, "message": "Task deleted"}))
                } else {
                    Err(format!("Task {} not found", task_id).into())
                }
            }
        }
    }
}

// ============================================================================
// TOOL 3: Text Analyzer (Using simple_tool! Macro - Quick Inline)
// ============================================================================

fn create_text_analyzer() -> Box<dyn Tool> {
    simple_tool!(
        name: "analyze_text",
        description: "Analyzes text and returns statistics",
        parameters: json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze"
                }
            },
            "required": ["text"]
        }),
        execute: |args| {
            let text = args["text"].as_str().unwrap_or("");
            json!({
                "length": text.len(),
                "words": text.split_whitespace().count(),
                "lines": text.lines().count(),
                "chars": text.chars().count(),
                "uppercase": text.chars().filter(|c| c.is_uppercase()).count(),
                "lowercase": text.chars().filter(|c| c.is_lowercase()).count(),
                "digits": text.chars().filter(|c| c.is_numeric()).count()
            })
        }
    )
}

// ============================================================================
// TOOL 4: Data Converter (Manual JSON - Full Control)
// ============================================================================

struct DataConverter;

impl Tool for DataConverter {
    fn name(&self) -> &str {
        "convert_data"
    }

    fn description(&self) -> &str {
        "Converts data between different formats"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The data to convert"
                },
                "from_format": {
                    "type": "string",
                    "enum": ["json", "csv", "text"],
                    "description": "Source format"
                },
                "to_format": {
                    "type": "string",
                    "enum": ["json", "csv", "text"],
                    "description": "Target format"
                }
            },
            "required": ["data", "from_format", "to_format"]
        })
    }

    fn execute(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let data = args["data"].as_str().ok_or("Missing data")?;
        let from_format = args["from_format"].as_str().ok_or("Missing from_format")?;
        let to_format = args["to_format"].as_str().ok_or("Missing to_format")?;

        // Simple mock conversion
        Ok(json!({
            "converted": format!("Converted from {} to {}: {}", from_format, to_format, data),
            "from": from_format,
            "to": to_format,
            "length": data.len()
        }))
    }
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 Complete Tool Calling Application");
    println!("=====================================\n");
    println!("Scenario: Productivity Assistant with Multiple Tools\n");

    // ========================================================================
    // STEP 1: Initialize Tool Registry
    // ========================================================================

    println!("📦 STEP 1: Initialize Tool Registry");
    println!("────────────────────────────────────\n");

    let mut registry = ToolRegistry::new();

    // Register all tools using different approaches
    registry.register(Box::new(CalculatorTool))?;
    println!("   ✅ Calculator (macro)");

    registry.register(Box::new(TaskManager::new()))?;
    println!("   ✅ Task Manager (SchemaBasedTool)");

    registry.register(create_text_analyzer())?;
    println!("   ✅ Text Analyzer (simple_tool!)");

    registry.register(Box::new(DataConverter))?;
    println!("   ✅ Data Converter (manual JSON)");

    println!("\n   📊 Total tools: {}\n", registry.len());

    // ========================================================================
    // STEP 2: Show Tool Definitions (What gets sent to LLM)
    // ========================================================================

    println!("📋 STEP 2: Tool Definitions for LLM");
    println!("────────────────────────────────────\n");

    for def in registry.tool_definitions() {
        println!("   📝 {}", def.name);
        println!("      Description: {}", def.description);
        println!("      Type: {}", def.tool_type);
    }
    println!();

    // ========================================================================
    // STEP 3: Execute Individual Tools
    // ========================================================================

    println!("🔧 STEP 3: Execute Individual Tools");
    println!("────────────────────────────────────\n");

    // 3.1: Calculator
    println!("   3.1: Calculator - Multiply 15 * 8");
    let calc_result = registry.execute(&ToolCallExec::new(
        "calc-1",
        "calculator",
        json!({"operation": "multiply", "a": 15, "b": 8}),
    ));
    print_tool_result(&calc_result);

    // 3.2: Task Manager - Create task
    println!("\n   3.2: Task Manager - Create new task");
    let create_task = registry.execute(&ToolCallExec::new(
        "task-1",
        "task_manager",
        json!({
            "action": "create",
            "description": "Review pull requests",
            "priority": "high"
        }),
    ));
    print_tool_result(&create_task);

    // 3.3: Task Manager - List tasks
    println!("\n   3.3: Task Manager - List all tasks");
    let list_tasks = registry.execute(&ToolCallExec::new(
        "task-2",
        "task_manager",
        json!({"action": "list"}),
    ));
    print_tool_result(&list_tasks);

    // 3.4: Text Analyzer
    println!("\n   3.4: Text Analyzer - Analyze sample text");
    let analyze_result = registry.execute(&ToolCallExec::new(
        "text-1",
        "analyze_text",
        json!({"text": "Hello World! This is RSLLM tool calling. Very COOL!"}),
    ));
    print_tool_result(&analyze_result);

    // 3.5: Data Converter
    println!("\n   3.5: Data Converter - Convert JSON to CSV");
    let convert_result = registry.execute(&ToolCallExec::new(
        "convert-1",
        "convert_data",
        json!({
            "data": "{\"name\": \"John\", \"age\": 30}",
            "from_format": "json",
            "to_format": "csv"
        }),
    ));
    print_tool_result(&convert_result);

    // ========================================================================
    // STEP 4: Batch Execution
    // ========================================================================

    println!("\n\n🔄 STEP 4: Batch Tool Execution");
    println!("────────────────────────────────────\n");

    let batch_calls = vec![
        ToolCallExec::new(
            "batch-1",
            "calculator",
            json!({"operation": "add", "a": 100, "b": 50}),
        ),
        ToolCallExec::new(
            "batch-2",
            "calculator",
            json!({"operation": "divide", "a": 144, "b": 12}),
        ),
        ToolCallExec::new(
            "batch-3",
            "task_manager",
            json!({"action": "create", "description": "Test batch execution"}),
        ),
    ];

    println!("   Executing {} tools in batch...", batch_calls.len());
    let batch_results = registry.execute_batch(&batch_calls);

    for (i, result) in batch_results.iter().enumerate() {
        println!("\n   Batch #{}: {}", i + 1, result.tool_name);
        if result.success {
            println!(
                "      ✅ Success: {}",
                serde_json::to_string(&result.content)?
            );
        } else {
            println!("      ❌ Error: {}", result.error.as_ref().unwrap());
        }
    }

    // ========================================================================
    // STEP 5: Error Handling
    // ========================================================================

    println!("\n\n⚠️  STEP 5: Error Handling");
    println!("────────────────────────────────────\n");

    // 5.1: Division by zero
    println!("   5.1: Testing division by zero");
    let error_result = registry.execute(&ToolCallExec::new(
        "error-1",
        "calculator",
        json!({"operation": "divide", "a": 10, "b": 0}),
    ));
    print_tool_result(&error_result);

    // 5.2: Invalid tool name
    println!("\n   5.2: Testing invalid tool name");
    let invalid_tool =
        registry.execute(&ToolCallExec::new("error-2", "nonexistent_tool", json!({})));
    print_tool_result(&invalid_tool);

    // 5.3: Missing required parameter
    println!("\n   5.3: Testing missing required parameter");
    let missing_param = registry.execute(&ToolCallExec::new(
        "error-3",
        "task_manager",
        json!({"action": "create"}), // Missing description
    ));
    print_tool_result(&missing_param);

    // ========================================================================
    // STEP 6: Tool Discovery & Introspection
    // ========================================================================

    println!("\n\n🔍 STEP 6: Tool Discovery");
    println!("────────────────────────────────────\n");

    println!("   Available tools:");
    for tool_name in registry.tool_names() {
        println!("      - {}", tool_name);
    }

    println!("\n   Registry status:");
    println!("      Total tools: {}", registry.len());
    println!("      Is empty: {}", registry.is_empty());

    // Check if specific tools exist
    println!("\n   Tool availability checks:");
    println!(
        "      calculator exists: {}",
        registry.contains("calculator")
    );
    println!(
        "      task_manager exists: {}",
        registry.contains("task_manager")
    );
    println!(
        "      nonexistent exists: {}",
        registry.contains("nonexistent")
    );

    // ========================================================================
    // STEP 7: Complete Workflow Example
    // ========================================================================

    println!("\n\n🎬 STEP 7: Complete Workflow");
    println!("────────────────────────────────────\n");
    println!("   Simulating: User asks 'Calculate 25% of my 5 tasks'\n");

    // Step 1: List tasks
    println!("   → Calling task_manager to count tasks");
    let count_tasks = registry.execute(&ToolCallExec::new(
        "workflow-1",
        "task_manager",
        json!({"action": "list"}),
    ));

    let task_count = if count_tasks.success {
        count_tasks.content["count"].as_u64().unwrap_or(0)
    } else {
        0
    };
    println!("      Found {} tasks", task_count);

    // Step 2: Calculate 25%
    println!("\n   → Calling calculator to compute 25% of {}", task_count);
    let calc_25_percent = registry.execute(&ToolCallExec::new(
        "workflow-2",
        "calculator",
        json!({
            "operation": "multiply",
            "a": task_count,
            "b": 0.25
        }),
    ));

    if calc_25_percent.success {
        let result = calc_25_percent.content["result"].as_f64().unwrap_or(0.0);
        println!("      25% of {} tasks = {} tasks", task_count, result);
    }

    // Step 3: Create analysis report
    println!("\n   → Creating analysis report");
    let report = format!(
        "Task Analysis: You have {} tasks. 25% completion would be {:.1} tasks.",
        task_count,
        task_count as f64 * 0.25
    );

    let analyze_report = registry.execute(&ToolCallExec::new(
        "workflow-3",
        "analyze_text",
        json!({"text": report}),
    ));

    if analyze_report.success {
        println!(
            "      Report stats: {} words, {} chars",
            analyze_report.content["words"], analyze_report.content["chars"]
        );
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================

    println!("\n\n📊 APPLICATION SUMMARY");
    println!("══════════════════════════════════════════════════\n");

    println!("✅ Features Demonstrated:");
    println!("   1. ✅ Multiple tools with different approaches");
    println!("   2. ✅ #[tool] macro (easiest - auto-generates everything)");
    println!("   3. ✅ SchemaBasedTool (automatic schema, custom logic)");
    println!("   4. ✅ simple_tool! macro (quick inline tools)");
    println!("   5. ✅ Manual JSON (full control)");
    println!("   6. ✅ Type-safe parameter handling");
    println!("   7. ✅ Automatic JSON schema generation");
    println!("   8. ✅ Custom validation logic");
    println!("   9. ✅ Batch execution");
    println!("   10. ✅ Error handling");
    println!("   11. ✅ Tool discovery & introspection");
    println!("   12. ✅ Multi-step workflows");

    println!("\n💡 Key Takeaways:");
    println!("   • Use #[tool] macro for new tools (15 lines per tool)");
    println!("   • Use SchemaBasedTool for stateful/complex tools");
    println!("   • Use simple_tool! for quick prototypes");
    println!("   • All approaches work together seamlessly");
    println!("   • Zero manual JSON schema writing needed");
    println!("   • Full type safety throughout");

    println!("\n🎉 Complete tool calling system ready for production!");

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn print_tool_result(result: &ToolResult) {
    if result.success {
        match serde_json::to_string_pretty(&result.content) {
            Ok(json_str) => println!("      ✅ Success:\n{}", indent(&json_str, 9)),
            Err(_) => println!("      ✅ Success: {:?}", result.content),
        }
    } else {
        println!("      ❌ Error: {}", result.error.as_ref().unwrap());
    }
}

fn indent(text: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", prefix, line))
        .collect::<Vec<_>>()
        .join("\n")
}
