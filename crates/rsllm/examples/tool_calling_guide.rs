//! 🛠️ RSLLM Tool Calling - Complete Guide
//!
//! This is THE comprehensive example for tool calling in RSLLM.
//! It covers ALL features, best practices, and common patterns.
//!
//! Run: cargo run -p rsllm --example tool_calling_guide --all-features
//!
//! TABLE OF CONTENTS:
//! 1. Quick Start (13 lines to create a tool!)
//! 2. All 4 Approaches Compared
//! 3. Best Practices (Descriptions & Validation)
//! 4. Advanced Features (Stateful tools, Batch execution)
//! 5. Error Handling
//! 6. Real-World Application
//!
//! 📖 This is the ONLY tool calling example you need!

use rsllm::tools::{SchemaBasedTool, Tool, ToolCall as ToolCallExec, ToolRegistry};
use rsllm::{simple_tool, tool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// SECTION 1: QUICK START - Minimum Code (13 Lines!)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct QuickAddParams {
    /// First number to add
    pub a: f64,
    /// Second number to add
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct QuickAddResult {
    /// The sum of the two numbers
    pub sum: f64,
}

#[tool(description = "Adds two numbers - quickstart example")]
fn quick_add(params: QuickAddParams) -> Result<QuickAddResult, Box<dyn Error + Send + Sync>> {
    Ok(QuickAddResult {
        sum: params.a + params.b,
    })
}

// ============================================================================
// SECTION 2: ALL 4 APPROACHES - Choose What Fits Your Needs
// ============================================================================

// ───────────────────────────────────────────────────────────────────────────
// APPROACH 1: #[tool] Macro (⭐ RECOMMENDED - Easiest!)
// ───────────────────────────────────────────────────────────────────────────

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcParams {
    /// Operation: "add", "subtract", "multiply", or "divide"
    pub operation: String,
    /// First number
    #[schemars(range(min = -1000.0, max = 1000.0))]
    pub a: f64,
    /// Second number
    #[schemars(range(min = -1000.0, max = 1000.0))]
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcResult {
    /// The calculation result
    pub result: f64,
}

#[tool(description = "Calculator using #[tool] macro - easiest approach!")]
fn calculator(params: CalcParams) -> Result<CalcResult, Box<dyn Error + Send + Sync>> {
    let result = match params.operation.as_str() {
        "add" => params.a + params.b,
        "subtract" => params.a - params.b,
        "multiply" => params.a * params.b,
        "divide" if params.b != 0.0 => params.a / params.b,
        "divide" => return Err("Cannot divide by zero".into()),
        _ => return Err(format!("Unknown operation: {}", params.operation).into()),
    };
    Ok(CalcResult { result })
}

// ───────────────────────────────────────────────────────────────────────────
// APPROACH 2: SchemaBasedTool (For Stateful/Complex Tools)
// ───────────────────────────────────────────────────────────────────────────

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CounterParams {
    /// Action: "increment", "decrement", "get", or "reset"
    pub action: String,
    /// Amount to increment/decrement (default: 1)
    #[serde(default = "default_amount")]
    pub amount: i32,
}

fn default_amount() -> i32 {
    1
}

pub struct Counter {
    value: std::sync::Mutex<i32>,
}

impl Counter {
    fn new() -> Self {
        Self {
            value: std::sync::Mutex::new(0),
        }
    }
}

impl SchemaBasedTool for Counter {
    type Params = CounterParams;

    fn name(&self) -> &str {
        "counter"
    }

    fn description(&self) -> &str {
        "Stateful counter - increment, decrement, get, or reset"
    }

    fn execute_typed(
        &self,
        params: Self::Params,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let mut value = self.value.lock().unwrap();

        match params.action.as_str() {
            "increment" => {
                *value += params.amount;
                Ok(json!({"value": *value, "action": "incremented"}))
            }
            "decrement" => {
                *value -= params.amount;
                Ok(json!({"value": *value, "action": "decremented"}))
            }
            "get" => Ok(json!({"value": *value})),
            "reset" => {
                *value = 0;
                Ok(json!({"value": *value, "action": "reset"}))
            }
            _ => Err(format!("Unknown action: {}", params.action).into()),
        }
    }

    fn validate_typed(&self, params: &Self::Params) -> Result<(), Box<dyn Error + Send + Sync>> {
        if params.amount == 0 && (params.action == "increment" || params.action == "decrement") {
            return Err("Amount cannot be zero for increment/decrement".into());
        }
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────────────
// APPROACH 3: simple_tool! Macro (For Quick Inline Tools)
// ───────────────────────────────────────────────────────────────────────────

fn create_text_tool() -> Box<dyn Tool> {
    simple_tool!(
        name: "text_analyzer",
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
                "uppercase": text.chars().filter(|c| c.is_uppercase()).count(),
                "lowercase": text.chars().filter(|c| c.is_lowercase()).count()
            })
        }
    )
}

// ───────────────────────────────────────────────────────────────────────────
// APPROACH 4: Manual JSON (For Full Control)
// ───────────────────────────────────────────────────────────────────────────

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes back the input with manual JSON schema"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back"
                }
            },
            "required": ["message"]
        })
    }

    fn execute(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
        let message = args["message"].as_str().ok_or("Missing message")?;
        Ok(json!({"echo": message, "length": message.len()}))
    }
}

// ============================================================================
// SECTION 3: BEST PRACTICES - Descriptions & Validation
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct BestPracticeParams {
    /// Search query (1-200 characters)
    /// Example: "rust programming tutorial"
    #[schemars(length(min = 1, max = 200))]
    pub query: String,

    /// Maximum results (1-100)
    #[schemars(range(min = 1, max = 100))]
    #[serde(default = "default_limit")]
    pub limit: u32,

    /// Category filter (optional)
    /// Valid: "docs", "code", "issues"
    pub category: Option<Category>,

    /// Include archived results
    #[serde(default)]
    pub include_archived: bool,
}

fn default_limit() -> u32 {
    10
}

#[derive(JsonSchema, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Docs,
    Code,
    Issues,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct SearchResult {
    /// Search results
    pub results: Vec<String>,
    /// Total results found
    pub total: usize,
}

#[tool(description = "Search with best practices: descriptions, validation, defaults")]
fn best_practice_search(
    params: BestPracticeParams,
) -> Result<SearchResult, Box<dyn Error + Send + Sync>> {
    Ok(SearchResult {
        results: vec![format!("Result for: {}", params.query)],
        total: 1,
    })
}

// ============================================================================
// MAIN - Interactive Guide
// ============================================================================

fn main() -> Result<(), Box<dyn Error>> {
    tracing::debug!("╔═══════════════════════════════════════════════════════════╗");
    tracing::debug!("║  🛠️  RSLLM Tool Calling - Complete Guide                 ║");
    tracing::debug!("╚═══════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════
    // SECTION 1: QUICK START
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ 📚 SECTION 1: Quick Start (13 Lines!)                  │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(QuickAddTool))?;

    tracing::debug!("✅ Created a tool in just 13 lines!");
    tracing::debug!("   Code: Define params (4) + result (3) + function (5) = 13 lines");

    let result = registry.execute(&ToolCallExec::new(
        "q1",
        "quick_add",
        json!({"a": 10, "b": 20}),
    ));
    tracing::debug!("   Test: quick_add(10, 20) = {}\n", result.content);

    // ═══════════════════════════════════════════════════════════════════════
    // SECTION 2: ALL 4 APPROACHES
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ 🎯 SECTION 2: All 4 Approaches Compared                │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    // Register all approaches
    registry.register(Box::new(CalculatorTool))?; // Approach 1: #[tool] macro
    registry.register(Box::new(Counter::new()))?; // Approach 2: SchemaBasedTool
    registry.register(create_text_tool())?; // Approach 3: simple_tool!
    registry.register(Box::new(EchoTool))?; // Approach 4: Manual JSON

    tracing::debug!("📊 Comparison Table:");
    tracing::debug!("┌──────────────────┬──────┬───────────┬─────────────┐");
    tracing::debug!("│ Approach         │ Lines│ Type Safe │ Auto Schema │");
    tracing::debug!("├──────────────────┼──────┼───────────┼─────────────┤");
    tracing::debug!("│ #[tool] macro    │  15  │    ✅     │     ✅      │");
    tracing::debug!("│ SchemaBasedTool  │  30  │    ✅     │     ✅      │");
    tracing::debug!("│ simple_tool!     │  20  │    ❌     │     ❌      │");
    tracing::debug!("│ Manual JSON      │  50  │    ❌     │     ❌      │");
    tracing::debug!("└──────────────────┴──────┴───────────┴─────────────┘\n");

    tracing::debug!("💡 Recommendation: Use #[tool] macro for 90% of cases\n");

    // Execute all approaches with same input
    tracing::debug!("🚀 Testing all approaches execute correctly:");
    for tool_name in &["calculator", "counter", "text_analyzer", "echo"] {
        let call = match *tool_name {
            "calculator" => ToolCallExec::new(
                "t1",
                *tool_name,
                json!({"operation": "add", "a": 5, "b": 3}),
            ),
            "counter" => ToolCallExec::new(
                "t2",
                *tool_name,
                json!({"action": "increment", "amount": 5}),
            ),
            "text_analyzer" => ToolCallExec::new("t3", *tool_name, json!({"text": "Hello World"})),
            "echo" => ToolCallExec::new("t4", *tool_name, json!({"message": "test"})),
            _ => continue,
        };

        let result = registry.execute(&call);
        let status = if result.success { "✅" } else { "❌" };
        let error_msg = if result.success {
            String::new()
        } else {
            format!("({})", result.error.unwrap_or_default())
        };
        tracing::debug!("   {} {} {}", status, tool_name, error_msg);
    }
    tracing::debug!();

    // ═══════════════════════════════════════════════════════════════════════
    // SECTION 3: BEST PRACTICES
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ 📝 SECTION 3: Best Practices (Prevent Hallucination!)  │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    registry.register(Box::new(BestPracticeSearchTool))?;

    tracing::debug!("✅ ALWAYS add descriptions using doc comments (///)");
    tracing::debug!("✅ Use #[schemars(...)] attributes for validation:");
    tracing::debug!("   • range(min = X, max = Y) - for numbers");
    tracing::debug!("   • length(min = X, max = Y) - for strings");
    tracing::debug!("   • Use enums for limited value sets");
    tracing::debug!("   • Add #[serde(default)] for optional params\n");

    tracing::debug!("Example schema generated:");
    let search_def = registry
        .tool_definitions()
        .into_iter()
        .find(|d| d.name == "best_practice_search")
        .unwrap();

    tracing::debug!(
        "{}\n",
        serde_json::to_string_pretty(&search_def.parameters)?
    );

    // ═══════════════════════════════════════════════════════════════════════
    // SECTION 4: ADVANCED FEATURES
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ 🔧 SECTION 4: Advanced Features                        │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    // 4.1: Batch Execution
    tracing::debug!("📦 Batch Execution (multiple tools at once):");
    let batch_calls = vec![
        ToolCallExec::new(
            "b1",
            "calculator",
            json!({"operation": "multiply", "a": 6, "b": 7}),
        ),
        ToolCallExec::new(
            "b2",
            "counter",
            json!({"action": "increment", "amount": 10}),
        ),
        ToolCallExec::new("b3", "text_analyzer", json!({"text": "Batch processing!"})),
    ];

    let batch_results = registry.execute_batch(&batch_calls);
    tracing::debug!("   Executed {} tools in batch:", batch_results.len());
    for r in &batch_results {
        tracing::debug!(
            "      {} {}",
            if r.success { "✅" } else { "❌" },
            r.tool_name
        );
    }
    tracing::debug!();

    // 4.2: Tool Discovery
    tracing::debug!("🔍 Tool Discovery & Introspection:");
    tracing::debug!("   Total tools: {}", registry.len());
    tracing::debug!("   Available: {:?}", registry.tool_names());
    tracing::debug!(
        "   'calculator' exists: {}",
        registry.contains("calculator")
    );
    tracing::debug!(
        "   'nonexistent' exists: {}\n",
        registry.contains("nonexistent")
    );

    // 4.3: Stateful Tool
    tracing::debug!("🔄 Stateful Tool (Counter maintains state):");
    let c1 = registry.execute(&ToolCallExec::new(
        "s1",
        "counter",
        json!({"action": "increment", "amount": 5}),
    ));
    tracing::debug!("   Increment by 5: {}", c1.content);

    let c2 = registry.execute(&ToolCallExec::new(
        "s2",
        "counter",
        json!({"action": "increment", "amount": 3}),
    ));
    tracing::debug!("   Increment by 3: {}", c2.content);

    let c3 = registry.execute(&ToolCallExec::new(
        "s3",
        "counter",
        json!({"action": "get"}),
    ));
    tracing::debug!("   Get current: {}", c3.content);
    tracing::debug!("   Notice: State is maintained between calls!\n");

    // ═══════════════════════════════════════════════════════════════════════
    // SECTION 5: ERROR HANDLING
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ ⚠️  SECTION 5: Error Handling                           │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    // Error 1: Division by zero
    tracing::debug!("🔴 Error 1: Division by zero");
    let err1 = registry.execute(&ToolCallExec::new(
        "e1",
        "calculator",
        json!({"operation": "divide", "a": 10, "b": 0}),
    ));
    let err1_msg = err1
        .error
        .unwrap_or_else(|| "unexpected success".to_string());
    tracing::debug!("   Result: {}", err1_msg);

    // Error 2: Invalid tool
    tracing::debug!("\n🔴 Error 2: Invalid tool name");
    let err2 = registry.execute(&ToolCallExec::new("e2", "nonexistent", json!({})));
    tracing::debug!("   Result: {}", err2.error.unwrap());

    // Error 3: Missing required parameter
    tracing::debug!("\n🔴 Error 3: Missing required parameter");
    let err3 = registry.execute(&ToolCallExec::new(
        "e3",
        "calculator",
        json!({"operation": "add"}),
    ));
    tracing::debug!("   Result: {}", err3.error.unwrap());

    // Error 4: Validation failure
    tracing::debug!("\n🔴 Error 4: Validation failure (amount = 0)");
    let err4 = registry.execute(&ToolCallExec::new(
        "e4",
        "counter",
        json!({"action": "increment", "amount": 0}),
    ));
    tracing::debug!("   Result: {}\n", err4.error.unwrap());

    // ═══════════════════════════════════════════════════════════════════════
    // SECTION 6: REAL-WORLD WORKFLOW
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ 🎬 SECTION 6: Real-World Workflow Example              │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    tracing::debug!("Scenario: Calculate average of counter value and multiply by 2\n");

    tracing::debug!("Step 1: Get counter value");
    let step1 = registry.execute(&ToolCallExec::new(
        "w1",
        "counter",
        json!({"action": "get"}),
    ));
    let counter_val = step1.content["value"].as_i64().unwrap_or(0) as f64;
    tracing::debug!("   Counter value: {}", counter_val);

    tracing::debug!("\nStep 2: Calculate (counter + 100) / 2");
    let step2 = registry.execute(&ToolCallExec::new(
        "w2",
        "calculator",
        json!({"operation": "add", "a": counter_val, "b": 100.0}),
    ));
    let sum = step2.content["result"].as_f64().unwrap_or(0.0);
    let avg = sum / 2.0;
    tracing::debug!("   Average: {}", avg);

    tracing::debug!("\nStep 3: Multiply by 2");
    let step3 = registry.execute(&ToolCallExec::new(
        "w3",
        "calculator",
        json!({"operation": "multiply", "a": avg, "b": 2.0}),
    ));
    tracing::debug!("   Final result: {}\n", step3.content["result"]);

    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY & RECOMMENDATIONS
    // ═══════════════════════════════════════════════════════════════════════

    tracing::debug!("┌─────────────────────────────────────────────────────────┐");
    tracing::debug!("│ 📊 SUMMARY & RECOMMENDATIONS                            │");
    tracing::debug!("└─────────────────────────────────────────────────────────┘\n");

    tracing::debug!("✨ What You Learned:");
    tracing::debug!("   1. ✅ Create tools in 13 lines with #[tool] macro");
    tracing::debug!("   2. ✅ 4 different approaches for different needs");
    tracing::debug!("   3. ✅ ALWAYS add descriptions to prevent hallucination");
    tracing::debug!("   4. ✅ Use #[schemars(...)] for validation");
    tracing::debug!("   5. ✅ Stateful tools with SchemaBasedTool");
    tracing::debug!("   6. ✅ Batch execution for performance");
    tracing::debug!("   7. ✅ Proper error handling");
    tracing::debug!("   8. ✅ Multi-step workflows\n");

    tracing::debug!("🎯 Quick Reference:");
    tracing::debug!("   ┌─────────────────────────────────────────────────────┐");
    tracing::debug!("   │ #[derive(JsonSchema, Serialize, Deserialize)]      │");
    tracing::debug!("   │ pub struct MyParams {{                              │");
    tracing::debug!("   │     /// Description here! (Critical!)               │");
    tracing::debug!("   │     #[schemars(range(min = 0, max = 100))]          │");
    tracing::debug!("   │     pub field: i32,                                 │");
    tracing::debug!("   │ }}                                                  │");
    tracing::debug!("   │                                                     │");
    tracing::debug!("   │ #[tool(description = \"Tool description\")]          │");
    tracing::debug!("   │ fn my_tool(p: MyParams) -> Result<MyResult, Error> │");
    tracing::debug!("   └─────────────────────────────────────────────────────┘\n");

    tracing::debug!("📚 Key Takeaways:");
    tracing::debug!("   • Use #[tool] macro for 90% of tools");
    tracing::debug!("   • Use SchemaBasedTool for stateful tools");
    tracing::debug!("   • ALWAYS add /// doc comments to fields");
    tracing::debug!("   • Use #[schemars(...)] for validation");
    tracing::debug!("   • Test with batch execution");
    tracing::debug!("   • Handle errors gracefully\n");

    tracing::debug!("🎉 You're now ready to build production tool calling apps!");
    tracing::debug!("\n📖 This example covers everything. You don't need others!");

    Ok(())
}
