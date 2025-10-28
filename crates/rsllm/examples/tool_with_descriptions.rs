//! Tool Descriptions - Critical for LLM Understanding!
//!
//! This example shows the CORRECT way to add descriptions to prevent
//! LLM hallucination. Descriptions are CRITICAL!
//!
//! Run with: cargo run -p rsllm --example tool_with_descriptions --all-features

use rsllm::tool;
use rsllm::tools::ToolRegistry;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// ❌ BAD EXAMPLE - No Descriptions (LLM will hallucinate!)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct BadParams {
    pub a: f64,  // ❌ No description - LLM doesn't know what this is!
    pub b: f64,  // ❌ No description - LLM will guess!
    pub op: String,  // ❌ What operations are valid? LLM doesn't know!
}

// ============================================================================
// ✅ GOOD EXAMPLE - With Descriptions (LLM knows exactly what to do!)
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct GoodParams {
    /// First number for the operation (must be positive)
    #[schemars(range(min = 0.0))]
    pub a: f64,

    /// Second number for the operation (must be positive)
    #[schemars(range(min = 0.0))]
    pub b: f64,

    /// Operation to perform: "add", "subtract", "multiply", or "divide"
    #[schemars(regex(pattern = "^(add|subtract|multiply|divide)$"))]
    pub op: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalcResult {
    /// The result of the calculation
    pub result: f64,

    /// The operation that was performed
    pub operation: String,
}

#[tool(description = "Performs arithmetic operations with well-documented parameters")]
fn good_calculator(params: GoodParams) -> Result<CalcResult, Box<dyn Error + Send + Sync>> {
    let result = match params.op.as_str() {
        "add" => params.a + params.b,
        "subtract" => params.a - params.b,
        "multiply" => params.a * params.b,
        "divide" => {
            if params.b == 0.0 {
                return Err("Cannot divide by zero".into());
            }
            params.a / params.b
        }
        _ => return Err(format!("Invalid operation: {}", params.op).into()),
    };

    Ok(CalcResult {
        result,
        operation: params.op,
    })
}

// ============================================================================
// ✅ EXCELLENT EXAMPLE - Rich Descriptions with Constraints
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct SearchParams {
    /// Search query string (what to search for)
    /// Example: "rust programming tutorial"
    #[schemars(length(min = 1, max = 200))]
    pub query: String,

    /// Maximum number of results to return
    /// Must be between 1 and 100
    #[schemars(range(min = 1, max = 100))]
    #[serde(default = "default_limit")]
    pub limit: u32,

    /// Search category filter (optional)
    /// Valid values: "docs", "code", "issues", "discussions"
    pub category: Option<String>,

    /// Sort order for results
    /// Default is "relevance"
    #[serde(default = "default_sort")]
    pub sort: SortOrder,
}

fn default_limit() -> u32 {
    10
}

fn default_sort() -> SortOrder {
    SortOrder::Relevance
}

#[derive(JsonSchema, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    /// Sort by relevance score
    Relevance,
    /// Sort by newest first
    Newest,
    /// Sort by oldest first
    Oldest,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct SearchResult {
    /// List of search results
    pub results: Vec<String>,
    /// Total number of results found
    pub total: usize,
    /// The query that was searched
    pub query: String,
}

#[tool(description = "Search with comprehensive parameter descriptions and validation")]
fn search(params: SearchParams) -> Result<SearchResult, Box<dyn Error + Send + Sync>> {
    Ok(SearchResult {
        results: vec![
            format!("Result 1 for '{}'", params.query),
            format!("Result 2 for '{}'", params.query),
        ],
        total: 42,
        query: params.query,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("📝 Tool Descriptions - Preventing LLM Hallucination");
    println!("=====================================================\n");

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(GoodCalculatorTool))?;
    registry.register(Box::new(SearchTool))?;

    println!("✅ Registered {} tools\n", registry.len());

    // Show the generated schemas
    println!("🔍 Generated JSON Schemas:\n");

    for def in registry.tool_definitions() {
        println!("📝 Tool: {}", def.name);
        println!("   Description: {}", def.description);
        println!("   Schema:");
        println!("{}\n", serde_json::to_string_pretty(&def.parameters)?);
    }

    println!("💡 Key Takeaways:\n");
    println!("1. ✅ ALWAYS add doc comments (///) to struct fields");
    println!("   These become JSON Schema descriptions!");
    println!();
    println!("2. ✅ Use #[schemars(...)] for validation:");
    println!("   - range(min = 0, max = 100) - numeric ranges");
    println!("   - min_length/max_length - string length");
    println!("   - regex(pattern = \"...\") - pattern matching");
    println!();
    println!("3. ✅ Use #[serde(default)] for optional params");
    println!("   Provides default values");
    println!();
    println!("4. ✅ Use enums with #[serde(rename_all)]");
    println!("   Makes valid values explicit");
    println!();
    println!("5. ✅ Add examples in doc comments");
    println!("   Helps LLM understand usage");

    println!("\n⚠️  WITHOUT descriptions:");
    println!("   LLM sees: {{\"a\": number, \"b\": number}}");
    println!("   LLM thinks: \"What are a and b? I'll guess!\" ❌");

    println!("\n✅ WITH descriptions:");
    println!("   LLM sees: {{\"a\": \"First number (positive)\", \"b\": \"Second number\"}}");
    println!("   LLM knows: \"a is first number, b is second, both positive!\" ✅");

    println!("\n🎉 Always use descriptions to guide the LLM correctly!");

    Ok(())
}
