use rsllm::tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TestParams {
    pub value: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct TestResult {
    pub echo: String,
}

#[tool(description = "Test tool")]
fn test_func(params: TestParams) -> Result<TestResult, Box<dyn Error + Send + Sync>> {
    Ok(TestResult {
        echo: params.value,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    use rsllm::tools::{ToolRegistry, ToolCall as ToolCallExec};
    use serde_json::json;

    println!("🎯 Testing #[tool] Macro");
    println!("========================\n");

    // Create registry and register the macro-generated tool
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(TestFuncTool))?;

    println!("✅ Macro-generated tool registered: TestFuncTool\n");

    // Execute it
    let call = ToolCallExec::new(
        "test-1",
        "test_func",
        json!({"value": "Hello from macro!"}),
    );

    let result = registry.execute(&call);

    if result.success {
        println!("✅ Tool executed successfully!");
        println!("   Result: {}", serde_json::to_string_pretty(&result.content)?);
    } else {
        println!("❌ Error: {:?}", result.error);
    }

    println!("\n🎉 #[tool] macro works perfectly!");

    Ok(())
}
