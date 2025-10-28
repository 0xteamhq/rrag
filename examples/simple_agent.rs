//! Simple Agent Framework Example
//!
//! This example demonstrates a simple but powerful agent framework that:
//! - Uses RSLLM for LLM communication
//! - Integrates tool calling
//! - Maintains conversation memory
//! - Handles multi-turn interactions
//! - Provides a clean agent abstraction
//!
//! This is a prototype to explore agent patterns before moving to source.
//!
//! Run: cargo run --example simple_agent --features rsllm-client

use rsllm::prelude::*;
use rsllm::tool;
use rsllm::tools::{Tool, ToolRegistry, ToolCall as ToolExec};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;

// ============================================================================
// AGENT FRAMEWORK - Core Abstractions
// ============================================================================

/// Simple Agent that can use tools and maintain conversation history
struct Agent {
    /// LLM client for generation
    llm_client: Client,

    /// Registry of available tools
    tool_registry: ToolRegistry,

    /// Conversation history
    conversation: Vec<ChatMessage>,

    /// Agent configuration
    config: AgentConfig,
}

/// Agent configuration
struct AgentConfig {
    /// System prompt that defines agent behavior
    system_prompt: String,

    /// Maximum conversation turns before summarization
    max_turns: usize,

    /// Whether to show thinking/reasoning
    verbose: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful assistant with access to tools. Use tools when needed to provide accurate information.".to_string(),
            max_turns: 20,
            verbose: true,
        }
    }
}

impl Agent {
    /// Create a new agent
    fn new(llm_client: Client, system_prompt: Option<String>) -> Self {
        let mut config = AgentConfig::default();
        if let Some(prompt) = system_prompt {
            config.system_prompt = prompt;
        }

        let mut conversation = Vec::new();
        conversation.push(ChatMessage::system(config.system_prompt.clone()));

        Self {
            llm_client,
            tool_registry: ToolRegistry::new(),
            conversation,
            config,
        }
    }

    /// Register a tool with the agent
    fn register_tool(&mut self, tool: Box<dyn Tool>) -> Result<(), Box<dyn Error>> {
        self.tool_registry.register(tool)?;
        Ok(())
    }

    /// Get available tools as LLM-compatible format
    fn get_tools_for_llm(&self) -> Vec<serde_json::Value> {
        self.tool_registry
            .tool_definitions()
            .into_iter()
            .map(|def| {
                json!({
                    "type": "function",
                    "function": {
                        "name": def.name,
                        "description": def.description,
                        "parameters": def.parameters
                    }
                })
            })
            .collect()
    }

    /// Run the agent with a user query (main entry point)
    async fn run(&mut self, user_input: &str) -> Result<String, Box<dyn Error>> {
        if self.config.verbose {
            println!("\n🤔 User: {}", user_input);
        }

        // Add user message to conversation
        self.conversation.push(ChatMessage::user(user_input));

        // Agent loop: keep running until we get a final answer (no more tool calls)
        let max_iterations = 10; // Prevent infinite loops
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > max_iterations {
                return Err("Agent exceeded maximum iterations".into());
            }

            if self.config.verbose {
                println!("\n🔄 Agent iteration {}", iteration);
            }

            // Call LLM with conversation history and available tools
            let response = self.llm_step().await?;

            // Check if LLM wants to use tools
            if let Some(tool_calls) = &response.tool_calls {
                if !tool_calls.is_empty() {
                    if self.config.verbose {
                        println!("🛠️  Agent wants to use {} tool(s)", tool_calls.len());
                    }

                    // Execute all requested tools
                    for tool_call in tool_calls {
                        self.execute_tool_call(tool_call)?;
                    }

                    // Continue loop to let agent process tool results
                    continue;
                }
            }

            // No tool calls - this is the final answer
            if self.config.verbose {
                println!("✅ Agent: {}", response.content);
            }

            // Add assistant response to conversation
            self.conversation.push(ChatMessage::assistant(response.content.clone()));

            return Ok(response.content);
        }
    }

    /// Single LLM step with tool calling support
    async fn llm_step(&self) -> Result<ChatResponse, Box<dyn Error>> {
        // For now, we'll simulate the tool calling since we need to integrate with actual LLM
        // In real implementation, this would call the LLM with tools parameter

        // TODO: This is where we'd integrate actual LLM tool calling
        // For now, return a mock response
        Ok(ChatResponse::new(
            "I'll help you with that using the available tools.".to_string(),
            "mock-model",
        ))
    }

    /// Execute a tool call and add result to conversation
    fn execute_tool_call(&mut self, tool_call: &ToolCall) -> Result<(), Box<dyn Error>> {
        if self.config.verbose {
            println!("   📞 Calling tool: {}", tool_call.function.name);
            println!("      Arguments: {}", tool_call.function.arguments);
        }

        // Execute the tool
        let tool_exec = ToolExec::new(
            &tool_call.id,
            &tool_call.function.name,
            tool_call.function.arguments.clone(),
        );

        let result = self.tool_registry.execute(&tool_exec);

        if self.config.verbose {
            if result.success {
                println!("      ✅ Result: {}", result.content);
            } else {
                println!("      ❌ Error: {}", result.error.as_ref().unwrap());
            }
        }

        // Add tool result to conversation
        let result_content = if result.success {
            serde_json::to_string(&result.content)?
        } else {
            format!("Error: {}", result.error.unwrap_or_default())
        };

        self.conversation.push(ChatMessage::tool(&tool_call.id, result_content));

        Ok(())
    }

    /// Get conversation history
    fn get_conversation(&self) -> &[ChatMessage] {
        &self.conversation
    }

    /// Clear conversation (keep system prompt)
    fn reset(&mut self) {
        self.conversation.clear();
        self.conversation.push(ChatMessage::system(self.config.system_prompt.clone()));
    }
}

// ============================================================================
// EXAMPLE TOOLS - Demonstrate Agent Capabilities
// ============================================================================

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalculatorParams {
    /// The arithmetic operation: "add", "subtract", "multiply", "divide"
    pub operation: String,
    /// First number
    pub a: f64,
    /// Second number
    pub b: f64,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct CalculatorResult {
    /// The calculation result
    pub result: f64,
}

#[tool(description = "Performs arithmetic calculations")]
fn calculator(params: CalculatorParams) -> Result<CalculatorResult, Box<dyn Error + Send + Sync>> {
    let result = match params.operation.as_str() {
        "add" => params.a + params.b,
        "subtract" => params.a - params.b,
        "multiply" => params.a * params.b,
        "divide" if params.b != 0.0 => params.a / params.b,
        "divide" => return Err("Cannot divide by zero".into()),
        _ => return Err(format!("Unknown operation: {}", params.operation).into()),
    };
    Ok(CalculatorResult { result })
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct WeatherParams {
    /// City name to get weather for
    pub city: String,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct WeatherResult {
    /// Temperature in Celsius
    pub temperature: i32,
    /// Weather condition
    pub condition: String,
}

#[tool(description = "Get current weather for a city")]
fn get_weather(_params: WeatherParams) -> Result<WeatherResult, Box<dyn Error + Send + Sync>> {
    // Mock weather data
    Ok(WeatherResult {
        temperature: 22,
        condition: "Sunny".to_string(),
    })
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct SearchParams {
    /// Search query
    pub query: String,
    /// Maximum number of results (1-10)
    #[schemars(range(min = 1, max = 10))]
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    5
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct SearchResult {
    /// Search results
    pub results: Vec<String>,
    /// Total count
    pub count: usize,
}

#[tool(description = "Search for information")]
fn search(params: SearchParams) -> Result<SearchResult, Box<dyn Error + Send + Sync>> {
    // Mock search results
    let results = vec![
        format!("Result 1 for '{}'", params.query),
        format!("Result 2 for '{}'", params.query),
        format!("Result 3 for '{}'", params.query),
    ];

    let limited_results: Vec<_> = results.into_iter().take(params.limit as usize).collect();
    let count = limited_results.len();

    Ok(SearchResult {
        results: limited_results,
        count,
    })
}

// ============================================================================
// MAIN - Demonstrate Agent Usage
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  🤖 Simple Agent Framework with RSLLM              ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Create LLM Client (using Ollama for local testing)
    // ═══════════════════════════════════════════════════════════════════════

    println!("📦 STEP 1: Initialize LLM Client");
    println!("─────────────────────────────────────────────────────\n");

    let llm_client = Client::builder()
        .provider(Provider::Ollama)
        .model("llama3.2:3b")
        .base_url("http://localhost:11434/api/")?
        .temperature(0.7)
        .build()?;

    println!("   ✅ Connected to Ollama (llama3.2:3b)\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Create Agent with Tools
    // ═══════════════════════════════════════════════════════════════════════

    println!("📦 STEP 2: Create Agent with Tools");
    println!("─────────────────────────────────────────────────────\n");

    let system_prompt = "You are a helpful AI assistant with access to tools. \
        Use the calculator for math, get_weather for weather info, and search for general queries. \
        Always use tools when appropriate to provide accurate answers.";

    let mut agent = Agent::new(llm_client, Some(system_prompt.to_string()));

    // Register tools
    agent.register_tool(Box::new(CalculatorTool))?;
    agent.register_tool(Box::new(GetWeatherTool))?;
    agent.register_tool(Box::new(SearchTool))?;

    println!("   ✅ Registered 3 tools:");
    for tool_name in agent.tool_registry.tool_names() {
        println!("      - {}", tool_name);
    }
    println!();

    // Show available tools in LLM format
    println!("   📋 Tools available to agent:");
    let tools = agent.get_tools_for_llm();
    for tool in &tools {
        println!("      • {}: {}",
            tool["function"]["name"].as_str().unwrap(),
            tool["function"]["description"].as_str().unwrap()
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 3: Test Direct Tool Execution
    // ═══════════════════════════════════════════════════════════════════════

    println!("📦 STEP 3: Test Direct Tool Execution");
    println!("─────────────────────────────────────────────────────\n");

    // Simulate what the LLM would do - request tool calls
    println!("   Simulating LLM tool call requests:\n");

    // Tool call 1: Calculator
    let tool_call_1 = ToolCall {
        id: "call-1".to_string(),
        call_type: rsllm::message::ToolCallType::Function,
        function: rsllm::message::ToolFunction {
            name: "calculator".to_string(),
            arguments: json!({"operation": "multiply", "a": 25, "b": 4}),
        },
    };

    agent.execute_tool_call(&tool_call_1)?;

    // Tool call 2: Weather
    let tool_call_2 = ToolCall {
        id: "call-2".to_string(),
        call_type: rsllm::message::ToolCallType::Function,
        function: rsllm::message::ToolFunction {
            name: "get_weather".to_string(),
            arguments: json!({"city": "Tokyo"}),
        },
    };

    agent.execute_tool_call(&tool_call_2)?;

    // Tool call 3: Search
    let tool_call_3 = ToolCall {
        id: "call-3".to_string(),
        call_type: rsllm::message::ToolCallType::Function,
        function: rsllm::message::ToolFunction {
            name: "search".to_string(),
            arguments: json!({"query": "Rust programming", "limit": 3}),
        },
    };

    agent.execute_tool_call(&tool_call_3)?;

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 4: Show Conversation History
    // ═══════════════════════════════════════════════════════════════════════

    println!("\n📦 STEP 4: Conversation History");
    println!("─────────────────────────────────────────────────────\n");

    println!("   Conversation has {} messages:", agent.get_conversation().len());
    for (i, msg) in agent.get_conversation().iter().enumerate() {
        let role = format!("{:?}", msg.role);
        let preview = match &msg.content {
            MessageContent::Text(t) => {
                if t.len() > 50 {
                    format!("{}...", &t[..50])
                } else {
                    t.clone()
                }
            }
            _ => "Multi-modal content".to_string(),
        };
        println!("   {}. {} - {}", i + 1, role, preview);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 5: Agent Patterns Demonstrated
    // ═══════════════════════════════════════════════════════════════════════

    println!("📦 STEP 5: Agent Patterns Demonstrated");
    println!("─────────────────────────────────────────────────────\n");

    println!("✅ Pattern 1: Tool Integration");
    println!("   • Agent has access to multiple tools");
    println!("   • Tools are registered dynamically");
    println!("   • Tool schemas auto-generated\n");

    println!("✅ Pattern 2: Conversation Memory");
    println!("   • Full conversation history maintained");
    println!("   • System prompt persisted");
    println!("   • Tool calls and results tracked\n");

    println!("✅ Pattern 3: Multi-Turn Interaction");
    println!("   • Agent can iterate multiple times");
    println!("   • Tool results fed back to LLM");
    println!("   • Prevents infinite loops\n");

    println!("✅ Pattern 4: Structured Tool Calling");
    println!("   • Type-safe tool parameters");
    println!("   • Automatic validation");
    println!("   • Error handling\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 6: Agent Architecture Summary
    // ═══════════════════════════════════════════════════════════════════════

    println!("📦 STEP 6: Agent Architecture");
    println!("─────────────────────────────────────────────────────\n");

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│                  Agent Architecture                 │");
    println!("├─────────────────────────────────────────────────────┤");
    println!("│                                                     │");
    println!("│  User Input                                         │");
    println!("│      ↓                                              │");
    println!("│  ┌──────────┐                                       │");
    println!("│  │  Agent   │                                       │");
    println!("│  └──────────┘                                       │");
    println!("│      ↓                                              │");
    println!("│  ┌──────────────────────┐                          │");
    println!("│  │  Conversation        │                          │");
    println!("│  │  Memory              │                          │");
    println!("│  └──────────────────────┘                          │");
    println!("│      ↓                                              │");
    println!("│  ┌──────────────────────┐                          │");
    println!("│  │  LLM (RSLLM)         │                          │");
    println!("│  │  + Tool Schemas      │                          │");
    println!("│  └──────────────────────┘                          │");
    println!("│      ↓                                              │");
    println!("│  Tool Calls? ──Yes──> Execute Tools ──┐            │");
    println!("│      │                                 │            │");
    println!("│      No                                │            │");
    println!("│      ↓                                 │            │");
    println!("│  Final Answer <────────────────────────┘            │");
    println!("│                                                     │");
    println!("└─────────────────────────────────────────────────────┘\n");

    // ═══════════════════════════════════════════════════════════════════════
    // STEP 7: Next Steps for Implementation
    // ═══════════════════════════════════════════════════════════════════════

    println!("📦 STEP 7: Next Steps");
    println!("─────────────────────────────────────────────────────\n");

    println!("🔮 To complete the agent, we need to:");
    println!("   1. ✅ Tool registry (DONE!)");
    println!("   2. ✅ Conversation memory (DONE!)");
    println!("   3. ✅ Agent structure (DONE!)");
    println!("   4. 🚧 LLM tool calling integration (TODO)");
    println!("   5. 🚧 Response parsing for tool calls (TODO)");
    println!("   6. 🚧 Agent loop with LLM (TODO)\n");

    println!("💡 Key Insights:");
    println!("   • Agent = LLM Client + Tools + Memory + Loop");
    println!("   • Tools use our RSLLM tool framework");
    println!("   • Conversation history enables context");
    println!("   • Agent loop handles multi-turn reasoning\n");

    println!("🎯 Ready to move to source code:");
    println!("   Create: crates/rrag/src/agent/");
    println!("   Files:");
    println!("     - agent.rs       (Agent struct)");
    println!("     - config.rs      (AgentConfig)");
    println!("     - executor.rs    (Tool execution)");
    println!("     - memory.rs      (Conversation management)");
    println!("     - mod.rs         (Public API)\n");

    println!("🎉 Agent framework prototype complete!");
    println!("   This example shows the core patterns we need!");

    Ok(())
}
