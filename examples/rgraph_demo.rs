//! # RGraph Demo - Graph-based Agent System
//! 
//! This example demonstrates RGraph's capabilities:
//! - Building workflow graphs
//! - Agent nodes with tool usage
//! - State management and flow control
//! - Multi-step reasoning and decision making
//! 
//! Run with: `cargo run --bin rgraph_demo`

use rgraph::prelude::*;
use rgraph::tools::{EchoTool, CalculatorTool, ToolRegistry};
use rgraph::nodes::agent::{AgentNode, AgentNodeConfig};
use rgraph::nodes::tool::{ToolNode, ToolNodeConfig};
use rgraph::nodes::condition::{ConditionNode, ConditionNodeConfig};
use rgraph::nodes::transform::{TransformNode, TransformNodeConfig, TransformType};
use rgraph::execution::{ExecutionEngine, ExecutionConfig, ExecutionMode};
use std::sync::Arc;
use std::collections::HashMap;
use tokio;

#[tokio::main]
async fn main() -> RGraphResult<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    println!("🤖 RGraph - Graph-based Agent System Demo");
    println!("==========================================\n");

    // Demo 1: Simple Agent Workflow
    println!("📋 Demo 1: Simple Agent Workflow");
    println!("--------------------------------");
    simple_agent_workflow().await?;
    println!();

    // Demo 2: Multi-Agent Collaboration
    println!("👥 Demo 2: Multi-Agent Collaboration");
    println!("------------------------------------");
    multi_agent_collaboration().await?;
    println!();

    // Demo 3: Tool Integration
    println!("🔧 Demo 3: Tool Integration and Usage");
    println!("------------------------------------");
    tool_integration_demo().await?;
    println!();

    // Demo 4: Conditional Routing
    println!("🛤️  Demo 4: Conditional Routing");
    println!("-------------------------------");
    conditional_routing_demo().await?;
    println!();

    // Demo 5: Data Processing Pipeline
    println!("⚙️  Demo 5: Data Processing Pipeline");
    println!("-----------------------------------");
    data_processing_pipeline().await?;
    println!();

    println!("🎉 All RGraph demos completed successfully!");
    Ok(())
}

async fn simple_agent_workflow() -> RGraphResult<()> {
    // Create a simple workflow with a single agent
    let agent_config = AgentNodeConfig {
        name: "helpful_assistant".to_string(),
        system_prompt: "You are a helpful AI assistant that provides clear and concise answers.".to_string(),
        max_steps: 3,
        temperature: 0.7,
        ..Default::default()
    };
    
    let agent_node = Arc::new(AgentNode::new("assistant", agent_config));
    
    // Build the graph
    let mut graph = WorkflowGraph::new("simple_agent_workflow");
    graph.add_node("assistant", agent_node).await?;
    
    // Execute the workflow
    let initial_state = GraphState::new()
        .with_input("user_input", "What is the capital of France?");
    
    let engine = ExecutionEngine::new();
    let result = engine.execute(&graph, initial_state).await?;
    
    println!("✅ Agent Response: {}", 
             result.final_state.get("agent_response")
                 .unwrap_or_default()
                 .as_string()
                 .unwrap_or("No response"));
    
    println!("📊 Execution took: {:?}", result.metrics.total_duration);
    println!("🔄 Nodes executed: {}", result.metrics.nodes_executed);
    
    Ok(())
}

async fn multi_agent_collaboration() -> RGraphResult<()> {
    // Create multiple agents with different roles
    let researcher_config = AgentNodeConfig {
        name: "researcher".to_string(),
        system_prompt: "You are a research specialist. Gather and analyze information thoroughly.".to_string(),
        max_steps: 5,
        ..Default::default()
    };
    
    let analyst_config = AgentNodeConfig {
        name: "analyst".to_string(),
        system_prompt: "You are a data analyst. Analyze information and provide insights.".to_string(),
        max_steps: 3,
        ..Default::default()
    };
    
    let writer_config = AgentNodeConfig {
        name: "writer".to_string(),
        system_prompt: "You are a technical writer. Create clear, comprehensive reports.".to_string(),
        max_steps: 3,
        ..Default::default()
    };
    
    // Create nodes
    let researcher = Arc::new(AgentNode::new("researcher", researcher_config));
    let analyst = Arc::new(AgentNode::new("analyst", analyst_config));
    let writer = Arc::new(AgentNode::new("writer", writer_config));
    
    // Build the collaboration graph
    let graph = GraphBuilder::new("multi_agent_collaboration")
        .description("Collaborative research and analysis workflow")
        .add_node("researcher", researcher).await?
        .add_node("analyst", analyst).await?
        .add_node("writer", writer).await?
        .add_edge("researcher", "analyst")?
        .add_edge("analyst", "writer")?
        .entry_points(vec![NodeId::new("researcher")])
        .build()?;
    
    // Execute the collaborative workflow
    let initial_state = GraphState::new()
        .with_input("research_topic", "Impact of AI on software development");
    
    let config = ExecutionConfig {
        mode: ExecutionMode::Sequential,
        verbose_logging: true,
        ..Default::default()
    };
    
    let engine = ExecutionEngine::with_config(config);
    let result = engine.execute(&graph, initial_state).await?;
    
    println!("🔬 Research Phase: Completed");
    println!("📊 Analysis Phase: Completed");
    println!("✍️  Writing Phase: Completed");
    
    if let Ok(final_report) = result.final_state.get("agent_response") {
        if let Some(report) = final_report.as_string() {
            println!("📄 Final Report: {}", 
                     report.chars().take(100).collect::<String>() + "...");
        }
    }
    
    println!("⏱️  Total collaboration time: {:?}", result.metrics.total_duration);
    
    Ok(())
}

async fn tool_integration_demo() -> RGraphResult<()> {
    // Create tools
    let echo_tool = Arc::new(EchoTool::new());
    let calculator_tool = Arc::new(CalculatorTool::new());
    
    // Create agent with tools
    let agent_config = AgentNodeConfig {
        name: "tool_agent".to_string(),
        system_prompt: "You are an AI assistant with access to tools. Use them to help users.".to_string(),
        tools: vec!["echo".to_string(), "calculator".to_string()],
        max_steps: 5,
        ..Default::default()
    };
    
    let agent = Arc::new(
        AgentNode::new("tool_agent", agent_config)
            .with_tool("echo".to_string(), echo_tool.clone())
            .with_tool("calculator".to_string(), calculator_tool.clone())
    );
    
    // Create tool nodes for direct tool usage
    let calc_config = ToolNodeConfig {
        tool_name: "calculator".to_string(),
        argument_mappings: [
            ("operation".to_string(), "operation".to_string()),
            ("a".to_string(), "a".to_string()),
            ("b".to_string(), "b".to_string()),
        ].iter().cloned().collect(),
        output_key: "calculation_result".to_string(),
    };
    
    let calc_node = Arc::new(ToolNode::new(
        "calculator_node",
        "Direct Calculator",
        calculator_tool,
        calc_config,
    ));
    
    // Build graph
    let graph = GraphBuilder::new("tool_integration")
        .add_node("tool_agent", agent).await?
        .add_node("calculator_node", calc_node).await?
        .add_edge("tool_agent", "calculator_node")?
        .entry_points(vec![NodeId::new("tool_agent")])
        .build()?;
    
    // Test tool integration
    let initial_state = GraphState::new()
        .with_input("user_input", "Please calculate 15 * 23 and then echo the result")
        .with_input("operation", "multiply")
        .with_input("a", 15.0)
        .with_input("b", 23.0);
    
    let result = graph.execute(initial_state).await?;
    
    println!("🤖 Agent used tools to process request");
    
    if let Ok(calc_result) = result.final_state.get("calculation_result") {
        println!("🔢 Calculation result: {:?}", calc_result);
    }
    
    if let Ok(agent_response) = result.final_state.get("agent_response") {
        if let Some(response) = agent_response.as_string() {
            println!("💬 Agent response: {}", response);
        }
    }
    
    Ok(())
}

async fn conditional_routing_demo() -> RGraphResult<()> {
    // Create different processing nodes
    let positive_agent = Arc::new(AgentNode::new(
        "positive_agent",
        AgentNodeConfig {
            name: "positive_responder".to_string(),
            system_prompt: "You provide positive, encouraging responses.".to_string(),
            ..Default::default()
        }
    ));
    
    let negative_agent = Arc::new(AgentNode::new(
        "negative_agent", 
        AgentNodeConfig {
            name: "critical_analyzer".to_string(),
            system_prompt: "You provide critical analysis and point out potential issues.".to_string(),
            ..Default::default()
        }
    ));
    
    // Create condition node for sentiment routing
    let condition_config = ConditionNodeConfig {
        condition_key: "sentiment".to_string(),
        condition_value: serde_json::Value::String("positive".to_string()),
        true_route: "positive_agent".to_string(),
        false_route: "negative_agent".to_string(),
    };
    
    let router = Arc::new(ConditionNode::new(
        "sentiment_router",
        "Sentiment Router", 
        condition_config
    ));
    
    // Build routing graph
    let mut graph = WorkflowGraph::new("conditional_routing");
    graph.add_node("sentiment_router", router).await?;
    graph.add_node("positive_agent", positive_agent).await?;
    graph.add_node("negative_agent", negative_agent).await?;
    graph.set_entry_points(vec![NodeId::new("sentiment_router")]);
    
    // Test positive sentiment routing
    println!("🟢 Testing positive sentiment routing:");
    let positive_state = GraphState::new()
        .with_input("sentiment", "positive")
        .with_input("user_input", "I love working with AI systems!");
    
    let result = graph.execute(positive_state).await?;
    println!("   Route taken: positive_agent");
    
    // Test negative sentiment routing  
    println!("🔴 Testing negative sentiment routing:");
    let negative_state = GraphState::new()
        .with_input("sentiment", "negative")
        .with_input("user_input", "I'm having trouble with this AI system.");
    
    let result = graph.execute(negative_state).await?;
    println!("   Route taken: negative_agent");
    
    Ok(())
}

async fn data_processing_pipeline() -> RGraphResult<()> {
    // Create data processing nodes
    let uppercase_transform = Arc::new(TransformNode::new(
        "uppercase",
        "Uppercase Transform",
        TransformNodeConfig {
            input_key: "raw_text".to_string(),
            output_key: "uppercase_text".to_string(),
            transform_type: TransformType::ToUpperCase,
        }
    ));
    
    let replace_transform = Arc::new(TransformNode::new(
        "replace",
        "Replace Transform",
        TransformNodeConfig {
            input_key: "uppercase_text".to_string(),
            output_key: "processed_text".to_string(),
            transform_type: TransformType::Replace {
                from: "HELLO".to_string(),
                to: "GREETINGS".to_string(),
            },
        }
    ));
    
    let json_transform = Arc::new(TransformNode::new(
        "json_stringify",
        "JSON Transform",
        TransformNodeConfig {
            input_key: "processed_text".to_string(),
            output_key: "json_output".to_string(),
            transform_type: TransformType::JsonStringify,
        }
    ));
    
    // Build processing pipeline
    let graph = GraphBuilder::new("data_processing_pipeline")
        .description("Multi-stage data processing pipeline")
        .add_node("uppercase", uppercase_transform).await?
        .add_node("replace", replace_transform).await?
        .add_node("json_stringify", json_transform).await?
        .add_edge("uppercase", "replace")?
        .add_edge("replace", "json_stringify")?
        .entry_points(vec![NodeId::new("uppercase")])
        .build()?;
    
    // Process data through pipeline
    let initial_state = GraphState::new()
        .with_input("raw_text", "hello world from rgraph pipeline");
    
    let result = graph.execute(initial_state).await?;
    
    println!("📝 Input: 'hello world from rgraph pipeline'");
    
    if let Ok(uppercase) = result.final_state.get("uppercase_text") {
        if let Some(text) = uppercase.as_string() {
            println!("🔠 After uppercase: '{}'", text);
        }
    }
    
    if let Ok(replaced) = result.final_state.get("processed_text") {
        if let Some(text) = replaced.as_string() {
            println!("🔄 After replace: '{}'", text);
        }
    }
    
    if let Ok(json) = result.final_state.get("json_output") {
        if let Some(text) = json.as_string() {
            println!("📋 Final JSON: {}", text);
        }
    }
    
    println!("⚙️  Pipeline stages: {} → {} → {}", 
             "uppercase", "replace", "json_stringify");
    println!("⏱️  Processing time: {:?}", result.metrics.total_duration);
    
    Ok(())
}