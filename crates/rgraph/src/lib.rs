//! # RGraph - Graph-based Agent Orchestration System
//! 
//! RGraph is a powerful graph-based workflow orchestration system designed for building
//! sophisticated AI agent applications. Inspired by LangGraph, it provides a declarative
//! way to define complex agent workflows with state management, conditional execution,
//! and seamless integration with the RRAG framework.
//! 
//! ## Key Features
//! 
//! - **Graph-Based Workflows**: Define agent behavior as directed graphs
//! - **State Management**: Persistent state across node executions
//! - **Conditional Routing**: Dynamic workflow paths based on execution results
//! - **Agent Orchestration**: Coordinate multiple AI agents and tools
//! - **RRAG Integration**: Built-in support for RAG-powered agents
//! - **Async Execution**: High-performance concurrent execution
//! - **Observability**: Comprehensive monitoring and debugging
//! - **Persistence**: Durable workflow state and history
//! 
//! ## Quick Start
//! 
//! ```rust
//! use rgraph::prelude::*;
//! 
//! # async fn example() -> RGraphResult<()> {
//! // Define a simple agent workflow
//! let mut graph = WorkflowGraph::new("research_assistant");
//! 
//! // Add nodes
//! graph.add_node("understand_query", QueryAnalysisNode::new()).await?;
//! graph.add_node("search_knowledge", RagSearchNode::new()).await?;
//! graph.add_node("synthesize_response", ResponseGenerationNode::new()).await?;
//! 
//! // Define edges
//! graph.add_edge("understand_query", "search_knowledge")?;
//! graph.add_edge("search_knowledge", "synthesize_response")?;
//! 
//! // Execute the workflow
//! let initial_state = GraphState::new()
//!     .with_input("user_query", "What is machine learning?");
//! 
//! let result = graph.execute(initial_state).await?;
//! println!("Response: {}", result.get_output("final_response")?);
//! # Ok(())
//! # }
//! ```
//! 
//! ## Advanced Examples
//! 
//! ### Multi-Agent Collaboration
//! ```rust
//! use rgraph::prelude::*;
//! 
//! # async fn example() -> RGraphResult<()> {
//! let mut graph = WorkflowGraph::new("multi_agent_system");
//! 
//! // Research agent
//! graph.add_node("researcher", 
//!     AgentNode::new("research_agent")
//!         .with_system_prompt("You are a research specialist...")
//!         .with_tools(vec![web_search_tool(), database_query_tool()])
//! ).await?;
//! 
//! // Analysis agent
//! graph.add_node("analyst",
//!     AgentNode::new("analysis_agent")
//!         .with_system_prompt("You analyze research data...")
//!         .with_tools(vec![data_analysis_tool(), visualization_tool()])
//! ).await?;
//! 
//! // Writer agent
//! graph.add_node("writer",
//!     AgentNode::new("writing_agent")
//!         .with_system_prompt("You write comprehensive reports...")
//! ).await?;
//! 
//! // Conditional routing based on research results
//! graph.add_conditional_edge("researcher", |state: &GraphState| {
//!     if state.get("research_quality_score")? > 0.8 {
//!         Ok("analyst".to_string())
//!     } else {
//!         Ok("researcher".to_string()) // Loop back for more research
//!     }
//! })?;
//! 
//! graph.add_edge("analyst", "writer")?;
//! 
//! let result = graph.execute(
//!     GraphState::new().with_input("research_topic", "Climate Change Impact")
//! ).await?;
//! # Ok(())
//! # }
//! ```
//! 
//! ### RAG-Powered Knowledge Agent
//! ```rust
//! use rgraph::prelude::*;
//! # #[cfg(feature = "rrag-integration")]
//! # async fn example() -> RGraphResult<()> {
//! let mut graph = WorkflowGraph::new("knowledge_agent");
//! 
//! // RAG retrieval node
//! graph.add_node("retrieve_context", 
//!     RagRetrievalNode::new()
//!         .with_rrag_system(rrag_system)
//!         .with_top_k(5)
//!         .with_score_threshold(0.7)
//! ).await?;
//! 
//! // Context evaluation node
//! graph.add_node("evaluate_context",
//!     ContextEvaluationNode::new()
//!         .with_relevance_threshold(0.6)
//! ).await?;
//! 
//! // Response generation with context
//! graph.add_node("generate_response",
//!     ContextualGenerationNode::new()
//!         .with_context_window(4096)
//! ).await?;
//! 
//! // Conditional routing based on context quality
//! graph.add_conditional_edge("evaluate_context", |state: &GraphState| {
//!     let context_score: f32 = state.get("context_relevance_score")?;
//!     if context_score > 0.6 {
//!         Ok("generate_response".to_string())
//!     } else {
//!         Ok("retrieve_context".to_string()) // Retry with different strategy
//!     }
//! })?;
//! 
//! let result = graph.execute(
//!     GraphState::new().with_input("query", "Explain quantum computing")
//! ).await?;
//! # Ok(())
//! # }
//! ```
//! 
//! ## Architecture
//! 
//! RGraph is built around several core concepts:
//! 
//! ### Workflow Graph
//! A directed graph representing the agent workflow, where nodes are execution units
//! and edges define the flow of control and data.
//! 
//! ### Graph State
//! A shared state object that flows through the graph, accumulating results and
//! providing context for decision-making.
//! 
//! ### Nodes
//! Execution units that perform specific tasks:
//! - **Agent Nodes**: LLM-powered agents with tools
//! - **Tool Nodes**: Direct tool execution
//! - **RAG Nodes**: Retrieval-augmented generation
//! - **Condition Nodes**: Decision points in the workflow
//! - **Transform Nodes**: Data transformation and processing
//! 
//! ### Execution Engine
//! The runtime system that executes graphs with support for:
//! - Parallel execution where possible
//! - State management and persistence
//! - Error handling and recovery
//! - Observability and debugging
//! 
//! ## Integration with RRAG
//! 
//! RGraph seamlessly integrates with the RRAG framework to provide:
//! - RAG-powered agent nodes
//! - Knowledge retrieval capabilities
//! - Document processing workflows
//! - Embedding-based routing decisions
//! - Multi-modal processing support

pub mod prelude;
pub mod core;
pub mod nodes;
pub mod execution;
pub mod state;
pub mod tools;
pub mod agents;
pub mod routing;
pub mod observability;

#[cfg(feature = "rrag-integration")]
pub mod rrag_integration;

#[cfg(feature = "persistence")]
pub mod persistence;

// Re-export core types for easy access
pub use crate::core::{
    WorkflowGraph, GraphBuilder, Node, NodeId, Edge, EdgeId,
    ExecutionContext, ExecutionResult
};
pub use crate::state::{GraphState, StateValue, StatePath};
pub use crate::execution::{ExecutionEngine, ExecutionConfig, ExecutionMode, ExecutionResults, ExecutionMetrics, ExecutionError};
pub use crate::nodes::{
    AgentNode, ToolNode, ConditionNode, TransformNode
};

#[cfg(feature = "rrag-integration")]
pub use crate::rrag_integration::{
    RagRetrievalNode, RagGenerationNode, ContextEvaluationNode,
    RagWorkflowBuilder, RagRetrievalConfig, RagGenerationConfig, ContextEvaluationConfig
};

// Error handling
use thiserror::Error;

/// Result type for RGraph operations
pub type RGraphResult<T> = Result<T, RGraphError>;

/// Error types for RGraph operations
#[derive(Debug, Error)]
pub enum RGraphError {
    #[error("Graph execution error: {message}")]
    Execution { message: String },
    
    #[error("Node error in '{node_id}': {message}")]
    Node { node_id: String, message: String },
    
    #[error("State error: {message}")]
    State { message: String },
    
    #[error("Graph validation error: {message}")]
    Validation { message: String },
    
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    #[error("Agent error: {message}")]
    Agent { message: String },
    
    #[error("Tool error: {message}")]
    Tool { message: String },
    
    #[error("Routing error: {message}")]
    Routing { message: String },
    
    #[cfg(feature = "rrag-integration")]
    #[error("RRAG integration error: {0}")]
    Rrag(#[from] rrag::RragError),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

impl RGraphError {
    /// Create an execution error
    pub fn execution(message: impl Into<String>) -> Self {
        Self::Execution { message: message.into() }
    }
    
    /// Create a node error
    pub fn node(node_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Node { 
            node_id: node_id.into(), 
            message: message.into() 
        }
    }
    
    /// Create a state error
    pub fn state(message: impl Into<String>) -> Self {
        Self::State { message: message.into() }
    }
    
    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation { message: message.into() }
    }
    
    /// Create a config error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config { message: message.into() }
    }
    
    /// Create a configuration error (alias for config)
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Config { message: message.into() }
    }
    
    /// Create an agent error
    pub fn agent(message: impl Into<String>) -> Self {
        Self::Agent { message: message.into() }
    }
    
    /// Create a tool error
    pub fn tool(message: impl Into<String>) -> Self {
        Self::Tool { message: message.into() }
    }
    
    /// Create a routing error
    pub fn routing(message: impl Into<String>) -> Self {
        Self::Routing { message: message.into() }
    }
}

/// Framework constants
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "RGraph";
pub const DESCRIPTION: &str = "Graph-based Agent Orchestration System";

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let exec_err = RGraphError::execution("test execution error");
        assert!(matches!(exec_err, RGraphError::Execution { .. }));
        
        let node_err = RGraphError::node("test_node", "test node error");
        assert!(matches!(node_err, RGraphError::Node { .. }));
    }
    
    #[test]
    fn test_constants() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "RGraph");
        assert!(!DESCRIPTION.is_empty());
    }
}