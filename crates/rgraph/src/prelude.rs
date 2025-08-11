//! # RGraph Prelude
//! 
//! This module provides a convenient way to import the most commonly used RGraph types and traits.
//! 
//! ```rust
//! use rgraph::prelude::*;
//! ```

// Core types
pub use crate::core::{
    WorkflowGraph, GraphBuilder, Node, NodeId, Edge, EdgeId,
    ExecutionContext, ExecutionResult
};

// State management
pub use crate::state::{GraphState, StateValue, StatePath};

// Execution engine
pub use crate::execution::{ExecutionEngine, ExecutionConfig, ExecutionMode};

// Node types
pub use crate::nodes::{
    AgentNode, ToolNode, ConditionNode, TransformNode,
    NodeConfig, NodeMetadata
};

// Agent system
pub use crate::agents::{Agent, AgentConfig, AgentBuilder};

// Tool system
pub use crate::tools::{Tool, ToolConfig, ToolResult, ToolError};

// Routing
pub use crate::routing::{
    RoutingCondition, ConditionalEdge, Router, RoutingDecision
};

// Error handling
pub use crate::{RGraphError, RGraphResult};

// Re-export commonly used async trait
pub use async_trait::async_trait;

// Re-export UUID for node IDs
pub use uuid::Uuid;

// Re-export commonly used future types
pub use futures::{Future, Stream, StreamExt, FutureExt};

// RRAG integration (when feature is enabled)
#[cfg(feature = "rrag-integration")]
pub use crate::rrag_integration::{
    RagRetrievalNode, RagGenerationNode, ContextEvaluationNode,
    RagWorkflowBuilder, RagRetrievalConfig, RagGenerationConfig, ContextEvaluationConfig
};

// Observability (when feature is enabled)
#[cfg(feature = "observability")]
pub use crate::observability::{
    GraphObserver, ExecutionMetrics, NodeMetrics, ObservabilityConfig
};

// Persistence (when feature is enabled)
#[cfg(feature = "persistence")]
pub use crate::persistence::{
    PersistentState, StateStore, SqliteStateStore, PostgresStateStore
};