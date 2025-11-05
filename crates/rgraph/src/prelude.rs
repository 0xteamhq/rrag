//! # RGraph Prelude
//!
//! This module provides a convenient way to import the most commonly used RGraph types and traits.
//!
//! ```rust
//! use rgraph::prelude::*;
//! ```

// Core types
pub use crate::core::{
    Edge, EdgeId, ExecutionContext, ExecutionResult, GraphBuilder, Node, NodeId, WorkflowGraph,
};

// State management
pub use crate::state::{GraphState, StatePath, StateValue};

// Execution engine
pub use crate::execution::{ExecutionConfig, ExecutionEngine, ExecutionMode};

// Node types
pub use crate::nodes::{
    AgentNode, ConditionNode, NodeConfig, NodeMetadata, ToolNode, TransformNode,
};

// Agent system
pub use crate::agents::{Agent, AgentBuilder, AgentConfig};

// Tool system
pub use crate::tools::{Tool, ToolConfig, ToolError, ToolResult};

// Routing
pub use crate::routing::{ConditionalEdge, Router, RoutingCondition, RoutingDecision};

// Error handling
pub use crate::{RGraphError, RGraphResult};

// Re-export commonly used async trait
pub use async_trait::async_trait;

// Re-export UUID for node IDs
pub use uuid::Uuid;

// Re-export commonly used future types
pub use futures::{Future, FutureExt, Stream, StreamExt};

// RRAG integration (when feature is enabled)
#[cfg(feature = "rrag-integration")]
pub use crate::rrag_integration::{
    ContextEvaluationConfig, ContextEvaluationNode, RagGenerationConfig, RagGenerationNode,
    RagRetrievalConfig, RagRetrievalNode, RagWorkflowBuilder,
};

// Observability (when feature is enabled)
#[cfg(feature = "observability")]
pub use crate::observability::{ExecutionMetrics, GraphObserver, NodeMetrics, ObservabilityConfig};
