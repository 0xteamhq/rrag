//! # Simple Execution Engine
//! 
//! A simplified execution engine that avoids complex lifetime issues.

use crate::core::{WorkflowGraph, ExecutionContext, ExecutionResult, NodeId};
use crate::state::GraphState;
use crate::{RGraphError, RGraphResult};
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the execution engine
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionConfig {
    /// Maximum number of nodes to execute
    pub max_nodes: usize,
    /// Continue execution on node errors
    pub continue_on_error: bool,
    /// Enable verbose logging
    pub verbose_logging: bool,
    /// Execution timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Maximum execution depth to prevent infinite loops
    pub max_execution_depth: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1000,
            continue_on_error: false,
            verbose_logging: false,
            timeout_seconds: Some(300), // 5 minutes
            max_execution_depth: 100,
        }
    }
}

/// Execution mode for the graph
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExecutionMode {
    /// Execute nodes sequentially
    Sequential,
    /// Execute nodes in parallel where possible
    Parallel,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Sequential
    }
}

/// Results from graph execution
#[derive(Debug, Clone)]
pub struct ExecutionResults {
    /// The final state after execution
    pub final_state: GraphState,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    /// Any errors that occurred
    pub errors: Vec<ExecutionError>,
}

/// Metrics collected during execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Number of nodes executed
    pub nodes_executed: usize,
    /// Total execution duration
    pub total_duration: Duration,
    /// Success indicator
    pub success: bool,
}

/// Error that occurred during execution
#[derive(Debug, Clone)]
pub struct ExecutionError {
    /// ID of the node that failed
    pub node_id: String,
    /// Error message
    pub error_message: String,
    /// When the error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Type of error
    pub error_type: String,
}

/// Simple execution engine
#[derive(Debug, Clone)]
pub struct ExecutionEngine {
    config: ExecutionConfig,
}

impl ExecutionEngine {
    /// Create a new execution engine with default configuration
    pub fn new() -> Self {
        Self {
            config: ExecutionConfig::default(),
        }
    }

    /// Create a new execution engine with custom configuration
    pub fn with_config(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Execute a workflow graph
    pub async fn execute(
        &self,
        graph: &WorkflowGraph,
        mut state: GraphState,
    ) -> RGraphResult<ExecutionResults> {
        let start_time = Instant::now();
        let mut errors = Vec::new();
        let mut nodes_executed = 0;

        if self.config.verbose_logging {
            #[cfg(feature = "observability")]
            tracing::info!("Starting graph execution: {}", graph.id());
            #[cfg(not(feature = "observability"))]
            eprintln!("Starting graph execution: {}", graph.id());
        }

        // Get entry points
        let entry_points = graph.entry_points_owned();
        if entry_points.is_empty() {
            return Err(RGraphError::execution("No entry points defined for graph"));
        }

        // Execute each entry point
        for entry_node_id in &entry_points {
            match self.execute_single_node(graph, &mut state, entry_node_id).await {
                Ok(_) => {
                    nodes_executed += 1;
                }
                Err(e) => {
                    let error = ExecutionError {
                        node_id: entry_node_id.as_str().to_string(),
                        error_message: e.to_string(),
                        timestamp: chrono::Utc::now(),
                        error_type: "NodeExecutionError".to_string(),
                    };
                    errors.push(error);

                    if !self.config.continue_on_error {
                        break;
                    }
                }
            }

            if nodes_executed >= self.config.max_nodes {
                break;
            }
        }

        let total_duration = start_time.elapsed();
        let success = errors.is_empty() || self.config.continue_on_error;

        if self.config.verbose_logging {
            #[cfg(feature = "observability")]
            tracing::info!("Graph execution completed: {} (success: {}, duration: {:?})", graph.id(), success, total_duration);
            #[cfg(not(feature = "observability"))]
            eprintln!("Graph execution completed: {} (success: {}, duration: {:?})", graph.id(), success, total_duration);
        }

        Ok(ExecutionResults {
            final_state: state,
            metrics: ExecutionMetrics {
                nodes_executed,
                total_duration,
                success,
            },
            errors,
        })
    }

    /// Execute a single node
    async fn execute_single_node(
        &self,
        graph: &WorkflowGraph,
        state: &mut GraphState,
        node_id: &NodeId,
    ) -> RGraphResult<()> {
        // Get the node
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| RGraphError::execution(format!("Node '{}' not found", node_id.as_str())))?;

        // Create execution context
        let context = ExecutionContext::new(graph.id().to_string(), node_id.clone());

        if self.config.verbose_logging {
            #[cfg(feature = "observability")]
            tracing::debug!("Executing node: {}", node_id.as_str());
            #[cfg(not(feature = "observability"))]
            eprintln!("Executing node: {}", node_id.as_str());
        }

        // Execute the node
        match node.execute(state, &context).await {
            Ok(ExecutionResult::Continue) => {
                if self.config.verbose_logging {
                    #[cfg(feature = "observability")]
                    tracing::debug!("Node '{}' completed successfully", node_id.as_str());
                    #[cfg(not(feature = "observability"))]
                    eprintln!("Node '{}' completed successfully", node_id.as_str());
                }
                Ok(())
            }
            Ok(ExecutionResult::Stop) => {
                if self.config.verbose_logging {
                    #[cfg(feature = "observability")]
                    tracing::info!("Node '{}' requested execution stop", node_id.as_str());
                    #[cfg(not(feature = "observability"))]
                    eprintln!("Node '{}' requested execution stop", node_id.as_str());
                }
                Ok(())
            }
            Ok(ExecutionResult::Route(_next_node)) => {
                // For now, we'll treat routing as completion
                // In a more complex implementation, we'd follow the route
                if self.config.verbose_logging {
                    #[cfg(feature = "observability")]
                    tracing::debug!("Node '{}' requested routing", node_id.as_str());
                    #[cfg(not(feature = "observability"))]
                    eprintln!("Node '{}' requested routing", node_id.as_str());
                }
                Ok(())
            }
            Ok(ExecutionResult::JumpTo(_target_node)) => {
                // For now, we'll treat jump as completion
                // In a more complex implementation, we'd jump to the target
                if self.config.verbose_logging {
                    #[cfg(feature = "observability")]
                    tracing::debug!("Node '{}' requested jump", node_id.as_str());
                    #[cfg(not(feature = "observability"))]
                    eprintln!("Node '{}' requested jump", node_id.as_str());
                }
                Ok(())
            }
            Err(e) => {
                if self.config.verbose_logging {
                    #[cfg(feature = "observability")]
                    tracing::error!("Node '{}' failed: {}", node_id.as_str(), e);
                    #[cfg(not(feature = "observability"))]
                    eprintln!("Node '{}' failed: {}", node_id.as_str(), e);
                }
                Err(e)
            }
        }
    }
}

impl Default for ExecutionEngine {
    fn default() -> Self {
        Self::new()
    }
}