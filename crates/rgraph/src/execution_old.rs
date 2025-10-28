//! # Graph Execution Engine
//! 
//! This module contains the execution engine that runs workflow graphs,
//! managing state, handling errors, and coordinating the flow of execution
//! through the graph nodes.

use crate::core::{WorkflowGraph, Node, NodeId, ExecutionContext, ExecutionResult};
use crate::state::GraphState;
use crate::{RGraphError, RGraphResult};
use std::sync::Arc;
use std::time::Duration;
use std::collections::{HashMap, VecDeque};
use tokio::time::timeout;
use futures::future::BoxFuture;
use uuid::Uuid;

#[cfg(feature = "observability")]
use crate::observability::{GraphObserver, ExecutionMetrics};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for graph execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionConfig {
    /// Maximum execution time for the entire graph
    pub max_execution_time: Option<Duration>,
    
    /// Maximum execution time for a single node
    pub max_node_execution_time: Option<Duration>,
    
    /// Maximum number of nodes that can be executed
    pub max_nodes: Option<usize>,
    
    /// Whether to continue execution on node failures
    pub continue_on_error: bool,
    
    /// Execution mode
    pub mode: ExecutionMode,
    
    /// Whether to enable detailed logging
    pub verbose_logging: bool,
    
    /// Maximum depth for recursive execution (prevents infinite loops)
    pub max_recursion_depth: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Some(Duration::from_secs(300)), // 5 minutes
            max_node_execution_time: Some(Duration::from_secs(60)), // 1 minute per node
            max_nodes: Some(1000),
            continue_on_error: false,
            mode: ExecutionMode::Sequential,
            verbose_logging: false,
            max_recursion_depth: 100,
        }
    }
}

/// Execution modes for the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExecutionMode {
    /// Execute nodes sequentially
    Sequential,
    /// Execute independent nodes in parallel
    Parallel,
    /// Execute with maximum parallelism
    MaxParallel,
}

/// Result of graph execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphExecutionResult {
    /// Execution ID
    pub execution_id: String,
    
    /// Final state after execution
    pub final_state: GraphState,
    
    /// Execution metrics
    pub metrics: ExecutionMetrics,
    
    /// Path taken through the graph
    pub execution_path: Vec<NodeId>,
    
    /// Any errors that occurred
    pub errors: Vec<ExecutionError>,
    
    /// Whether execution completed successfully
    pub success: bool,
}

/// Information about execution errors
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionError {
    pub node_id: String,
    pub error_message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_type: String,
}

/// Execution metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionMetrics {
    /// Total execution time
    pub total_duration: Duration,
    
    /// Number of nodes executed
    pub nodes_executed: usize,
    
    /// Per-node execution times
    pub node_durations: HashMap<String, Duration>,
    
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    
    /// End time
    pub end_time: chrono::DateTime<chrono::Utc>,
    
    /// Memory usage metrics
    pub peak_memory_usage: Option<usize>,
}

/// The main execution engine for workflow graphs
pub struct ExecutionEngine {
    config: ExecutionConfig,
    
    #[cfg(feature = "observability")]
    observers: Vec<Arc<dyn GraphObserver>>,
}

impl ExecutionEngine {
    /// Create a new execution engine with default configuration
    pub fn new() -> Self {
        Self {
            config: ExecutionConfig::default(),
            #[cfg(feature = "observability")]
            observers: Vec::new(),
        }
    }
    
    /// Create a new execution engine with custom configuration
    pub fn with_config(config: ExecutionConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "observability")]
            observers: Vec::new(),
        }
    }
    
    /// Add an observer for execution monitoring
    #[cfg(feature = "observability")]
    pub fn add_observer(&mut self, observer: Arc<dyn GraphObserver>) {
        self.observers.push(observer);
    }
    
    /// Execute a workflow graph
    pub async fn execute(
        &self,
        graph: &WorkflowGraph,
        mut state: GraphState,
    ) -> RGraphResult<GraphExecutionResult> {
        let execution_id = Uuid::new_v4().to_string();
        let start_time = chrono::Utc::now();
        
        if self.config.verbose_logging {
            #[cfg(feature = "observability")]
            tracing::info!("Starting graph execution: {}", execution_id);
            #[cfg(not(feature = "observability"))]
            tracing::debug!("Starting graph execution: {}", execution_id);
        }
        
        // Validate the graph
        graph.validate()?;
        
        // Set up execution context
        let mut execution_path = Vec::new();
        let mut errors = Vec::new();
        let mut node_durations = HashMap::new();
        let mut nodes_executed = 0;
        
        // Execute the graph with timeout if specified
        let execution_future = self.execute_internal(
            graph,
            &mut state,
            &execution_id,
            &mut execution_path,
            &mut errors,
            &mut node_durations,
            &mut nodes_executed,
        );
        
        let execution_result = if let Some(max_time) = self.config.max_execution_time {
            match timeout(max_time, execution_future).await {
                Ok(result) => result,
                Err(_) => {
                    errors.push(ExecutionError {
                        node_id: "timeout".to_string(),
                        error_message: "Graph execution timed out".to_string(),
                        timestamp: chrono::Utc::now(),
                        error_type: "Timeout".to_string(),
                    });
                    Err(RGraphError::execution("Graph execution timed out"))
                }
            }
        } else {
            execution_future.await
        };
        
        let end_time = chrono::Utc::now();
        let total_duration = (end_time - start_time).to_std().unwrap_or(Duration::ZERO);
        
        let success = execution_result.is_ok() && errors.is_empty();
        
        if self.config.verbose_logging {
            if success {
                #[cfg(feature = "observability")]
                tracing::info!("Graph execution completed successfully: {}", execution_id);
                #[cfg(not(feature = "observability"))]
                tracing::debug!("Graph execution completed successfully: {}", execution_id);
            } else {
                #[cfg(feature = "observability")]
                tracing::error!("Graph execution failed: {}", execution_id);
                #[cfg(not(feature = "observability"))]
                tracing::debug!("Graph execution failed: {}", execution_id);
            }
        }
        
        Ok(GraphExecutionResult {
            execution_id,
            final_state: state,
            metrics: ExecutionMetrics {
                total_duration,
                nodes_executed,
                node_durations,
                start_time,
                end_time,
                peak_memory_usage: None, // Would be tracked in a real implementation
            },
            execution_path,
            errors,
            success,
        })
    }
    
    /// Internal execution logic
    async fn execute_internal(
        &self,
        graph: &WorkflowGraph,
        state: &mut GraphState,
        execution_id: &str,
        execution_path: &mut Vec<NodeId>,
        errors: &mut Vec<ExecutionError>,
        node_durations: &mut HashMap<String, Duration>,
        nodes_executed: &mut usize,
    ) -> RGraphResult<()> {
        let entry_points = graph.entry_points();
        
        if entry_points.is_empty() {
            return Err(RGraphError::execution("No entry points defined"));
        }
        
        // Start execution from entry points
        match self.config.mode {
            ExecutionMode::Sequential => {
                // Execute entry points sequentially
                for entry_point in entry_points {
                    self.execute_node_sequence(
                        graph,
                        state,
                        &entry_point,
                        execution_id,
                        execution_path,
                        errors,
                        node_durations,
                        nodes_executed,
                        0, // recursion depth
                    ).await?;
                }
            }
            ExecutionMode::Parallel | ExecutionMode::MaxParallel => {
                // Execute entry points in parallel
                let futures: Vec<BoxFuture<'_, RGraphResult<()>>> = entry_points
                    .into_iter()
                    .map(|entry_point| {
                        Box::pin(self.execute_node_sequence(
                            graph,
                            state,
                            &entry_point,
                            execution_id,
                            execution_path,
                            errors,
                            node_durations,
                            nodes_executed,
                            0,
                        )) as BoxFuture<'_, RGraphResult<()>>
                    })
                    .collect();
                
                // Wait for all parallel executions
                let results = futures::future::join_all(futures).await;
                
                // Check for errors
                for result in results {
                    if let Err(e) = result {
                        if !self.config.continue_on_error {
                            return Err(e);
                        }
                        errors.push(ExecutionError {
                            node_id: "parallel_execution".to_string(),
                            error_message: e.to_string(),
                            timestamp: chrono::Utc::now(),
                            error_type: "ParallelExecutionError".to_string(),
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute a sequence of nodes starting from a given node
    fn execute_node_sequence(
        &self,
        graph: &WorkflowGraph,
        state: &mut GraphState,
        current_node_id: &NodeId,
        execution_id: &str,
        execution_path: &mut Vec<NodeId>,
        errors: &mut Vec<ExecutionError>,
        node_durations: &mut HashMap<String, Duration>,
        nodes_executed: &mut usize,
        recursion_depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = RGraphResult<()>> + '_>> {
        Box::pin(async move {
        // Check recursion depth
        if recursion_depth >= self.config.max_recursion_depth {
            return Err(RGraphError::execution(
                format!("Maximum recursion depth {} exceeded", self.config.max_recursion_depth)
            ));
        }
        
        // Check node limit
        if let Some(max_nodes) = self.config.max_nodes {
            if *nodes_executed >= max_nodes {
                return Err(RGraphError::execution(
                    format!("Maximum node limit {} exceeded", max_nodes)
                ));
            }
        }
        
        // Get the node (in a real implementation, we'd look it up from the graph)
        // For now, we'll simulate node execution
        let node = self.get_node_from_graph(graph, current_node_id)?;
        
        // Create execution context
        let mut context = ExecutionContext::new(
            graph.id().to_string(),
            current_node_id.clone(),
        );
        context.execution_path = execution_path.clone();
        
        // Execute the node with timeout if specified
        let node_start = std::time::Instant::now();
        let execution_result = if let Some(max_time) = self.config.max_node_execution_time {
            match timeout(max_time, node.execute(state, &context)).await {
                Ok(result) => result,
                Err(_) => {
                    let error = ExecutionError {
                        node_id: current_node_id.as_str().to_string(),
                        error_message: "Node execution timed out".to_string(),
                        timestamp: chrono::Utc::now(),
                        error_type: "NodeTimeout".to_string(),
                    };
                    errors.push(error);
                    
                    if self.config.continue_on_error {
                        return Ok(());
                    } else {
                        return Err(RGraphError::node(
                            current_node_id.as_str(),
                            "Node execution timed out"
                        ));
                    }
                }
            }
        } else {
            node.execute(state, &context).await
        };
        
        let node_duration = node_start.elapsed();
        node_durations.insert(current_node_id.as_str().to_string(), node_duration);
        *nodes_executed += 1;
        execution_path.push(current_node_id.clone());
        
        if self.config.verbose_logging {
            #[cfg(feature = "observability")]
            tracing::debug!(
                "Executed node '{}' in {:.2}ms",
                current_node_id.as_str(),
                node_duration.as_millis()
            );
            #[cfg(not(feature = "observability"))]
            tracing::debug!(
                "Executed node '{}' in {:.2}ms",
                current_node_id.as_str(),
                node_duration.as_millis()
            );
        }
        
        // Handle execution result
        match execution_result {
            Ok(ExecutionResult::Continue) => {
                // Continue to next nodes (would be determined by graph structure)
                self.continue_to_next_nodes(
                    graph,
                    state,
                    current_node_id,
                    execution_id,
                    execution_path,
                    errors,
                    node_durations,
                    nodes_executed,
                    recursion_depth + 1,
                ).await?;
            }
            Ok(ExecutionResult::Stop) => {
                // Stop execution
                if self.config.verbose_logging {
                    #[cfg(feature = "observability")]
                    tracing::info!("Node '{}' requested execution stop", current_node_id.as_str());
                    #[cfg(not(feature = "observability"))]
                    tracing::debug!("Node '{}' requested execution stop", current_node_id.as_str());
                }
            }
            Ok(ExecutionResult::JumpTo(next_node_id)) => {
                // Jump to specific node
                self.execute_node_sequence(
                    graph,
                    state,
                    &next_node_id,
                    execution_id,
                    execution_path,
                    errors,
                    node_durations,
                    nodes_executed,
                    recursion_depth + 1,
                ).await?;
            }
            Ok(ExecutionResult::Route(next_node_id)) => {
                // Route to next node based on routing logic
                let next_node_id = NodeId::new(next_node_id);
                self.execute_node_sequence(
                    graph,
                    state,
                    &next_node_id,
                    execution_id,
                    execution_path,
                    errors,
                    node_durations,
                    nodes_executed,
                    recursion_depth + 1,
                ).await?;
            }
            Err(e) => {
                let error = ExecutionError {
                    node_id: current_node_id.as_str().to_string(),
                    error_message: e.to_string(),
                    timestamp: chrono::Utc::now(),
                    error_type: "NodeExecutionError".to_string(),
                };
                errors.push(error);
                
                if !self.config.continue_on_error {
                    return Err(e);
                }
            }
        }
        
        Ok(())
        })
    }
    
    /// Continue execution to the next nodes in the graph
    async fn continue_to_next_nodes(
        &self,
        graph: &WorkflowGraph,
        state: &mut GraphState,
        current_node_id: &NodeId,
        execution_id: &str,
        execution_path: &mut Vec<NodeId>,
        errors: &mut Vec<ExecutionError>,
        node_durations: &mut HashMap<String, Duration>,
        nodes_executed: &mut usize,
        recursion_depth: usize,
    ) -> RGraphResult<()> {
        // In a real implementation, we'd traverse the graph edges
        // For now, we'll simulate this by checking for predefined next nodes
        
        // This is where we'd implement the actual graph traversal logic
        // based on the petgraph structure and edge conditions
        
        Ok(())
    }
    
    /// Get a node from the graph (placeholder implementation)
    fn get_node_from_graph(
        &self,
        _graph: &WorkflowGraph,
        node_id: &NodeId,
    ) -> RGraphResult<Arc<dyn Node>> {
        // In a real implementation, we'd look up the node from the graph's internal structure
        // For now, return an error indicating this is not implemented
        Err(RGraphError::execution(
            format!("Node lookup not implemented for node: {}", node_id.as_str())
        ))
    }
}

impl Default for ExecutionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience method to execute a graph with default engine
impl WorkflowGraph {
    /// Execute the graph with the given initial state
    pub async fn execute(&self, state: GraphState) -> RGraphResult<GraphExecutionResult> {
        let engine = ExecutionEngine::new();
        engine.execute(self, state).await
    }
    
    /// Execute the graph with custom execution configuration
    pub async fn execute_with_config(
        &self,
        state: GraphState,
        config: ExecutionConfig,
    ) -> RGraphResult<GraphExecutionResult> {
        let engine = ExecutionEngine::with_config(config);
        engine.execute(self, state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Node, ExecutionContext, ExecutionResult};
    use crate::state::{GraphState, StateValue};
    use async_trait::async_trait;
    
    // Mock node for testing
    struct MockNode {
        id: NodeId,
        name: String,
        result: ExecutionResult,
    }
    
    impl MockNode {
        fn new(
            id: impl Into<NodeId>,
            name: impl Into<String>,
            result: ExecutionResult,
        ) -> Arc<Self> {
            Arc::new(Self {
                id: id.into(),
                name: name.into(),
                result,
            })
        }
    }
    
    #[async_trait]
    impl Node for MockNode {
        async fn execute(
            &self,
            state: &mut GraphState,
            _context: &ExecutionContext,
        ) -> RGraphResult<ExecutionResult> {
            // Add execution marker to state
            let mut executed_nodes = state.get("executed_nodes")
                .unwrap_or(StateValue::Array(vec![]));
            
            if let StateValue::Array(ref mut nodes) = executed_nodes {
                nodes.push(StateValue::String(self.name.clone()));
            }
            
            state.set("executed_nodes", executed_nodes);
            
            Ok(self.result.clone())
        }
        
        fn id(&self) -> &NodeId {
            &self.id
        }
        
        fn name(&self) -> &str {
            &self.name
        }
    }
    
    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        
        assert_eq!(config.max_execution_time, Some(Duration::from_secs(300)));
        assert_eq!(config.max_node_execution_time, Some(Duration::from_secs(60)));
        assert_eq!(config.max_nodes, Some(1000));
        assert!(!config.continue_on_error);
        assert_eq!(config.mode, ExecutionMode::Sequential);
        assert!(!config.verbose_logging);
        assert_eq!(config.max_recursion_depth, 100);
    }
    
    #[test]
    fn test_execution_metrics() {
        let start_time = chrono::Utc::now();
        let end_time = start_time + chrono::Duration::seconds(10);
        
        let metrics = ExecutionMetrics {
            total_duration: Duration::from_secs(10),
            nodes_executed: 5,
            node_durations: HashMap::new(),
            start_time,
            end_time,
            peak_memory_usage: Some(1024 * 1024), // 1MB
        };
        
        assert_eq!(metrics.total_duration, Duration::from_secs(10));
        assert_eq!(metrics.nodes_executed, 5);
        assert_eq!(metrics.peak_memory_usage, Some(1024 * 1024));
    }
    
    #[test]
    fn test_execution_error() {
        let error = ExecutionError {
            node_id: "test_node".to_string(),
            error_message: "Test error".to_string(),
            timestamp: chrono::Utc::now(),
            error_type: "TestError".to_string(),
        };
        
        assert_eq!(error.node_id, "test_node");
        assert_eq!(error.error_message, "Test error");
        assert_eq!(error.error_type, "TestError");
    }
    
    #[tokio::test]
    async fn test_execution_engine_creation() {
        let engine = ExecutionEngine::new();
        assert_eq!(engine.config.mode, ExecutionMode::Sequential);
        
        let custom_config = ExecutionConfig {
            mode: ExecutionMode::Parallel,
            verbose_logging: true,
            ..Default::default()
        };
        
        let custom_engine = ExecutionEngine::with_config(custom_config);
        assert_eq!(custom_engine.config.mode, ExecutionMode::Parallel);
        assert!(custom_engine.config.verbose_logging);
    }
}