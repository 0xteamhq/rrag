//! # Observability
//!
//! Monitoring and observability for graph execution.

use crate::core::{ExecutionContext, NodeId};
// Future use for observability features
use async_trait::async_trait;
use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Graph execution observer
#[async_trait]
pub trait GraphObserver: Send + Sync {
    /// Called when graph execution starts
    async fn on_execution_start(&self, graph_id: &str, context: &ExecutionContext);

    /// Called when graph execution ends
    async fn on_execution_end(&self, graph_id: &str, success: bool, duration: Duration);

    /// Called when node execution starts
    async fn on_node_start(&self, node_id: &NodeId, context: &ExecutionContext);

    /// Called when node execution ends
    async fn on_node_end(&self, node_id: &NodeId, success: bool, duration: Duration);

    /// Called when state changes
    async fn on_state_change(&self, key: &str, old_value: Option<&str>, new_value: &str);
}

/// Basic logging observer
pub struct LoggingObserver;

#[async_trait]
impl GraphObserver for LoggingObserver {
    async fn on_execution_start(&self, graph_id: &str, _context: &ExecutionContext) {
        #[cfg(feature = "observability")]
        tracing::info!("Graph execution started: {}", graph_id);
        #[cfg(not(feature = "observability"))]
        {
            let _ = (graph_id, _context);
            eprintln!("Graph execution started: {}", graph_id);
        }
    }

    async fn on_execution_end(&self, graph_id: &str, success: bool, duration: Duration) {
        #[cfg(feature = "observability")]
        tracing::info!(
            "Graph execution ended: {} (success: {}, duration: {:?})",
            graph_id,
            success,
            duration
        );
        #[cfg(not(feature = "observability"))]
        eprintln!(
            "Graph execution ended: {} (success: {}, duration: {:?})",
            graph_id, success, duration
        );
    }

    async fn on_node_start(&self, node_id: &NodeId, _context: &ExecutionContext) {
        #[cfg(feature = "observability")]
        tracing::debug!("Node execution started: {}", node_id.as_str());
        #[cfg(not(feature = "observability"))]
        {
            let _ = _context;
            eprintln!("Node execution started: {}", node_id.as_str());
        }
    }

    async fn on_node_end(&self, node_id: &NodeId, success: bool, duration: Duration) {
        #[cfg(feature = "observability")]
        tracing::debug!(
            "Node execution ended: {} (success: {}, duration: {:?})",
            node_id.as_str(),
            success,
            duration
        );
        #[cfg(not(feature = "observability"))]
        eprintln!(
            "Node execution ended: {} (success: {}, duration: {:?})",
            node_id.as_str(),
            success,
            duration
        );
    }

    async fn on_state_change(&self, key: &str, old_value: Option<&str>, new_value: &str) {
        #[cfg(feature = "observability")]
        tracing::trace!(
            "State change: {} = {} (was: {:?})",
            key,
            new_value,
            old_value
        );
        #[cfg(not(feature = "observability"))]
        eprintln!(
            "State change: {} = {} (was: {:?})",
            key, new_value, old_value
        );
    }
}

/// Execution metrics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExecutionMetrics {
    pub nodes_executed: usize,
    pub total_duration: Duration,
    pub node_metrics: Vec<NodeMetrics>,
}

/// Node-specific metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NodeMetrics {
    pub node_id: String,
    pub execution_count: usize,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub success_rate: f32,
}

/// Observability configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ObservabilityConfig {
    pub enable_logging: bool,
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub log_level: String,
}
