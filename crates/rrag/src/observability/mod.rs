//! # RRAG Observability System
//!
//! Enterprise-grade observability and monitoring for production RAG deployments.
//!
//! This module provides comprehensive monitoring, metrics collection, alerting,
//! and visualization capabilities to ensure your RAG system operates reliably
//! at scale. It includes real-time dashboards, intelligent alerting, performance
//! profiling, and data export capabilities.
//!
//! ## Features
//!
//! - **Metrics Collection**: Prometheus-compatible metrics with custom dashboards
//! - **Real-time Monitoring**: Live system health and performance tracking
//! - **Intelligent Alerting**: Smart alerts with multiple notification channels
//! - **Performance Profiling**: Bottleneck detection and optimization insights
//! - **Health Monitoring**: Component-level health checks and diagnostics
//! - **Log Aggregation**: Structured logging with search and analysis
//! - **Data Export**: Export metrics and logs for external analysis
//! - **Data Retention**: Configurable retention policies for long-term storage
//!
//! ## Quick Start
//!
//! ### Basic Observability Setup
//! ```rust
//! use rrag::observability::{ObservabilitySystem, ObservabilityConfig};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let observability = ObservabilitySystem::new(
//!     ObservabilityConfig::default()
//!         .with_metrics(true)
//!         .with_monitoring(true)
//!         .with_alerting(true)
//!         .with_dashboard(true)
//! ).await?;
//!
//! // Start the observability system
//! observability.start().await?;
//!
//! // Access components
//! let metrics = observability.metrics();
//! let monitoring = observability.monitoring();
//! let alerting = observability.alerting();
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Metrics Collection
//! ```rust
//! use rrag::observability::{MetricsCollector, MetricType};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let metrics = MetricsCollector::new();
//!
//! // Counter metrics
//! metrics.inc_counter("requests_total").await?;
//! metrics.inc_counter_by("documents_processed", 10).await?;
//!
//! // Gauge metrics
//! metrics.set_gauge("active_users", 150.0).await?;
//! metrics.set_gauge("memory_usage_mb", 512.0).await?;
//!
//! // Histogram metrics for latency
//! metrics.observe_histogram("request_duration_ms", 45.2).await?;
//!
//! // Timer metrics
//! let timer = metrics.start_timer("query_processing_time");
//! // ... do work ...
//! timer.stop().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Alert Configuration
//! ```rust
//! use rrag::observability::{AlertManager, AlertRule, AlertSeverity};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let alert_manager = AlertManager::new();
//!
//! // High latency alert
//! let latency_alert = AlertRule::new("high_latency")
//!     .condition("avg(request_duration_ms) > 1000")
//!     .severity(AlertSeverity::High)
//!     .description("Query latency is too high")
//!     .notification_channels(vec!["slack", "email"])
//!     .cooldown_minutes(5);
//!
//! alert_manager.add_rule(latency_alert).await?;
//!
//! // Error rate alert
//! let error_alert = AlertRule::new("high_error_rate")
//!     .condition("rate(error_count) > 0.05")
//!     .severity(AlertSeverity::Critical)
//!     .description("Error rate exceeded 5%")
//!     .notification_channels(vec!["pagerduty", "slack"]);
//!
//! alert_manager.add_rule(error_alert).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Health Monitoring
//! ```rust
//! use rrag::observability::{HealthMonitor, HealthCheck};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let health_monitor = HealthMonitor::new();
//!
//! // Add custom health checks
//! health_monitor.add_check(
//!     "database",
//!     Box::new(|_| async {
//!         // Check database connectivity
//!         Ok(true)
//!     })
//! ).await?;
//!
//! health_monitor.add_check(
//!     "embedding_service",
//!     Box::new(|_| async {
//!         // Check embedding service
//!         Ok(true)
//!     })
//! ).await?;
//!
//! // Get overall health status
//! let status = health_monitor.check_all().await?;
//! println!("System health: {:?}", status.overall_status);
//! # Ok(())
//! # }
//! ```
//!
//! ### Performance Profiling
//! ```rust
//! use rrag::observability::{PerformanceProfiler, ProfileConfig};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let profiler = PerformanceProfiler::new(ProfileConfig::default());
//!
//! // Start profiling a specific operation
//! let profile_id = profiler.start_profile("document_processing").await?;
//!
//! // ... perform work ...
//!
//! let profile = profiler.stop_profile(profile_id).await?;
//!
//! // Analyze bottlenecks
//! let bottlenecks = profiler.analyze_bottlenecks(5).await?;
//! for bottleneck in bottlenecks.bottlenecks {
//!     println!("Bottleneck: {} took {:.2}ms",
//!              bottleneck.operation,
//!              bottleneck.average_duration_ms);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Dashboard and Visualization
//! ```rust
//! use rrag::observability::{DashboardServer, DashboardConfig};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let dashboard = DashboardServer::new(
//!     DashboardConfig::default()
//!         .with_port(3000)
//!         .with_realtime_updates(true)
//!         .with_custom_charts(vec![
//!             "query_latency_histogram",
//!             "documents_processed_rate",
//!             "error_rate_by_component"
//!         ])
//! );
//!
//! // Start dashboard server
//! dashboard.start().await?;
//! println!("Dashboard available at: http://localhost:3000");
//! # Ok(())
//! # }
//! ```
//!
//! ## Integration Examples
//!
//! ### With RAG System
//! ```rust
//! use rrag::{RragSystemBuilder, observability::ObservabilityConfig};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let rag = RragSystemBuilder::new()
//!     .with_observability(
//!         ObservabilityConfig::production()
//!             .with_prometheus_endpoint(true)
//!             .with_health_checks(true)
//!             .with_performance_profiling(true)
//!     )
//!     .build()
//!     .await?;
//!
//! // System automatically reports metrics
//! let results = rag.search("query", Some(10)).await?;
//! // Metrics like query_count, search_latency, results_returned are automatic
//! # Ok(())
//! # }
//! ```

pub mod alerting;
pub mod dashboard;
pub mod export;
pub mod health;
pub mod logging;
pub mod metrics;
pub mod monitoring;
pub mod profiling;
pub mod retention;

// Core observability system
pub use alerting::{
    AlertCondition, AlertConfig, AlertManager, AlertNotification, AlertRule, AlertSeverity,
    NotificationChannel,
};
pub use dashboard::{
    ChartData, DashboardConfig, DashboardHandler, DashboardMetrics, DashboardServer,
    RealtimeMetrics, WebSocketManager,
};
pub use export::{
    ExportConfig, ExportFormat, ExportManager, MetricsExporter, ReportConfig, ReportGenerator,
};
pub use health::{
    ComponentStatus, HealthChecker, HealthConfig, HealthMonitor, HealthReport, ServiceHealth,
};
pub use logging::{
    LogAggregator, LogConfig, LogEntry, LogFilter, LogLevel, LogQuery, StructuredLogger,
};
pub use metrics::{
    CounterMetric, GaugeMetric, HistogramMetric, Metric, MetricType, MetricValue, MetricsCollector,
    MetricsRegistry, TimerMetric,
};
pub use monitoring::{
    MonitoringConfig, MonitoringService, PerformanceMonitor, SearchAnalyzer, SystemMonitor,
    UserActivityTracker,
};
pub use profiling::{
    BottleneckAnalysis, PerformanceProfiler, PerformanceReport, ProfileData, Profiler,
    ProfilingConfig,
};
pub use retention::{
    ArchiveManager, DataRetention, HistoricalAnalyzer, RetentionConfig, RetentionPolicy,
};

use crate::{RragError, RragResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main observability system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// System identification
    pub system_id: String,
    pub environment: String,

    /// Component configurations
    pub metrics: metrics::MetricsConfig,
    pub monitoring: monitoring::MonitoringConfig,
    pub alerting: alerting::AlertConfig,
    pub dashboard: dashboard::DashboardConfig,
    pub logging: logging::LogConfig,
    pub health: health::HealthConfig,
    pub profiling: profiling::ProfilingConfig,
    pub export: export::ExportConfig,
    pub retention: retention::RetentionConfig,

    /// Global settings
    pub enabled: bool,
    pub sample_rate: f64,
    pub batch_size: usize,
    pub flush_interval_seconds: u64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            system_id: "rrag-system".to_string(),
            environment: "production".to_string(),
            metrics: metrics::MetricsConfig::default(),
            monitoring: monitoring::MonitoringConfig::default(),
            alerting: alerting::AlertConfig::default(),
            dashboard: dashboard::DashboardConfig::default(),
            logging: logging::LogConfig::default(),
            health: health::HealthConfig::default(),
            profiling: profiling::ProfilingConfig::default(),
            export: export::ExportConfig::default(),
            retention: retention::RetentionConfig::default(),
            enabled: true,
            sample_rate: 1.0,
            batch_size: 100,
            flush_interval_seconds: 30,
        }
    }
}

/// Main observability system
pub struct ObservabilitySystem {
    config: ObservabilityConfig,
    metrics: Arc<MetricsCollector>,
    monitoring: Arc<SystemMonitor>,
    alerting: Arc<AlertManager>,
    dashboard: Arc<DashboardServer>,
    logging: Arc<LogAggregator>,
    health: Arc<HealthMonitor>,
    profiling: Arc<PerformanceProfiler>,
    export: Arc<ExportManager>,
    retention: Arc<DataRetention>,

    // Internal state
    start_time: DateTime<Utc>,
    is_running: Arc<RwLock<bool>>,
}

impl ObservabilitySystem {
    /// Create new observability system
    pub async fn new(config: ObservabilityConfig) -> RragResult<Self> {
        if !config.enabled {
            return Err(RragError::config("observability.enabled", "true", "false"));
        }

        let metrics = Arc::new(MetricsCollector::new(config.metrics.clone()).await?);
        let monitoring =
            Arc::new(SystemMonitor::new(config.monitoring.clone(), metrics.clone()).await?);

        let alerting = Arc::new(AlertManager::new(config.alerting.clone(), metrics.clone()).await?);

        let dashboard = Arc::new(
            DashboardServer::new(
                config.dashboard.clone(),
                metrics.clone(),
                monitoring.clone(),
            )
            .await?,
        );

        let logging = Arc::new(LogAggregator::new(config.logging.clone()).await?);
        let health = Arc::new(HealthMonitor::new(config.health.clone()).await?);
        let profiling = Arc::new(PerformanceProfiler::new(config.profiling.clone()).await?);
        let export = Arc::new(ExportManager::new(config.export.clone()).await?);
        let retention = Arc::new(DataRetention::new(config.retention.clone()).await?);

        Ok(Self {
            config,
            metrics,
            monitoring,
            alerting,
            dashboard,
            logging,
            health,
            profiling,
            export,
            retention,
            start_time: Utc::now(),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the observability system
    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config(
                "observability",
                "stopped",
                "already running",
            ));
        }

        // Start all components
        self.metrics.start().await?;
        self.monitoring.start().await?;
        self.alerting.start().await?;
        self.dashboard.start().await?;
        self.logging.start().await?;
        self.health.start().await?;
        self.profiling.start().await?;
        self.export.start().await?;
        self.retention.start().await?;

        *running = true;
        tracing::info!("Observability system started successfully");

        Ok(())
    }

    /// Stop the observability system
    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        // Stop all components in reverse order
        self.retention.stop().await?;
        self.export.stop().await?;
        self.profiling.stop().await?;
        self.health.stop().await?;
        self.logging.stop().await?;
        self.dashboard.stop().await?;
        self.alerting.stop().await?;
        self.monitoring.stop().await?;
        self.metrics.stop().await?;

        *running = false;
        tracing::info!("Observability system stopped successfully");

        Ok(())
    }

    /// Check if system is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get metrics collector
    pub fn metrics(&self) -> &Arc<MetricsCollector> {
        &self.metrics
    }

    /// Get system monitor
    pub fn monitoring(&self) -> &Arc<SystemMonitor> {
        &self.monitoring
    }

    /// Get alert manager
    pub fn alerting(&self) -> &Arc<AlertManager> {
        &self.alerting
    }

    /// Get dashboard server
    pub fn dashboard(&self) -> &Arc<DashboardServer> {
        &self.dashboard
    }

    /// Get log aggregator
    pub fn logging(&self) -> &Arc<LogAggregator> {
        &self.logging
    }

    /// Get health monitor
    pub fn health(&self) -> &Arc<HealthMonitor> {
        &self.health
    }

    /// Get profiler
    pub fn profiling(&self) -> &Arc<PerformanceProfiler> {
        &self.profiling
    }

    /// Get export manager
    pub fn export(&self) -> &Arc<ExportManager> {
        &self.export
    }

    /// Get retention manager
    pub fn retention(&self) -> &Arc<DataRetention> {
        &self.retention
    }

    /// Get system configuration
    pub fn config(&self) -> &ObservabilityConfig {
        &self.config
    }

    /// Get system uptime
    pub fn uptime(&self) -> chrono::Duration {
        Utc::now() - self.start_time
    }

    /// Get comprehensive system status
    pub async fn status(&self) -> ObservabilityStatus {
        ObservabilityStatus {
            running: self.is_running().await,
            uptime_seconds: self.uptime().num_seconds(),
            components: HashMap::from([
                ("metrics".to_string(), self.metrics.is_healthy().await),
                ("monitoring".to_string(), self.monitoring.is_healthy().await),
                ("alerting".to_string(), self.alerting.is_healthy().await),
                ("dashboard".to_string(), self.dashboard.is_healthy().await),
                ("logging".to_string(), self.logging.is_healthy().await),
                ("health".to_string(), self.health.is_healthy().await),
                ("profiling".to_string(), self.profiling.is_healthy().await),
                ("export".to_string(), self.export.is_healthy().await),
                ("retention".to_string(), self.retention.is_healthy().await),
            ]),
            last_check: Utc::now(),
        }
    }
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityStatus {
    pub running: bool,
    pub uptime_seconds: i64,
    pub components: HashMap<String, bool>,
    pub last_check: DateTime<Utc>,
}

/// Builder for observability system
pub struct ObservabilityBuilder {
    config: ObservabilityConfig,
}

impl ObservabilityBuilder {
    pub fn new() -> Self {
        Self {
            config: ObservabilityConfig::default(),
        }
    }

    pub fn with_system_id(mut self, id: impl Into<String>) -> Self {
        self.config.system_id = id.into();
        self
    }

    pub fn with_environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = env.into();
        self
    }

    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.config.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn with_flush_interval(mut self, seconds: u64) -> Self {
        self.config.flush_interval_seconds = seconds;
        self
    }

    pub fn enable_dashboard(mut self, enabled: bool) -> Self {
        self.config.dashboard.enabled = enabled;
        self
    }

    pub fn with_dashboard_port(mut self, port: u16) -> Self {
        self.config.dashboard.port = port;
        self
    }

    pub fn enable_alerts(mut self, enabled: bool) -> Self {
        self.config.alerting.enabled = enabled;
        self
    }

    pub fn enable_profiling(mut self, enabled: bool) -> Self {
        self.config.profiling.enabled = enabled;
        self
    }

    pub fn with_retention_days(mut self, days: u32) -> Self {
        self.config.retention.retention_days = days;
        self
    }

    pub async fn build(self) -> RragResult<ObservabilitySystem> {
        ObservabilitySystem::new(self.config).await
    }
}

impl Default for ObservabilityBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_observability_system_creation() {
        let config = ObservabilityConfig::default();
        let system = ObservabilitySystem::new(config).await.unwrap();

        assert!(!system.is_running().await);
        assert_eq!(system.config.system_id, "rrag-system");
    }

    #[tokio::test]
    async fn test_observability_builder() {
        let system = ObservabilityBuilder::new()
            .with_system_id("test-system")
            .with_environment("test")
            .with_sample_rate(0.5)
            .build()
            .await
            .unwrap();

        assert_eq!(system.config.system_id, "test-system");
        assert_eq!(system.config.environment, "test");
        assert_eq!(system.config.sample_rate, 0.5);
    }

    #[tokio::test]
    async fn test_system_lifecycle() {
        let system = ObservabilityBuilder::new()
            .with_system_id("test-lifecycle")
            .build()
            .await
            .unwrap();

        assert!(!system.is_running().await);

        system.start().await.unwrap();
        assert!(system.is_running().await);

        system.stop().await.unwrap();
        assert!(!system.is_running().await);
    }

    #[tokio::test]
    async fn test_system_status() {
        let system = ObservabilityBuilder::new().build().await.unwrap();
        let status = system.status().await;

        assert!(!status.running);
        assert!(status.uptime_seconds >= 0);
        assert_eq!(status.components.len(), 9); // All components
    }
}
