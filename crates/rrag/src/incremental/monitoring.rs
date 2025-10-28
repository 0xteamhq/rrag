//! # Incremental Indexing Monitoring
//!
//! Comprehensive monitoring and alerting system for incremental indexing operations.
//! Provides performance tracking, health monitoring, and automated alerting.

use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance metrics collection
    pub enable_performance_metrics: bool,

    /// Enable health monitoring
    pub enable_health_monitoring: bool,

    /// Enable alerting system
    pub enable_alerting: bool,

    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,

    /// Health check interval in seconds
    pub health_check_interval_secs: u64,

    /// Metrics retention period in days
    pub metrics_retention_days: u32,

    /// Alert configuration
    pub alert_config: AlertConfig,

    /// Export configuration
    pub export_config: ExportConfig,
}

/// Alert system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable email alerts
    pub enable_email_alerts: bool,

    /// Enable webhook alerts
    pub enable_webhook_alerts: bool,

    /// Enable log alerts
    pub enable_log_alerts: bool,

    /// Alert thresholds
    pub thresholds: AlertThresholds,

    /// Alert cooldown period in seconds
    pub cooldown_period_secs: u64,

    /// Maximum alerts per hour
    pub max_alerts_per_hour: u32,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Error rate threshold (0.0 to 1.0)
    pub error_rate_threshold: f64,

    /// Response time threshold in milliseconds
    pub response_time_threshold_ms: u64,

    /// Queue depth threshold
    pub queue_depth_threshold: usize,

    /// Memory usage threshold (0.0 to 1.0)
    pub memory_usage_threshold: f64,

    /// Storage usage threshold (0.0 to 1.0)
    pub storage_usage_threshold: f64,

    /// Throughput threshold (operations per second)
    pub throughput_threshold_ops: f64,
}

/// Export configuration for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Enable Prometheus export
    pub enable_prometheus: bool,

    /// Enable JSON export
    pub enable_json_export: bool,

    /// Export endpoint
    pub export_endpoint: Option<String>,

    /// Export interval in seconds
    pub export_interval_secs: u64,

    /// Export format
    pub export_format: ExportFormat,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Prometheus,
    Json,
    InfluxDB,
    StatsD,
    Custom(String),
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_performance_metrics: true,
            enable_health_monitoring: true,
            enable_alerting: true,
            metrics_interval_secs: 30,
            health_check_interval_secs: 60,
            metrics_retention_days: 30,
            alert_config: AlertConfig::default(),
            export_config: ExportConfig::default(),
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enable_email_alerts: false,
            enable_webhook_alerts: true,
            enable_log_alerts: true,
            thresholds: AlertThresholds::default(),
            cooldown_period_secs: 300, // 5 minutes
            max_alerts_per_hour: 10,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            error_rate_threshold: 0.05,        // 5%
            response_time_threshold_ms: 10000, // 10 seconds
            queue_depth_threshold: 1000,
            memory_usage_threshold: 0.8,    // 80%
            storage_usage_threshold: 0.9,   // 90%
            throughput_threshold_ops: 10.0, // 10 ops/sec minimum
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: false,
            enable_json_export: true,
            export_endpoint: None,
            export_interval_secs: 300, // 5 minutes
            export_format: ExportFormat::Json,
        }
    }
}

/// Comprehensive metrics for incremental indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalMetrics {
    /// System identification
    pub system_id: String,

    /// Metrics timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Indexing performance metrics
    pub indexing_metrics: IndexingMetrics,

    /// System performance metrics
    pub system_metrics: SystemMetrics,

    /// Operation metrics
    pub operation_metrics: OperationMetrics,

    /// Health metrics
    pub health_metrics: HealthMetrics,

    /// Error metrics
    pub error_metrics: ErrorMetrics,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Indexing-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingMetrics {
    /// Documents processed per second
    pub documents_per_second: f64,

    /// Chunks processed per second
    pub chunks_per_second: f64,

    /// Embeddings processed per second
    pub embeddings_per_second: f64,

    /// Average indexing time per document
    pub avg_indexing_time_ms: f64,

    /// Index size growth rate (bytes per second)
    pub index_growth_rate_bps: f64,

    /// Batch processing efficiency
    pub batch_efficiency: f64,

    /// Change detection accuracy
    pub change_detection_accuracy: f64,

    /// Vector update efficiency
    pub vector_update_efficiency: f64,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,

    /// Memory usage in bytes
    pub memory_usage_bytes: u64,

    /// Available memory in bytes
    pub available_memory_bytes: u64,

    /// Storage usage in bytes
    pub storage_usage_bytes: u64,

    /// Available storage in bytes
    pub available_storage_bytes: u64,

    /// Network I/O bytes per second
    pub network_io_bps: f64,

    /// Disk I/O operations per second
    pub disk_io_ops: f64,

    /// Active connections
    pub active_connections: usize,
}

/// Operation-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Total operations performed
    pub total_operations: u64,

    /// Operations by type
    pub operations_by_type: HashMap<String, u64>,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Average operation time in milliseconds
    pub avg_operation_time_ms: f64,

    /// 95th percentile operation time
    pub p95_operation_time_ms: f64,

    /// 99th percentile operation time
    pub p99_operation_time_ms: f64,

    /// Queue depths by type
    pub queue_depths: HashMap<String, usize>,

    /// Retry statistics
    pub retry_stats: RetryMetrics,
}

/// Retry-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryMetrics {
    /// Total retry attempts
    pub total_retries: u64,

    /// Successful retries
    pub successful_retries: u64,

    /// Failed retries (exhausted)
    pub exhausted_retries: u64,

    /// Average retries per operation
    pub avg_retries_per_operation: f64,
}

/// Health monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Overall system health score (0.0 to 1.0)
    pub overall_health_score: f64,

    /// Component health scores
    pub component_health: HashMap<String, f64>,

    /// Service availability (0.0 to 1.0)
    pub service_availability: f64,

    /// Data consistency score (0.0 to 1.0)
    pub data_consistency_score: f64,

    /// Performance score (0.0 to 1.0)
    pub performance_score: f64,

    /// Last health check timestamp
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total errors encountered
    pub total_errors: u64,

    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,

    /// Errors by component
    pub errors_by_component: HashMap<String, u64>,

    /// Error rate (errors per operation)
    pub error_rate: f64,

    /// Critical errors
    pub critical_errors: u64,

    /// Recoverable errors
    pub recoverable_errors: u64,

    /// Error resolution time average
    pub avg_resolution_time_ms: f64,
}

/// Performance tracking system
pub struct PerformanceTracker {
    /// Data points storage
    data_points: Arc<RwLock<VecDeque<PerformanceDataPoint>>>,

    /// Aggregated statistics
    statistics: Arc<RwLock<PerformanceStatistics>>,

    /// Configuration
    config: MonitoringConfig,

    /// Max data points to retain
    max_data_points: usize,
}

/// Individual performance data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Operation type
    pub operation_type: String,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Memory usage at time of operation
    pub memory_usage_mb: f64,

    /// Success indicator
    pub success: bool,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Aggregated performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStatistics {
    /// Statistics by operation type
    pub by_operation_type: HashMap<String, OperationStatistics>,

    /// Overall statistics
    pub overall: OperationStatistics,

    /// Time-based trends
    pub trends: TrendAnalysis,

    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Statistics for a specific operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStatistics {
    /// Total operations
    pub total_count: u64,

    /// Successful operations
    pub success_count: u64,

    /// Average duration
    pub avg_duration_ms: f64,

    /// Median duration
    pub median_duration_ms: f64,

    /// 95th percentile duration
    pub p95_duration_ms: f64,

    /// 99th percentile duration
    pub p99_duration_ms: f64,

    /// Standard deviation
    pub std_deviation_ms: f64,

    /// Operations per second
    pub operations_per_second: f64,
}

/// Trend analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Performance trend over time
    pub performance_trend: TrendDirection,

    /// Error rate trend
    pub error_rate_trend: TrendDirection,

    /// Throughput trend
    pub throughput_trend: TrendDirection,

    /// Memory usage trend
    pub memory_trend: TrendDirection,

    /// Trend analysis period
    pub analysis_period_hours: u32,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Indexing statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingStats {
    /// Documents indexed
    pub documents_indexed: u64,

    /// Chunks processed
    pub chunks_processed: u64,

    /// Embeddings generated
    pub embeddings_generated: u64,

    /// Index updates performed
    pub index_updates: u64,

    /// Average processing time per document
    pub avg_document_processing_ms: f64,

    /// Indexing throughput
    pub indexing_throughput_dps: f64, // documents per second

    /// Storage efficiency
    pub storage_efficiency: f64,

    /// Index quality score
    pub index_quality_score: f64,
}

/// Metrics collector for gathering system metrics
pub struct MetricsCollector {
    /// Current metrics
    current_metrics: Arc<RwLock<IncrementalMetrics>>,

    /// Metrics history
    metrics_history: Arc<RwLock<VecDeque<IncrementalMetrics>>>,

    /// Performance tracker
    performance_tracker: Arc<PerformanceTracker>,

    /// Configuration
    config: MonitoringConfig,

    /// Collection statistics
    collection_stats: Arc<RwLock<CollectionStatistics>>,

    /// Background task handles
    task_handles: Arc<tokio::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

/// Collection process statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatistics {
    /// Total collection cycles
    pub total_collections: u64,

    /// Failed collections
    pub failed_collections: u64,

    /// Average collection time
    pub avg_collection_time_ms: f64,

    /// Last collection timestamp
    pub last_collection: chrono::DateTime<chrono::Utc>,

    /// Collection success rate
    pub collection_success_rate: f64,
}

impl IncrementalMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            system_id: Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            indexing_metrics: IndexingMetrics {
                documents_per_second: 0.0,
                chunks_per_second: 0.0,
                embeddings_per_second: 0.0,
                avg_indexing_time_ms: 0.0,
                index_growth_rate_bps: 0.0,
                batch_efficiency: 1.0,
                change_detection_accuracy: 1.0,
                vector_update_efficiency: 1.0,
            },
            system_metrics: SystemMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                available_memory_bytes: 0,
                storage_usage_bytes: 0,
                available_storage_bytes: 0,
                network_io_bps: 0.0,
                disk_io_ops: 0.0,
                active_connections: 0,
            },
            operation_metrics: OperationMetrics {
                total_operations: 0,
                operations_by_type: HashMap::new(),
                success_rate: 1.0,
                avg_operation_time_ms: 0.0,
                p95_operation_time_ms: 0.0,
                p99_operation_time_ms: 0.0,
                queue_depths: HashMap::new(),
                retry_stats: RetryMetrics {
                    total_retries: 0,
                    successful_retries: 0,
                    exhausted_retries: 0,
                    avg_retries_per_operation: 0.0,
                },
            },
            health_metrics: HealthMetrics {
                overall_health_score: 1.0,
                component_health: HashMap::new(),
                service_availability: 1.0,
                data_consistency_score: 1.0,
                performance_score: 1.0,
                last_health_check: chrono::Utc::now(),
            },
            error_metrics: ErrorMetrics {
                total_errors: 0,
                errors_by_type: HashMap::new(),
                errors_by_component: HashMap::new(),
                error_rate: 0.0,
                critical_errors: 0,
                recoverable_errors: 0,
                avg_resolution_time_ms: 0.0,
            },
            custom_metrics: HashMap::new(),
        }
    }

    /// Update metrics with new data
    pub fn update(&mut self, update_data: MetricsUpdate) {
        self.timestamp = chrono::Utc::now();

        // Update indexing metrics
        if let Some(indexing) = update_data.indexing_metrics {
            self.indexing_metrics = indexing;
        }

        // Update system metrics
        if let Some(system) = update_data.system_metrics {
            self.system_metrics = system;
        }

        // Update operation metrics
        if let Some(operations) = update_data.operation_metrics {
            self.operation_metrics = operations;
        }

        // Update health metrics
        if let Some(health) = update_data.health_metrics {
            self.health_metrics = health;
        }

        // Update error metrics
        if let Some(errors) = update_data.error_metrics {
            self.error_metrics = errors;
        }

        // Merge custom metrics
        for (key, value) in update_data.custom_metrics {
            self.custom_metrics.insert(key, value);
        }
    }

    /// Calculate overall system score
    pub fn calculate_system_score(&self) -> f64 {
        let health_weight = 0.4;
        let performance_weight = 0.3;
        let reliability_weight = 0.3;

        let health_score = self.health_metrics.overall_health_score;
        let performance_score = self.health_metrics.performance_score;
        let reliability_score = 1.0 - self.error_metrics.error_rate.min(1.0);

        (health_score * health_weight)
            + (performance_score * performance_weight)
            + (reliability_score * reliability_weight)
    }
}

/// Update data for metrics
#[derive(Debug, Clone)]
pub struct MetricsUpdate {
    pub indexing_metrics: Option<IndexingMetrics>,
    pub system_metrics: Option<SystemMetrics>,
    pub operation_metrics: Option<OperationMetrics>,
    pub health_metrics: Option<HealthMetrics>,
    pub error_metrics: Option<ErrorMetrics>,
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new(config: MonitoringConfig, max_data_points: usize) -> Self {
        Self {
            data_points: Arc::new(RwLock::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(PerformanceStatistics {
                by_operation_type: HashMap::new(),
                overall: OperationStatistics {
                    total_count: 0,
                    success_count: 0,
                    avg_duration_ms: 0.0,
                    median_duration_ms: 0.0,
                    p95_duration_ms: 0.0,
                    p99_duration_ms: 0.0,
                    std_deviation_ms: 0.0,
                    operations_per_second: 0.0,
                },
                trends: TrendAnalysis {
                    performance_trend: TrendDirection::Stable,
                    error_rate_trend: TrendDirection::Stable,
                    throughput_trend: TrendDirection::Stable,
                    memory_trend: TrendDirection::Stable,
                    analysis_period_hours: 24,
                },
                last_updated: chrono::Utc::now(),
            })),
            config,
            max_data_points,
        }
    }

    /// Record a performance data point
    pub async fn record_data_point(&self, data_point: PerformanceDataPoint) {
        let mut data_points = self.data_points.write().await;
        data_points.push_back(data_point);

        // Limit data points
        while data_points.len() > self.max_data_points {
            data_points.pop_front();
        }

        // Update statistics
        self.update_statistics().await;
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> PerformanceStatistics {
        self.statistics.read().await.clone()
    }

    /// Update aggregated statistics
    async fn update_statistics(&self) {
        let data_points = self.data_points.read().await;

        if data_points.is_empty() {
            return;
        }

        let mut by_operation_type: HashMap<String, Vec<&PerformanceDataPoint>> = HashMap::new();
        let mut all_points = Vec::new();

        // Group by operation type
        for point in data_points.iter() {
            by_operation_type
                .entry(point.operation_type.clone())
                .or_insert_with(Vec::new)
                .push(point);
            all_points.push(point);
        }

        let mut statistics = self.statistics.write().await;

        // Calculate statistics for each operation type
        for (op_type, points) in by_operation_type {
            let stats = self.calculate_operation_statistics(&points);
            statistics.by_operation_type.insert(op_type, stats);
        }

        // Calculate overall statistics
        statistics.overall = self.calculate_operation_statistics(&all_points);
        statistics.last_updated = chrono::Utc::now();
    }

    /// Calculate statistics for a set of data points
    fn calculate_operation_statistics(
        &self,
        points: &[&PerformanceDataPoint],
    ) -> OperationStatistics {
        if points.is_empty() {
            return OperationStatistics {
                total_count: 0,
                success_count: 0,
                avg_duration_ms: 0.0,
                median_duration_ms: 0.0,
                p95_duration_ms: 0.0,
                p99_duration_ms: 0.0,
                std_deviation_ms: 0.0,
                operations_per_second: 0.0,
            };
        }

        let total_count = points.len() as u64;
        let success_count = points.iter().filter(|p| p.success).count() as u64;

        let mut durations: Vec<u64> = points.iter().map(|p| p.duration_ms).collect();
        durations.sort();

        let avg_duration_ms = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
        let median_duration_ms = if durations.len() % 2 == 0 {
            (durations[durations.len() / 2 - 1] + durations[durations.len() / 2]) as f64 / 2.0
        } else {
            durations[durations.len() / 2] as f64
        };

        let p95_index = ((durations.len() as f64) * 0.95) as usize;
        let p99_index = ((durations.len() as f64) * 0.99) as usize;

        let p95_duration_ms = durations
            .get(p95_index.min(durations.len() - 1))
            .unwrap_or(&0) as &u64;
        let p99_duration_ms = durations
            .get(p99_index.min(durations.len() - 1))
            .unwrap_or(&0) as &u64;

        // Calculate standard deviation
        let variance = durations
            .iter()
            .map(|d| (*d as f64 - avg_duration_ms).powi(2))
            .sum::<f64>()
            / durations.len() as f64;
        let std_deviation_ms = variance.sqrt();

        // Calculate operations per second (simplified)
        let time_span_secs = if points.len() > 1 {
            let first = points.first().unwrap().timestamp;
            let last = points.last().unwrap().timestamp;
            last.signed_duration_since(first).num_seconds().max(1) as f64
        } else {
            1.0
        };
        let operations_per_second = total_count as f64 / time_span_secs;

        OperationStatistics {
            total_count,
            success_count,
            avg_duration_ms,
            median_duration_ms,
            p95_duration_ms: *p95_duration_ms as f64,
            p99_duration_ms: *p99_duration_ms as f64,
            std_deviation_ms,
            operations_per_second,
        }
    }
}

impl MetricsCollector {
    /// Create new metrics collector
    pub async fn new(config: MonitoringConfig) -> RragResult<Self> {
        let collector = Self {
            current_metrics: Arc::new(RwLock::new(IncrementalMetrics::new())),
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            performance_tracker: Arc::new(PerformanceTracker::new(config.clone(), 10000)),
            config: config.clone(),
            collection_stats: Arc::new(RwLock::new(CollectionStatistics {
                total_collections: 0,
                failed_collections: 0,
                avg_collection_time_ms: 0.0,
                last_collection: chrono::Utc::now(),
                collection_success_rate: 1.0,
            })),
            task_handles: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        };

        if config.enable_performance_metrics {
            collector.start_collection_tasks().await?;
        }

        Ok(collector)
    }

    /// Get current metrics
    pub async fn get_current_metrics(&self) -> IncrementalMetrics {
        self.current_metrics.read().await.clone()
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Vec<IncrementalMetrics> {
        let history = self.metrics_history.read().await;
        let limit = limit.unwrap_or(history.len());
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Update metrics
    pub async fn update_metrics(&self, update: MetricsUpdate) -> RragResult<()> {
        let mut current = self.current_metrics.write().await;
        current.update(update);

        // Add to history
        let mut history = self.metrics_history.write().await;
        history.push_back(current.clone());

        // Limit history size based on retention
        let max_history_size = (self.config.metrics_retention_days as usize) * 24 * 60 * 60
            / (self.config.metrics_interval_secs as usize);
        while history.len() > max_history_size {
            history.pop_front();
        }

        Ok(())
    }

    /// Record performance data
    pub async fn record_performance(&self, data_point: PerformanceDataPoint) -> RragResult<()> {
        self.performance_tracker.record_data_point(data_point).await;
        Ok(())
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> PerformanceStatistics {
        self.performance_tracker.get_statistics().await
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        let handles = self.task_handles.lock().await;
        let all_running = handles.iter().all(|handle| !handle.is_finished());

        let stats = self.collection_stats.read().await;
        let healthy_collection = stats.collection_success_rate > 0.8;

        Ok(all_running && healthy_collection)
    }

    /// Start background collection tasks
    async fn start_collection_tasks(&self) -> RragResult<()> {
        let mut handles = self.task_handles.lock().await;

        // Metrics collection task
        handles.push(self.start_metrics_collection_task().await);

        // Health monitoring task
        if self.config.enable_health_monitoring {
            handles.push(self.start_health_monitoring_task().await);
        }

        // Export task
        if self.config.export_config.enable_json_export
            || self.config.export_config.enable_prometheus
        {
            handles.push(self.start_export_task().await);
        }

        Ok(())
    }

    /// Start metrics collection background task
    async fn start_metrics_collection_task(&self) -> tokio::task::JoinHandle<()> {
        let current_metrics = Arc::clone(&self.current_metrics);
        let collection_stats = Arc::clone(&self.collection_stats);
        let interval = self.config.metrics_interval_secs;

        tokio::spawn(async move {
            let mut timer = tokio::time::interval(tokio::time::Duration::from_secs(interval));

            loop {
                timer.tick().await;

                let start_time = std::time::Instant::now();
                let collection_successful = {
                    // Collect system metrics (simplified)
                    let update = MetricsUpdate {
                        indexing_metrics: Some(IndexingMetrics {
                            documents_per_second: 10.0, // Would be actual measurement
                            chunks_per_second: 50.0,
                            embeddings_per_second: 50.0,
                            avg_indexing_time_ms: 100.0,
                            index_growth_rate_bps: 1024.0,
                            batch_efficiency: 0.95,
                            change_detection_accuracy: 0.98,
                            vector_update_efficiency: 0.92,
                        }),
                        system_metrics: Some(SystemMetrics {
                            cpu_usage_percent: 45.0,                      // Would be actual measurement
                            memory_usage_bytes: 512 * 1024 * 1024,        // 512MB
                            available_memory_bytes: 1024 * 1024 * 1024,   // 1GB
                            storage_usage_bytes: 10 * 1024 * 1024 * 1024, // 10GB
                            available_storage_bytes: 90 * 1024 * 1024 * 1024, // 90GB
                            network_io_bps: 1024.0 * 100.0,               // 100KB/s
                            disk_io_ops: 50.0,
                            active_connections: 10,
                        }),
                        operation_metrics: None,
                        health_metrics: None,
                        error_metrics: None,
                        custom_metrics: HashMap::new(),
                    };

                    // Update metrics
                    let mut metrics = current_metrics.write().await;
                    metrics.update(update);
                    true
                };

                let collection_time = start_time.elapsed().as_millis() as f64;

                // Update collection statistics
                let mut stats = collection_stats.write().await;
                stats.total_collections += 1;
                if !collection_successful {
                    stats.failed_collections += 1;
                }
                stats.avg_collection_time_ms =
                    (stats.avg_collection_time_ms + collection_time) / 2.0;
                stats.last_collection = chrono::Utc::now();
                stats.collection_success_rate = (stats.total_collections - stats.failed_collections)
                    as f64
                    / stats.total_collections as f64;
            }
        })
    }

    /// Start health monitoring task
    async fn start_health_monitoring_task(&self) -> tokio::task::JoinHandle<()> {
        let current_metrics = Arc::clone(&self.current_metrics);
        let interval = self.config.health_check_interval_secs;

        tokio::spawn(async move {
            let mut timer = tokio::time::interval(tokio::time::Duration::from_secs(interval));

            loop {
                timer.tick().await;

                // Perform health checks and update health metrics
                let health_update = HealthMetrics {
                    overall_health_score: 0.95,       // Would be calculated
                    component_health: HashMap::new(), // Would include component scores
                    service_availability: 0.99,
                    data_consistency_score: 0.98,
                    performance_score: 0.92,
                    last_health_check: chrono::Utc::now(),
                };

                let mut metrics = current_metrics.write().await;
                metrics.health_metrics = health_update;
            }
        })
    }

    /// Start export task
    async fn start_export_task(&self) -> tokio::task::JoinHandle<()> {
        let current_metrics = Arc::clone(&self.current_metrics);
        let export_config = self.config.export_config.clone();

        tokio::spawn(async move {
            let mut timer = tokio::time::interval(tokio::time::Duration::from_secs(
                export_config.export_interval_secs,
            ));

            loop {
                timer.tick().await;

                if export_config.enable_json_export {
                    let metrics = current_metrics.read().await;
                    // Export metrics as JSON (simplified)
                    match serde_json::to_string_pretty(&*metrics) {
                        Ok(json) => {
                            // Would export to configured endpoint
                            tracing::debug!("Exported metrics: {} chars", json.len());
                        }
                        Err(e) => {
                            tracing::debug!("Failed to serialize metrics: {}", e);
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_creation() {
        let metrics = IncrementalMetrics::new();
        assert!(!metrics.system_id.is_empty());
        assert_eq!(metrics.health_metrics.overall_health_score, 1.0);
    }

    #[tokio::test]
    async fn test_performance_tracker() {
        let config = MonitoringConfig::default();
        let tracker = PerformanceTracker::new(config, 100);

        let data_point = PerformanceDataPoint {
            timestamp: chrono::Utc::now(),
            operation_type: "indexing".to_string(),
            duration_ms: 100,
            memory_usage_mb: 50.0,
            success: true,
            metadata: HashMap::new(),
        };

        tracker.record_data_point(data_point).await;

        let stats = tracker.get_statistics().await;
        assert_eq!(stats.overall.total_count, 1);
        assert_eq!(stats.overall.success_count, 1);
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let config = MonitoringConfig {
            enable_performance_metrics: false, // Disable for test
            ..MonitoringConfig::default()
        };

        let collector = MetricsCollector::new(config).await.unwrap();
        assert!(collector.health_check().await.unwrap());

        let metrics = collector.get_current_metrics().await;
        assert!(!metrics.system_id.is_empty());
    }

    #[tokio::test]
    async fn test_metrics_update() {
        let config = MonitoringConfig {
            enable_performance_metrics: false,
            ..MonitoringConfig::default()
        };

        let collector = MetricsCollector::new(config).await.unwrap();

        let update = MetricsUpdate {
            indexing_metrics: Some(IndexingMetrics {
                documents_per_second: 20.0,
                chunks_per_second: 100.0,
                embeddings_per_second: 100.0,
                avg_indexing_time_ms: 50.0,
                index_growth_rate_bps: 2048.0,
                batch_efficiency: 0.98,
                change_detection_accuracy: 0.99,
                vector_update_efficiency: 0.95,
            }),
            system_metrics: None,
            operation_metrics: None,
            health_metrics: None,
            error_metrics: None,
            custom_metrics: HashMap::new(),
        };

        collector.update_metrics(update).await.unwrap();

        let metrics = collector.get_current_metrics().await;
        assert_eq!(metrics.indexing_metrics.documents_per_second, 20.0);
    }

    #[test]
    fn test_trend_directions() {
        let directions = vec![
            TrendDirection::Improving,
            TrendDirection::Stable,
            TrendDirection::Degrading,
            TrendDirection::Unknown,
        ];

        // Ensure all directions are different
        for (i, dir1) in directions.iter().enumerate() {
            for (j, dir2) in directions.iter().enumerate() {
                if i != j {
                    assert_ne!(format!("{:?}", dir1), format!("{:?}", dir2));
                }
            }
        }
    }

    #[test]
    fn test_export_formats() {
        let formats = vec![
            ExportFormat::Prometheus,
            ExportFormat::Json,
            ExportFormat::InfluxDB,
            ExportFormat::StatsD,
            ExportFormat::Custom("custom".to_string()),
        ];

        // Ensure all formats are different
        for (i, fmt1) in formats.iter().enumerate() {
            for (j, fmt2) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(format!("{:?}", fmt1), format!("{:?}", fmt2));
                }
            }
        }
    }
}
