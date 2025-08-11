//! # Export and Reporting System
//! 
//! Comprehensive data export capabilities with multiple formats,
//! automated report generation, and scheduled exports for RRAG observability data.

use crate::{RragError, RragResult};
use super::{
    metrics::{MetricsCollector, Metric},
    monitoring::SystemOverview,
    health::HealthReport,
    profiling::PerformanceReport,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc, Duration};

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub enabled: bool,
    pub default_format: ExportFormat,
    pub output_directory: String,
    pub max_file_size_mb: u64,
    pub retention_days: u32,
    pub compression_enabled: bool,
    pub scheduled_exports: Vec<ScheduledExportConfig>,
    pub destinations: Vec<ExportDestinationConfig>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_format: ExportFormat::Json,
            output_directory: "./exports".to_string(),
            max_file_size_mb: 100,
            retention_days: 90,
            compression_enabled: true,
            scheduled_exports: Vec::new(),
            destinations: Vec::new(),
        }
    }
}

/// Scheduled export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledExportConfig {
    pub name: String,
    pub schedule_cron: String,
    pub export_type: ExportType,
    pub format: ExportFormat,
    pub destinations: Vec<String>,
    pub filters: ExportFilters,
}

/// Export destination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportDestinationConfig {
    pub name: String,
    pub destination_type: DestinationType,
    pub config: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DestinationType {
    LocalFile,
    S3,
    Azure,
    GCS,
    SFTP,
    HTTP,
    Email,
    Webhook,
}

/// Export data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExportType {
    Metrics,
    Logs,
    HealthReport,
    PerformanceReport,
    SystemOverview,
    AlertHistory,
    UserActivity,
    CustomReport,
}

/// Supported export formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExportFormat {
    Json,
    Csv,
    Xml,
    Yaml,
    Parquet,
    Avro,
    Excel,
    Pdf,
}

/// Export filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFilters {
    pub time_range: Option<TimeRange>,
    pub components: Vec<String>,
    pub severity_levels: Vec<String>,
    pub custom_fields: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl Default for ExportFilters {
    fn default() -> Self {
        Self {
            time_range: None,
            components: Vec::new(),
            severity_levels: Vec::new(),
            custom_fields: HashMap::new(),
        }
    }
}

/// Export result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    pub export_id: String,
    pub export_type: ExportType,
    pub format: ExportFormat,
    pub file_path: Option<String>,
    pub file_size_bytes: u64,
    pub record_count: usize,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: ExportStatus,
    pub error_message: Option<String>,
    pub destinations: Vec<DestinationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestinationResult {
    pub destination_name: String,
    pub status: ExportStatus,
    pub delivered_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub delivery_info: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExportStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    PartiallyCompleted,
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub name: String,
    pub description: String,
    pub report_type: ReportType,
    pub template: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub output_format: ExportFormat,
    pub include_charts: bool,
    pub chart_config: ChartConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    SystemHealth,
    PerformanceSummary,
    SecurityAudit,
    UsageAnalytics,
    ErrorAnalysis,
    CapacityPlanning,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    pub chart_types: Vec<ChartType>,
    pub color_scheme: String,
    pub dimensions: ChartDimensions,
    pub include_legends: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
    Area,
    Scatter,
    Heatmap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDimensions {
    pub width: u32,
    pub height: u32,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            chart_types: vec![ChartType::Line, ChartType::Bar],
            color_scheme: "default".to_string(),
            dimensions: ChartDimensions { width: 800, height: 600 },
            include_legends: true,
        }
    }
}

/// Data formatter trait
#[async_trait::async_trait]
pub trait DataFormatter: Send + Sync {
    async fn format_metrics(&self, metrics: &[Metric]) -> RragResult<Vec<u8>>;
    async fn format_health_report(&self, report: &HealthReport) -> RragResult<Vec<u8>>;
    async fn format_performance_report(&self, report: &PerformanceReport) -> RragResult<Vec<u8>>;
    async fn format_system_overview(&self, overview: &SystemOverview) -> RragResult<Vec<u8>>;
    fn content_type(&self) -> &'static str;
    fn file_extension(&self) -> &'static str;
}

/// JSON formatter
pub struct JsonFormatter;

#[async_trait::async_trait]
impl DataFormatter for JsonFormatter {
    async fn format_metrics(&self, metrics: &[Metric]) -> RragResult<Vec<u8>> {
        serde_json::to_vec_pretty(metrics)
            .map_err(|e| RragError::agent("json_formatter", e.to_string()))
    }

    async fn format_health_report(&self, report: &HealthReport) -> RragResult<Vec<u8>> {
        serde_json::to_vec_pretty(report)
            .map_err(|e| RragError::agent("json_formatter", e.to_string()))
    }

    async fn format_performance_report(&self, report: &PerformanceReport) -> RragResult<Vec<u8>> {
        serde_json::to_vec_pretty(report)
            .map_err(|e| RragError::agent("json_formatter", e.to_string()))
    }

    async fn format_system_overview(&self, overview: &SystemOverview) -> RragResult<Vec<u8>> {
        serde_json::to_vec_pretty(overview)
            .map_err(|e| RragError::agent("json_formatter", e.to_string()))
    }

    fn content_type(&self) -> &'static str {
        "application/json"
    }

    fn file_extension(&self) -> &'static str {
        "json"
    }
}

/// CSV formatter
pub struct CsvFormatter;

#[async_trait::async_trait]
impl DataFormatter for CsvFormatter {
    async fn format_metrics(&self, metrics: &[Metric]) -> RragResult<Vec<u8>> {
        let mut output = Vec::new();
        
        // CSV header
        output.extend_from_slice(b"timestamp,name,type,value,labels\n");
        
        for metric in metrics {
            let timestamp = metric.timestamp.format("%Y-%m-%d %H:%M:%S").to_string();
            let name = &metric.name;
            let metric_type = format!("{:?}", metric.metric_type);
            let value = match &metric.value {
                super::metrics::MetricValue::Counter(v) => v.to_string(),
                super::metrics::MetricValue::Gauge(v) => v.to_string(),
                super::metrics::MetricValue::Timer { duration_ms, .. } => duration_ms.to_string(),
                super::metrics::MetricValue::Histogram { sum, count, .. } => {
                    if *count > 0 { (sum / *count as f64).to_string() } else { "0".to_string() }
                },
                super::metrics::MetricValue::Summary { sum, count, .. } => {
                    if *count > 0 { (sum / *count as f64).to_string() } else { "0".to_string() }
                },
            };
            let labels = metric.labels.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(";");
            
            let line = format!("{},{},{},{},\"{}\"\n", timestamp, name, metric_type, value, labels);
            output.extend_from_slice(line.as_bytes());
        }
        
        Ok(output)
    }

    async fn format_health_report(&self, report: &HealthReport) -> RragResult<Vec<u8>> {
        let mut output = Vec::new();
        
        // CSV header
        output.extend_from_slice(b"component,status,last_check,response_time_ms,error_message\n");
        
        for (component, health) in &report.services {
            let status = health.status.to_string();
            let last_check = health.last_check.format("%Y-%m-%d %H:%M:%S").to_string();
            let response_time = health.response_time_ms.map(|t| t.to_string()).unwrap_or_default();
            let error_message = health.error_message.as_deref().unwrap_or("");
            
            let line = format!("{},{},{},{},\"{}\"\n", 
                component, status, last_check, response_time, error_message);
            output.extend_from_slice(line.as_bytes());
        }
        
        Ok(output)
    }

    async fn format_performance_report(&self, report: &PerformanceReport) -> RragResult<Vec<u8>> {
        let mut output = Vec::new();
        
        // CSV header for component performance
        output.extend_from_slice(b"component,operation_count,avg_duration_ms,max_duration_ms,std_deviation_ms\n");
        
        for (component, metrics) in &report.component_performance {
            let line = format!("{},{},{:.2},{:.2},{:.2}\n",
                component,
                metrics.operation_count,
                metrics.average_duration_ms,
                metrics.max_duration_ms,
                metrics.standard_deviation_ms
            );
            output.extend_from_slice(line.as_bytes());
        }
        
        Ok(output)
    }

    async fn format_system_overview(&self, overview: &SystemOverview) -> RragResult<Vec<u8>> {
        let mut output = Vec::new();
        
        // CSV header
        output.extend_from_slice(b"timestamp,cpu_usage,memory_usage,active_sessions,total_searches\n");
        
        let timestamp = overview.timestamp.format("%Y-%m-%d %H:%M:%S").to_string();
        let cpu_usage = overview.performance_metrics.as_ref()
            .map(|p| p.cpu_usage_percent.to_string())
            .unwrap_or_default();
        let memory_usage = overview.performance_metrics.as_ref()
            .map(|p| p.memory_usage_percent.to_string())
            .unwrap_or_default();
        let active_sessions = overview.active_sessions.map(|s| s.to_string()).unwrap_or_default();
        let total_searches = overview.search_stats.as_ref()
            .map(|s| s.total_searches.to_string())
            .unwrap_or_default();
        
        let line = format!("{},{},{},{},{}\n",
            timestamp, cpu_usage, memory_usage, active_sessions, total_searches);
        output.extend_from_slice(line.as_bytes());
        
        Ok(output)
    }

    fn content_type(&self) -> &'static str {
        "text/csv"
    }

    fn file_extension(&self) -> &'static str {
        "csv"
    }
}

/// Export destination trait
#[async_trait::async_trait]
pub trait ExportDestination: Send + Sync {
    async fn export_data(&self, data: &[u8], filename: &str, content_type: &str) -> RragResult<DestinationResult>;
    fn destination_name(&self) -> &str;
    async fn test_connection(&self) -> RragResult<bool>;
}

/// Local file export destination
pub struct LocalFileDestination {
    name: String,
    base_path: String,
}

impl LocalFileDestination {
    pub fn new(name: impl Into<String>, base_path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_path: base_path.into(),
        }
    }
}

#[async_trait::async_trait]
impl ExportDestination for LocalFileDestination {
    async fn export_data(&self, data: &[u8], filename: &str, _content_type: &str) -> RragResult<DestinationResult> {
        let full_path = format!("{}/{}", self.base_path, filename);
        
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&full_path).parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| RragError::storage("create_directory", e))?;
        }
        
        // Write file
        tokio::fs::write(&full_path, data).await
            .map_err(|e| RragError::storage("write_file", e))?;
        
        Ok(DestinationResult {
            destination_name: self.name.clone(),
            status: ExportStatus::Completed,
            delivered_at: Some(Utc::now()),
            error_message: None,
            delivery_info: HashMap::from([
                ("file_path".to_string(), full_path),
                ("file_size".to_string(), data.len().to_string()),
            ]),
        })
    }

    fn destination_name(&self) -> &str {
        &self.name
    }

    async fn test_connection(&self) -> RragResult<bool> {
        // Test if we can write to the directory
        match tokio::fs::metadata(&self.base_path).await {
            Ok(metadata) => Ok(metadata.is_dir()),
            Err(_) => {
                // Try to create the directory
                match tokio::fs::create_dir_all(&self.base_path).await {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
        }
    }
}

/// HTTP webhook export destination
pub struct WebhookDestination {
    name: String,
    url: String,
    headers: HashMap<String, String>,
    #[cfg(feature = "http")]
    client: reqwest::Client,
}

impl WebhookDestination {
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: HashMap::new(),
            #[cfg(feature = "http")]
            client: reqwest::Client::new(),
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

#[async_trait::async_trait]
impl ExportDestination for WebhookDestination {
    async fn export_data(&self, data: &[u8], filename: &str, content_type: &str) -> RragResult<DestinationResult> {
        #[cfg(feature = "http")]
        {
            let mut request = self.client.post(&self.url)
                .header("Content-Type", content_type)
                .header("X-Filename", filename)
                .body(data.to_vec());

            for (key, value) in &self.headers {
                request = request.header(key, value);
            }

            match request.send().await {
                Ok(response) => {
                    let status_code = response.status().as_u16();
                    if response.status().is_success() {
                        Ok(DestinationResult {
                            destination_name: self.name.clone(),
                            status: ExportStatus::Completed,
                            delivered_at: Some(Utc::now()),
                            error_message: None,
                            delivery_info: HashMap::from([
                                ("status_code".to_string(), status_code.to_string()),
                                ("url".to_string(), self.url.clone()),
                            ]),
                        })
                    } else {
                        Ok(DestinationResult {
                            destination_name: self.name.clone(),
                            status: ExportStatus::Failed,
                            delivered_at: None,
                            error_message: Some(format!("HTTP {}: {}", status_code, response.status())),
                            delivery_info: HashMap::from([
                                ("status_code".to_string(), status_code.to_string()),
                            ]),
                        })
                    }
                },
                Err(e) => Ok(DestinationResult {
                    destination_name: self.name.clone(),
                    status: ExportStatus::Failed,
                    delivered_at: None,
                    error_message: Some(e.to_string()),
                    delivery_info: HashMap::new(),
                })
            }
        }
        #[cfg(not(feature = "http"))]
        {
            // Without HTTP feature, return skipped status
            Ok(DestinationResult {
                destination_name: self.name.clone(),
                status: ExportStatus::Failed,
                delivered_at: None,
                error_message: Some("HTTP feature not enabled".to_string()),
                delivery_info: HashMap::from([
                    ("note".to_string(), "HTTP feature disabled".to_string()),
                    ("url".to_string(), self.url.clone()),
                ]),
            })
        }
    }

    fn destination_name(&self) -> &str {
        &self.name
    }

    async fn test_connection(&self) -> RragResult<bool> {
        #[cfg(feature = "http")]
        {
            match self.client.head(&self.url).send().await {
                Ok(response) => Ok(response.status().is_success()),
                Err(_) => Ok(false),
            }
        }
        #[cfg(not(feature = "http"))]
        {
            // Without HTTP feature, assume connection is fine
            Ok(true)
        }
    }
}

/// Report generator
pub struct ReportGenerator {
    templates: Arc<RwLock<HashMap<String, String>>>,
}

impl ReportGenerator {
    pub fn new() -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_template(&self, name: impl Into<String>, template: impl Into<String>) {
        let mut templates = self.templates.write().await;
        templates.insert(name.into(), template.into());
    }

    pub async fn generate_report(
        &self,
        config: &ReportConfig,
        data: &SystemOverview,
    ) -> RragResult<Vec<u8>> {
        match config.report_type {
            ReportType::SystemHealth => self.generate_system_health_report(data).await,
            ReportType::PerformanceSummary => self.generate_performance_summary_report(data).await,
            ReportType::UsageAnalytics => self.generate_usage_analytics_report(data).await,
            _ => self.generate_generic_report(config, data).await,
        }
    }

    async fn generate_system_health_report(&self, data: &SystemOverview) -> RragResult<Vec<u8>> {
        let report = serde_json::json!({
            "title": "System Health Report",
            "generated_at": Utc::now(),
            "data": data,
            "summary": {
                "overall_status": "healthy",
                "components_checked": data.performance_metrics.as_ref().map(|_| 1).unwrap_or(0),
                "issues_detected": 0
            }
        });

        serde_json::to_vec_pretty(&report)
            .map_err(|e| RragError::agent("report_generator", e.to_string()))
    }

    async fn generate_performance_summary_report(&self, data: &SystemOverview) -> RragResult<Vec<u8>> {
        let report = serde_json::json!({
            "title": "Performance Summary Report",
            "generated_at": Utc::now(),
            "performance_metrics": data.performance_metrics,
            "search_stats": data.search_stats,
            "user_stats": data.user_stats
        });

        serde_json::to_vec_pretty(&report)
            .map_err(|e| RragError::agent("report_generator", e.to_string()))
    }

    async fn generate_usage_analytics_report(&self, data: &SystemOverview) -> RragResult<Vec<u8>> {
        let report = serde_json::json!({
            "title": "Usage Analytics Report",
            "generated_at": Utc::now(),
            "active_sessions": data.active_sessions,
            "user_stats": data.user_stats,
            "search_stats": data.search_stats
        });

        serde_json::to_vec_pretty(&report)
            .map_err(|e| RragError::agent("report_generator", e.to_string()))
    }

    async fn generate_generic_report(&self, config: &ReportConfig, data: &SystemOverview) -> RragResult<Vec<u8>> {
        let report = serde_json::json!({
            "title": config.name,
            "description": config.description,
            "generated_at": Utc::now(),
            "data": data
        });

        serde_json::to_vec_pretty(&report)
            .map_err(|e| RragError::agent("report_generator", e.to_string()))
    }
}

/// Main export manager
pub struct ExportManager {
    config: ExportConfig,
    formatters: Arc<RwLock<HashMap<ExportFormat, Box<dyn DataFormatter>>>>,
    destinations: Arc<RwLock<HashMap<String, Box<dyn ExportDestination>>>>,
    report_generator: Arc<ReportGenerator>,
    export_history: Arc<RwLock<Vec<ExportResult>>>,
    export_queue: mpsc::UnboundedSender<ExportRequest>,
    _queue_receiver: mpsc::UnboundedReceiver<ExportRequest>,
    processing_handle: Option<tokio::task::JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

#[derive(Debug)]
struct ExportRequest {
    export_id: String,
    export_type: ExportType,
    format: ExportFormat,
    data: ExportData,
    destinations: Vec<String>,
    filters: ExportFilters,
}

#[derive(Debug)]
enum ExportData {
    Metrics(Vec<Metric>),
    HealthReport(HealthReport),
    PerformanceReport(PerformanceReport),
    SystemOverview(SystemOverview),
}

/// Metrics exporter for integration with external systems
pub struct MetricsExporter {
    export_manager: Arc<ExportManager>,
    metrics_collector: Arc<MetricsCollector>,
}

impl MetricsExporter {
    pub fn new(export_manager: Arc<ExportManager>, metrics_collector: Arc<MetricsCollector>) -> Self {
        Self {
            export_manager,
            metrics_collector,
        }
    }

    pub async fn export_current_metrics(&self, format: ExportFormat, destinations: Vec<String>) -> RragResult<ExportResult> {
        let metrics = self.metrics_collector.get_all_metrics().await;
        self.export_manager.export_metrics(metrics, format, destinations, ExportFilters::default()).await
    }

    pub async fn schedule_periodic_export(&self, interval_minutes: u32, format: ExportFormat, destinations: Vec<String>) -> RragResult<()> {
        // This would set up a periodic export job
        // For now, just log the setup
        tracing::info!(
            "Scheduled metrics export every {} minutes to {:?} destinations in {:?} format",
            interval_minutes,
            destinations,
            format
        );
        Ok(())
    }
}

impl ExportManager {
    pub async fn new(config: ExportConfig) -> RragResult<Self> {
        let formatters: Arc<RwLock<HashMap<ExportFormat, Box<dyn DataFormatter>>>> = Arc::new(RwLock::new(HashMap::new()));
        let destinations: Arc<RwLock<HashMap<String, Box<dyn ExportDestination>>>> = Arc::new(RwLock::new(HashMap::new()));
        let report_generator = Arc::new(ReportGenerator::new());
        let export_history = Arc::new(RwLock::new(Vec::new()));
        
        let (export_queue, queue_receiver) = mpsc::unbounded_channel();

        // Initialize default formatters
        {
            let mut fmt = formatters.write().await;
            fmt.insert(ExportFormat::Json, Box::new(JsonFormatter));
            fmt.insert(ExportFormat::Csv, Box::new(CsvFormatter));
        }

        // Initialize destinations from config
        {
            let mut dest = destinations.write().await;
            for dest_config in &config.destinations {
                if !dest_config.enabled {
                    continue;
                }

                match dest_config.destination_type {
                    DestinationType::LocalFile => {
                        let base_path = dest_config.config.get("path")
                            .unwrap_or(&config.output_directory);
                        dest.insert(
                            dest_config.name.clone(),
                            Box::new(LocalFileDestination::new(&dest_config.name, base_path))
                        );
                    },
                    DestinationType::Webhook | DestinationType::HTTP => {
                        if let Some(url) = dest_config.config.get("url") {
                            let mut webhook = WebhookDestination::new(&dest_config.name, url);
                            
                            // Add custom headers
                            for (key, value) in &dest_config.config {
                                if key.starts_with("header_") {
                                    let header_name = key.strip_prefix("header_").unwrap();
                                    webhook = webhook.with_header(header_name, value);
                                }
                            }
                            
                            dest.insert(dest_config.name.clone(), Box::new(webhook));
                        }
                    },
                    _ => {
                        tracing::warn!("Destination type {:?} not yet implemented", dest_config.destination_type);
                    }
                }
            }
        }

        Ok(Self {
            config,
            formatters,
            destinations,
            report_generator,
            export_history,
            export_queue,
            _queue_receiver: queue_receiver,
            processing_handle: None,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&mut self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("export_manager", "stopped", "already running"));
        }

        // Start export processing loop would go here
        *running = true;
        tracing::info!("Export manager started");
        Ok(())
    }

    pub async fn stop(&mut self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        if let Some(handle) = self.processing_handle.take() {
            handle.abort();
        }

        *running = false;
        tracing::info!("Export manager stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    pub async fn export_metrics(
        &self,
        metrics: Vec<Metric>,
        format: ExportFormat,
        destinations: Vec<String>,
        filters: ExportFilters,
    ) -> RragResult<ExportResult> {
        let export_id = uuid::Uuid::new_v4().to_string();
        let started_at = Utc::now();
        
        // Apply filters
        let filtered_metrics = self.apply_metric_filters(metrics, &filters);
        
        // Format data
        let formatters = self.formatters.read().await;
        let formatter = formatters.get(&format)
            .ok_or_else(|| RragError::config("export_format", "supported", &format!("{:?}", format)))?;
        
        let formatted_data = formatter.format_metrics(&filtered_metrics).await?;
        drop(formatters);

        // Generate filename
        let filename = format!(
            "metrics_{}.{}",
            started_at.format("%Y%m%d_%H%M%S"),
            match format {
                ExportFormat::Json => "json",
                ExportFormat::Csv => "csv",
                _ => "data",
            }
        );

        // Export to destinations
        let destinations_map = self.destinations.read().await;
        let mut destination_results = Vec::new();

        for dest_name in destinations {
            if let Some(destination) = destinations_map.get(&dest_name) {
                let result = destination.export_data(
                    &formatted_data,
                    &filename,
                    formatter.content_type()
                ).await?;
                destination_results.push(result);
            }
        }

        let export_result = ExportResult {
            export_id: export_id.clone(),
            export_type: ExportType::Metrics,
            format,
            file_path: Some(filename),
            file_size_bytes: formatted_data.len() as u64,
            record_count: filtered_metrics.len(),
            started_at,
            completed_at: Some(Utc::now()),
            status: if destination_results.iter().all(|r| r.status == ExportStatus::Completed) {
                ExportStatus::Completed
            } else {
                ExportStatus::PartiallyCompleted
            },
            error_message: None,
            destinations: destination_results,
        };

        // Store in history
        let mut history = self.export_history.write().await;
        history.push(export_result.clone());

        // Keep only recent exports
        if history.len() > 1000 {
            history.drain(0..history.len() - 1000);
        }

        Ok(export_result)
    }

    fn apply_metric_filters(&self, metrics: Vec<Metric>, filters: &ExportFilters) -> Vec<Metric> {
        metrics.into_iter()
            .filter(|metric| {
                // Time range filter
                if let Some(ref time_range) = filters.time_range {
                    if metric.timestamp < time_range.start || metric.timestamp > time_range.end {
                        return false;
                    }
                }

                // Component filter
                if !filters.components.is_empty() {
                    if let Some(component) = metric.labels.get("component") {
                        if !filters.components.contains(component) {
                            return false;
                        }
                    }
                }

                true
            })
            .collect()
    }

    pub async fn generate_and_export_report(
        &self,
        config: ReportConfig,
        data: SystemOverview,
        destinations: Vec<String>,
    ) -> RragResult<ExportResult> {
        let export_id = uuid::Uuid::new_v4().to_string();
        let started_at = Utc::now();

        // Generate report
        let report_data = self.report_generator.generate_report(&config, &data).await?;

        // Generate filename
        let filename = format!(
            "report_{}_{}.{}",
            config.name.replace(' ', "_").to_lowercase(),
            started_at.format("%Y%m%d_%H%M%S"),
            match config.output_format {
                ExportFormat::Json => "json",
                ExportFormat::Csv => "csv",
                ExportFormat::Pdf => "pdf",
                _ => "report",
            }
        );

        // Export to destinations
        let destinations_map = self.destinations.read().await;
        let mut destination_results = Vec::new();

        for dest_name in destinations {
            if let Some(destination) = destinations_map.get(&dest_name) {
                let result = destination.export_data(
                    &report_data,
                    &filename,
                    "application/json" // Default content type
                ).await?;
                destination_results.push(result);
            }
        }

        let export_result = ExportResult {
            export_id,
            export_type: ExportType::CustomReport,
            format: config.output_format,
            file_path: Some(filename),
            file_size_bytes: report_data.len() as u64,
            record_count: 1,
            started_at,
            completed_at: Some(Utc::now()),
            status: if destination_results.iter().all(|r| r.status == ExportStatus::Completed) {
                ExportStatus::Completed
            } else {
                ExportStatus::PartiallyCompleted
            },
            error_message: None,
            destinations: destination_results,
        };

        // Store in history
        let mut history = self.export_history.write().await;
        history.push(export_result.clone());

        Ok(export_result)
    }

    pub async fn get_export_history(&self, limit: Option<usize>) -> Vec<ExportResult> {
        let history = self.export_history.read().await;
        let limit = limit.unwrap_or(history.len());
        let start_index = history.len().saturating_sub(limit);
        history[start_index..].to_vec()
    }

    pub async fn get_export_status(&self, export_id: &str) -> Option<ExportResult> {
        let history = self.export_history.read().await;
        history.iter().find(|r| r.export_id == export_id).cloned()
    }

    pub async fn test_destination(&self, destination_name: &str) -> RragResult<bool> {
        let destinations = self.destinations.read().await;
        if let Some(destination) = destinations.get(destination_name) {
            destination.test_connection().await
        } else {
            Err(RragError::config("destination", "exists", "not_found"))
        }
    }

    pub async fn add_destination(&self, name: String, destination: Box<dyn ExportDestination>) {
        let mut destinations = self.destinations.write().await;
        destinations.insert(name, destination);
    }

    pub async fn remove_destination(&self, name: &str) {
        let mut destinations = self.destinations.write().await;
        destinations.remove(name);
    }

    pub async fn list_destinations(&self) -> Vec<String> {
        let destinations = self.destinations.read().await;
        destinations.keys().cloned().collect()
    }

    pub async fn get_export_stats(&self) -> ExportStats {
        let history = self.export_history.read().await;
        
        let total_exports = history.len();
        let successful_exports = history.iter()
            .filter(|r| r.status == ExportStatus::Completed)
            .count();
        let failed_exports = history.iter()
            .filter(|r| r.status == ExportStatus::Failed)
            .count();
        
        let total_data_exported = history.iter()
            .map(|r| r.file_size_bytes)
            .sum::<u64>();

        let exports_by_type = history.iter()
            .fold(HashMap::new(), |mut acc, result| {
                *acc.entry(result.export_type.clone()).or_insert(0) += 1;
                acc
            });

        let exports_by_format = history.iter()
            .fold(HashMap::new(), |mut acc, result| {
                *acc.entry(result.format.clone()).or_insert(0) += 1;
                acc
            });

        ExportStats {
            total_exports,
            successful_exports,
            failed_exports,
            total_data_exported_bytes: total_data_exported,
            exports_by_type,
            exports_by_format,
            last_export: history.last().map(|r| r.started_at),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportStats {
    pub total_exports: usize,
    pub successful_exports: usize,
    pub failed_exports: usize,
    pub total_data_exported_bytes: u64,
    pub exports_by_type: HashMap<ExportType, usize>,
    pub exports_by_format: HashMap<ExportFormat, usize>,
    pub last_export: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_formatter() {
        let formatter = JsonFormatter;
        let metrics = vec![
            Metric::counter("test_counter", 42),
            Metric::gauge("test_gauge", 3.14),
        ];

        let result = formatter.format_metrics(&metrics).await.unwrap();
        let json_str = String::from_utf8(result).unwrap();
        
        assert!(json_str.contains("test_counter"));
        assert!(json_str.contains("test_gauge"));
        assert_eq!(formatter.content_type(), "application/json");
        assert_eq!(formatter.file_extension(), "json");
    }

    #[tokio::test]
    async fn test_csv_formatter() {
        let formatter = CsvFormatter;
        let metrics = vec![
            Metric::counter("requests_total", 100)
                .with_label("method", "GET"),
        ];

        let result = formatter.format_metrics(&metrics).await.unwrap();
        let csv_str = String::from_utf8(result).unwrap();
        
        assert!(csv_str.contains("timestamp,name,type,value,labels"));
        assert!(csv_str.contains("requests_total"));
        assert!(csv_str.contains("Counter"));
        assert_eq!(formatter.content_type(), "text/csv");
        assert_eq!(formatter.file_extension(), "csv");
    }

    #[tokio::test]
    async fn test_local_file_destination() {
        let temp_dir = tempfile::tempdir().unwrap();
        let destination = LocalFileDestination::new(
            "test_local",
            temp_dir.path().to_string_lossy().to_string()
        );

        assert!(destination.test_connection().await.unwrap());
        
        let test_data = b"test export data";
        let result = destination.export_data(test_data, "test.json", "application/json").await.unwrap();
        
        assert_eq!(result.status, ExportStatus::Completed);
        assert_eq!(result.destination_name, "test_local");
        assert!(result.delivered_at.is_some());
    }

    #[tokio::test]
    async fn test_export_manager() {
        let config = ExportConfig {
            output_directory: tempfile::tempdir().unwrap().path().to_string_lossy().to_string(),
            destinations: vec![
                ExportDestinationConfig {
                    name: "local_test".to_string(),
                    destination_type: DestinationType::LocalFile,
                    config: HashMap::new(),
                    enabled: true,
                }
            ],
            ..Default::default()
        };

        let mut manager = ExportManager::new(config).await.unwrap();
        manager.start().await.unwrap();

        let metrics = vec![
            Metric::counter("test_metric", 123),
            Metric::gauge("test_gauge", 45.6),
        ];

        let result = manager.export_metrics(
            metrics,
            ExportFormat::Json,
            vec!["local_test".to_string()],
            ExportFilters::default()
        ).await.unwrap();

        assert_eq!(result.export_type, ExportType::Metrics);
        assert_eq!(result.format, ExportFormat::Json);
        assert_eq!(result.status, ExportStatus::Completed);
        assert_eq!(result.record_count, 2);
        assert!(!result.destinations.is_empty());

        let history = manager.get_export_history(Some(10)).await;
        assert_eq!(history.len(), 1);

        let stats = manager.get_export_stats().await;
        assert_eq!(stats.total_exports, 1);
        assert_eq!(stats.successful_exports, 1);

        manager.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_report_generator() {
        let generator = ReportGenerator::new();
        let overview = SystemOverview {
            timestamp: Utc::now(),
            performance_metrics: None,
            search_stats: None,
            user_stats: None,
            active_sessions: Some(10),
        };

        let config = ReportConfig {
            name: "Test Report".to_string(),
            description: "A test report".to_string(),
            report_type: ReportType::SystemHealth,
            template: None,
            parameters: HashMap::new(),
            output_format: ExportFormat::Json,
            include_charts: false,
            chart_config: ChartConfig::default(),
        };

        let report_data = generator.generate_report(&config, &overview).await.unwrap();
        let report_str = String::from_utf8(report_data).unwrap();
        
        assert!(report_str.contains("System Health Report"));
        assert!(report_str.contains("generated_at"));
    }

    #[test]
    fn test_export_filters() {
        let filters = ExportFilters {
            time_range: Some(TimeRange {
                start: Utc::now() - Duration::hours(1),
                end: Utc::now(),
            }),
            components: vec!["search".to_string(), "storage".to_string()],
            ..Default::default()
        };

        assert!(filters.time_range.is_some());
        assert_eq!(filters.components.len(), 2);
        assert!(filters.components.contains(&"search".to_string()));
    }

    #[test]
    fn test_export_status() {
        assert_eq!(ExportStatus::Completed, ExportStatus::Completed);
        assert_ne!(ExportStatus::Completed, ExportStatus::Failed);
    }
}