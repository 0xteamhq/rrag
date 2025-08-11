//! # Data Retention and Historical Analysis
//! 
//! Automated data lifecycle management with configurable retention policies,
//! historical analysis capabilities, and efficient archiving for RRAG observability data.

use crate::{RragError, RragResult};
use super::{
    metrics::Metric,
    logging::LogEntry,
    health::ServiceHealth,
    profiling::ProfileData,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc, Duration};

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    pub enabled: bool,
    pub retention_days: u32,
    pub archive_enabled: bool,
    pub archive_compression: bool,
    pub cleanup_interval_hours: u32,
    pub policies: Vec<RetentionPolicyConfig>,
    pub historical_analysis_enabled: bool,
    pub trend_analysis_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_days: 90,
            archive_enabled: true,
            archive_compression: true,
            cleanup_interval_hours: 24,
            policies: vec![
                RetentionPolicyConfig::default_metrics(),
                RetentionPolicyConfig::default_logs(),
                RetentionPolicyConfig::default_health(),
                RetentionPolicyConfig::default_profiles(),
            ],
            historical_analysis_enabled: true,
            trend_analysis_days: 30,
        }
    }
}

/// Individual retention policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicyConfig {
    pub name: String,
    pub data_type: DataType,
    pub retention_days: u32,
    pub archive_after_days: Option<u32>,
    pub compression_enabled: bool,
    pub priority: RetentionPriority,
    pub conditions: Vec<RetentionCondition>,
}

impl RetentionPolicyConfig {
    pub fn default_metrics() -> Self {
        Self {
            name: "metrics_policy".to_string(),
            data_type: DataType::Metrics,
            retention_days: 90,
            archive_after_days: Some(30),
            compression_enabled: true,
            priority: RetentionPriority::Medium,
            conditions: vec![],
        }
    }

    pub fn default_logs() -> Self {
        Self {
            name: "logs_policy".to_string(),
            data_type: DataType::Logs,
            retention_days: 30,
            archive_after_days: Some(7),
            compression_enabled: true,
            priority: RetentionPriority::High,
            conditions: vec![
                RetentionCondition::SeverityLevel("ERROR".to_string(), 60),
                RetentionCondition::SeverityLevel("WARN".to_string(), 30),
            ],
        }
    }

    pub fn default_health() -> Self {
        Self {
            name: "health_policy".to_string(),
            data_type: DataType::HealthChecks,
            retention_days: 180,
            archive_after_days: Some(60),
            compression_enabled: true,
            priority: RetentionPriority::High,
            conditions: vec![],
        }
    }

    pub fn default_profiles() -> Self {
        Self {
            name: "profiles_policy".to_string(),
            data_type: DataType::Profiles,
            retention_days: 60,
            archive_after_days: Some(14),
            compression_enabled: true,
            priority: RetentionPriority::Medium,
            conditions: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Metrics,
    Logs,
    HealthChecks,
    Profiles,
    Alerts,
    UserActivity,
    SystemEvents,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RetentionPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Retention condition for selective data retention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionCondition {
    SeverityLevel(String, u32), // level, retention_days
    ComponentName(String, u32), // component, retention_days
    UserDefined(String, String, u32), // field, value, retention_days
    DataSize(u64, u32), // max_size_bytes, retention_days
}

/// Retention policy for specific data types
pub struct RetentionPolicy {
    config: RetentionPolicyConfig,
    last_cleanup: Option<DateTime<Utc>>,
    items_processed: u64,
    items_archived: u64,
    items_deleted: u64,
}

impl RetentionPolicy {
    pub fn new(config: RetentionPolicyConfig) -> Self {
        Self {
            config,
            last_cleanup: None,
            items_processed: 0,
            items_archived: 0,
            items_deleted: 0,
        }
    }

    pub fn should_retain(&self, timestamp: DateTime<Utc>) -> RetentionAction {
        let age = Utc::now() - timestamp;
        let age_days = age.num_days() as u32;

        // Check if data should be deleted
        if age_days > self.config.retention_days {
            return RetentionAction::Delete;
        }

        // Check if data should be archived
        if let Some(archive_days) = self.config.archive_after_days {
            if age_days > archive_days {
                return RetentionAction::Archive;
            }
        }

        RetentionAction::Keep
    }

    pub fn apply_conditions(&self, data: &dyn RetentionData) -> Option<u32> {
        for condition in &self.config.conditions {
            match condition {
                RetentionCondition::SeverityLevel(level, days) => {
                    if let Some(severity) = data.severity_level() {
                        if severity == *level {
                            return Some(*days);
                        }
                    }
                },
                RetentionCondition::ComponentName(component, days) => {
                    if let Some(comp) = data.component_name() {
                        if comp == *component {
                            return Some(*days);
                        }
                    }
                },
                RetentionCondition::UserDefined(field, value, days) => {
                    if let Some(field_value) = data.custom_field(field) {
                        if field_value == *value {
                            return Some(*days);
                        }
                    }
                },
                RetentionCondition::DataSize(max_size, days) => {
                    if data.data_size() > *max_size {
                        return Some(*days);
                    }
                },
            }
        }
        None
    }

    pub fn stats(&self) -> RetentionPolicyStats {
        RetentionPolicyStats {
            name: self.config.name.clone(),
            data_type: self.config.data_type.clone(),
            retention_days: self.config.retention_days,
            last_cleanup: self.last_cleanup,
            items_processed: self.items_processed,
            items_archived: self.items_archived,
            items_deleted: self.items_deleted,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicyStats {
    pub name: String,
    pub data_type: DataType,
    pub retention_days: u32,
    pub last_cleanup: Option<DateTime<Utc>>,
    pub items_processed: u64,
    pub items_archived: u64,
    pub items_deleted: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RetentionAction {
    Keep,
    Archive,
    Delete,
}

/// Trait for data that can be subject to retention policies
pub trait RetentionData {
    fn timestamp(&self) -> DateTime<Utc>;
    fn data_size(&self) -> u64;
    fn severity_level(&self) -> Option<String> { None }
    fn component_name(&self) -> Option<String> { None }
    fn custom_field(&self, _field: &str) -> Option<String> { None }
    fn data_type(&self) -> DataType;
}

impl RetentionData for Metric {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn data_size(&self) -> u64 {
        // Rough estimate of metric size
        std::mem::size_of::<Metric>() as u64 + 
        self.name.len() as u64 + 
        self.labels.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() as u64
    }

    fn component_name(&self) -> Option<String> {
        self.labels.get("component").cloned()
    }

    fn custom_field(&self, field: &str) -> Option<String> {
        self.labels.get(field).cloned()
    }

    fn data_type(&self) -> DataType {
        DataType::Metrics
    }
}

impl RetentionData for LogEntry {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn data_size(&self) -> u64 {
        std::mem::size_of::<LogEntry>() as u64 + 
        self.message.len() as u64 + 
        self.component.len() as u64 +
        self.fields.iter().map(|(k, v)| k.len() + v.to_string().len()).sum::<usize>() as u64
    }

    fn severity_level(&self) -> Option<String> {
        Some(self.level.to_string())
    }

    fn component_name(&self) -> Option<String> {
        Some(self.component.clone())
    }

    fn custom_field(&self, field: &str) -> Option<String> {
        self.fields.get(field).and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    fn data_type(&self) -> DataType {
        DataType::Logs
    }
}

impl RetentionData for ServiceHealth {
    fn timestamp(&self) -> DateTime<Utc> {
        self.last_check
    }

    fn data_size(&self) -> u64 {
        std::mem::size_of::<ServiceHealth>() as u64 + 
        self.component_name.len() as u64 +
        self.error_message.as_ref().map(|s| s.len()).unwrap_or(0) as u64
    }

    fn component_name(&self) -> Option<String> {
        Some(self.component_name.clone())
    }

    fn severity_level(&self) -> Option<String> {
        Some(self.status.to_string())
    }

    fn data_type(&self) -> DataType {
        DataType::HealthChecks
    }
}

impl RetentionData for ProfileData {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn data_size(&self) -> u64 {
        std::mem::size_of::<ProfileData>() as u64 + 
        self.operation.len() as u64 + 
        self.component.len() as u64 +
        self.custom_metrics.iter().map(|(k, _)| k.len()).sum::<usize>() as u64
    }

    fn component_name(&self) -> Option<String> {
        Some(self.component.clone())
    }

    fn custom_field(&self, field: &str) -> Option<String> {
        self.tags.get(field).cloned()
    }

    fn data_type(&self) -> DataType {
        DataType::Profiles
    }
}

/// Archive manager for compressed storage
pub struct ArchiveManager {
    archive_path: String,
    compression_enabled: bool,
    max_archive_size_mb: u64,
}

impl ArchiveManager {
    pub fn new(archive_path: impl Into<String>, compression_enabled: bool) -> Self {
        Self {
            archive_path: archive_path.into(),
            compression_enabled,
            max_archive_size_mb: 1024, // 1 GB default
        }
    }

    pub async fn archive_data(&self, data: &[u8], filename: &str) -> RragResult<ArchiveResult> {
        let full_path = format!("{}/{}", self.archive_path, filename);
        
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&full_path).parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| RragError::storage("create_archive_directory", e))?;
        }

        let final_data = if self.compression_enabled {
            // In a real implementation, this would use a compression library like flate2
            // For now, we'll simulate compression
            let compressed_data = data.to_vec(); // Mock: no actual compression
            compressed_data
        } else {
            data.to_vec()
        };

        // Write archived data
        tokio::fs::write(&full_path, &final_data).await
            .map_err(|e| RragError::storage("write_archive", e))?;

        Ok(ArchiveResult {
            archived_at: Utc::now(),
            file_path: full_path,
            original_size: data.len() as u64,
            archived_size: final_data.len() as u64,
            compression_ratio: if data.len() > 0 {
                final_data.len() as f64 / data.len() as f64
            } else {
                1.0
            },
        })
    }

    pub async fn restore_data(&self, filename: &str) -> RragResult<Vec<u8>> {
        let full_path = format!("{}/{}", self.archive_path, filename);
        
        let archived_data = tokio::fs::read(&full_path).await
            .map_err(|e| RragError::storage("read_archive", e))?;

        // If compression was enabled, decompress here
        // For now, return as-is since we're not actually compressing
        Ok(archived_data)
    }

    pub async fn delete_archive(&self, filename: &str) -> RragResult<()> {
        let full_path = format!("{}/{}", self.archive_path, filename);
        tokio::fs::remove_file(&full_path).await
            .map_err(|e| RragError::storage("delete_archive", e))
    }

    pub async fn list_archives(&self) -> RragResult<Vec<ArchiveInfo>> {
        let mut archives = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.archive_path).await
            .map_err(|e| RragError::storage("read_archive_directory", e))?;

        while let Some(entry) = dir.next_entry().await
            .map_err(|e| RragError::storage("read_directory_entry", e))? {
            
            if let Ok(metadata) = entry.metadata().await {
                if metadata.is_file() {
                    archives.push(ArchiveInfo {
                        filename: entry.file_name().to_string_lossy().to_string(),
                        size_bytes: metadata.len(),
                        created_at: metadata.created().ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| Utc::now() - Duration::seconds(d.as_secs() as i64))
                            .unwrap_or(Utc::now()),
                    });
                }
            }
        }

        Ok(archives)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveResult {
    pub archived_at: DateTime<Utc>,
    pub file_path: String,
    pub original_size: u64,
    pub archived_size: u64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveInfo {
    pub filename: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
}

/// Historical analyzer for trend analysis
pub struct HistoricalAnalyzer {
    analysis_window_days: u32,
}

impl HistoricalAnalyzer {
    pub fn new(analysis_window_days: u32) -> Self {
        Self { analysis_window_days }
    }

    pub async fn analyze_trends<T: RetentionData>(&self, data: &[T]) -> TrendAnalysis {
        let cutoff_time = Utc::now() - Duration::days(self.analysis_window_days as i64);
        let recent_data: Vec<_> = data.iter()
            .filter(|item| item.timestamp() >= cutoff_time)
            .collect();

        if recent_data.is_empty() {
            return TrendAnalysis::empty();
        }

        // Analyze data volume trends
        let volume_trend = self.analyze_volume_trend(&recent_data).await;
        
        // Analyze component trends
        let component_trends = self.analyze_component_trends(&recent_data).await;
        
        // Analyze severity trends (for logs/alerts)
        let severity_trends = self.analyze_severity_trends(&recent_data).await;

        TrendAnalysis {
            analysis_period_days: self.analysis_window_days,
            generated_at: Utc::now(),
            total_data_points: recent_data.len(),
            volume_trend,
            component_trends,
            severity_trends,
            recommendations: self.generate_recommendations(&volume_trend, &component_trends).await,
        }
    }

    async fn analyze_volume_trend<T: RetentionData>(&self, data: &[&T]) -> VolumeTrend {
        let days = self.analysis_window_days.min(30) as usize; // Max 30 days for daily buckets
        let mut daily_counts = vec![0; days];
        let mut daily_sizes = vec![0u64; days];

        for item in data {
            let days_ago = (Utc::now() - item.timestamp()).num_days().max(0) as usize;
            if days_ago < days {
                let index = days - 1 - days_ago; // Reverse order (recent first)
                daily_counts[index] += 1;
                daily_sizes[index] += item.data_size();
            }
        }

        let avg_daily_count = if days > 0 {
            daily_counts.iter().sum::<usize>() as f64 / days as f64
        } else {
            0.0
        };

        let avg_daily_size = if days > 0 {
            daily_sizes.iter().sum::<u64>() as f64 / days as f64
        } else {
            0.0
        };

        // Calculate trend direction
        let recent_half = &daily_counts[days/2..];
        let older_half = &daily_counts[..days/2];
        
        let recent_avg = if recent_half.len() > 0 {
            recent_half.iter().sum::<usize>() as f64 / recent_half.len() as f64
        } else {
            0.0
        };
        
        let older_avg = if older_half.len() > 0 {
            older_half.iter().sum::<usize>() as f64 / older_half.len() as f64
        } else {
            0.0
        };

        let trend_direction = if recent_avg > older_avg * 1.1 {
            TrendDirection::Increasing
        } else if recent_avg < older_avg * 0.9 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        VolumeTrend {
            daily_counts,
            daily_sizes,
            average_daily_count: avg_daily_count,
            average_daily_size_bytes: avg_daily_size,
            trend_direction,
            growth_rate_percent: if older_avg > 0.0 {
                ((recent_avg - older_avg) / older_avg) * 100.0
            } else {
                0.0
            },
        }
    }

    async fn analyze_component_trends<T: RetentionData>(&self, data: &[&T]) -> HashMap<String, ComponentTrend> {
        let mut component_data: HashMap<String, Vec<&T>> = HashMap::new();

        for item in data {
            if let Some(component) = item.component_name() {
                component_data.entry(component).or_default().push(*item);
            }
        }

        let mut trends = HashMap::new();
        for (component, items) in component_data {
            let volume_trend = self.analyze_volume_trend(&items).await;
            trends.insert(component.clone(), ComponentTrend {
                component_name: component,
                data_count: items.len(),
                volume_trend,
            });
        }

        trends
    }

    async fn analyze_severity_trends<T: RetentionData>(&self, data: &[&T]) -> HashMap<String, SeverityTrend> {
        let mut severity_data: HashMap<String, Vec<&T>> = HashMap::new();

        for item in data {
            if let Some(severity) = item.severity_level() {
                severity_data.entry(severity).or_default().push(*item);
            }
        }

        let mut trends = HashMap::new();
        for (severity, items) in severity_data {
            let volume_trend = self.analyze_volume_trend(&items).await;
            trends.insert(severity.clone(), SeverityTrend {
                severity_level: severity,
                occurrence_count: items.len(),
                volume_trend,
            });
        }

        trends
    }

    async fn generate_recommendations(&self, volume_trend: &VolumeTrend, component_trends: &HashMap<String, ComponentTrend>) -> Vec<RetentionRecommendation> {
        let mut recommendations = Vec::new();

        // Volume-based recommendations
        if volume_trend.growth_rate_percent > 50.0 {
            recommendations.push(RetentionRecommendation {
                category: RecommendationCategory::Storage,
                priority: RecommendationPriority::High,
                message: "Data volume is growing rapidly. Consider reducing retention periods or implementing more aggressive archiving.".to_string(),
                estimated_savings_percent: 30.0,
            });
        }

        if volume_trend.average_daily_size_bytes > 1_000_000_000.0 { // > 1GB daily
            recommendations.push(RetentionRecommendation {
                category: RecommendationCategory::Compression,
                priority: RecommendationPriority::Medium,
                message: "Large daily data volume detected. Enable compression to reduce storage costs.".to_string(),
                estimated_savings_percent: 60.0,
            });
        }

        // Component-based recommendations
        for (component, trend) in component_trends {
            if trend.volume_trend.growth_rate_percent > 100.0 {
                recommendations.push(RetentionRecommendation {
                    category: RecommendationCategory::ComponentSpecific,
                    priority: RecommendationPriority::High,
                    message: format!("Component '{}' is producing data at an increasing rate. Review logging levels or implement component-specific retention policies.", component),
                    estimated_savings_percent: 25.0,
                });
            }
        }

        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub analysis_period_days: u32,
    pub generated_at: DateTime<Utc>,
    pub total_data_points: usize,
    pub volume_trend: VolumeTrend,
    pub component_trends: HashMap<String, ComponentTrend>,
    pub severity_trends: HashMap<String, SeverityTrend>,
    pub recommendations: Vec<RetentionRecommendation>,
}

impl TrendAnalysis {
    fn empty() -> Self {
        Self {
            analysis_period_days: 0,
            generated_at: Utc::now(),
            total_data_points: 0,
            volume_trend: VolumeTrend {
                daily_counts: Vec::new(),
                daily_sizes: Vec::new(),
                average_daily_count: 0.0,
                average_daily_size_bytes: 0.0,
                trend_direction: TrendDirection::Stable,
                growth_rate_percent: 0.0,
            },
            component_trends: HashMap::new(),
            severity_trends: HashMap::new(),
            recommendations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeTrend {
    pub daily_counts: Vec<usize>,
    pub daily_sizes: Vec<u64>,
    pub average_daily_count: f64,
    pub average_daily_size_bytes: f64,
    pub trend_direction: TrendDirection,
    pub growth_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTrend {
    pub component_name: String,
    pub data_count: usize,
    pub volume_trend: VolumeTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityTrend {
    pub severity_level: String,
    pub occurrence_count: usize,
    pub volume_trend: VolumeTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub message: String,
    pub estimated_savings_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Storage,
    Compression,
    RetentionPeriod,
    ComponentSpecific,
    ArchivingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Main data retention system
pub struct DataRetention {
    config: RetentionConfig,
    policies: Arc<RwLock<HashMap<String, RetentionPolicy>>>,
    archive_manager: Arc<ArchiveManager>,
    historical_analyzer: Arc<HistoricalAnalyzer>,
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

impl DataRetention {
    pub async fn new(config: RetentionConfig) -> RragResult<Self> {
        let policies = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize policies from config
        {
            let mut policy_map = policies.write().await;
            for policy_config in &config.policies {
                let policy = RetentionPolicy::new(policy_config.clone());
                policy_map.insert(policy_config.name.clone(), policy);
            }
        }

        let archive_manager = Arc::new(ArchiveManager::new(
            "./archives",
            config.archive_compression
        ));

        let historical_analyzer = Arc::new(HistoricalAnalyzer::new(
            config.trend_analysis_days
        ));

        Ok(Self {
            config,
            policies,
            archive_manager,
            historical_analyzer,
            cleanup_handle: None,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&mut self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("data_retention", "stopped", "already running"));
        }

        if self.config.enabled {
            let handle = self.start_cleanup_loop().await?;
            self.cleanup_handle = Some(handle);
        }

        *running = true;
        tracing::info!("Data retention system started");
        Ok(())
    }

    pub async fn stop(&mut self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }

        *running = false;
        tracing::info!("Data retention system stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    async fn start_cleanup_loop(&self) -> RragResult<tokio::task::JoinHandle<()>> {
        let config = self.config.clone();
        let policies = self.policies.clone();
        let is_running = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(config.cleanup_interval_hours as u64 * 3600)
            );

            while *is_running.read().await {
                interval.tick().await;
                
                tracing::info!("Running data retention cleanup");
                
                // In a real implementation, this would:
                // 1. Query the data stores for items to process
                // 2. Apply retention policies
                // 3. Archive or delete data as needed
                // 4. Update policy statistics
                
                let mut policy_map = policies.write().await;
                for (name, policy) in policy_map.iter_mut() {
                    tracing::debug!("Processing retention policy: {}", name);
                    // Mock processing
                    policy.items_processed += 10;
                    policy.last_cleanup = Some(Utc::now());
                }
            }
        });

        Ok(handle)
    }

    pub async fn apply_retention_policy<T: RetentionData + Clone>(&self, data: Vec<T>, data_type: DataType) -> RragResult<RetentionResult> {
        let policies = self.policies.read().await;
        
        // Find applicable policy for this data type
        let policy = policies.values()
            .find(|p| p.config.data_type == data_type)
            .ok_or_else(|| RragError::config("retention_policy", "exists", &format!("{:?}", data_type)))?;

        let mut result = RetentionResult {
            processed_count: 0,
            kept_count: 0,
            archived_count: 0,
            deleted_count: 0,
            errors: Vec::new(),
        };

        for item in data {
            result.processed_count += 1;

            // Check custom conditions first
            let retention_days = policy.apply_conditions(&item)
                .unwrap_or(policy.config.retention_days);

            let age = Utc::now() - item.timestamp();
            let age_days = age.num_days() as u32;

            if age_days > retention_days {
                // Delete item
                result.deleted_count += 1;
            } else if let Some(archive_days) = policy.config.archive_after_days {
                if age_days > archive_days {
                    // Archive item
                    match self.archive_item(&item).await {
                        Ok(_) => result.archived_count += 1,
                        Err(e) => result.errors.push(e.to_string()),
                    }
                } else {
                    result.kept_count += 1;
                }
            } else {
                result.kept_count += 1;
            }
        }

        Ok(result)
    }

    async fn archive_item<T: RetentionData>(&self, item: &T) -> RragResult<ArchiveResult> {
        // Serialize the item (this is a simplified version)
        let serialized_data = serde_json::to_vec(&serde_json::json!({
            "timestamp": item.timestamp(),
            "data_type": item.data_type(),
            "size": item.data_size(),
            "component": item.component_name(),
            "severity": item.severity_level(),
        })).map_err(|e| RragError::agent("serialization", e.to_string()))?;

        let filename = format!(
            "{}_{}.json",
            format!("{:?}", item.data_type()).to_lowercase(),
            item.timestamp().format("%Y%m%d_%H%M%S")
        );

        self.archive_manager.archive_data(&serialized_data, &filename).await
    }

    pub async fn analyze_historical_data<T: RetentionData>(&self, data: &[T]) -> TrendAnalysis {
        self.historical_analyzer.analyze_trends(data).await
    }

    pub async fn get_retention_stats(&self) -> Vec<RetentionPolicyStats> {
        let policies = self.policies.read().await;
        policies.values().map(|p| p.stats()).collect()
    }

    pub async fn add_policy(&self, policy_config: RetentionPolicyConfig) -> RragResult<()> {
        let mut policies = self.policies.write().await;
        let policy = RetentionPolicy::new(policy_config.clone());
        policies.insert(policy_config.name.clone(), policy);
        Ok(())
    }

    pub async fn remove_policy(&self, policy_name: &str) -> RragResult<()> {
        let mut policies = self.policies.write().await;
        policies.remove(policy_name);
        Ok(())
    }

    pub async fn update_policy(&self, policy_config: RetentionPolicyConfig) -> RragResult<()> {
        let mut policies = self.policies.write().await;
        let policy = RetentionPolicy::new(policy_config.clone());
        policies.insert(policy_config.name.clone(), policy);
        Ok(())
    }

    pub async fn get_archive_info(&self) -> RragResult<Vec<ArchiveInfo>> {
        self.archive_manager.list_archives().await
    }

    pub async fn restore_from_archive(&self, filename: &str) -> RragResult<Vec<u8>> {
        self.archive_manager.restore_data(filename).await
    }

    pub async fn delete_archive(&self, filename: &str) -> RragResult<()> {
        self.archive_manager.delete_archive(filename).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionResult {
    pub processed_count: usize,
    pub kept_count: usize,
    pub archived_count: usize,
    pub deleted_count: usize,
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::{metrics::MetricType, logging::LogLevel};

    #[test]
    fn test_retention_policy_creation() {
        let config = RetentionPolicyConfig::default_logs();
        let policy = RetentionPolicy::new(config);
        
        assert_eq!(policy.config.data_type, DataType::Logs);
        assert_eq!(policy.config.retention_days, 30);
        assert!(policy.config.archive_after_days.is_some());
        assert_eq!(policy.config.archive_after_days.unwrap(), 7);
    }

    #[test]
    fn test_retention_action() {
        let config = RetentionPolicyConfig {
            name: "test_policy".to_string(),
            data_type: DataType::Metrics,
            retention_days: 30,
            archive_after_days: Some(7),
            compression_enabled: true,
            priority: RetentionPriority::Medium,
            conditions: vec![],
        };
        
        let policy = RetentionPolicy::new(config);
        
        // Test recent data (should keep)
        let recent_time = Utc::now() - Duration::days(5);
        assert_eq!(policy.should_retain(recent_time), RetentionAction::Keep);
        
        // Test old data for archiving
        let archive_time = Utc::now() - Duration::days(10);
        assert_eq!(policy.should_retain(archive_time), RetentionAction::Archive);
        
        // Test very old data (should delete)
        let delete_time = Utc::now() - Duration::days(40);
        assert_eq!(policy.should_retain(delete_time), RetentionAction::Delete);
    }

    #[tokio::test]
    async fn test_archive_manager() {
        let temp_dir = tempfile::tempdir().unwrap();
        let archive_manager = ArchiveManager::new(
            temp_dir.path().to_string_lossy().to_string(),
            true
        );

        let test_data = b"test archive data";
        let filename = "test_archive.json";

        let result = archive_manager.archive_data(test_data, filename).await.unwrap();
        assert_eq!(result.original_size, test_data.len() as u64);
        assert!(!result.file_path.is_empty());

        let restored_data = archive_manager.restore_data(filename).await.unwrap();
        assert_eq!(restored_data, test_data);

        let archives = archive_manager.list_archives().await.unwrap();
        assert_eq!(archives.len(), 1);
        assert_eq!(archives[0].filename, filename);

        archive_manager.delete_archive(filename).await.unwrap();
        let archives_after_delete = archive_manager.list_archives().await.unwrap();
        assert_eq!(archives_after_delete.len(), 0);
    }

    #[tokio::test]
    async fn test_historical_analyzer() {
        let analyzer = HistoricalAnalyzer::new(7); // 7 days analysis window

        // Create test metrics with different timestamps
        let mut test_metrics = Vec::new();
        for i in 0..10 {
            let timestamp = Utc::now() - Duration::days(i);
            let metric = Metric::counter("test_counter", (i * 10) as u64)
                .with_label("component", "test_component");
            let mut metric = metric;
            metric.timestamp = timestamp;
            test_metrics.push(metric);
        }

        let analysis = analyzer.analyze_trends(&test_metrics).await;
        assert_eq!(analysis.total_data_points, 8); // Only last 7 days + today
        assert!(!analysis.volume_trend.daily_counts.is_empty());
        assert!(!analysis.component_trends.is_empty());
        assert!(analysis.component_trends.contains_key("test_component"));
    }

    #[tokio::test]
    async fn test_data_retention_system() {
        let config = RetentionConfig::default();
        let mut retention = DataRetention::new(config).await.unwrap();

        assert!(!retention.is_healthy().await);

        retention.start().await.unwrap();
        assert!(retention.is_healthy().await);

        // Test adding a custom policy
        let custom_policy = RetentionPolicyConfig {
            name: "custom_test_policy".to_string(),
            data_type: DataType::Metrics,
            retention_days: 60,
            archive_after_days: Some(14),
            compression_enabled: true,
            priority: RetentionPriority::High,
            conditions: vec![],
        };

        retention.add_policy(custom_policy).await.unwrap();

        let stats = retention.get_retention_stats().await;
        assert!(stats.iter().any(|s| s.name == "custom_test_policy"));

        retention.stop().await.unwrap();
        assert!(!retention.is_healthy().await);
    }

    #[tokio::test]
    async fn test_retention_data_trait() {
        // Test Metric implementation
        let metric = Metric::counter("test_metric", 100)
            .with_label("component", "test_component");
        
        assert_eq!(metric.data_type(), DataType::Metrics);
        assert!(metric.data_size() > 0);
        assert_eq!(metric.component_name(), Some("test_component".to_string()));

        // Test LogEntry implementation
        let log_entry = super::super::logging::LogEntry::new(LogLevel::Error, "Test error", "test_component");
        assert_eq!(log_entry.data_type(), DataType::Logs);
        assert_eq!(log_entry.severity_level(), Some("ERROR".to_string()));
        assert_eq!(log_entry.component_name(), Some("test_component".to_string()));
    }

    #[test]
    fn test_retention_conditions() {
        let config = RetentionPolicyConfig {
            name: "conditional_policy".to_string(),
            data_type: DataType::Logs,
            retention_days: 30,
            archive_after_days: None,
            compression_enabled: false,
            priority: RetentionPriority::Medium,
            conditions: vec![
                RetentionCondition::SeverityLevel("ERROR".to_string(), 90),
                RetentionCondition::ComponentName("critical_component".to_string(), 180),
            ],
        };

        let policy = RetentionPolicy::new(config);
        
        let error_log = super::super::logging::LogEntry::new(LogLevel::Error, "Error message", "normal_component");
        assert_eq!(policy.apply_conditions(&error_log), Some(90));

        let critical_log = super::super::logging::LogEntry::new(LogLevel::Info, "Info message", "critical_component");
        assert_eq!(policy.apply_conditions(&critical_log), Some(180));

        let normal_log = super::super::logging::LogEntry::new(LogLevel::Info, "Info message", "normal_component");
        assert_eq!(policy.apply_conditions(&normal_log), None);
    }
}