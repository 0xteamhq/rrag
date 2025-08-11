//! # System Monitoring
//! 
//! Real-time monitoring of RRAG system performance, health, and usage patterns.
//! Provides insights into search analytics, performance bottlenecks, and user behavior.

use crate::{RragError, RragResult};
use super::metrics::MetricsCollector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub collection_interval_seconds: u64,
    pub performance_window_minutes: u32,
    pub search_analytics_enabled: bool,
    pub user_tracking_enabled: bool,
    pub resource_monitoring_enabled: bool,
    pub alert_thresholds: AlertThresholds,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_seconds: 30,
            performance_window_minutes: 5,
            search_analytics_enabled: true,
            user_tracking_enabled: true,
            resource_monitoring_enabled: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert threshold configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub error_rate_percent: f64,
    pub response_time_ms: f64,
    pub disk_usage_percent: f64,
    pub queue_size: usize,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 80.0,
            memory_usage_percent: 85.0,
            error_rate_percent: 5.0,
            response_time_ms: 1000.0,
            disk_usage_percent: 90.0,
            queue_size: 1000,
        }
    }
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_mb: f64,
    pub disk_usage_percent: f64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub active_connections: u32,
    pub thread_count: u32,
    pub gc_collections: u64,
    pub gc_pause_time_ms: f64,
}

/// Search analytics data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchAnalytics {
    pub timestamp: DateTime<Utc>,
    pub query: String,
    pub query_type: QueryType,
    pub results_count: usize,
    pub processing_time_ms: f64,
    pub success: bool,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub similarity_scores: Vec<f64>,
    pub rerank_applied: bool,
    pub cache_hit: bool,
}

/// Query classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryType {
    Factual,
    Conceptual,
    Procedural,
    Conversational,
    Unknown,
}

/// User activity tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserActivity {
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub session_id: String,
    pub action: UserAction,
    pub query: Option<String>,
    pub response_time_ms: f64,
    pub success: bool,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserAction {
    Search,
    Chat,
    DocumentUpload,
    DocumentView,
    SystemHealth,
    Other(String),
}

/// Performance monitoring service
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics_collector: Arc<MetricsCollector>,
    performance_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    collection_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl PerformanceMonitor {
    pub async fn new(config: MonitoringConfig, metrics_collector: Arc<MetricsCollector>) -> RragResult<Self> {
        Ok(Self {
            config,
            metrics_collector,
            performance_history: Arc::new(RwLock::new(Vec::new())),
            collection_handle: Arc::new(RwLock::new(None)),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("performance_monitor", "stopped", "already running"));
        }

        let config = self.config.clone();
        let metrics_collector = self.metrics_collector.clone();
        let performance_history = self.performance_history.clone();
        let is_running_clone = self.is_running.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(config.collection_interval_seconds)
            );

            while *is_running_clone.read().await {
                interval.tick().await;
                
                if let Ok(metrics) = Self::collect_system_metrics().await {
                    // Store in history
                    let mut history = performance_history.write().await;
                    history.push(metrics.clone());
                    
                    // Keep only recent data
                    let retention_size = (config.performance_window_minutes * 60 / config.collection_interval_seconds as u32) as usize;
                    let current_len = history.len();
                    if current_len > retention_size {
                        history.drain(0..current_len - retention_size);
                    }
                    drop(history);

                    // Update metrics
                    let _ = metrics_collector.set_gauge("system_cpu_usage_percent", metrics.cpu_usage_percent).await;
                    let _ = metrics_collector.set_gauge("system_memory_usage_mb", metrics.memory_usage_mb).await;
                    let _ = metrics_collector.set_gauge("system_memory_usage_percent", metrics.memory_usage_percent).await;
                    let _ = metrics_collector.set_gauge("system_disk_usage_mb", metrics.disk_usage_mb).await;
                    let _ = metrics_collector.set_gauge("system_active_connections", metrics.active_connections as f64).await;
                    let _ = metrics_collector.set_gauge("system_thread_count", metrics.thread_count as f64).await;
                }
            }
        });

        {
            let mut handle_guard = self.collection_handle.write().await;
            *handle_guard = Some(handle);
        }
        *running = true;
        tracing::info!("Performance monitor started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        *running = false;
        
        {
            let mut handle_guard = self.collection_handle.write().await;
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }

        tracing::info!("Performance monitor stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    pub async fn get_current_metrics(&self) -> RragResult<PerformanceMetrics> {
        Self::collect_system_metrics().await
    }

    pub async fn get_metrics_history(&self) -> Vec<PerformanceMetrics> {
        self.performance_history.read().await.clone()
    }

    pub async fn get_average_metrics(&self, minutes: u32) -> RragResult<PerformanceMetrics> {
        let history = self.performance_history.read().await;
        if history.is_empty() {
            return self.get_current_metrics().await;
        }

        let cutoff_time = Utc::now() - Duration::minutes(minutes as i64);
        let recent_metrics: Vec<_> = history
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .collect();

        if recent_metrics.is_empty() {
            return self.get_current_metrics().await;
        }

        let count = recent_metrics.len() as f64;
        let avg_cpu = recent_metrics.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / count;
        let avg_memory = recent_metrics.iter().map(|m| m.memory_usage_mb).sum::<f64>() / count;
        let avg_memory_percent = recent_metrics.iter().map(|m| m.memory_usage_percent).sum::<f64>() / count;
        let avg_disk = recent_metrics.iter().map(|m| m.disk_usage_mb).sum::<f64>() / count;
        let avg_disk_percent = recent_metrics.iter().map(|m| m.disk_usage_percent).sum::<f64>() / count;
        let avg_connections = recent_metrics.iter().map(|m| m.active_connections as f64).sum::<f64>() / count;
        let avg_threads = recent_metrics.iter().map(|m| m.thread_count as f64).sum::<f64>() / count;

        Ok(PerformanceMetrics {
            timestamp: Utc::now(),
            cpu_usage_percent: avg_cpu,
            memory_usage_mb: avg_memory,
            memory_usage_percent: avg_memory_percent,
            disk_usage_mb: avg_disk,
            disk_usage_percent: avg_disk_percent,
            network_bytes_sent: 0, // Averages don't make sense for cumulative metrics
            network_bytes_received: 0,
            active_connections: avg_connections as u32,
            thread_count: avg_threads as u32,
            gc_collections: 0,
            gc_pause_time_ms: 0.0,
        })
    }

    async fn collect_system_metrics() -> RragResult<PerformanceMetrics> {
        // This would integrate with actual system monitoring libraries
        // For now, providing mock data
        Ok(PerformanceMetrics {
            timestamp: Utc::now(),
            cpu_usage_percent: 25.0 + (rand::random::<f64>() * 50.0), // Mock: 25-75%
            memory_usage_mb: 512.0 + (rand::random::<f64>() * 1024.0), // Mock: 512-1536 MB
            memory_usage_percent: 30.0 + (rand::random::<f64>() * 40.0), // Mock: 30-70%
            disk_usage_mb: 2048.0 + (rand::random::<f64>() * 1024.0), // Mock: 2-3 GB
            disk_usage_percent: 40.0 + (rand::random::<f64>() * 30.0), // Mock: 40-70%
            network_bytes_sent: rand::random::<u64>() % 1_000_000,
            network_bytes_received: rand::random::<u64>() % 1_000_000,
            active_connections: (10 + rand::random::<u32>() % 100) as u32,
            thread_count: (50 + rand::random::<u32>() % 50) as u32,
            gc_collections: rand::random::<u64>() % 10,
            gc_pause_time_ms: rand::random::<f64>() * 10.0,
        })
    }
}

/// Search analytics service
pub struct SearchAnalyzer {
    config: MonitoringConfig,
    metrics_collector: Arc<MetricsCollector>,
    search_history: Arc<RwLock<Vec<SearchAnalytics>>>,
    query_patterns: Arc<RwLock<HashMap<String, u64>>>,
    is_running: Arc<RwLock<bool>>,
}

impl SearchAnalyzer {
    pub async fn new(config: MonitoringConfig, metrics_collector: Arc<MetricsCollector>) -> Self {
        Self {
            config,
            metrics_collector,
            search_history: Arc::new(RwLock::new(Vec::new())),
            query_patterns: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("search_analyzer", "stopped", "already running"));
        }
        *running = true;
        tracing::info!("Search analyzer started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }
        *running = false;
        tracing::info!("Search analyzer stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    pub async fn record_search(&self, analytics: SearchAnalytics) -> RragResult<()> {
        if !*self.is_running.read().await {
            return Err(RragError::config("search_analyzer", "running", "stopped"));
        }

        // Update query patterns
        let mut patterns = self.query_patterns.write().await;
        *patterns.entry(analytics.query.clone()).or_insert(0) += 1;
        drop(patterns);

        // Store in history
        let mut history = self.search_history.write().await;
        history.push(analytics.clone());
        
        // Keep only recent data (last 1000 searches)
        let current_len = history.len();
        if current_len > 1000 {
            history.drain(0..current_len - 1000);
        }

        // Update metrics
        let _ = self.metrics_collector.inc_counter("search_queries_total").await;
        let _ = self.metrics_collector.record_timer("search_processing_time_ms", analytics.processing_time_ms).await;
        let _ = self.metrics_collector.observe_histogram("search_results_count", analytics.results_count as f64, None).await;

        if analytics.success {
            let _ = self.metrics_collector.inc_counter("search_queries_successful").await;
        } else {
            let _ = self.metrics_collector.inc_counter("search_queries_failed").await;
        }

        if analytics.cache_hit {
            let _ = self.metrics_collector.inc_counter("search_cache_hits").await;
        } else {
            let _ = self.metrics_collector.inc_counter("search_cache_misses").await;
        }

        Ok(())
    }

    pub async fn get_popular_queries(&self, limit: usize) -> Vec<(String, u64)> {
        let patterns = self.query_patterns.read().await;
        let mut query_counts: Vec<_> = patterns.iter()
            .map(|(query, count)| (query.clone(), *count))
            .collect();
        
        query_counts.sort_by(|a, b| b.1.cmp(&a.1));
        query_counts.into_iter().take(limit).collect()
    }

    pub async fn get_search_stats(&self) -> SearchStats {
        let history = self.search_history.read().await;
        
        if history.is_empty() {
            return SearchStats::default();
        }

        let total_searches = history.len();
        let successful_searches = history.iter().filter(|s| s.success).count();
        let cache_hits = history.iter().filter(|s| s.cache_hit).count();
        let rerank_applied = history.iter().filter(|s| s.rerank_applied).count();

        let avg_processing_time = history.iter()
            .map(|s| s.processing_time_ms)
            .sum::<f64>() / total_searches as f64;

        let avg_results_count = history.iter()
            .map(|s| s.results_count as f64)
            .sum::<f64>() / total_searches as f64;

        SearchStats {
            total_searches,
            successful_searches,
            success_rate: (successful_searches as f64 / total_searches as f64) * 100.0,
            cache_hit_rate: (cache_hits as f64 / total_searches as f64) * 100.0,
            rerank_usage_rate: (rerank_applied as f64 / total_searches as f64) * 100.0,
            average_processing_time_ms: avg_processing_time,
            average_results_count: avg_results_count,
            query_types: self.analyze_query_types(&history),
        }
    }

    fn analyze_query_types(&self, history: &[SearchAnalytics]) -> HashMap<QueryType, usize> {
        let mut counts = HashMap::new();
        for search in history {
            *counts.entry(search.query_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn classify_query(&self, query: &str) -> QueryType {
        // Simple heuristic-based classification
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("what") || query_lower.contains("who") || 
           query_lower.contains("when") || query_lower.contains("where") {
            QueryType::Factual
        } else if query_lower.contains("how") || query_lower.contains("explain") || 
                  query_lower.contains("describe") {
            QueryType::Procedural
        } else if query_lower.contains("why") || query_lower.contains("concept") ||
                  query_lower.contains("theory") {
            QueryType::Conceptual
        } else if query_lower.contains("can you") || query_lower.contains("please") ||
                  query_lower.len() > 50 {
            QueryType::Conversational
        } else {
            QueryType::Unknown
        }
    }
}

/// Search statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    pub total_searches: usize,
    pub successful_searches: usize,
    pub success_rate: f64,
    pub cache_hit_rate: f64,
    pub rerank_usage_rate: f64,
    pub average_processing_time_ms: f64,
    pub average_results_count: f64,
    pub query_types: HashMap<QueryType, usize>,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            total_searches: 0,
            successful_searches: 0,
            success_rate: 0.0,
            cache_hit_rate: 0.0,
            rerank_usage_rate: 0.0,
            average_processing_time_ms: 0.0,
            average_results_count: 0.0,
            query_types: HashMap::new(),
        }
    }
}

/// User activity tracking service
pub struct UserActivityTracker {
    config: MonitoringConfig,
    metrics_collector: Arc<MetricsCollector>,
    activity_history: Arc<RwLock<Vec<UserActivity>>>,
    active_sessions: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl UserActivityTracker {
    pub async fn new(config: MonitoringConfig, metrics_collector: Arc<MetricsCollector>) -> Self {
        Self {
            config,
            metrics_collector,
            activity_history: Arc::new(RwLock::new(Vec::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("user_activity_tracker", "stopped", "already running"));
        }
        *running = true;
        tracing::info!("User activity tracker started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }
        *running = false;
        tracing::info!("User activity tracker stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    pub async fn track_activity(&self, activity: UserActivity) -> RragResult<()> {
        if !*self.is_running.read().await {
            return Err(RragError::config("user_activity_tracker", "running", "stopped"));
        }

        // Update active sessions
        let mut sessions = self.active_sessions.write().await;
        sessions.insert(activity.session_id.clone(), activity.timestamp);
        drop(sessions);

        // Store activity
        let mut history = self.activity_history.write().await;
        history.push(activity.clone());
        
        // Keep only recent activity (last 10000 actions)
        let current_len = history.len();
        if current_len > 10000 {
            history.drain(0..current_len - 10000);
        }

        // Update metrics
        let _ = self.metrics_collector.inc_counter("user_actions_total").await;
        let _ = self.metrics_collector.record_timer("user_action_response_time_ms", activity.response_time_ms).await;

        match activity.action {
            UserAction::Search => {
                let _ = self.metrics_collector.inc_counter("user_searches_total").await;
            },
            UserAction::Chat => {
                let _ = self.metrics_collector.inc_counter("user_chats_total").await;
            },
            UserAction::DocumentUpload => {
                let _ = self.metrics_collector.inc_counter("user_document_uploads_total").await;
            },
            _ => {}
        }

        Ok(())
    }

    pub async fn get_active_sessions_count(&self) -> usize {
        // Clean up old sessions (older than 1 hour)
        let cutoff = Utc::now() - Duration::hours(1);
        let mut sessions = self.active_sessions.write().await;
        sessions.retain(|_, timestamp| *timestamp > cutoff);
        sessions.len()
    }

    pub async fn get_user_stats(&self, time_window_hours: i64) -> UserStats {
        let history = self.activity_history.read().await;
        let cutoff_time = Utc::now() - Duration::hours(time_window_hours);
        
        let recent_activity: Vec<_> = history
            .iter()
            .filter(|a| a.timestamp >= cutoff_time)
            .collect();

        if recent_activity.is_empty() {
            return UserStats::default();
        }

        let unique_users: std::collections::HashSet<_> = recent_activity
            .iter()
            .map(|a| a.user_id.as_str())
            .collect();

        let unique_sessions: std::collections::HashSet<_> = recent_activity
            .iter()
            .map(|a| a.session_id.as_str())
            .collect();

        let action_counts = self.count_actions(&recent_activity);
        let avg_response_time = recent_activity
            .iter()
            .map(|a| a.response_time_ms)
            .sum::<f64>() / recent_activity.len() as f64;

        UserStats {
            total_actions: recent_activity.len(),
            unique_users: unique_users.len(),
            unique_sessions: unique_sessions.len(),
            action_breakdown: action_counts,
            average_response_time_ms: avg_response_time,
            time_window_hours,
        }
    }

    fn count_actions(&self, activities: &[&UserActivity]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for activity in activities {
            let action_name = match &activity.action {
                UserAction::Search => "search",
                UserAction::Chat => "chat",
                UserAction::DocumentUpload => "document_upload",
                UserAction::DocumentView => "document_view",
                UserAction::SystemHealth => "system_health",
                UserAction::Other(name) => name,
            };
            *counts.entry(action_name.to_string()).or_insert(0) += 1;
        }
        counts
    }
}

/// User activity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStats {
    pub total_actions: usize,
    pub unique_users: usize,
    pub unique_sessions: usize,
    pub action_breakdown: HashMap<String, usize>,
    pub average_response_time_ms: f64,
    pub time_window_hours: i64,
}

impl Default for UserStats {
    fn default() -> Self {
        Self {
            total_actions: 0,
            unique_users: 0,
            unique_sessions: 0,
            action_breakdown: HashMap::new(),
            average_response_time_ms: 0.0,
            time_window_hours: 24,
        }
    }
}

/// Main system monitor orchestrating all monitoring services
pub struct SystemMonitor {
    config: MonitoringConfig,
    performance_monitor: Arc<PerformanceMonitor>,
    search_analyzer: Arc<SearchAnalyzer>,
    user_tracker: Arc<UserActivityTracker>,
    is_running: Arc<RwLock<bool>>,
}

impl SystemMonitor {
    pub async fn new(config: MonitoringConfig, metrics_collector: Arc<MetricsCollector>) -> RragResult<Self> {
        let performance_monitor = Arc::new(PerformanceMonitor::new(config.clone(), metrics_collector.clone()).await?);
        let search_analyzer = Arc::new(SearchAnalyzer::new(config.clone(), metrics_collector.clone()).await);
        let user_tracker = Arc::new(UserActivityTracker::new(config.clone(), metrics_collector).await);

        Ok(Self {
            config,
            performance_monitor,
            search_analyzer,
            user_tracker,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("system_monitor", "stopped", "already running"));
        }

        // Start all monitoring services
        if self.config.resource_monitoring_enabled {
            self.performance_monitor.start().await?;
        }
        
        if self.config.search_analytics_enabled {
            self.search_analyzer.start().await?;
        }
        
        if self.config.user_tracking_enabled {
            self.user_tracker.start().await?;
        }

        *running = true;
        tracing::info!("System monitor started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }

        // Stop all monitoring services
        self.performance_monitor.stop().await?;
        self.search_analyzer.stop().await?;
        self.user_tracker.stop().await?;

        *running = false;
        tracing::info!("System monitor stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await &&
            self.performance_monitor.is_healthy().await &&
            self.search_analyzer.is_healthy().await &&
            self.user_tracker.is_healthy().await
    }

    pub fn performance(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    pub fn search_analytics(&self) -> &SearchAnalyzer {
        &self.search_analyzer
    }

    pub fn user_activity(&self) -> &UserActivityTracker {
        &self.user_tracker
    }

    pub async fn get_system_overview(&self) -> SystemOverview {
        let performance = if self.config.resource_monitoring_enabled {
            self.performance_monitor.get_current_metrics().await.ok()
        } else {
            None
        };

        let search_stats = if self.config.search_analytics_enabled {
            Some(self.search_analyzer.get_search_stats().await)
        } else {
            None
        };

        let user_stats = if self.config.user_tracking_enabled {
            Some(self.user_tracker.get_user_stats(24).await)
        } else {
            None
        };

        SystemOverview {
            timestamp: Utc::now(),
            performance_metrics: performance,
            search_stats,
            user_stats,
            active_sessions: if self.config.user_tracking_enabled {
                Some(self.user_tracker.get_active_sessions_count().await)
            } else {
                None
            },
        }
    }
}

/// Complete system overview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOverview {
    pub timestamp: DateTime<Utc>,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub search_stats: Option<SearchStats>,
    pub user_stats: Option<UserStats>,
    pub active_sessions: Option<usize>,
}

/// Monitoring service interface
pub struct MonitoringService {
    system_monitor: Arc<RwLock<SystemMonitor>>,
}

impl MonitoringService {
    pub async fn new(config: MonitoringConfig, metrics_collector: Arc<MetricsCollector>) -> RragResult<Self> {
        let system_monitor = SystemMonitor::new(config, metrics_collector).await?;
        Ok(Self {
            system_monitor: Arc::new(RwLock::new(system_monitor)),
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        let monitor = self.system_monitor.read().await;
        monitor.start().await
    }

    pub async fn stop(&self) -> RragResult<()> {
        let monitor = self.system_monitor.read().await;
        monitor.stop().await
    }

    pub async fn is_healthy(&self) -> bool {
        let monitor = self.system_monitor.read().await;
        monitor.is_healthy().await
    }

    pub async fn get_overview(&self) -> SystemOverview {
        let monitor = self.system_monitor.read().await;
        monitor.get_system_overview().await
    }

    pub async fn record_search(&self, analytics: SearchAnalytics) -> RragResult<()> {
        let monitor = self.system_monitor.read().await;
        monitor.search_analytics().record_search(analytics).await
    }

    pub async fn track_user_activity(&self, activity: UserActivity) -> RragResult<()> {
        let monitor = self.system_monitor.read().await;
        monitor.user_activity().track_activity(activity).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::metrics::MetricsConfig;

    async fn create_test_metrics_collector() -> Arc<MetricsCollector> {
        Arc::new(MetricsCollector::new(MetricsConfig::default()).await.unwrap())
    }

    #[tokio::test]
    async fn test_performance_monitor() {
        let metrics_collector = create_test_metrics_collector().await;
        let config = MonitoringConfig::default();
        let mut monitor = PerformanceMonitor::new(config, metrics_collector).await.unwrap();
        
        assert!(!monitor.is_healthy().await);
        
        monitor.start().await.unwrap();
        assert!(monitor.is_healthy().await);
        
        let current_metrics = monitor.get_current_metrics().await.unwrap();
        assert!(current_metrics.cpu_usage_percent >= 0.0);
        assert!(current_metrics.memory_usage_mb >= 0.0);
        
        monitor.stop().await.unwrap();
        assert!(!monitor.is_healthy().await);
    }

    #[tokio::test]
    async fn test_search_analyzer() {
        let metrics_collector = create_test_metrics_collector().await;
        let config = MonitoringConfig::default();
        let analyzer = SearchAnalyzer::new(config, metrics_collector).await;
        
        analyzer.start().await.unwrap();
        assert!(analyzer.is_healthy().await);
        
        let search_analytics = SearchAnalytics {
            timestamp: Utc::now(),
            query: "test query".to_string(),
            query_type: QueryType::Factual,
            results_count: 5,
            processing_time_ms: 150.0,
            success: true,
            user_id: Some("user123".to_string()),
            session_id: Some("session456".to_string()),
            similarity_scores: vec![0.9, 0.8, 0.7],
            rerank_applied: true,
            cache_hit: false,
        };
        
        analyzer.record_search(search_analytics).await.unwrap();
        
        let stats = analyzer.get_search_stats().await;
        assert_eq!(stats.total_searches, 1);
        assert_eq!(stats.successful_searches, 1);
        assert_eq!(stats.success_rate, 100.0);
        
        analyzer.stop().await.unwrap();
        assert!(!analyzer.is_healthy().await);
    }

    #[tokio::test]
    async fn test_user_activity_tracker() {
        let metrics_collector = create_test_metrics_collector().await;
        let config = MonitoringConfig::default();
        let tracker = UserActivityTracker::new(config, metrics_collector).await;
        
        tracker.start().await.unwrap();
        assert!(tracker.is_healthy().await);
        
        let activity = UserActivity {
            timestamp: Utc::now(),
            user_id: "user123".to_string(),
            session_id: "session456".to_string(),
            action: UserAction::Search,
            query: Some("test query".to_string()),
            response_time_ms: 200.0,
            success: true,
            ip_address: Some("127.0.0.1".to_string()),
            user_agent: Some("test-agent".to_string()),
        };
        
        tracker.track_activity(activity).await.unwrap();
        
        let stats = tracker.get_user_stats(24).await;
        assert_eq!(stats.total_actions, 1);
        assert_eq!(stats.unique_users, 1);
        assert_eq!(stats.unique_sessions, 1);
        
        tracker.stop().await.unwrap();
        assert!(!tracker.is_healthy().await);
    }

    #[tokio::test]
    async fn test_system_monitor() {
        let metrics_collector = create_test_metrics_collector().await;
        let config = MonitoringConfig::default();
        let mut monitor = SystemMonitor::new(config, metrics_collector).await.unwrap();
        
        assert!(!monitor.is_healthy().await);
        
        monitor.start().await.unwrap();
        assert!(monitor.is_healthy().await);
        
        let overview = monitor.get_system_overview().await;
        assert!(overview.performance_metrics.is_some());
        assert!(overview.search_stats.is_some());
        assert!(overview.user_stats.is_some());
        
        monitor.stop().await.unwrap();
        assert!(!monitor.is_healthy().await);
    }

    #[test]
    fn test_query_classification() {
        let metrics_collector = futures::executor::block_on(create_test_metrics_collector());
        let config = MonitoringConfig::default();
        let analyzer = futures::executor::block_on(SearchAnalyzer::new(config, metrics_collector));
        
        assert_eq!(analyzer.classify_query("What is machine learning?"), QueryType::Factual);
        assert_eq!(analyzer.classify_query("How do I implement a neural network?"), QueryType::Procedural);
        assert_eq!(analyzer.classify_query("Why does backpropagation work?"), QueryType::Conceptual);
        assert_eq!(analyzer.classify_query("Can you help me understand this concept please?"), QueryType::Conversational);
        assert_eq!(analyzer.classify_query("neural networks"), QueryType::Unknown);
    }

    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_usage_percent, 80.0);
        assert_eq!(thresholds.memory_usage_percent, 85.0);
        assert_eq!(thresholds.error_rate_percent, 5.0);
        assert_eq!(thresholds.response_time_ms, 1000.0);
    }
}