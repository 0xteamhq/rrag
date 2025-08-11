//! # Cache Metrics and Monitoring
//!
//! Performance metrics and monitoring for the caching layer.

use super::{CacheStats, OverallCacheMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Cache metrics collector
pub struct MetricsCollector {
    /// Per-cache metrics
    cache_metrics: HashMap<String, CacheStats>,

    /// Operation timings
    operation_timings: OperationTimings,

    /// Memory tracking
    memory_tracker: MemoryTracker,

    /// Performance analyzer
    analyzer: PerformanceAnalyzer,

    /// Metrics history
    history: MetricsHistory,
}

/// Operation timing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTimings {
    /// Get operation timings
    pub get_timings: TimingStats,

    /// Put operation timings
    pub put_timings: TimingStats,

    /// Remove operation timings
    pub remove_timings: TimingStats,

    /// Eviction timings
    pub eviction_timings: TimingStats,

    /// Compression timings
    pub compression_timings: TimingStats,
}

/// Timing statistics for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Total operations
    pub count: u64,

    /// Total time in microseconds
    pub total_us: u64,

    /// Average time in microseconds
    pub avg_us: f32,

    /// Minimum time
    pub min_us: u64,

    /// Maximum time
    pub max_us: u64,

    /// 50th percentile
    pub p50_us: u64,

    /// 95th percentile
    pub p95_us: u64,

    /// 99th percentile
    pub p99_us: u64,
}

/// Memory usage tracker
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current memory usage
    pub current_bytes: usize,

    /// Peak memory usage
    pub peak_bytes: usize,

    /// Memory saved through compression
    pub compression_saved_bytes: usize,

    /// Memory saved through deduplication
    pub deduplication_saved_bytes: usize,

    /// Memory pressure events
    pub pressure_events: Vec<MemoryPressureEvent>,
}

/// Memory pressure event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureEvent {
    /// When the event occurred
    pub timestamp: SystemTime,

    /// Memory usage at time of event
    pub memory_bytes: usize,

    /// Pressure level (0.0 to 1.0)
    pub pressure_level: f32,

    /// Action taken
    pub action: PressureAction,

    /// Memory freed
    pub freed_bytes: usize,
}

/// Actions taken under memory pressure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PressureAction {
    /// Evicted entries
    Eviction { count: usize },

    /// Compressed entries
    Compression { count: usize },

    /// Cleared entire cache
    ClearCache,

    /// No action needed
    None,
}

/// Performance analyzer
#[derive(Debug, Clone)]
pub struct PerformanceAnalyzer {
    /// Hit rate over time
    pub hit_rate_history: Vec<(SystemTime, f32)>,

    /// Operations per second history
    pub ops_history: Vec<(SystemTime, f32)>,

    /// Latency percentiles over time
    pub latency_history: Vec<(SystemTime, LatencySnapshot)>,

    /// Efficiency score history
    pub efficiency_history: Vec<(SystemTime, f32)>,
}

/// Latency snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySnapshot {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
}

/// Metrics history for trend analysis
#[derive(Debug, Clone)]
pub struct MetricsHistory {
    /// Historical snapshots
    pub snapshots: Vec<MetricsSnapshot>,

    /// Maximum history size
    pub max_size: usize,

    /// Snapshot interval
    pub interval: Duration,

    /// Last snapshot time
    pub last_snapshot: SystemTime,
}

/// Point-in-time metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Snapshot timestamp
    pub timestamp: SystemTime,

    /// Overall metrics
    pub overall: OverallCacheMetrics,

    /// Individual cache stats
    pub cache_stats: HashMap<String, CacheStats>,

    /// Memory usage
    pub memory_bytes: usize,

    /// Active operations
    pub active_operations: u32,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            cache_metrics: HashMap::new(),
            operation_timings: OperationTimings::default(),
            memory_tracker: MemoryTracker::new(),
            analyzer: PerformanceAnalyzer::new(),
            history: MetricsHistory::new(1000, Duration::from_secs(60)),
        }
    }

    /// Record cache operation
    pub fn record_operation(&mut self, cache: &str, operation: Operation, duration: Duration) {
        let duration_us = duration.as_micros() as u64;

        match operation {
            Operation::Get { hit } => {
                self.operation_timings.get_timings.record(duration_us);

                if let Some(stats) = self.cache_metrics.get_mut(cache) {
                    if hit {
                        stats.hits += 1;
                    } else {
                        stats.misses += 1;
                    }
                    stats.hit_rate = stats.hits as f32 / (stats.hits + stats.misses) as f32;
                }
            }
            Operation::Put => {
                self.operation_timings.put_timings.record(duration_us);
            }
            Operation::Remove => {
                self.operation_timings.remove_timings.record(duration_us);
            }
            Operation::Evict => {
                self.operation_timings.eviction_timings.record(duration_us);

                if let Some(stats) = self.cache_metrics.get_mut(cache) {
                    stats.evictions += 1;
                }
            }
        }
    }

    /// Update memory usage
    pub fn update_memory(&mut self, cache: &str, bytes: usize) {
        self.memory_tracker.current_bytes = bytes;
        self.memory_tracker.peak_bytes = self.memory_tracker.peak_bytes.max(bytes);

        if let Some(stats) = self.cache_metrics.get_mut(cache) {
            stats.memory_usage = bytes;
        }

        // Check for memory pressure
        let pressure = self.calculate_memory_pressure();
        if pressure > 0.8 {
            self.memory_tracker
                .pressure_events
                .push(MemoryPressureEvent {
                    timestamp: SystemTime::now(),
                    memory_bytes: bytes,
                    pressure_level: pressure,
                    action: PressureAction::None,
                    freed_bytes: 0,
                });
        }
    }

    /// Calculate memory pressure (0.0 to 1.0)
    fn calculate_memory_pressure(&self) -> f32 {
        // Simplified - would use system memory in real implementation
        const MAX_MEMORY: usize = 1024 * 1024 * 1024; // 1GB
        (self.memory_tracker.current_bytes as f32 / MAX_MEMORY as f32).min(1.0)
    }

    /// Take metrics snapshot
    pub fn snapshot(&mut self) -> MetricsSnapshot {
        let overall = self.calculate_overall_metrics();

        MetricsSnapshot {
            timestamp: SystemTime::now(),
            overall,
            cache_stats: self.cache_metrics.clone(),
            memory_bytes: self.memory_tracker.current_bytes,
            active_operations: 0, // Would track active operations
        }
    }

    /// Calculate overall metrics
    fn calculate_overall_metrics(&self) -> OverallCacheMetrics {
        let total_hits: u64 = self.cache_metrics.values().map(|s| s.hits).sum();
        let total_misses: u64 = self.cache_metrics.values().map(|s| s.misses).sum();
        let total_ops = total_hits + total_misses;

        let hit_rate = if total_ops > 0 {
            total_hits as f32 / total_ops as f32
        } else {
            0.0
        };

        // Calculate time saved (estimated)
        let avg_cache_time = self.operation_timings.get_timings.avg_us;
        let avg_miss_time = avg_cache_time * 10.0; // Assume cache is 10x faster
        let time_saved_ms = (total_hits as f32 * (avg_miss_time - avg_cache_time)) / 1000.0;

        // Calculate efficiency score
        let efficiency_score = hit_rate * 0.4
            + (1.0 - self.calculate_memory_pressure()) * 0.3
            + (time_saved_ms / 1000.0).min(1.0) * 0.3;

        OverallCacheMetrics {
            memory_saved: self.memory_tracker.compression_saved_bytes
                + self.memory_tracker.deduplication_saved_bytes,
            time_saved_ms,
            efficiency_score,
            memory_pressure: self.calculate_memory_pressure(),
            ops_per_second: self.calculate_ops_per_second(),
        }
    }

    /// Calculate operations per second
    fn calculate_ops_per_second(&self) -> f32 {
        // Would calculate based on recent operations
        100.0 // Placeholder
    }

    /// Get performance report
    pub fn get_report(&self) -> PerformanceReport {
        PerformanceReport {
            summary: self.get_summary(),
            recommendations: self.generate_recommendations(),
            alerts: self.generate_alerts(),
            trends: self.analyze_trends(),
        }
    }

    /// Get summary statistics
    fn get_summary(&self) -> SummaryStats {
        let total_hits: u64 = self.cache_metrics.values().map(|s| s.hits).sum();
        let total_misses: u64 = self.cache_metrics.values().map(|s| s.misses).sum();

        SummaryStats {
            total_operations: total_hits + total_misses,
            overall_hit_rate: if total_hits + total_misses > 0 {
                total_hits as f32 / (total_hits + total_misses) as f32
            } else {
                0.0
            },
            memory_usage_mb: self.memory_tracker.current_bytes as f32 / (1024.0 * 1024.0),
            avg_latency_us: self.operation_timings.get_timings.avg_us,
            efficiency_score: self.calculate_overall_metrics().efficiency_score,
        }
    }

    /// Generate performance recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check hit rate
        let summary = self.get_summary();
        if summary.overall_hit_rate < 0.5 {
            recommendations.push(
                "Low hit rate detected. Consider increasing cache size or adjusting eviction policy.".to_string()
            );
        }

        // Check memory pressure
        if self.calculate_memory_pressure() > 0.8 {
            recommendations
                .push("High memory pressure. Enable compression or reduce cache size.".to_string());
        }

        // Check latency
        if self.operation_timings.get_timings.p99_us > 1000 {
            recommendations.push(
                "High cache latency detected. Consider optimizing data structures.".to_string(),
            );
        }

        recommendations
    }

    /// Generate alerts for issues
    fn generate_alerts(&self) -> Vec<Alert> {
        let mut alerts = Vec::new();

        if self.calculate_memory_pressure() > 0.9 {
            alerts.push(Alert {
                severity: AlertSeverity::Critical,
                message: "Critical memory pressure - cache may start dropping entries".to_string(),
                timestamp: SystemTime::now(),
            });
        }

        if self.get_summary().overall_hit_rate < 0.3 {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                message: "Very low cache hit rate - cache may not be effective".to_string(),
                timestamp: SystemTime::now(),
            });
        }

        alerts
    }

    /// Analyze performance trends
    fn analyze_trends(&self) -> TrendAnalysis {
        TrendAnalysis {
            hit_rate_trend: Trend::Stable,
            memory_trend: Trend::Increasing,
            latency_trend: Trend::Stable,
            efficiency_trend: Trend::Stable,
        }
    }
}

/// Cache operation types
#[derive(Debug, Clone)]
pub enum Operation {
    Get { hit: bool },
    Put,
    Remove,
    Evict,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub summary: SummaryStats,
    pub recommendations: Vec<String>,
    pub alerts: Vec<Alert>,
    pub trends: TrendAnalysis,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryStats {
    pub total_operations: u64,
    pub overall_hit_rate: f32,
    pub memory_usage_mb: f32,
    pub avg_latency_us: f32,
    pub efficiency_score: f32,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub hit_rate_trend: Trend,
    pub memory_trend: Trend,
    pub latency_trend: Trend,
    pub efficiency_trend: Trend,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

// Implementations

impl Default for OperationTimings {
    fn default() -> Self {
        Self {
            get_timings: TimingStats::default(),
            put_timings: TimingStats::default(),
            remove_timings: TimingStats::default(),
            eviction_timings: TimingStats::default(),
            compression_timings: TimingStats::default(),
        }
    }
}

impl Default for TimingStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_us: 0,
            avg_us: 0.0,
            min_us: u64::MAX,
            max_us: 0,
            p50_us: 0,
            p95_us: 0,
            p99_us: 0,
        }
    }
}

impl TimingStats {
    /// Record a timing measurement
    pub fn record(&mut self, duration_us: u64) {
        self.count += 1;
        self.total_us += duration_us;
        self.avg_us = self.total_us as f32 / self.count as f32;
        self.min_us = self.min_us.min(duration_us);
        self.max_us = self.max_us.max(duration_us);

        // Update percentiles (simplified - would use proper algorithm)
        self.p50_us = self.avg_us as u64;
        self.p95_us = (self.avg_us * 1.5) as u64;
        self.p99_us = (self.avg_us * 2.0) as u64;
    }
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            current_bytes: 0,
            peak_bytes: 0,
            compression_saved_bytes: 0,
            deduplication_saved_bytes: 0,
            pressure_events: Vec::new(),
        }
    }
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            hit_rate_history: Vec::new(),
            ops_history: Vec::new(),
            latency_history: Vec::new(),
            efficiency_history: Vec::new(),
        }
    }
}

impl MetricsHistory {
    pub fn new(max_size: usize, interval: Duration) -> Self {
        Self {
            snapshots: Vec::new(),
            max_size,
            interval,
            last_snapshot: SystemTime::now(),
        }
    }

    pub fn add_snapshot(&mut self, snapshot: MetricsSnapshot) {
        self.snapshots.push(snapshot);
        if self.snapshots.len() > self.max_size {
            self.snapshots.remove(0);
        }
        self.last_snapshot = SystemTime::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_stats() {
        let mut stats = TimingStats::default();

        stats.record(100);
        stats.record(200);
        stats.record(150);

        assert_eq!(stats.count, 3);
        assert_eq!(stats.avg_us, 150.0);
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 200);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        collector.cache_metrics.insert(
            "test".to_string(),
            CacheStats {
                total_entries: 100,
                hits: 80,
                misses: 20,
                hit_rate: 0.8,
                memory_usage: 1024,
                avg_access_time_us: 10.0,
                evictions: 5,
                last_cleanup: SystemTime::now(),
            },
        );

        collector.record_operation(
            "test",
            Operation::Get { hit: true },
            Duration::from_micros(10),
        );

        let report = collector.get_report();
        assert!(report.summary.overall_hit_rate > 0.0);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();

        tracker.current_bytes = 1024;
        tracker.peak_bytes = 2048;
        tracker.compression_saved_bytes = 512;

        assert_eq!(tracker.peak_bytes, 2048);
    }
}
