//! # Metrics Collection System
//! 
//! Comprehensive metrics collection and aggregation for RRAG system performance,
//! usage patterns, and operational insights.

use crate::{RragError, RragResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc, Duration};
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub collection_interval_seconds: u64,
    pub buffer_size: usize,
    pub export_interval_seconds: u64,
    pub retention_days: u32,
    pub labels: HashMap<String, String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_seconds: 10,
            buffer_size: 10000,
            export_interval_seconds: 60,
            retention_days: 30,
            labels: HashMap::new(),
        }
    }
}

/// Metric types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
    Summary,
}

/// Metric value variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram { buckets: Vec<(f64, u64)>, sum: f64, count: u64 },
    Timer { duration_ms: f64, count: u64 },
    Summary { sum: f64, count: u64, quantiles: Vec<(f64, f64)> },
}

/// Individual metric instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub value: MetricValue,
    pub labels: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub help: Option<String>,
}

impl Metric {
    pub fn counter(name: impl Into<String>, value: u64) -> Self {
        Self {
            name: name.into(),
            metric_type: MetricType::Counter,
            value: MetricValue::Counter(value),
            labels: HashMap::new(),
            timestamp: Utc::now(),
            help: None,
        }
    }

    pub fn gauge(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            metric_type: MetricType::Gauge,
            value: MetricValue::Gauge(value),
            labels: HashMap::new(),
            timestamp: Utc::now(),
            help: None,
        }
    }

    pub fn timer(name: impl Into<String>, duration_ms: f64) -> Self {
        Self {
            name: name.into(),
            metric_type: MetricType::Timer,
            value: MetricValue::Timer { duration_ms, count: 1 },
            labels: HashMap::new(),
            timestamp: Utc::now(),
            help: None,
        }
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels.extend(labels);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

/// Counter metric implementation
pub struct CounterMetric {
    name: String,
    value: AtomicU64,
    labels: HashMap<String, String>,
    help: Option<String>,
}

impl CounterMetric {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: AtomicU64::new(0),
            labels: HashMap::new(),
            help: None,
        }
    }

    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }

    pub fn to_metric(&self) -> Metric {
        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Counter,
            value: MetricValue::Counter(self.get()),
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: self.help.clone(),
        }
    }
}

/// Gauge metric implementation
pub struct GaugeMetric {
    name: String,
    value: AtomicI64, // Using i64 to support negative values with bit manipulation
    labels: HashMap<String, String>,
    help: Option<String>,
}

impl GaugeMetric {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: AtomicI64::new(0),
            labels: HashMap::new(),
            help: None,
        }
    }

    pub fn set(&self, value: f64) {
        self.value.store(value.to_bits() as i64, Ordering::Relaxed);
    }

    pub fn inc(&self) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed) as u64);
        self.set(current + 1.0);
    }

    pub fn dec(&self) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed) as u64);
        self.set(current - 1.0);
    }

    pub fn add(&self, value: f64) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed) as u64);
        self.set(current + value);
    }

    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed) as u64)
    }

    pub fn to_metric(&self) -> Metric {
        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Gauge,
            value: MetricValue::Gauge(self.get()),
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: self.help.clone(),
        }
    }
}

/// Histogram metric for tracking distributions
pub struct HistogramMetric {
    name: String,
    buckets: Vec<(f64, AtomicU64)>,
    sum: AtomicI64, // Using i64 for atomic f64
    count: AtomicU64,
    labels: HashMap<String, String>,
    help: Option<String>,
}

impl HistogramMetric {
    pub fn new(name: impl Into<String>, buckets: Vec<f64>) -> Self {
        let mut histogram_buckets = Vec::new();
        for bucket in buckets {
            histogram_buckets.push((bucket, AtomicU64::new(0)));
        }
        // Add +Inf bucket
        histogram_buckets.push((f64::INFINITY, AtomicU64::new(0)));

        Self {
            name: name.into(),
            buckets: histogram_buckets,
            sum: AtomicI64::new(0),
            count: AtomicU64::new(0),
            labels: HashMap::new(),
            help: None,
        }
    }

    pub fn observe(&self, value: f64) {
        // Update buckets
        for (le, counter) in &self.buckets {
            if value <= *le {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update sum and count
        let current_sum = f64::from_bits(self.sum.load(Ordering::Relaxed) as u64);
        self.sum.store((current_sum + value).to_bits() as i64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn to_metric(&self) -> Metric {
        let buckets: Vec<(f64, u64)> = self.buckets
            .iter()
            .map(|(le, counter)| (*le, counter.load(Ordering::Relaxed)))
            .collect();

        let sum = f64::from_bits(self.sum.load(Ordering::Relaxed) as u64);
        let count = self.count.load(Ordering::Relaxed);

        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Histogram,
            value: MetricValue::Histogram { buckets, sum, count },
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: self.help.clone(),
        }
    }
}

/// Timer metric for measuring durations
pub struct TimerMetric {
    name: String,
    total_duration_ms: AtomicI64,
    count: AtomicU64,
    labels: HashMap<String, String>,
    help: Option<String>,
}

impl TimerMetric {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            total_duration_ms: AtomicI64::new(0),
            count: AtomicU64::new(0),
            labels: HashMap::new(),
            help: None,
        }
    }

    pub fn record(&self, duration_ms: f64) {
        let current_total = f64::from_bits(self.total_duration_ms.load(Ordering::Relaxed) as u64);
        self.total_duration_ms.store(
            (current_total + duration_ms).to_bits() as i64,
            Ordering::Relaxed
        );
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn average_duration(&self) -> f64 {
        let total = f64::from_bits(self.total_duration_ms.load(Ordering::Relaxed) as u64);
        let count = self.count.load(Ordering::Relaxed);
        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    pub fn to_metric(&self) -> Metric {
        let duration_ms = self.average_duration();
        let count = self.count.load(Ordering::Relaxed);

        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Timer,
            value: MetricValue::Timer { duration_ms, count },
            labels: self.labels.clone(),
            timestamp: Utc::now(),
            help: self.help.clone(),
        }
    }
}

/// Metrics registry for managing all metrics
pub struct MetricsRegistry {
    counters: Arc<RwLock<HashMap<String, Arc<CounterMetric>>>>,
    gauges: Arc<RwLock<HashMap<String, Arc<GaugeMetric>>>>,
    histograms: Arc<RwLock<HashMap<String, Arc<HistogramMetric>>>>,
    timers: Arc<RwLock<HashMap<String, Arc<TimerMetric>>>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            timers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_or_create_counter(&self, name: &str) -> Arc<CounterMetric> {
        let counters = self.counters.read().await;
        if let Some(counter) = counters.get(name) {
            return counter.clone();
        }
        drop(counters);

        let mut counters = self.counters.write().await;
        counters.entry(name.to_string())
            .or_insert_with(|| Arc::new(CounterMetric::new(name)))
            .clone()
    }

    pub async fn get_or_create_gauge(&self, name: &str) -> Arc<GaugeMetric> {
        let gauges = self.gauges.read().await;
        if let Some(gauge) = gauges.get(name) {
            return gauge.clone();
        }
        drop(gauges);

        let mut gauges = self.gauges.write().await;
        gauges.entry(name.to_string())
            .or_insert_with(|| Arc::new(GaugeMetric::new(name)))
            .clone()
    }

    pub async fn get_or_create_histogram(&self, name: &str, buckets: Vec<f64>) -> Arc<HistogramMetric> {
        let histograms = self.histograms.read().await;
        if let Some(histogram) = histograms.get(name) {
            return histogram.clone();
        }
        drop(histograms);

        let mut histograms = self.histograms.write().await;
        histograms.entry(name.to_string())
            .or_insert_with(|| Arc::new(HistogramMetric::new(name, buckets)))
            .clone()
    }

    pub async fn get_or_create_timer(&self, name: &str) -> Arc<TimerMetric> {
        let timers = self.timers.read().await;
        if let Some(timer) = timers.get(name) {
            return timer.clone();
        }
        drop(timers);

        let mut timers = self.timers.write().await;
        timers.entry(name.to_string())
            .or_insert_with(|| Arc::new(TimerMetric::new(name)))
            .clone()
    }

    pub async fn collect_all_metrics(&self) -> Vec<Metric> {
        let mut metrics = Vec::new();

        // Collect counters
        let counters = self.counters.read().await;
        for counter in counters.values() {
            metrics.push(counter.to_metric());
        }
        drop(counters);

        // Collect gauges
        let gauges = self.gauges.read().await;
        for gauge in gauges.values() {
            metrics.push(gauge.to_metric());
        }
        drop(gauges);

        // Collect histograms
        let histograms = self.histograms.read().await;
        for histogram in histograms.values() {
            metrics.push(histogram.to_metric());
        }
        drop(histograms);

        // Collect timers
        let timers = self.timers.read().await;
        for timer in timers.values() {
            metrics.push(timer.to_metric());
        }

        metrics
    }
}

/// Main metrics collector
pub struct MetricsCollector {
    config: MetricsConfig,
    registry: Arc<MetricsRegistry>,
    buffer: Arc<RwLock<Vec<Metric>>>,
    sender: mpsc::UnboundedSender<Metric>,
    _receiver_handle: tokio::task::JoinHandle<()>,
    is_running: Arc<RwLock<bool>>,
}

impl MetricsCollector {
    pub async fn new(config: MetricsConfig) -> RragResult<Self> {
        let registry = Arc::new(MetricsRegistry::new());
        let buffer = Arc::new(RwLock::new(Vec::with_capacity(config.buffer_size)));
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let is_running = Arc::new(RwLock::new(false));

        let buffer_clone = buffer.clone();
        let is_running_clone = is_running.clone();
        let buffer_size = config.buffer_size;

        let receiver_handle = tokio::spawn(async move {
            while let Some(metric) = receiver.recv().await {
                if !*is_running_clone.read().await {
                    break;
                }

                let mut buffer = buffer_clone.write().await;
                buffer.push(metric);

                // Prevent buffer overflow
                if buffer.len() >= buffer_size {
                    buffer.drain(0..buffer_size / 2);
                }
            }
        });

        Ok(Self {
            config,
            registry,
            buffer,
            sender,
            _receiver_handle: receiver_handle,
            is_running,
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("metrics", "stopped", "already running"));
        }
        *running = true;
        tracing::info!("Metrics collector started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }
        *running = false;
        tracing::info!("Metrics collector stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    pub async fn record_metric(&self, metric: Metric) -> RragResult<()> {
        if !*self.is_running.read().await {
            return Err(RragError::config("metrics", "running", "stopped"));
        }

        self.sender.send(metric)
            .map_err(|e| RragError::agent("metrics", e.to_string()))?;
        
        Ok(())
    }

    pub async fn inc_counter(&self, name: &str) -> RragResult<()> {
        let counter = self.registry.get_or_create_counter(name).await;
        counter.inc();
        Ok(())
    }

    pub async fn inc_counter_by(&self, name: &str, value: u64) -> RragResult<()> {
        let counter = self.registry.get_or_create_counter(name).await;
        counter.inc_by(value);
        Ok(())
    }

    pub async fn set_gauge(&self, name: &str, value: f64) -> RragResult<()> {
        let gauge = self.registry.get_or_create_gauge(name).await;
        gauge.set(value);
        Ok(())
    }

    pub async fn observe_histogram(&self, name: &str, value: f64, buckets: Option<Vec<f64>>) -> RragResult<()> {
        let default_buckets = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];
        let histogram = self.registry.get_or_create_histogram(
            name, 
            buckets.unwrap_or(default_buckets)
        ).await;
        histogram.observe(value);
        Ok(())
    }

    pub async fn record_timer(&self, name: &str, duration_ms: f64) -> RragResult<()> {
        let timer = self.registry.get_or_create_timer(name).await;
        timer.record(duration_ms);
        Ok(())
    }

    pub async fn get_all_metrics(&self) -> Vec<Metric> {
        let registry_metrics = self.registry.collect_all_metrics().await;
        let buffer_metrics = self.buffer.read().await.clone();
        
        let mut all_metrics = registry_metrics;
        all_metrics.extend(buffer_metrics);
        all_metrics
    }

    pub async fn get_metrics_count(&self) -> usize {
        self.buffer.read().await.len()
    }

    pub async fn clear_buffer(&self) -> Vec<Metric> {
        let mut buffer = self.buffer.write().await;
        let metrics = buffer.clone();
        buffer.clear();
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_counter_metric() {
        let counter = CounterMetric::new("test_counter");
        
        assert_eq!(counter.get(), 0);
        
        counter.inc();
        assert_eq!(counter.get(), 1);
        
        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
        
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[tokio::test]
    async fn test_gauge_metric() {
        let gauge = GaugeMetric::new("test_gauge");
        
        assert_eq!(gauge.get(), 0.0);
        
        gauge.set(10.5);
        assert_eq!(gauge.get(), 10.5);
        
        gauge.inc();
        assert_eq!(gauge.get(), 11.5);
        
        gauge.dec();
        assert_eq!(gauge.get(), 10.5);
        
        gauge.add(-5.0);
        assert_eq!(gauge.get(), 5.5);
    }

    #[tokio::test]
    async fn test_histogram_metric() {
        let histogram = HistogramMetric::new("test_histogram", vec![1.0, 5.0, 10.0]);
        
        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);
        histogram.observe(15.0);
        
        let metric = histogram.to_metric();
        if let MetricValue::Histogram { buckets, sum, count } = metric.value {
            assert_eq!(count, 4);
            assert_eq!(sum, 25.5);
            
            // Check bucket counts
            assert_eq!(buckets[0], (1.0, 1)); // 0.5 <= 1.0
            assert_eq!(buckets[1], (5.0, 2)); // 0.5, 3.0 <= 5.0
            assert_eq!(buckets[2], (10.0, 3)); // 0.5, 3.0, 7.0 <= 10.0
            assert_eq!(buckets[3], (f64::INFINITY, 4)); // All values <= +Inf
        }
    }

    #[tokio::test]
    async fn test_timer_metric() {
        let timer = TimerMetric::new("test_timer");
        
        timer.record(100.0);
        timer.record(200.0);
        timer.record(300.0);
        
        assert_eq!(timer.average_duration(), 200.0);
        
        let metric = timer.to_metric();
        if let MetricValue::Timer { duration_ms, count } = metric.value {
            assert_eq!(duration_ms, 200.0);
            assert_eq!(count, 3);
        }
    }

    #[tokio::test]
    async fn test_metrics_registry() {
        let registry = MetricsRegistry::new();
        
        // Test counter
        let counter = registry.get_or_create_counter("test_counter").await;
        counter.inc();
        
        // Test gauge
        let gauge = registry.get_or_create_gauge("test_gauge").await;
        gauge.set(42.0);
        
        // Collect all metrics
        let metrics = registry.collect_all_metrics().await;
        assert_eq!(metrics.len(), 2);
        
        // Verify metrics
        let counter_metric = metrics.iter().find(|m| m.name == "test_counter").unwrap();
        assert_eq!(counter_metric.metric_type, MetricType::Counter);
        if let MetricValue::Counter(value) = counter_metric.value {
            assert_eq!(value, 1);
        }
        
        let gauge_metric = metrics.iter().find(|m| m.name == "test_gauge").unwrap();
        assert_eq!(gauge_metric.metric_type, MetricType::Gauge);
        if let MetricValue::Gauge(value) = gauge_metric.value {
            assert_eq!(value, 42.0);
        }
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config).await.unwrap();
        
        collector.start().await.unwrap();
        assert!(collector.is_healthy().await);
        
        // Test counter operations
        collector.inc_counter("requests_total").await.unwrap();
        collector.inc_counter_by("requests_total", 5).await.unwrap();
        
        // Test gauge operations
        collector.set_gauge("active_connections", 10.0).await.unwrap();
        
        // Test histogram operations
        collector.observe_histogram("request_duration", 0.5, None).await.unwrap();
        
        // Test timer operations
        collector.record_timer("process_time", 150.0).await.unwrap();
        
        let metrics = collector.get_all_metrics().await;
        assert!(!metrics.is_empty());
        
        collector.stop().await.unwrap();
        assert!(!collector.is_healthy().await);
    }

    #[test]
    fn test_metric_creation() {
        let counter = Metric::counter("test_counter", 10);
        assert_eq!(counter.name, "test_counter");
        assert_eq!(counter.metric_type, MetricType::Counter);
        if let MetricValue::Counter(value) = counter.value {
            assert_eq!(value, 10);
        }
        
        let gauge = Metric::gauge("test_gauge", 42.5)
            .with_label("host", "server1")
            .with_help("Test gauge metric");
        
        assert_eq!(gauge.name, "test_gauge");
        assert_eq!(gauge.metric_type, MetricType::Gauge);
        assert!(gauge.labels.contains_key("host"));
        assert_eq!(gauge.labels["host"], "server1");
        assert_eq!(gauge.help.as_ref().unwrap(), "Test gauge metric");
    }
}