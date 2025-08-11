//! # Performance Profiling System
//!
//! Advanced profiling capabilities for identifying bottlenecks,
//! performance trends, and optimization opportunities in RRAG systems.

use crate::{RragError, RragResult};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub enabled: bool,
    pub sample_rate: f64,
    pub max_profiles: usize,
    pub profile_duration_seconds: u64,
    pub enable_cpu_profiling: bool,
    pub enable_memory_profiling: bool,
    pub enable_io_profiling: bool,
    pub enable_custom_metrics: bool,
    pub bottleneck_threshold_ms: f64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 0.1, // 10% sampling
            max_profiles: 1000,
            profile_duration_seconds: 60,
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_io_profiling: true,
            enable_custom_metrics: true,
            bottleneck_threshold_ms: 100.0,
        }
    }
}

/// Profile data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileData {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: f64,
    pub operation: String,
    pub component: String,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub trace_id: Option<String>,
    pub cpu_usage_percent: Option<f64>,
    pub memory_usage_mb: Option<f64>,
    pub io_read_bytes: Option<u64>,
    pub io_write_bytes: Option<u64>,
    pub custom_metrics: HashMap<String, f64>,
    pub stack_trace: Option<String>,
    pub tags: HashMap<String, String>,
}

impl ProfileData {
    pub fn new(operation: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            duration_ms: 0.0,
            operation: operation.into(),
            component: component.into(),
            user_id: None,
            session_id: None,
            trace_id: None,
            cpu_usage_percent: None,
            memory_usage_mb: None,
            io_read_bytes: None,
            io_write_bytes: None,
            custom_metrics: HashMap::new(),
            stack_trace: None,
            tags: HashMap::new(),
        }
    }

    pub fn with_duration(mut self, duration_ms: f64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_trace(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    pub fn with_cpu_usage(mut self, cpu_percent: f64) -> Self {
        self.cpu_usage_percent = Some(cpu_percent);
        self
    }

    pub fn with_memory_usage(mut self, memory_mb: f64) -> Self {
        self.memory_usage_mb = Some(memory_mb);
        self
    }

    pub fn with_io(mut self, read_bytes: u64, write_bytes: u64) -> Self {
        self.io_read_bytes = Some(read_bytes);
        self.io_write_bytes = Some(write_bytes);
        self
    }

    pub fn with_custom_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.custom_metrics.insert(name.into(), value);
        self
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn with_stack_trace(mut self, trace: impl Into<String>) -> Self {
        self.stack_trace = Some(trace.into());
        self
    }
}

/// Bottleneck analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub timestamp: DateTime<Utc>,
    pub analysis_period_minutes: u32,
    pub bottlenecks: Vec<Bottleneck>,
    pub performance_trends: Vec<PerformanceTrend>,
    pub recommendations: Vec<OptimizationRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub component: String,
    pub operation: String,
    pub average_duration_ms: f64,
    pub max_duration_ms: f64,
    pub occurrence_count: usize,
    pub impact_score: f64,
    pub bottleneck_type: BottleneckType,
    pub affected_users: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Database,
    Cache,
    Algorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub component: String,
    pub trend_direction: TrendDirection,
    pub change_rate_percent: f64,
    pub significance: TrendSignificance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendSignificance {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub component: String,
    pub operation: String,
    pub recommendation: String,
    pub priority: RecommendationPriority,
    pub estimated_improvement_percent: f64,
    pub implementation_effort: ImplementationEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: DateTime<Utc>,
    pub report_period: Duration,
    pub summary: PerformanceSummary,
    pub component_performance: HashMap<String, ComponentPerformanceMetrics>,
    pub operation_performance: HashMap<String, OperationPerformanceMetrics>,
    pub user_experience_metrics: UserExperienceMetrics,
    pub resource_utilization: ResourceUtilizationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_operations: usize,
    pub average_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub throughput_ops_per_second: f64,
    pub bottlenecks_detected: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentPerformanceMetrics {
    pub component_name: String,
    pub operation_count: usize,
    pub average_duration_ms: f64,
    pub max_duration_ms: f64,
    pub min_duration_ms: f64,
    pub standard_deviation_ms: f64,
    pub error_count: usize,
    pub cpu_usage_percent: Option<f64>,
    pub memory_usage_mb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPerformanceMetrics {
    pub operation_name: String,
    pub invocation_count: usize,
    pub average_duration_ms: f64,
    pub percentiles: HashMap<u32, f64>, // P50, P95, P99, etc.
    pub concurrent_executions_max: usize,
    pub success_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserExperienceMetrics {
    pub average_session_duration_minutes: f64,
    pub bounce_rate_percent: f64,
    pub user_satisfaction_score: Option<f64>,
    pub conversion_rate_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    pub cpu_utilization_percent: f64,
    pub memory_utilization_percent: f64,
    pub disk_utilization_percent: f64,
    pub network_utilization_percent: f64,
    pub connection_pool_utilization_percent: f64,
}

/// Performance profiler
pub struct PerformanceProfiler {
    config: ProfilingConfig,
    profiles: Arc<RwLock<Vec<ProfileData>>>,
    active_profiles: Arc<RwLock<HashMap<String, std::time::Instant>>>,
    is_running: Arc<RwLock<bool>>,
}

impl PerformanceProfiler {
    pub async fn new(config: ProfilingConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            profiles: Arc::new(RwLock::new(Vec::new())),
            active_profiles: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn start(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if *running {
            return Err(RragError::config("profiler", "stopped", "already running"));
        }
        *running = true;
        tracing::info!("Performance profiler started");
        Ok(())
    }

    pub async fn stop(&self) -> RragResult<()> {
        let mut running = self.is_running.write().await;
        if !*running {
            return Ok(());
        }
        *running = false;
        tracing::info!("Performance profiler stopped");
        Ok(())
    }

    pub async fn is_healthy(&self) -> bool {
        *self.is_running.read().await
    }

    /// Start profiling an operation
    pub async fn start_profile(&self, operation_id: impl Into<String>) -> RragResult<()> {
        if !*self.is_running.read().await {
            return Ok(()); // Silently ignore if profiler is disabled
        }

        // Check sampling rate
        if rand::random::<f64>() > self.config.sample_rate {
            return Ok(()); // Skip this profile based on sampling
        }

        let mut active = self.active_profiles.write().await;
        active.insert(operation_id.into(), std::time::Instant::now());
        Ok(())
    }

    /// End profiling an operation and record the profile
    pub async fn end_profile(
        &self,
        operation_id: impl Into<String>,
        operation: impl Into<String>,
        component: impl Into<String>,
    ) -> RragResult<Option<ProfileData>> {
        if !*self.is_running.read().await {
            return Ok(None);
        }

        let operation_id = operation_id.into();
        let mut active = self.active_profiles.write().await;

        if let Some(start_time) = active.remove(&operation_id) {
            let duration_ms = start_time.elapsed().as_millis() as f64;
            drop(active);

            let mut profile = ProfileData::new(operation, component)
                .with_duration(duration_ms)
                .with_trace(operation_id);

            // Collect system metrics if enabled
            if self.config.enable_cpu_profiling || self.config.enable_memory_profiling {
                // In a real implementation, this would collect actual system metrics
                profile = profile
                    .with_cpu_usage(rand::random::<f64>() * 100.0)
                    .with_memory_usage(rand::random::<f64>() * 1024.0);
            }

            // Store profile
            let mut profiles = self.profiles.write().await;
            profiles.push(profile.clone());

            // Keep only recent profiles
            let profiles_len = profiles.len();
            if profiles_len > self.config.max_profiles {
                profiles.drain(0..profiles_len - self.config.max_profiles);
            }

            Ok(Some(profile))
        } else {
            Ok(None)
        }
    }

    /// Record a complete profile data
    pub async fn record_profile(&self, profile: ProfileData) -> RragResult<()> {
        if !*self.is_running.read().await {
            return Ok(());
        }

        let mut profiles = self.profiles.write().await;
        profiles.push(profile);

        // Keep only recent profiles
        let profiles_len = profiles.len();
        if profiles_len > self.config.max_profiles {
            profiles.drain(0..profiles_len - self.config.max_profiles);
        }

        Ok(())
    }

    /// Analyze bottlenecks in the collected profiles
    pub async fn analyze_bottlenecks(&self, analysis_period_minutes: u32) -> BottleneckAnalysis {
        let profiles = self.profiles.read().await;
        let cutoff_time = Utc::now() - Duration::minutes(analysis_period_minutes as i64);

        let recent_profiles: Vec<_> = profiles
            .iter()
            .filter(|p| p.timestamp >= cutoff_time)
            .collect();

        if recent_profiles.is_empty() {
            return BottleneckAnalysis {
                timestamp: Utc::now(),
                analysis_period_minutes,
                bottlenecks: Vec::new(),
                performance_trends: Vec::new(),
                recommendations: Vec::new(),
            };
        }

        let bottlenecks = self.identify_bottlenecks(&recent_profiles);
        let trends = self.analyze_performance_trends(&recent_profiles);
        let recommendations = self.generate_recommendations(&bottlenecks);

        BottleneckAnalysis {
            timestamp: Utc::now(),
            analysis_period_minutes,
            bottlenecks,
            performance_trends: trends,
            recommendations,
        }
    }

    fn identify_bottlenecks(&self, profiles: &[&ProfileData]) -> Vec<Bottleneck> {
        let mut component_operations: HashMap<String, Vec<f64>> = HashMap::new();
        let mut user_impact: HashMap<String, std::collections::HashSet<String>> = HashMap::new();

        for profile in profiles {
            let key = format!("{}:{}", profile.component, profile.operation);
            component_operations
                .entry(key.clone())
                .or_default()
                .push(profile.duration_ms);

            if let Some(ref user_id) = profile.user_id {
                user_impact.entry(key).or_default().insert(user_id.clone());
            }
        }

        let mut bottlenecks = Vec::new();

        for (key, durations) in component_operations {
            let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
            let max_duration = durations.iter().fold(0.0f64, |a, &b| a.max(b));

            if avg_duration > self.config.bottleneck_threshold_ms {
                let parts: Vec<&str> = key.split(':').collect();
                let component = parts[0].to_string();
                let operation = parts[1].to_string();

                let impact_score =
                    self.calculate_impact_score(avg_duration, durations.len(), max_duration);
                let bottleneck_type = self.determine_bottleneck_type(&component, avg_duration);
                let affected_users = user_impact.get(&key).map(|set| set.len()).unwrap_or(0);

                bottlenecks.push(Bottleneck {
                    component,
                    operation,
                    average_duration_ms: avg_duration,
                    max_duration_ms: max_duration,
                    occurrence_count: durations.len(),
                    impact_score,
                    bottleneck_type,
                    affected_users,
                });
            }
        }

        // Sort by impact score
        bottlenecks.sort_by(|a, b| {
            b.impact_score
                .partial_cmp(&a.impact_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bottlenecks
    }

    fn calculate_impact_score(
        &self,
        avg_duration: f64,
        frequency: usize,
        max_duration: f64,
    ) -> f64 {
        // Impact score considers duration, frequency, and peak impact
        let duration_weight = avg_duration / 1000.0; // Convert to seconds
        let frequency_weight = (frequency as f64).log10();
        let peak_weight = max_duration / avg_duration;

        duration_weight * frequency_weight * peak_weight
    }

    fn determine_bottleneck_type(&self, component: &str, avg_duration: f64) -> BottleneckType {
        match component.to_lowercase().as_str() {
            s if s.contains("database") || s.contains("db") => BottleneckType::Database,
            s if s.contains("cache") => BottleneckType::Cache,
            s if s.contains("network") || s.contains("http") => BottleneckType::Network,
            s if s.contains("io") || s.contains("file") => BottleneckType::IO,
            _ => {
                if avg_duration > 1000.0 {
                    BottleneckType::Algorithm
                } else if avg_duration > 500.0 {
                    BottleneckType::CPU
                } else {
                    BottleneckType::Memory
                }
            }
        }
    }

    fn analyze_performance_trends(&self, _profiles: &[&ProfileData]) -> Vec<PerformanceTrend> {
        // This would implement sophisticated trend analysis
        // For now, return a simple mock trend
        vec![PerformanceTrend {
            metric_name: "response_time".to_string(),
            component: "search".to_string(),
            trend_direction: TrendDirection::Stable,
            change_rate_percent: -2.5,
            significance: TrendSignificance::Low,
        }]
    }

    fn generate_recommendations(
        &self,
        bottlenecks: &[Bottleneck],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for bottleneck in bottlenecks.iter().take(5) {
            // Top 5 bottlenecks
            let recommendation = match bottleneck.bottleneck_type {
                BottleneckType::Database => OptimizationRecommendation {
                    component: bottleneck.component.clone(),
                    operation: bottleneck.operation.clone(),
                    recommendation: "Consider adding database indexes or optimizing queries".to_string(),
                    priority: if bottleneck.impact_score > 10.0 { RecommendationPriority::High } else { RecommendationPriority::Medium },
                    estimated_improvement_percent: 40.0,
                    implementation_effort: ImplementationEffort::Medium,
                },
                BottleneckType::Cache => OptimizationRecommendation {
                    component: bottleneck.component.clone(),
                    operation: bottleneck.operation.clone(),
                    recommendation: "Implement caching strategy or increase cache size".to_string(),
                    priority: RecommendationPriority::Medium,
                    estimated_improvement_percent: 60.0,
                    implementation_effort: ImplementationEffort::Low,
                },
                BottleneckType::Algorithm => OptimizationRecommendation {
                    component: bottleneck.component.clone(),
                    operation: bottleneck.operation.clone(),
                    recommendation: "Review algorithm complexity and optimize data structures".to_string(),
                    priority: RecommendationPriority::High,
                    estimated_improvement_percent: 70.0,
                    implementation_effort: ImplementationEffort::High,
                },
                _ => OptimizationRecommendation {
                    component: bottleneck.component.clone(),
                    operation: bottleneck.operation.clone(),
                    recommendation: "Profile detailed resource usage to identify specific optimization opportunities".to_string(),
                    priority: RecommendationPriority::Low,
                    estimated_improvement_percent: 25.0,
                    implementation_effort: ImplementationEffort::Medium,
                },
            };

            recommendations.push(recommendation);
        }

        recommendations
    }

    /// Generate comprehensive performance report
    pub async fn generate_performance_report(&self, period: Duration) -> PerformanceReport {
        let profiles = self.profiles.read().await;
        let cutoff_time = Utc::now() - period;

        let recent_profiles: Vec<_> = profiles
            .iter()
            .filter(|p| p.timestamp >= cutoff_time)
            .collect();

        if recent_profiles.is_empty() {
            return PerformanceReport {
                timestamp: Utc::now(),
                report_period: period,
                summary: PerformanceSummary {
                    total_operations: 0,
                    average_response_time_ms: 0.0,
                    p95_response_time_ms: 0.0,
                    p99_response_time_ms: 0.0,
                    error_rate_percent: 0.0,
                    throughput_ops_per_second: 0.0,
                    bottlenecks_detected: 0,
                },
                component_performance: HashMap::new(),
                operation_performance: HashMap::new(),
                user_experience_metrics: UserExperienceMetrics {
                    average_session_duration_minutes: 0.0,
                    bounce_rate_percent: 0.0,
                    user_satisfaction_score: None,
                    conversion_rate_percent: None,
                },
                resource_utilization: ResourceUtilizationMetrics {
                    cpu_utilization_percent: 0.0,
                    memory_utilization_percent: 0.0,
                    disk_utilization_percent: 0.0,
                    network_utilization_percent: 0.0,
                    connection_pool_utilization_percent: 0.0,
                },
            };
        }

        let summary = self.calculate_performance_summary(&recent_profiles);
        let component_performance = self.calculate_component_performance(&recent_profiles);
        let operation_performance = self.calculate_operation_performance(&recent_profiles);
        let user_experience = self.calculate_user_experience_metrics(&recent_profiles);
        let resource_utilization = self.calculate_resource_utilization(&recent_profiles);

        PerformanceReport {
            timestamp: Utc::now(),
            report_period: period,
            summary,
            component_performance,
            operation_performance,
            user_experience_metrics: user_experience,
            resource_utilization,
        }
    }

    fn calculate_performance_summary(&self, profiles: &[&ProfileData]) -> PerformanceSummary {
        let durations: Vec<f64> = profiles.iter().map(|p| p.duration_ms).collect();
        let total_operations = durations.len();

        if durations.is_empty() {
            return PerformanceSummary {
                total_operations: 0,
                average_response_time_ms: 0.0,
                p95_response_time_ms: 0.0,
                p99_response_time_ms: 0.0,
                error_rate_percent: 0.0,
                throughput_ops_per_second: 0.0,
                bottlenecks_detected: 0,
            };
        }

        let average_response_time_ms = durations.iter().sum::<f64>() / durations.len() as f64;

        let mut sorted_durations = durations.clone();
        sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p95_index = (durations.len() as f64 * 0.95) as usize;
        let p99_index = (durations.len() as f64 * 0.99) as usize;

        let p95_response_time_ms = sorted_durations.get(p95_index).copied().unwrap_or(0.0);
        let p99_response_time_ms = sorted_durations.get(p99_index).copied().unwrap_or(0.0);

        // Mock error rate and throughput calculations
        let error_rate_percent = 2.0; // Would be calculated from actual error data
        let throughput_ops_per_second = total_operations as f64 / 60.0; // Assuming 1-minute window

        let bottlenecks_detected = durations
            .iter()
            .filter(|&&d| d > self.config.bottleneck_threshold_ms)
            .count();

        PerformanceSummary {
            total_operations,
            average_response_time_ms,
            p95_response_time_ms,
            p99_response_time_ms,
            error_rate_percent,
            throughput_ops_per_second,
            bottlenecks_detected,
        }
    }

    fn calculate_component_performance(
        &self,
        profiles: &[&ProfileData],
    ) -> HashMap<String, ComponentPerformanceMetrics> {
        let mut component_data: HashMap<String, Vec<f64>> = HashMap::new();
        let error_counts: HashMap<String, usize> = HashMap::new();

        for profile in profiles {
            component_data
                .entry(profile.component.clone())
                .or_default()
                .push(profile.duration_ms);
        }

        let mut result = HashMap::new();

        for (component_name, durations) in component_data {
            let operation_count = durations.len();
            let average_duration_ms = durations.iter().sum::<f64>() / durations.len() as f64;
            let max_duration_ms = durations.iter().fold(0.0f64, |a, &b| a.max(b));
            let min_duration_ms = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            let variance = durations
                .iter()
                .map(|&d| (d - average_duration_ms).powi(2))
                .sum::<f64>()
                / durations.len() as f64;
            let standard_deviation_ms = variance.sqrt();

            let error_count = error_counts.get(&component_name).copied().unwrap_or(0);

            result.insert(
                component_name.clone(),
                ComponentPerformanceMetrics {
                    component_name,
                    operation_count,
                    average_duration_ms,
                    max_duration_ms,
                    min_duration_ms,
                    standard_deviation_ms,
                    error_count,
                    cpu_usage_percent: Some(rand::random::<f64>() * 100.0), // Mock data
                    memory_usage_mb: Some(rand::random::<f64>() * 1024.0),  // Mock data
                },
            );
        }

        result
    }

    fn calculate_operation_performance(
        &self,
        profiles: &[&ProfileData],
    ) -> HashMap<String, OperationPerformanceMetrics> {
        let mut operation_data: HashMap<String, Vec<f64>> = HashMap::new();

        for profile in profiles {
            operation_data
                .entry(profile.operation.clone())
                .or_default()
                .push(profile.duration_ms);
        }

        let mut result = HashMap::new();

        for (operation_name, durations) in operation_data {
            let invocation_count = durations.len();
            let average_duration_ms = durations.iter().sum::<f64>() / durations.len() as f64;

            let mut sorted_durations = durations.clone();
            sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mut percentiles = HashMap::new();
            percentiles.insert(
                50,
                sorted_durations[(durations.len() * 50 / 100).min(durations.len() - 1)],
            );
            percentiles.insert(
                95,
                sorted_durations[(durations.len() * 95 / 100).min(durations.len() - 1)],
            );
            percentiles.insert(
                99,
                sorted_durations[(durations.len() * 99 / 100).min(durations.len() - 1)],
            );

            result.insert(
                operation_name.clone(),
                OperationPerformanceMetrics {
                    operation_name,
                    invocation_count,
                    average_duration_ms,
                    percentiles,
                    concurrent_executions_max: 10, // Mock data
                    success_rate_percent: 98.5,    // Mock data
                },
            );
        }

        result
    }

    fn calculate_user_experience_metrics(
        &self,
        _profiles: &[&ProfileData],
    ) -> UserExperienceMetrics {
        UserExperienceMetrics {
            average_session_duration_minutes: 15.5,
            bounce_rate_percent: 12.3,
            user_satisfaction_score: Some(4.2),
            conversion_rate_percent: Some(3.1),
        }
    }

    fn calculate_resource_utilization(
        &self,
        profiles: &[&ProfileData],
    ) -> ResourceUtilizationMetrics {
        let cpu_values: Vec<f64> = profiles
            .iter()
            .filter_map(|p| p.cpu_usage_percent)
            .collect();
        let memory_values: Vec<f64> = profiles.iter().filter_map(|p| p.memory_usage_mb).collect();

        let cpu_utilization = if !cpu_values.is_empty() {
            cpu_values.iter().sum::<f64>() / cpu_values.len() as f64
        } else {
            45.0 // Mock data
        };

        let memory_utilization = if !memory_values.is_empty() {
            (memory_values.iter().sum::<f64>() / memory_values.len() as f64) / 1024.0 * 100.0
        // Convert to percentage
        } else {
            62.0 // Mock data
        };

        ResourceUtilizationMetrics {
            cpu_utilization_percent: cpu_utilization,
            memory_utilization_percent: memory_utilization,
            disk_utilization_percent: 23.0,            // Mock data
            network_utilization_percent: 15.0,         // Mock data
            connection_pool_utilization_percent: 78.0, // Mock data
        }
    }

    pub async fn clear_profiles(&self) -> RragResult<()> {
        let mut profiles = self.profiles.write().await;
        profiles.clear();
        Ok(())
    }

    pub async fn get_profile_count(&self) -> usize {
        self.profiles.read().await.len()
    }
}

/// Profiler trait for custom profiling implementations
#[async_trait::async_trait]
pub trait Profiler: Send + Sync {
    async fn start_profile(&self, operation_id: &str) -> RragResult<()>;
    async fn end_profile(
        &self,
        operation_id: &str,
        operation: &str,
        component: &str,
    ) -> RragResult<Option<ProfileData>>;
    async fn record_profile(&self, profile: ProfileData) -> RragResult<()>;
    async fn analyze_bottlenecks(&self, period_minutes: u32) -> BottleneckAnalysis;
    async fn generate_report(&self, period: Duration) -> PerformanceReport;
}

#[async_trait::async_trait]
impl Profiler for PerformanceProfiler {
    async fn start_profile(&self, operation_id: &str) -> RragResult<()> {
        self.start_profile(operation_id).await
    }

    async fn end_profile(
        &self,
        operation_id: &str,
        operation: &str,
        component: &str,
    ) -> RragResult<Option<ProfileData>> {
        self.end_profile(operation_id, operation, component).await
    }

    async fn record_profile(&self, profile: ProfileData) -> RragResult<()> {
        self.record_profile(profile).await
    }

    async fn analyze_bottlenecks(&self, period_minutes: u32) -> BottleneckAnalysis {
        self.analyze_bottlenecks(period_minutes).await
    }

    async fn generate_report(&self, period: Duration) -> PerformanceReport {
        self.generate_performance_report(period).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profile_data_creation() {
        let profile = ProfileData::new("search_query", "search_engine")
            .with_duration(150.5)
            .with_user("user123")
            .with_session("session456")
            .with_trace("trace789")
            .with_cpu_usage(45.2)
            .with_memory_usage(512.0)
            .with_io(1024, 2048)
            .with_custom_metric("cache_hits", 85.0)
            .with_tag("priority", "high");

        assert_eq!(profile.operation, "search_query");
        assert_eq!(profile.component, "search_engine");
        assert_eq!(profile.duration_ms, 150.5);
        assert_eq!(profile.user_id.as_ref().unwrap(), "user123");
        assert_eq!(profile.cpu_usage_percent.unwrap(), 45.2);
        assert_eq!(profile.io_read_bytes.unwrap(), 1024);
        assert!(profile.custom_metrics.contains_key("cache_hits"));
        assert!(profile.tags.contains_key("priority"));
    }

    #[tokio::test]
    async fn test_performance_profiler() {
        let config = ProfilingConfig {
            sample_rate: 1.0, // 100% sampling for testing
            max_profiles: 100,
            ..Default::default()
        };

        let profiler = PerformanceProfiler::new(config).await.unwrap();

        assert!(!profiler.is_healthy().await);

        profiler.start().await.unwrap();
        assert!(profiler.is_healthy().await);

        // Test operation profiling
        profiler.start_profile("op1").await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let profile = profiler
            .end_profile("op1", "test_operation", "test_component")
            .await
            .unwrap();

        assert!(profile.is_some());
        let profile = profile.unwrap();
        assert_eq!(profile.operation, "test_operation");
        assert_eq!(profile.component, "test_component");
        assert!(profile.duration_ms > 0.0);

        assert_eq!(profiler.get_profile_count().await, 1);

        profiler.stop().await.unwrap();
        assert!(!profiler.is_healthy().await);
    }

    #[tokio::test]
    async fn test_bottleneck_analysis() {
        let config = ProfilingConfig {
            sample_rate: 1.0,
            bottleneck_threshold_ms: 50.0,
            ..Default::default()
        };

        let profiler = PerformanceProfiler::new(config).await.unwrap();
        profiler.start().await.unwrap();

        // Add some profiles with different performance characteristics
        let profiles = vec![
            ProfileData::new("fast_operation", "component1").with_duration(25.0),
            ProfileData::new("slow_operation", "component1")
                .with_duration(150.0)
                .with_user("user1"),
            ProfileData::new("slow_operation", "component1")
                .with_duration(200.0)
                .with_user("user2"),
            ProfileData::new("moderate_operation", "component2").with_duration(75.0),
        ];

        for profile in profiles {
            profiler.record_profile(profile).await.unwrap();
        }

        let analysis = profiler.analyze_bottlenecks(60).await;

        assert!(!analysis.bottlenecks.is_empty());
        let bottleneck = &analysis.bottlenecks[0];
        assert_eq!(bottleneck.operation, "slow_operation");
        assert!(bottleneck.average_duration_ms > 50.0);
        assert_eq!(bottleneck.affected_users, 2);

        assert!(!analysis.recommendations.is_empty());

        profiler.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_performance_report() {
        let config = ProfilingConfig::default();
        let profiler = PerformanceProfiler::new(config).await.unwrap();
        profiler.start().await.unwrap();

        // Add sample profiles
        let profiles = vec![
            ProfileData::new("operation1", "component1").with_duration(100.0),
            ProfileData::new("operation1", "component1").with_duration(120.0),
            ProfileData::new("operation2", "component2").with_duration(50.0),
            ProfileData::new("operation2", "component2").with_duration(60.0),
        ];

        for profile in profiles {
            profiler.record_profile(profile).await.unwrap();
        }

        let report = profiler
            .generate_performance_report(Duration::hours(1))
            .await;

        assert_eq!(report.summary.total_operations, 4);
        assert!(report.summary.average_response_time_ms > 0.0);
        assert!(!report.component_performance.is_empty());
        assert!(!report.operation_performance.is_empty());

        assert!(report.component_performance.contains_key("component1"));
        assert!(report.component_performance.contains_key("component2"));

        profiler.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_sampling_rate() {
        let config = ProfilingConfig {
            sample_rate: 0.0, // No sampling
            ..Default::default()
        };

        let profiler = PerformanceProfiler::new(config).await.unwrap();
        profiler.start().await.unwrap();

        // These operations should be ignored due to 0% sampling rate
        for i in 0..10 {
            profiler.start_profile(&format!("op{}", i)).await.unwrap();
            let profile = profiler
                .end_profile(&format!("op{}", i), "test", "component")
                .await
                .unwrap();
            // Due to 0% sampling, profiles should not be recorded
            assert!(profile.is_none());
        }

        assert_eq!(profiler.get_profile_count().await, 0);

        profiler.stop().await.unwrap();
    }

    #[test]
    fn test_bottleneck_types() {
        let config = ProfilingConfig::default();
        let profiler = futures::executor::block_on(PerformanceProfiler::new(config)).unwrap();

        assert!(matches!(
            profiler.determine_bottleneck_type("database_service", 500.0),
            BottleneckType::Database
        ));

        assert!(matches!(
            profiler.determine_bottleneck_type("cache_manager", 100.0),
            BottleneckType::Cache
        ));

        assert!(matches!(
            profiler.determine_bottleneck_type("network_handler", 200.0),
            BottleneckType::Network
        ));

        assert!(matches!(
            profiler.determine_bottleneck_type("regular_service", 1500.0),
            BottleneckType::Algorithm
        ));
    }
}
