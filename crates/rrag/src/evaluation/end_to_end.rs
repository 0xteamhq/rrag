//! # End-to-End Evaluation Module
//!
//! Comprehensive evaluation that considers the entire RAG pipeline
//! including user experience, system performance, and holistic quality.

use super::{
    EvaluationData, EvaluationMetadata, EvaluationResult, EvaluationSummary, Evaluator,
    EvaluatorConfig, EvaluatorPerformance, PerformanceStats,
};
use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// End-to-end evaluator
pub struct EndToEndEvaluator {
    config: EndToEndConfig,
    metrics: Vec<Box<dyn E2EMetric>>,
}

/// Configuration for end-to-end evaluation
#[derive(Debug, Clone)]
pub struct EndToEndConfig {
    /// Enabled metrics
    pub enabled_metrics: Vec<E2EMetricType>,

    /// User experience weight
    pub user_experience_weight: f32,

    /// System performance weight
    pub system_performance_weight: f32,

    /// Quality weight
    pub quality_weight: f32,

    /// Robustness weight
    pub robustness_weight: f32,

    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,

    /// User satisfaction config
    pub user_satisfaction_config: UserSatisfactionConfig,

    /// System reliability config
    pub system_reliability_config: SystemReliabilityConfig,
}

impl Default for EndToEndConfig {
    fn default() -> Self {
        Self {
            enabled_metrics: vec![
                E2EMetricType::UserSatisfaction,
                E2EMetricType::SystemLatency,
                E2EMetricType::SystemThroughput,
                E2EMetricType::OverallQuality,
                E2EMetricType::Robustness,
                E2EMetricType::Consistency,
                E2EMetricType::Usability,
            ],
            user_experience_weight: 0.4,
            system_performance_weight: 0.3,
            quality_weight: 0.2,
            robustness_weight: 0.1,
            performance_thresholds: PerformanceThresholds::default(),
            user_satisfaction_config: UserSatisfactionConfig::default(),
            system_reliability_config: SystemReliabilityConfig::default(),
        }
    }
}

/// Types of end-to-end metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum E2EMetricType {
    /// Overall user satisfaction
    UserSatisfaction,
    /// System response latency
    SystemLatency,
    /// System throughput
    SystemThroughput,
    /// Overall quality (combination of all quality metrics)
    OverallQuality,
    /// System robustness to different inputs
    Robustness,
    /// Consistency across similar queries
    Consistency,
    /// System usability and user experience
    Usability,
    /// Resource efficiency
    ResourceEfficiency,
    /// Error rate and reliability
    ErrorRate,
    /// Scalability under load
    Scalability,
    /// User engagement metrics
    UserEngagement,
    /// Trust and credibility
    TrustScore,
}

/// Performance thresholds for evaluation
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,

    /// Minimum throughput (queries per second)
    pub min_throughput_qps: f32,

    /// Maximum error rate (%)
    pub max_error_rate: f32,

    /// Minimum quality score
    pub min_quality_score: f32,

    /// Maximum resource usage (MB)
    pub max_memory_usage_mb: f32,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 2000.0,
            min_throughput_qps: 10.0,
            max_error_rate: 5.0,
            min_quality_score: 0.7,
            max_memory_usage_mb: 1000.0,
        }
    }
}

/// User satisfaction configuration
#[derive(Debug, Clone)]
pub struct UserSatisfactionConfig {
    /// Weight for answer quality
    pub answer_quality_weight: f32,

    /// Weight for response time
    pub response_time_weight: f32,

    /// Weight for relevance
    pub relevance_weight: f32,

    /// Weight for completeness
    pub completeness_weight: f32,

    /// Weight for clarity
    pub clarity_weight: f32,
}

impl Default for UserSatisfactionConfig {
    fn default() -> Self {
        Self {
            answer_quality_weight: 0.3,
            response_time_weight: 0.2,
            relevance_weight: 0.25,
            completeness_weight: 0.15,
            clarity_weight: 0.1,
        }
    }
}

/// System reliability configuration
#[derive(Debug, Clone)]
pub struct SystemReliabilityConfig {
    /// Acceptable failure rate
    pub acceptable_failure_rate: f32,

    /// Recovery time threshold
    pub recovery_time_threshold_ms: f32,

    /// Consistency threshold
    pub consistency_threshold: f32,
}

impl Default for SystemReliabilityConfig {
    fn default() -> Self {
        Self {
            acceptable_failure_rate: 0.01,
            recovery_time_threshold_ms: 5000.0,
            consistency_threshold: 0.9,
        }
    }
}

/// Trait for end-to-end metrics
pub trait E2EMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;

    /// Metric type
    fn metric_type(&self) -> E2EMetricType;

    /// Evaluate metric across all queries
    fn evaluate_system(
        &self,
        evaluation_data: &EvaluationData,
        system_metrics: &SystemMetrics,
    ) -> RragResult<f32>;

    /// Get metric configuration
    fn get_config(&self) -> E2EMetricConfig;
}

/// Configuration for E2E metrics
#[derive(Debug, Clone)]
pub struct E2EMetricConfig {
    /// Metric name
    pub name: String,

    /// Requires system performance data
    pub requires_performance_data: bool,

    /// Requires user feedback
    pub requires_user_feedback: bool,

    /// Score range
    pub score_range: (f32, f32),

    /// Higher is better
    pub higher_is_better: bool,

    /// Evaluation level
    pub evaluation_level: EvaluationLevel,
}

/// Level of evaluation
#[derive(Debug, Clone)]
pub enum EvaluationLevel {
    /// Query-level evaluation
    Query,
    /// Session-level evaluation
    Session,
    /// System-level evaluation
    System,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Average response time
    pub avg_response_time_ms: f32,

    /// Throughput (queries per second)
    pub throughput_qps: f32,

    /// Error rate
    pub error_rate: f32,

    /// Memory usage
    pub memory_usage_mb: f32,

    /// CPU usage
    pub cpu_usage_percent: f32,

    /// System availability
    pub availability_percent: f32,

    /// Cache hit rate
    pub cache_hit_rate: f32,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            avg_response_time_ms: 1000.0,
            throughput_qps: 50.0,
            error_rate: 1.0,
            memory_usage_mb: 512.0,
            cpu_usage_percent: 60.0,
            availability_percent: 99.5,
            cache_hit_rate: 0.8,
        }
    }
}

impl EndToEndEvaluator {
    /// Create new end-to-end evaluator
    pub fn new(config: EndToEndConfig) -> Self {
        let mut evaluator = Self {
            config: config.clone(),
            metrics: Vec::new(),
        };

        // Initialize metrics based on configuration
        evaluator.initialize_metrics();

        evaluator
    }

    /// Initialize metrics based on configuration
    fn initialize_metrics(&mut self) {
        for metric_type in &self.config.enabled_metrics {
            let metric: Box<dyn E2EMetric> = match metric_type {
                E2EMetricType::UserSatisfaction => Box::new(UserSatisfactionMetric::new(
                    self.config.user_satisfaction_config.clone(),
                )),
                E2EMetricType::SystemLatency => Box::new(SystemLatencyMetric::new(
                    self.config.performance_thresholds.clone(),
                )),
                E2EMetricType::SystemThroughput => Box::new(SystemThroughputMetric::new(
                    self.config.performance_thresholds.clone(),
                )),
                E2EMetricType::OverallQuality => Box::new(OverallQualityMetric::new()),
                E2EMetricType::Robustness => Box::new(RobustnessMetric::new()),
                E2EMetricType::Consistency => Box::new(ConsistencyMetric::new(
                    self.config.system_reliability_config.clone(),
                )),
                E2EMetricType::Usability => Box::new(UsabilityMetric::new()),
                E2EMetricType::ResourceEfficiency => Box::new(ResourceEfficiencyMetric::new(
                    self.config.performance_thresholds.clone(),
                )),
                E2EMetricType::ErrorRate => Box::new(ErrorRateMetric::new(
                    self.config.system_reliability_config.clone(),
                )),
                _ => continue, // Skip unsupported metrics
            };

            self.metrics.push(metric);
        }
    }
}

impl Evaluator for EndToEndEvaluator {
    fn name(&self) -> &str {
        "EndToEnd"
    }

    fn evaluate(&self, data: &EvaluationData) -> RragResult<EvaluationResult> {
        let start_time = std::time::Instant::now();
        let mut overall_scores = HashMap::new();
        let per_query_results = Vec::new(); // E2E metrics are typically system-level

        // Calculate system metrics from evaluation data
        let system_metrics = self.calculate_system_metrics(data);

        // Evaluate each metric
        for metric in &self.metrics {
            match metric.evaluate_system(data, &system_metrics) {
                Ok(score) => {
                    overall_scores.insert(metric.name().to_string(), score);
                }
                Err(e) => {
                    ewarn!(" Failed to evaluate {}: {}", metric.name(), e);
                }
            }
        }

        // Calculate weighted overall score
        let overall_score = self.calculate_overall_score(&overall_scores);
        overall_scores.insert("overall_e2e_score".to_string(), overall_score);

        let total_time = start_time.elapsed().as_millis() as f32;

        // Generate insights and recommendations
        let insights = self.generate_insights(&overall_scores, &system_metrics);
        let recommendations = self.generate_recommendations(&overall_scores, &system_metrics);

        Ok(EvaluationResult {
            id: uuid::Uuid::new_v4().to_string(),
            evaluation_type: "EndToEnd".to_string(),
            overall_scores: overall_scores.clone(),
            per_query_results,
            summary: EvaluationSummary {
                total_queries: data.queries.len(),
                avg_scores: overall_scores.clone(),
                std_deviations: HashMap::new(), // Not applicable for system-level metrics
                performance_stats: PerformanceStats {
                    avg_eval_time_ms: total_time,
                    total_eval_time_ms: total_time,
                    peak_memory_usage_mb: system_metrics.memory_usage_mb,
                    throughput_qps: system_metrics.throughput_qps,
                },
                insights,
                recommendations,
            },
            metadata: EvaluationMetadata {
                timestamp: chrono::Utc::now(),
                evaluation_version: "1.0.0".to_string(),
                system_config: HashMap::new(),
                environment: std::env::vars().collect(),
                git_commit: None,
            },
        })
    }

    fn supported_metrics(&self) -> Vec<String> {
        self.metrics.iter().map(|m| m.name().to_string()).collect()
    }

    fn get_config(&self) -> EvaluatorConfig {
        EvaluatorConfig {
            name: "EndToEnd".to_string(),
            version: "1.0.0".to_string(),
            metrics: self.supported_metrics(),
            performance: EvaluatorPerformance {
                avg_time_per_sample_ms: 200.0,
                memory_usage_mb: 100.0,
                accuracy: 0.9,
            },
        }
    }
}

impl EndToEndEvaluator {
    /// Calculate system metrics from evaluation data
    fn calculate_system_metrics(&self, data: &EvaluationData) -> SystemMetrics {
        let mut total_time = 0.0;
        let mut error_count = 0;
        let mut valid_responses = 0;

        // Aggregate timing and error information
        for response in &data.system_responses {
            total_time += response.timing.total_time_ms;
            valid_responses += 1;

            // Check for errors (simplified)
            if response.generated_answer.is_none() || response.retrieved_docs.is_empty() {
                error_count += 1;
            }
        }

        let avg_response_time = if valid_responses > 0 {
            total_time / valid_responses as f32
        } else {
            0.0
        };

        let error_rate = if data.queries.len() > 0 {
            (error_count as f32 / data.queries.len() as f32) * 100.0
        } else {
            0.0
        };

        let throughput = if total_time > 0.0 {
            (valid_responses as f32 * 1000.0) / total_time // Convert to QPS
        } else {
            0.0
        };

        SystemMetrics {
            avg_response_time_ms: avg_response_time,
            throughput_qps: throughput,
            error_rate,
            memory_usage_mb: 256.0,  // Estimated
            cpu_usage_percent: 45.0, // Estimated
            availability_percent: 99.0,
            cache_hit_rate: 0.7,
        }
    }

    /// Calculate overall weighted score
    fn calculate_overall_score(&self, scores: &HashMap<String, f32>) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        // User experience metrics
        if let Some(&user_satisfaction) = scores.get("user_satisfaction") {
            weighted_sum += user_satisfaction * self.config.user_experience_weight;
            total_weight += self.config.user_experience_weight;
        }

        // System performance metrics
        let performance_metrics = ["system_latency", "system_throughput", "resource_efficiency"];
        let mut performance_score = 0.0;
        let mut performance_count = 0;

        for metric in &performance_metrics {
            if let Some(&score) = scores.get(*metric) {
                performance_score += score;
                performance_count += 1;
            }
        }

        if performance_count > 0 {
            performance_score /= performance_count as f32;
            weighted_sum += performance_score * self.config.system_performance_weight;
            total_weight += self.config.system_performance_weight;
        }

        // Quality metrics
        if let Some(&quality) = scores.get("overall_quality") {
            weighted_sum += quality * self.config.quality_weight;
            total_weight += self.config.quality_weight;
        }

        // Robustness metrics
        if let Some(&robustness) = scores.get("robustness") {
            weighted_sum += robustness * self.config.robustness_weight;
            total_weight += self.config.robustness_weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Generate insights based on evaluation results
    fn generate_insights(
        &self,
        scores: &HashMap<String, f32>,
        metrics: &SystemMetrics,
    ) -> Vec<String> {
        let mut insights = Vec::new();

        // Overall performance insights
        if let Some(&overall_score) = scores.get("overall_e2e_score") {
            if overall_score > 0.8 {
                insights.push("🎯 Excellent end-to-end system performance".to_string());
            } else if overall_score < 0.6 {
                insights.push("⚠️ End-to-end system performance needs improvement".to_string());
            }
        }

        // Latency insights
        if metrics.avg_response_time_ms > self.config.performance_thresholds.max_latency_ms {
            insights.push(format!(
                "🐌 High latency detected: {:.1}ms (threshold: {:.1}ms)",
                metrics.avg_response_time_ms, self.config.performance_thresholds.max_latency_ms
            ));
        }

        // Throughput insights
        if metrics.throughput_qps < self.config.performance_thresholds.min_throughput_qps {
            insights.push(format!(
                "📊 Low throughput: {:.1} QPS (minimum: {:.1} QPS)",
                metrics.throughput_qps, self.config.performance_thresholds.min_throughput_qps
            ));
        }

        // Error rate insights
        if metrics.error_rate > self.config.performance_thresholds.max_error_rate {
            insights.push(format!(
                "🚨 High error rate: {:.1}% (threshold: {:.1}%)",
                metrics.error_rate, self.config.performance_thresholds.max_error_rate
            ));
        }

        // Resource efficiency insights
        if metrics.memory_usage_mb > self.config.performance_thresholds.max_memory_usage_mb {
            insights.push(format!(
                "💾 High memory usage: {:.1}MB (threshold: {:.1}MB)",
                metrics.memory_usage_mb, self.config.performance_thresholds.max_memory_usage_mb
            ));
        }

        // User satisfaction insights
        if let Some(&user_satisfaction) = scores.get("user_satisfaction") {
            if user_satisfaction < 0.7 {
                insights.push(
                    "👥 User satisfaction below expectations - focus on UX improvements"
                        .to_string(),
                );
            }
        }

        insights
    }

    /// Generate recommendations based on evaluation results
    fn generate_recommendations(
        &self,
        scores: &HashMap<String, f32>,
        metrics: &SystemMetrics,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if metrics.avg_response_time_ms > self.config.performance_thresholds.max_latency_ms {
            recommendations
                .push("⚡ Optimize response time with caching and parallel processing".to_string());
            recommendations
                .push("🔧 Consider upgrading hardware or scaling horizontally".to_string());
        }

        if metrics.throughput_qps < self.config.performance_thresholds.min_throughput_qps {
            recommendations.push("📈 Implement load balancing and connection pooling".to_string());
            recommendations.push("🚀 Consider async processing for better throughput".to_string());
        }

        if metrics.error_rate > self.config.performance_thresholds.max_error_rate {
            recommendations
                .push("🛡️ Implement better error handling and retry mechanisms".to_string());
            recommendations.push("📊 Add comprehensive monitoring and alerting".to_string());
        }

        // User experience recommendations
        if let Some(&user_satisfaction) = scores.get("user_satisfaction") {
            if user_satisfaction < 0.7 {
                recommendations
                    .push("👤 Conduct user research to identify pain points".to_string());
                recommendations
                    .push("🎨 Improve user interface and interaction design".to_string());
            }
        }

        // Quality recommendations
        if let Some(&quality) = scores.get("overall_quality") {
            if quality < 0.7 {
                recommendations
                    .push("📚 Improve training data quality and model fine-tuning".to_string());
                recommendations
                    .push("🔍 Implement better content filtering and validation".to_string());
            }
        }

        // System reliability recommendations
        if let Some(&consistency) = scores.get("consistency") {
            if consistency < 0.8 {
                recommendations.push(
                    "🎯 Improve system consistency with better configuration management"
                        .to_string(),
                );
                recommendations
                    .push("🔄 Implement chaos engineering to test system resilience".to_string());
            }
        }

        recommendations
    }
}

// Individual E2E metric implementations
struct UserSatisfactionMetric {
    config: UserSatisfactionConfig,
}

impl UserSatisfactionMetric {
    fn new(config: UserSatisfactionConfig) -> Self {
        Self { config }
    }
}

impl E2EMetric for UserSatisfactionMetric {
    fn name(&self) -> &str {
        "user_satisfaction"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::UserSatisfaction
    }

    fn evaluate_system(&self, data: &EvaluationData, metrics: &SystemMetrics) -> RragResult<f32> {
        // Simulate user satisfaction based on various factors
        let response_time_score = if metrics.avg_response_time_ms < 1000.0 {
            1.0
        } else if metrics.avg_response_time_ms < 3000.0 {
            0.8 - (metrics.avg_response_time_ms - 1000.0) / 2000.0 * 0.3
        } else {
            0.5
        };

        // Quality score (based on having answers)
        let answered_queries = data
            .system_responses
            .iter()
            .filter(|r| r.generated_answer.is_some())
            .count();
        let answer_quality_score = answered_queries as f32 / data.queries.len() as f32;

        // Relevance score (simplified)
        let relevance_score = 0.8; // Placeholder

        // Completeness score (based on retrieved documents)
        let avg_docs = data
            .system_responses
            .iter()
            .map(|r| r.retrieved_docs.len())
            .sum::<usize>() as f32
            / data.system_responses.len() as f32;
        let completeness_score = (avg_docs / 5.0).min(1.0); // Normalize to 5 docs = 1.0

        // Clarity score (simplified)
        let clarity_score = 0.75; // Placeholder

        // Weighted combination
        let satisfaction = response_time_score * self.config.response_time_weight
            + answer_quality_score * self.config.answer_quality_weight
            + relevance_score * self.config.relevance_weight
            + completeness_score * self.config.completeness_weight
            + clarity_score * self.config.clarity_weight;

        Ok(satisfaction.min(1.0))
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "user_satisfaction".to_string(),
            requires_performance_data: true,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

struct SystemLatencyMetric {
    thresholds: PerformanceThresholds,
}

impl SystemLatencyMetric {
    fn new(thresholds: PerformanceThresholds) -> Self {
        Self { thresholds }
    }
}

impl E2EMetric for SystemLatencyMetric {
    fn name(&self) -> &str {
        "system_latency"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::SystemLatency
    }

    fn evaluate_system(&self, _data: &EvaluationData, metrics: &SystemMetrics) -> RragResult<f32> {
        // Score based on how well latency meets thresholds
        let score = if metrics.avg_response_time_ms <= self.thresholds.max_latency_ms {
            1.0 - (metrics.avg_response_time_ms / self.thresholds.max_latency_ms) * 0.2
        } else {
            // Penalty for exceeding threshold
            let excess = metrics.avg_response_time_ms - self.thresholds.max_latency_ms;
            let penalty = excess / self.thresholds.max_latency_ms;
            (0.8 - penalty * 0.5).max(0.0)
        };

        Ok(score)
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "system_latency".to_string(),
            requires_performance_data: true,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

struct SystemThroughputMetric {
    thresholds: PerformanceThresholds,
}

impl SystemThroughputMetric {
    fn new(thresholds: PerformanceThresholds) -> Self {
        Self { thresholds }
    }
}

impl E2EMetric for SystemThroughputMetric {
    fn name(&self) -> &str {
        "system_throughput"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::SystemThroughput
    }

    fn evaluate_system(&self, _data: &EvaluationData, metrics: &SystemMetrics) -> RragResult<f32> {
        // Score based on throughput relative to minimum threshold
        let score = if metrics.throughput_qps >= self.thresholds.min_throughput_qps {
            (metrics.throughput_qps / self.thresholds.min_throughput_qps).min(2.0) / 2.0
        } else {
            metrics.throughput_qps / self.thresholds.min_throughput_qps
        };

        Ok(score.min(1.0))
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "system_throughput".to_string(),
            requires_performance_data: true,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

// Placeholder implementations for other metrics
macro_rules! impl_simple_e2e_metric {
    ($name:ident, $metric_name:literal, $metric_type:expr, $default_score:expr) => {
        struct $name;

        impl $name {
            fn new() -> Self {
                Self
            }
        }

        impl E2EMetric for $name {
            fn name(&self) -> &str {
                $metric_name
            }

            fn metric_type(&self) -> E2EMetricType {
                $metric_type
            }

            fn evaluate_system(
                &self,
                _data: &EvaluationData,
                _metrics: &SystemMetrics,
            ) -> RragResult<f32> {
                Ok($default_score)
            }

            fn get_config(&self) -> E2EMetricConfig {
                E2EMetricConfig {
                    name: $metric_name.to_string(),
                    requires_performance_data: false,
                    requires_user_feedback: false,
                    score_range: (0.0, 1.0),
                    higher_is_better: true,
                    evaluation_level: EvaluationLevel::System,
                }
            }
        }
    };
}

struct OverallQualityMetric;

impl OverallQualityMetric {
    fn new() -> Self {
        Self
    }
}

impl E2EMetric for OverallQualityMetric {
    fn name(&self) -> &str {
        "overall_quality"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::OverallQuality
    }

    fn evaluate_system(&self, data: &EvaluationData, _metrics: &SystemMetrics) -> RragResult<f32> {
        // Aggregate quality score based on successful responses
        let successful_responses = data
            .system_responses
            .iter()
            .filter(|r| r.generated_answer.is_some() && !r.retrieved_docs.is_empty())
            .count();

        let quality_score = successful_responses as f32 / data.queries.len() as f32;
        Ok(quality_score)
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "overall_quality".to_string(),
            requires_performance_data: false,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

struct ConsistencyMetric {
    config: SystemReliabilityConfig,
}

impl ConsistencyMetric {
    fn new(config: SystemReliabilityConfig) -> Self {
        Self { config }
    }
}

impl E2EMetric for ConsistencyMetric {
    fn name(&self) -> &str {
        "consistency"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::Consistency
    }

    fn evaluate_system(&self, data: &EvaluationData, _metrics: &SystemMetrics) -> RragResult<f32> {
        // Measure consistency in response times and quality
        let response_times: Vec<f32> = data
            .system_responses
            .iter()
            .map(|r| r.timing.total_time_ms)
            .collect();

        if response_times.is_empty() {
            return Ok(0.0);
        }

        let mean_time = response_times.iter().sum::<f32>() / response_times.len() as f32;
        let variance = response_times
            .iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f32>()
            / response_times.len() as f32;
        let std_dev = variance.sqrt();

        // Consistency score based on coefficient of variation
        let cv = if mean_time > 0.0 {
            std_dev / mean_time
        } else {
            0.0
        };
        let consistency = (1.0 - cv).max(0.0);

        Ok(consistency)
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "consistency".to_string(),
            requires_performance_data: true,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

struct ResourceEfficiencyMetric {
    thresholds: PerformanceThresholds,
}

impl ResourceEfficiencyMetric {
    fn new(thresholds: PerformanceThresholds) -> Self {
        Self { thresholds }
    }
}

impl E2EMetric for ResourceEfficiencyMetric {
    fn name(&self) -> &str {
        "resource_efficiency"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::ResourceEfficiency
    }

    fn evaluate_system(&self, _data: &EvaluationData, metrics: &SystemMetrics) -> RragResult<f32> {
        // Score based on resource usage efficiency
        let memory_score = if metrics.memory_usage_mb <= self.thresholds.max_memory_usage_mb {
            1.0 - (metrics.memory_usage_mb / self.thresholds.max_memory_usage_mb) * 0.3
        } else {
            0.7 * (self.thresholds.max_memory_usage_mb / metrics.memory_usage_mb)
        };

        let cpu_score = if metrics.cpu_usage_percent <= 80.0 {
            1.0 - (metrics.cpu_usage_percent / 100.0) * 0.2
        } else {
            0.8 * (80.0 / metrics.cpu_usage_percent)
        };

        let efficiency = (memory_score + cpu_score) / 2.0;
        Ok(efficiency.min(1.0))
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "resource_efficiency".to_string(),
            requires_performance_data: true,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

struct ErrorRateMetric {
    config: SystemReliabilityConfig,
}

impl ErrorRateMetric {
    fn new(config: SystemReliabilityConfig) -> Self {
        Self { config }
    }
}

impl E2EMetric for ErrorRateMetric {
    fn name(&self) -> &str {
        "error_rate"
    }

    fn metric_type(&self) -> E2EMetricType {
        E2EMetricType::ErrorRate
    }

    fn evaluate_system(&self, _data: &EvaluationData, metrics: &SystemMetrics) -> RragResult<f32> {
        // Score based on error rate (lower error rate = higher score)
        let score = if metrics.error_rate <= self.config.acceptable_failure_rate * 100.0 {
            1.0 - (metrics.error_rate / 100.0) * 0.1
        } else {
            let excess = metrics.error_rate - (self.config.acceptable_failure_rate * 100.0);
            (0.9 - excess / 100.0 * 2.0).max(0.0)
        };

        Ok(score)
    }

    fn get_config(&self) -> E2EMetricConfig {
        E2EMetricConfig {
            name: "error_rate".to_string(),
            requires_performance_data: true,
            requires_user_feedback: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_level: EvaluationLevel::System,
        }
    }
}

impl_simple_e2e_metric!(
    RobustnessMetric,
    "robustness",
    E2EMetricType::Robustness,
    0.8
);
impl_simple_e2e_metric!(UsabilityMetric, "usability", E2EMetricType::Usability, 0.85);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{
        GroundTruth, RetrievedDocument, SystemResponse, SystemTiming, TestQuery,
    };

    #[test]
    fn test_user_satisfaction_metric() {
        let config = UserSatisfactionConfig::default();
        let metric = UserSatisfactionMetric::new(config);

        let data = create_test_data();
        let system_metrics = SystemMetrics::default();

        let score = metric.evaluate_system(&data, &system_metrics).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_system_latency_metric() {
        let thresholds = PerformanceThresholds::default();
        let metric = SystemLatencyMetric::new(thresholds);

        let data = create_test_data();
        let mut system_metrics = SystemMetrics::default();
        system_metrics.avg_response_time_ms = 1500.0; // Within threshold

        let score = metric.evaluate_system(&data, &system_metrics).unwrap();
        assert!(score > 0.5); // Should be good score for reasonable latency
    }

    #[test]
    fn test_end_to_end_evaluator() {
        let config = EndToEndConfig::default();
        let evaluator = EndToEndEvaluator::new(config);

        assert_eq!(evaluator.name(), "EndToEnd");
        assert!(!evaluator.supported_metrics().is_empty());
    }

    fn create_test_data() -> EvaluationData {
        use super::super::*;

        EvaluationData {
            queries: vec![TestQuery {
                id: "q1".to_string(),
                query: "What is machine learning?".to_string(),
                query_type: None,
                metadata: HashMap::new(),
            }],
            ground_truth: vec![GroundTruth {
                query_id: "q1".to_string(),
                relevant_docs: vec!["doc1".to_string()],
                expected_answer: Some("ML is AI subset".to_string()),
                relevance_judgments: HashMap::new(),
                metadata: HashMap::new(),
            }],
            system_responses: vec![SystemResponse {
                query_id: "q1".to_string(),
                retrieved_docs: vec![RetrievedDocument {
                    doc_id: "doc1".to_string(),
                    content: "Machine learning content".to_string(),
                    score: 0.9,
                    rank: 0,
                    metadata: HashMap::new(),
                }],
                generated_answer: Some("Machine learning is...".to_string()),
                timing: SystemTiming {
                    total_time_ms: 1000.0,
                    retrieval_time_ms: 500.0,
                    generation_time_ms: Some(400.0),
                    reranking_time_ms: Some(100.0),
                },
                metadata: HashMap::new(),
            }],
            context: HashMap::new(),
        }
    }
}
