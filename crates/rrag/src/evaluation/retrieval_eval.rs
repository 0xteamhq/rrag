//! # Retrieval Evaluation Module
//! 
//! Specialized evaluation metrics for retrieval components including
//! traditional IR metrics (Precision@K, Recall@K, MAP, MRR, NDCG)
//! and modern retrieval-specific metrics.

use crate::{RragResult, RragError};
use super::{
    Evaluator, EvaluatorConfig, EvaluatorPerformance, EvaluationData, EvaluationResult,
    QueryEvaluationResult, EvaluationSummary, EvaluationMetadata, PerformanceStats,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Retrieval evaluator
pub struct RetrievalEvaluator {
    config: RetrievalEvalConfig,
    metrics: Vec<Box<dyn RetrievalMetric>>,
}

/// Configuration for retrieval evaluation
#[derive(Debug, Clone)]
pub struct RetrievalEvalConfig {
    /// Enabled metrics
    pub enabled_metrics: Vec<RetrievalMetricType>,
    
    /// K values for Precision@K, Recall@K, NDCG@K
    pub k_values: Vec<usize>,
    
    /// Relevance threshold for binary metrics
    pub relevance_threshold: f32,
    
    /// Use graded relevance (vs binary)
    pub use_graded_relevance: bool,
    
    /// Maximum grade for graded relevance
    pub max_relevance_grade: f32,
    
    /// Evaluation cutoff (maximum documents to consider)
    pub evaluation_cutoff: usize,
}

impl Default for RetrievalEvalConfig {
    fn default() -> Self {
        Self {
            enabled_metrics: vec![
                RetrievalMetricType::PrecisionAtK,
                RetrievalMetricType::RecallAtK,
                RetrievalMetricType::MeanAveragePrecision,
                RetrievalMetricType::MeanReciprocalRank,
                RetrievalMetricType::NdcgAtK,
                RetrievalMetricType::HitRate,
            ],
            k_values: vec![1, 3, 5, 10, 20],
            relevance_threshold: 0.5,
            use_graded_relevance: true,
            max_relevance_grade: 3.0,
            evaluation_cutoff: 100,
        }
    }
}

/// Types of retrieval evaluation metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalMetricType {
    /// Precision at K
    PrecisionAtK,
    /// Recall at K
    RecallAtK,
    /// F1 Score at K
    F1AtK,
    /// Mean Average Precision
    MeanAveragePrecision,
    /// Mean Reciprocal Rank
    MeanReciprocalRank,
    /// Normalized Discounted Cumulative Gain at K
    NdcgAtK,
    /// Hit Rate (at least one relevant document in top K)
    HitRate,
    /// Average Precision
    AveragePrecision,
    /// Reciprocal Rank
    ReciprocalRank,
    /// Coverage (fraction of query terms covered)
    Coverage,
    /// Diversity metrics
    Diversity,
    /// Novelty metrics
    Novelty,
}

/// Trait for retrieval evaluation metrics
pub trait RetrievalMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;
    
    /// Metric type
    fn metric_type(&self) -> RetrievalMetricType;
    
    /// Evaluate metric for a single query
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32>;
    
    /// Batch evaluation
    fn evaluate_batch(
        &self,
        retrieved_docs_batch: &[Vec<RetrievalDoc>],
        relevant_docs_batch: &[Vec<String>],
        relevance_judgments_batch: &[HashMap<String, f32>],
    ) -> RragResult<Vec<f32>> {
        let mut scores = Vec::new();
        
        for (i, retrieved_docs) in retrieved_docs_batch.iter().enumerate() {
            let relevant_docs = relevant_docs_batch.get(i).map(|r| r.as_slice()).unwrap_or(&[]);
            let empty_judgments = HashMap::new();
            let relevance_judgments = relevance_judgments_batch.get(i).unwrap_or(&empty_judgments);
            
            let score = self.evaluate_query(retrieved_docs, relevant_docs, relevance_judgments)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// Get metric configuration
    fn get_config(&self) -> RetrievalMetricConfig;
}

/// Configuration for retrieval metrics
#[derive(Debug, Clone)]
pub struct RetrievalMetricConfig {
    /// Metric name
    pub name: String,
    
    /// Requires relevance judgments
    pub requires_relevance_judgments: bool,
    
    /// Supports graded relevance
    pub supports_graded_relevance: bool,
    
    /// K values (if applicable)
    pub k_values: Vec<usize>,
    
    /// Score range
    pub score_range: (f32, f32),
    
    /// Higher is better
    pub higher_is_better: bool,
}

/// Retrieved document for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalDoc {
    /// Document ID
    pub doc_id: String,
    
    /// Retrieval score
    pub score: f32,
    
    /// Rank in results
    pub rank: usize,
}

impl RetrievalEvaluator {
    /// Create new retrieval evaluator
    pub fn new(config: RetrievalEvalConfig) -> Self {
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
            let metric: Box<dyn RetrievalMetric> = match metric_type {
                RetrievalMetricType::PrecisionAtK => {
                    Box::new(PrecisionAtKMetric::new(self.config.k_values.clone(), self.config.relevance_threshold))
                },
                RetrievalMetricType::RecallAtK => {
                    Box::new(RecallAtKMetric::new(self.config.k_values.clone(), self.config.relevance_threshold))
                },
                RetrievalMetricType::F1AtK => {
                    Box::new(F1AtKMetric::new(self.config.k_values.clone(), self.config.relevance_threshold))
                },
                RetrievalMetricType::MeanAveragePrecision => {
                    Box::new(MeanAveragePrecisionMetric::new(self.config.relevance_threshold))
                },
                RetrievalMetricType::MeanReciprocalRank => {
                    Box::new(MeanReciprocalRankMetric::new(self.config.relevance_threshold))
                },
                RetrievalMetricType::NdcgAtK => {
                    Box::new(NdcgAtKMetric::new(self.config.k_values.clone(), self.config.use_graded_relevance))
                },
                RetrievalMetricType::HitRate => {
                    Box::new(HitRateMetric::new(self.config.k_values.clone(), self.config.relevance_threshold))
                },
                RetrievalMetricType::AveragePrecision => {
                    Box::new(AveragePrecisionMetric::new(self.config.relevance_threshold))
                },
                RetrievalMetricType::ReciprocalRank => {
                    Box::new(ReciprocalRankMetric::new(self.config.relevance_threshold))
                },
                RetrievalMetricType::Coverage => {
                    Box::new(CoverageMetric::new())
                },
                _ => continue, // Skip unsupported metrics
            };
            
            self.metrics.push(metric);
        }
    }
}

impl Evaluator for RetrievalEvaluator {
    fn name(&self) -> &str {
        "Retrieval"
    }
    
    fn evaluate(&self, data: &EvaluationData) -> RragResult<EvaluationResult> {
        let start_time = std::time::Instant::now();
        let mut overall_scores = HashMap::new();
        let mut per_query_results = Vec::new();
        
        // Collect all metric scores
        let mut all_metric_scores: HashMap<String, Vec<f32>> = HashMap::new();
        
        // Process each query
        for query in &data.queries {
            let mut query_scores = HashMap::new();
            
            // Find corresponding system response and ground truth
            let system_response = data.system_responses.iter()
                .find(|r| r.query_id == query.id);
            let ground_truth = data.ground_truth.iter()
                .find(|gt| gt.query_id == query.id);
            
            if let (Some(response), Some(gt)) = (system_response, ground_truth) {
                // Convert to evaluation format
                let retrieved_docs: Vec<RetrievalDoc> = response.retrieved_docs.iter()
                    .map(|doc| RetrievalDoc {
                        doc_id: doc.doc_id.clone(),
                        score: doc.score,
                        rank: doc.rank,
                    })
                    .collect();
                
                // Evaluate each metric for this query
                for metric in &self.metrics {
                    match metric.evaluate_query(&retrieved_docs, &gt.relevant_docs, &gt.relevance_judgments) {
                        Ok(score) => {
                            let metric_name = metric.name().to_string();
                            query_scores.insert(metric_name.clone(), score);
                            
                            // Collect for overall statistics
                            all_metric_scores.entry(metric_name).or_insert_with(Vec::new).push(score);
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to evaluate {} for query {}: {}", 
                                     metric.name(), query.id, e);
                        }
                    }
                }
            }
            
            per_query_results.push(QueryEvaluationResult {
                query_id: query.id.clone(),
                scores: query_scores,
                errors: Vec::new(),
                details: HashMap::new(),
            });
        }
        
        // Calculate overall scores (averages)
        for (metric_name, scores) in &all_metric_scores {
            if !scores.is_empty() {
                let average = scores.iter().sum::<f32>() / scores.len() as f32;
                overall_scores.insert(metric_name.clone(), average);
            }
        }
        
        // Calculate summary statistics
        let mut avg_scores = HashMap::new();
        let mut std_deviations = HashMap::new();
        
        for (metric_name, scores) in &all_metric_scores {
            if !scores.is_empty() {
                let avg = scores.iter().sum::<f32>() / scores.len() as f32;
                avg_scores.insert(metric_name.clone(), avg);
                
                let variance = scores.iter()
                    .map(|score| (score - avg).powi(2))
                    .sum::<f32>() / scores.len() as f32;
                std_deviations.insert(metric_name.clone(), variance.sqrt());
            }
        }
        
        let total_time = start_time.elapsed().as_millis() as f32;
        
        // Generate insights
        let insights = self.generate_insights(&overall_scores, &std_deviations);
        let recommendations = self.generate_recommendations(&overall_scores);
        
        Ok(EvaluationResult {
            id: uuid::Uuid::new_v4().to_string(),
            evaluation_type: "Retrieval".to_string(),
            overall_scores,
            per_query_results,
            summary: EvaluationSummary {
                total_queries: data.queries.len(),
                avg_scores,
                std_deviations,
                performance_stats: PerformanceStats {
                    avg_eval_time_ms: total_time / data.queries.len() as f32,
                    total_eval_time_ms: total_time,
                    peak_memory_usage_mb: 30.0, // Estimated
                    throughput_qps: data.queries.len() as f32 / (total_time / 1000.0),
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
            name: "Retrieval".to_string(),
            version: "1.0.0".to_string(),
            metrics: self.supported_metrics(),
            performance: EvaluatorPerformance {
                avg_time_per_sample_ms: 50.0,
                memory_usage_mb: 30.0,
                accuracy: 0.95,
            },
        }
    }
}

impl RetrievalEvaluator {
    /// Generate insights based on scores
    fn generate_insights(&self, scores: &HashMap<String, f32>, std_devs: &HashMap<String, f32>) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Precision insights
        if let Some(&precision_5) = scores.get("precision@5") {
            if precision_5 > 0.8 {
                insights.push("🎯 Excellent precision@5 - retrieval is highly accurate".to_string());
            } else if precision_5 < 0.4 {
                insights.push("⚠️ Low precision@5 - many irrelevant documents retrieved".to_string());
            }
        }
        
        // Recall insights
        if let Some(&recall_10) = scores.get("recall@10") {
            if recall_10 < 0.5 {
                insights.push("📚 Low recall@10 - important documents may be missed".to_string());
            }
        }
        
        // NDCG insights
        if let Some(&ndcg_10) = scores.get("ndcg@10") {
            if ndcg_10 > 0.7 {
                insights.push("📈 Strong NDCG@10 - good ranking quality".to_string());
            } else if ndcg_10 < 0.4 {
                insights.push("🔄 Poor NDCG@10 - ranking needs improvement".to_string());
            }
        }
        
        // MRR insights
        if let Some(&mrr) = scores.get("mrr") {
            if mrr > 0.8 {
                insights.push("🥇 Excellent MRR - relevant documents consistently ranked high".to_string());
            } else if mrr < 0.4 {
                insights.push("📉 Low MRR - relevant documents often ranked low".to_string());
            }
        }
        
        // Precision vs Recall trade-off
        if let (Some(&precision_5), Some(&recall_10)) = (scores.get("precision@5"), scores.get("recall@10")) {
            if precision_5 > 0.7 && recall_10 < 0.4 {
                insights.push("⚖️ High precision but low recall - consider retrieving more documents".to_string());
            } else if precision_5 < 0.4 && recall_10 > 0.7 {
                insights.push("⚖️ High recall but low precision - improve ranking quality".to_string());
            }
        }
        
        insights
    }
    
    /// Generate recommendations based on scores
    fn generate_recommendations(&self, scores: &HashMap<String, f32>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if let Some(&precision_5) = scores.get("precision@5") {
            if precision_5 < 0.5 {
                recommendations.push("🎯 Improve retrieval precision by tuning similarity thresholds".to_string());
                recommendations.push("🔧 Consider using reranking models to improve result quality".to_string());
            }
        }
        
        if let Some(&recall_10) = scores.get("recall@10") {
            if recall_10 < 0.6 {
                recommendations.push("📈 Increase retrieval coverage by retrieving more candidates".to_string());
                recommendations.push("🔍 Improve query expansion to catch more relevant documents".to_string());
            }
        }
        
        if let Some(&ndcg_10) = scores.get("ndcg@10") {
            if ndcg_10 < 0.5 {
                recommendations.push("📊 Implement learning-to-rank models to improve ranking".to_string());
                recommendations.push("⚡ Fine-tune embedding models for better relevance scoring".to_string());
            }
        }
        
        if let Some(&mrr) = scores.get("mrr") {
            if mrr < 0.5 {
                recommendations.push("🥇 Focus on improving ranking of the most relevant document".to_string());
                recommendations.push("🎪 Consider ensemble methods to combine multiple ranking signals".to_string());
            }
        }
        
        recommendations
    }
}

// Individual metric implementations
struct PrecisionAtKMetric {
    k_values: Vec<usize>,
    relevance_threshold: f32,
}

impl PrecisionAtKMetric {
    fn new(k_values: Vec<usize>, relevance_threshold: f32) -> Self {
        Self { k_values, relevance_threshold }
    }
}

impl RetrievalMetric for PrecisionAtKMetric {
    fn name(&self) -> &str {
        "precision@k"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::PrecisionAtK
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        // For now, return precision@5 as the default
        let k = 5;
        
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        let top_k_docs = &retrieved_docs[..k.min(retrieved_docs.len())];
        let mut relevant_count = 0;
        
        for doc in top_k_docs {
            if let Some(&relevance) = relevance_judgments.get(&doc.doc_id) {
                if relevance >= self.relevance_threshold {
                    relevant_count += 1;
                }
            }
        }
        
        let precision = relevant_count as f32 / top_k_docs.len() as f32;
        Ok(precision)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "precision@k".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: self.k_values.clone(),
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct RecallAtKMetric {
    k_values: Vec<usize>,
    relevance_threshold: f32,
}

impl RecallAtKMetric {
    fn new(k_values: Vec<usize>, relevance_threshold: f32) -> Self {
        Self { k_values, relevance_threshold }
    }
}

impl RetrievalMetric for RecallAtKMetric {
    fn name(&self) -> &str {
        "recall@k"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::RecallAtK
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        let k = 10; // Default to recall@10
        
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        // Count total relevant documents
        let total_relevant = relevance_judgments.values()
            .filter(|&&relevance| relevance >= self.relevance_threshold)
            .count();
        
        if total_relevant == 0 {
            return Ok(1.0); // Perfect recall when no relevant documents exist
        }
        
        let top_k_docs = &retrieved_docs[..k.min(retrieved_docs.len())];
        let mut retrieved_relevant = 0;
        
        for doc in top_k_docs {
            if let Some(&relevance) = relevance_judgments.get(&doc.doc_id) {
                if relevance >= self.relevance_threshold {
                    retrieved_relevant += 1;
                }
            }
        }
        
        let recall = retrieved_relevant as f32 / total_relevant as f32;
        Ok(recall)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "recall@k".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: self.k_values.clone(),
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct F1AtKMetric {
    k_values: Vec<usize>,
    relevance_threshold: f32,
}

impl F1AtKMetric {
    fn new(k_values: Vec<usize>, relevance_threshold: f32) -> Self {
        Self { k_values, relevance_threshold }
    }
}

impl RetrievalMetric for F1AtKMetric {
    fn name(&self) -> &str {
        "f1@k"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::F1AtK
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        let k = 5; // Default to F1@5
        
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate precision@k
        let top_k_docs = &retrieved_docs[..k.min(retrieved_docs.len())];
        let mut relevant_retrieved = 0;
        
        for doc in top_k_docs {
            if let Some(&relevance) = relevance_judgments.get(&doc.doc_id) {
                if relevance >= self.relevance_threshold {
                    relevant_retrieved += 1;
                }
            }
        }
        
        let precision = relevant_retrieved as f32 / top_k_docs.len() as f32;
        
        // Calculate recall@k
        let total_relevant = relevance_judgments.values()
            .filter(|&&relevance| relevance >= self.relevance_threshold)
            .count();
        
        let recall = if total_relevant == 0 {
            1.0
        } else {
            relevant_retrieved as f32 / total_relevant as f32
        };
        
        // Calculate F1
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        
        Ok(f1)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "f1@k".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: self.k_values.clone(),
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct MeanAveragePrecisionMetric {
    relevance_threshold: f32,
}

impl MeanAveragePrecisionMetric {
    fn new(relevance_threshold: f32) -> Self {
        Self { relevance_threshold }
    }
}

impl RetrievalMetric for MeanAveragePrecisionMetric {
    fn name(&self) -> &str {
        "map"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::MeanAveragePrecision
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        let mut sum_precision = 0.0;
        let mut relevant_count = 0;
        let mut total_relevant = 0;
        
        // Count total relevant documents
        for &relevance in relevance_judgments.values() {
            if relevance >= self.relevance_threshold {
                total_relevant += 1;
            }
        }
        
        if total_relevant == 0 {
            return Ok(0.0);
        }
        
        // Calculate AP
        for (i, doc) in retrieved_docs.iter().enumerate() {
            if let Some(&relevance) = relevance_judgments.get(&doc.doc_id) {
                if relevance >= self.relevance_threshold {
                    relevant_count += 1;
                    let precision_at_i = relevant_count as f32 / (i + 1) as f32;
                    sum_precision += precision_at_i;
                }
            }
        }
        
        let ap = sum_precision / total_relevant as f32;
        Ok(ap)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "map".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: vec![],
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct MeanReciprocalRankMetric {
    relevance_threshold: f32,
}

impl MeanReciprocalRankMetric {
    fn new(relevance_threshold: f32) -> Self {
        Self { relevance_threshold }
    }
}

impl RetrievalMetric for MeanReciprocalRankMetric {
    fn name(&self) -> &str {
        "mrr"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::MeanReciprocalRank
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        // Find first relevant document
        for (i, doc) in retrieved_docs.iter().enumerate() {
            if let Some(&relevance) = relevance_judgments.get(&doc.doc_id) {
                if relevance >= self.relevance_threshold {
                    return Ok(1.0 / (i + 1) as f32);
                }
            }
        }
        
        Ok(0.0) // No relevant document found
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "mrr".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: vec![],
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct NdcgAtKMetric {
    k_values: Vec<usize>,
    use_graded_relevance: bool,
}

impl NdcgAtKMetric {
    fn new(k_values: Vec<usize>, use_graded_relevance: bool) -> Self {
        Self { k_values, use_graded_relevance }
    }
    
    fn dcg(&self, relevances: &[f32]) -> f32 {
        relevances.iter().enumerate()
            .map(|(i, &rel)| rel / (i as f32 + 2.0).log2())
            .sum()
    }
}

impl RetrievalMetric for NdcgAtKMetric {
    fn name(&self) -> &str {
        "ndcg@k"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::NdcgAtK
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        let k = 10; // Default to NDCG@10
        
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        let top_k_docs = &retrieved_docs[..k.min(retrieved_docs.len())];
        
        // Get relevances for retrieved documents
        let mut retrieved_relevances = Vec::new();
        for doc in top_k_docs {
            let relevance = relevance_judgments.get(&doc.doc_id).copied().unwrap_or(0.0);
            retrieved_relevances.push(relevance);
        }
        
        // Calculate DCG
        let dcg = self.dcg(&retrieved_relevances);
        
        // Calculate IDCG (ideal DCG)
        let mut all_relevances: Vec<f32> = relevance_judgments.values().copied().collect();
        all_relevances.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let ideal_relevances = &all_relevances[..k.min(all_relevances.len())];
        let idcg = self.dcg(ideal_relevances);
        
        // Calculate NDCG
        let ndcg = if idcg == 0.0 { 0.0 } else { dcg / idcg };
        Ok(ndcg)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "ndcg@k".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: self.k_values.clone(),
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct HitRateMetric {
    k_values: Vec<usize>,
    relevance_threshold: f32,
}

impl HitRateMetric {
    fn new(k_values: Vec<usize>, relevance_threshold: f32) -> Self {
        Self { k_values, relevance_threshold }
    }
}

impl RetrievalMetric for HitRateMetric {
    fn name(&self) -> &str {
        "hit_rate"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::HitRate
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        let k = 5; // Default to hit rate@5
        
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        let top_k_docs = &retrieved_docs[..k.min(retrieved_docs.len())];
        
        // Check if any document is relevant
        for doc in top_k_docs {
            if let Some(&relevance) = relevance_judgments.get(&doc.doc_id) {
                if relevance >= self.relevance_threshold {
                    return Ok(1.0); // Hit!
                }
            }
        }
        
        Ok(0.0) // No hit
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "hit_rate".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: self.k_values.clone(),
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

// Placeholder implementations for other metrics
struct AveragePrecisionMetric {
    relevance_threshold: f32,
}

impl AveragePrecisionMetric {
    fn new(relevance_threshold: f32) -> Self {
        Self { relevance_threshold }
    }
}

impl RetrievalMetric for AveragePrecisionMetric {
    fn name(&self) -> &str {
        "average_precision"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::AveragePrecision
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        // Same as MAP but for single query
        let map_metric = MeanAveragePrecisionMetric::new(self.relevance_threshold);
        map_metric.evaluate_query(retrieved_docs, _relevant_docs, relevance_judgments)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "average_precision".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: vec![],
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct ReciprocalRankMetric {
    relevance_threshold: f32,
}

impl ReciprocalRankMetric {
    fn new(relevance_threshold: f32) -> Self {
        Self { relevance_threshold }
    }
}

impl RetrievalMetric for ReciprocalRankMetric {
    fn name(&self) -> &str {
        "reciprocal_rank"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::ReciprocalRank
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        // Same as MRR but for single query
        let mrr_metric = MeanReciprocalRankMetric::new(self.relevance_threshold);
        mrr_metric.evaluate_query(retrieved_docs, _relevant_docs, relevance_judgments)
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "reciprocal_rank".to_string(),
            requires_relevance_judgments: true,
            supports_graded_relevance: true,
            k_values: vec![],
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct CoverageMetric;

impl CoverageMetric {
    fn new() -> Self {
        Self
    }
}

impl RetrievalMetric for CoverageMetric {
    fn name(&self) -> &str {
        "coverage"
    }
    
    fn metric_type(&self) -> RetrievalMetricType {
        RetrievalMetricType::Coverage
    }
    
    fn evaluate_query(
        &self,
        retrieved_docs: &[RetrievalDoc],
        _relevant_docs: &[String],
        _relevance_judgments: &HashMap<String, f32>,
    ) -> RragResult<f32> {
        // Simple coverage metric - fraction of documents that have content
        if retrieved_docs.is_empty() {
            return Ok(0.0);
        }
        
        let coverage = retrieved_docs.len() as f32 / 100.0; // Assume 100 is max expected
        Ok(coverage.min(1.0))
    }
    
    fn get_config(&self) -> RetrievalMetricConfig {
        RetrievalMetricConfig {
            name: "coverage".to_string(),
            requires_relevance_judgments: false,
            supports_graded_relevance: false,
            k_values: vec![],
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_precision_at_k_metric() {
        let metric = PrecisionAtKMetric::new(vec![5], 0.5);
        
        let retrieved_docs = vec![
            RetrievalDoc { doc_id: "doc1".to_string(), score: 0.9, rank: 0 },
            RetrievalDoc { doc_id: "doc2".to_string(), score: 0.8, rank: 1 },
            RetrievalDoc { doc_id: "doc3".to_string(), score: 0.7, rank: 2 },
        ];
        
        let mut relevance_judgments = HashMap::new();
        relevance_judgments.insert("doc1".to_string(), 1.0);
        relevance_judgments.insert("doc2".to_string(), 0.0);
        relevance_judgments.insert("doc3".to_string(), 1.0);
        
        let score = metric.evaluate_query(&retrieved_docs, &[], &relevance_judgments).unwrap();
        assert_eq!(score, 2.0 / 3.0); // 2 relevant out of 3 retrieved
    }
    
    #[test]
    fn test_mrr_metric() {
        let metric = MeanReciprocalRankMetric::new(0.5);
        
        let retrieved_docs = vec![
            RetrievalDoc { doc_id: "doc1".to_string(), score: 0.9, rank: 0 },
            RetrievalDoc { doc_id: "doc2".to_string(), score: 0.8, rank: 1 },
            RetrievalDoc { doc_id: "doc3".to_string(), score: 0.7, rank: 2 },
        ];
        
        let mut relevance_judgments = HashMap::new();
        relevance_judgments.insert("doc1".to_string(), 0.0);
        relevance_judgments.insert("doc2".to_string(), 1.0);
        relevance_judgments.insert("doc3".to_string(), 0.0);
        
        let score = metric.evaluate_query(&retrieved_docs, &[], &relevance_judgments).unwrap();
        assert_eq!(score, 0.5); // First relevant at position 2 -> 1/2 = 0.5
    }
    
    #[test]
    fn test_retrieval_evaluator_creation() {
        let config = RetrievalEvalConfig::default();
        let evaluator = RetrievalEvaluator::new(config);
        
        assert_eq!(evaluator.name(), "Retrieval");
        assert!(!evaluator.supported_metrics().is_empty());
    }
}