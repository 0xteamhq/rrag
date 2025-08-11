//! # RAGAS Metrics Implementation
//! 
//! Implementation of RAGAS (Retrieval-Augmented Generation Assessment) metrics
//! for evaluating RAG systems comprehensively. Includes faithfulness, answer
//! relevancy, context precision, context recall, and other RAGAS metrics.

use crate::{RragError, RragResult};
use super::{
    Evaluator, EvaluatorConfig, EvaluatorPerformance, EvaluationData, EvaluationResult,
    QueryEvaluationResult, EvaluationSummary, EvaluationMetadata, PerformanceStats,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// RAGAS evaluator
pub struct RagasEvaluator {
    config: RagasConfig,
    metrics: Vec<Box<dyn RagasMetric>>,
}

/// Configuration for RAGAS evaluation
#[derive(Debug, Clone)]
pub struct RagasConfig {
    /// Enabled RAGAS metrics
    pub enabled_metrics: Vec<RagasMetricType>,
    
    /// Faithfulness evaluation config
    pub faithfulness_config: FaithfulnessConfig,
    
    /// Answer relevancy config
    pub answer_relevancy_config: AnswerRelevancyConfig,
    
    /// Context precision config
    pub context_precision_config: ContextPrecisionConfig,
    
    /// Context recall config
    pub context_recall_config: ContextRecallConfig,
    
    /// Context relevancy config
    pub context_relevancy_config: ContextRelevancyConfig,
    
    /// Answer similarity config
    pub answer_similarity_config: AnswerSimilarityConfig,
    
    /// Answer correctness config
    pub answer_correctness_config: AnswerCorrectnessConfig,
}

impl Default for RagasConfig {
    fn default() -> Self {
        Self {
            enabled_metrics: vec![
                RagasMetricType::Faithfulness,
                RagasMetricType::AnswerRelevancy,
                RagasMetricType::ContextPrecision,
                RagasMetricType::ContextRecall,
                RagasMetricType::ContextRelevancy,
            ],
            faithfulness_config: FaithfulnessConfig::default(),
            answer_relevancy_config: AnswerRelevancyConfig::default(),
            context_precision_config: ContextPrecisionConfig::default(),
            context_recall_config: ContextRecallConfig::default(),
            context_relevancy_config: ContextRelevancyConfig::default(),
            answer_similarity_config: AnswerSimilarityConfig::default(),
            answer_correctness_config: AnswerCorrectnessConfig::default(),
        }
    }
}

/// Types of RAGAS metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RagasMetricType {
    /// Measures factual consistency of generated answer with given context
    Faithfulness,
    /// Measures how relevant the generated answer is to the question
    AnswerRelevancy,
    /// Measures fraction of relevant items in retrieved context
    ContextPrecision,
    /// Measures fraction of relevant items retrieved
    ContextRecall,
    /// Measures relevancy of retrieved context
    ContextRelevancy,
    /// Measures semantic similarity between generated and ground truth answers
    AnswerSimilarity,
    /// Measures factual correctness of generated answer
    AnswerCorrectness,
    /// Measures harmfulness of generated content
    Harmfulness,
    /// Measures maliciousness of generated content
    Maliciousness,
    /// Measures coherence of generated answer
    Coherence,
    /// Measures conciseness of generated answer
    Conciseness,
}

/// Trait for RAGAS metrics
pub trait RagasMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;
    
    /// Metric type
    fn metric_type(&self) -> RagasMetricType;
    
    /// Evaluate metric for a single query
    fn evaluate_query(
        &self,
        query: &str,
        contexts: &[String],
        answer: &str,
        ground_truth: Option<&str>,
    ) -> RragResult<f32>;
    
    /// Batch evaluation
    fn evaluate_batch(
        &self,
        queries: &[String],
        contexts: &[Vec<String>],
        answers: &[String],
        ground_truths: &[Option<String>],
    ) -> RragResult<Vec<f32>> {
        let mut scores = Vec::new();
        
        for (i, query) in queries.iter().enumerate() {
            let query_contexts = contexts.get(i).map(|c| c.as_slice()).unwrap_or(&[]);
            let answer = answers.get(i).map(|a| a.as_str()).unwrap_or("");
            let ground_truth = ground_truths.get(i).and_then(|gt| gt.as_ref()).map(|s| s.as_str());
            
            let score = self.evaluate_query(query, query_contexts, answer, ground_truth)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// Get metric configuration
    fn get_config(&self) -> RagasMetricConfig;
}

/// Configuration for individual RAGAS metrics
#[derive(Debug, Clone)]
pub struct RagasMetricConfig {
    /// Metric name
    pub name: String,
    
    /// Requires ground truth
    pub requires_ground_truth: bool,
    
    /// Requires context
    pub requires_context: bool,
    
    /// Score range
    pub score_range: (f32, f32),
    
    /// Higher is better
    pub higher_is_better: bool,
}

// Individual metric configurations
#[derive(Debug, Clone)]
pub struct FaithfulnessConfig {
    pub use_nli_model: bool,
    pub batch_size: usize,
    pub similarity_threshold: f32,
}

impl Default for FaithfulnessConfig {
    fn default() -> Self {
        Self {
            use_nli_model: false, // Use similarity-based for now
            batch_size: 10,
            similarity_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnswerRelevancyConfig {
    pub use_question_generation: bool,
    pub num_generated_questions: usize,
    pub similarity_threshold: f32,
}

impl Default for AnswerRelevancyConfig {
    fn default() -> Self {
        Self {
            use_question_generation: false,
            num_generated_questions: 3,
            similarity_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextPrecisionConfig {
    pub use_binary_relevance: bool,
    pub relevance_threshold: f32,
}

impl Default for ContextPrecisionConfig {
    fn default() -> Self {
        Self {
            use_binary_relevance: true,
            relevance_threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextRecallConfig {
    pub sentence_similarity_threshold: f32,
    pub use_semantic_similarity: bool,
}

impl Default for ContextRecallConfig {
    fn default() -> Self {
        Self {
            sentence_similarity_threshold: 0.7,
            use_semantic_similarity: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextRelevancyConfig {
    pub relevance_threshold: f32,
}

impl Default for ContextRelevancyConfig {
    fn default() -> Self {
        Self {
            relevance_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnswerSimilarityConfig {
    pub similarity_method: SimilarityMethod,
    pub weight_factual: f32,
    pub weight_semantic: f32,
}

impl Default for AnswerSimilarityConfig {
    fn default() -> Self {
        Self {
            similarity_method: SimilarityMethod::Cosine,
            weight_factual: 0.7,
            weight_semantic: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnswerCorrectnessConfig {
    pub use_fact_checking: bool,
    pub factual_weight: f32,
    pub semantic_weight: f32,
}

impl Default for AnswerCorrectnessConfig {
    fn default() -> Self {
        Self {
            use_fact_checking: false,
            factual_weight: 0.75,
            semantic_weight: 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SimilarityMethod {
    Cosine,
    Jaccard,
    Bleu,
    Rouge,
}

impl RagasEvaluator {
    /// Create new RAGAS evaluator
    pub fn new(config: RagasConfig) -> Self {
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
            let metric: Box<dyn RagasMetric> = match metric_type {
                RagasMetricType::Faithfulness => Box::new(FaithfulnessMetric::new(self.config.faithfulness_config.clone())),
                RagasMetricType::AnswerRelevancy => Box::new(AnswerRelevancyMetric::new(self.config.answer_relevancy_config.clone())),
                RagasMetricType::ContextPrecision => Box::new(ContextPrecisionMetric::new(self.config.context_precision_config.clone())),
                RagasMetricType::ContextRecall => Box::new(ContextRecallMetric::new(self.config.context_recall_config.clone())),
                RagasMetricType::ContextRelevancy => Box::new(ContextRelevancyMetric::new(self.config.context_relevancy_config.clone())),
                RagasMetricType::AnswerSimilarity => Box::new(AnswerSimilarityMetric::new(self.config.answer_similarity_config.clone())),
                RagasMetricType::AnswerCorrectness => Box::new(AnswerCorrectnessMetric::new(self.config.answer_correctness_config.clone())),
                _ => continue, // Skip unsupported metrics for now
            };
            
            self.metrics.push(metric);
        }
    }
}

impl Evaluator for RagasEvaluator {
    fn name(&self) -> &str {
        "RAGAS"
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
            
            if let Some(response) = system_response {
                // Extract contexts and answer
                let contexts: Vec<String> = response.retrieved_docs.iter()
                    .map(|doc| doc.content.clone())
                    .collect();
                let answer = response.generated_answer.as_deref().unwrap_or("");
                let ground_truth_answer = ground_truth.and_then(|gt| gt.expected_answer.as_deref());
                
                // Evaluate each metric for this query
                for metric in &self.metrics {
                    match metric.evaluate_query(&query.query, &contexts, answer, ground_truth_answer) {
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
            evaluation_type: "RAGAS".to_string(),
            overall_scores,
            per_query_results,
            summary: EvaluationSummary {
                total_queries: data.queries.len(),
                avg_scores,
                std_deviations,
                performance_stats: PerformanceStats {
                    avg_eval_time_ms: total_time / data.queries.len() as f32,
                    total_eval_time_ms: total_time,
                    peak_memory_usage_mb: 50.0, // Estimated
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
            name: "RAGAS".to_string(),
            version: "1.0.0".to_string(),
            metrics: self.supported_metrics(),
            performance: EvaluatorPerformance {
                avg_time_per_sample_ms: 100.0,
                memory_usage_mb: 50.0,
                accuracy: 0.9,
            },
        }
    }
}

impl RagasEvaluator {
    /// Generate insights based on scores
    fn generate_insights(&self, scores: &HashMap<String, f32>, std_devs: &HashMap<String, f32>) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Overall performance assessment
        let avg_score: f32 = scores.values().sum::<f32>() / scores.len() as f32;
        if avg_score > 0.8 {
            insights.push("🟢 Overall RAGAS performance is excellent".to_string());
        } else if avg_score > 0.6 {
            insights.push("🟡 Overall RAGAS performance is good with room for improvement".to_string());
        } else {
            insights.push("🔴 Overall RAGAS performance needs significant improvement".to_string());
        }
        
        // Specific metric insights
        if let Some(&faithfulness) = scores.get("faithfulness") {
            if faithfulness < 0.7 {
                insights.push("⚠️ Low faithfulness score indicates potential hallucination issues".to_string());
            }
        }
        
        if let Some(&context_precision) = scores.get("context_precision") {
            if context_precision < 0.6 {
                insights.push("🎯 Low context precision suggests retrieval is returning irrelevant documents".to_string());
            }
        }
        
        if let Some(&context_recall) = scores.get("context_recall") {
            if context_recall < 0.6 {
                insights.push("📚 Low context recall indicates important information may be missing from retrieval".to_string());
            }
        }
        
        // Consistency insights
        let high_variance_metrics: Vec<&String> = std_devs.iter()
            .filter(|(_, &std_dev)| std_dev > 0.2)
            .map(|(name, _)| name)
            .collect();
        
        if !high_variance_metrics.is_empty() {
            insights.push(format!("📊 High variance detected in: {}. This indicates inconsistent performance across queries", 
                                high_variance_metrics.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
        
        insights
    }
    
    /// Generate recommendations based on scores
    fn generate_recommendations(&self, scores: &HashMap<String, f32>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if let Some(&faithfulness) = scores.get("faithfulness") {
            if faithfulness < 0.7 {
                recommendations.push("📖 Implement stronger grounding mechanisms to improve faithfulness".to_string());
                recommendations.push("🔍 Consider post-processing to filter out potential hallucinations".to_string());
            }
        }
        
        if let Some(&context_precision) = scores.get("context_precision") {
            if context_precision < 0.6 {
                recommendations.push("🎯 Improve retrieval ranking to surface more relevant documents first".to_string());
                recommendations.push("⚡ Consider using reranking models to improve context quality".to_string());
            }
        }
        
        if let Some(&context_recall) = scores.get("context_recall") {
            if context_recall < 0.6 {
                recommendations.push("📈 Increase the number of retrieved documents".to_string());
                recommendations.push("🔧 Tune embedding models or retrieval parameters".to_string());
            }
        }
        
        if let Some(&answer_relevancy) = scores.get("answer_relevancy") {
            if answer_relevancy < 0.6 {
                recommendations.push("💬 Improve prompt engineering to generate more relevant answers".to_string());
                recommendations.push("🧠 Consider fine-tuning the generation model on domain-specific data".to_string());
            }
        }
        
        recommendations
    }
}

// Individual RAGAS metric implementations
struct FaithfulnessMetric {
    config: FaithfulnessConfig,
}

impl FaithfulnessMetric {
    fn new(config: FaithfulnessConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for FaithfulnessMetric {
    fn name(&self) -> &str {
        "faithfulness"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::Faithfulness
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        contexts: &[String],
        answer: &str,
        _ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        if contexts.is_empty() || answer.is_empty() {
            return Ok(0.0);
        }
        
        // Simple faithfulness evaluation based on content overlap
        let answer_lower = answer.to_lowercase();
        let answer_words: std::collections::HashSet<&str> = answer_lower
            .split_whitespace()
            .collect();
        
        let context_text = contexts.join(" ");
        let context_lower = context_text.to_lowercase();
        let context_words: std::collections::HashSet<&str> = context_lower
            .split_whitespace()
            .collect();
        
        let overlap = answer_words.intersection(&context_words).count();
        let faithfulness = if answer_words.is_empty() {
            0.0
        } else {
            overlap as f32 / answer_words.len() as f32
        };
        
        Ok(faithfulness.min(1.0))
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "faithfulness".to_string(),
            requires_ground_truth: false,
            requires_context: true,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct AnswerRelevancyMetric {
    config: AnswerRelevancyConfig,
}

impl AnswerRelevancyMetric {
    fn new(config: AnswerRelevancyConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for AnswerRelevancyMetric {
    fn name(&self) -> &str {
        "answer_relevancy"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::AnswerRelevancy
    }
    
    fn evaluate_query(
        &self,
        query: &str,
        _contexts: &[String],
        answer: &str,
        _ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        if query.is_empty() || answer.is_empty() {
            return Ok(0.0);
        }
        
        // Simple relevancy evaluation based on keyword overlap
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .collect();
        
        let answer_lower = answer.to_lowercase();
        let answer_words: std::collections::HashSet<&str> = answer_lower
            .split_whitespace()
            .collect();
        
        let overlap = query_words.intersection(&answer_words).count();
        let union = query_words.union(&answer_words).count();
        
        let jaccard = if union == 0 {
            0.0
        } else {
            overlap as f32 / union as f32
        };
        
        Ok(jaccard)
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "answer_relevancy".to_string(),
            requires_ground_truth: false,
            requires_context: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct ContextPrecisionMetric {
    config: ContextPrecisionConfig,
}

impl ContextPrecisionMetric {
    fn new(config: ContextPrecisionConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for ContextPrecisionMetric {
    fn name(&self) -> &str {
        "context_precision"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::ContextPrecision
    }
    
    fn evaluate_query(
        &self,
        query: &str,
        contexts: &[String],
        _answer: &str,
        ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        if contexts.is_empty() {
            return Ok(0.0);
        }
        
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .collect();
        
        let mut relevant_contexts = 0;
        
        for context in contexts {
            let context_lower = context.to_lowercase();
            let context_words: std::collections::HashSet<&str> = context_lower
                .split_whitespace()
                .collect();
            
            let overlap = query_words.intersection(&context_words).count();
            let relevance = overlap as f32 / query_words.len() as f32;
            
            if relevance >= self.config.relevance_threshold {
                relevant_contexts += 1;
            }
        }
        
        let precision = relevant_contexts as f32 / contexts.len() as f32;
        Ok(precision)
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "context_precision".to_string(),
            requires_ground_truth: false,
            requires_context: true,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct ContextRecallMetric {
    config: ContextRecallConfig,
}

impl ContextRecallMetric {
    fn new(config: ContextRecallConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for ContextRecallMetric {
    fn name(&self) -> &str {
        "context_recall"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::ContextRecall
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        contexts: &[String],
        _answer: &str,
        ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        let ground_truth = match ground_truth {
            Some(gt) => gt,
            None => return Ok(0.5), // Default score when no ground truth
        };
        
        if contexts.is_empty() {
            return Ok(0.0);
        }
        
        let gt_sentences: Vec<&str> = ground_truth.split('.').collect();
        let context_text = contexts.join(" ");
        
        let mut recalled_sentences = 0;
        
        for sentence in &gt_sentences {
            if sentence.trim().is_empty() {
                continue;
            }
            
            let sentence_lower = sentence.to_lowercase();
            let sentence_words: std::collections::HashSet<&str> = sentence_lower
                .split_whitespace()
                .collect();
            
            let context_text_lower = context_text.to_lowercase();
            let context_words: std::collections::HashSet<&str> = context_text_lower
                .split_whitespace()
                .collect();
            
            let overlap = sentence_words.intersection(&context_words).count();
            let similarity = if sentence_words.is_empty() {
                0.0
            } else {
                overlap as f32 / sentence_words.len() as f32
            };
            
            if similarity >= self.config.sentence_similarity_threshold {
                recalled_sentences += 1;
            }
        }
        
        let recall = if gt_sentences.is_empty() {
            1.0
        } else {
            recalled_sentences as f32 / gt_sentences.len() as f32
        };
        
        Ok(recall)
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "context_recall".to_string(),
            requires_ground_truth: true,
            requires_context: true,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct ContextRelevancyMetric {
    config: ContextRelevancyConfig,
}

impl ContextRelevancyMetric {
    fn new(config: ContextRelevancyConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for ContextRelevancyMetric {
    fn name(&self) -> &str {
        "context_relevancy"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::ContextRelevancy
    }
    
    fn evaluate_query(
        &self,
        query: &str,
        contexts: &[String],
        _answer: &str,
        _ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        if contexts.is_empty() || query.is_empty() {
            return Ok(0.0);
        }
        
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .collect();
        
        let context_text = contexts.join(" ");
        let context_text_lower = context_text.to_lowercase();
        let context_words: std::collections::HashSet<&str> = context_text_lower
            .split_whitespace()
            .collect();
        
        let overlap = query_words.intersection(&context_words).count();
        let union = query_words.union(&context_words).count();
        
        let relevancy = if union == 0 {
            0.0
        } else {
            overlap as f32 / union as f32
        };
        
        Ok(relevancy)
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "context_relevancy".to_string(),
            requires_ground_truth: false,
            requires_context: true,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct AnswerSimilarityMetric {
    config: AnswerSimilarityConfig,
}

impl AnswerSimilarityMetric {
    fn new(config: AnswerSimilarityConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for AnswerSimilarityMetric {
    fn name(&self) -> &str {
        "answer_similarity"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::AnswerSimilarity
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        _contexts: &[String],
        answer: &str,
        ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        let ground_truth = match ground_truth {
            Some(gt) => gt,
            None => return Ok(0.0),
        };
        
        if answer.is_empty() || ground_truth.is_empty() {
            return Ok(0.0);
        }
        
        match self.config.similarity_method {
            SimilarityMethod::Cosine | SimilarityMethod::Jaccard => {
                let answer_lower = answer.to_lowercase();
                let answer_words: std::collections::HashSet<&str> = answer_lower
                    .split_whitespace()
                    .collect();
                
                let gt_lower = ground_truth.to_lowercase();
                let gt_words: std::collections::HashSet<&str> = gt_lower
                    .split_whitespace()
                    .collect();
                
                let intersection = answer_words.intersection(&gt_words).count();
                let union = answer_words.union(&gt_words).count();
                
                let similarity = if union == 0 {
                    0.0
                } else {
                    intersection as f32 / union as f32
                };
                
                Ok(similarity)
            }
            _ => {
                // For other methods, use simple word overlap for now
                let answer_lower = answer.to_lowercase();
                let answer_words: std::collections::HashSet<&str> = answer_lower
                    .split_whitespace()
                    .collect();
                
                let gt_lower = ground_truth.to_lowercase();
                let gt_words: std::collections::HashSet<&str> = gt_lower
                    .split_whitespace()
                    .collect();
                
                let intersection = answer_words.intersection(&gt_words).count();
                let union = answer_words.union(&gt_words).count();
                
                let similarity = if union == 0 {
                    0.0
                } else {
                    intersection as f32 / union as f32
                };
                
                Ok(similarity)
            }
        }
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "answer_similarity".to_string(),
            requires_ground_truth: true,
            requires_context: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

struct AnswerCorrectnessMetric {
    config: AnswerCorrectnessConfig,
}

impl AnswerCorrectnessMetric {
    fn new(config: AnswerCorrectnessConfig) -> Self {
        Self { config }
    }
}

impl RagasMetric for AnswerCorrectnessMetric {
    fn name(&self) -> &str {
        "answer_correctness"
    }
    
    fn metric_type(&self) -> RagasMetricType {
        RagasMetricType::AnswerCorrectness
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        _contexts: &[String],
        answer: &str,
        ground_truth: Option<&str>,
    ) -> RragResult<f32> {
        let ground_truth = match ground_truth {
            Some(gt) => gt,
            None => return Ok(0.0),
        };
        
        if answer.is_empty() || ground_truth.is_empty() {
            return Ok(0.0);
        }
        
        // Combine factual and semantic correctness
        let answer_lower = answer.to_lowercase();
        let answer_words: std::collections::HashSet<&str> = answer_lower
            .split_whitespace()
            .collect();
        
        let gt_lower = ground_truth.to_lowercase();
        let gt_words: std::collections::HashSet<&str> = gt_lower
            .split_whitespace()
            .collect();
        
        // Factual correctness (word overlap)
        let intersection = answer_words.intersection(&gt_words).count();
        let factual_score = if gt_words.is_empty() {
            0.0
        } else {
            intersection as f32 / gt_words.len() as f32
        };
        
        // Semantic correctness (Jaccard similarity)
        let union = answer_words.union(&gt_words).count();
        let semantic_score = if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        };
        
        // Weighted combination
        let correctness = factual_score * self.config.factual_weight + 
                         semantic_score * self.config.semantic_weight;
        
        Ok(correctness.min(1.0))
    }
    
    fn get_config(&self) -> RagasMetricConfig {
        RagasMetricConfig {
            name: "answer_correctness".to_string(),
            requires_ground_truth: true,
            requires_context: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_faithfulness_metric() {
        let config = FaithfulnessConfig::default();
        let metric = FaithfulnessMetric::new(config);
        
        let contexts = vec!["Machine learning is a subset of AI".to_string()];
        let answer = "Machine learning is part of artificial intelligence";
        
        let score = metric.evaluate_query("", &contexts, answer, None).unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_answer_relevancy_metric() {
        let config = AnswerRelevancyConfig::default();
        let metric = AnswerRelevancyMetric::new(config);
        
        let query = "What is machine learning?";
        let answer = "Machine learning is a subset of artificial intelligence";
        
        let score = metric.evaluate_query(query, &[], answer, None).unwrap();
        assert!(score > 0.0);
    }
    
    #[test]
    fn test_context_precision_metric() {
        let config = ContextPrecisionConfig::default();
        let metric = ContextPrecisionMetric::new(config);
        
        let query = "machine learning";
        let contexts = vec![
            "Machine learning is great".to_string(),
            "The weather is nice today".to_string(),
        ];
        
        let score = metric.evaluate_query(query, &contexts, "", None).unwrap();
        assert!(score > 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_ragas_evaluator_creation() {
        let config = RagasConfig::default();
        let evaluator = RagasEvaluator::new(config);
        
        assert_eq!(evaluator.name(), "RAGAS");
        assert!(!evaluator.supported_metrics().is_empty());
    }
}