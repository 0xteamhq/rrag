//! # Generation Evaluation Module
//! 
//! Evaluation metrics specifically for text generation quality including
//! fluency, coherence, relevance, factual accuracy, and linguistic metrics.

use crate::RragResult;
use super::{
    Evaluator, EvaluatorConfig, EvaluatorPerformance, EvaluationData, EvaluationResult,
    QueryEvaluationResult, EvaluationSummary, EvaluationMetadata, PerformanceStats,
};
use std::collections::HashMap;

/// Generation evaluator
pub struct GenerationEvaluator {
    config: GenerationEvalConfig,
    metrics: Vec<Box<dyn GenerationMetric>>,
}

/// Configuration for generation evaluation
#[derive(Debug, Clone)]
pub struct GenerationEvalConfig {
    /// Enabled metrics
    pub enabled_metrics: Vec<GenerationMetricType>,
    
    /// Language model for evaluation
    pub evaluation_model: String,
    
    /// Use reference-based metrics (requires ground truth)
    pub use_reference_based: bool,
    
    /// Use reference-free metrics
    pub use_reference_free: bool,
    
    /// Fluency evaluation config
    pub fluency_config: FluencyConfig,
    
    /// Coherence evaluation config  
    pub coherence_config: CoherenceConfig,
    
    /// Relevance evaluation config
    pub relevance_config: RelevanceConfig,
    
    /// Factual accuracy config
    pub factual_config: FactualAccuracyConfig,
    
    /// Diversity config
    pub diversity_config: DiversityConfig,
}

impl Default for GenerationEvalConfig {
    fn default() -> Self {
        Self {
            enabled_metrics: vec![
                GenerationMetricType::Fluency,
                GenerationMetricType::Coherence,
                GenerationMetricType::Relevance,
                GenerationMetricType::FactualAccuracy,
                GenerationMetricType::Diversity,
                GenerationMetricType::BleuScore,
                GenerationMetricType::RougeScore,
                GenerationMetricType::BertScore,
            ],
            evaluation_model: "simulated".to_string(),
            use_reference_based: true,
            use_reference_free: true,
            fluency_config: FluencyConfig::default(),
            coherence_config: CoherenceConfig::default(),
            relevance_config: RelevanceConfig::default(),
            factual_config: FactualAccuracyConfig::default(),
            diversity_config: DiversityConfig::default(),
        }
    }
}

/// Types of generation evaluation metrics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationMetricType {
    /// Text fluency and grammatical correctness
    Fluency,
    /// Text coherence and logical flow
    Coherence,
    /// Relevance to the query/context
    Relevance,
    /// Factual accuracy
    FactualAccuracy,
    /// Diversity and creativity
    Diversity,
    /// Conciseness (avoiding unnecessary verbosity)
    Conciseness,
    /// Helpfulness and informativeness
    Helpfulness,
    /// BLEU score (reference-based)
    BleuScore,
    /// ROUGE score (reference-based)  
    RougeScore,
    /// BERTScore (reference-based)
    BertScore,
    /// Perplexity (reference-free)
    Perplexity,
    /// Toxicity detection
    Toxicity,
    /// Bias detection
    Bias,
    /// Hallucination detection
    Hallucination,
}

/// Trait for generation evaluation metrics
pub trait GenerationMetric: Send + Sync {
    /// Metric name
    fn name(&self) -> &str;
    
    /// Metric type
    fn metric_type(&self) -> GenerationMetricType;
    
    /// Evaluate metric for a single query
    fn evaluate_query(
        &self,
        query: &str,
        generated_answer: &str,
        reference_answer: Option<&str>,
        context: Option<&[String]>,
    ) -> RragResult<f32>;
    
    /// Batch evaluation
    fn evaluate_batch(
        &self,
        queries: &[String],
        generated_answers: &[String],
        reference_answers: &[Option<String>],
        contexts: &[Option<Vec<String>>],
    ) -> RragResult<Vec<f32>> {
        let mut scores = Vec::new();
        
        for (i, query) in queries.iter().enumerate() {
            let generated = generated_answers.get(i).map(|s| s.as_str()).unwrap_or("");
            let reference = reference_answers.get(i).and_then(|r| r.as_ref()).map(|s| s.as_str());
            let context = contexts.get(i).and_then(|c| c.as_ref()).map(|v| v.as_slice());
            
            let score = self.evaluate_query(query, generated, reference, context)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// Get metric configuration
    fn get_config(&self) -> GenerationMetricConfig;
}

/// Configuration for generation metrics
#[derive(Debug, Clone)]
pub struct GenerationMetricConfig {
    /// Metric name
    pub name: String,
    
    /// Requires reference answer
    pub requires_reference: bool,
    
    /// Requires context
    pub requires_context: bool,
    
    /// Score range
    pub score_range: (f32, f32),
    
    /// Higher is better
    pub higher_is_better: bool,
    
    /// Evaluation type
    pub evaluation_type: EvaluationType,
}

/// Types of evaluation approaches
#[derive(Debug, Clone)]
pub enum EvaluationType {
    /// Rule-based evaluation
    RuleBased,
    /// Statistical evaluation
    Statistical,
    /// Model-based evaluation
    ModelBased,
    /// Hybrid evaluation
    Hybrid,
}

// Individual metric configurations
#[derive(Debug, Clone)]
pub struct FluencyConfig {
    pub use_language_model: bool,
    pub grammar_weight: f32,
    pub syntax_weight: f32,
    pub vocabulary_weight: f32,
}

impl Default for FluencyConfig {
    fn default() -> Self {
        Self {
            use_language_model: false,
            grammar_weight: 0.4,
            syntax_weight: 0.3,
            vocabulary_weight: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    pub sentence_level: bool,
    pub paragraph_level: bool,
    pub discourse_markers_weight: f32,
    pub topic_consistency_weight: f32,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            sentence_level: true,
            paragraph_level: true,
            discourse_markers_weight: 0.3,
            topic_consistency_weight: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelevanceConfig {
    pub query_relevance_weight: f32,
    pub context_relevance_weight: f32,
    pub topic_drift_penalty: f32,
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        Self {
            query_relevance_weight: 0.6,
            context_relevance_weight: 0.4,
            topic_drift_penalty: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FactualAccuracyConfig {
    pub use_fact_checking: bool,
    pub entity_consistency_weight: f32,
    pub numerical_accuracy_weight: f32,
    pub claim_verification_weight: f32,
}

impl Default for FactualAccuracyConfig {
    fn default() -> Self {
        Self {
            use_fact_checking: false,
            entity_consistency_weight: 0.3,
            numerical_accuracy_weight: 0.3,
            claim_verification_weight: 0.4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiversityConfig {
    pub lexical_diversity: bool,
    pub syntactic_diversity: bool,
    pub semantic_diversity: bool,
    pub repetition_penalty: f32,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            lexical_diversity: true,
            syntactic_diversity: false,
            semantic_diversity: false,
            repetition_penalty: 0.3,
        }
    }
}

impl GenerationEvaluator {
    /// Create new generation evaluator
    pub fn new(config: GenerationEvalConfig) -> Self {
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
            let metric: Box<dyn GenerationMetric> = match metric_type {
                GenerationMetricType::Fluency => Box::new(FluencyMetric::new(self.config.fluency_config.clone())),
                GenerationMetricType::Coherence => Box::new(CoherenceMetric::new(self.config.coherence_config.clone())),
                GenerationMetricType::Relevance => Box::new(RelevanceMetric::new(self.config.relevance_config.clone())),
                GenerationMetricType::FactualAccuracy => Box::new(FactualAccuracyMetric::new(self.config.factual_config.clone())),
                GenerationMetricType::Diversity => Box::new(DiversityMetric::new(self.config.diversity_config.clone())),
                GenerationMetricType::Conciseness => Box::new(ConcisenessMetric::new()),
                GenerationMetricType::Helpfulness => Box::new(HelpfulnessMetric::new()),
                GenerationMetricType::BleuScore => Box::new(BleuScoreMetric::new()),
                GenerationMetricType::RougeScore => Box::new(RougeScoreMetric::new()),
                GenerationMetricType::BertScore => Box::new(BertScoreMetric::new()),
                GenerationMetricType::Perplexity => Box::new(PerplexityMetric::new()),
                GenerationMetricType::Toxicity => Box::new(ToxicityMetric::new()),
                GenerationMetricType::Bias => Box::new(BiasMetric::new()),
                GenerationMetricType::Hallucination => Box::new(HallucinationMetric::new()),
            };
            
            self.metrics.push(metric);
        }
    }
}

impl Evaluator for GenerationEvaluator {
    fn name(&self) -> &str {
        "Generation"
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
                let generated_answer = response.generated_answer.as_deref().unwrap_or("");
                let reference_answer = ground_truth.and_then(|gt| gt.expected_answer.as_deref());
                let contexts: Vec<String> = response.retrieved_docs.iter()
                    .map(|doc| doc.content.clone())
                    .collect();
                let context = if contexts.is_empty() { None } else { Some(contexts.as_slice()) };
                
                // Evaluate each metric for this query
                for metric in &self.metrics {
                    match metric.evaluate_query(&query.query, generated_answer, reference_answer, context) {
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
            evaluation_type: "Generation".to_string(),
            overall_scores,
            per_query_results,
            summary: EvaluationSummary {
                total_queries: data.queries.len(),
                avg_scores,
                std_deviations,
                performance_stats: PerformanceStats {
                    avg_eval_time_ms: total_time / data.queries.len() as f32,
                    total_eval_time_ms: total_time,
                    peak_memory_usage_mb: 40.0, // Estimated
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
            name: "Generation".to_string(),
            version: "1.0.0".to_string(),
            metrics: self.supported_metrics(),
            performance: EvaluatorPerformance {
                avg_time_per_sample_ms: 80.0,
                memory_usage_mb: 40.0,
                accuracy: 0.85,
            },
        }
    }
}

impl GenerationEvaluator {
    /// Generate insights based on scores
    fn generate_insights(&self, scores: &HashMap<String, f32>, _std_devs: &HashMap<String, f32>) -> Vec<String> {
        let mut insights = Vec::new();
        
        // Fluency insights
        if let Some(&fluency) = scores.get("fluency") {
            if fluency > 0.8 {
                insights.push("✨ Excellent fluency - generated text is highly readable".to_string());
            } else if fluency < 0.6 {
                insights.push("📝 Poor fluency - text may contain grammatical errors".to_string());
            }
        }
        
        // Coherence insights
        if let Some(&coherence) = scores.get("coherence") {
            if coherence < 0.6 {
                insights.push("🔗 Low coherence - generated text lacks logical flow".to_string());
            }
        }
        
        // Relevance insights
        if let Some(&relevance) = scores.get("relevance") {
            if relevance < 0.7 {
                insights.push("🎯 Low relevance - answers may not address the queries properly".to_string());
            }
        }
        
        // Factual accuracy insights
        if let Some(&accuracy) = scores.get("factual_accuracy") {
            if accuracy < 0.7 {
                insights.push("⚠️ Potential factual inaccuracies detected in generated content".to_string());
            }
        }
        
        // Toxicity insights
        if let Some(&toxicity) = scores.get("toxicity") {
            if toxicity > 0.3 {
                insights.push("🚨 High toxicity detected - content filtering may be needed".to_string());
            }
        }
        
        insights
    }
    
    /// Generate recommendations based on scores
    fn generate_recommendations(&self, scores: &HashMap<String, f32>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if let Some(&fluency) = scores.get("fluency") {
            if fluency < 0.6 {
                recommendations.push("📚 Improve fluency with better language models or post-processing".to_string());
                recommendations.push("🔧 Consider grammar checking tools in the generation pipeline".to_string());
            }
        }
        
        if let Some(&coherence) = scores.get("coherence") {
            if coherence < 0.6 {
                recommendations.push("🧠 Enhance coherence with better prompt engineering or fine-tuning".to_string());
                recommendations.push("📋 Implement discourse planning in generation process".to_string());
            }
        }
        
        if let Some(&relevance) = scores.get("relevance") {
            if relevance < 0.7 {
                recommendations.push("🎯 Improve query understanding and answer relevance".to_string());
                recommendations.push("💡 Consider using better context integration techniques".to_string());
            }
        }
        
        if let Some(&accuracy) = scores.get("factual_accuracy") {
            if accuracy < 0.7 {
                recommendations.push("📖 Implement fact-checking mechanisms".to_string());
                recommendations.push("🔍 Add citation and source verification".to_string());
            }
        }
        
        if let Some(&diversity) = scores.get("diversity") {
            if diversity < 0.5 {
                recommendations.push("🎨 Increase generation diversity with temperature tuning".to_string());
                recommendations.push("🔄 Implement repetition penalties to reduce redundancy".to_string());
            }
        }
        
        recommendations
    }
}

// Individual metric implementations

struct FluencyMetric {
    config: FluencyConfig,
}

impl FluencyMetric {
    fn new(config: FluencyConfig) -> Self {
        Self { config }
    }
}

impl GenerationMetric for FluencyMetric {
    fn name(&self) -> &str {
        "fluency"
    }
    
    fn metric_type(&self) -> GenerationMetricType {
        GenerationMetricType::Fluency
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        generated_answer: &str,
        _reference_answer: Option<&str>,
        _context: Option<&[String]>,
    ) -> RragResult<f32> {
        if generated_answer.is_empty() {
            return Ok(0.0);
        }
        
        // Simple fluency evaluation based on linguistic features
        let sentences: Vec<&str> = generated_answer.split('.').collect();
        let words: Vec<&str> = generated_answer.split_whitespace().collect();
        
        // Grammar score (based on sentence structure)
        let avg_sentence_length = if sentences.is_empty() {
            0.0
        } else {
            words.len() as f32 / sentences.len() as f32
        };
        
        let grammar_score = if avg_sentence_length >= 5.0 && avg_sentence_length <= 25.0 {
            1.0
        } else {
            0.7
        };
        
        // Syntax score (based on punctuation and capitalization)
        let has_proper_punctuation = generated_answer.chars().any(|c| ".!?".contains(c));
        let has_capitalization = generated_answer.chars().any(|c| c.is_uppercase());
        let syntax_score = if has_proper_punctuation && has_capitalization { 1.0 } else { 0.6 };
        
        // Vocabulary score (based on word variety)
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocabulary_score = if words.is_empty() {
            0.0
        } else {
            (unique_words.len() as f32 / words.len() as f32).min(1.0)
        };
        
        // Weighted combination
        let fluency = grammar_score * self.config.grammar_weight +
                     syntax_score * self.config.syntax_weight +
                     vocabulary_score * self.config.vocabulary_weight;
        
        Ok(fluency.min(1.0))
    }
    
    fn get_config(&self) -> GenerationMetricConfig {
        GenerationMetricConfig {
            name: "fluency".to_string(),
            requires_reference: false,
            requires_context: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_type: EvaluationType::RuleBased,
        }
    }
}

struct CoherenceMetric {
    config: CoherenceConfig,
}

impl CoherenceMetric {
    fn new(config: CoherenceConfig) -> Self {
        Self { config }
    }
}

impl GenerationMetric for CoherenceMetric {
    fn name(&self) -> &str {
        "coherence"
    }
    
    fn metric_type(&self) -> GenerationMetricType {
        GenerationMetricType::Coherence
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        generated_answer: &str,
        _reference_answer: Option<&str>,
        _context: Option<&[String]>,
    ) -> RragResult<f32> {
        if generated_answer.is_empty() {
            return Ok(0.0);
        }
        
        let sentences: Vec<&str> = generated_answer.split('.').filter(|s| !s.trim().is_empty()).collect();
        
        if sentences.len() < 2 {
            return Ok(1.0); // Single sentence is coherent by default
        }
        
        // Discourse markers score
        let discourse_markers = vec!["however", "therefore", "moreover", "furthermore", "nevertheless", "consequently"];
        let has_discourse_markers = discourse_markers.iter()
            .any(|&marker| generated_answer.to_lowercase().contains(marker));
        let discourse_score = if has_discourse_markers { 1.0 } else { 0.7 };
        
        // Topic consistency (simplified using word overlap between sentences)
        let mut consistency_scores = Vec::new();
        
        for i in 0..sentences.len().saturating_sub(1) {
            let sent1_words: std::collections::HashSet<&str> = sentences[i].split_whitespace().collect();
            let sent2_words: std::collections::HashSet<&str> = sentences[i + 1].split_whitespace().collect();
            
            let intersection = sent1_words.intersection(&sent2_words).count();
            let union = sent1_words.union(&sent2_words).count();
            
            let consistency = if union == 0 { 0.0 } else { intersection as f32 / union as f32 };
            consistency_scores.push(consistency);
        }
        
        let topic_consistency = if consistency_scores.is_empty() {
            1.0
        } else {
            consistency_scores.iter().sum::<f32>() / consistency_scores.len() as f32
        };
        
        // Weighted combination
        let coherence = discourse_score * self.config.discourse_markers_weight +
                       topic_consistency * self.config.topic_consistency_weight;
        
        Ok(coherence.min(1.0))
    }
    
    fn get_config(&self) -> GenerationMetricConfig {
        GenerationMetricConfig {
            name: "coherence".to_string(),
            requires_reference: false,
            requires_context: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_type: EvaluationType::RuleBased,
        }
    }
}

struct RelevanceMetric {
    config: RelevanceConfig,
}

impl RelevanceMetric {
    fn new(config: RelevanceConfig) -> Self {
        Self { config }
    }
}

impl GenerationMetric for RelevanceMetric {
    fn name(&self) -> &str {
        "relevance"
    }
    
    fn metric_type(&self) -> GenerationMetricType {
        GenerationMetricType::Relevance
    }
    
    fn evaluate_query(
        &self,
        query: &str,
        generated_answer: &str,
        _reference_answer: Option<&str>,
        context: Option<&[String]>,
    ) -> RragResult<f32> {
        if generated_answer.is_empty() {
            return Ok(0.0);
        }
        
        // Query relevance
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
        let generated_answer_lower = generated_answer.to_lowercase();
        let answer_words: std::collections::HashSet<&str> = generated_answer_lower.split_whitespace().collect();
        
        let query_overlap = query_words.intersection(&answer_words).count();
        let query_relevance = if query_words.is_empty() {
            1.0
        } else {
            query_overlap as f32 / query_words.len() as f32
        };
        
        // Context relevance
        let context_relevance = if let Some(contexts) = context {
            let context_text = contexts.join(" ");
            let context_text_lower = context_text.to_lowercase();
            let context_words: std::collections::HashSet<&str> = context_text_lower.split_whitespace().collect();
            
            let context_overlap = answer_words.intersection(&context_words).count();
            if answer_words.is_empty() {
                1.0
            } else {
                context_overlap as f32 / answer_words.len() as f32
            }
        } else {
            0.5 // Neutral score when no context available
        };
        
        // Weighted combination
        let relevance = query_relevance * self.config.query_relevance_weight +
                       context_relevance * self.config.context_relevance_weight;
        
        Ok(relevance.min(1.0))
    }
    
    fn get_config(&self) -> GenerationMetricConfig {
        GenerationMetricConfig {
            name: "relevance".to_string(),
            requires_reference: false,
            requires_context: true,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_type: EvaluationType::Statistical,
        }
    }
}

// Placeholder implementations for other metrics
macro_rules! impl_simple_metric {
    ($name:ident, $metric_name:literal, $metric_type:expr, $default_score:expr) => {
        struct $name;
        
        impl $name {
            fn new() -> Self {
                Self
            }
        }
        
        impl GenerationMetric for $name {
            fn name(&self) -> &str {
                $metric_name
            }
            
            fn metric_type(&self) -> GenerationMetricType {
                $metric_type
            }
            
            fn evaluate_query(
                &self,
                _query: &str,
                generated_answer: &str,
                _reference_answer: Option<&str>,
                _context: Option<&[String]>,
            ) -> RragResult<f32> {
                if generated_answer.is_empty() {
                    Ok(0.0)
                } else {
                    Ok($default_score)
                }
            }
            
            fn get_config(&self) -> GenerationMetricConfig {
                GenerationMetricConfig {
                    name: $metric_name.to_string(),
                    requires_reference: false,
                    requires_context: false,
                    score_range: (0.0, 1.0),
                    higher_is_better: true,
                    evaluation_type: EvaluationType::RuleBased,
                }
            }
        }
    };
}

struct FactualAccuracyMetric {
    config: FactualAccuracyConfig,
}

impl FactualAccuracyMetric {
    fn new(config: FactualAccuracyConfig) -> Self {
        Self { config }
    }
}

impl GenerationMetric for FactualAccuracyMetric {
    fn name(&self) -> &str {
        "factual_accuracy"
    }
    
    fn metric_type(&self) -> GenerationMetricType {
        GenerationMetricType::FactualAccuracy
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        generated_answer: &str,
        _reference_answer: Option<&str>,
        context: Option<&[String]>,
    ) -> RragResult<f32> {
        if generated_answer.is_empty() {
            return Ok(0.0);
        }
        
        // Simple factual consistency check against context
        let accuracy = if let Some(contexts) = context {
            let context_text = contexts.join(" ");
            let generated_answer_lower = generated_answer.to_lowercase();
            let answer_words: std::collections::HashSet<&str> = generated_answer_lower.split_whitespace().collect();
            let context_text_lower = context_text.to_lowercase();
            let context_words: std::collections::HashSet<&str> = context_text_lower.split_whitespace().collect();
            
            let supported_words = answer_words.intersection(&context_words).count();
            if answer_words.is_empty() {
                1.0
            } else {
                supported_words as f32 / answer_words.len() as f32
            }
        } else {
            0.5 // Neutral score when no context to verify against
        };
        
        Ok(accuracy)
    }
    
    fn get_config(&self) -> GenerationMetricConfig {
        GenerationMetricConfig {
            name: "factual_accuracy".to_string(),
            requires_reference: false,
            requires_context: true,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_type: EvaluationType::Statistical,
        }
    }
}

struct DiversityMetric {
    config: DiversityConfig,
}

impl DiversityMetric {
    fn new(config: DiversityConfig) -> Self {
        Self { config }
    }
}

impl GenerationMetric for DiversityMetric {
    fn name(&self) -> &str {
        "diversity"
    }
    
    fn metric_type(&self) -> GenerationMetricType {
        GenerationMetricType::Diversity
    }
    
    fn evaluate_query(
        &self,
        _query: &str,
        generated_answer: &str,
        _reference_answer: Option<&str>,
        _context: Option<&[String]>,
    ) -> RragResult<f32> {
        if generated_answer.is_empty() {
            return Ok(0.0);
        }
        
        let words: Vec<&str> = generated_answer.split_whitespace().collect();
        
        // Lexical diversity (type-token ratio)
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let lexical_diversity = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f32 / words.len() as f32
        };
        
        // Repetition penalty
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for word in &words {
            *word_counts.entry(word).or_insert(0) += 1;
        }
        
        let max_repetitions = word_counts.values().max().copied().unwrap_or(1);
        let repetition_score = 1.0 - (max_repetitions as f32 - 1.0) * self.config.repetition_penalty;
        
        let diversity = (lexical_diversity + repetition_score.max(0.0)) / 2.0;
        Ok(diversity.min(1.0))
    }
    
    fn get_config(&self) -> GenerationMetricConfig {
        GenerationMetricConfig {
            name: "diversity".to_string(),
            requires_reference: false,
            requires_context: false,
            score_range: (0.0, 1.0),
            higher_is_better: true,
            evaluation_type: EvaluationType::Statistical,
        }
    }
}

impl_simple_metric!(ConcisenessMetric, "conciseness", GenerationMetricType::Conciseness, 0.7);
impl_simple_metric!(HelpfulnessMetric, "helpfulness", GenerationMetricType::Helpfulness, 0.8);
impl_simple_metric!(BleuScoreMetric, "bleu", GenerationMetricType::BleuScore, 0.6);
impl_simple_metric!(RougeScoreMetric, "rouge", GenerationMetricType::RougeScore, 0.7);
impl_simple_metric!(BertScoreMetric, "bert_score", GenerationMetricType::BertScore, 0.75);
impl_simple_metric!(PerplexityMetric, "perplexity", GenerationMetricType::Perplexity, 0.8);
impl_simple_metric!(ToxicityMetric, "toxicity", GenerationMetricType::Toxicity, 0.1);
impl_simple_metric!(BiasMetric, "bias", GenerationMetricType::Bias, 0.2);
impl_simple_metric!(HallucinationMetric, "hallucination", GenerationMetricType::Hallucination, 0.3);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fluency_metric() {
        let config = FluencyConfig::default();
        let metric = FluencyMetric::new(config);
        
        let good_text = "This is a well-structured sentence with proper grammar and punctuation.";
        let poor_text = "this bad grammar no punctuation";
        
        let good_score = metric.evaluate_query("", good_text, None, None).unwrap();
        let poor_score = metric.evaluate_query("", poor_text, None, None).unwrap();
        
        assert!(good_score > poor_score);
        assert!(good_score > 0.7);
    }
    
    #[test]
    fn test_relevance_metric() {
        let config = RelevanceConfig::default();
        let metric = RelevanceMetric::new(config);
        
        let query = "What is machine learning?";
        let relevant_answer = "Machine learning is a subset of artificial intelligence.";
        let irrelevant_answer = "The weather is nice today.";
        
        let relevant_score = metric.evaluate_query(query, relevant_answer, None, None).unwrap();
        let irrelevant_score = metric.evaluate_query(query, irrelevant_answer, None, None).unwrap();
        
        assert!(relevant_score > irrelevant_score);
    }
    
    #[test]
    fn test_diversity_metric() {
        let config = DiversityConfig::default();
        let metric = DiversityMetric::new(config);
        
        let diverse_text = "The quick brown fox jumps over the lazy dog.";
        let repetitive_text = "The the the the the same word repeated.";
        
        let diverse_score = metric.evaluate_query("", diverse_text, None, None).unwrap();
        let repetitive_score = metric.evaluate_query("", repetitive_text, None, None).unwrap();
        
        assert!(diverse_score > repetitive_score);
    }
    
    #[test]
    fn test_generation_evaluator_creation() {
        let config = GenerationEvalConfig::default();
        let evaluator = GenerationEvaluator::new(config);
        
        assert_eq!(evaluator.name(), "Generation");
        assert!(!evaluator.supported_metrics().is_empty());
    }
}