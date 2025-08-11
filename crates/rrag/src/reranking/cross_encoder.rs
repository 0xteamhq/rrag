//! # Cross-Encoder Reranking
//!
//! Cross-encoder models that jointly encode query and document pairs
//! to produce more accurate relevance scores than bi-encoder approaches.

use crate::{RragResult, SearchResult};
use std::collections::HashMap;

/// Cross-encoder reranker for query-document relevance scoring
pub struct CrossEncoderReranker {
    /// Configuration
    config: CrossEncoderConfig,

    /// Model interface
    model: Box<dyn CrossEncoderModel>,

    /// Scoring cache
    score_cache: HashMap<String, f32>,
}

/// Configuration for cross-encoder reranking
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// Model type to use
    pub model_type: CrossEncoderModelType,

    /// Maximum sequence length for input
    pub max_sequence_length: usize,

    /// Batch size for processing
    pub batch_size: usize,

    /// Score aggregation method
    pub score_aggregation: ScoreAggregation,

    /// Reranking strategy
    pub strategy: RerankingStrategy,

    /// Confidence threshold
    pub confidence_threshold: f32,

    /// Enable caching
    pub enable_caching: bool,

    /// Temperature for score calibration
    pub temperature: f32,
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: CrossEncoderModelType::SimulatedBert,
            max_sequence_length: 512,
            batch_size: 16,
            score_aggregation: ScoreAggregation::Mean,
            strategy: RerankingStrategy::TopK(50),
            confidence_threshold: 0.5,
            enable_caching: true,
            temperature: 1.0,
        }
    }
}

/// Types of cross-encoder models
#[derive(Debug, Clone, PartialEq)]
pub enum CrossEncoderModelType {
    /// BERT-based cross-encoder
    Bert,
    /// RoBERTa-based cross-encoder
    RoBERTa,
    /// DistilBERT for faster inference
    DistilBert,
    /// Custom model
    Custom(String),
    /// Simulated model for demonstration
    SimulatedBert,
}

/// Score aggregation methods
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreAggregation {
    /// Average all scores
    Mean,
    /// Maximum score
    Max,
    /// Minimum score
    Min,
    /// Weighted average
    Weighted(Vec<f32>),
    /// Median score
    Median,
}

/// Reranking strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RerankingStrategy {
    /// Rerank top-k candidates
    TopK(usize),
    /// Rerank all candidates above threshold
    Threshold(f32),
    /// Adaptive reranking based on score distribution
    Adaptive,
    /// Stage-wise reranking
    Staged(Vec<usize>),
}

/// Result from cross-encoder reranking
#[derive(Debug, Clone)]
pub struct RerankedResult {
    /// Document identifier
    pub document_id: String,

    /// Cross-encoder relevance score
    pub cross_encoder_score: f32,

    /// Original retrieval score
    pub original_score: f32,

    /// Combined score
    pub combined_score: f32,

    /// Confidence in the score
    pub confidence: f32,

    /// Token-level attention scores (if available)
    pub attention_scores: Option<Vec<f32>>,

    /// Processing metadata
    pub metadata: CrossEncoderMetadata,
}

/// Metadata from cross-encoder processing
#[derive(Debug, Clone)]
pub struct CrossEncoderMetadata {
    /// Model used
    pub model_type: String,

    /// Input sequence length
    pub sequence_length: usize,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Number of tokens processed
    pub num_tokens: usize,

    /// Whether result was cached
    pub from_cache: bool,
}

/// Trait for cross-encoder models
pub trait CrossEncoderModel: Send + Sync {
    /// Score a single query-document pair
    fn score(&self, query: &str, document: &str) -> RragResult<f32>;

    /// Score multiple query-document pairs in batch
    fn score_batch(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Get attention scores if supported
    fn get_attention_scores(&self, query: &str, document: &str) -> RragResult<Option<Vec<f32>>> {
        let _ = (query, document);
        Ok(None)
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Model size in parameters
    pub parameters: Option<usize>,

    /// Whether model supports attention extraction
    pub supports_attention: bool,
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker
    pub fn new(config: CrossEncoderConfig) -> Self {
        let model = Self::create_model(&config.model_type);

        Self {
            config,
            model,
            score_cache: HashMap::new(),
        }
    }

    /// Create model based on configuration
    fn create_model(model_type: &CrossEncoderModelType) -> Box<dyn CrossEncoderModel> {
        match model_type {
            CrossEncoderModelType::SimulatedBert => Box::new(SimulatedBertCrossEncoder::new()),
            CrossEncoderModelType::Bert => Box::new(SimulatedBertCrossEncoder::new()), // Would be real BERT
            CrossEncoderModelType::RoBERTa => Box::new(SimulatedRobertaCrossEncoder::new()),
            CrossEncoderModelType::DistilBert => Box::new(SimulatedDistilBertCrossEncoder::new()),
            CrossEncoderModelType::Custom(name) => Box::new(CustomCrossEncoder::new(name.clone())),
        }
    }

    /// Rerank search results using cross-encoder
    pub async fn rerank(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> RragResult<HashMap<usize, f32>> {
        let _start_time = std::time::Instant::now();

        // Apply reranking strategy to select candidates
        let candidates = self.select_candidates(results)?;

        // Prepare query-document pairs
        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|&idx| (query.to_string(), results[idx].content.clone()))
            .collect();

        // Score the pairs
        let scores = if self.config.batch_size > 1 && pairs.len() > 1 {
            self.score_batch(&pairs).await?
        } else {
            self.score_sequential(&pairs).await?
        };

        // Create result mapping
        let mut score_map = HashMap::new();
        for (i, &candidate_idx) in candidates.iter().enumerate() {
            if let Some(&score) = scores.get(i) {
                score_map.insert(candidate_idx, score);
            }
        }

        Ok(score_map)
    }

    /// Select candidates for reranking based on strategy
    fn select_candidates(&self, results: &[SearchResult]) -> RragResult<Vec<usize>> {
        match &self.config.strategy {
            RerankingStrategy::TopK(k) => Ok((0..results.len().min(*k)).collect()),
            RerankingStrategy::Threshold(threshold) => Ok(results
                .iter()
                .enumerate()
                .filter(|(_, result)| result.score >= *threshold)
                .map(|(idx, _)| idx)
                .collect()),
            RerankingStrategy::Adaptive => {
                // Adaptive strategy based on score distribution
                let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
                let mean = scores.iter().sum::<f32>() / scores.len() as f32;
                let std_dev = {
                    let variance = scores
                        .iter()
                        .map(|score| (score - mean).powi(2))
                        .sum::<f32>()
                        / scores.len() as f32;
                    variance.sqrt()
                };

                let adaptive_threshold = mean - std_dev * 0.5;
                Ok(results
                    .iter()
                    .enumerate()
                    .filter(|(_, result)| result.score >= adaptive_threshold)
                    .map(|(idx, _)| idx)
                    .take(self.config.batch_size * 3) // Reasonable upper limit
                    .collect())
            }
            RerankingStrategy::Staged(stages) => {
                // Take the first stage size for now
                let stage_size = stages.first().copied().unwrap_or(10);
                Ok((0..results.len().min(stage_size)).collect())
            }
        }
    }

    /// Score pairs sequentially
    async fn score_sequential(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>> {
        let mut scores = Vec::new();

        for (query, document) in pairs {
            let cache_key = format!("{}|{}", query, document);

            let score = if self.config.enable_caching && self.score_cache.contains_key(&cache_key) {
                *self.score_cache.get(&cache_key).unwrap()
            } else {
                let score = self.model.score(query, document)?;
                if self.config.enable_caching {
                    // Note: In a real implementation, we'd need mutable access or use interior mutability
                }
                score
            };

            scores.push(score);
        }

        Ok(scores)
    }

    /// Score pairs in batches
    async fn score_batch(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>> {
        let mut all_scores = Vec::new();

        for chunk in pairs.chunks(self.config.batch_size) {
            let batch_scores = self.model.score_batch(chunk)?;
            all_scores.extend(batch_scores);
        }

        Ok(all_scores)
    }

    /// Apply temperature scaling to scores
    fn apply_temperature(&self, score: f32) -> f32 {
        if self.config.temperature == 1.0 {
            score
        } else {
            score / self.config.temperature
        }
    }

    /// Get model information
    pub fn get_model_info(&self) -> ModelInfo {
        self.model.model_info()
    }
}

/// Simulated BERT cross-encoder for demonstration
struct SimulatedBertCrossEncoder;

impl SimulatedBertCrossEncoder {
    fn new() -> Self {
        Self
    }
}

impl CrossEncoderModel for SimulatedBertCrossEncoder {
    fn score(&self, query: &str, document: &str) -> RragResult<f32> {
        // Simulate BERT cross-encoder scoring
        let query_tokens: Vec<&str> = query.split_whitespace().collect();
        let doc_tokens: Vec<&str> = document.split_whitespace().collect();

        // Simulate attention-based scoring
        let mut score = 0.0;
        let mut matches = 0;

        for q_token in &query_tokens {
            for d_token in &doc_tokens {
                let similarity = self.token_similarity(q_token, d_token);
                if similarity > 0.3 {
                    score += similarity;
                    matches += 1;
                }
            }
        }

        // Normalize by document length and add position bias
        let length_penalty = 1.0 / (1.0 + (doc_tokens.len() as f32 / 100.0));
        let coverage_bonus = if matches as f32 / query_tokens.len() as f32 > 0.5 {
            0.2
        } else {
            0.0
        };

        let final_score = ((score / query_tokens.len() as f32) * length_penalty + coverage_bonus)
            .max(0.0)
            .min(1.0);

        Ok(final_score)
    }

    fn score_batch(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>> {
        pairs
            .iter()
            .map(|(query, document)| self.score(query, document))
            .collect()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "SimulatedBERT-CrossEncoder".to_string(),
            version: "1.0".to_string(),
            max_sequence_length: 512,
            parameters: Some(110_000_000),
            supports_attention: true,
        }
    }

    fn get_attention_scores(&self, query: &str, document: &str) -> RragResult<Option<Vec<f32>>> {
        // Simulate attention scores
        let query_tokens: Vec<&str> = query.split_whitespace().collect();
        let doc_tokens: Vec<&str> = document.split_whitespace().collect();

        let mut attention_scores = Vec::new();
        for d_token in &doc_tokens {
            let max_attention = query_tokens
                .iter()
                .map(|q_token| self.token_similarity(q_token, d_token))
                .fold(0.0f32, |a, b| a.max(b));
            attention_scores.push(max_attention);
        }

        Ok(Some(attention_scores))
    }
}

impl SimulatedBertCrossEncoder {
    /// Simulate token-level similarity (would be learned embeddings in real model)
    fn token_similarity(&self, token1: &str, token2: &str) -> f32 {
        let t1_lower = token1.to_lowercase();
        let t2_lower = token2.to_lowercase();

        // Exact match
        if t1_lower == t2_lower {
            return 1.0;
        }

        // Partial matches
        if t1_lower.contains(&t2_lower) || t2_lower.contains(&t1_lower) {
            return 0.7;
        }

        // Character-level similarity (simplified Jaccard)
        let chars1: std::collections::HashSet<char> = t1_lower.chars().collect();
        let chars2: std::collections::HashSet<char> = t2_lower.chars().collect();

        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();

        if union == 0 {
            0.0
        } else {
            (intersection as f32 / union as f32) * 0.5
        }
    }
}

/// Simulated RoBERTa cross-encoder
struct SimulatedRobertaCrossEncoder;

impl SimulatedRobertaCrossEncoder {
    fn new() -> Self {
        Self
    }
}

impl CrossEncoderModel for SimulatedRobertaCrossEncoder {
    fn score(&self, query: &str, document: &str) -> RragResult<f32> {
        // Simulate RoBERTa with slightly different scoring
        let bert_encoder = SimulatedBertCrossEncoder::new();
        let base_score = bert_encoder.score(query, document)?;

        // RoBERTa might have different biases
        let roberta_adjustment = 0.05 * (document.len() as f32).log10().sin().abs();
        Ok((base_score + roberta_adjustment).min(1.0))
    }

    fn score_batch(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>> {
        pairs
            .iter()
            .map(|(query, document)| self.score(query, document))
            .collect()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "SimulatedRoBERTa-CrossEncoder".to_string(),
            version: "1.0".to_string(),
            max_sequence_length: 512,
            parameters: Some(125_000_000),
            supports_attention: true,
        }
    }
}

/// Simulated DistilBERT cross-encoder (faster, smaller)
struct SimulatedDistilBertCrossEncoder;

impl SimulatedDistilBertCrossEncoder {
    fn new() -> Self {
        Self
    }
}

impl CrossEncoderModel for SimulatedDistilBertCrossEncoder {
    fn score(&self, query: &str, document: &str) -> RragResult<f32> {
        // Simulate DistilBERT with faster but slightly less accurate scoring
        let bert_encoder = SimulatedBertCrossEncoder::new();
        let base_score = bert_encoder.score(query, document)?;

        // DistilBERT might be slightly less accurate
        let distillation_noise = 0.02 * (query.len() as f32 % 7.0) / 7.0;
        Ok((base_score - distillation_noise).max(0.0))
    }

    fn score_batch(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>> {
        pairs
            .iter()
            .map(|(query, document)| self.score(query, document))
            .collect()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "SimulatedDistilBERT-CrossEncoder".to_string(),
            version: "1.0".to_string(),
            max_sequence_length: 512,
            parameters: Some(66_000_000),
            supports_attention: false, // Simplified model
        }
    }
}

/// Custom cross-encoder model
struct CustomCrossEncoder {
    name: String,
}

impl CustomCrossEncoder {
    fn new(name: String) -> Self {
        Self { name }
    }
}

impl CrossEncoderModel for CustomCrossEncoder {
    fn score(&self, query: &str, document: &str) -> RragResult<f32> {
        // Placeholder for custom model
        let _ = (query, document);
        Ok(0.5) // Neutral score
    }

    fn score_batch(&self, pairs: &[(String, String)]) -> RragResult<Vec<f32>> {
        Ok(vec![0.5; pairs.len()])
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.name.clone(),
            version: "custom".to_string(),
            max_sequence_length: 512,
            parameters: None,
            supports_attention: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchResult;

    #[tokio::test]
    async fn test_cross_encoder_reranking() {
        let config = CrossEncoderConfig::default();
        let reranker = CrossEncoderReranker::new(config);

        let results = vec![
            SearchResult {
                id: "doc1".to_string(),
                content: "Machine learning is a subset of artificial intelligence".to_string(),
                score: 0.8,
                rank: 0,
                metadata: Default::default(),
                embedding: None,
            },
            SearchResult {
                id: "doc2".to_string(),
                content: "Deep learning uses neural networks with multiple layers".to_string(),
                score: 0.6,
                rank: 1,
                metadata: Default::default(),
                embedding: None,
            },
        ];

        let query = "What is machine learning?";
        let reranked_scores = reranker.rerank(query, &results).await.unwrap();

        assert!(!reranked_scores.is_empty());
        assert!(reranked_scores.contains_key(&0));
    }

    #[test]
    fn test_simulated_bert_scoring() {
        let model = SimulatedBertCrossEncoder::new();

        let score = model
            .score(
                "machine learning",
                "artificial intelligence and machine learning",
            )
            .unwrap();
        assert!(score > 0.0);
        assert!(score <= 1.0);

        // Should score higher for better matches
        let high_score = model
            .score("rust programming", "rust is a programming language")
            .unwrap();
        let low_score = model
            .score("rust programming", "cooking recipes for dinner")
            .unwrap();
        assert!(high_score > low_score);
    }

    #[test]
    fn test_batch_scoring() {
        let model = SimulatedBertCrossEncoder::new();

        let pairs = vec![
            ("query1".to_string(), "relevant document".to_string()),
            ("query2".to_string(), "another document".to_string()),
        ];

        let scores = model.score_batch(&pairs).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_attention_scores() {
        let model = SimulatedBertCrossEncoder::new();

        let attention = model
            .get_attention_scores("machine learning", "artificial intelligence")
            .unwrap();
        assert!(attention.is_some());

        let scores = attention.unwrap();
        assert_eq!(scores.len(), 2); // "artificial" and "intelligence"
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }
}
