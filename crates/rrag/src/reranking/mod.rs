//! # Advanced Reranking Module
//! 
//! Sophisticated reranking algorithms to improve retrieval precision after initial
//! candidate retrieval. Includes cross-encoder models, learning-to-rank, and
//! multi-signal reranking approaches.

use crate::RragResult;

pub mod cross_encoder;
pub mod learning_to_rank;
pub mod multi_signal;
pub mod neural_reranker;

// Re-exports
pub use cross_encoder::{
    CrossEncoderReranker, CrossEncoderConfig, CrossEncoderModel,
    RerankedResult, RerankingStrategy, ScoreAggregation
};
pub use learning_to_rank::{
    LearningToRankReranker, LTRConfig, LTRModel, LTRFeatures,
    RankingFeature, FeatureExtractor, FeatureType
};
pub use multi_signal::{
    MultiSignalReranker, MultiSignalConfig, SignalWeight,
    RelevanceSignal, SignalType, SignalAggregation
};
pub use neural_reranker::{
    NeuralReranker, NeuralConfig, AttentionMechanism,
    TransformerReranker, BertReranker, RobertaReranker
};

/// Main reranking interface that coordinates different reranking strategies
pub struct AdvancedReranker {
    /// Cross-encoder for query-document relevance
    cross_encoder: Option<CrossEncoderReranker>,
    
    /// Learning-to-rank model
    ltr_model: Option<LearningToRankReranker>,
    
    /// Multi-signal aggregation
    multi_signal: Option<MultiSignalReranker>,
    
    /// Neural reranking models
    neural_reranker: Option<NeuralReranker>,
    
    /// Configuration
    config: AdvancedRerankingConfig,
}

/// Configuration for advanced reranking
#[derive(Debug, Clone)]
pub struct AdvancedRerankingConfig {
    /// Enable cross-encoder reranking
    pub enable_cross_encoder: bool,
    
    /// Enable learning-to-rank
    pub enable_ltr: bool,
    
    /// Enable multi-signal reranking
    pub enable_multi_signal: bool,
    
    /// Enable neural reranking
    pub enable_neural: bool,
    
    /// Maximum number of candidates to rerank
    pub max_candidates: usize,
    
    /// Minimum score threshold
    pub score_threshold: f32,
    
    /// Reranking strategy priority order
    pub strategy_order: Vec<RerankingStrategyType>,
    
    /// Score combination method
    pub score_combination: ScoreCombination,
    
    /// Cache reranking results
    pub enable_caching: bool,
    
    /// Batch size for neural models
    pub batch_size: usize,
}

impl Default for AdvancedRerankingConfig {
    fn default() -> Self {
        Self {
            enable_cross_encoder: true,
            enable_ltr: false,
            enable_multi_signal: true,
            enable_neural: false,
            max_candidates: 100,
            score_threshold: 0.1,
            strategy_order: vec![
                RerankingStrategyType::CrossEncoder,
                RerankingStrategyType::MultiSignal,
            ],
            score_combination: ScoreCombination::Weighted(vec![0.7, 0.3]),
            enable_caching: true,
            batch_size: 32,
        }
    }
}

/// Types of reranking strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RerankingStrategyType {
    CrossEncoder,
    LearningToRank,
    MultiSignal,
    Neural,
}

/// Methods for combining scores from multiple rerankers
#[derive(Debug, Clone)]
pub enum ScoreCombination {
    /// Average all scores
    Average,
    /// Weighted combination
    Weighted(Vec<f32>),
    /// Maximum score
    Max,
    /// Minimum score
    Min,
    /// Learned combination (requires training)
    Learned,
}

/// Result from advanced reranking
#[derive(Debug, Clone)]
pub struct AdvancedRerankedResult {
    /// Document identifier
    pub document_id: String,
    
    /// Final combined score
    pub final_score: f32,
    
    /// Individual scores from each reranker
    pub component_scores: std::collections::HashMap<String, f32>,
    
    /// Original retrieval rank
    pub original_rank: usize,
    
    /// New rank after reranking
    pub new_rank: usize,
    
    /// Confidence in the reranking decision
    pub confidence: f32,
    
    /// Explanation of the reranking decision
    pub explanation: Option<String>,
    
    /// Processing metadata
    pub metadata: RerankingMetadata,
}

/// Metadata about the reranking process
#[derive(Debug, Clone)]
pub struct RerankingMetadata {
    /// Time taken for reranking
    pub reranking_time_ms: u64,
    
    /// Rerankers used
    pub rerankers_used: Vec<String>,
    
    /// Features extracted
    pub features_extracted: usize,
    
    /// Model versions used
    pub model_versions: std::collections::HashMap<String, String>,
    
    /// Warnings or notices
    pub warnings: Vec<String>,
}

impl AdvancedReranker {
    /// Create a new advanced reranker
    pub fn new(config: AdvancedRerankingConfig) -> Self {
        Self {
            cross_encoder: if config.enable_cross_encoder {
                Some(CrossEncoderReranker::new(CrossEncoderConfig::default()))
            } else {
                None
            },
            ltr_model: if config.enable_ltr {
                Some(LearningToRankReranker::new(LTRConfig::default()))
            } else {
                None
            },
            multi_signal: if config.enable_multi_signal {
                Some(MultiSignalReranker::new(MultiSignalConfig::default()))
            } else {
                None
            },
            neural_reranker: if config.enable_neural {
                Some(NeuralReranker::new(NeuralConfig::default()))
            } else {
                None
            },
            config,
        }
    }
    
    /// Rerank a list of initial retrieval results
    pub async fn rerank(
        &self,
        query: &str,
        initial_results: Vec<crate::SearchResult>,
    ) -> RragResult<Vec<AdvancedRerankedResult>> {
        let start_time = std::time::Instant::now();
        
        // Limit candidates if needed
        let candidates: Vec<_> = initial_results
            .into_iter()
            .take(self.config.max_candidates)
            .enumerate()
            .collect();
        
        let mut component_scores = std::collections::HashMap::new();
        let mut rerankers_used = Vec::new();
        let mut warnings = Vec::new();
        
        // Apply reranking strategies in order
        for strategy in &self.config.strategy_order {
            match strategy {
                RerankingStrategyType::CrossEncoder => {
                    if let Some(ref cross_encoder) = self.cross_encoder {
                        let candidate_results: Vec<_> = candidates.iter().map(|(_, result)| result.clone()).collect();
                        match cross_encoder.rerank(query, &candidate_results).await {
                            Ok(scores) => {
                                component_scores.insert("cross_encoder".to_string(), scores);
                                rerankers_used.push("cross_encoder".to_string());
                            }
                            Err(e) => {
                                warnings.push(format!("Cross-encoder failed: {}", e));
                            }
                        }
                    }
                }
                RerankingStrategyType::MultiSignal => {
                    if let Some(ref multi_signal) = self.multi_signal {
                        let candidate_results: Vec<_> = candidates.iter().map(|(_, result)| result.clone()).collect();
                        match multi_signal.rerank(query, &candidate_results).await {
                            Ok(scores) => {
                                component_scores.insert("multi_signal".to_string(), scores);
                                rerankers_used.push("multi_signal".to_string());
                            }
                            Err(e) => {
                                warnings.push(format!("Multi-signal failed: {}", e));
                            }
                        }
                    }
                }
                RerankingStrategyType::LearningToRank => {
                    if let Some(ref ltr) = self.ltr_model {
                        let candidate_results: Vec<_> = candidates.iter().map(|(_, result)| result.clone()).collect();
                        match ltr.rerank(query, &candidate_results).await {
                            Ok(scores) => {
                                component_scores.insert("ltr".to_string(), scores);
                                rerankers_used.push("ltr".to_string());
                            }
                            Err(e) => {
                                warnings.push(format!("LTR failed: {}", e));
                            }
                        }
                    }
                }
                RerankingStrategyType::Neural => {
                    if let Some(ref neural) = self.neural_reranker {
                        let candidate_results: Vec<_> = candidates.iter().map(|(_, result)| result.clone()).collect();
                        match neural.rerank(query, &candidate_results).await {
                            Ok(scores) => {
                                component_scores.insert("neural".to_string(), scores);
                                rerankers_used.push("neural".to_string());
                            }
                            Err(e) => {
                                warnings.push(format!("Neural reranker failed: {}", e));
                            }
                        }
                    }
                }
            }
        }
        
        // Combine scores
        let final_scores = self.combine_scores(&component_scores, candidates.len());
        
        // Create reranked results
        let mut reranked_results: Vec<_> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, (original_rank, result))| AdvancedRerankedResult {
                document_id: result.id.clone(),
                final_score: final_scores.get(&idx).copied().unwrap_or(result.score),
                component_scores: component_scores
                    .iter()
                    .map(|(name, scores)| (name.clone(), scores.get(&idx).copied().unwrap_or(0.0)))
                    .collect(),
                original_rank,
                new_rank: 0, // Will be filled after sorting
                confidence: self.calculate_confidence(&component_scores, idx),
                explanation: self.generate_explanation(&component_scores, idx),
                metadata: RerankingMetadata {
                    reranking_time_ms: start_time.elapsed().as_millis() as u64,
                    rerankers_used: rerankers_used.clone(),
                    features_extracted: 0, // Would be set by individual rerankers
                    model_versions: std::collections::HashMap::new(),
                    warnings: warnings.clone(),
                },
            })
            .collect();
        
        // Sort by final score
        reranked_results.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Update new ranks
        for (idx, result) in reranked_results.iter_mut().enumerate() {
            result.new_rank = idx;
        }
        
        // Filter by score threshold
        reranked_results.retain(|result| result.final_score >= self.config.score_threshold);
        
        Ok(reranked_results)
    }
    
    /// Combine scores from different rerankers
    fn combine_scores(
        &self,
        component_scores: &std::collections::HashMap<String, std::collections::HashMap<usize, f32>>,
        num_candidates: usize,
    ) -> std::collections::HashMap<usize, f32> {
        let mut final_scores = std::collections::HashMap::new();
        
        for idx in 0..num_candidates {
            let scores: Vec<f32> = component_scores
                .values()
                .map(|scores| scores.get(&idx).copied().unwrap_or(0.0))
                .collect();
            
            let final_score = match &self.config.score_combination {
                ScoreCombination::Average => {
                    if scores.is_empty() { 0.0 } else { scores.iter().sum::<f32>() / scores.len() as f32 }
                }
                ScoreCombination::Weighted(weights) => {
                    scores.iter().zip(weights.iter())
                        .map(|(score, weight)| score * weight)
                        .sum::<f32>()
                }
                ScoreCombination::Max => {
                    scores.iter().fold(0.0f32, |a, &b| a.max(b))
                }
                ScoreCombination::Min => {
                    scores.iter().fold(1.0f32, |a, &b| a.min(b))
                }
                ScoreCombination::Learned => {
                    // Would use a learned combination model
                    if scores.is_empty() { 0.0 } else { scores.iter().sum::<f32>() / scores.len() as f32 }
                }
            };
            
            final_scores.insert(idx, final_score);
        }
        
        final_scores
    }
    
    /// Calculate confidence in the reranking decision
    fn calculate_confidence(
        &self,
        component_scores: &std::collections::HashMap<String, std::collections::HashMap<usize, f32>>,
        idx: usize,
    ) -> f32 {
        // Simple confidence calculation based on score agreement
        let scores: Vec<f32> = component_scores
            .values()
            .map(|scores| scores.get(&idx).copied().unwrap_or(0.0))
            .collect();
        
        if scores.len() < 2 {
            return 0.5; // Low confidence with only one scorer
        }
        
        // Calculate standard deviation as inverse confidence
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
        let std_dev = variance.sqrt();
        
        // Convert to confidence (lower std_dev = higher confidence)
        (1.0 - std_dev.min(1.0)).max(0.0)
    }
    
    /// Generate explanation for reranking decision
    fn generate_explanation(
        &self,
        component_scores: &std::collections::HashMap<String, std::collections::HashMap<usize, f32>>,
        idx: usize,
    ) -> Option<String> {
        let scores: Vec<(String, f32)> = component_scores
            .iter()
            .map(|(name, scores)| (name.clone(), scores.get(&idx).copied().unwrap_or(0.0)))
            .collect();
        
        if scores.is_empty() {
            return None;
        }
        
        let mut explanations = Vec::new();
        
        for (reranker, score) in &scores {
            match reranker.as_str() {
                "cross_encoder" => {
                    explanations.push(format!("Cross-encoder relevance: {:.3}", score));
                }
                "multi_signal" => {
                    explanations.push(format!("Multi-signal analysis: {:.3}", score));
                }
                "ltr" => {
                    explanations.push(format!("Learning-to-rank: {:.3}", score));
                }
                "neural" => {
                    explanations.push(format!("Neural reranker: {:.3}", score));
                }
                _ => {
                    explanations.push(format!("{}: {:.3}", reranker, score));
                }
            }
        }
        
        Some(explanations.join("; "))
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: AdvancedRerankingConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &AdvancedRerankingConfig {
        &self.config
    }
}