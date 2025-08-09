//! # Rank Fusion Algorithms
//! 
//! Advanced algorithms for combining results from multiple retrieval methods.
//! Implements state-of-the-art fusion techniques for optimal ranking.

use crate::{RragResult, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Trait for rank fusion algorithms
pub trait RankFusion: Send + Sync {
    /// Fuse multiple result sets into a single ranked list
    fn fuse(
        &self,
        result_sets: Vec<Vec<SearchResult>>,
        limit: usize,
    ) -> RragResult<Vec<SearchResult>>;
}

/// Reciprocal Rank Fusion (RRF)
/// 
/// RRF is a simple yet effective fusion method that combines rankings
/// by summing reciprocal ranks. It's robust to outliers and doesn't
/// require score calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReciprocalRankFusion {
    /// Constant k to avoid division by zero (typically 60)
    pub k: f32,
    
    /// Whether to normalize final scores
    pub normalize_scores: bool,
}

impl Default for ReciprocalRankFusion {
    fn default() -> Self {
        Self {
            k: 60.0,
            normalize_scores: true,
        }
    }
}

impl RankFusion for ReciprocalRankFusion {
    fn fuse(
        &self,
        result_sets: Vec<Vec<SearchResult>>,
        limit: usize,
    ) -> RragResult<Vec<SearchResult>> {
        let mut fusion_scores: HashMap<String, f32> = HashMap::new();
        let mut doc_contents: HashMap<String, (String, HashMap<String, serde_json::Value>)> = HashMap::new();
        
        // Calculate RRF scores
        for results in &result_sets {
            for (rank, result) in results.iter().enumerate() {
                // RRF formula: 1 / (k + rank)
                let rrf_score = 1.0 / (self.k + rank as f32 + 1.0);
                
                *fusion_scores.entry(result.id.clone()).or_insert(0.0) += rrf_score;
                
                // Store document content and metadata
                doc_contents.entry(result.id.clone())
                    .or_insert((result.content.clone(), result.metadata.clone()));
            }
        }
        
        // Sort by fusion score
        let mut sorted_results: Vec<_> = fusion_scores
            .into_iter()
            .filter_map(|(id, score)| {
                doc_contents.get(&id).map(|(content, metadata)| {
                    SearchResult {
                        id: id.clone(),
                        content: content.clone(),
                        score,
                        rank: 0,
                        metadata: metadata.clone(),
                        embedding: None,
                    }
                })
            })
            .collect();
        
        sorted_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        // Normalize scores if requested
        if self.normalize_scores && !sorted_results.is_empty() {
            let max_score = sorted_results[0].score;
            for result in &mut sorted_results {
                result.score /= max_score;
            }
        }
        
        // Truncate and update ranks
        sorted_results.truncate(limit);
        for (i, result) in sorted_results.iter_mut().enumerate() {
            result.rank = i;
        }
        
        Ok(sorted_results)
    }
}

/// Weighted linear combination fusion
/// 
/// Combines scores from different retrievers using weighted linear combination.
/// Requires score calibration for best results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedFusion {
    /// Weights for each retriever (should sum to 1.0)
    pub weights: Vec<f32>,
    
    /// Score normalization method
    pub normalization: ScoreNormalization,
}

/// Score normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// Min-max normalization
    MinMax,
    /// Z-score normalization
    ZScore,
    /// No normalization
    None,
}

impl WeightedFusion {
    pub fn new(weights: Vec<f32>) -> Self {
        // Normalize weights to sum to 1.0
        let sum: f32 = weights.iter().sum();
        let normalized_weights = if sum > 0.0 {
            weights.iter().map(|w| w / sum).collect()
        } else {
            weights
        };
        
        Self {
            weights: normalized_weights,
            normalization: ScoreNormalization::MinMax,
        }
    }
    
    fn normalize_scores(&self, results: &mut Vec<SearchResult>) {
        match self.normalization {
            ScoreNormalization::MinMax => {
                if results.is_empty() {
                    return;
                }
                
                let min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
                let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
                
                if max > min {
                    for result in results {
                        result.score = (result.score - min) / (max - min);
                    }
                }
            }
            ScoreNormalization::ZScore => {
                if results.is_empty() {
                    return;
                }
                
                let mean: f32 = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
                let variance: f32 = results.iter()
                    .map(|r| (r.score - mean).powi(2))
                    .sum::<f32>() / results.len() as f32;
                let std_dev = variance.sqrt();
                
                if std_dev > 0.0 {
                    for result in results {
                        result.score = (result.score - mean) / std_dev;
                    }
                }
            }
            ScoreNormalization::None => {}
        }
    }
}

impl RankFusion for WeightedFusion {
    fn fuse(
        &self,
        mut result_sets: Vec<Vec<SearchResult>>,
        limit: usize,
    ) -> RragResult<Vec<SearchResult>> {
        // Normalize scores in each result set
        for results in &mut result_sets {
            self.normalize_scores(results);
        }
        
        let mut fusion_scores: HashMap<String, f32> = HashMap::new();
        let mut doc_contents: HashMap<String, (String, HashMap<String, serde_json::Value>)> = HashMap::new();
        
        // Apply weighted combination
        for (i, results) in result_sets.iter().enumerate() {
            let weight = self.weights.get(i).copied().unwrap_or(1.0 / result_sets.len() as f32);
            
            for result in results {
                *fusion_scores.entry(result.id.clone()).or_insert(0.0) += result.score * weight;
                
                doc_contents.entry(result.id.clone())
                    .or_insert((result.content.clone(), result.metadata.clone()));
            }
        }
        
        // Sort by fusion score
        let mut sorted_results: Vec<_> = fusion_scores
            .into_iter()
            .filter_map(|(id, score)| {
                doc_contents.get(&id).map(|(content, metadata)| {
                    SearchResult {
                        id: id.clone(),
                        content: content.clone(),
                        score,
                        rank: 0,
                        metadata: metadata.clone(),
                        embedding: None,
                    }
                })
            })
            .collect();
        
        sorted_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        sorted_results.truncate(limit);
        
        // Update ranks
        for (i, result) in sorted_results.iter_mut().enumerate() {
            result.rank = i;
        }
        
        Ok(sorted_results)
    }
}

/// Advanced fusion with learning-to-rank capabilities
#[derive(Debug, Clone)]
pub struct LearnedFusion {
    /// Feature weights learned from training data
    feature_weights: Vec<f32>,
    
    /// Interaction features between retrievers
    use_interactions: bool,
}

impl LearnedFusion {
    pub fn new(feature_weights: Vec<f32>) -> Self {
        Self {
            feature_weights,
            use_interactions: true,
        }
    }
    
    /// Extract features from result sets for learning
    pub fn extract_features(&self, result_sets: &[Vec<SearchResult>], doc_id: &str) -> Vec<f32> {
        let mut features = Vec::new();
        
        for results in result_sets {
            // Find document in this result set
            let doc_result = results.iter().find(|r| r.id == doc_id);
            
            if let Some(result) = doc_result {
                // Position features
                features.push(1.0 / (result.rank as f32 + 1.0)); // Reciprocal rank
                features.push(result.score); // Raw score
                features.push((results.len() - result.rank) as f32 / results.len() as f32); // Normalized position
            } else {
                // Document not found in this retriever
                features.push(0.0);
                features.push(0.0);
                features.push(0.0);
            }
        }
        
        // Add interaction features if enabled
        if self.use_interactions && result_sets.len() > 1 {
            for i in 0..result_sets.len() {
                for j in i + 1..result_sets.len() {
                    let score_i = result_sets[i].iter()
                        .find(|r| r.id == doc_id)
                        .map(|r| r.score)
                        .unwrap_or(0.0);
                    let score_j = result_sets[j].iter()
                        .find(|r| r.id == doc_id)
                        .map(|r| r.score)
                        .unwrap_or(0.0);
                    
                    // Interaction features
                    features.push(score_i * score_j); // Product
                    features.push((score_i - score_j).abs()); // Difference
                    features.push(score_i.max(score_j)); // Max
                }
            }
        }
        
        features
    }
}

impl RankFusion for LearnedFusion {
    fn fuse(
        &self,
        result_sets: Vec<Vec<SearchResult>>,
        limit: usize,
    ) -> RragResult<Vec<SearchResult>> {
        // Collect all unique document IDs
        let mut all_docs: HashSet<String> = HashSet::new();
        let mut doc_contents: HashMap<String, (String, HashMap<String, serde_json::Value>)> = HashMap::new();
        
        for results in &result_sets {
            for result in results {
                all_docs.insert(result.id.clone());
                doc_contents.entry(result.id.clone())
                    .or_insert((result.content.clone(), result.metadata.clone()));
            }
        }
        
        // Score each document using learned weights
        let mut scored_docs: Vec<(String, f32)> = all_docs
            .into_iter()
            .map(|doc_id| {
                let features = self.extract_features(&result_sets, &doc_id);
                let score: f32 = features.iter()
                    .zip(self.feature_weights.iter())
                    .map(|(f, w)| f * w)
                    .sum();
                (doc_id, score)
            })
            .collect();
        
        // Sort by learned score
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_docs.truncate(limit);
        
        // Build final results
        let results: Vec<SearchResult> = scored_docs
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (doc_id, score))| {
                doc_contents.get(&doc_id).map(|(content, metadata)| {
                    SearchResult {
                        id: doc_id,
                        content: content.clone(),
                        score,
                        rank,
                        metadata: metadata.clone(),
                        embedding: None,
                    }
                })
            })
            .collect();
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_results() -> Vec<Vec<SearchResult>> {
        vec![
            vec![
                SearchResult::new("1", "Doc 1", 0.9, 0),
                SearchResult::new("2", "Doc 2", 0.8, 1),
                SearchResult::new("3", "Doc 3", 0.7, 2),
            ],
            vec![
                SearchResult::new("2", "Doc 2", 0.95, 0),
                SearchResult::new("3", "Doc 3", 0.85, 1),
                SearchResult::new("4", "Doc 4", 0.75, 2),
            ],
        ]
    }
    
    #[test]
    fn test_reciprocal_rank_fusion() {
        let rrf = ReciprocalRankFusion::default();
        let results = rrf.fuse(create_test_results(), 3).unwrap();
        
        assert_eq!(results.len(), 3);
        // Doc 2 should rank highest (appears in both lists at high positions)
        assert_eq!(results[0].id, "2");
    }
    
    #[test]
    fn test_weighted_fusion() {
        let fusion = WeightedFusion::new(vec![0.3, 0.7]);
        let results = fusion.fuse(create_test_results(), 3).unwrap();
        
        assert_eq!(results.len(), 3);
        // Results should be weighted towards second retriever
    }
}