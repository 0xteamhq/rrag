//! # Hybrid Retrieval System
//!
//! Combines semantic and keyword-based retrieval for optimal performance.
//! Implements multiple fusion strategies and adaptive weighting.

use super::{
    BM25Config, BM25Retriever, RankFusion, ReciprocalRankFusion, SemanticConfig, SemanticRetriever,
    WeightedFusion,
};
use crate::{Document, EmbeddingProvider, RragResult, SearchResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Hybrid retriever configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// BM25 configuration
    pub bm25_config: BM25Config,

    /// Semantic search configuration
    pub semantic_config: SemanticConfig,

    /// Fusion strategy to use
    pub fusion_strategy: FusionStrategy,

    /// Whether to use adaptive weighting
    pub adaptive_weights: bool,

    /// Initial weight for semantic search (0.0 to 1.0)
    pub semantic_weight: f32,

    /// Whether to run retrievers in parallel
    pub parallel_retrieval: bool,

    /// Minimum confidence score to include results
    pub min_confidence: f32,

    /// Enable query analysis for better routing
    pub enable_query_analysis: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            bm25_config: BM25Config::default(),
            semantic_config: SemanticConfig::default(),
            fusion_strategy: FusionStrategy::ReciprocalRankFusion,
            adaptive_weights: true,
            semantic_weight: 0.6,
            parallel_retrieval: true,
            min_confidence: 0.0,
            enable_query_analysis: true,
        }
    }
}

/// Fusion strategies for combining results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    ReciprocalRankFusion,

    /// Weighted linear combination
    WeightedCombination,

    /// Learned fusion with ML model
    LearnedFusion,

    /// Custom fusion function
    Custom,
}

/// Query characteristics for adaptive routing
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QueryCharacteristics {
    /// Number of tokens in query
    num_tokens: usize,

    /// Contains named entities
    has_entities: bool,

    /// Is a question
    is_question: bool,

    /// Contains technical terms
    has_technical_terms: bool,

    /// Query complexity score
    complexity: f32,
}

/// Hybrid retriever combining multiple strategies
pub struct HybridRetriever {
    /// Configuration
    config: Arc<HybridConfig>,

    /// BM25 keyword retriever
    bm25_retriever: Arc<BM25Retriever>,

    /// Semantic vector retriever
    semantic_retriever: Arc<SemanticRetriever>,

    /// Fusion algorithm
    fusion: Arc<dyn RankFusion>,

    /// Adaptive weight history
    weight_history: Arc<RwLock<Vec<(f32, f32)>>>, // (semantic_weight, performance_score)

    /// Query performance metrics
    query_metrics: Arc<RwLock<Vec<QueryMetrics>>>,
}

/// Query performance metrics for learning
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct QueryMetrics {
    query: String,
    characteristics: QueryCharacteristics,
    semantic_weight_used: f32,
    response_time_ms: u64,
    user_satisfaction: Option<f32>, // Optional user feedback
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new(config: HybridConfig, embedding_service: Arc<dyn EmbeddingProvider>) -> Self {
        let bm25_retriever = Arc::new(BM25Retriever::new(config.bm25_config.clone()));
        let semantic_retriever = Arc::new(SemanticRetriever::new(
            config.semantic_config.clone(),
            embedding_service,
        ));

        let fusion: Arc<dyn RankFusion> = match &config.fusion_strategy {
            FusionStrategy::ReciprocalRankFusion => Arc::new(ReciprocalRankFusion::default()),
            FusionStrategy::WeightedCombination => Arc::new(WeightedFusion::new(vec![
                1.0 - config.semantic_weight,
                config.semantic_weight,
            ])),
            _ => Arc::new(ReciprocalRankFusion::default()),
        };

        Self {
            config: Arc::new(config),
            bm25_retriever,
            semantic_retriever,
            fusion,
            weight_history: Arc::new(RwLock::new(Vec::new())),
            query_metrics: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Index a document in both retrievers
    pub async fn index_document(&self, doc: &Document) -> RragResult<()> {
        if self.config.parallel_retrieval {
            // Index in parallel
            let (bm25_result, semantic_result) = tokio::join!(
                self.bm25_retriever.index_document(doc),
                self.semantic_retriever.index_document(doc)
            );

            bm25_result?;
            semantic_result?;
        } else {
            // Index sequentially
            self.bm25_retriever.index_document(doc).await?;
            self.semantic_retriever.index_document(doc).await?;
        }

        Ok(())
    }

    /// Batch index multiple documents
    pub async fn index_batch(&self, documents: Vec<Document>) -> RragResult<()> {
        if self.config.parallel_retrieval {
            let (bm25_result, semantic_result) = tokio::join!(
                self.bm25_retriever.index_batch(documents.clone()),
                self.semantic_retriever.index_batch(documents)
            );

            bm25_result?;
            semantic_result?;
        } else {
            self.bm25_retriever.index_batch(documents.clone()).await?;
            self.semantic_retriever.index_batch(documents).await?;
        }

        Ok(())
    }

    /// Perform hybrid search
    pub async fn search(&self, query: &str, limit: usize) -> RragResult<Vec<SearchResult>> {
        let start_time = Instant::now();

        // Analyze query characteristics
        let characteristics = if self.config.enable_query_analysis {
            self.analyze_query(query)
        } else {
            self.simple_query_analysis(query)
        };

        // Determine weights based on query characteristics and history
        let semantic_weight = if self.config.adaptive_weights {
            self.calculate_adaptive_weight(&characteristics).await
        } else {
            self.config.semantic_weight
        };

        // Perform searches
        let (bm25_results, semantic_results) = if self.config.parallel_retrieval {
            tokio::join!(
                self.bm25_retriever.search(query, limit * 2),
                self.semantic_retriever
                    .search(query, limit * 2, Some(self.config.min_confidence))
            )
        } else {
            let bm25 = self.bm25_retriever.search(query, limit * 2).await;
            let semantic = self
                .semantic_retriever
                .search(query, limit * 2, Some(self.config.min_confidence))
                .await;
            (bm25, semantic)
        };

        let bm25_results = bm25_results?;
        let semantic_results = semantic_results?;

        // Combine results using fusion strategy
        let fused_results = match self.config.fusion_strategy {
            FusionStrategy::WeightedCombination => {
                let fusion = WeightedFusion::new(vec![1.0 - semantic_weight, semantic_weight]);
                fusion.fuse(vec![bm25_results, semantic_results], limit)?
            }
            _ => self
                .fusion
                .fuse(vec![bm25_results, semantic_results], limit)?,
        };

        // Record metrics
        let elapsed = start_time.elapsed().as_millis() as u64;
        let metrics = QueryMetrics {
            query: query.to_string(),
            characteristics,
            semantic_weight_used: semantic_weight,
            response_time_ms: elapsed,
            user_satisfaction: None,
        };

        let mut query_metrics = self.query_metrics.write().await;
        query_metrics.push(metrics);

        Ok(fused_results)
    }

    /// Advanced search with multiple strategies
    pub async fn advanced_search(
        &self,
        query: &str,
        limit: usize,
        strategies: Vec<SearchStrategy>,
    ) -> RragResult<Vec<SearchResult>> {
        let mut all_results = Vec::new();

        for strategy in strategies {
            let results = match strategy {
                SearchStrategy::ExactMatch => {
                    // Boost BM25 for exact matches
                    self.bm25_retriever.search(query, limit).await?
                }
                SearchStrategy::Semantic => {
                    // Pure semantic search
                    self.semantic_retriever.search(query, limit, None).await?
                }
                SearchStrategy::Hybrid => {
                    // Standard hybrid search
                    self.search(query, limit).await?
                }
                SearchStrategy::QueryExpansion => {
                    // Expand query with synonyms and search
                    let expanded = self.expand_query(query);
                    self.search(&expanded, limit).await?
                }
            };

            all_results.push(results);
        }

        // Fuse all strategy results
        self.fusion.fuse(all_results, limit)
    }

    /// Analyze query characteristics
    fn analyze_query(&self, query: &str) -> QueryCharacteristics {
        let tokens: Vec<&str> = query.split_whitespace().collect();
        let num_tokens = tokens.len();

        // Check if it's a question
        let is_question = query.contains('?')
            || query.starts_with("what")
            || query.starts_with("how")
            || query.starts_with("why")
            || query.starts_with("when")
            || query.starts_with("where")
            || query.starts_with("who");

        // Simple entity detection (could use NER model)
        let has_entities = tokens
            .iter()
            .any(|t| t.chars().next().map_or(false, |c| c.is_uppercase()));

        // Technical term detection (simplified)
        let technical_terms = [
            "algorithm",
            "function",
            "method",
            "system",
            "protocol",
            "framework",
        ];
        let has_technical_terms = tokens
            .iter()
            .any(|t| technical_terms.contains(&t.to_lowercase().as_str()));

        // Calculate complexity
        let complexity = (num_tokens as f32 / 10.0).min(1.0);

        QueryCharacteristics {
            num_tokens,
            has_entities,
            is_question,
            has_technical_terms,
            complexity,
        }
    }

    /// Simple query analysis without NLP
    fn simple_query_analysis(&self, query: &str) -> QueryCharacteristics {
        let num_tokens = query.split_whitespace().count();

        QueryCharacteristics {
            num_tokens,
            has_entities: false,
            is_question: query.contains('?'),
            has_technical_terms: false,
            complexity: (num_tokens as f32 / 10.0).min(1.0),
        }
    }

    /// Calculate adaptive weight based on query characteristics and history
    async fn calculate_adaptive_weight(&self, characteristics: &QueryCharacteristics) -> f32 {
        let mut base_weight = self.config.semantic_weight;

        // Adjust based on query characteristics
        if characteristics.is_question {
            base_weight += 0.1; // Questions benefit from semantic understanding
        }

        if characteristics.has_entities {
            base_weight -= 0.1; // Named entities benefit from exact matching
        }

        if characteristics.has_technical_terms {
            base_weight -= 0.05; // Technical terms often need exact matches
        }

        // Adjust based on query complexity
        base_weight += characteristics.complexity * 0.1;

        // Learn from history if available
        let history = self.weight_history.read().await;
        if history.len() > 10 {
            // Simple moving average of successful weights
            let recent_weights: Vec<f32> = history
                .iter()
                .rev()
                .take(10)
                .filter(|(_, score)| *score > 0.7)
                .map(|(weight, _)| *weight)
                .collect();

            if !recent_weights.is_empty() {
                let avg_weight: f32 =
                    recent_weights.iter().sum::<f32>() / recent_weights.len() as f32;
                base_weight = 0.7 * base_weight + 0.3 * avg_weight;
            }
        }

        // Clamp to valid range
        base_weight.max(0.0).min(1.0)
    }

    /// Expand query with synonyms and related terms
    fn expand_query(&self, query: &str) -> String {
        // Simple query expansion (in production, use WordNet or embeddings)
        let expansions = vec![
            ("ML", "machine learning"),
            ("AI", "artificial intelligence"),
            ("NLP", "natural language processing"),
            ("DB", "database"),
        ];

        let mut expanded = query.to_string();
        for (abbr, full) in expansions {
            if query.contains(abbr) && !query.contains(full) {
                expanded.push_str(&format!(" {}", full));
            }
        }

        expanded
    }

    /// Record user feedback for learning
    pub async fn record_feedback(&self, query: &str, satisfaction: f32) -> RragResult<()> {
        let mut metrics = self.query_metrics.write().await;

        // Find the most recent query matching this text
        if let Some(metric) = metrics.iter_mut().rev().find(|m| m.query == query) {
            metric.user_satisfaction = Some(satisfaction);

            // Update weight history if satisfied
            if satisfaction > 0.7 {
                let mut history = self.weight_history.write().await;
                history.push((metric.semantic_weight_used, satisfaction));

                // Keep only recent history
                if history.len() > 100 {
                    history.drain(0..50);
                }
            }
        }

        Ok(())
    }

    /// Get retrieval statistics
    pub async fn stats(&self) -> HybridStats {
        let bm25_stats = self.bm25_retriever.stats().await;
        let semantic_stats = self.semantic_retriever.stats().await;
        let metrics = self.query_metrics.read().await;

        let avg_response_time = if metrics.is_empty() {
            0
        } else {
            metrics.iter().map(|m| m.response_time_ms).sum::<u64>() / metrics.len() as u64
        };

        HybridStats {
            bm25_stats,
            semantic_stats,
            total_queries: metrics.len(),
            avg_response_time_ms: avg_response_time,
            fusion_strategy: format!("{:?}", self.config.fusion_strategy),
        }
    }
}

/// Search strategies for advanced search
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exact keyword matching
    ExactMatch,
    /// Pure semantic search
    Semantic,
    /// Hybrid search
    Hybrid,
    /// Query expansion with synonyms
    QueryExpansion,
}

/// Hybrid retriever statistics
#[derive(Debug, Serialize)]
pub struct HybridStats {
    pub bm25_stats: std::collections::HashMap<String, serde_json::Value>,
    pub semantic_stats: std::collections::HashMap<String, serde_json::Value>,
    pub total_queries: usize,
    pub avg_response_time_ms: u64,
    pub fusion_strategy: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingService;

    #[tokio::test]
    async fn test_hybrid_search() {
        let config = HybridConfig::default();
        let embedding_service = Arc::new(MockEmbeddingService::new());
        let retriever = HybridRetriever::new(config, embedding_service);

        let docs = vec![
            Document::with_id("1", "The quick brown fox jumps over the lazy dog"),
            Document::with_id(
                "2",
                "Machine learning is a subset of artificial intelligence",
            ),
            Document::with_id(
                "3",
                "Natural language processing enables text understanding",
            ),
        ];

        retriever.index_batch(docs).await.unwrap();

        let results = retriever.search("machine learning AI", 2).await.unwrap();
        assert!(!results.is_empty());

        // Test adaptive weighting
        retriever
            .record_feedback("machine learning AI", 0.9)
            .await
            .unwrap();
    }
}
