//! # Graph Retrieval Configuration
//!
//! Centralized configuration structures for the graph-based retrieval system.

use super::{
    algorithms::PageRankConfig, entity::EntityExtractionConfig, query_expansion::ExpansionConfig,
    storage::GraphStorageConfig,
};
use serde::{Deserialize, Serialize};

/// Main configuration for graph-based retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Entity extraction configuration
    pub entity_extraction: EntityExtractionConfig,

    /// Graph storage configuration
    pub storage: GraphStorageConfig,

    /// Query expansion configuration
    pub query_expansion: ExpansionConfig,

    /// Algorithm configurations
    pub algorithms: AlgorithmConfig,

    /// Performance configuration
    pub performance: PerformanceConfig,

    /// Feature flags
    pub features: FeatureFlags,
}

/// Algorithm-specific configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// PageRank algorithm configuration
    pub pagerank: PageRankConfig,

    /// Graph traversal limits
    pub traversal: TraversalConfig,

    /// Similarity computation settings
    pub similarity: SimilarityConfig,

    /// Path-finding configuration
    pub pathfinding: PathFindingConfig,
}

/// Graph traversal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalConfig {
    /// Maximum depth for graph traversal
    pub max_depth: usize,

    /// Maximum number of nodes to visit
    pub max_nodes: usize,

    /// Maximum distance for weighted traversals
    pub max_distance: f32,

    /// Enable early termination optimizations
    pub enable_early_termination: bool,
}

/// Similarity computation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Default similarity metric
    pub default_metric: SimilarityMetric,

    /// Threshold for considering nodes similar
    pub similarity_threshold: f32,

    /// Enable embedding-based similarity
    pub enable_embedding_similarity: bool,

    /// Enable structural similarity
    pub enable_structural_similarity: bool,

    /// Weights for different similarity factors
    pub similarity_weights: SimilarityWeights,
}

/// Path-finding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathFindingConfig {
    /// Maximum path length to consider
    pub max_path_length: usize,

    /// Maximum number of paths to find
    pub max_paths: usize,

    /// Minimum path score threshold
    pub min_path_score: f32,

    /// Enable bidirectional search
    pub enable_bidirectional_search: bool,

    /// Path scoring method
    pub scoring_method: PathScoringMethod,
}

/// Performance-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub enable_parallel_processing: bool,

    /// Number of worker threads
    pub num_workers: usize,

    /// Batch size for bulk operations
    pub batch_size: usize,

    /// Cache size limits
    pub cache_limits: CacheLimits,

    /// Memory usage limits
    pub memory_limits: MemoryLimits,

    /// Timeout settings
    pub timeouts: TimeoutConfig,
}

/// Feature flags for enabling/disabling functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable entity extraction
    pub entity_extraction: bool,

    /// Enable relationship extraction
    pub relationship_extraction: bool,

    /// Enable query expansion
    pub query_expansion: bool,

    /// Enable PageRank scoring
    pub pagerank_scoring: bool,

    /// Enable path-based retrieval
    pub path_based_retrieval: bool,

    /// Enable result diversification
    pub result_diversification: bool,

    /// Enable semantic search
    pub semantic_search: bool,

    /// Enable graph-based re-ranking
    pub graph_reranking: bool,

    /// Enable incremental updates
    pub incremental_updates: bool,

    /// Enable distributed processing
    pub distributed_processing: bool,
}

/// Similarity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Cosine similarity
    Cosine,

    /// Euclidean distance (converted to similarity)
    Euclidean,

    /// Jaccard similarity
    Jaccard,

    /// Dice coefficient
    Dice,

    /// Custom similarity function
    Custom(String),
}

/// Weights for different similarity factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityWeights {
    /// Weight for content similarity
    pub content: f32,

    /// Weight for structural similarity
    pub structural: f32,

    /// Weight for semantic similarity
    pub semantic: f32,

    /// Weight for temporal similarity
    pub temporal: f32,

    /// Weight for metadata similarity
    pub metadata: f32,
}

/// Path scoring methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathScoringMethod {
    /// Simple path length-based scoring
    Length,

    /// Edge weight-based scoring
    EdgeWeight,

    /// PageRank-based scoring
    PageRank,

    /// Combined scoring using multiple factors
    Combined(Vec<PathScoringFactor>),
}

/// Factors for path scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathScoringFactor {
    /// Factor type
    pub factor_type: PathFactorType,

    /// Weight of this factor
    pub weight: f32,
}

/// Types of path scoring factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathFactorType {
    /// Path length
    Length,

    /// Average edge weight
    AverageEdgeWeight,

    /// Minimum edge weight
    MinEdgeWeight,

    /// PageRank of nodes in path
    NodePageRank,

    /// Semantic coherence of path
    SemanticCoherence,
}

/// Cache size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLimits {
    /// Maximum number of cached queries
    pub max_cached_queries: usize,

    /// Maximum number of cached PageRank scores
    pub max_cached_pagerank: usize,

    /// Maximum number of cached entity embeddings
    pub max_cached_embeddings: usize,

    /// Maximum number of cached paths
    pub max_cached_paths: usize,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

/// Memory usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum graph size in MB
    pub max_graph_size_mb: usize,

    /// Maximum number of nodes
    pub max_nodes: usize,

    /// Maximum number of edges
    pub max_edges: usize,

    /// Memory threshold for triggering cleanup
    pub cleanup_threshold_mb: usize,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Query timeout in seconds
    pub query_timeout_seconds: u64,

    /// Entity extraction timeout in seconds
    pub extraction_timeout_seconds: u64,

    /// Graph traversal timeout in seconds
    pub traversal_timeout_seconds: u64,

    /// PageRank computation timeout in seconds
    pub pagerank_timeout_seconds: u64,
}

/// Default implementations
impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            entity_extraction: EntityExtractionConfig::default(),
            storage: GraphStorageConfig::default(),
            query_expansion: ExpansionConfig::default(),
            algorithms: AlgorithmConfig::default(),
            performance: PerformanceConfig::default(),
            features: FeatureFlags::default(),
        }
    }
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            pagerank: PageRankConfig::default(),
            traversal: TraversalConfig::default(),
            similarity: SimilarityConfig::default(),
            pathfinding: PathFindingConfig::default(),
        }
    }
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_nodes: 1000,
            max_distance: 10.0,
            enable_early_termination: true,
        }
    }
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            default_metric: SimilarityMetric::Cosine,
            similarity_threshold: 0.7,
            enable_embedding_similarity: true,
            enable_structural_similarity: true,
            similarity_weights: SimilarityWeights::default(),
        }
    }
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            content: 0.4,
            structural: 0.2,
            semantic: 0.3,
            temporal: 0.05,
            metadata: 0.05,
        }
    }
}

impl Default for PathFindingConfig {
    fn default() -> Self {
        Self {
            max_path_length: 6,
            max_paths: 10,
            min_path_score: 0.1,
            enable_bidirectional_search: true,
            scoring_method: PathScoringMethod::Combined(vec![
                PathScoringFactor {
                    factor_type: PathFactorType::Length,
                    weight: 0.3,
                },
                PathScoringFactor {
                    factor_type: PathFactorType::AverageEdgeWeight,
                    weight: 0.4,
                },
                PathScoringFactor {
                    factor_type: PathFactorType::NodePageRank,
                    weight: 0.3,
                },
            ]),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel_processing: true,
            num_workers: num_cpus::get(),
            batch_size: 100,
            cache_limits: CacheLimits::default(),
            memory_limits: MemoryLimits::default(),
            timeouts: TimeoutConfig::default(),
        }
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            entity_extraction: true,
            relationship_extraction: true,
            query_expansion: true,
            pagerank_scoring: true,
            path_based_retrieval: true,
            result_diversification: true,
            semantic_search: true,
            graph_reranking: true,
            incremental_updates: false,
            distributed_processing: false,
        }
    }
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_cached_queries: 1000,
            max_cached_pagerank: 1,
            max_cached_embeddings: 10000,
            max_cached_paths: 5000,
            cache_ttl_seconds: 3600, // 1 hour
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_graph_size_mb: 1024, // 1 GB
            max_nodes: 1_000_000,
            max_edges: 5_000_000,
            cleanup_threshold_mb: 800,
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            query_timeout_seconds: 30,
            extraction_timeout_seconds: 300, // 5 minutes
            traversal_timeout_seconds: 60,
            pagerank_timeout_seconds: 600, // 10 minutes
        }
    }
}

/// Configuration builder for easier setup
pub struct GraphConfigBuilder {
    config: GraphConfig,
}

impl GraphConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: GraphConfig::default(),
        }
    }

    /// Enable/disable entity extraction
    pub fn with_entity_extraction(mut self, enabled: bool) -> Self {
        self.config.features.entity_extraction = enabled;
        self
    }

    /// Set entity extraction confidence threshold
    pub fn with_entity_confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.entity_extraction.min_confidence = threshold;
        self
    }

    /// Enable/disable query expansion
    pub fn with_query_expansion(mut self, enabled: bool) -> Self {
        self.config.features.query_expansion = enabled;
        self
    }

    /// Set maximum expansion terms
    pub fn with_max_expansion_terms(mut self, max_terms: usize) -> Self {
        self.config.query_expansion.max_expansion_terms = max_terms;
        self
    }

    /// Enable/disable PageRank scoring
    pub fn with_pagerank_scoring(mut self, enabled: bool) -> Self {
        self.config.features.pagerank_scoring = enabled;
        self
    }

    /// Set PageRank damping factor
    pub fn with_pagerank_damping_factor(mut self, damping_factor: f32) -> Self {
        self.config.algorithms.pagerank.damping_factor = damping_factor;
        self
    }

    /// Set graph traversal limits
    pub fn with_traversal_limits(mut self, max_depth: usize, max_nodes: usize) -> Self {
        self.config.algorithms.traversal.max_depth = max_depth;
        self.config.algorithms.traversal.max_nodes = max_nodes;
        self
    }

    /// Set similarity threshold
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.config.algorithms.similarity.similarity_threshold = threshold;
        self
    }

    /// Enable/disable parallel processing
    pub fn with_parallel_processing(mut self, enabled: bool) -> Self {
        self.config.performance.enable_parallel_processing = enabled;
        self
    }

    /// Set number of worker threads
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.config.performance.num_workers = num_workers;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.performance.batch_size = batch_size;
        self
    }

    /// Set memory limits
    pub fn with_memory_limits(
        mut self,
        max_graph_size_mb: usize,
        max_nodes: usize,
        max_edges: usize,
    ) -> Self {
        self.config.performance.memory_limits.max_graph_size_mb = max_graph_size_mb;
        self.config.performance.memory_limits.max_nodes = max_nodes;
        self.config.performance.memory_limits.max_edges = max_edges;
        self
    }

    /// Set query timeout
    pub fn with_query_timeout(mut self, timeout_seconds: u64) -> Self {
        self.config.performance.timeouts.query_timeout_seconds = timeout_seconds;
        self
    }

    /// Enable all features (for development/testing)
    pub fn with_all_features(mut self) -> Self {
        self.config.features = FeatureFlags {
            entity_extraction: true,
            relationship_extraction: true,
            query_expansion: true,
            pagerank_scoring: true,
            path_based_retrieval: true,
            result_diversification: true,
            semantic_search: true,
            graph_reranking: true,
            incremental_updates: true,
            distributed_processing: false, // Keep disabled for single-node setup
        };
        self
    }

    /// Enable minimal features (for lightweight deployment)
    pub fn with_minimal_features(mut self) -> Self {
        self.config.features = FeatureFlags {
            entity_extraction: true,
            relationship_extraction: false,
            query_expansion: true,
            pagerank_scoring: false,
            path_based_retrieval: false,
            result_diversification: false,
            semantic_search: true,
            graph_reranking: false,
            incremental_updates: false,
            distributed_processing: false,
        };
        self
    }

    /// Build the configuration
    pub fn build(self) -> GraphConfig {
        self.config
    }
}

impl Default for GraphConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation methods for configuration
impl GraphConfig {
    /// Validate the configuration and return warnings/errors
    pub fn validate(&self) -> Result<Vec<String>, Vec<String>> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Validate entity extraction settings
        if self.features.entity_extraction {
            if self.entity_extraction.min_confidence < 0.0
                || self.entity_extraction.min_confidence > 1.0
            {
                errors.push("Entity extraction confidence must be between 0.0 and 1.0".to_string());
            }

            if self.entity_extraction.max_entity_length == 0 {
                errors.push("Maximum entity length must be greater than 0".to_string());
            }
        }

        // Validate query expansion settings
        if self.features.query_expansion {
            if self.query_expansion.max_expansion_terms == 0 {
                warnings.push(
                    "Maximum expansion terms is 0, query expansion will be ineffective".to_string(),
                );
            }
        }

        // Validate algorithm settings
        if self.algorithms.pagerank.damping_factor < 0.0
            || self.algorithms.pagerank.damping_factor > 1.0
        {
            errors.push("PageRank damping factor must be between 0.0 and 1.0".to_string());
        }

        if self.algorithms.traversal.max_depth == 0 {
            errors.push("Maximum traversal depth must be greater than 0".to_string());
        }

        if self.algorithms.similarity.similarity_threshold < 0.0
            || self.algorithms.similarity.similarity_threshold > 1.0
        {
            errors.push("Similarity threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate performance settings
        if self.performance.num_workers == 0 {
            errors.push("Number of workers must be greater than 0".to_string());
        }

        if self.performance.batch_size == 0 {
            errors.push("Batch size must be greater than 0".to_string());
        }

        // Validate memory limits
        if self.performance.memory_limits.max_nodes == 0 {
            errors.push("Maximum number of nodes must be greater than 0".to_string());
        }

        if self.performance.memory_limits.max_edges == 0 {
            errors.push("Maximum number of edges must be greater than 0".to_string());
        }

        // Check for logical inconsistencies
        if !self.features.entity_extraction && self.features.relationship_extraction {
            warnings.push(
                "Relationship extraction requires entity extraction to be enabled".to_string(),
            );
        }

        if !self.features.pagerank_scoring
            && self.algorithms.pathfinding.scoring_method.uses_pagerank()
        {
            warnings
                .push("Path scoring uses PageRank but PageRank scoring is disabled".to_string());
        }

        if errors.is_empty() {
            Ok(warnings)
        } else {
            Err(errors)
        }
    }
}

impl PathScoringMethod {
    /// Check if this scoring method uses PageRank
    pub fn uses_pagerank(&self) -> bool {
        match self {
            PathScoringMethod::PageRank => true,
            PathScoringMethod::Combined(factors) => factors
                .iter()
                .any(|f| matches!(f.factor_type, PathFactorType::NodePageRank)),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GraphConfig::default();

        // Verify that all features are enabled by default
        assert!(config.features.entity_extraction);
        assert!(config.features.query_expansion);
        assert!(config.features.pagerank_scoring);

        // Verify reasonable default values
        assert!(config.algorithms.pagerank.damping_factor > 0.0);
        assert!(config.algorithms.pagerank.damping_factor < 1.0);
        assert!(config.algorithms.traversal.max_depth > 0);
        assert!(config.performance.batch_size > 0);
    }

    #[test]
    fn test_config_builder() {
        let config = GraphConfigBuilder::new()
            .with_entity_extraction(true)
            .with_entity_confidence_threshold(0.8)
            .with_query_expansion(true)
            .with_max_expansion_terms(15)
            .with_pagerank_scoring(true)
            .with_pagerank_damping_factor(0.9)
            .with_parallel_processing(true)
            .with_num_workers(4)
            .with_batch_size(50)
            .build();

        assert!(config.features.entity_extraction);
        assert_eq!(config.entity_extraction.min_confidence, 0.8);
        assert!(config.features.query_expansion);
        assert_eq!(config.query_expansion.max_expansion_terms, 15);
        assert!(config.features.pagerank_scoring);
        assert_eq!(config.algorithms.pagerank.damping_factor, 0.9);
        assert!(config.performance.enable_parallel_processing);
        assert_eq!(config.performance.num_workers, 4);
        assert_eq!(config.performance.batch_size, 50);
    }

    #[test]
    fn test_config_validation() {
        let mut config = GraphConfig::default();

        // Valid configuration should pass
        let result = config.validate();
        assert!(result.is_ok());

        // Invalid damping factor should fail
        config.algorithms.pagerank.damping_factor = 1.5;
        let result = config.validate();
        assert!(result.is_err());

        // Reset and test another invalid setting
        config.algorithms.pagerank.damping_factor = 0.85;
        config.performance.num_workers = 0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_minimal_and_full_features() {
        let minimal_config = GraphConfigBuilder::new().with_minimal_features().build();

        assert!(minimal_config.features.entity_extraction);
        assert!(!minimal_config.features.relationship_extraction);
        assert!(!minimal_config.features.pagerank_scoring);

        let full_config = GraphConfigBuilder::new().with_all_features().build();

        assert!(full_config.features.entity_extraction);
        assert!(full_config.features.relationship_extraction);
        assert!(full_config.features.pagerank_scoring);
        assert!(full_config.features.incremental_updates);
    }
}
