//! # Query Expansion Using Graph Structure
//!
//! Leverage knowledge graph structure to expand and enhance queries for improved retrieval.

use super::{algorithms::GraphAlgorithms, KnowledgeGraph};
use crate::RragResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Query expander trait for different expansion strategies
#[async_trait]
pub trait QueryExpander: Send + Sync {
    /// Expand a text query using the knowledge graph
    async fn expand_query(
        &self,
        query: &str,
        options: &ExpansionOptions,
    ) -> RragResult<ExpansionResult>;

    /// Expand query terms using graph structure
    async fn expand_terms(
        &self,
        terms: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<String>>;

    /// Find related entities for query expansion
    async fn find_related_entities(
        &self,
        entities: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<String>>;

    /// Get expansion suggestions for a query
    async fn get_suggestions(&self, query: &str, max_suggestions: usize)
        -> RragResult<Vec<String>>;
}

/// Graph-based query expander
pub struct GraphQueryExpander {
    /// Knowledge graph
    graph: KnowledgeGraph,

    /// Expansion configuration
    config: ExpansionConfig,

    /// Pre-computed expansion cache
    expansion_cache: tokio::sync::RwLock<HashMap<String, Vec<String>>>,
}

/// Query expansion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionConfig {
    /// Maximum expansion depth in the graph
    pub max_expansion_depth: usize,

    /// Maximum number of expanded terms
    pub max_expansion_terms: usize,

    /// Minimum similarity threshold for term expansion
    pub min_similarity_threshold: f32,

    /// Weights for different expansion strategies
    pub strategy_weights: HashMap<ExpansionStrategy, f32>,

    /// Enable semantic expansion using embeddings
    pub enable_semantic_expansion: bool,

    /// Enable structural expansion using graph topology
    pub enable_structural_expansion: bool,

    /// Enable statistical expansion using co-occurrence
    pub enable_statistical_expansion: bool,

    /// Cache expansion results
    pub enable_caching: bool,

    /// Stop words to avoid in expansion
    pub stop_words: HashSet<String>,
}

/// Expansion options for individual queries
#[derive(Debug, Clone)]
pub struct ExpansionOptions {
    /// Specific expansion strategies to use
    pub strategies: Vec<ExpansionStrategy>,

    /// Maximum number of terms to add
    pub max_terms: Option<usize>,

    /// Minimum confidence for expanded terms
    pub min_confidence: f32,

    /// Focus entities (boost terms related to these)
    pub focus_entities: Vec<String>,

    /// Context for expansion (document domain, etc.)
    pub context: Option<String>,

    /// Whether to include original query terms
    pub include_original: bool,
}

/// Query expansion strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpansionStrategy {
    /// Semantic expansion using entity relationships
    Semantic,

    /// Hierarchical expansion (parent/child concepts)
    Hierarchical,

    /// Similarity-based expansion
    Similarity,

    /// Co-occurrence based expansion
    CoOccurrence,

    /// Synonym expansion
    Synonym,

    /// Entity type expansion
    EntityType,

    /// Path-based expansion (following graph paths)
    PathBased,

    /// PageRank-based expansion (importance-weighted)
    PageRank,

    /// Custom expansion strategy
    Custom(String),
}

/// Expansion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionResult {
    /// Original query
    pub original_query: String,

    /// Expanded query terms
    pub expanded_terms: Vec<ExpandedTerm>,

    /// Expansion statistics
    pub stats: ExpansionStats,

    /// Used expansion strategies
    pub strategies_used: Vec<ExpansionStrategy>,

    /// Expansion confidence score
    pub confidence: f32,
}

/// Expanded term with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedTerm {
    /// The expanded term
    pub term: String,

    /// Expansion strategy that generated this term
    pub strategy: ExpansionStrategy,

    /// Confidence score
    pub confidence: f32,

    /// Weight/importance score
    pub weight: f32,

    /// Source entities that led to this expansion
    pub source_entities: Vec<String>,

    /// Semantic relationship to original query
    pub relationship: Option<String>,
}

/// Expansion statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionStats {
    /// Number of entities found in original query
    pub entities_found: usize,

    /// Number of terms added per strategy
    pub terms_per_strategy: HashMap<String, usize>,

    /// Expansion time in milliseconds
    pub expansion_time_ms: u64,

    /// Graph nodes examined
    pub nodes_examined: usize,

    /// Graph edges examined  
    pub edges_examined: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        let mut strategy_weights = HashMap::new();
        strategy_weights.insert(ExpansionStrategy::Semantic, 1.0);
        strategy_weights.insert(ExpansionStrategy::Hierarchical, 0.8);
        strategy_weights.insert(ExpansionStrategy::Similarity, 0.7);
        strategy_weights.insert(ExpansionStrategy::CoOccurrence, 0.6);
        strategy_weights.insert(ExpansionStrategy::EntityType, 0.5);
        strategy_weights.insert(ExpansionStrategy::PathBased, 0.4);

        let mut stop_words = HashSet::new();
        stop_words.extend(
            vec![
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                "by", "from", "up", "about", "into", "through", "during", "before", "after",
                "above", "below", "between", "among", "this", "that",
            ]
            .into_iter()
            .map(|s| s.to_string()),
        );

        Self {
            max_expansion_depth: 2,
            max_expansion_terms: 20,
            min_similarity_threshold: 0.3,
            strategy_weights,
            enable_semantic_expansion: true,
            enable_structural_expansion: true,
            enable_statistical_expansion: true,
            enable_caching: true,
            stop_words,
        }
    }
}

impl Default for ExpansionOptions {
    fn default() -> Self {
        Self {
            strategies: vec![
                ExpansionStrategy::Semantic,
                ExpansionStrategy::Similarity,
                ExpansionStrategy::CoOccurrence,
            ],
            max_terms: Some(10),
            min_confidence: 0.3,
            focus_entities: Vec::new(),
            context: None,
            include_original: true,
        }
    }
}

impl GraphQueryExpander {
    /// Create a new graph query expander
    pub fn new(graph: KnowledgeGraph, config: ExpansionConfig) -> Self {
        Self {
            graph,
            config,
            expansion_cache: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Update the knowledge graph
    pub async fn update_graph(&mut self, graph: KnowledgeGraph) {
        self.graph = graph;
        // Clear cache when graph changes
        if self.config.enable_caching {
            self.expansion_cache.write().await.clear();
        }
    }

    /// Extract entities from query text using graph nodes
    async fn extract_query_entities(&self, query: &str) -> Vec<String> {
        let mut entities = Vec::new();
        let query_lower = query.to_lowercase();

        // Find matching node labels in the query
        for (_, node) in &self.graph.nodes {
            let label_lower = node.label.to_lowercase();
            if query_lower.contains(&label_lower) && !self.config.stop_words.contains(&label_lower)
            {
                entities.push(node.id.clone());
            }
        }

        entities
    }

    /// Expand using semantic relationships
    async fn semantic_expansion(
        &self,
        entity_ids: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<ExpandedTerm>> {
        let mut expanded_terms = Vec::new();
        let strategy_weight = self
            .config
            .strategy_weights
            .get(&ExpansionStrategy::Semantic)
            .copied()
            .unwrap_or(1.0);

        for entity_id in entity_ids {
            if let Some(_entity_node) = self.graph.get_node(entity_id) {
                // Find semantic relationships
                let semantic_edges: Vec<_> = self
                    .graph
                    .edges
                    .values()
                    .filter(|edge| {
                        (edge.source_id == *entity_id || edge.target_id == *entity_id)
                            && matches!(edge.edge_type, super::EdgeType::Semantic(_))
                    })
                    .collect();

                for edge in semantic_edges {
                    let related_node_id = if edge.source_id == *entity_id {
                        &edge.target_id
                    } else {
                        &edge.source_id
                    };

                    if let Some(related_node) = self.graph.get_node(related_node_id) {
                        let confidence = edge.confidence * strategy_weight;
                        if confidence >= options.min_confidence {
                            let expanded_term = ExpandedTerm {
                                term: related_node.label.clone(),
                                strategy: ExpansionStrategy::Semantic,
                                confidence,
                                weight: edge.weight * strategy_weight,
                                source_entities: vec![entity_id.clone()],
                                relationship: Some(edge.label.clone()),
                            };
                            expanded_terms.push(expanded_term);
                        }
                    }
                }
            }
        }

        Ok(expanded_terms)
    }

    /// Expand using hierarchical relationships
    async fn hierarchical_expansion(
        &self,
        entity_ids: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<ExpandedTerm>> {
        let mut expanded_terms = Vec::new();
        let strategy_weight = self
            .config
            .strategy_weights
            .get(&ExpansionStrategy::Hierarchical)
            .copied()
            .unwrap_or(0.8);

        for entity_id in entity_ids {
            // Find hierarchical edges (parent/child relationships)
            let hierarchical_edges: Vec<_> = self
                .graph
                .edges
                .values()
                .filter(|edge| {
                    (edge.source_id == *entity_id || edge.target_id == *entity_id)
                        && matches!(edge.edge_type, super::EdgeType::Hierarchical)
                })
                .collect();

            for edge in hierarchical_edges {
                let related_node_id = if edge.source_id == *entity_id {
                    &edge.target_id
                } else {
                    &edge.source_id
                };

                if let Some(related_node) = self.graph.get_node(related_node_id) {
                    let confidence = edge.confidence * strategy_weight;
                    if confidence >= options.min_confidence {
                        let expanded_term = ExpandedTerm {
                            term: related_node.label.clone(),
                            strategy: ExpansionStrategy::Hierarchical,
                            confidence,
                            weight: edge.weight * strategy_weight,
                            source_entities: vec![entity_id.clone()],
                            relationship: Some(if edge.source_id == *entity_id {
                                "parent".to_string()
                            } else {
                                "child".to_string()
                            }),
                        };
                        expanded_terms.push(expanded_term);
                    }
                }
            }
        }

        Ok(expanded_terms)
    }

    /// Expand using similarity relationships
    async fn similarity_expansion(
        &self,
        entity_ids: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<ExpandedTerm>> {
        let mut expanded_terms = Vec::new();
        let strategy_weight = self
            .config
            .strategy_weights
            .get(&ExpansionStrategy::Similarity)
            .copied()
            .unwrap_or(0.7);

        for entity_id in entity_ids {
            if let Some(entity_node) = self.graph.get_node(entity_id) {
                // If entity has embedding, find similar nodes by embedding similarity
                if let Some(entity_embedding) = &entity_node.embedding {
                    for (other_id, other_node) in &self.graph.nodes {
                        if other_id == entity_id {
                            continue;
                        }

                        if let Some(other_embedding) = &other_node.embedding {
                            if let Ok(similarity) =
                                entity_embedding.cosine_similarity(other_embedding)
                            {
                                if similarity >= self.config.min_similarity_threshold {
                                    let confidence = similarity * strategy_weight;
                                    if confidence >= options.min_confidence {
                                        let expanded_term = ExpandedTerm {
                                            term: other_node.label.clone(),
                                            strategy: ExpansionStrategy::Similarity,
                                            confidence,
                                            weight: similarity * strategy_weight,
                                            source_entities: vec![entity_id.clone()],
                                            relationship: Some(format!(
                                                "similarity:{:.2}",
                                                similarity
                                            )),
                                        };
                                        expanded_terms.push(expanded_term);
                                    }
                                }
                            }
                        }
                    }
                }

                // Also find explicit similarity edges
                let similarity_edges: Vec<_> = self
                    .graph
                    .edges
                    .values()
                    .filter(|edge| {
                        (edge.source_id == *entity_id || edge.target_id == *entity_id)
                            && matches!(edge.edge_type, super::EdgeType::Similar)
                    })
                    .collect();

                for edge in similarity_edges {
                    let related_node_id = if edge.source_id == *entity_id {
                        &edge.target_id
                    } else {
                        &edge.source_id
                    };

                    if let Some(related_node) = self.graph.get_node(related_node_id) {
                        let confidence = edge.confidence * strategy_weight;
                        if confidence >= options.min_confidence {
                            let expanded_term = ExpandedTerm {
                                term: related_node.label.clone(),
                                strategy: ExpansionStrategy::Similarity,
                                confidence,
                                weight: edge.weight * strategy_weight,
                                source_entities: vec![entity_id.clone()],
                                relationship: Some("explicit_similarity".to_string()),
                            };
                            expanded_terms.push(expanded_term);
                        }
                    }
                }
            }
        }

        Ok(expanded_terms)
    }

    /// Expand using co-occurrence relationships
    async fn cooccurrence_expansion(
        &self,
        entity_ids: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<ExpandedTerm>> {
        let mut expanded_terms = Vec::new();
        let strategy_weight = self
            .config
            .strategy_weights
            .get(&ExpansionStrategy::CoOccurrence)
            .copied()
            .unwrap_or(0.6);

        for entity_id in entity_ids {
            // Find co-occurrence edges
            let cooccurrence_edges: Vec<_> = self
                .graph
                .edges
                .values()
                .filter(|edge| {
                    (edge.source_id == *entity_id || edge.target_id == *entity_id)
                        && matches!(edge.edge_type, super::EdgeType::CoOccurs)
                })
                .collect();

            for edge in cooccurrence_edges {
                let related_node_id = if edge.source_id == *entity_id {
                    &edge.target_id
                } else {
                    &edge.source_id
                };

                if let Some(related_node) = self.graph.get_node(related_node_id) {
                    let confidence = edge.confidence * strategy_weight;
                    if confidence >= options.min_confidence {
                        let expanded_term = ExpandedTerm {
                            term: related_node.label.clone(),
                            strategy: ExpansionStrategy::CoOccurrence,
                            confidence,
                            weight: edge.weight * strategy_weight,
                            source_entities: vec![entity_id.clone()],
                            relationship: Some("co_occurrence".to_string()),
                        };
                        expanded_terms.push(expanded_term);
                    }
                }
            }
        }

        Ok(expanded_terms)
    }

    /// Expand using entity type relationships
    async fn entity_type_expansion(
        &self,
        entity_ids: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<ExpandedTerm>> {
        let mut expanded_terms = Vec::new();
        let strategy_weight = self
            .config
            .strategy_weights
            .get(&ExpansionStrategy::EntityType)
            .copied()
            .unwrap_or(0.5);

        // Group entities by type
        let mut entities_by_type: HashMap<String, Vec<String>> = HashMap::new();
        for entity_id in entity_ids {
            if let Some(entity_node) = self.graph.get_node(entity_id) {
                let type_key = match &entity_node.node_type {
                    super::NodeType::Entity(entity_type) => entity_type.clone(),
                    super::NodeType::Concept => "Concept".to_string(),
                    super::NodeType::Document => "Document".to_string(),
                    super::NodeType::DocumentChunk => "DocumentChunk".to_string(),
                    super::NodeType::Keyword => "Keyword".to_string(),
                    super::NodeType::Custom(custom) => custom.clone(),
                };

                entities_by_type
                    .entry(type_key)
                    .or_default()
                    .push(entity_id.clone());
            }
        }

        // For each type, find other entities of the same type
        for (entity_type, type_entities) in entities_by_type {
            let similar_type_nodes: Vec<_> = self
                .graph
                .nodes
                .values()
                .filter(|node| {
                    let node_type_key = match &node.node_type {
                        super::NodeType::Entity(et) => et.clone(),
                        super::NodeType::Concept => "Concept".to_string(),
                        super::NodeType::Document => "Document".to_string(),
                        super::NodeType::DocumentChunk => "DocumentChunk".to_string(),
                        super::NodeType::Keyword => "Keyword".to_string(),
                        super::NodeType::Custom(custom) => custom.clone(),
                    };
                    node_type_key == entity_type && !type_entities.contains(&node.id)
                })
                .collect();

            for node in similar_type_nodes.into_iter().take(5) {
                // Limit to avoid too many results
                let confidence = strategy_weight * 0.5; // Lower confidence for type-based expansion
                if confidence >= options.min_confidence {
                    let expanded_term = ExpandedTerm {
                        term: node.label.clone(),
                        strategy: ExpansionStrategy::EntityType,
                        confidence,
                        weight: strategy_weight * 0.5,
                        source_entities: type_entities.clone(),
                        relationship: Some(format!("same_type:{}", entity_type)),
                    };
                    expanded_terms.push(expanded_term);
                }
            }
        }

        Ok(expanded_terms)
    }

    /// Expand using graph paths
    async fn path_based_expansion(
        &self,
        entity_ids: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<ExpandedTerm>> {
        let mut expanded_terms = Vec::new();
        let strategy_weight = self
            .config
            .strategy_weights
            .get(&ExpansionStrategy::PathBased)
            .copied()
            .unwrap_or(0.4);

        // Use BFS to find nodes within expansion depth
        for entity_id in entity_ids {
            let traversal_config = super::algorithms::TraversalConfig {
                max_depth: self.config.max_expansion_depth,
                max_nodes: 50, // Limit to avoid performance issues
                ..Default::default()
            };

            if let Ok(visited_nodes) =
                GraphAlgorithms::bfs_search(&self.graph, entity_id, &traversal_config)
            {
                for visited_node_id in visited_nodes.iter().skip(1) {
                    // Skip the source node
                    if let Some(visited_node) = self.graph.get_node(visited_node_id) {
                        // Calculate confidence based on distance from source
                        let distance = visited_nodes
                            .iter()
                            .position(|id| id == visited_node_id)
                            .unwrap_or(0);
                        let distance_factor = 1.0 / (distance as f32 + 1.0);
                        let confidence = strategy_weight * distance_factor;

                        if confidence >= options.min_confidence {
                            let expanded_term = ExpandedTerm {
                                term: visited_node.label.clone(),
                                strategy: ExpansionStrategy::PathBased,
                                confidence,
                                weight: confidence,
                                source_entities: vec![entity_id.clone()],
                                relationship: Some(format!("path_distance:{}", distance)),
                            };
                            expanded_terms.push(expanded_term);
                        }
                    }
                }
            }
        }

        Ok(expanded_terms)
    }

    /// Apply focus entity boosting
    fn apply_focus_boosting(&self, terms: &mut [ExpandedTerm], focus_entities: &[String]) {
        if focus_entities.is_empty() {
            return;
        }

        for term in terms {
            // Boost terms that are related to focus entities
            let is_related = term
                .source_entities
                .iter()
                .any(|source| focus_entities.contains(source));

            if is_related {
                term.confidence *= 1.5;
                term.weight *= 1.5;
            }
        }
    }

    /// Deduplicate and rank expanded terms
    fn deduplicate_and_rank(&self, terms: &mut Vec<ExpandedTerm>, max_terms: Option<usize>) {
        // Remove duplicates by term text, keeping the one with highest confidence
        let mut seen_terms: HashMap<String, usize> = HashMap::new();
        let mut unique_terms: Vec<ExpandedTerm> = Vec::new();

        for term in terms.drain(..) {
            match seen_terms.get(&term.term) {
                Some(&existing_index) => {
                    if term.confidence > unique_terms[existing_index].confidence {
                        unique_terms[existing_index] = term;
                    }
                }
                None => {
                    seen_terms.insert(term.term.clone(), unique_terms.len());
                    unique_terms.push(term);
                }
            }
        }

        // Sort by weight (descending) then by confidence
        unique_terms.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Limit results
        if let Some(limit) = max_terms {
            unique_terms.truncate(limit);
        }

        *terms = unique_terms;
    }
}

#[async_trait]
impl QueryExpander for GraphQueryExpander {
    async fn expand_query(
        &self,
        query: &str,
        options: &ExpansionOptions,
    ) -> RragResult<ExpansionResult> {
        let start_time = std::time::Instant::now();

        // Check cache first
        if self.config.enable_caching {
            let cache_key = format!("{}:{:?}", query, options.strategies);
            if let Some(cached_terms) = self.expansion_cache.read().await.get(&cache_key) {
                let result = ExpansionResult {
                    original_query: query.to_string(),
                    expanded_terms: cached_terms
                        .iter()
                        .map(|term| ExpandedTerm {
                            term: term.clone(),
                            strategy: ExpansionStrategy::Semantic, // Default for cached results
                            confidence: 0.8,
                            weight: 0.8,
                            source_entities: Vec::new(),
                            relationship: None,
                        })
                        .collect(),
                    stats: ExpansionStats {
                        entities_found: 0,
                        terms_per_strategy: HashMap::new(),
                        expansion_time_ms: start_time.elapsed().as_millis() as u64,
                        nodes_examined: 0,
                        edges_examined: 0,
                    },
                    strategies_used: options.strategies.clone(),
                    confidence: 0.8,
                };
                return Ok(result);
            }
        }

        // Extract entities from query
        let entity_ids = self.extract_query_entities(query).await;
        let mut expanded_terms = Vec::new();
        let mut terms_per_strategy = HashMap::new();
        let mut nodes_examined = 0;
        let mut edges_examined = 0;

        // Apply expansion strategies
        for strategy in &options.strategies {
            let strategy_terms = match strategy {
                ExpansionStrategy::Semantic if self.config.enable_semantic_expansion => {
                    self.semantic_expansion(&entity_ids, options).await?
                }
                ExpansionStrategy::Hierarchical if self.config.enable_structural_expansion => {
                    self.hierarchical_expansion(&entity_ids, options).await?
                }
                ExpansionStrategy::Similarity => {
                    self.similarity_expansion(&entity_ids, options).await?
                }
                ExpansionStrategy::CoOccurrence if self.config.enable_statistical_expansion => {
                    self.cooccurrence_expansion(&entity_ids, options).await?
                }
                ExpansionStrategy::EntityType => {
                    self.entity_type_expansion(&entity_ids, options).await?
                }
                ExpansionStrategy::PathBased if self.config.enable_structural_expansion => {
                    self.path_based_expansion(&entity_ids, options).await?
                }
                _ => Vec::new(), // Strategy not enabled or supported
            };

            terms_per_strategy.insert(strategy.to_string(), strategy_terms.len());
            expanded_terms.extend(strategy_terms);

            // Update examination counters (simplified)
            nodes_examined += entity_ids.len();
            edges_examined += entity_ids.len() * 5; // Rough estimate
        }

        // Apply focus entity boosting if specified
        self.apply_focus_boosting(&mut expanded_terms, &options.focus_entities);

        // Deduplicate and rank
        self.deduplicate_and_rank(&mut expanded_terms, options.max_terms);

        // Add original query terms if requested
        if options.include_original {
            let original_terms: Vec<_> = query
                .split_whitespace()
                .filter(|term| !self.config.stop_words.contains(&term.to_lowercase()))
                .map(|term| ExpandedTerm {
                    term: term.to_string(),
                    strategy: ExpansionStrategy::Custom("original".to_string()),
                    confidence: 1.0,
                    weight: 1.0,
                    source_entities: Vec::new(),
                    relationship: Some("original_query".to_string()),
                })
                .collect();

            expanded_terms.splice(0..0, original_terms);
        }

        // Calculate overall confidence
        let confidence = if !expanded_terms.is_empty() {
            expanded_terms.iter().map(|t| t.confidence).sum::<f32>() / expanded_terms.len() as f32
        } else {
            0.0
        };

        // Cache results if enabled
        if self.config.enable_caching {
            let cache_key = format!("{}:{:?}", query, options.strategies);
            let cache_terms: Vec<_> = expanded_terms.iter().map(|t| t.term.clone()).collect();
            self.expansion_cache
                .write()
                .await
                .insert(cache_key, cache_terms);
        }

        let expansion_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExpansionResult {
            original_query: query.to_string(),
            expanded_terms,
            stats: ExpansionStats {
                entities_found: entity_ids.len(),
                terms_per_strategy,
                expansion_time_ms,
                nodes_examined,
                edges_examined,
            },
            strategies_used: options.strategies.clone(),
            confidence,
        })
    }

    async fn expand_terms(
        &self,
        terms: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<String>> {
        let combined_query = terms.join(" ");
        let expansion_result = self.expand_query(&combined_query, options).await?;
        Ok(expansion_result
            .expanded_terms
            .into_iter()
            .map(|t| t.term)
            .collect())
    }

    async fn find_related_entities(
        &self,
        entities: &[String],
        options: &ExpansionOptions,
    ) -> RragResult<Vec<String>> {
        // Find entity IDs matching the given entity names
        let entity_ids: Vec<_> = entities
            .iter()
            .filter_map(|entity_name| {
                self.graph
                    .nodes
                    .values()
                    .find(|node| node.label.eq_ignore_ascii_case(entity_name))
                    .map(|node| node.id.clone())
            })
            .collect();

        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Use semantic expansion to find related entities
        let expanded_terms = self.semantic_expansion(&entity_ids, options).await?;
        Ok(expanded_terms.into_iter().map(|t| t.term).collect())
    }

    async fn get_suggestions(
        &self,
        query: &str,
        max_suggestions: usize,
    ) -> RragResult<Vec<String>> {
        let options = ExpansionOptions {
            strategies: vec![ExpansionStrategy::Semantic, ExpansionStrategy::Similarity],
            max_terms: Some(max_suggestions),
            min_confidence: 0.2, // Lower threshold for suggestions
            ..Default::default()
        };

        let expansion_result = self.expand_query(query, &options).await?;
        Ok(expansion_result
            .expanded_terms
            .into_iter()
            .map(|t| t.term)
            .collect())
    }
}

impl std::fmt::Display for ExpansionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpansionStrategy::Semantic => write!(f, "semantic"),
            ExpansionStrategy::Hierarchical => write!(f, "hierarchical"),
            ExpansionStrategy::Similarity => write!(f, "similarity"),
            ExpansionStrategy::CoOccurrence => write!(f, "co_occurrence"),
            ExpansionStrategy::Synonym => write!(f, "synonym"),
            ExpansionStrategy::EntityType => write!(f, "entity_type"),
            ExpansionStrategy::PathBased => write!(f, "path_based"),
            ExpansionStrategy::PageRank => write!(f, "pagerank"),
            ExpansionStrategy::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_retrieval::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn create_test_graph() -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();

        // Add nodes
        let node1 = GraphNode::new("machine learning", NodeType::Concept);
        let node2 = GraphNode::new("artificial intelligence", NodeType::Concept);
        let node3 = GraphNode::new("deep learning", NodeType::Concept);
        let node4 = GraphNode::new("neural networks", NodeType::Concept);

        let node1_id = node1.id.clone();
        let node2_id = node2.id.clone();
        let node3_id = node3.id.clone();
        let node4_id = node4.id.clone();

        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();
        graph.add_node(node3).unwrap();
        graph.add_node(node4).unwrap();

        // Add semantic relationships
        graph
            .add_edge(
                GraphEdge::new(
                    node3_id.clone(),
                    node1_id.clone(),
                    "is_a",
                    EdgeType::Semantic("is_a".to_string()),
                )
                .with_confidence(0.9)
                .with_weight(0.9),
            )
            .unwrap();

        graph
            .add_edge(
                GraphEdge::new(
                    node1_id.clone(),
                    node2_id.clone(),
                    "part_of",
                    EdgeType::Semantic("part_of".to_string()),
                )
                .with_confidence(0.8)
                .with_weight(0.8),
            )
            .unwrap();

        graph
            .add_edge(
                GraphEdge::new(
                    node4_id.clone(),
                    node3_id.clone(),
                    "used_in",
                    EdgeType::Semantic("used_in".to_string()),
                )
                .with_confidence(0.7)
                .with_weight(0.7),
            )
            .unwrap();

        graph
    }

    #[tokio::test]
    async fn test_query_expansion() {
        let graph = create_test_graph();
        let config = ExpansionConfig::default();
        let expander = GraphQueryExpander::new(graph, config);

        let options = ExpansionOptions {
            strategies: vec![ExpansionStrategy::Semantic],
            max_terms: Some(5),
            min_confidence: 0.3,
            ..Default::default()
        };

        let result = expander
            .expand_query("machine learning", &options)
            .await
            .unwrap();

        assert!(!result.expanded_terms.is_empty());
        assert!(result.stats.entities_found > 0);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_semantic_expansion() {
        let graph = create_test_graph();
        let config = ExpansionConfig::default();
        let expander = GraphQueryExpander::new(graph.clone(), config);

        // Find the machine learning node ID
        let ml_node_id = graph
            .nodes
            .values()
            .find(|node| node.label == "machine learning")
            .unwrap()
            .id
            .clone();

        let options = ExpansionOptions::default();
        let expanded_terms = expander
            .semantic_expansion(&[ml_node_id], &options)
            .await
            .unwrap();

        // Should find related terms through semantic relationships
        assert!(!expanded_terms.is_empty());

        // Check that we found "artificial intelligence" and "deep learning"
        let term_texts: Vec<_> = expanded_terms.iter().map(|t| &t.term).collect();
        assert!(
            term_texts.contains(&&"artificial intelligence".to_string())
                || term_texts.contains(&&"deep learning".to_string())
        );
    }

    #[tokio::test]
    async fn test_term_expansion() {
        let graph = create_test_graph();
        let config = ExpansionConfig::default();
        let expander = GraphQueryExpander::new(graph, config);

        let options = ExpansionOptions::default();
        let expanded_terms = expander
            .expand_terms(&["machine learning".to_string()], &options)
            .await
            .unwrap();

        assert!(!expanded_terms.is_empty());

        // Should include related AI terms
        let has_ai_terms = expanded_terms.iter().any(|term| {
            term.contains("artificial") || term.contains("deep") || term.contains("neural")
        });
        assert!(has_ai_terms);
    }

    #[tokio::test]
    async fn test_get_suggestions() {
        let graph = create_test_graph();
        let config = ExpansionConfig::default();
        let expander = GraphQueryExpander::new(graph, config);

        let suggestions = expander.get_suggestions("machine", 3).await.unwrap();

        // Should return some suggestions for the partial query
        assert!(!suggestions.is_empty());
        assert!(suggestions.len() <= 3);
    }
}
