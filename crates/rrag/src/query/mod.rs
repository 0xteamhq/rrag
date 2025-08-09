//! # Query Processing Module
//! 
//! Advanced query rewriting and expansion capabilities for improved retrieval.
//! Implements state-of-the-art techniques like HyDE, query decomposition, and semantic expansion.

pub mod rewriter;
pub mod expander;
pub mod classifier;
pub mod decomposer;
pub mod hyde;

pub use rewriter::{QueryRewriter, QueryRewriteConfig, RewriteStrategy, RewriteResult};
pub use expander::{QueryExpander, ExpansionConfig, ExpansionStrategy, ExpansionResult};
pub use classifier::{QueryClassifier, QueryIntent, QueryType, ClassificationResult};
pub use decomposer::{QueryDecomposer, DecompositionStrategy, SubQuery};
pub use hyde::{HyDEGenerator, HyDEConfig, HyDEResult};

use crate::{RragResult, EmbeddingProvider};
use std::sync::Arc;

/// Main query processor that orchestrates all query enhancement techniques
pub struct QueryProcessor {
    /// Query rewriter for transforming queries
    rewriter: QueryRewriter,
    
    /// Query expander for adding related terms
    expander: QueryExpander,
    
    /// Query classifier for intent detection
    classifier: QueryClassifier,
    
    /// Query decomposer for breaking down complex queries
    decomposer: QueryDecomposer,
    
    /// HyDE generator for hypothetical document embeddings
    hyde: Option<HyDEGenerator>,
    
    /// Configuration
    config: QueryProcessorConfig,
}

/// Configuration for the query processor
#[derive(Debug, Clone)]
pub struct QueryProcessorConfig {
    /// Whether to enable query rewriting
    pub enable_rewriting: bool,
    
    /// Whether to enable query expansion
    pub enable_expansion: bool,
    
    /// Whether to enable intent classification
    pub enable_classification: bool,
    
    /// Whether to enable query decomposition
    pub enable_decomposition: bool,
    
    /// Whether to enable HyDE
    pub enable_hyde: bool,
    
    /// Maximum number of query variants to generate
    pub max_variants: usize,
    
    /// Confidence threshold for classifications
    pub confidence_threshold: f32,
}

impl Default for QueryProcessorConfig {
    fn default() -> Self {
        Self {
            enable_rewriting: true,
            enable_expansion: true,
            enable_classification: true,
            enable_decomposition: true,
            enable_hyde: true,
            max_variants: 5,
            confidence_threshold: 0.7,
        }
    }
}

/// Complete query processing result
#[derive(Debug, Clone)]
pub struct QueryProcessingResult {
    /// Original query
    pub original_query: String,
    
    /// Rewritten queries
    pub rewritten_queries: Vec<RewriteResult>,
    
    /// Expanded queries with additional terms
    pub expanded_queries: Vec<ExpansionResult>,
    
    /// Query classification results
    pub classification: Option<ClassificationResult>,
    
    /// Decomposed sub-queries
    pub sub_queries: Vec<SubQuery>,
    
    /// HyDE generated hypothetical documents
    pub hyde_results: Vec<HyDEResult>,
    
    /// Final optimized queries for retrieval
    pub final_queries: Vec<String>,
    
    /// Processing metadata
    pub metadata: QueryProcessingMetadata,
}

/// Metadata about query processing
#[derive(Debug, Clone)]
pub struct QueryProcessingMetadata {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Number of techniques applied
    pub techniques_applied: Vec<String>,
    
    /// Confidence scores
    pub confidence_scores: std::collections::HashMap<String, f32>,
    
    /// Any warnings or notes
    pub warnings: Vec<String>,
}

impl QueryProcessor {
    /// Create a new query processor
    pub fn new(config: QueryProcessorConfig) -> Self {
        let rewriter = QueryRewriter::new(QueryRewriteConfig::default());
        let expander = QueryExpander::new(ExpansionConfig::default());
        let classifier = QueryClassifier::new();
        let decomposer = QueryDecomposer::new();
        
        Self {
            rewriter,
            expander,
            classifier,
            decomposer,
            hyde: None,
            config,
        }
    }
    
    /// Create with embedding provider for HyDE support
    pub fn with_embedding_provider(
        mut self, 
        embedding_provider: Arc<dyn EmbeddingProvider>
    ) -> Self {
        if self.config.enable_hyde {
            self.hyde = Some(HyDEGenerator::new(
                HyDEConfig::default(),
                embedding_provider
            ));
        }
        self
    }
    
    /// Process a query through all enabled techniques
    pub async fn process_query(&self, query: &str) -> RragResult<QueryProcessingResult> {
        let start_time = std::time::Instant::now();
        let mut techniques_applied = Vec::new();
        let mut confidence_scores = std::collections::HashMap::new();
        let mut warnings = Vec::new();
        
        // 1. Classify the query intent
        let classification = if self.config.enable_classification {
            techniques_applied.push("classification".to_string());
            let result = self.classifier.classify(query).await?;
            confidence_scores.insert("classification".to_string(), result.confidence);
            Some(result)
        } else {
            None
        };
        
        // 2. Rewrite the query
        let rewritten_queries = if self.config.enable_rewriting {
            techniques_applied.push("rewriting".to_string());
            let results = self.rewriter.rewrite(query).await?;
            if results.is_empty() {
                warnings.push("Query rewriting produced no results".to_string());
            }
            results
        } else {
            Vec::new()
        };
        
        // 3. Expand the query with synonyms and related terms
        let expanded_queries = if self.config.enable_expansion {
            techniques_applied.push("expansion".to_string());
            let results = self.expander.expand(query).await?;
            confidence_scores.insert("expansion".to_string(), 
                results.iter().map(|r| r.confidence).fold(0.0, f32::max));
            results
        } else {
            Vec::new()
        };
        
        // 4. Decompose complex queries
        let sub_queries = if self.config.enable_decomposition {
            techniques_applied.push("decomposition".to_string());
            self.decomposer.decompose(query).await?
        } else {
            Vec::new()
        };
        
        // 5. Generate HyDE hypothetical documents
        let hyde_results = if self.config.enable_hyde && self.hyde.is_some() {
            techniques_applied.push("hyde".to_string());
            let results = self.hyde.as_ref().unwrap().generate(query).await?;
            confidence_scores.insert("hyde".to_string(), 
                results.iter().map(|r| r.confidence).fold(0.0, f32::max));
            results
        } else {
            Vec::new()
        };
        
        // 6. Generate final optimized queries
        let final_queries = self.generate_final_queries(
            query,
            &rewritten_queries,
            &expanded_queries,
            &sub_queries,
            &hyde_results,
            &classification
        );
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(QueryProcessingResult {
            original_query: query.to_string(),
            rewritten_queries,
            expanded_queries,
            classification,
            sub_queries,
            hyde_results,
            final_queries,
            metadata: QueryProcessingMetadata {
                processing_time_ms: processing_time,
                techniques_applied,
                confidence_scores,
                warnings,
            },
        })
    }
    
    /// Generate final optimized queries from all processing results
    fn generate_final_queries(
        &self,
        original: &str,
        rewritten: &[RewriteResult],
        expanded: &[ExpansionResult],
        sub_queries: &[SubQuery],
        hyde: &[HyDEResult],
        classification: &Option<ClassificationResult>,
    ) -> Vec<String> {
        let mut queries = Vec::new();
        
        // Always include the original query
        queries.push(original.to_string());
        
        // Add high-confidence rewritten queries
        for rewrite in rewritten {
            if rewrite.confidence >= self.config.confidence_threshold {
                queries.push(rewrite.rewritten_query.clone());
            }
        }
        
        // Add expanded queries based on intent
        if let Some(classification) = classification {
            match classification.intent {
                QueryIntent::Factual => {
                    // For factual queries, prefer exact matches
                    queries.extend(expanded.iter()
                        .filter(|e| e.expansion_type == ExpansionStrategy::Synonyms)
                        .map(|e| e.expanded_query.clone()));
                }
                QueryIntent::Conceptual => {
                    // For conceptual queries, prefer broader expansions
                    queries.extend(expanded.iter()
                        .filter(|e| e.expansion_type == ExpansionStrategy::Semantic)
                        .map(|e| e.expanded_query.clone()));
                }
                _ => {
                    // Default: include all high-confidence expansions
                    queries.extend(expanded.iter()
                        .filter(|e| e.confidence >= self.config.confidence_threshold)
                        .map(|e| e.expanded_query.clone()));
                }
            }
        } else {
            queries.extend(expanded.iter()
                .filter(|e| e.confidence >= self.config.confidence_threshold)
                .map(|e| e.expanded_query.clone()));
        }
        
        // Add sub-queries for complex queries
        queries.extend(sub_queries.iter().map(|sq| sq.query.clone()));
        
        // Add HyDE queries for semantic search
        queries.extend(hyde.iter()
            .filter(|h| h.confidence >= self.config.confidence_threshold)
            .map(|h| h.hypothetical_answer.clone()));
        
        // Deduplicate and limit
        queries.sort();
        queries.dedup();
        queries.truncate(self.config.max_variants);
        
        queries
    }
}