//! # Query Expander
//!
//! Intelligent query expansion using synonyms, related terms, and semantic similarity.
//! Improves recall by adding relevant terms that might appear in target documents.

use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query expander for adding related terms
pub struct QueryExpander {
    /// Configuration
    config: ExpansionConfig,

    /// Synonym dictionary
    synonyms: HashMap<String, Vec<String>>,

    /// Related terms dictionary
    related_terms: HashMap<String, Vec<String>>,

    /// Domain-specific expansions
    domain_expansions: HashMap<String, HashMap<String, Vec<String>>>,
}

/// Configuration for query expansion
#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    /// Maximum number of synonyms to add
    pub max_synonyms: usize,

    /// Maximum number of related terms to add
    pub max_related_terms: usize,

    /// Enable synonym expansion
    pub enable_synonyms: bool,

    /// Enable related term expansion
    pub enable_related_terms: bool,

    /// Enable semantic expansion
    pub enable_semantic_expansion: bool,

    /// Enable domain-specific expansion
    pub enable_domain_expansion: bool,

    /// Minimum relevance score for expansions
    pub min_relevance_score: f32,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_synonyms: 3,
            max_related_terms: 2,
            enable_synonyms: true,
            enable_related_terms: true,
            enable_semantic_expansion: true,
            enable_domain_expansion: true,
            min_relevance_score: 0.6,
        }
    }
}

/// Expansion strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpansionStrategy {
    /// Add synonyms
    Synonyms,
    /// Add related terms
    RelatedTerms,
    /// Semantic expansion using embeddings
    Semantic,
    /// Domain-specific expansion
    DomainSpecific,
    /// Contextual expansion
    Contextual,
}

/// Result of query expansion
#[derive(Debug, Clone)]
pub struct ExpansionResult {
    /// Original query
    pub original_query: String,

    /// Expanded query
    pub expanded_query: String,

    /// Terms that were added
    pub added_terms: Vec<String>,

    /// Expansion strategy used
    pub expansion_type: ExpansionStrategy,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Relevance scores for added terms
    pub term_scores: HashMap<String, f32>,
}

impl QueryExpander {
    /// Create a new query expander
    pub fn new(config: ExpansionConfig) -> Self {
        let synonyms = Self::init_synonyms();
        let related_terms = Self::init_related_terms();
        let domain_expansions = Self::init_domain_expansions();

        Self {
            config,
            synonyms,
            related_terms,
            domain_expansions,
        }
    }

    /// Expand a query using all enabled strategies
    pub async fn expand(&self, query: &str) -> RragResult<Vec<ExpansionResult>> {
        let mut results = Vec::new();

        // Tokenize query
        let tokens = self.tokenize(query);

        // Apply synonym expansion
        if self.config.enable_synonyms {
            if let Some(result) = self.expand_with_synonyms(query, &tokens) {
                if result.confidence >= self.config.min_relevance_score {
                    results.push(result);
                }
            }
        }

        // Apply related terms expansion
        if self.config.enable_related_terms {
            if let Some(result) = self.expand_with_related_terms(query, &tokens) {
                if result.confidence >= self.config.min_relevance_score {
                    results.push(result);
                }
            }
        }

        // Apply semantic expansion
        if self.config.enable_semantic_expansion {
            if let Some(result) = self.expand_semantically(query, &tokens) {
                if result.confidence >= self.config.min_relevance_score {
                    results.push(result);
                }
            }
        }

        // Apply domain-specific expansion
        if self.config.enable_domain_expansion {
            let domain_results = self.expand_domain_specific(query, &tokens);
            results.extend(
                domain_results
                    .into_iter()
                    .filter(|r| r.confidence >= self.config.min_relevance_score),
            );
        }

        Ok(results)
    }

    /// Expand query with synonyms
    fn expand_with_synonyms(&self, query: &str, tokens: &[String]) -> Option<ExpansionResult> {
        let mut added_terms = Vec::new();
        let mut term_scores = HashMap::new();

        for token in tokens {
            if let Some(synonyms) = self.synonyms.get(&token.to_lowercase()) {
                for synonym in synonyms.iter().take(self.config.max_synonyms) {
                    if !tokens
                        .iter()
                        .any(|t| t.to_lowercase() == synonym.to_lowercase())
                    {
                        added_terms.push(synonym.clone());
                        term_scores.insert(synonym.clone(), 0.8); // Fixed score for synonyms
                    }
                }
            }
        }

        if !added_terms.is_empty() {
            let expanded_query = format!("{} {}", query, added_terms.join(" "));
            Some(ExpansionResult {
                original_query: query.to_string(),
                expanded_query,
                added_terms,
                expansion_type: ExpansionStrategy::Synonyms,
                confidence: 0.8,
                term_scores,
            })
        } else {
            None
        }
    }

    /// Expand query with related terms
    fn expand_with_related_terms(&self, query: &str, tokens: &[String]) -> Option<ExpansionResult> {
        let mut added_terms = Vec::new();
        let mut term_scores = HashMap::new();

        for token in tokens {
            if let Some(related) = self.related_terms.get(&token.to_lowercase()) {
                for term in related.iter().take(self.config.max_related_terms) {
                    if !tokens
                        .iter()
                        .any(|t| t.to_lowercase() == term.to_lowercase())
                    {
                        added_terms.push(term.clone());
                        term_scores.insert(term.clone(), 0.7); // Slightly lower than synonyms
                    }
                }
            }
        }

        if !added_terms.is_empty() {
            let expanded_query = format!("{} {}", query, added_terms.join(" "));
            Some(ExpansionResult {
                original_query: query.to_string(),
                expanded_query,
                added_terms,
                expansion_type: ExpansionStrategy::RelatedTerms,
                confidence: 0.7,
                term_scores,
            })
        } else {
            None
        }
    }

    /// Expand query semantically
    fn expand_semantically(&self, query: &str, _tokens: &[String]) -> Option<ExpansionResult> {
        // For now, implement a simple semantic expansion
        // In production, this would use word embeddings or language models
        let semantic_expansions = self.get_semantic_expansions(query);

        if !semantic_expansions.is_empty() {
            let mut term_scores = HashMap::new();
            for term in &semantic_expansions {
                term_scores.insert(term.clone(), 0.6);
            }

            let expanded_query = format!("{} {}", query, semantic_expansions.join(" "));
            Some(ExpansionResult {
                original_query: query.to_string(),
                expanded_query,
                added_terms: semantic_expansions,
                expansion_type: ExpansionStrategy::Semantic,
                confidence: 0.6,
                term_scores,
            })
        } else {
            None
        }
    }

    /// Apply domain-specific expansions
    fn expand_domain_specific(&self, query: &str, tokens: &[String]) -> Vec<ExpansionResult> {
        let mut results = Vec::new();

        // Detect domain
        let domain = self.detect_domain(tokens);

        if let Some(domain_dict) = self.domain_expansions.get(&domain) {
            for token in tokens {
                if let Some(expansions) = domain_dict.get(&token.to_lowercase()) {
                    let mut term_scores = HashMap::new();
                    for term in expansions {
                        term_scores.insert(term.clone(), 0.75);
                    }

                    let expanded_query = format!("{} {}", query, expansions.join(" "));
                    results.push(ExpansionResult {
                        original_query: query.to_string(),
                        expanded_query,
                        added_terms: expansions.clone(),
                        expansion_type: ExpansionStrategy::DomainSpecific,
                        confidence: 0.75,
                        term_scores,
                    });
                }
            }
        }

        results
    }

    /// Get semantic expansions for a query
    fn get_semantic_expansions(&self, query: &str) -> Vec<String> {
        // Simple rule-based semantic expansion
        // In production, use proper semantic models
        let mut expansions = Vec::new();

        let query_lower = query.to_lowercase();

        if query_lower.contains("learn") || query_lower.contains("study") {
            expansions.extend_from_slice(&["education", "training", "tutorial"]);
        }

        if query_lower.contains("build") || query_lower.contains("create") {
            expansions.extend_from_slice(&["develop", "construct", "implement"]);
        }

        if query_lower.contains("fast") || query_lower.contains("quick") {
            expansions.extend_from_slice(&["rapid", "efficient", "performance"]);
        }

        if query_lower.contains("problem") || query_lower.contains("issue") {
            expansions.extend_from_slice(&["solution", "fix", "troubleshoot"]);
        }

        expansions.into_iter().map(String::from).collect()
    }

    /// Detect the domain of a query
    fn detect_domain(&self, tokens: &[String]) -> String {
        let tech_terms = [
            "code",
            "programming",
            "software",
            "api",
            "database",
            "algorithm",
        ];
        let business_terms = ["market", "sales", "revenue", "customer", "profit"];
        let science_terms = ["research", "study", "experiment", "theory", "analysis"];

        let tokens_lower: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();

        let tech_count = tech_terms
            .iter()
            .filter(|&&term| tokens_lower.iter().any(|t| t.contains(term)))
            .count();
        let business_count = business_terms
            .iter()
            .filter(|&&term| tokens_lower.iter().any(|t| t.contains(term)))
            .count();
        let science_count = science_terms
            .iter()
            .filter(|&&term| tokens_lower.iter().any(|t| t.contains(term)))
            .count();

        if tech_count > business_count && tech_count > science_count {
            "technology".to_string()
        } else if business_count > science_count {
            "business".to_string()
        } else if science_count > 0 {
            "science".to_string()
        } else {
            "general".to_string()
        }
    }

    /// Tokenize query into individual terms
    fn tokenize(&self, query: &str) -> Vec<String> {
        query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() > 2) // Filter out very short words
            .map(String::from)
            .collect()
    }

    /// Initialize synonym dictionary
    fn init_synonyms() -> HashMap<String, Vec<String>> {
        let mut synonyms = HashMap::new();

        // Technology synonyms
        synonyms.insert(
            "fast".to_string(),
            vec![
                "quick".to_string(),
                "rapid".to_string(),
                "speedy".to_string(),
            ],
        );
        synonyms.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "massive".to_string(),
            ],
        );
        synonyms.insert(
            "small".to_string(),
            vec![
                "tiny".to_string(),
                "little".to_string(),
                "compact".to_string(),
            ],
        );
        synonyms.insert(
            "good".to_string(),
            vec![
                "excellent".to_string(),
                "great".to_string(),
                "quality".to_string(),
            ],
        );
        synonyms.insert(
            "bad".to_string(),
            vec![
                "poor".to_string(),
                "terrible".to_string(),
                "awful".to_string(),
            ],
        );
        synonyms.insert(
            "simple".to_string(),
            vec![
                "easy".to_string(),
                "basic".to_string(),
                "straightforward".to_string(),
            ],
        );
        synonyms.insert(
            "difficult".to_string(),
            vec![
                "hard".to_string(),
                "challenging".to_string(),
                "complex".to_string(),
            ],
        );
        synonyms.insert(
            "method".to_string(),
            vec![
                "approach".to_string(),
                "technique".to_string(),
                "way".to_string(),
            ],
        );
        synonyms.insert(
            "create".to_string(),
            vec![
                "build".to_string(),
                "make".to_string(),
                "develop".to_string(),
            ],
        );
        synonyms.insert(
            "use".to_string(),
            vec![
                "utilize".to_string(),
                "employ".to_string(),
                "apply".to_string(),
            ],
        );

        synonyms
    }

    /// Initialize related terms dictionary
    fn init_related_terms() -> HashMap<String, Vec<String>> {
        let mut related = HashMap::new();

        // Technology related terms
        related.insert(
            "programming".to_string(),
            vec![
                "coding".to_string(),
                "development".to_string(),
                "software".to_string(),
            ],
        );
        related.insert(
            "database".to_string(),
            vec![
                "data".to_string(),
                "storage".to_string(),
                "query".to_string(),
            ],
        );
        related.insert(
            "algorithm".to_string(),
            vec![
                "logic".to_string(),
                "computation".to_string(),
                "optimization".to_string(),
            ],
        );
        related.insert(
            "machine".to_string(),
            vec![
                "learning".to_string(),
                "ai".to_string(),
                "model".to_string(),
            ],
        );
        related.insert(
            "web".to_string(),
            vec![
                "website".to_string(),
                "internet".to_string(),
                "browser".to_string(),
            ],
        );
        related.insert(
            "api".to_string(),
            vec![
                "interface".to_string(),
                "endpoint".to_string(),
                "service".to_string(),
            ],
        );
        related.insert(
            "security".to_string(),
            vec![
                "encryption".to_string(),
                "authentication".to_string(),
                "protection".to_string(),
            ],
        );
        related.insert(
            "performance".to_string(),
            vec![
                "speed".to_string(),
                "optimization".to_string(),
                "efficiency".to_string(),
            ],
        );

        related
    }

    /// Initialize domain-specific expansions
    fn init_domain_expansions() -> HashMap<String, HashMap<String, Vec<String>>> {
        let mut domains = HashMap::new();

        // Technology domain
        let mut tech_expansions = HashMap::new();
        tech_expansions.insert(
            "ml".to_string(),
            vec![
                "machine learning".to_string(),
                "artificial intelligence".to_string(),
            ],
        );
        tech_expansions.insert(
            "ai".to_string(),
            vec![
                "artificial intelligence".to_string(),
                "machine learning".to_string(),
                "neural networks".to_string(),
            ],
        );
        tech_expansions.insert(
            "nlp".to_string(),
            vec![
                "natural language processing".to_string(),
                "text analysis".to_string(),
            ],
        );
        tech_expansions.insert(
            "api".to_string(),
            vec![
                "rest".to_string(),
                "endpoint".to_string(),
                "microservice".to_string(),
            ],
        );
        tech_expansions.insert(
            "db".to_string(),
            vec![
                "database".to_string(),
                "sql".to_string(),
                "storage".to_string(),
            ],
        );

        domains.insert("technology".to_string(), tech_expansions);

        // Business domain
        let mut business_expansions = HashMap::new();
        business_expansions.insert(
            "roi".to_string(),
            vec![
                "return on investment".to_string(),
                "profitability".to_string(),
            ],
        );
        business_expansions.insert(
            "kpi".to_string(),
            vec![
                "key performance indicator".to_string(),
                "metrics".to_string(),
            ],
        );
        business_expansions.insert(
            "b2b".to_string(),
            vec!["business to business".to_string(), "enterprise".to_string()],
        );
        business_expansions.insert(
            "b2c".to_string(),
            vec!["business to consumer".to_string(), "retail".to_string()],
        );

        domains.insert("business".to_string(), business_expansions);

        domains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synonym_expansion() {
        let expander = QueryExpander::new(ExpansionConfig::default());

        let results = expander.expand("fast algorithm").await.unwrap();

        let synonym_result = results
            .iter()
            .find(|r| r.expansion_type == ExpansionStrategy::Synonyms);
        assert!(synonym_result.is_some());

        let result = synonym_result.unwrap();
        assert!(result.expanded_query.contains("quick") || result.expanded_query.contains("rapid"));
    }

    #[tokio::test]
    async fn test_domain_expansion() {
        let expander = QueryExpander::new(ExpansionConfig::default());

        let results = expander.expand("ML model").await.unwrap();

        let domain_result = results
            .iter()
            .find(|r| r.expansion_type == ExpansionStrategy::DomainSpecific);
        assert!(domain_result.is_some());

        let result = domain_result.unwrap();
        assert!(result.expanded_query.contains("machine learning"));
    }
}
