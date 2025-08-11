//! # Query Rewriter
//!
//! Intelligent query rewriting for improved search quality.
//! Implements multiple rewriting strategies including grammar correction,
//! clarification, and style normalization.

use crate::RragResult;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query rewriter for transforming user queries
pub struct QueryRewriter {
    /// Configuration
    config: QueryRewriteConfig,

    /// Grammar patterns for correction
    grammar_patterns: Vec<GrammarPattern>,

    /// Query templates for different domains
    templates: HashMap<String, Vec<String>>,

    /// Common query transformations
    transformations: Vec<QueryTransformation>,
}

/// Configuration for query rewriting
#[derive(Debug, Clone)]
pub struct QueryRewriteConfig {
    /// Enable grammar correction
    pub enable_grammar_correction: bool,

    /// Enable query clarification
    pub enable_clarification: bool,

    /// Enable style normalization
    pub enable_style_normalization: bool,

    /// Enable domain-specific rewriting
    pub enable_domain_rewriting: bool,

    /// Maximum number of rewrites per query
    pub max_rewrites: usize,

    /// Minimum confidence for accepting rewrites
    pub min_confidence: f32,
}

impl Default for QueryRewriteConfig {
    fn default() -> Self {
        Self {
            enable_grammar_correction: true,
            enable_clarification: true,
            enable_style_normalization: true,
            enable_domain_rewriting: true,
            max_rewrites: 3,
            min_confidence: 0.6,
        }
    }
}

/// Rewriting strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RewriteStrategy {
    /// Grammar and spelling correction
    GrammarCorrection,
    /// Add clarifying information
    Clarification,
    /// Normalize writing style
    StyleNormalization,
    /// Domain-specific transformations
    DomainSpecific,
    /// Template-based rewriting
    TemplateBasedRewriting,
}

/// Grammar pattern for correction
struct GrammarPattern {
    /// Pattern to match
    pattern: Regex,
    /// Replacement template
    replacement: String,
    /// Confidence score
    confidence: f32,
}

/// Query transformation rule
struct QueryTransformation {
    /// Name of the transformation
    name: String,
    /// Function to apply transformation
    transform: fn(&str) -> Option<String>,
    /// Confidence score
    confidence: f32,
    /// Strategy type
    strategy: RewriteStrategy,
}

/// Result of query rewriting
#[derive(Debug, Clone)]
pub struct RewriteResult {
    /// Original query
    pub original_query: String,

    /// Rewritten query
    pub rewritten_query: String,

    /// Strategy used for rewriting
    pub strategy: RewriteStrategy,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Explanation of the rewrite
    pub explanation: String,
}

impl QueryRewriter {
    /// Create a new query rewriter
    pub fn new(config: QueryRewriteConfig) -> Self {
        let grammar_patterns = Self::init_grammar_patterns();
        let templates = Self::init_templates();
        let transformations = Self::init_transformations();

        Self {
            config,
            grammar_patterns,
            templates,
            transformations,
        }
    }

    /// Rewrite a query using all enabled strategies
    pub async fn rewrite(&self, query: &str) -> RragResult<Vec<RewriteResult>> {
        let mut results = Vec::new();

        // Apply grammar correction
        if self.config.enable_grammar_correction {
            if let Some(result) = self.apply_grammar_correction(query) {
                if result.confidence >= self.config.min_confidence {
                    results.push(result);
                }
            }
        }

        // Apply clarification
        if self.config.enable_clarification {
            if let Some(result) = self.apply_clarification(query) {
                if result.confidence >= self.config.min_confidence {
                    results.push(result);
                }
            }
        }

        // Apply style normalization
        if self.config.enable_style_normalization {
            if let Some(result) = self.apply_style_normalization(query) {
                if result.confidence >= self.config.min_confidence {
                    results.push(result);
                }
            }
        }

        // Apply domain-specific rewriting
        if self.config.enable_domain_rewriting {
            let domain_results = self.apply_domain_rewriting(query);
            results.extend(
                domain_results
                    .into_iter()
                    .filter(|r| r.confidence >= self.config.min_confidence),
            );
        }

        // Limit results
        results.truncate(self.config.max_rewrites);

        Ok(results)
    }

    /// Apply grammar correction patterns
    fn apply_grammar_correction(&self, query: &str) -> Option<RewriteResult> {
        for pattern in &self.grammar_patterns {
            if let Some(rewritten) = pattern.apply(query) {
                if rewritten != query {
                    return Some(RewriteResult {
                        original_query: query.to_string(),
                        rewritten_query: rewritten,
                        strategy: RewriteStrategy::GrammarCorrection,
                        confidence: pattern.confidence,
                        explanation: "Applied grammar correction".to_string(),
                    });
                }
            }
        }
        None
    }

    /// Apply query clarification
    fn apply_clarification(&self, query: &str) -> Option<RewriteResult> {
        // Check if query is too vague or ambiguous
        if self.is_vague_query(query) {
            let clarified = self.clarify_query(query);
            if let Some(clarified_query) = clarified {
                return Some(RewriteResult {
                    original_query: query.to_string(),
                    rewritten_query: clarified_query,
                    strategy: RewriteStrategy::Clarification,
                    confidence: 0.7,
                    explanation: "Added clarifying information to vague query".to_string(),
                });
            }
        }
        None
    }

    /// Apply style normalization
    fn apply_style_normalization(&self, query: &str) -> Option<RewriteResult> {
        let normalized = self.normalize_style(query);
        if normalized != query {
            Some(RewriteResult {
                original_query: query.to_string(),
                rewritten_query: normalized,
                strategy: RewriteStrategy::StyleNormalization,
                confidence: 0.8,
                explanation: "Normalized query style".to_string(),
            })
        } else {
            None
        }
    }

    /// Apply domain-specific rewriting
    fn apply_domain_rewriting(&self, query: &str) -> Vec<RewriteResult> {
        let mut results = Vec::new();

        // Apply transformations
        for transformation in &self.transformations {
            if let Some(transformed) = (transformation.transform)(query) {
                if transformed != query {
                    results.push(RewriteResult {
                        original_query: query.to_string(),
                        rewritten_query: transformed,
                        strategy: transformation.strategy.clone(),
                        confidence: transformation.confidence,
                        explanation: format!("Applied {}", transformation.name),
                    });
                }
            }
        }

        results
    }

    /// Check if a query is too vague
    fn is_vague_query(&self, query: &str) -> bool {
        let vague_patterns = [
            r"^(what|how|why|when|where)\s+is\s+\w+\?*$",
            r"^(tell me about|about|info on)\s+\w+\?*$",
            r"^\w{1,3}\?*$", // Very short queries
        ];

        let query_lower = query.to_lowercase();
        for pattern in &vague_patterns {
            if Regex::new(pattern).unwrap().is_match(&query_lower) {
                return true;
            }
        }

        false
    }

    /// Clarify a vague query
    fn clarify_query(&self, query: &str) -> Option<String> {
        let query_lower = query.to_lowercase();

        // Add context based on common patterns
        if query_lower.starts_with("what is") {
            return Some(format!(
                "{} and how does it work?",
                query.trim_end_matches('?')
            ));
        }

        if query_lower.starts_with("how") {
            return Some(format!("{} step by step", query.trim_end_matches('?')));
        }

        if query_lower.starts_with("tell me about") {
            return Some(query_lower.replace("tell me about", "explain the concept of"));
        }

        None
    }

    /// Normalize query style
    fn normalize_style(&self, query: &str) -> String {
        let mut normalized = query.to_string();

        // Remove excessive punctuation
        normalized = Regex::new(r"[!]{2,}")
            .unwrap()
            .replace_all(&normalized, "!")
            .to_string();
        normalized = Regex::new(r"[?]{2,}")
            .unwrap()
            .replace_all(&normalized, "?")
            .to_string();

        // Fix spacing
        normalized = Regex::new(r"\s+")
            .unwrap()
            .replace_all(&normalized, " ")
            .to_string();

        // Capitalize first letter
        if let Some(first_char) = normalized.chars().next() {
            normalized = first_char.to_uppercase().collect::<String>() + &normalized[1..];
        }

        // Ensure questions end with question mark
        if self.is_question(&normalized) && !normalized.ends_with('?') {
            normalized.push('?');
        }

        normalized.trim().to_string()
    }

    /// Check if query is a question
    fn is_question(&self, query: &str) -> bool {
        let question_words = [
            "what", "how", "why", "when", "where", "who", "which", "can", "is", "are", "do", "does",
        ];
        let query_lower = query.to_lowercase();
        question_words
            .iter()
            .any(|&word| query_lower.starts_with(word))
    }

    /// Initialize grammar patterns
    fn init_grammar_patterns() -> Vec<GrammarPattern> {
        vec![
            GrammarPattern {
                pattern: Regex::new(r"\bteh\b").unwrap(),
                replacement: "the".to_string(),
                confidence: 0.9,
            },
            GrammarPattern {
                pattern: Regex::new(r"\badn\b").unwrap(),
                replacement: "and".to_string(),
                confidence: 0.9,
            },
            GrammarPattern {
                pattern: Regex::new(r"\bwat\b").unwrap(),
                replacement: "what".to_string(),
                confidence: 0.8,
            },
            // Add more patterns as needed
        ]
    }

    /// Initialize query templates
    fn init_templates() -> HashMap<String, Vec<String>> {
        let mut templates = HashMap::new();

        templates.insert(
            "technical".to_string(),
            vec![
                "How does {concept} work?".to_string(),
                "What are the key features of {concept}?".to_string(),
                "Explain {concept} in detail".to_string(),
            ],
        );

        templates.insert(
            "comparison".to_string(),
            vec![
                "Compare {item1} and {item2}".to_string(),
                "What are the differences between {item1} and {item2}?".to_string(),
                "{item1} vs {item2} pros and cons".to_string(),
            ],
        );

        templates
    }

    /// Initialize transformations
    fn init_transformations() -> Vec<QueryTransformation> {
        vec![
            QueryTransformation {
                name: "Convert abbreviations".to_string(),
                transform: |query| {
                    let mut result = query.to_string();
                    let abbreviations = [
                        ("ML", "machine learning"),
                        ("AI", "artificial intelligence"),
                        ("NLP", "natural language processing"),
                        ("API", "application programming interface"),
                        ("UI", "user interface"),
                        ("UX", "user experience"),
                    ];

                    for (abbr, full) in &abbreviations {
                        result = result.replace(abbr, full);
                    }

                    if result != query {
                        Some(result)
                    } else {
                        None
                    }
                },
                confidence: 0.8,
                strategy: RewriteStrategy::DomainSpecific,
            },
            QueryTransformation {
                name: "Add technical context".to_string(),
                transform: |query| {
                    let tech_terms = ["algorithm", "framework", "library", "system"];
                    if tech_terms
                        .iter()
                        .any(|term| query.to_lowercase().contains(term))
                    {
                        Some(format!("{} implementation and usage", query))
                    } else {
                        None
                    }
                },
                confidence: 0.6,
                strategy: RewriteStrategy::DomainSpecific,
            },
        ]
    }
}

impl GrammarPattern {
    /// Apply the pattern to a query
    fn apply(&self, query: &str) -> Option<String> {
        if self.pattern.is_match(query) {
            Some(
                self.pattern
                    .replace_all(query, &self.replacement)
                    .to_string(),
            )
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_rewriter() {
        let rewriter = QueryRewriter::new(QueryRewriteConfig::default());

        let results = rewriter.rewrite("wat is ML?").await.unwrap();
        assert!(!results.is_empty());

        // Should correct "wat" to "what" and expand "ML"
        let grammar_corrected = results
            .iter()
            .find(|r| r.strategy == RewriteStrategy::GrammarCorrection);
        assert!(grammar_corrected.is_some());
    }

    #[tokio::test]
    async fn test_style_normalization() {
        let rewriter = QueryRewriter::new(QueryRewriteConfig::default());

        let results = rewriter.rewrite("how   does  this work???").await.unwrap();
        let normalized = results
            .iter()
            .find(|r| r.strategy == RewriteStrategy::StyleNormalization);

        assert!(normalized.is_some());
        assert_eq!(normalized.unwrap().rewritten_query, "How does this work?");
    }
}
