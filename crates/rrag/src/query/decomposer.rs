//! # Query Decomposer
//!
//! Intelligent decomposition of complex queries into simpler sub-queries.
//! Helps improve retrieval by breaking down multi-part questions into focused searches.

use crate::RragResult;
use serde::{Deserialize, Serialize};

/// Query decomposer for breaking down complex queries
pub struct QueryDecomposer {
    /// Configuration
    config: DecompositionConfig,

    /// Patterns for identifying decomposable queries
    patterns: Vec<DecompositionPattern>,

    /// Keywords that indicate complex queries
    complexity_indicators: Vec<String>,
}

/// Configuration for query decomposition
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Maximum number of sub-queries to generate
    pub max_sub_queries: usize,

    /// Minimum length for a sub-query
    pub min_sub_query_length: usize,

    /// Enable temporal decomposition (time-based queries)
    pub enable_temporal_decomposition: bool,

    /// Enable logical decomposition (AND/OR queries)
    pub enable_logical_decomposition: bool,

    /// Enable topical decomposition (multi-topic queries)
    pub enable_topical_decomposition: bool,

    /// Enable comparative decomposition (comparison queries)
    pub enable_comparative_decomposition: bool,

    /// Confidence threshold for accepting decompositions
    pub confidence_threshold: f32,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_sub_queries: 5,
            min_sub_query_length: 5,
            enable_temporal_decomposition: true,
            enable_logical_decomposition: true,
            enable_topical_decomposition: true,
            enable_comparative_decomposition: true,
            confidence_threshold: 0.6,
        }
    }
}

/// Decomposition strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Logical decomposition (AND, OR clauses)
    Logical,
    /// Temporal decomposition (time-based aspects)
    Temporal,
    /// Topical decomposition (different subjects)
    Topical,
    /// Comparative decomposition (A vs B)
    Comparative,
    /// Sequential decomposition (step-by-step)
    Sequential,
    /// Causal decomposition (cause and effect)
    Causal,
}

/// Pattern for identifying decomposable queries
struct DecompositionPattern {
    /// Name of the pattern
    name: String,
    /// Keywords that trigger this pattern
    triggers: Vec<String>,
    /// Decomposition strategy to apply
    strategy: DecompositionStrategy,
    /// Function to extract sub-queries
    extractor: fn(&str) -> Vec<String>,
    /// Confidence score
    confidence: f32,
}

/// Sub-query generated from decomposition
#[derive(Debug, Clone)]
pub struct SubQuery {
    /// The sub-query text
    pub query: String,

    /// Strategy used to generate this sub-query
    pub strategy: DecompositionStrategy,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Priority/importance (higher = more important)
    pub priority: f32,

    /// Metadata about the sub-query
    pub metadata: SubQueryMetadata,
}

/// Metadata for sub-queries
#[derive(Debug, Clone)]
pub struct SubQueryMetadata {
    /// Position in the original query
    pub position: usize,

    /// Relationship to other sub-queries
    pub relationships: Vec<String>,

    /// Expected answer type
    pub expected_answer_type: String,

    /// Dependencies on other sub-queries
    pub dependencies: Vec<usize>,
}

impl QueryDecomposer {
    /// Create a new query decomposer
    pub fn new() -> Self {
        Self::with_config(DecompositionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DecompositionConfig) -> Self {
        let patterns = Self::init_patterns();
        let complexity_indicators = Self::init_complexity_indicators();

        Self {
            config,
            patterns,
            complexity_indicators,
        }
    }

    /// Decompose a complex query into sub-queries
    pub async fn decompose(&self, query: &str) -> RragResult<Vec<SubQuery>> {
        let mut sub_queries = Vec::new();

        // Check if query needs decomposition
        if !self.should_decompose(query) {
            return Ok(sub_queries);
        }

        // Apply different decomposition strategies
        if self.config.enable_logical_decomposition {
            sub_queries.extend(self.logical_decomposition(query));
        }

        if self.config.enable_temporal_decomposition {
            sub_queries.extend(self.temporal_decomposition(query));
        }

        if self.config.enable_topical_decomposition {
            sub_queries.extend(self.topical_decomposition(query));
        }

        if self.config.enable_comparative_decomposition {
            sub_queries.extend(self.comparative_decomposition(query));
        }

        // Filter by confidence and limit results
        sub_queries.retain(|sq| sq.confidence >= self.config.confidence_threshold);
        sub_queries.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sub_queries.truncate(self.config.max_sub_queries);

        // Add metadata and dependencies
        self.enrich_sub_queries(&mut sub_queries);

        Ok(sub_queries)
    }

    /// Check if a query should be decomposed
    fn should_decompose(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();

        // Check for complexity indicators
        let has_complexity_indicators = self
            .complexity_indicators
            .iter()
            .any(|indicator| query_lower.contains(indicator));

        // Check for multiple questions
        let question_count = query.matches('?').count();

        // Check length threshold
        let word_count = query.split_whitespace().count();

        has_complexity_indicators || question_count > 1 || word_count > 15
    }

    /// Logical decomposition (AND, OR, BUT clauses)
    fn logical_decomposition(&self, query: &str) -> Vec<SubQuery> {
        let mut sub_queries = Vec::new();

        // Split on logical connectors
        let logical_connectors = ["and", "or", "but", "however", "also", "additionally"];

        for connector in &logical_connectors {
            if query.to_lowercase().contains(connector) {
                let parts: Vec<&str> = query.split(&format!(" {} ", connector)).collect();
                if parts.len() > 1 {
                    for (i, part) in parts.iter().enumerate() {
                        let trimmed = part.trim();
                        if trimmed.len() >= self.config.min_sub_query_length {
                            sub_queries.push(SubQuery {
                                query: self.complete_sub_query(trimmed),
                                strategy: DecompositionStrategy::Logical,
                                confidence: 0.8,
                                priority: 1.0 - (i as f32 * 0.1), // First parts are more important
                                metadata: SubQueryMetadata {
                                    position: i,
                                    relationships: vec![connector.to_string()],
                                    expected_answer_type: "factual".to_string(),
                                    dependencies: vec![],
                                },
                            });
                        }
                    }
                    break; // Only use the first matching connector
                }
            }
        }

        sub_queries
    }

    /// Temporal decomposition for time-based queries
    fn temporal_decomposition(&self, query: &str) -> Vec<SubQuery> {
        let mut sub_queries = Vec::new();
        let query_lower = query.to_lowercase();

        // Time indicators
        let time_indicators = [
            "when", "before", "after", "during", "since", "until", "timeline",
        ];

        if time_indicators
            .iter()
            .any(|&indicator| query_lower.contains(indicator))
        {
            // Extract temporal aspects
            let temporal_aspects = self.extract_temporal_aspects(query);

            for (i, aspect) in temporal_aspects.iter().enumerate() {
                sub_queries.push(SubQuery {
                    query: aspect.clone(),
                    strategy: DecompositionStrategy::Temporal,
                    confidence: 0.7,
                    priority: 0.8,
                    metadata: SubQueryMetadata {
                        position: i,
                        relationships: vec!["temporal".to_string()],
                        expected_answer_type: "temporal".to_string(),
                        dependencies: vec![],
                    },
                });
            }
        }

        sub_queries
    }

    /// Topical decomposition for multi-topic queries
    fn topical_decomposition(&self, query: &str) -> Vec<SubQuery> {
        let mut sub_queries = Vec::new();

        // Look for multiple topics or subjects
        let topics = self.extract_topics(query);

        if topics.len() > 1 {
            for (i, topic) in topics.iter().enumerate() {
                let topic_query = format!("What is {}?", topic);
                sub_queries.push(SubQuery {
                    query: topic_query,
                    strategy: DecompositionStrategy::Topical,
                    confidence: 0.6,
                    priority: 0.7,
                    metadata: SubQueryMetadata {
                        position: i,
                        relationships: vec!["topical".to_string()],
                        expected_answer_type: "conceptual".to_string(),
                        dependencies: vec![],
                    },
                });
            }
        }

        sub_queries
    }

    /// Comparative decomposition for comparison queries
    fn comparative_decomposition(&self, query: &str) -> Vec<SubQuery> {
        let mut sub_queries = Vec::new();
        let query_lower = query.to_lowercase();

        // Comparison indicators
        let comparison_indicators = [
            "vs",
            "versus",
            "compare",
            "difference",
            "similar",
            "different",
        ];

        if comparison_indicators
            .iter()
            .any(|&indicator| query_lower.contains(indicator))
        {
            let items = self.extract_comparison_items(query);

            if items.len() >= 2 {
                for item in &items {
                    sub_queries.push(SubQuery {
                        query: format!("What are the features of {}?", item),
                        strategy: DecompositionStrategy::Comparative,
                        confidence: 0.75,
                        priority: 0.8,
                        metadata: SubQueryMetadata {
                            position: 0,
                            relationships: vec!["comparative".to_string()],
                            expected_answer_type: "comparative".to_string(),
                            dependencies: vec![],
                        },
                    });
                }

                // Add a synthesis query
                sub_queries.push(SubQuery {
                    query: format!("Compare {} and {}", items[0], items[1]),
                    strategy: DecompositionStrategy::Comparative,
                    confidence: 0.9,
                    priority: 1.0,
                    metadata: SubQueryMetadata {
                        position: items.len(),
                        relationships: vec!["synthesis".to_string()],
                        expected_answer_type: "comparative".to_string(),
                        dependencies: (0..items.len()).collect(),
                    },
                });
            }
        }

        sub_queries
    }

    /// Complete a sub-query to make it grammatically correct
    fn complete_sub_query(&self, partial: &str) -> String {
        let trimmed = partial.trim();

        // If it doesn't start with a question word or have proper structure, add context
        let question_words = ["what", "how", "why", "when", "where", "who", "which"];
        let starts_with_question = question_words
            .iter()
            .any(|&word| trimmed.to_lowercase().starts_with(word));

        if starts_with_question || trimmed.ends_with('?') {
            trimmed.to_string()
        } else {
            format!("What is {}?", trimmed)
        }
    }

    /// Extract temporal aspects from a query
    fn extract_temporal_aspects(&self, query: &str) -> Vec<String> {
        let mut aspects = Vec::new();

        // Simple temporal extraction - in production, this would be more sophisticated
        if query.to_lowercase().contains("when") {
            aspects.push(format!(
                "When did {} happen?",
                self.extract_main_subject(query)
            ));
        }

        if query.to_lowercase().contains("before") {
            aspects.push(format!(
                "What happened before {}?",
                self.extract_main_subject(query)
            ));
        }

        if query.to_lowercase().contains("after") {
            aspects.push(format!(
                "What happened after {}?",
                self.extract_main_subject(query)
            ));
        }

        aspects
    }

    /// Extract topics from a query
    fn extract_topics(&self, query: &str) -> Vec<String> {
        let mut topics = Vec::new();

        // Simple topic extraction based on nouns and capitalized words
        let words: Vec<&str> = query.split_whitespace().collect();

        for window in words.windows(2) {
            let word = window[0];
            // Look for capitalized words (potential proper nouns/topics)
            if word.chars().next().map_or(false, |c| c.is_uppercase()) && word.len() > 2 {
                topics.push(word.to_string());
            }
        }

        // Remove duplicates
        topics.sort();
        topics.dedup();

        topics
    }

    /// Extract comparison items from a query
    fn extract_comparison_items(&self, query: &str) -> Vec<String> {
        let mut items = Vec::new();

        // Look for patterns like "A vs B" or "A and B"
        if let Some(vs_pos) = query.to_lowercase().find(" vs ") {
            let before = &query[..vs_pos].trim();
            let after = &query[vs_pos + 4..].trim();

            items.push(self.extract_last_noun(before).to_string());
            items.push(self.extract_first_noun(after).to_string());
        } else if query.to_lowercase().contains("compare") {
            // Extract nouns after "compare"
            let words: Vec<&str> = query.split_whitespace().collect();
            let mut collecting = false;

            for word in words {
                if word.to_lowercase() == "compare" {
                    collecting = true;
                    continue;
                }

                if collecting
                    && word.len() > 2
                    && !["and", "with", "to"].contains(&word.to_lowercase().as_str())
                {
                    items.push(
                        word.trim_matches(|c: char| !c.is_alphanumeric())
                            .to_string(),
                    );
                    if items.len() >= 2 {
                        break;
                    }
                }
            }
        }

        items
    }

    /// Extract the main subject from a query
    fn extract_main_subject(&self, query: &str) -> String {
        // Simple subject extraction - would be more sophisticated in production
        let words: Vec<&str> = query.split_whitespace().collect();

        // Look for the first meaningful noun
        for word in words {
            if word.len() > 3
                && !["what", "when", "where", "how", "why", "who", "the", "and"]
                    .contains(&word.to_lowercase().as_str())
            {
                return word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();
            }
        }

        "this".to_string()
    }

    /// Extract the last meaningful noun from text
    fn extract_last_noun<'a>(&self, text: &'a str) -> &'a str {
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words.iter().rev() {
            if word.len() > 2
                && !["the", "and", "or", "of", "in", "on", "at"]
                    .contains(&word.to_lowercase().as_str())
            {
                return word;
            }
        }
        text
    }

    /// Extract the first meaningful noun from text
    fn extract_first_noun<'a>(&self, text: &'a str) -> &'a str {
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if word.len() > 2
                && !["the", "and", "or", "of", "in", "on", "at"]
                    .contains(&word.to_lowercase().as_str())
            {
                return word;
            }
        }
        text
    }

    /// Enrich sub-queries with additional metadata
    fn enrich_sub_queries(&self, sub_queries: &mut [SubQuery]) {
        for (i, sub_query) in sub_queries.iter_mut().enumerate() {
            // Add position metadata
            sub_query.metadata.position = i;

            // Determine expected answer type based on query structure
            sub_query.metadata.expected_answer_type = self.determine_answer_type(&sub_query.query);
        }
    }

    /// Determine the expected answer type for a query
    fn determine_answer_type(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();

        if query_lower.starts_with("what is") || query_lower.starts_with("define") {
            "definitional".to_string()
        } else if query_lower.starts_with("how") {
            "procedural".to_string()
        } else if query_lower.starts_with("when") {
            "temporal".to_string()
        } else if query_lower.starts_with("where") {
            "locational".to_string()
        } else if query_lower.starts_with("why") {
            "causal".to_string()
        } else if query_lower.contains("compare") || query_lower.contains("vs") {
            "comparative".to_string()
        } else {
            "factual".to_string()
        }
    }

    /// Initialize decomposition patterns
    fn init_patterns() -> Vec<DecompositionPattern> {
        vec![
            DecompositionPattern {
                name: "Logical AND".to_string(),
                triggers: vec![
                    "and".to_string(),
                    "also".to_string(),
                    "additionally".to_string(),
                ],
                strategy: DecompositionStrategy::Logical,
                extractor: |query| {
                    query
                        .split(" and ")
                        .map(|s| s.trim().to_string())
                        .filter(|s| s.len() > 5)
                        .collect()
                },
                confidence: 0.8,
            },
            DecompositionPattern {
                name: "Comparative".to_string(),
                triggers: vec![
                    "vs".to_string(),
                    "compare".to_string(),
                    "difference".to_string(),
                ],
                strategy: DecompositionStrategy::Comparative,
                extractor: |query| {
                    if query.contains(" vs ") {
                        query
                            .split(" vs ")
                            .map(|s| format!("What is {}?", s.trim()))
                            .collect()
                    } else {
                        vec![]
                    }
                },
                confidence: 0.9,
            },
        ]
    }

    /// Initialize complexity indicators
    fn init_complexity_indicators() -> Vec<String> {
        vec![
            "and".to_string(),
            "or".to_string(),
            "but".to_string(),
            "however".to_string(),
            "also".to_string(),
            "additionally".to_string(),
            "furthermore".to_string(),
            "moreover".to_string(),
            "vs".to_string(),
            "versus".to_string(),
            "compare".to_string(),
            "difference".to_string(),
            "similar".to_string(),
            "different".to_string(),
            "before".to_string(),
            "after".to_string(),
            "during".to_string(),
            "while".to_string(),
            "meanwhile".to_string(),
        ]
    }
}

impl Default for QueryDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_logical_decomposition() {
        let decomposer = QueryDecomposer::new();

        let query = "What is machine learning and how does deep learning work?";
        let sub_queries = decomposer.decompose(query).await.unwrap();

        assert!(!sub_queries.is_empty());
        assert!(sub_queries
            .iter()
            .any(|sq| sq.strategy == DecompositionStrategy::Logical));
    }

    #[tokio::test]
    async fn test_comparative_decomposition() {
        let decomposer = QueryDecomposer::new();

        let query = "What are the differences between Python vs Rust for system programming?";
        let sub_queries = decomposer.decompose(query).await.unwrap();

        assert!(!sub_queries.is_empty());
        let comparative_queries: Vec<_> = sub_queries
            .iter()
            .filter(|sq| sq.strategy == DecompositionStrategy::Comparative)
            .collect();
        assert!(!comparative_queries.is_empty());
    }

    #[tokio::test]
    async fn test_should_not_decompose_simple_query() {
        let decomposer = QueryDecomposer::new();

        let query = "What is Rust?";
        let sub_queries = decomposer.decompose(query).await.unwrap();

        // Simple queries should not be decomposed
        assert!(sub_queries.is_empty());
    }

    #[tokio::test]
    async fn test_temporal_decomposition() {
        let decomposer = QueryDecomposer::new();

        let query = "When did the Renaissance start and what happened before it?";
        let sub_queries = decomposer.decompose(query).await.unwrap();

        assert!(!sub_queries.is_empty());
        let temporal_queries: Vec<_> = sub_queries
            .iter()
            .filter(|sq| sq.strategy == DecompositionStrategy::Temporal)
            .collect();
        assert!(!temporal_queries.is_empty());
    }
}
