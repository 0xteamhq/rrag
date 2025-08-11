//! # Query Classifier
//!
//! Intelligent classification of user queries to determine intent and appropriate search strategies.
//! Helps optimize retrieval by understanding what the user is looking for.

use crate::RragResult;
use serde::{Deserialize, Serialize};

/// Query classifier for intent detection
pub struct QueryClassifier {
    /// Intent patterns for classification
    patterns: Vec<IntentPattern>,

    /// Type patterns for classification
    type_patterns: Vec<TypePattern>,
}

/// Query intent categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryIntent {
    /// Factual information seeking
    Factual,
    /// Conceptual understanding
    Conceptual,
    /// Procedural how-to questions
    Procedural,
    /// Comparative analysis
    Comparative,
    /// Troubleshooting and problem-solving
    Troubleshooting,
    /// Exploratory research
    Exploratory,
    /// Definitional queries
    Definitional,
    /// Opinion or recommendation seeking
    OpinionSeeking,
}

/// Query type categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryType {
    /// Direct question
    Question,
    /// Command or request
    Command,
    /// Keyword search
    Keywords,
    /// Natural language statement
    Statement,
    /// Complex multi-part query
    Complex,
}

/// Result of query classification
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Original query
    pub query: String,

    /// Detected intent
    pub intent: QueryIntent,

    /// Detected query type
    pub query_type: QueryType,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Additional metadata about the query
    pub metadata: ClassificationMetadata,
}

/// Additional metadata from classification
#[derive(Debug, Clone)]
pub struct ClassificationMetadata {
    /// Key entities detected in the query
    pub entities: Vec<String>,

    /// Domain/topic detected
    pub domain: Option<String>,

    /// Complexity score
    pub complexity: f32,

    /// Whether query requires context
    pub needs_context: bool,

    /// Suggested search strategies
    pub suggested_strategies: Vec<String>,
}

/// Pattern for intent detection
struct IntentPattern {
    /// Intent this pattern detects
    intent: QueryIntent,
    /// Keywords that indicate this intent
    keywords: Vec<String>,
    /// Phrase patterns
    patterns: Vec<String>,
    /// Confidence score
    confidence: f32,
}

/// Pattern for type detection
struct TypePattern {
    /// Query type this pattern detects
    query_type: QueryType,
    /// Indicators for this type
    indicators: Vec<String>,
    /// Confidence score
    confidence: f32,
}

impl QueryClassifier {
    /// Create a new query classifier
    pub fn new() -> Self {
        let patterns = Self::init_intent_patterns();
        let type_patterns = Self::init_type_patterns();

        Self {
            patterns,
            type_patterns,
        }
    }

    /// Classify a query to determine intent and type
    pub async fn classify(&self, query: &str) -> RragResult<ClassificationResult> {
        let query_lower = query.to_lowercase();
        let tokens = self.tokenize(&query_lower);

        // Detect intent
        let (intent, intent_confidence) = self.detect_intent(&query_lower, &tokens);

        // Detect query type
        let (query_type, type_confidence) = self.detect_query_type(&query_lower, &tokens);

        // Extract entities
        let entities = self.extract_entities(&tokens);

        // Detect domain
        let domain = self.detect_domain(&tokens);

        // Calculate complexity
        let complexity = self.calculate_complexity(query, &tokens);

        // Determine if context is needed
        let needs_context = self.needs_context(query, &tokens);

        // Suggest search strategies
        let suggested_strategies = self.suggest_strategies(&intent, &query_type, complexity);

        // Overall confidence is the minimum of intent and type confidence
        let confidence = intent_confidence.min(type_confidence);

        Ok(ClassificationResult {
            query: query.to_string(),
            intent,
            query_type,
            confidence,
            metadata: ClassificationMetadata {
                entities,
                domain,
                complexity,
                needs_context,
                suggested_strategies,
            },
        })
    }

    /// Detect query intent
    fn detect_intent(&self, query: &str, tokens: &[String]) -> (QueryIntent, f32) {
        let mut best_intent = QueryIntent::Factual;
        let mut best_confidence = 0.0;

        for pattern in &self.patterns {
            let mut score = 0.0;
            let mut matches = 0;

            // Check keyword matches
            for keyword in &pattern.keywords {
                if tokens.iter().any(|t| t.contains(keyword)) {
                    score += 1.0;
                    matches += 1;
                }
            }

            // Check phrase patterns
            for phrase in &pattern.patterns {
                if query.contains(phrase) {
                    score += 2.0; // Phrase matches are stronger
                    matches += 1;
                }
            }

            if matches > 0 {
                // Normalize score
                let normalized_score = (score
                    / (pattern.keywords.len() + pattern.patterns.len()) as f32)
                    * pattern.confidence;

                if normalized_score > best_confidence {
                    best_intent = pattern.intent.clone();
                    best_confidence = normalized_score;
                }
            }
        }

        // Default fallback based on simple heuristics
        if best_confidence < 0.3 {
            if query.starts_with("what is") || query.starts_with("define") {
                best_intent = QueryIntent::Definitional;
                best_confidence = 0.6;
            } else if query.starts_with("how to") || query.contains("step") {
                best_intent = QueryIntent::Procedural;
                best_confidence = 0.6;
            } else if query.contains("compare")
                || query.contains("vs")
                || query.contains("difference")
            {
                best_intent = QueryIntent::Comparative;
                best_confidence = 0.6;
            }
        }

        (best_intent, best_confidence)
    }

    /// Detect query type
    fn detect_query_type(&self, query: &str, tokens: &[String]) -> (QueryType, f32) {
        let mut best_type = QueryType::Keywords;
        let mut best_confidence = 0.0;

        for pattern in &self.type_patterns {
            let mut matches = 0;

            for indicator in &pattern.indicators {
                if query.contains(indicator) || tokens.iter().any(|t| t == indicator) {
                    matches += 1;
                }
            }

            if matches > 0 {
                let confidence =
                    (matches as f32 / pattern.indicators.len() as f32) * pattern.confidence;
                if confidence > best_confidence {
                    best_type = pattern.query_type.clone();
                    best_confidence = confidence;
                }
            }
        }

        // Fallback heuristics
        if best_confidence < 0.5 {
            if query.ends_with('?') {
                best_type = QueryType::Question;
                best_confidence = 0.8;
            } else if tokens.len() <= 3 {
                best_type = QueryType::Keywords;
                best_confidence = 0.7;
            } else if tokens.len() > 10 {
                best_type = QueryType::Complex;
                best_confidence = 0.6;
            } else {
                best_type = QueryType::Statement;
                best_confidence = 0.5;
            }
        }

        (best_type, best_confidence)
    }

    /// Extract entities from query tokens
    fn extract_entities(&self, tokens: &[String]) -> Vec<String> {
        let mut entities = Vec::new();

        // Simple entity extraction based on capitalization and known patterns
        for token in tokens {
            // Check if it looks like a proper noun (capitalized)
            if token.chars().next().map_or(false, |c| c.is_uppercase()) {
                entities.push(token.clone());
            }

            // Check for technical terms
            let tech_terms = [
                "api",
                "sql",
                "json",
                "html",
                "css",
                "javascript",
                "python",
                "rust",
                "docker",
            ];
            if tech_terms.contains(&token.to_lowercase().as_str()) {
                entities.push(token.clone());
            }
        }

        entities
    }

    /// Detect query domain
    fn detect_domain(&self, tokens: &[String]) -> Option<String> {
        let domains = [
            (
                "technology",
                vec![
                    "code",
                    "programming",
                    "software",
                    "api",
                    "database",
                    "algorithm",
                    "computer",
                ],
            ),
            (
                "science",
                vec![
                    "research",
                    "study",
                    "experiment",
                    "theory",
                    "analysis",
                    "data",
                    "scientific",
                ],
            ),
            (
                "business",
                vec![
                    "market",
                    "sales",
                    "revenue",
                    "customer",
                    "profit",
                    "strategy",
                    "management",
                ],
            ),
            (
                "health",
                vec![
                    "medical",
                    "health",
                    "disease",
                    "treatment",
                    "doctor",
                    "medicine",
                    "patient",
                ],
            ),
            (
                "education",
                vec![
                    "learn",
                    "study",
                    "school",
                    "university",
                    "course",
                    "education",
                    "teach",
                ],
            ),
        ];

        for (domain, keywords) in &domains {
            let matches = keywords
                .iter()
                .filter(|&&keyword| tokens.iter().any(|t| t.contains(keyword)))
                .count();

            if matches >= 2 || (matches == 1 && tokens.len() <= 5) {
                return Some(domain.to_string());
            }
        }

        None
    }

    /// Calculate query complexity
    fn calculate_complexity(&self, query: &str, tokens: &[String]) -> f32 {
        let mut complexity = 0.0;

        // Length factor
        complexity += (tokens.len() as f32 / 20.0).min(1.0) * 0.3;

        // Question words
        let question_words = ["what", "how", "why", "when", "where", "which", "who"];
        let question_count = question_words
            .iter()
            .filter(|&&word| tokens.iter().any(|t| t == word))
            .count();
        complexity += (question_count as f32 * 0.1).min(0.3);

        // Conjunctions indicating complexity
        let conjunctions = ["and", "or", "but", "however", "also", "additionally"];
        let conjunction_count = conjunctions
            .iter()
            .filter(|&&word| tokens.iter().any(|t| t == word))
            .count();
        complexity += (conjunction_count as f32 * 0.15).min(0.2);

        // Nested questions
        if query.matches('?').count() > 1 {
            complexity += 0.2;
        }

        complexity.min(1.0)
    }

    /// Determine if query needs conversational context
    fn needs_context(&self, _query: &str, tokens: &[String]) -> bool {
        let context_indicators = [
            "this",
            "that",
            "it",
            "they",
            "them",
            "previous",
            "above",
            "following",
        ];
        let pronouns = ["it", "this", "that", "these", "those"];

        // Check for pronouns without clear antecedents
        let has_pronouns = pronouns
            .iter()
            .any(|&pronoun| tokens.contains(&pronoun.to_string()));

        // Check for context indicators
        let has_context_indicators = context_indicators
            .iter()
            .any(|&indicator| tokens.contains(&indicator.to_string()));

        // Very short queries often need context
        let is_very_short = tokens.len() <= 2;

        has_pronouns || has_context_indicators || is_very_short
    }

    /// Suggest search strategies based on classification
    fn suggest_strategies(
        &self,
        intent: &QueryIntent,
        query_type: &QueryType,
        complexity: f32,
    ) -> Vec<String> {
        let mut strategies = Vec::new();

        match intent {
            QueryIntent::Factual => {
                strategies.push("keyword_search".to_string());
                strategies.push("exact_match".to_string());
            }
            QueryIntent::Conceptual => {
                strategies.push("semantic_search".to_string());
                strategies.push("related_documents".to_string());
            }
            QueryIntent::Procedural => {
                strategies.push("step_by_step".to_string());
                strategies.push("tutorial_search".to_string());
            }
            QueryIntent::Comparative => {
                strategies.push("comparative_analysis".to_string());
                strategies.push("side_by_side".to_string());
            }
            QueryIntent::Troubleshooting => {
                strategies.push("problem_solution".to_string());
                strategies.push("diagnostic".to_string());
            }
            QueryIntent::Exploratory => {
                strategies.push("broad_search".to_string());
                strategies.push("topic_exploration".to_string());
            }
            QueryIntent::Definitional => {
                strategies.push("definition_search".to_string());
                strategies.push("glossary_lookup".to_string());
            }
            QueryIntent::OpinionSeeking => {
                strategies.push("review_search".to_string());
                strategies.push("opinion_mining".to_string());
            }
        }

        match query_type {
            QueryType::Complex => {
                strategies.push("query_decomposition".to_string());
                strategies.push("multi_step_search".to_string());
            }
            QueryType::Keywords => {
                strategies.push("keyword_expansion".to_string());
                strategies.push("term_matching".to_string());
            }
            _ => {}
        }

        if complexity > 0.7 {
            strategies.push("complex_reasoning".to_string());
            strategies.push("multi_document_synthesis".to_string());
        }

        strategies
    }

    /// Tokenize query
    fn tokenize(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Initialize intent detection patterns
    fn init_intent_patterns() -> Vec<IntentPattern> {
        vec![
            IntentPattern {
                intent: QueryIntent::Definitional,
                keywords: vec![
                    "define".to_string(),
                    "definition".to_string(),
                    "meaning".to_string(),
                ],
                patterns: vec![
                    "what is".to_string(),
                    "what does".to_string(),
                    "define".to_string(),
                ],
                confidence: 0.9,
            },
            IntentPattern {
                intent: QueryIntent::Procedural,
                keywords: vec![
                    "how".to_string(),
                    "step".to_string(),
                    "tutorial".to_string(),
                    "guide".to_string(),
                ],
                patterns: vec![
                    "how to".to_string(),
                    "step by step".to_string(),
                    "how do i".to_string(),
                ],
                confidence: 0.9,
            },
            IntentPattern {
                intent: QueryIntent::Comparative,
                keywords: vec![
                    "compare".to_string(),
                    "difference".to_string(),
                    "better".to_string(),
                    "versus".to_string(),
                ],
                patterns: vec![
                    "vs".to_string(),
                    "compared to".to_string(),
                    "difference between".to_string(),
                ],
                confidence: 0.8,
            },
            IntentPattern {
                intent: QueryIntent::Troubleshooting,
                keywords: vec![
                    "problem".to_string(),
                    "error".to_string(),
                    "fix".to_string(),
                    "issue".to_string(),
                    "broken".to_string(),
                ],
                patterns: vec![
                    "not working".to_string(),
                    "how to fix".to_string(),
                    "troubleshoot".to_string(),
                ],
                confidence: 0.8,
            },
            IntentPattern {
                intent: QueryIntent::Factual,
                keywords: vec![
                    "when".to_string(),
                    "where".to_string(),
                    "who".to_string(),
                    "which".to_string(),
                ],
                patterns: vec![
                    "when did".to_string(),
                    "where is".to_string(),
                    "who created".to_string(),
                ],
                confidence: 0.7,
            },
        ]
    }

    /// Initialize query type patterns
    fn init_type_patterns() -> Vec<TypePattern> {
        vec![
            TypePattern {
                query_type: QueryType::Question,
                indicators: vec![
                    "?".to_string(),
                    "what".to_string(),
                    "how".to_string(),
                    "why".to_string(),
                    "when".to_string(),
                    "where".to_string(),
                ],
                confidence: 0.9,
            },
            TypePattern {
                query_type: QueryType::Command,
                indicators: vec![
                    "show".to_string(),
                    "find".to_string(),
                    "get".to_string(),
                    "list".to_string(),
                    "give".to_string(),
                ],
                confidence: 0.8,
            },
            TypePattern {
                query_type: QueryType::Complex,
                indicators: vec![
                    "and".to_string(),
                    "or".to_string(),
                    "but".to_string(),
                    "however".to_string(),
                    "also".to_string(),
                ],
                confidence: 0.7,
            },
        ]
    }
}

impl Default for QueryClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_definitional_query() {
        let classifier = QueryClassifier::new();

        let result = classifier
            .classify("What is machine learning?")
            .await
            .unwrap();
        assert_eq!(result.intent, QueryIntent::Definitional);
        assert_eq!(result.query_type, QueryType::Question);
        assert!(result.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_procedural_query() {
        let classifier = QueryClassifier::new();

        let result = classifier
            .classify("How to implement a REST API?")
            .await
            .unwrap();
        assert_eq!(result.intent, QueryIntent::Procedural);
        assert!(result.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_comparative_query() {
        let classifier = QueryClassifier::new();

        let result = classifier
            .classify("Python vs Rust performance comparison")
            .await
            .unwrap();
        assert_eq!(result.intent, QueryIntent::Comparative);
        assert!(result.confidence > 0.5);
    }
}
