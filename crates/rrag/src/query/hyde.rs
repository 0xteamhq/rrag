//! # HyDE (Hypothetical Document Embeddings)
//!
//! Generates hypothetical documents that would answer the user's query,
//! then uses their embeddings for more effective semantic search.
//! Based on the paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"

use crate::{EmbeddingProvider, RragResult};
use std::collections::HashMap;
use std::sync::Arc;

/// HyDE generator for creating hypothetical document embeddings
pub struct HyDEGenerator {
    /// Configuration
    config: HyDEConfig,

    /// Embedding provider for generating embeddings
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Document templates for different query types
    templates: HashMap<String, Vec<DocumentTemplate>>,

    /// Answer generation patterns
    answer_patterns: Vec<AnswerPattern>,
}

/// Configuration for HyDE generation
#[derive(Debug, Clone)]
pub struct HyDEConfig {
    /// Number of hypothetical documents to generate
    pub num_hypothetical_docs: usize,

    /// Maximum length for generated documents
    pub max_document_length: usize,

    /// Minimum length for generated documents
    pub min_document_length: usize,

    /// Enable query-specific document generation
    pub enable_query_specific_generation: bool,

    /// Enable domain-aware generation
    pub enable_domain_awareness: bool,

    /// Confidence threshold for accepting generated documents
    pub confidence_threshold: f32,

    /// Temperature for generation (creativity vs accuracy)
    pub generation_temperature: f32,
}

impl Default for HyDEConfig {
    fn default() -> Self {
        Self {
            num_hypothetical_docs: 3,
            max_document_length: 500,
            min_document_length: 50,
            enable_query_specific_generation: true,
            enable_domain_awareness: true,
            confidence_threshold: 0.6,
            generation_temperature: 0.7,
        }
    }
}

/// Document template for generating hypothetical answers
#[derive(Debug, Clone)]
struct DocumentTemplate {
    /// Template name
    name: String,
    /// Template pattern with placeholders
    pattern: String,
    /// Query types this template works best for
    query_types: Vec<String>,
    /// Confidence score for this template
    confidence: f32,
}

/// Pattern for generating answers
#[derive(Debug, Clone)]
struct AnswerPattern {
    /// Pattern name
    name: String,
    /// Trigger keywords
    triggers: Vec<String>,
    /// Generation function
    generator: fn(&str, &HyDEConfig) -> Vec<String>,
    /// Confidence score
    confidence: f32,
}

/// Result of HyDE generation
#[derive(Debug, Clone)]
pub struct HyDEResult {
    /// Original query
    pub query: String,

    /// Generated hypothetical answer/document
    pub hypothetical_answer: String,

    /// Embedding of the hypothetical document
    pub embedding: Option<crate::embeddings::Embedding>,

    /// Generation method used
    pub generation_method: String,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Generation metadata
    pub metadata: HyDEMetadata,
}

/// Metadata for HyDE generation
#[derive(Debug, Clone)]
pub struct HyDEMetadata {
    /// Generation time in milliseconds
    pub generation_time_ms: u64,

    /// Document length in characters
    pub document_length: usize,

    /// Document length in tokens (approximate)
    pub estimated_tokens: usize,

    /// Query type detected
    pub detected_query_type: String,

    /// Domain detected
    pub detected_domain: Option<String>,

    /// Template used
    pub template_used: Option<String>,
}

impl HyDEGenerator {
    /// Create a new HyDE generator
    pub fn new(config: HyDEConfig, embedding_provider: Arc<dyn EmbeddingProvider>) -> Self {
        let templates = Self::init_templates();
        let answer_patterns = Self::init_answer_patterns();

        Self {
            config,
            embedding_provider,
            templates,
            answer_patterns,
        }
    }

    /// Generate hypothetical documents for a query
    pub async fn generate(&self, query: &str) -> RragResult<Vec<HyDEResult>> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        // Detect query characteristics
        let query_type = self.detect_query_type(query);
        let domain = if self.config.enable_domain_awareness {
            self.detect_domain(query)
        } else {
            None
        };

        // Generate hypothetical documents using different strategies
        let hypothetical_docs = self.generate_hypothetical_documents(query, &query_type, &domain);

        for (i, doc) in hypothetical_docs.iter().enumerate() {
            if doc.len() < self.config.min_document_length
                || doc.len() > self.config.max_document_length
            {
                continue;
            }

            // Generate embedding for the hypothetical document
            let embedding = match self.embedding_provider.embed_text(doc).await {
                Ok(emb) => Some(emb),
                Err(_) => None, // Continue without embedding if it fails
            };

            let confidence = self.calculate_confidence(query, doc, &query_type);

            if confidence >= self.config.confidence_threshold {
                results.push(HyDEResult {
                    query: query.to_string(),
                    hypothetical_answer: doc.clone(),
                    embedding,
                    generation_method: format!("pattern_{}", i),
                    confidence,
                    metadata: HyDEMetadata {
                        generation_time_ms: start_time.elapsed().as_millis() as u64,
                        document_length: doc.len(),
                        estimated_tokens: doc.split_whitespace().count(),
                        detected_query_type: query_type.clone(),
                        detected_domain: domain.clone(),
                        template_used: Some(format!("template_{}", i)),
                    },
                });
            }

            if results.len() >= self.config.num_hypothetical_docs {
                break;
            }
        }

        Ok(results)
    }

    /// Generate hypothetical documents using various strategies
    fn generate_hypothetical_documents(
        &self,
        query: &str,
        query_type: &str,
        domain: &Option<String>,
    ) -> Vec<String> {
        let mut documents = Vec::new();

        // Strategy 1: Template-based generation
        if let Some(templates) = self.templates.get(query_type) {
            for template in templates {
                let doc = self.apply_template(query, template, domain);
                documents.push(doc);
            }
        }

        // Strategy 2: Pattern-based generation
        for pattern in &self.answer_patterns {
            if pattern
                .triggers
                .iter()
                .any(|trigger| query.to_lowercase().contains(&trigger.to_lowercase()))
            {
                let generated_docs = (pattern.generator)(query, &self.config);
                documents.extend(generated_docs);
            }
        }

        // Strategy 3: Fallback generic generation
        if documents.is_empty() {
            documents.extend(self.generate_generic_documents(query, query_type));
        }

        // Limit and deduplicate
        documents.sort();
        documents.dedup();
        documents.truncate(self.config.num_hypothetical_docs * 2); // Generate more, filter later

        documents
    }

    /// Apply a template to generate a hypothetical document
    fn apply_template(
        &self,
        query: &str,
        template: &DocumentTemplate,
        domain: &Option<String>,
    ) -> String {
        let mut result = template.pattern.clone();

        // Extract key terms from query
        let key_terms = self.extract_key_terms(query);
        let main_subject = self.extract_main_subject(query);

        // Replace placeholders
        result = result.replace("{query}", query);
        result = result.replace("{subject}", &main_subject);
        result = result.replace("{key_terms}", &key_terms.join(", "));

        if let Some(domain_name) = domain {
            result = result.replace("{domain}", domain_name);
        }

        // Clean up the result
        self.clean_generated_text(&result)
    }

    /// Generate generic hypothetical documents
    fn generate_generic_documents(&self, query: &str, query_type: &str) -> Vec<String> {
        let mut documents = Vec::new();
        let main_subject = self.extract_main_subject(query);

        match query_type {
            "definitional" => {
                documents.push(format!(
                    "{} is a concept that refers to the fundamental principles and mechanisms underlying this topic. \
                    It encompasses various aspects including its core definition, key characteristics, and primary applications. \
                    Understanding {} requires examining its historical development, theoretical foundations, and practical implications. \
                    The concept plays a crucial role in its respective field and has significant impact on related areas.",
                    main_subject, main_subject
                ));
            }
            "procedural" => {
                documents.push(format!(
                    "To accomplish {} successfully, there are several important steps to follow. \
                    First, it's essential to understand the underlying principles and requirements. \
                    The process typically involves careful planning, systematic execution, and continuous monitoring. \
                    Key considerations include proper preparation, attention to detail, and adherence to best practices. \
                    Following these guidelines will help ensure optimal results and avoid common pitfalls.",
                    main_subject
                ));
            }
            "comparative" => {
                documents.push(format!(
                    "When comparing different approaches to {}, several factors must be considered. \
                    Each option has distinct advantages and disadvantages that affect their suitability for various use cases. \
                    The comparison involves analyzing performance characteristics, resource requirements, and implementation complexity. \
                    Understanding these differences helps in making informed decisions based on specific needs and constraints.",
                    main_subject
                ));
            }
            "factual" => {
                documents.push(format!(
                    "Regarding {}, there are several important facts and key information points to consider. \
                    The available evidence and research data provide insights into various aspects of this topic. \
                    Historical context, current developments, and future trends all contribute to a comprehensive understanding. \
                    These facts form the foundation for deeper analysis and informed decision-making.",
                    main_subject
                ));
            }
            _ => {
                documents.push(format!(
                    "{} represents an important topic that deserves careful examination. \
                    The subject encompasses multiple dimensions including theoretical aspects, practical applications, and real-world implications. \
                    Understanding this topic requires considering various perspectives, analyzing available information, and drawing meaningful conclusions. \
                    This comprehensive approach ensures a thorough grasp of the subject matter.",
                    main_subject
                ));
            }
        }

        documents
    }

    /// Detect the type of query
    fn detect_query_type(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();

        if query_lower.starts_with("what is") || query_lower.starts_with("define") {
            "definitional".to_string()
        } else if query_lower.starts_with("how to") || query_lower.contains("step") {
            "procedural".to_string()
        } else if query_lower.contains("compare")
            || query_lower.contains("vs")
            || query_lower.contains("difference")
        {
            "comparative".to_string()
        } else if query_lower.starts_with("when")
            || query_lower.starts_with("where")
            || query_lower.starts_with("who")
        {
            "factual".to_string()
        } else if query_lower.starts_with("why") {
            "causal".to_string()
        } else if query_lower.starts_with("list") || query_lower.contains("examples") {
            "enumerative".to_string()
        } else {
            "general".to_string()
        }
    }

    /// Detect domain from query
    fn detect_domain(&self, query: &str) -> Option<String> {
        let query_lower = query.to_lowercase();

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
                    "tech",
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
                    "scientific",
                    "hypothesis",
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
                    "company",
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
                    "healthcare",
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
                    "academic",
                ],
            ),
            (
                "finance",
                vec![
                    "money",
                    "investment",
                    "financial",
                    "bank",
                    "trading",
                    "economics",
                    "cost",
                    "price",
                ],
            ),
        ];

        for (domain, keywords) in &domains {
            let matches = keywords
                .iter()
                .filter(|&&keyword| query_lower.contains(keyword))
                .count();

            if matches >= 2 || (matches == 1 && query_lower.split_whitespace().count() <= 5) {
                return Some(domain.to_string());
            }
        }

        None
    }

    /// Extract key terms from query
    fn extract_key_terms(&self, query: &str) -> Vec<String> {
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can", "what", "how", "why",
            "when", "where", "who", "which",
        ];

        query
            .split_whitespace()
            .filter(|word| {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                !stop_words.contains(&clean_word.as_str()) && clean_word.len() > 2
            })
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .collect()
    }

    /// Extract main subject from query
    fn extract_main_subject(&self, query: &str) -> String {
        let key_terms = self.extract_key_terms(query);
        if !key_terms.is_empty() {
            key_terms[0].clone()
        } else {
            "the topic".to_string()
        }
    }

    /// Clean generated text
    fn clean_generated_text(&self, text: &str) -> String {
        text.trim()
            .replace("  ", " ")
            .replace("\n\n", "\n")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Calculate confidence score for generated document
    fn calculate_confidence(&self, query: &str, document: &str, query_type: &str) -> f32 {
        let mut confidence = 0.5; // Base confidence

        // Check length appropriateness
        if document.len() >= self.config.min_document_length
            && document.len() <= self.config.max_document_length
        {
            confidence += 0.1;
        }

        // Check if key terms from query appear in document
        let query_terms = self.extract_key_terms(query);
        let document_lower = document.to_lowercase();
        let term_matches = query_terms
            .iter()
            .filter(|term| document_lower.contains(&term.to_lowercase()))
            .count();

        if !query_terms.is_empty() {
            confidence += (term_matches as f32 / query_terms.len() as f32) * 0.3;
        }

        // Bonus for appropriate query type handling
        match query_type {
            "definitional" if document.contains("is") || document.contains("refers to") => {
                confidence += 0.1
            }
            "procedural" if document.contains("step") || document.contains("process") => {
                confidence += 0.1
            }
            "comparative" if document.contains("compare") || document.contains("difference") => {
                confidence += 0.1
            }
            _ => {}
        }

        confidence.min(1.0)
    }

    /// Initialize document templates
    fn init_templates() -> HashMap<String, Vec<DocumentTemplate>> {
        let mut templates = HashMap::new();

        // Definitional templates
        templates.insert("definitional".to_string(), vec![
            DocumentTemplate {
                name: "concept_definition".to_string(),
                pattern: "{subject} is a fundamental concept in {domain} that encompasses several key aspects. It refers to the systematic approach and principles underlying this area of study. The definition includes both theoretical foundations and practical applications, making it essential for understanding related topics.".to_string(),
                query_types: vec!["definitional".to_string()],
                confidence: 0.8,
            },
        ]);

        // Procedural templates
        templates.insert("procedural".to_string(), vec![
            DocumentTemplate {
                name: "how_to_guide".to_string(),
                pattern: "To effectively accomplish {subject}, follow these systematic steps and best practices. The process requires careful planning, proper execution, and continuous monitoring. Begin by understanding the requirements, then proceed with methodical implementation while considering potential challenges and solutions.".to_string(),
                query_types: vec!["procedural".to_string()],
                confidence: 0.8,
            },
        ]);

        // Comparative templates
        templates.insert("comparative".to_string(), vec![
            DocumentTemplate {
                name: "comparison_analysis".to_string(),
                pattern: "When analyzing {subject}, several important factors distinguish different approaches and options. Each alternative offers unique advantages and limitations that affect performance, cost, and suitability for various use cases. The comparison reveals critical differences in functionality, efficiency, and implementation requirements.".to_string(),
                query_types: vec!["comparative".to_string()],
                confidence: 0.8,
            },
        ]);

        templates
    }

    /// Initialize answer patterns
    fn init_answer_patterns() -> Vec<AnswerPattern> {
        vec![
            AnswerPattern {
                name: "technical_explanation".to_string(),
                triggers: vec![
                    "algorithm".to_string(),
                    "system".to_string(),
                    "technology".to_string(),
                ],
                generator: |query, _config| {
                    vec![format!(
                        "The technical implementation of {} involves several sophisticated components working together. \
                        The system architecture incorporates advanced algorithms and optimized data structures to ensure \
                        efficient performance and scalability. Key technical considerations include resource management, \
                        error handling, and performance optimization strategies.",
                        query
                    )]
                },
                confidence: 0.7,
            },
            AnswerPattern {
                name: "research_summary".to_string(),
                triggers: vec![
                    "research".to_string(),
                    "study".to_string(),
                    "analysis".to_string(),
                ],
                generator: |query, _config| {
                    vec![format!(
                        "Recent research on {} has revealed significant insights and findings that advance our understanding \
                        of this field. Multiple studies have examined various aspects, employing rigorous methodologies \
                        and comprehensive data analysis. The research findings contribute valuable knowledge and inform \
                        evidence-based practices and future investigations.",
                        query
                    )]
                },
                confidence: 0.7,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddingProvider;

    #[tokio::test]
    async fn test_hyde_generation() {
        let provider = Arc::new(MockEmbeddingProvider::new());
        let hyde = HyDEGenerator::new(HyDEConfig::default(), provider);

        let results = hyde.generate("What is machine learning?").await.unwrap();

        assert!(!results.is_empty());
        assert!(results[0].confidence > 0.0);
        assert!(results[0].hypothetical_answer.len() > 50);
        assert_eq!(results[0].metadata.detected_query_type, "definitional");
    }

    #[tokio::test]
    async fn test_procedural_query() {
        let provider = Arc::new(MockEmbeddingProvider::new());
        let hyde = HyDEGenerator::new(HyDEConfig::default(), provider);

        let results = hyde.generate("How to implement a REST API?").await.unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].metadata.detected_query_type, "procedural");
        assert!(
            results[0].hypothetical_answer.contains("step")
                || results[0].hypothetical_answer.contains("process")
        );
    }

    #[tokio::test]
    async fn test_comparative_query() {
        let provider = Arc::new(MockEmbeddingProvider::new());
        let hyde = HyDEGenerator::new(HyDEConfig::default(), provider);

        let results = hyde
            .generate("Python vs Rust performance comparison")
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].metadata.detected_query_type, "comparative");
    }

    #[test]
    fn test_query_type_detection() {
        let provider = Arc::new(MockEmbeddingProvider::new());
        let hyde = HyDEGenerator::new(HyDEConfig::default(), provider);

        assert_eq!(hyde.detect_query_type("What is AI?"), "definitional");
        assert_eq!(hyde.detect_query_type("How to code?"), "procedural");
        assert_eq!(hyde.detect_query_type("Python vs Java"), "comparative");
        assert_eq!(hyde.detect_query_type("When was it built?"), "factual");
    }

    #[test]
    fn test_domain_detection() {
        let provider = Arc::new(MockEmbeddingProvider::new());
        let hyde = HyDEGenerator::new(HyDEConfig::default(), provider);

        assert_eq!(
            hyde.detect_domain("machine learning algorithm"),
            Some("technology".to_string())
        );
        assert_eq!(
            hyde.detect_domain("medical research study"),
            Some("health".to_string())
        );
        assert_eq!(
            hyde.detect_domain("market analysis strategy"),
            Some("business".to_string())
        );
        assert_eq!(hyde.detect_domain("simple question"), None);
    }
}
