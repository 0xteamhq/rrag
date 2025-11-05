//! # Query Processing Demo
//!
//! Demonstrates the complete query rewriting and expansion system including:
//! - Query classification and intent detection
//! - Query rewriting for grammar and style
//! - Query expansion with synonyms and related terms
//! - Query decomposition for complex queries
//! - HyDE generation for hypothetical documents

use rrag::embeddings::MockEmbeddingProvider;
use rrag::query::{
    ExpansionConfig, HyDEConfig, HyDEGenerator, QueryClassifier, QueryDecomposer, QueryExpander,
    QueryProcessor, QueryProcessorConfig, QueryRewriteConfig, QueryRewriter,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🔍 RRAG Query Processing Demo");
    tracing::debug!("=====================================\n");

    // Create mock embedding provider for HyDE
    let embedding_provider = Arc::new(MockEmbeddingProvider::new());

    // Create query processor with all features enabled
    let query_processor = QueryProcessor::new(QueryProcessorConfig::default())
        .with_embedding_provider(embedding_provider.clone());

    // Test different types of queries
    let test_queries = vec![
        "wat is machien lerning and how dose deep lerning work???",
        "What are the differences between Python vs Rust for system programming?",
        "How to implement REST API authentication and also secure database access?",
        "When did the Renaissance start and what happened before it in European history?",
        "Compare machine learning algorithms performance",
        "What is AI?", // Simple query that shouldn't be over-processed
    ];

    for (i, query) in test_queries.iter().enumerate() {
        tracing::debug!("🔍 Query {}: {}", i + 1, query);
        tracing::debug!("{}", "─".repeat(50));

        // Process the query through our complete system
        match query_processor.process_query(query).await {
            Ok(result) => {
                tracing::debug!("📊 Processing Results:");
                tracing::debug!(
                    "  • Techniques Applied: {:?}",
                    result.metadata.techniques_applied
                );
                tracing::debug!(
                    "  • Processing Time: {}ms",
                    result.metadata.processing_time_ms
                );

                // Show classification results
                if let Some(classification) = &result.classification {
                    tracing::debug!("\n🎯 Query Classification:");
                    tracing::debug!("  • Intent: {:?}", classification.intent);
                    tracing::debug!("  • Type: {:?}", classification.query_type);
                    tracing::debug!("  • Confidence: {:.2}", classification.confidence);
                    tracing::debug!("  • Complexity: {:.2}", classification.metadata.complexity);
                    tracing::debug!(
                        "  • Needs Context: {}",
                        classification.metadata.needs_context
                    );
                    tracing::debug!(
                        "  • Suggested Strategies: {:?}",
                        classification.metadata.suggested_strategies
                    );
                }

                // Show rewritten queries
                if !result.rewritten_queries.is_empty() {
                    tracing::debug!("\n✏️ Rewritten Queries:");
                    for (j, rewrite) in result.rewritten_queries.iter().enumerate() {
                        tracing::debug!(
                            "  {}. [{}] {} (confidence: {:.2})",
                            j + 1,
                            format!("{:?}", rewrite.strategy),
                            rewrite.rewritten_query,
                            rewrite.confidence
                        );
                    }
                }

                // Show expanded queries
                if !result.expanded_queries.is_empty() {
                    tracing::debug!("\n🔍 Expanded Queries:");
                    for (j, expansion) in result.expanded_queries.iter().enumerate() {
                        tracing::debug!(
                            "  {}. [{}] {} (confidence: {:.2})",
                            j + 1,
                            format!("{:?}", expansion.expansion_type),
                            expansion.expanded_query,
                            expansion.confidence
                        );
                        if !expansion.added_terms.is_empty() {
                            tracing::debug!("     Added terms: {}", expansion.added_terms.join(", "));
                        }
                    }
                }

                // Show sub-queries from decomposition
                if !result.sub_queries.is_empty() {
                    tracing::debug!("\n🔄 Decomposed Sub-queries:");
                    for (j, sub_query) in result.sub_queries.iter().enumerate() {
                        tracing::debug!(
                            "  {}. [{}] {} (priority: {:.2}, confidence: {:.2})",
                            j + 1,
                            format!("{:?}", sub_query.strategy),
                            sub_query.query,
                            sub_query.priority,
                            sub_query.confidence
                        );
                    }
                }

                // Show HyDE results
                if !result.hyde_results.is_empty() {
                    tracing::debug!("\n📄 HyDE Hypothetical Documents:");
                    for (j, hyde) in result.hyde_results.iter().enumerate() {
                        tracing::debug!(
                            "  {}. [{}] (confidence: {:.2})",
                            j + 1,
                            hyde.generation_method,
                            hyde.confidence
                        );
                        tracing::debug!("     Query Type: {}", hyde.metadata.detected_query_type);
                        if let Some(domain) = &hyde.metadata.detected_domain {
                            tracing::debug!("     Domain: {}", domain);
                        }
                        // Show first 150 chars of generated document
                        let preview = if hyde.hypothetical_answer.len() > 150 {
                            format!("{}...", &hyde.hypothetical_answer[..150])
                        } else {
                            hyde.hypothetical_answer.clone()
                        };
                        tracing::debug!("     Generated: {}", preview);
                    }
                }

                // Show final optimized queries
                tracing::debug!("\n🎯 Final Optimized Queries:");
                for (j, final_query) in result.final_queries.iter().enumerate() {
                    tracing::debug!("  {}. {}", j + 1, final_query);
                }

                // Show warnings if any
                if !result.metadata.warnings.is_empty() {
                    tracing::debug!("\n⚠️ Warnings:");
                    for warning in &result.metadata.warnings {
                        tracing::debug!("  • {}", warning);
                    }
                }
            }
            Err(e) => {
                error!(" Error processing query: {}", e);
            }
        }

        tracing::debug!("\n{}\n", "=".repeat(70));
    }

    // Demonstrate individual components
    tracing::debug!("🔧 Individual Component Demonstrations");
    tracing::debug!("=====================================\n");

    demo_classifier().await?;
    demo_rewriter().await?;
    demo_expander().await?;
    demo_decomposer().await?;
    demo_hyde(embedding_provider).await?;

    Ok(())
}

async fn demo_classifier() -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🎯 Query Classifier Demo:");

    let classifier = QueryClassifier::new();
    let queries = [
        "What is machine learning?",
        "How to implement a REST API?",
        "Compare Python vs Rust performance",
        "Fix my broken code",
    ];

    for query in &queries {
        let result = classifier.classify(query).await?;
        tracing::debug!(
            "  • '{}' → Intent: {:?}, Type: {:?}",
            query, result.intent, result.query_type
        );
    }
    Ok(())
}

async fn demo_rewriter() -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("✏️ Query Rewriter Demo:");

    let rewriter = QueryRewriter::new(QueryRewriteConfig::default());
    let queries = [
        "wat is ML",
        "how   does  this work???",
        "teh API is not working",
    ];

    for query in &queries {
        let results = rewriter.rewrite(query).await?;
        if !results.is_empty() {
            tracing::debug!("  • '{}' → '{}'", query, results[0].rewritten_query);
        }
    }
    Ok(())
}

async fn demo_expander() -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🔍 Query Expander Demo:");

    let expander = QueryExpander::new(ExpansionConfig::default());
    let queries = ["fast algorithm", "ML model", "programming languages"];

    for query in &queries {
        let results = expander.expand(query).await?;
        for result in results {
            if !result.added_terms.is_empty() {
                tracing::debug!("  • '{}' + [{}]", query, result.added_terms.join(", "));
            }
        }
    }
    Ok(())
}

async fn demo_decomposer() -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🔄 Query Decomposer Demo:");

    let decomposer = QueryDecomposer::new();
    let queries = [
        "What is machine learning and how does deep learning work?",
        "Compare Python vs Rust for system programming",
    ];

    for query in &queries {
        let results = decomposer.decompose(query).await?;
        if !results.is_empty() {
            tracing::debug!("  • '{}' →", query);
            for sub_query in results {
                tracing::debug!("    - {}", sub_query.query);
            }
        }
    }
    Ok(())
}

async fn demo_hyde(
    embedding_provider: Arc<MockEmbeddingProvider>,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("📄 HyDE Generator Demo:");

    let hyde = HyDEGenerator::new(HyDEConfig::default(), embedding_provider);
    let queries = [
        "What is machine learning?",
        "How to implement authentication?",
    ];

    for query in &queries {
        let results = hyde.generate(query).await?;
        if !results.is_empty() {
            let preview = if results[0].hypothetical_answer.len() > 100 {
                format!("{}...", &results[0].hypothetical_answer[..100])
            } else {
                results[0].hypothetical_answer.clone()
            };
            tracing::debug!("  • '{}' → {}", query, preview);
        }
    }
    Ok(())
}
