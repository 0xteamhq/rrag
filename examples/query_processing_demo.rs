//! # Query Processing Demo
//! 
//! Demonstrates the complete query rewriting and expansion system including:
//! - Query classification and intent detection
//! - Query rewriting for grammar and style
//! - Query expansion with synonyms and related terms
//! - Query decomposition for complex queries
//! - HyDE generation for hypothetical documents

use rrag::query::{
    QueryProcessor, QueryProcessorConfig,
    QueryClassifier, QueryRewriter, QueryExpander, QueryDecomposer, HyDEGenerator,
    QueryRewriteConfig, ExpansionConfig, HyDEConfig,
};
use rrag::embeddings::MockEmbeddingProvider;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 RRAG Query Processing Demo");
    println!("=====================================\n");

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
        println!("🔍 Query {}: {}", i + 1, query);
        println!("{}", "─".repeat(50));
        
        // Process the query through our complete system
        match query_processor.process_query(query).await {
            Ok(result) => {
                println!("📊 Processing Results:");
                println!("  • Techniques Applied: {:?}", result.metadata.techniques_applied);
                println!("  • Processing Time: {}ms", result.metadata.processing_time_ms);
                
                // Show classification results
                if let Some(classification) = &result.classification {
                    println!("\n🎯 Query Classification:");
                    println!("  • Intent: {:?}", classification.intent);
                    println!("  • Type: {:?}", classification.query_type);
                    println!("  • Confidence: {:.2}", classification.confidence);
                    println!("  • Complexity: {:.2}", classification.metadata.complexity);
                    println!("  • Needs Context: {}", classification.metadata.needs_context);
                    println!("  • Suggested Strategies: {:?}", classification.metadata.suggested_strategies);
                }
                
                // Show rewritten queries
                if !result.rewritten_queries.is_empty() {
                    println!("\n✏️ Rewritten Queries:");
                    for (j, rewrite) in result.rewritten_queries.iter().enumerate() {
                        println!("  {}. [{}] {} (confidence: {:.2})", 
                                j + 1, 
                                format!("{:?}", rewrite.strategy),
                                rewrite.rewritten_query,
                                rewrite.confidence
                        );
                    }
                }
                
                // Show expanded queries
                if !result.expanded_queries.is_empty() {
                    println!("\n🔍 Expanded Queries:");
                    for (j, expansion) in result.expanded_queries.iter().enumerate() {
                        println!("  {}. [{}] {} (confidence: {:.2})", 
                                j + 1,
                                format!("{:?}", expansion.expansion_type),
                                expansion.expanded_query,
                                expansion.confidence
                        );
                        if !expansion.added_terms.is_empty() {
                            println!("     Added terms: {}", expansion.added_terms.join(", "));
                        }
                    }
                }
                
                // Show sub-queries from decomposition
                if !result.sub_queries.is_empty() {
                    println!("\n🔄 Decomposed Sub-queries:");
                    for (j, sub_query) in result.sub_queries.iter().enumerate() {
                        println!("  {}. [{}] {} (priority: {:.2}, confidence: {:.2})", 
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
                    println!("\n📄 HyDE Hypothetical Documents:");
                    for (j, hyde) in result.hyde_results.iter().enumerate() {
                        println!("  {}. [{}] (confidence: {:.2})", 
                                j + 1,
                                hyde.generation_method,
                                hyde.confidence
                        );
                        println!("     Query Type: {}", hyde.metadata.detected_query_type);
                        if let Some(domain) = &hyde.metadata.detected_domain {
                            println!("     Domain: {}", domain);
                        }
                        // Show first 150 chars of generated document
                        let preview = if hyde.hypothetical_answer.len() > 150 {
                            format!("{}...", &hyde.hypothetical_answer[..150])
                        } else {
                            hyde.hypothetical_answer.clone()
                        };
                        println!("     Generated: {}", preview);
                    }
                }
                
                // Show final optimized queries
                println!("\n🎯 Final Optimized Queries:");
                for (j, final_query) in result.final_queries.iter().enumerate() {
                    println!("  {}. {}", j + 1, final_query);
                }
                
                // Show warnings if any
                if !result.metadata.warnings.is_empty() {
                    println!("\n⚠️ Warnings:");
                    for warning in &result.metadata.warnings {
                        println!("  • {}", warning);
                    }
                }
            }
            Err(e) => {
                println!("❌ Error processing query: {}", e);
            }
        }
        
        println!("\n{}\n", "=".repeat(70));
    }

    // Demonstrate individual components
    println!("🔧 Individual Component Demonstrations");
    println!("=====================================\n");

    demo_classifier().await?;
    demo_rewriter().await?;
    demo_expander().await?;
    demo_decomposer().await?;
    demo_hyde(embedding_provider).await?;

    Ok(())
}

async fn demo_classifier() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Query Classifier Demo:");
    
    let classifier = QueryClassifier::new();
    let queries = [
        "What is machine learning?",
        "How to implement a REST API?",
        "Compare Python vs Rust performance",
        "Fix my broken code",
    ];
    
    for query in &queries {
        let result = classifier.classify(query).await?;
        println!("  • '{}' → Intent: {:?}, Type: {:?}", 
                query, result.intent, result.query_type);
    }
    println!();
    Ok(())
}

async fn demo_rewriter() -> Result<(), Box<dyn std::error::Error>> {
    println!("✏️ Query Rewriter Demo:");
    
    let rewriter = QueryRewriter::new(QueryRewriteConfig::default());
    let queries = [
        "wat is ML",
        "how   does  this work???",
        "teh API is not working",
    ];
    
    for query in &queries {
        let results = rewriter.rewrite(query).await?;
        if !results.is_empty() {
            println!("  • '{}' → '{}'", query, results[0].rewritten_query);
        }
    }
    println!();
    Ok(())
}

async fn demo_expander() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Query Expander Demo:");
    
    let expander = QueryExpander::new(ExpansionConfig::default());
    let queries = [
        "fast algorithm",
        "ML model",
        "programming languages",
    ];
    
    for query in &queries {
        let results = expander.expand(query).await?;
        for result in results {
            if !result.added_terms.is_empty() {
                println!("  • '{}' + [{}]", query, result.added_terms.join(", "));
            }
        }
    }
    println!();
    Ok(())
}

async fn demo_decomposer() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Query Decomposer Demo:");
    
    let decomposer = QueryDecomposer::new();
    let queries = [
        "What is machine learning and how does deep learning work?",
        "Compare Python vs Rust for system programming",
    ];
    
    for query in &queries {
        let results = decomposer.decompose(query).await?;
        if !results.is_empty() {
            println!("  • '{}' →", query);
            for sub_query in results {
                println!("    - {}", sub_query.query);
            }
        }
    }
    println!();
    Ok(())
}

async fn demo_hyde(embedding_provider: Arc<MockEmbeddingProvider>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 HyDE Generator Demo:");
    
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
            println!("  • '{}' → {}", query, preview);
        }
    }
    println!();
    Ok(())
}