//! # RRAG Getting Started Guide
//!
//! This comprehensive guide walks you through RRAG from basic setup to advanced features:
//! - Quick start with minimal setup
//! - Step-by-step tutorial with explanations
//! - Common patterns and best practices
//! - Troubleshooting and optimization tips
//! - Next steps for production deployment
//!
//! Run with: `cargo run --bin getting_started_guide`

use rrag::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("📚 RRAG Framework - Getting Started Guide");
    tracing::debug!("=========================================\n");

    tracing::debug!("Welcome to RRAG (Rust RAG Framework)! This guide will walk you through");
    tracing::debug!("the key concepts and help you build your first RAG application.\n");

    // Step 1: Understanding RRAG
    tracing::debug!("🎯 Step 1: Understanding RRAG");
    tracing::debug!("------------------------------");
    explain_rrag_concepts();

    // Step 2: Quick Start
    tracing::debug!("🚀 Step 2: Quick Start - Your First RAG System");
    tracing::debug!("-----------------------------------------------");
    let basic_system = quick_start_tutorial().await?;

    // Step 3: Document Processing
    tracing::debug!("📄 Step 3: Document Processing and Chunking");
    tracing::debug!("-------------------------------------------");
    document_processing_tutorial(&basic_system).await?;

    // Step 4: Advanced Search
    tracing::debug!("🔍 Step 4: Advanced Search Techniques");
    tracing::debug!("-------------------------------------");
    advanced_search_tutorial(&basic_system).await?;

    // Step 5: Query Enhancement
    tracing::debug!("✨ Step 5: Query Enhancement and Processing");
    tracing::debug!("------------------------------------------");
    query_enhancement_tutorial().await?;

    // Step 6: Performance Optimization
    tracing::debug!("⚡ Step 6: Performance Optimization");
    tracing::debug!("---------------------------------");
    performance_optimization_tutorial().await?;

    // Step 7: Production Considerations
    tracing::debug!("🏭 Step 7: Production Deployment");
    tracing::debug!("-------------------------------");
    production_considerations();

    // Step 8: Next Steps
    tracing::debug!("📈 Step 8: Next Steps and Advanced Features");
    tracing::debug!("------------------------------------------");
    next_steps_guide();

    tracing::debug!("🎉 Congratulations! You've completed the RRAG getting started guide!");
    tracing::debug!("You're now ready to build powerful RAG applications with Rust.");
    tracing::debug!("\nHappy coding! 🦀");

    Ok(())
}

fn explain_rrag_concepts() {
    tracing::debug!("RRAG is a high-performance Rust framework for building Retrieval-Augmented");
    tracing::debug!("Generation (RAG) systems. Here are the key concepts:");
    tracing::debug!("📖 Core Components:");
    tracing::debug!("  • Document Processing: Chunking, cleaning, and preprocessing");
    tracing::debug!("  • Embedding Generation: Converting text to vector representations");
    tracing::debug!("  • Vector Storage: Efficient storage and indexing of embeddings");
    tracing::debug!("  • Retrieval: Finding relevant documents using similarity search");
    tracing::debug!("  • Generation: Augmenting LLM responses with retrieved context");
    tracing::debug!("🎯 Key Benefits:");
    tracing::debug!("  • High Performance: Rust's speed for production workloads");
    tracing::debug!("  • Memory Safety: Zero-cost abstractions without runtime overhead");
    tracing::debug!("  • Async Support: Non-blocking I/O for scalable applications");
    tracing::debug!("  • Modular Design: Use only the components you need");
    tracing::debug!("  • Production Ready: Built-in monitoring, caching, and error handling");
}

async fn quick_start_tutorial() -> RragResult<RragSystem> {
    tracing::debug!("Let's build your first RAG system in just a few lines of code!");

    tracing::debug!("1. Create a basic RRAG system:");
    tracing::debug!("```rust");
    tracing::debug!("use rrag::prelude::*;");
    tracing::debug!("let system = RragSystemBuilder::new()");
    tracing::debug!("    .with_embedding_provider(LocalEmbeddingProvider::new())");
    tracing::debug!("    .with_vector_store(InMemoryVectorStore::new())");
    tracing::debug!("    .build()");
    tracing::debug!("    .await?;");
    tracing::debug!("```");

    // Actually build the system
    let system = RragSystemBuilder::new().build().await?;

    tracing::debug!("✅ System created successfully!");

    tracing::debug!("2. Add some sample documents:");
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information.",
    ];

    tracing::debug!("```rust");
    tracing::debug!("let documents = vec![");
    for doc in &documents {
        tracing::debug!("    \"{}\",", doc);
    }
    tracing::debug!("]);");
    tracing::debug!("```");

    // Process and index documents
    let start_time = Instant::now();
    for (i, doc_text) in documents.iter().enumerate() {
        let document = Document::with_id(format!("doc_{}", i), doc_text.to_string())
            .with_metadata_map(HashMap::from([(
                "topic".to_string(),
                serde_json::Value::String("AI/ML".to_string()),
            )]));

        system.process_document(document).await?;
    }
    let processing_time = start_time.elapsed();

    tracing::debug!(
        "✅ Added {} documents in {:.2}ms",
        documents.len(),
        processing_time.as_millis()
    );

    tracing::debug!("3. Perform your first search:");
    tracing::debug!("```rust");
    tracing::debug!("let results = system.search(\"neural networks\", Some(3)).await?;");
    tracing::debug!("```");

    let results = system
        .search("neural networks".to_string(), Some(3))
        .await?;

    tracing::debug!("📊 Search Results:");
    for (i, result) in results.results.iter().enumerate() {
        tracing::debug!(
            "  {}. Score: {:.3} - {}",
            i + 1,
            result.score,
            result.content.chars().take(60).collect::<String>() + "..."
        );
    }

    Ok(system)
}

async fn document_processing_tutorial(_system: &RragSystem) -> RragResult<()> {
    tracing::debug!("RRAG provides powerful document processing capabilities:");

    tracing::debug!("1. Text Preprocessing:");
    tracing::debug!("```rust");
    tracing::debug!("use rrag::preprocessing::{{TextPreprocessor, PreprocessingStep}};");
    tracing::debug!("let preprocessor = TextPreprocessor::new()");
    tracing::debug!("    .add_step(PreprocessingStep::RemoveExtraWhitespace)");
    tracing::debug!("    .add_step(PreprocessingStep::NormalizeUnicode)");
    tracing::debug!("    .add_step(PreprocessingStep::RemoveHtmlTags);");
    tracing::debug!("```");

    tracing::debug!("2. Document Chunking:");
    tracing::debug!("```rust");
    tracing::debug!("use rrag::chunking::{{DocumentChunker, ChunkingStrategy}};");
    tracing::debug!("let chunker = DocumentChunker::new()");
    tracing::debug!("    .with_strategy(ChunkingStrategy::Semantic)");
    tracing::debug!("    .with_chunk_size(512)");
    tracing::debug!("    .with_overlap(50);");
    tracing::debug!("```");

    // Demonstrate chunking
    let long_document = "This is a long document that will be split into chunks. \
        Machine learning is revolutionizing how we process data. Deep learning, \
        a subset of machine learning, uses neural networks with multiple layers. \
        These networks can learn complex patterns from large datasets. Natural \
        language processing is another important area that helps computers \
        understand human language. Computer vision enables machines to \
        interpret visual information from images and videos.";

    tracing::debug!("Example: Chunking a longer document");
    tracing::debug!("Original length: {} characters", long_document.len());

    // Simulate chunking (in a real implementation, you'd use the actual chunker)
    let chunks = simulate_chunking(long_document, 100);
    tracing::debug!("Number of chunks: {}", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        tracing::debug!(
            "  Chunk {}: {}...",
            i + 1,
            chunk.chars().take(50).collect::<String>()
        );
    }

    tracing::debug!("3. Metadata Enhancement:");
    tracing::debug!("```rust");
    tracing::debug!("let document = Document::new(content)");
    tracing::debug!("    .with_metadata(HashMap::from([");
    tracing::debug!("        (\"source\".to_string(), \"research_paper.pdf\".to_string()),");
    tracing::debug!("        (\"section\".to_string(), \"introduction\".to_string()),");
    tracing::debug!("        (\"page\".to_string(), \"1\".to_string()),");
    tracing::debug!("    ]));");
    tracing::debug!("```");

    Ok(())
}

async fn advanced_search_tutorial(system: &RragSystem) -> RragResult<()> {
    tracing::debug!("RRAG supports multiple search strategies and algorithms:");

    tracing::debug!("1. Similarity Search:");
    let similarity_results = system
        .search("artificial intelligence".to_string(), Some(2))
        .await?;
    tracing::debug!("Query: 'artificial intelligence'");
    for result in similarity_results.results {
        tracing::debug!(
            "  • Score: {:.3} - {}",
            result.score,
            result.content.chars().take(60).collect::<String>() + "..."
        );
    }

    tracing::debug!("2. Hybrid Search (combines multiple signals):");
    tracing::debug!("```rust");
    tracing::debug!("let config = HybridSearchConfig::new()");
    tracing::debug!("    .with_semantic_weight(0.7)");
    tracing::debug!("    .with_keyword_weight(0.3);");
    tracing::debug!("let results = system.hybrid_search(\"deep learning\", config).await?;");
    tracing::debug!("```");

    tracing::debug!("3. Filtered Search:");
    tracing::debug!("```rust");
    tracing::debug!("let filters = SearchFilters::new()");
    tracing::debug!("    .with_metadata_filter(\"topic\", \"AI/ML\")");
    tracing::debug!("    .with_score_threshold(0.5);");
    tracing::debug!("let results = system.search_with_filters(query, filters).await?;");
    tracing::debug!("```");

    tracing::debug!("4. Multi-Modal Search:");
    tracing::debug!("```rust");
    tracing::debug!("// Search across text, images, and tables");
    tracing::debug!("let results = system.multimodal_search(");
    tracing::debug!("    \"show me revenue charts\",");
    tracing::debug!("    MultiModalOptions::new()");
    tracing::debug!("        .include_text(true)");
    tracing::debug!("        .include_images(true)");
    tracing::debug!("        .include_tables(true)");
    tracing::debug!(").await?;");
    tracing::debug!("```");

    Ok(())
}

async fn query_enhancement_tutorial() -> RragResult<()> {
    tracing::debug!("RRAG can automatically improve your queries:");

    tracing::debug!("1. Query Rewriting:");
    tracing::debug!("```rust");
    tracing::debug!("use rrag::query::{{QueryProcessor, QueryProcessorConfig}};");
    tracing::debug!("let processor = QueryProcessor::new(QueryProcessorConfig::default());");
    tracing::debug!("let enhanced = processor.process_query(\"What's ML?\").await?;");
    tracing::debug!("// Original: \"What's ML?\"");
    tracing::debug!("// Enhanced: [\"What is machine learning?\", \"ML algorithms\", ...]");
    tracing::debug!("```");

    tracing::debug!("2. Query Expansion:");
    tracing::debug!("Original query: \"neural networks\"");
    tracing::debug!("Expanded to include:");
    tracing::debug!("  • deep learning");
    tracing::debug!("  • artificial neural networks");
    tracing::debug!("  • multi-layer perceptrons");
    tracing::debug!("  • backpropagation");

    tracing::debug!("3. HyDE (Hypothetical Document Embeddings):");
    tracing::debug!("```rust");
    tracing::debug!("// Generate hypothetical answer and search with it");
    tracing::debug!("let hyde_generator = HyDEGenerator::new();");
    tracing::debug!(
        "let hypothetical = hyde_generator.generate(\"How does photosynthesis work?\").await?;"
    );
    tracing::debug!("let results = system.search_with_hyde(query, hypothetical).await?;");
    tracing::debug!("```");

    Ok(())
}

async fn performance_optimization_tutorial() -> RragResult<()> {
    tracing::debug!("RRAG includes several performance optimization features:");

    tracing::debug!("1. Intelligent Caching:");
    tracing::debug!("```rust");
    tracing::debug!("let cache_config = CacheConfig::default()");
    tracing::debug!("    .with_query_cache(true)");
    tracing::debug!("    .with_embedding_cache(true)");
    tracing::debug!("    .with_semantic_cache(true);");
    tracing::debug!("let system = RragSystemBuilder::new()");
    tracing::debug!("    .with_caching(cache_config)");
    tracing::debug!("    .build().await?;");
    tracing::debug!("```");

    tracing::debug!("Cache Benefits:");
    tracing::debug!("  • Query Cache: 10-100x faster for repeated queries");
    tracing::debug!("  • Embedding Cache: Skip expensive embedding computation");
    tracing::debug!("  • Semantic Cache: Hit rate for similar queries");

    tracing::debug!("2. Async Processing:");
    tracing::debug!("```rust");
    tracing::debug!("// Process multiple queries concurrently");
    tracing::debug!("let queries = vec![\"query1\", \"query2\", \"query3\"];");
    tracing::debug!("let futures: Vec<_> = queries.iter()");
    tracing::debug!("    .map(|q| system.search(q, Some(5)))");
    tracing::debug!("    .collect();");
    tracing::debug!("let results = futures::future::join_all(futures).await;");
    tracing::debug!("```");

    tracing::debug!("3. Batch Operations:");
    tracing::debug!("```rust");
    tracing::debug!("// Add multiple documents efficiently");
    tracing::debug!("system.add_documents_batch(documents).await?;");
    tracing::debug!("// Generate embeddings in batches");
    tracing::debug!("let embeddings = embedding_provider.embed_batch(texts).await?;");
    tracing::debug!("```");

    tracing::debug!("4. Memory Management:");
    tracing::debug!("  • Streaming processing for large documents");
    tracing::debug!("  • Lazy loading of embeddings");
    tracing::debug!("  • Configurable memory limits");
    tracing::debug!("  • Automatic garbage collection");

    Ok(())
}

fn production_considerations() {
    tracing::debug!("Key considerations for production deployment:");

    tracing::debug!("1. Observability:");
    tracing::debug!("```rust");
    tracing::debug!("let observability = ObservabilityConfig::production()");
    tracing::debug!("    .with_metrics(true)");
    tracing::debug!("    .with_monitoring(true)");
    tracing::debug!("    .with_alerting(true)");
    tracing::debug!("    .with_dashboard(true);");
    tracing::debug!("```");

    tracing::debug!("2. High Availability:");
    tracing::debug!("  • Multiple embedding provider fallbacks");
    tracing::debug!("  • Vector store replication");
    tracing::debug!("  • Circuit breaker patterns");
    tracing::debug!("  • Health checks and readiness probes");

    tracing::debug!("3. Scaling Strategies:");
    tracing::debug!("  • Horizontal scaling with load balancing");
    tracing::debug!("  • Vertical scaling for memory-intensive operations");
    tracing::debug!("  • Distributed caching with Redis/Memcached");
    tracing::debug!("  • GPU acceleration for embeddings");

    tracing::debug!("4. Security:");
    tracing::debug!("  • Input sanitization and validation");
    tracing::debug!("  • Rate limiting and DDoS protection");
    tracing::debug!("  • Secure embedding provider connections");
    tracing::debug!("  • Audit logging for compliance");

    tracing::debug!("5. Configuration Management:");
    tracing::debug!("```rust");
    tracing::debug!("let config = RragConfig::from_env()?");
    tracing::debug!("    .or_from_file(\"config.toml\")?");
    tracing::debug!("    .with_defaults();");
    tracing::debug!("```");
}

fn next_steps_guide() {
    tracing::debug!("Explore RRAG's advanced features:");

    tracing::debug!("🎯 Specialized Examples:");
    tracing::debug!("  cargo run --bin advanced_reranking_demo");
    tracing::debug!("  cargo run --bin graph_retrieval_demo");
    tracing::debug!("  cargo run --bin multimodal_rag_demo");
    tracing::debug!("  cargo run --bin observability_dashboard_demo");
    tracing::debug!("  cargo run --bin production_deployment_demo");

    tracing::debug!("📚 Documentation:");
    tracing::debug!("  • API Documentation: docs.rs/rrag");
    tracing::debug!("  • GitHub Repository: github.com/levalhq/rrag");
    tracing::debug!("  • Examples Directory: ./examples/");
    tracing::debug!("  • Architecture Guide: ./ARCHITECTURE.md");

    tracing::debug!("🛠️  Advanced Features to Explore:");
    tracing::debug!("  • Graph Retrieval: Knowledge graph-based search");
    tracing::debug!("  • Incremental Indexing: Real-time document updates");
    tracing::debug!("  • Multi-Modal Processing: Images, tables, charts");
    tracing::debug!("  • Advanced Reranking: Neural reranking models");
    tracing::debug!("  • Custom Evaluations: RAGAS and custom metrics");

    tracing::debug!("🤝 Community:");
    tracing::debug!("  • Join discussions on GitHub Issues");
    tracing::debug!("  • Contribute features and bug fixes");
    tracing::debug!("  • Share your use cases and success stories");
    tracing::debug!("  • Help improve documentation");

    tracing::debug!("🚀 Performance Optimization:");
    tracing::debug!("  • Profile your application with built-in tools");
    tracing::debug!("  • Experiment with different embedding providers");
    tracing::debug!("  • Tune caching and memory settings");
    tracing::debug!("  • Consider GPU acceleration for large-scale deployments");
}

// Helper functions

fn simulate_chunking(text: &str, chunk_size: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_size = 0;

    for word in words {
        current_chunk.push(word);
        current_size += word.len() + 1; // +1 for space

        if current_size >= chunk_size {
            chunks.push(current_chunk.join(" "));
            current_chunk.clear();
            current_size = 0;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join(" "));
    }

    chunks
}
