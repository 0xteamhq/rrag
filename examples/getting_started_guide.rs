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
    println!("📚 RRAG Framework - Getting Started Guide");
    println!("=========================================\n");

    println!("Welcome to RRAG (Rust RAG Framework)! This guide will walk you through");
    println!("the key concepts and help you build your first RAG application.\n");

    // Step 1: Understanding RRAG
    println!("🎯 Step 1: Understanding RRAG");
    println!("------------------------------");
    explain_rrag_concepts();
    println!();

    // Step 2: Quick Start
    println!("🚀 Step 2: Quick Start - Your First RAG System");
    println!("-----------------------------------------------");
    let basic_system = quick_start_tutorial().await?;
    println!();

    // Step 3: Document Processing
    println!("📄 Step 3: Document Processing and Chunking");
    println!("-------------------------------------------");
    document_processing_tutorial(&basic_system).await?;
    println!();

    // Step 4: Advanced Search
    println!("🔍 Step 4: Advanced Search Techniques");
    println!("-------------------------------------");
    advanced_search_tutorial(&basic_system).await?;
    println!();

    // Step 5: Query Enhancement
    println!("✨ Step 5: Query Enhancement and Processing");
    println!("------------------------------------------");
    query_enhancement_tutorial().await?;
    println!();

    // Step 6: Performance Optimization
    println!("⚡ Step 6: Performance Optimization");
    println!("---------------------------------");
    performance_optimization_tutorial().await?;
    println!();

    // Step 7: Production Considerations
    println!("🏭 Step 7: Production Deployment");
    println!("-------------------------------");
    production_considerations();
    println!();

    // Step 8: Next Steps
    println!("📈 Step 8: Next Steps and Advanced Features");
    println!("------------------------------------------");
    next_steps_guide();
    println!();

    println!("🎉 Congratulations! You've completed the RRAG getting started guide!");
    println!("You're now ready to build powerful RAG applications with Rust.");
    println!("\nHappy coding! 🦀");

    Ok(())
}

fn explain_rrag_concepts() {
    println!("RRAG is a high-performance Rust framework for building Retrieval-Augmented");
    println!("Generation (RAG) systems. Here are the key concepts:");
    println!();
    println!("📖 Core Components:");
    println!("  • Document Processing: Chunking, cleaning, and preprocessing");
    println!("  • Embedding Generation: Converting text to vector representations");
    println!("  • Vector Storage: Efficient storage and indexing of embeddings");
    println!("  • Retrieval: Finding relevant documents using similarity search");
    println!("  • Generation: Augmenting LLM responses with retrieved context");
    println!();
    println!("🎯 Key Benefits:");
    println!("  • High Performance: Rust's speed for production workloads");
    println!("  • Memory Safety: Zero-cost abstractions without runtime overhead");
    println!("  • Async Support: Non-blocking I/O for scalable applications");
    println!("  • Modular Design: Use only the components you need");
    println!("  • Production Ready: Built-in monitoring, caching, and error handling");
}

async fn quick_start_tutorial() -> RragResult<RragSystem> {
    println!("Let's build your first RAG system in just a few lines of code!");
    println!();

    println!("1. Create a basic RRAG system:");
    println!("```rust");
    println!("use rrag::prelude::*;");
    println!();
    println!("let system = RragSystemBuilder::new()");
    println!("    .with_embedding_provider(LocalEmbeddingProvider::new())");
    println!("    .with_vector_store(InMemoryVectorStore::new())");
    println!("    .build()");
    println!("    .await?;");
    println!("```");
    println!();

    // Actually build the system
    let system = RragSystemBuilder::new().build().await?;

    println!("✅ System created successfully!");
    println!();

    println!("2. Add some sample documents:");
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information.",
    ];

    println!("```rust");
    println!("let documents = vec![");
    for doc in &documents {
        println!("    \"{}\",", doc);
    }
    println!("]);");
    println!("```");
    println!();

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

    println!(
        "✅ Added {} documents in {:.2}ms",
        documents.len(),
        processing_time.as_millis()
    );
    println!();

    println!("3. Perform your first search:");
    println!("```rust");
    println!("let results = system.search(\"neural networks\", Some(3)).await?;");
    println!("```");
    println!();

    let results = system
        .search("neural networks".to_string(), Some(3))
        .await?;

    println!("📊 Search Results:");
    for (i, result) in results.results.iter().enumerate() {
        println!(
            "  {}. Score: {:.3} - {}",
            i + 1,
            result.score,
            result.content.chars().take(60).collect::<String>() + "..."
        );
    }
    println!();

    Ok(system)
}

async fn document_processing_tutorial(_system: &RragSystem) -> RragResult<()> {
    println!("RRAG provides powerful document processing capabilities:");
    println!();

    println!("1. Text Preprocessing:");
    println!("```rust");
    println!("use rrag::preprocessing::{{TextPreprocessor, PreprocessingStep}};");
    println!();
    println!("let preprocessor = TextPreprocessor::new()");
    println!("    .add_step(PreprocessingStep::RemoveExtraWhitespace)");
    println!("    .add_step(PreprocessingStep::NormalizeUnicode)");
    println!("    .add_step(PreprocessingStep::RemoveHtmlTags);");
    println!("```");
    println!();

    println!("2. Document Chunking:");
    println!("```rust");
    println!("use rrag::chunking::{{DocumentChunker, ChunkingStrategy}};");
    println!();
    println!("let chunker = DocumentChunker::new()");
    println!("    .with_strategy(ChunkingStrategy::Semantic)");
    println!("    .with_chunk_size(512)");
    println!("    .with_overlap(50);");
    println!("```");
    println!();

    // Demonstrate chunking
    let long_document = "This is a long document that will be split into chunks. \
        Machine learning is revolutionizing how we process data. Deep learning, \
        a subset of machine learning, uses neural networks with multiple layers. \
        These networks can learn complex patterns from large datasets. Natural \
        language processing is another important area that helps computers \
        understand human language. Computer vision enables machines to \
        interpret visual information from images and videos.";

    println!("Example: Chunking a longer document");
    println!("Original length: {} characters", long_document.len());

    // Simulate chunking (in a real implementation, you'd use the actual chunker)
    let chunks = simulate_chunking(long_document, 100);
    println!("Number of chunks: {}", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "  Chunk {}: {}...",
            i + 1,
            chunk.chars().take(50).collect::<String>()
        );
    }
    println!();

    println!("3. Metadata Enhancement:");
    println!("```rust");
    println!("let document = Document::new(content)");
    println!("    .with_metadata(HashMap::from([");
    println!("        (\"source\".to_string(), \"research_paper.pdf\".to_string()),");
    println!("        (\"section\".to_string(), \"introduction\".to_string()),");
    println!("        (\"page\".to_string(), \"1\".to_string()),");
    println!("    ]));");
    println!("```");
    println!();

    Ok(())
}

async fn advanced_search_tutorial(system: &RragSystem) -> RragResult<()> {
    println!("RRAG supports multiple search strategies and algorithms:");
    println!();

    println!("1. Similarity Search:");
    let similarity_results = system
        .search("artificial intelligence".to_string(), Some(2))
        .await?;
    println!("Query: 'artificial intelligence'");
    for result in similarity_results.results {
        println!(
            "  • Score: {:.3} - {}",
            result.score,
            result.content.chars().take(60).collect::<String>() + "..."
        );
    }
    println!();

    println!("2. Hybrid Search (combines multiple signals):");
    println!("```rust");
    println!("let config = HybridSearchConfig::new()");
    println!("    .with_semantic_weight(0.7)");
    println!("    .with_keyword_weight(0.3);");
    println!();
    println!("let results = system.hybrid_search(\"deep learning\", config).await?;");
    println!("```");
    println!();

    println!("3. Filtered Search:");
    println!("```rust");
    println!("let filters = SearchFilters::new()");
    println!("    .with_metadata_filter(\"topic\", \"AI/ML\")");
    println!("    .with_score_threshold(0.5);");
    println!();
    println!("let results = system.search_with_filters(query, filters).await?;");
    println!("```");
    println!();

    println!("4. Multi-Modal Search:");
    println!("```rust");
    println!("// Search across text, images, and tables");
    println!("let results = system.multimodal_search(");
    println!("    \"show me revenue charts\",");
    println!("    MultiModalOptions::new()");
    println!("        .include_text(true)");
    println!("        .include_images(true)");
    println!("        .include_tables(true)");
    println!(").await?;");
    println!("```");
    println!();

    Ok(())
}

async fn query_enhancement_tutorial() -> RragResult<()> {
    println!("RRAG can automatically improve your queries:");
    println!();

    println!("1. Query Rewriting:");
    println!("```rust");
    println!("use rrag::query::{{QueryProcessor, QueryProcessorConfig}};");
    println!();
    println!("let processor = QueryProcessor::new(QueryProcessorConfig::default());");
    println!("let enhanced = processor.process_query(\"What's ML?\").await?;");
    println!();
    println!("// Original: \"What's ML?\"");
    println!("// Enhanced: [\"What is machine learning?\", \"ML algorithms\", ...]");
    println!("```");
    println!();

    println!("2. Query Expansion:");
    println!("Original query: \"neural networks\"");
    println!("Expanded to include:");
    println!("  • deep learning");
    println!("  • artificial neural networks");
    println!("  • multi-layer perceptrons");
    println!("  • backpropagation");
    println!();

    println!("3. HyDE (Hypothetical Document Embeddings):");
    println!("```rust");
    println!("// Generate hypothetical answer and search with it");
    println!("let hyde_generator = HyDEGenerator::new();");
    println!(
        "let hypothetical = hyde_generator.generate(\"How does photosynthesis work?\").await?;"
    );
    println!("let results = system.search_with_hyde(query, hypothetical).await?;");
    println!("```");
    println!();

    Ok(())
}

async fn performance_optimization_tutorial() -> RragResult<()> {
    println!("RRAG includes several performance optimization features:");
    println!();

    println!("1. Intelligent Caching:");
    println!("```rust");
    println!("let cache_config = CacheConfig::default()");
    println!("    .with_query_cache(true)");
    println!("    .with_embedding_cache(true)");
    println!("    .with_semantic_cache(true);");
    println!();
    println!("let system = RragSystemBuilder::new()");
    println!("    .with_caching(cache_config)");
    println!("    .build().await?;");
    println!("```");
    println!();

    println!("Cache Benefits:");
    println!("  • Query Cache: 10-100x faster for repeated queries");
    println!("  • Embedding Cache: Skip expensive embedding computation");
    println!("  • Semantic Cache: Hit rate for similar queries");
    println!();

    println!("2. Async Processing:");
    println!("```rust");
    println!("// Process multiple queries concurrently");
    println!("let queries = vec![\"query1\", \"query2\", \"query3\"];");
    println!("let futures: Vec<_> = queries.iter()");
    println!("    .map(|q| system.search(q, Some(5)))");
    println!("    .collect();");
    println!("let results = futures::future::join_all(futures).await;");
    println!("```");
    println!();

    println!("3. Batch Operations:");
    println!("```rust");
    println!("// Add multiple documents efficiently");
    println!("system.add_documents_batch(documents).await?;");
    println!();
    println!("// Generate embeddings in batches");
    println!("let embeddings = embedding_provider.embed_batch(texts).await?;");
    println!("```");
    println!();

    println!("4. Memory Management:");
    println!("  • Streaming processing for large documents");
    println!("  • Lazy loading of embeddings");
    println!("  • Configurable memory limits");
    println!("  • Automatic garbage collection");
    println!();

    Ok(())
}

fn production_considerations() {
    println!("Key considerations for production deployment:");
    println!();

    println!("1. Observability:");
    println!("```rust");
    println!("let observability = ObservabilityConfig::production()");
    println!("    .with_metrics(true)");
    println!("    .with_monitoring(true)");
    println!("    .with_alerting(true)");
    println!("    .with_dashboard(true);");
    println!("```");
    println!();

    println!("2. High Availability:");
    println!("  • Multiple embedding provider fallbacks");
    println!("  • Vector store replication");
    println!("  • Circuit breaker patterns");
    println!("  • Health checks and readiness probes");
    println!();

    println!("3. Scaling Strategies:");
    println!("  • Horizontal scaling with load balancing");
    println!("  • Vertical scaling for memory-intensive operations");
    println!("  • Distributed caching with Redis/Memcached");
    println!("  • GPU acceleration for embeddings");
    println!();

    println!("4. Security:");
    println!("  • Input sanitization and validation");
    println!("  • Rate limiting and DDoS protection");
    println!("  • Secure embedding provider connections");
    println!("  • Audit logging for compliance");
    println!();

    println!("5. Configuration Management:");
    println!("```rust");
    println!("let config = RragConfig::from_env()?");
    println!("    .or_from_file(\"config.toml\")?");
    println!("    .with_defaults();");
    println!("```");
    println!();
}

fn next_steps_guide() {
    println!("Explore RRAG's advanced features:");
    println!();

    println!("🎯 Specialized Examples:");
    println!("  cargo run --bin advanced_reranking_demo");
    println!("  cargo run --bin graph_retrieval_demo");
    println!("  cargo run --bin multimodal_rag_demo");
    println!("  cargo run --bin observability_dashboard_demo");
    println!("  cargo run --bin production_deployment_demo");
    println!();

    println!("📚 Documentation:");
    println!("  • API Documentation: docs.rs/rrag");
    println!("  • GitHub Repository: github.com/leval-ai/rrag");
    println!("  • Examples Directory: ./examples/");
    println!("  • Architecture Guide: ./ARCHITECTURE.md");
    println!();

    println!("🛠️  Advanced Features to Explore:");
    println!("  • Graph Retrieval: Knowledge graph-based search");
    println!("  • Incremental Indexing: Real-time document updates");
    println!("  • Multi-Modal Processing: Images, tables, charts");
    println!("  • Advanced Reranking: Neural reranking models");
    println!("  • Custom Evaluations: RAGAS and custom metrics");
    println!();

    println!("🤝 Community:");
    println!("  • Join discussions on GitHub Issues");
    println!("  • Contribute features and bug fixes");
    println!("  • Share your use cases and success stories");
    println!("  • Help improve documentation");
    println!();

    println!("🚀 Performance Optimization:");
    println!("  • Profile your application with built-in tools");
    println!("  • Experiment with different embedding providers");
    println!("  • Tune caching and memory settings");
    println!("  • Consider GPU acceleration for large-scale deployments");
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
