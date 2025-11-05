//! # Hybrid Search Demo
//!
//! Demonstrates the power of hybrid search combining semantic and keyword retrieval.
//! Shows how different types of queries benefit from different retrieval strategies.

use rrag::retrieval_enhanced::semantic::SimilarityMetric;
use rrag::{
    BM25Config, Document, FusionStrategy, HybridConfig, HybridRetriever, MockEmbeddingService,
    RragResult, SemanticConfig, TokenizerType,
};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("🔍 RRAG Hybrid Search Demonstration");
    tracing::debug!("====================================\n");

    // 1. Setup hybrid retriever
    tracing::debug!("1. Setting up Hybrid Retriever...");
    let hybrid_config = HybridConfig {
        bm25_config: BM25Config {
            k1: 1.5,
            b: 0.75,
            tokenizer: TokenizerType::Standard,
            min_token_length: 2,
            max_token_length: 50,
            use_stemming: true,
            remove_stopwords: true,
            custom_stopwords: None,
        },
        semantic_config: SemanticConfig {
            similarity_metric: SimilarityMetric::Cosine,
            embedding_dimension: 768,
            normalize_embeddings: true,
            ..Default::default()
        },
        fusion_strategy: FusionStrategy::ReciprocalRankFusion,
        adaptive_weights: true,
        semantic_weight: 0.6,
        parallel_retrieval: true,
        min_confidence: 0.3,
        enable_query_analysis: true,
    };

    // Create embedding service (mock for demo)
    let embedding_service = Arc::new(MockEmbeddingService::new());

    let hybrid_retriever = HybridRetriever::new(hybrid_config, embedding_service);
    tracing::debug!("   ✓ Hybrid retriever initialized\n");

    // 2. Index diverse documents
    tracing::debug!("2. Indexing Documents...");
    let documents = vec![
        // Technical documentation
        Document::with_id("1", "The BM25 algorithm is a probabilistic ranking function used in information retrieval. It considers term frequency (TF) and inverse document frequency (IDF) to rank documents based on their relevance to a given query.")
            .with_metadata("type", "technical".into())
            .with_metadata("topic", "algorithms".into()),
        // Product description
        Document::with_id("2", "Our new smartphone features a 6.7-inch OLED display, 5G connectivity, and a powerful A15 processor. The camera system includes a 48MP main sensor with optical image stabilization.")
            .with_metadata("type", "product".into())
            .with_metadata("category", "electronics".into()),
        // News article
        Document::with_id("3", "Breaking: Scientists discover a new method for carbon capture using algae-based bioreactors. The technology could reduce atmospheric CO2 by up to 40% more efficiently than current methods.")
            .with_metadata("type", "news".into())
            .with_metadata("topic", "environment".into()),
        // Tutorial content
        Document::with_id("4", "To implement hybrid search in Rust, first create a BM25 retriever for keyword matching, then add a semantic retriever using embeddings. Combine results using reciprocal rank fusion for optimal performance.")
            .with_metadata("type", "tutorial".into())
            .with_metadata("language", "rust".into()),
        // FAQ entry
        Document::with_id("5", "Q: What is the difference between semantic search and keyword search? A: Keyword search matches exact terms, while semantic search understands meaning and context, finding conceptually similar content even without exact matches.")
            .with_metadata("type", "faq".into())
            .with_metadata("topic", "search".into()),
        // Research paper abstract
        Document::with_id("6", "Abstract: This paper presents a novel approach to multi-modal retrieval combining visual and textual features. Our method achieves state-of-the-art performance on benchmark datasets with 15% improvement in mAP.")
            .with_metadata("type", "research".into())
            .with_metadata("field", "machine_learning".into()),
        // Code documentation
        Document::with_id("7", "The HybridRetriever struct combines BM25 and semantic search. It uses adaptive weighting based on query characteristics. Call search() method with your query string and desired limit for results.")
            .with_metadata("type", "code_doc".into())
            .with_metadata("component", "retriever".into()),
        // User review
        Document::with_id("8", "Amazing product! The search functionality is incredibly fast and accurate. It finds exactly what I'm looking for, even when I don't use the exact keywords. Highly recommend!")
            .with_metadata("type", "review".into())
            .with_metadata("rating", 5.into()),
    ];

    hybrid_retriever.index_batch(documents).await?;
    tracing::debug!("   ✓ Indexed {} documents\n", 8);

    // 3. Test different query types
    tracing::debug!("3. Testing Different Query Types\n");
    tracing::debug!("{}", "─".repeat(60));

    // Test 1: Technical exact match query
    tracing::debug!("\n📝 Query 1: 'BM25 algorithm IDF'");
    tracing::debug!("   Type: Technical/Exact Match");
    tracing::debug!("   Expected: Should favor BM25 retrieval\n");

    let results = hybrid_retriever.search("BM25 algorithm IDF", 3).await?;
    print_results(&results);

    // Test 2: Conceptual/semantic query
    tracing::debug!("\n💭 Query 2: 'How does meaning-based search work?'");
    tracing::debug!("   Type: Conceptual/Semantic");
    tracing::debug!("   Expected: Should favor semantic retrieval\n");

    let results = hybrid_retriever
        .search("How does meaning-based search work?", 3)
        .await?;
    print_results(&results);

    // Test 3: Product search query
    tracing::debug!("\n🛍️ Query 3: 'phone with good camera and fast processor'");
    tracing::debug!("   Type: Product Search");
    tracing::debug!("   Expected: Should use both BM25 and semantic\n");

    let results = hybrid_retriever
        .search("phone with good camera and fast processor", 3)
        .await?;
    print_results(&results);

    // Test 4: Question answering
    tracing::debug!("\n❓ Query 4: 'What is hybrid search?'");
    tracing::debug!("   Type: Question");
    tracing::debug!("   Expected: Should boost semantic understanding\n");

    let results = hybrid_retriever.search("What is hybrid search?", 3).await?;
    print_results(&results);

    // Test 5: Code search
    tracing::debug!("\n💻 Query 5: 'Rust implementation reciprocal rank fusion'");
    tracing::debug!("   Type: Code/Technical");
    tracing::debug!("   Expected: Mixed strategy\n");

    let results = hybrid_retriever
        .search("Rust implementation reciprocal rank fusion", 3)
        .await?;
    print_results(&results);

    // Test 6: Typo/fuzzy query
    tracing::debug!("\n🔤 Query 6: 'enviornmental carbon captur technology'");
    tracing::debug!("   Type: Query with typos");
    tracing::debug!("   Expected: Semantic should compensate for typos\n");

    let results = hybrid_retriever
        .search("enviornmental carbon captur technology", 3)
        .await?;
    print_results(&results);

    // 4. Demonstrate adaptive weighting
    tracing::debug!("\n4. Adaptive Weight Learning");
    tracing::debug!("{}", "─".repeat(60));

    // Simulate user feedback
    tracing::debug!("\n📊 Recording user satisfaction for query optimization...");

    hybrid_retriever
        .record_feedback("BM25 algorithm IDF", 0.9)
        .await?;
    hybrid_retriever
        .record_feedback("How does meaning-based search work?", 0.95)
        .await?;
    hybrid_retriever
        .record_feedback("phone with good camera and fast processor", 0.85)
        .await?;

    tracing::debug!("   ✓ Feedback recorded for adaptive weight learning\n");

    // 5. Advanced search with multiple strategies
    tracing::debug!("5. Advanced Multi-Strategy Search");
    tracing::debug!("{}", "─".repeat(60));

    // Note: In a real implementation, we would use different search strategies
    // For now, we'll demonstrate with regular search calls

    tracing::debug!("\n🚀 Query: 'information retrieval'");
    tracing::debug!("   Using multiple strategies: ExactMatch, Semantic, QueryExpansion\n");

    // For now, just demonstrate regular search
    let results = hybrid_retriever.search("information retrieval", 5).await?;

    print_results(&results);

    // 6. Show statistics
    tracing::debug!("\n6. Retriever Statistics");
    tracing::debug!("{}", "─".repeat(60));

    let stats = hybrid_retriever.stats().await;
    tracing::debug!("\n📈 Hybrid Retriever Stats:");
    tracing::debug!(
        "   • BM25 Index: {} unique terms",
        stats.bm25_stats.get("unique_terms").unwrap()
    );
    tracing::debug!(
        "   • Semantic Index: {} documents",
        stats.semantic_stats.get("total_documents").unwrap()
    );
    tracing::debug!("   • Total Queries: {}", stats.total_queries);
    tracing::debug!("   • Avg Response Time: {}ms", stats.avg_response_time_ms);
    tracing::debug!("   • Fusion Strategy: {}", stats.fusion_strategy);

    tracing::debug!("\n✅ Hybrid Search Demo Complete!");
    tracing::debug!("\nKey Insights:");
    tracing::debug!("• Hybrid search combines the best of both worlds");
    tracing::debug!("• BM25 excels at exact matches and rare terms");
    tracing::debug!("• Semantic search handles concepts and typos");
    tracing::debug!("• Adaptive weighting learns from user feedback");
    tracing::debug!("• Fusion strategies merge results optimally");

    Ok(())
}

/// Helper function to print search results
fn print_results(results: &[rrag::SearchResult]) {
    for (i, result) in results.iter().enumerate() {
        tracing::debug!(
            "   {}. [Score: {:.3}] Doc #{}",
            i + 1,
            result.score,
            result.id
        );

        // Print first 100 chars of content
        let preview = if result.content.len() > 100 {
            format!("{}...", &result.content[..100])
        } else {
            result.content.clone()
        };
        tracing::debug!("      \"{}\"", preview);

        // Show metadata
        if let Some(doc_type) = result.metadata.get("type") {
            tracing::debug!("      Type: {}", doc_type);
        }
    }
}
