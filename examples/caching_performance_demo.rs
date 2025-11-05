//! # Caching Performance Demo
//!
//! This example demonstrates RRAG's intelligent caching system:
//! - Multi-level cache architecture
//! - Semantic similarity caching
//! - Performance optimization with caching
//! - Cache analytics and monitoring
//! - Memory management and eviction policies
//!
//! Run with: `cargo run --bin caching_performance_demo`

use rrag::caching::{
    CacheConfig, CacheMetrics, CacheService, EmbeddingCacheConfig, EvictionPolicy,
    QueryCacheConfig, SemanticCacheConfig,
};
use rrag::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("🚀 RRAG Caching Performance Demo");
    tracing::debug!("================================\n");

    // 1. Setup Cache System
    tracing::debug!("1. Setting up intelligent caching system...");
    let mut cache = setup_cache_system().await?;
    tracing::debug!("   ✓ Multi-level cache system initialized\n");

    // 2. Cache Performance Comparison
    tracing::debug!("2. Cache performance comparison...");
    demo_cache_performance(&mut cache).await?;
    tracing::debug!("   ✓ Performance comparison completed\n");

    // 3. Semantic Similarity Caching
    tracing::debug!("3. Semantic similarity caching demo...");
    demo_semantic_caching(&mut cache).await?;
    tracing::debug!("   ✓ Semantic caching demonstrated\n");

    // 4. Embedding Cache Optimization
    tracing::debug!("4. Embedding cache optimization...");
    demo_embedding_caching(&mut cache).await?;
    tracing::debug!("   ✓ Embedding caching optimized\n");

    // 5. Cache Analytics and Monitoring
    tracing::debug!("5. Cache analytics and monitoring...");
    analyze_cache_performance(&cache).await?;
    tracing::debug!("   ✓ Cache analytics completed\n");

    // 6. Memory Management Demo
    tracing::debug!("6. Memory management and eviction policies...");
    demo_memory_management(&mut cache).await?;
    tracing::debug!("   ✓ Memory management demonstrated\n");

    tracing::debug!("🎉 Caching performance demo completed successfully!");
    Ok(())
}

async fn setup_cache_system() -> RragResult<CacheService> {
    let config = CacheConfig {
        enabled: true,
        query_cache: QueryCacheConfig {
            enabled: true,
            max_size: 1000,
            ttl: Duration::from_secs(3600), // 1 hour
            eviction_policy: EvictionPolicy::SemanticAware,
            similarity_threshold: 0.85,
        },
        embedding_cache: EmbeddingCacheConfig {
            enabled: true,
            max_size: 10000,
            ttl: Duration::from_secs(86400), // 24 hours
            eviction_policy: EvictionPolicy::LFU,
            compression_enabled: true,
        },
        semantic_cache: SemanticCacheConfig {
            enabled: true,
            max_size: 2000,
            ttl: Duration::from_secs(7200), // 2 hours
            similarity_threshold: 0.80,
            clustering_enabled: true,
            max_clusters: 50,
        },
        ..Default::default()
    };

    let cache = CacheService::new(config)?;

    tracing::debug!(
        "   - Query cache: {} entries, semantic-aware eviction",
        1000
    );
    tracing::debug!(
        "   - Embedding cache: {} entries, LFU eviction, compression",
        10000
    );
    tracing::debug!("   - Semantic cache: {} entries, clustering enabled", 2000);

    Ok(cache)
}

async fn demo_cache_performance(cache: &mut CacheService) -> RragResult<()> {
    tracing::debug!("   Comparing performance with and without caching:");

    let queries = vec![
        "What is machine learning?",
        "Explain neural networks",
        "Deep learning applications",
        "What is machine learning?", // Repeat for cache hit
        "How does AI work?",
        "Machine learning algorithms", // Similar to first query
    ];

    // First run - cache misses
    tracing::debug!("     🔥 Cold start (cache misses):");
    let cold_start = Instant::now();

    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();

        // Check cache first
        if let Some(_cached_result) = cache.get_query_results(query).await {
            let duration = query_start.elapsed();
            tracing::debug!(
                "       Query {}: CACHE HIT - {:.1}ms",
                i + 1,
                duration.as_millis()
            );
        } else {
            // Simulate expensive operation
            let result = simulate_expensive_search(query).await?;

            // Cache the result
            cache
                .cache_query_results(query.to_string(), create_cache_entry(query, &result))
                .await?;

            let duration = query_start.elapsed();
            tracing::debug!(
                "       Query {}: CACHE MISS - {:.1}ms",
                i + 1,
                duration.as_millis()
            );
        }
    }

    let total_cold = cold_start.elapsed();
    tracing::debug!(
        "     Total cold start time: {:.1}ms",
        total_cold.as_millis()
    );

    // Second run - cache hits
    tracing::debug!("\n     ⚡ Warm cache (cache hits):");
    let warm_start = Instant::now();

    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();

        if let Some(_cached_result) = cache.get_query_results(query).await {
            let duration = query_start.elapsed();
            tracing::debug!(
                "       Query {}: CACHE HIT - {:.1}ms",
                i + 1,
                duration.as_millis()
            );
        } else {
            tracing::debug!("       Query {}: Unexpected cache miss", i + 1);
        }
    }

    let total_warm = warm_start.elapsed();
    tracing::debug!(
        "     Total warm cache time: {:.1}ms",
        total_warm.as_millis()
    );

    let speedup = total_cold.as_millis() as f32 / total_warm.as_millis() as f32;
    tracing::debug!("     🚀 Performance improvement: {:.1}x faster", speedup);

    Ok(())
}

async fn demo_semantic_caching(cache: &mut CacheService) -> RragResult<()> {
    tracing::debug!("   Semantic similarity caching:");

    let similar_queries = vec![
        ("What is artificial intelligence?", "primary"),
        ("Explain AI concepts", "similar"),
        ("Define artificial intelligence", "similar"),
        ("How does AI work?", "similar"),
        ("AI fundamentals", "similar"),
    ];

    for (query, query_type) in similar_queries {
        if let Some(semantic_result) = cache.get_semantic_results(query).await {
            let similarity = calculate_similarity(query, &semantic_result.representative);
            tracing::debug!(
                "     🎯 {} query: '{}' (similarity: {:.2})",
                query_type, query, similarity
            );
            tracing::debug!(
                "       Matched cluster: {}",
                semantic_result.cluster_id.unwrap_or(0)
            );
        } else {
            tracing::debug!(
                "     🆕 {} query: '{}' (new semantic entry)",
                query_type, query
            );

            // Create semantic cache entry
            let semantic_entry = create_semantic_cache_entry(query);
            cache
                .cache_semantic_results(format!("ai_cluster_{}", query.len()), semantic_entry)
                .await?;
        }
    }

    tracing::debug!("     💡 Semantic caching groups similar queries for broader cache hits");

    Ok(())
}

async fn demo_embedding_caching(cache: &mut CacheService) -> RragResult<()> {
    tracing::debug!("   Embedding caching optimization:");

    let texts = vec![
        "Machine learning algorithms",
        "Deep neural networks",
        "Natural language processing",
        "Computer vision applications",
        "Machine learning algorithms", // Repeat
    ];

    let model = "sentence-transformers/all-MiniLM-L6-v2";

    for text in texts {
        let embedding_start = Instant::now();

        if let Some(cached_embedding) = cache.get_embedding(text, model).await {
            let duration = embedding_start.elapsed();
            tracing::debug!(
                "     ⚡ Cached embedding for '{}' - {:.1}ms (dim: {})",
                text,
                duration.as_micros() as f32 / 1000.0,
                cached_embedding.len()
            );
        } else {
            // Simulate expensive embedding computation
            let embedding = simulate_embedding_computation(text).await?;

            // Cache the embedding
            cache
                .cache_embedding(text.to_string(), model.to_string(), embedding.clone())
                .await?;

            let duration = embedding_start.elapsed();
            tracing::debug!(
                "     🔥 Computed embedding for '{}' - {:.1}ms (dim: {})",
                text,
                duration.as_millis(),
                embedding.len()
            );
        }
    }

    tracing::debug!("     📊 Embedding cache saves expensive computation time");

    Ok(())
}

async fn analyze_cache_performance(cache: &CacheService) -> RragResult<()> {
    let metrics = cache.get_metrics();

    tracing::debug!("   📈 Cache Performance Analytics:");
    tracing::debug!(
        "     
┌─────────────────┬──────────┬───────────┬──────────┬─────────────┐
│ Cache Type      │ Hit Rate │ Entries   │ Memory   │ Avg Time    │
├─────────────────┼──────────┼───────────┼──────────┼─────────────┤"
    );

    print_cache_stats("Query Cache", &metrics.query_cache);
    print_cache_stats("Embedding Cache", &metrics.embedding_cache);
    print_cache_stats("Semantic Cache", &metrics.semantic_cache);
    print_cache_stats("Result Cache", &metrics.result_cache);

    tracing::debug!("└─────────────────┴──────────┴───────────┴──────────┴─────────────┘");

    // Overall performance metrics
    tracing::debug!("\n     🎯 Overall Cache Efficiency:");
    tracing::debug!(
        "       • Memory saved: {:.1} MB",
        metrics.overall.memory_saved as f32 / 1024.0 / 1024.0
    );
    tracing::debug!(
        "       • Time saved: {:.1} seconds",
        metrics.overall.time_saved_ms / 1000.0
    );
    tracing::debug!(
        "       • Efficiency score: {:.1}%",
        metrics.overall.efficiency_score * 100.0
    );
    tracing::debug!(
        "       • Operations/sec: {:.1}",
        metrics.overall.ops_per_second
    );

    // Memory pressure analysis
    if metrics.overall.memory_pressure > 0.8 {
        tracing::debug!(
            "     ⚠️  High memory pressure detected: {:.1}%",
            metrics.overall.memory_pressure * 100.0
        );
        tracing::debug!("       Consider increasing cache sizes or enabling compression");
    } else {
        tracing::debug!(
            "     ✅ Memory pressure healthy: {:.1}%",
            metrics.overall.memory_pressure * 100.0
        );
    }

    Ok(())
}

async fn demo_memory_management(cache: &mut CacheService) -> RragResult<()> {
    tracing::debug!("   Memory management and eviction policies:");

    // Simulate cache filling up
    tracing::debug!("     🔄 Filling caches to capacity...");
    for i in 0..100 {
        let query = format!("test query number {}", i);
        let result = simulate_expensive_search(&query).await?;

        cache
            .cache_query_results(query.clone(), create_cache_entry(&query, &result))
            .await?;

        if i % 25 == 0 {
            tracing::debug!("       Cached {} queries", i + 1);
        }
    }

    // Trigger maintenance
    tracing::debug!("     🧹 Triggering cache maintenance...");
    cache.maintenance().await?;

    tracing::debug!("     📊 Eviction policy behaviors:");
    tracing::debug!("       • LRU: Removes least recently accessed items");
    tracing::debug!("       • LFU: Removes least frequently used items");
    tracing::debug!("       • Semantic-aware: Keeps diverse representative queries");
    tracing::debug!("       • TTL: Removes expired items based on time");

    let metrics = cache.get_metrics();
    tracing::debug!("     📈 Current eviction counts:");
    tracing::debug!(
        "       • Query cache: {} evictions",
        metrics.query_cache.evictions
    );
    tracing::debug!(
        "       • Embedding cache: {} evictions",
        metrics.embedding_cache.evictions
    );
    tracing::debug!(
        "       • Semantic cache: {} evictions",
        metrics.semantic_cache.evictions
    );

    Ok(())
}

// Helper functions

async fn simulate_expensive_search(query: &str) -> RragResult<Vec<String>> {
    // Simulate expensive search operation
    tokio::time::sleep(Duration::from_millis(100 + (query.len() as u64 * 5))).await;

    Ok(vec![
        format!("Result 1 for: {}", query),
        format!("Result 2 for: {}", query),
        format!("Result 3 for: {}", query),
    ])
}

async fn simulate_embedding_computation(text: &str) -> RragResult<Vec<f32>> {
    // Simulate expensive embedding computation
    tokio::time::sleep(Duration::from_millis(50 + (text.len() as u64 * 2))).await;

    // Generate mock embedding
    let mut embedding = Vec::with_capacity(384);
    for i in 0..384 {
        embedding.push((i as f32 + text.len() as f32) / 1000.0);
    }

    Ok(embedding)
}

fn create_cache_entry(query: &str, results: &[String]) -> rrag::caching::QueryCacheEntry {
    use rrag::caching::{CacheEntryMetadata, CachedSearchResult, QueryCacheEntry};

    let cached_results = results
        .iter()
        .enumerate()
        .map(|(i, result)| CachedSearchResult {
            document_id: format!("doc_{}", i),
            content: result.clone(),
            score: 0.9 - (i as f32 * 0.1),
            rank: i + 1,
            metadata: HashMap::new(),
        })
        .collect();

    QueryCacheEntry {
        query: query.to_string(),
        embedding_hash: format!("hash_{}", query.len()),
        results: cached_results,
        generated_answer: Some(format!("Generated answer for: {}", query)),
        metadata: CacheEntryMetadata::new(),
    }
}

fn create_semantic_cache_entry(query: &str) -> rrag::caching::SemanticCacheEntry {
    use rrag::caching::{CacheEntryMetadata, SemanticCacheEntry, SimilarEntry};

    SemanticCacheEntry {
        representative: query.to_string(),
        cluster_id: Some(1),
        similar_entries: vec![SimilarEntry {
            text: query.to_string(),
            similarity: 1.0,
            added_at: std::time::SystemTime::now(),
        }],
        results: vec![],
        metadata: CacheEntryMetadata::new(),
    }
}

fn calculate_similarity(query1: &str, query2: &str) -> f32 {
    // Simple similarity calculation - in production use proper embedding similarity
    let binding1 = query1.to_lowercase();
    let words1: std::collections::HashSet<&str> = binding1.split_whitespace().collect();
    let binding2 = query2.to_lowercase();
    let words2: std::collections::HashSet<&str> = binding2.split_whitespace().collect();

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn print_cache_stats(name: &str, stats: &rrag::caching::CacheStats) {
    tracing::debug!(
        "│ {:<15} │ {:>7.1}% │ {:>9} │ {:>6.1}MB │ {:>9.1}μs │",
        name,
        stats.hit_rate * 100.0,
        stats.total_entries,
        stats.memory_usage as f32 / 1024.0 / 1024.0,
        stats.avg_access_time_us
    );
}
