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

use rrag::prelude::*;
use rrag::caching::{
    CacheService, CacheConfig, QueryCacheConfig, EmbeddingCacheConfig,
    SemanticCacheConfig, EvictionPolicy, CacheMetrics
};
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> RragResult<()> {
    println!("🚀 RRAG Caching Performance Demo");
    println!("================================\n");

    // 1. Setup Cache System
    println!("1. Setting up intelligent caching system...");
    let mut cache = setup_cache_system().await?;
    println!("   ✓ Multi-level cache system initialized\n");

    // 2. Cache Performance Comparison
    println!("2. Cache performance comparison...");
    demo_cache_performance(&mut cache).await?;
    println!("   ✓ Performance comparison completed\n");

    // 3. Semantic Similarity Caching
    println!("3. Semantic similarity caching demo...");
    demo_semantic_caching(&mut cache).await?;
    println!("   ✓ Semantic caching demonstrated\n");

    // 4. Embedding Cache Optimization
    println!("4. Embedding cache optimization...");
    demo_embedding_caching(&mut cache).await?;
    println!("   ✓ Embedding caching optimized\n");

    // 5. Cache Analytics and Monitoring
    println!("5. Cache analytics and monitoring...");
    analyze_cache_performance(&cache).await?;
    println!("   ✓ Cache analytics completed\n");

    // 6. Memory Management Demo
    println!("6. Memory management and eviction policies...");
    demo_memory_management(&mut cache).await?;
    println!("   ✓ Memory management demonstrated\n");

    println!("🎉 Caching performance demo completed successfully!");
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
    
    println!("   - Query cache: {} entries, semantic-aware eviction", 1000);
    println!("   - Embedding cache: {} entries, LFU eviction, compression", 10000);
    println!("   - Semantic cache: {} entries, clustering enabled", 2000);
    
    Ok(cache)
}

async fn demo_cache_performance(cache: &mut CacheService) -> RragResult<()> {
    println!("   Comparing performance with and without caching:");
    
    let queries = vec![
        "What is machine learning?",
        "Explain neural networks",
        "Deep learning applications",
        "What is machine learning?", // Repeat for cache hit
        "How does AI work?",
        "Machine learning algorithms", // Similar to first query
    ];
    
    // First run - cache misses
    println!("     🔥 Cold start (cache misses):");
    let cold_start = Instant::now();
    
    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();
        
        // Check cache first
        if let Some(_cached_result) = cache.get_query_results(query).await {
            let duration = query_start.elapsed();
            println!("       Query {}: CACHE HIT - {:.1}ms", i+1, duration.as_millis());
        } else {
            // Simulate expensive operation
            let result = simulate_expensive_search(query).await?;
            
            // Cache the result
            cache.cache_query_results(
                query.to_string(), 
                create_cache_entry(query, &result)
            ).await?;
            
            let duration = query_start.elapsed();
            println!("       Query {}: CACHE MISS - {:.1}ms", i+1, duration.as_millis());
        }
    }
    
    let total_cold = cold_start.elapsed();
    println!("     Total cold start time: {:.1}ms", total_cold.as_millis());
    
    // Second run - cache hits
    println!("\n     ⚡ Warm cache (cache hits):");
    let warm_start = Instant::now();
    
    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();
        
        if let Some(_cached_result) = cache.get_query_results(query).await {
            let duration = query_start.elapsed();
            println!("       Query {}: CACHE HIT - {:.1}ms", i+1, duration.as_millis());
        } else {
            println!("       Query {}: Unexpected cache miss", i+1);
        }
    }
    
    let total_warm = warm_start.elapsed();
    println!("     Total warm cache time: {:.1}ms", total_warm.as_millis());
    
    let speedup = total_cold.as_millis() as f32 / total_warm.as_millis() as f32;
    println!("     🚀 Performance improvement: {:.1}x faster", speedup);
    
    Ok(())
}

async fn demo_semantic_caching(cache: &mut CacheService) -> RragResult<()> {
    println!("   Semantic similarity caching:");
    
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
            println!("     🎯 {} query: '{}' (similarity: {:.2})", 
                   query_type, query, similarity);
            println!("       Matched cluster: {}", 
                   semantic_result.cluster_id.unwrap_or(0));
        } else {
            println!("     🆕 {} query: '{}' (new semantic entry)", query_type, query);
            
            // Create semantic cache entry
            let semantic_entry = create_semantic_cache_entry(query);
            cache.cache_semantic_results(
                format!("ai_cluster_{}", query.len()), 
                semantic_entry
            ).await?;
        }
    }
    
    println!("     💡 Semantic caching groups similar queries for broader cache hits");
    
    Ok(())
}

async fn demo_embedding_caching(cache: &mut CacheService) -> RragResult<()> {
    println!("   Embedding caching optimization:");
    
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
            println!("     ⚡ Cached embedding for '{}' - {:.1}ms (dim: {})", 
                   text, duration.as_micros() as f32 / 1000.0, cached_embedding.len());
        } else {
            // Simulate expensive embedding computation
            let embedding = simulate_embedding_computation(text).await?;
            
            // Cache the embedding
            cache.cache_embedding(
                text.to_string(),
                model.to_string(), 
                embedding.clone()
            ).await?;
            
            let duration = embedding_start.elapsed();
            println!("     🔥 Computed embedding for '{}' - {:.1}ms (dim: {})", 
                   text, duration.as_millis(), embedding.len());
        }
    }
    
    println!("     📊 Embedding cache saves expensive computation time");
    
    Ok(())
}

async fn analyze_cache_performance(cache: &CacheService) -> RragResult<()> {
    let metrics = cache.get_metrics();
    
    println!("   📈 Cache Performance Analytics:");
    println!("     
┌─────────────────┬──────────┬───────────┬──────────┬─────────────┐
│ Cache Type      │ Hit Rate │ Entries   │ Memory   │ Avg Time    │
├─────────────────┼──────────┼───────────┼──────────┼─────────────┤");
    
    print_cache_stats("Query Cache", &metrics.query_cache);
    print_cache_stats("Embedding Cache", &metrics.embedding_cache);
    print_cache_stats("Semantic Cache", &metrics.semantic_cache);
    print_cache_stats("Result Cache", &metrics.result_cache);
    
    println!("└─────────────────┴──────────┴───────────┴──────────┴─────────────┘");
    
    // Overall performance metrics
    println!("\n     🎯 Overall Cache Efficiency:");
    println!("       • Memory saved: {:.1} MB", 
             metrics.overall.memory_saved as f32 / 1024.0 / 1024.0);
    println!("       • Time saved: {:.1} seconds", 
             metrics.overall.time_saved_ms / 1000.0);
    println!("       • Efficiency score: {:.1}%", 
             metrics.overall.efficiency_score * 100.0);
    println!("       • Operations/sec: {:.1}", 
             metrics.overall.ops_per_second);
    
    // Memory pressure analysis
    if metrics.overall.memory_pressure > 0.8 {
        println!("     ⚠️  High memory pressure detected: {:.1}%", 
                 metrics.overall.memory_pressure * 100.0);
        println!("       Consider increasing cache sizes or enabling compression");
    } else {
        println!("     ✅ Memory pressure healthy: {:.1}%", 
                 metrics.overall.memory_pressure * 100.0);
    }
    
    Ok(())
}

async fn demo_memory_management(cache: &mut CacheService) -> RragResult<()> {
    println!("   Memory management and eviction policies:");
    
    // Simulate cache filling up
    println!("     🔄 Filling caches to capacity...");
    for i in 0..100 {
        let query = format!("test query number {}", i);
        let result = simulate_expensive_search(&query).await?;
        
        cache.cache_query_results(
            query.clone(),
            create_cache_entry(&query, &result)
        ).await?;
        
        if i % 25 == 0 {
            println!("       Cached {} queries", i + 1);
        }
    }
    
    // Trigger maintenance
    println!("     🧹 Triggering cache maintenance...");
    cache.maintenance().await?;
    
    println!("     📊 Eviction policy behaviors:");
    println!("       • LRU: Removes least recently accessed items");
    println!("       • LFU: Removes least frequently used items");
    println!("       • Semantic-aware: Keeps diverse representative queries");
    println!("       • TTL: Removes expired items based on time");
    
    let metrics = cache.get_metrics();
    println!("     📈 Current eviction counts:");
    println!("       • Query cache: {} evictions", metrics.query_cache.evictions);
    println!("       • Embedding cache: {} evictions", metrics.embedding_cache.evictions);
    println!("       • Semantic cache: {} evictions", metrics.semantic_cache.evictions);
    
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
    use rrag::caching::{QueryCacheEntry, CachedSearchResult, CacheEntryMetadata};
    
    let cached_results = results.iter().enumerate().map(|(i, result)| {
        CachedSearchResult {
            document_id: format!("doc_{}", i),
            content: result.clone(),
            score: 0.9 - (i as f32 * 0.1),
            rank: i + 1,
            metadata: HashMap::new(),
        }
    }).collect();
    
    QueryCacheEntry {
        query: query.to_string(),
        embedding_hash: format!("hash_{}", query.len()),
        results: cached_results,
        generated_answer: Some(format!("Generated answer for: {}", query)),
        metadata: CacheEntryMetadata::new(),
    }
}

fn create_semantic_cache_entry(query: &str) -> rrag::caching::SemanticCacheEntry {
    use rrag::caching::{SemanticCacheEntry, SimilarEntry, CacheEntryMetadata};
    
    SemanticCacheEntry {
        representative: query.to_string(),
        cluster_id: Some(1),
        similar_entries: vec![
            SimilarEntry {
                text: query.to_string(),
                similarity: 1.0,
                added_at: std::time::SystemTime::now(),
            }
        ],
        results: vec![],
        metadata: CacheEntryMetadata::new(),
    }
}

fn calculate_similarity(query1: &str, query2: &str) -> f32 {
    // Simple similarity calculation - in production use proper embedding similarity
    let words1: std::collections::HashSet<&str> = query1.to_lowercase().split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = query2.to_lowercase().split_whitespace().collect();
    
    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();
    
    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

fn print_cache_stats(name: &str, stats: &rrag::caching::CacheStats) {
    println!("│ {:<15} │ {:>7.1}% │ {:>9} │ {:>6.1}MB │ {:>9.1}μs │",
             name,
             stats.hit_rate * 100.0,
             stats.total_entries,
             stats.memory_usage as f32 / 1024.0 / 1024.0,
             stats.avg_access_time_us);
}