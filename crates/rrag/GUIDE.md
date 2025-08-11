# RRAG Framework Guide

This comprehensive guide covers everything you need to know about building production-ready RAG applications with RRAG.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Building Your First RAG System](#building-your-first-rag-system)
5. [Advanced Features](#advanced-features)
6. [Production Deployment](#production-deployment)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Migration Guide](#migration-guide)

## Quick Start

### Installation

Add RRAG to your `Cargo.toml`:

```toml
[dependencies]
rrag = "0.1"
tokio = { version = "1.0", features = ["full"] }

# Optional features
rrag = { version = "0.1", features = ["rsllm-client", "observability", "security"] }
```

### Your First RAG Application

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // 1. Create a RAG system
    let system = RragSystemBuilder::new()
        .with_name("My RAG App")
        .with_environment("development")
        .build()
        .await?;
    
    // 2. Add documents
    let documents = vec![
        Document::new("Rust is a systems programming language focused on safety and performance."),
        Document::new("RRAG is a Rust framework for building RAG applications."),
        Document::new("Vector embeddings represent text as dense numerical vectors."),
    ];
    
    for doc in documents {
        system.process_document(doc).await?;
    }
    
    // 3. Query the system
    let response = system.search("What is Rust?".to_string(), Some(3)).await?;
    println!("Found {} results", response.total_results);
    
    Ok(())
}
```

## Core Concepts

### Documents and Chunking

RRAG processes documents through a sophisticated chunking system:

```rust
use rrag::prelude::*;

// Create a document with metadata
let document = Document::new("Large document content...")
    .with_metadata("source", "documentation".into())
    .with_metadata("category", "technical".into())
    .with_content_hash();

// Configure chunking strategy
let chunker = DocumentChunker::with_strategy(
    ChunkingStrategy::FixedSize { size: 512, overlap: 64 }
);

// Chunk the document
let chunks = chunker.chunk_document(&document)?;
println!("Created {} chunks", chunks.len());
```

### Embeddings and Vector Storage

Embeddings transform text into numerical vectors for semantic search:

```rust
use rrag::prelude::*;
use std::sync::Arc;

// Set up embedding provider
let provider = Arc::new(OpenAIEmbeddingProvider::new("your-api-key"));
let service = EmbeddingService::new(provider);

// Generate embeddings
let documents = vec![Document::new("Sample text")];
let embeddings = service.embed_documents(&documents).await?;

println!("Generated embedding with {} dimensions", embeddings[0].dimensions);
```

### Retrieval and Search

RRAG supports multiple retrieval strategies:

```rust
use rrag::prelude::*;

// Hybrid retrieval combining semantic and lexical search
let retriever = HybridRetriever::builder()
    .with_semantic_config(SemanticConfig {
        similarity_threshold: 0.8,
        max_results: 20,
        ..Default::default()
    })
    .with_bm25_config(BM25Config {
        tokenizer: TokenizerType::Simple,
        max_results: 20,
        ..Default::default()
    })
    .with_fusion_strategy(FusionStrategy::ReciprocalRankFusion)
    .build()
    .await?;

// Search with filters
let query = SearchQuery::new("machine learning")
    .with_filters(vec![("category", "technical")])
    .with_limit(10);

let results = retriever.search(query).await?;
```

## Architecture Overview

RRAG follows a modular, pipeline-based architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Processing    │───▶│   Vector Store  │
│   (Input)       │    │   Pipeline      │    │   (Embeddings)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Responses     │◀───│     Agent       │◀───│    Retriever    │
│   (Output)      │    │   (LLM+Tools)   │    │   (Search)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **Document Processing**: Chunking, metadata extraction, preprocessing
2. **Embedding Generation**: Multi-provider embedding support
3. **Vector Storage**: Efficient storage and indexing
4. **Retrieval**: Semantic and hybrid search capabilities
5. **Agent System**: LLM integration with tool calling
6. **Pipeline System**: Composable processing workflows

## Building Your First RAG System

### Step 1: Document Processing Pipeline

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create embedding service
    let provider = Arc::new(OpenAIEmbeddingProvider::new("your-api-key"));
    let embedding_service = Arc::new(EmbeddingService::new(provider));
    
    // Build processing pipeline
    let pipeline = RagPipelineBuilder::new()
        .add_step(TextPreprocessingStep::new(vec![
            TextOperation::NormalizeWhitespace,
            TextOperation::RemoveExtraWhitespace,
        ]))
        .add_step(DocumentChunkingStep::new(
            ChunkingStrategy::FixedSize { size: 512, overlap: 64 }
        ))
        .add_step(EmbeddingStep::new(embedding_service))
        .build();
    
    // Process documents
    let documents = vec![
        Document::new("Your document content here..."),
        // Add more documents
    ];
    
    let context = PipelineContext::new(PipelineData::Documents(documents));
    let result = pipeline.execute(context).await?;
    
    println!("Pipeline completed in {}ms", result.total_execution_time());
    Ok(())
}
```

### Step 2: Setting Up Retrieval

```rust
use rrag::prelude::*;

// Create a hybrid retriever for best results
let retriever = HybridRetriever::builder()
    .with_semantic_config(SemanticConfig {
        embedding_provider: "openai".to_string(),
        similarity_threshold: 0.7,
        max_results: 20,
    })
    .with_bm25_config(BM25Config {
        tokenizer: TokenizerType::Simple,
        max_results: 20,
        boost_factor: 1.2,
    })
    .with_fusion_strategy(FusionStrategy::ReciprocalRankFusion)
    .build()
    .await?;
```

### Step 3: Agent Integration

```rust
use rrag::prelude::*;
use std::sync::Arc;

// Create an agent with tools and memory
let memory = Arc::new(ConversationBufferMemory::new(1000));

let agent = AgentBuilder::new()
    .with_name("RAG Assistant")
    .with_model("openai", "gpt-4")
    .with_system_prompt("You are a helpful assistant with access to a knowledge base.")
    .with_temperature(0.7)
    .with_memory(memory)
    .with_tool(Arc::new(Calculator::new()))
    .build()?;

// Use the agent
let response = agent.process_message(
    "What can you tell me about Rust programming?", 
    Some("user-123".to_string())
).await?;

println!("Agent: {}", response.text);
```

## Advanced Features

### Incremental Indexing

For large-scale applications, use incremental indexing:

```rust
use rrag::prelude::*;

let indexing_service = IncrementalServiceBuilder::new()
    .with_batch_size(1000)
    .with_change_detection(true)
    .with_rollback_support(true)
    .with_integrity_checking(true)
    .build()
    .await?;

// Process document updates
let changes = vec![
    DocumentChange::Added(Document::new("New content")),
    DocumentChange::Modified("doc-123".to_string(), Document::new("Updated content")),
    DocumentChange::Deleted("doc-456".to_string()),
];

indexing_service.process_changes(changes).await?;
```

### Graph-Based Retrieval

For complex relationships:

```rust
use rrag::prelude::*;

let graph_retriever = GraphRetrievalBuilder::new()
    .with_entity_extraction(true)
    .with_relationship_mapping(true)
    .with_pagerank_scoring(true)
    .with_traversal_config(TraversalConfig {
        max_depth: 3,
        min_relevance: 0.5,
        ..Default::default()
    })
    .build()
    .await?;

let results = graph_retriever.query_with_context(
    "Tell me about the relationship between Rust and WebAssembly",
    QueryContext::new().with_expansion(true)
).await?;
```

### Multi-Modal Processing

Handle different content types:

```rust
use rrag::prelude::*;

let processor = MultiModalProcessor::builder()
    .with_text_handler(TextProcessor::new())
    .with_image_handler(ImageProcessor::new())
    .with_pdf_handler(PdfProcessor::new())
    .build();

let mixed_content = vec![
    MultiModalContent::Text("Text content".to_string()),
    MultiModalContent::Image(ImageData::from_path("image.jpg")?),
    MultiModalContent::Pdf(PdfData::from_path("document.pdf")?),
];

let results = processor.process_batch(mixed_content).await?;
```

## Production Deployment

### Configuration Management

```rust
use rrag::prelude::*;

// Load configuration from environment
let config = RragSystemConfig {
    name: std::env::var("RRAG_SYSTEM_NAME").unwrap_or_else(|_| "Production RAG".to_string()),
    environment: std::env::var("ENVIRONMENT").unwrap_or_else(|_| "production".to_string()),
    components: ComponentConfigs {
        embedding: EmbeddingConfig {
            provider: std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "openai".to_string()),
            batch_size: std::env::var("EMBEDDING_BATCH_SIZE")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),
            // ... more config
        },
        // ... other components
    },
    performance: PerformanceConfig {
        max_concurrency: 50,
        request_timeout_seconds: 30,
        connection_pool_size: 20,
        cache_size: 10000,
        rate_limit_per_second: Some(1000),
    },
    monitoring: MonitoringConfig {
        enable_metrics: true,
        enable_tracing: true,
        log_level: "info".to_string(),
        metrics_endpoint: Some("http://prometheus:9090".to_string()),
        health_check_interval_seconds: 30,
    },
    features: FeatureFlags {
        enable_caching: true,
        enable_async_processing: true,
        enable_auto_retry: true,
        enable_validation: true,
        ..Default::default()
    },
};

let system = RragSystem::new(config).await?;
```

### Observability

```rust
use rrag::prelude::*;

// Set up comprehensive monitoring
let observability = ObservabilityBuilder::new()
    .with_metrics_collection(true)
    .with_distributed_tracing(true)
    .with_real_time_dashboard(true)
    .with_alert_rules(vec![
        AlertRule::new("high_error_rate")
            .condition(AlertCondition::ErrorRateAbove(0.05))
            .severity(AlertSeverity::Critical),
        AlertRule::new("slow_response_time")
            .condition(AlertCondition::ResponseTimeAbove(5000))
            .severity(AlertSeverity::Warning),
    ])
    .build()
    .await?;

// Monitor system performance
let metrics = observability.get_metrics().await?;
println!("Current RPS: {:.2}", metrics.requests_per_second);
println!("P95 latency: {:.2}ms", metrics.p95_response_time_ms);
println!("Error rate: {:.2}%", metrics.error_rate * 100.0);
```

### Security Configuration

```rust
use rrag::prelude::*;

// Configure security features
let system = RragSystemBuilder::new()
    .with_security_config(SecurityConfig {
        enable_authentication: true,
        auth_provider: AuthProvider::JWT {
            secret_key_env: "JWT_SECRET".to_string(),
            token_expiry_hours: 24,
        },
        enable_authorization: true,
        rbac_config: RBACConfig {
            roles: vec![
                Role::new("admin").with_permissions(vec!["read", "write", "admin"]),
                Role::new("user").with_permissions(vec!["read"]),
            ],
        },
        rate_limiting: RateLimitConfig {
            requests_per_minute: 60,
            burst_size: 10,
        },
        input_validation: ValidationConfig {
            max_input_length: 10000,
            sanitize_html: true,
            validate_schemas: true,
        },
    })
    .build()
    .await?;
```

## Performance Optimization

### Caching Strategies

```rust
use rrag::prelude::*;

// Multi-level caching
let caching = CachingBuilder::new()
    .with_embedding_cache(EmbeddingCacheConfig {
        max_size: 10000,
        ttl_seconds: 3600,
        persistence: true,
    })
    .with_search_cache(SearchCacheConfig {
        max_size: 5000,
        ttl_seconds: 1800,
        cache_strategy: CacheStrategy::LRU,
    })
    .with_llm_cache(LLMCacheConfig {
        max_size: 1000,
        ttl_seconds: 600,
        semantic_similarity_threshold: 0.95,
    })
    .build();
```

### Batch Processing

```rust
use rrag::prelude::*;

// Optimize for high-throughput processing
let batch_config = BatchConfig {
    batch_size: 100,
    max_wait_time_ms: 1000,
    parallel_batches: 4,
    retry_strategy: RetryStrategy::ExponentialBackoff {
        max_attempts: 3,
        base_delay_ms: 1000,
    },
};

let batch_processor = BatchProcessor::new(batch_config);
```

### Connection Pooling

```rust
use rrag::prelude::*;

// Efficient resource management
let pool_config = ConnectionPoolConfig {
    max_connections: 20,
    min_connections: 5,
    connection_timeout_seconds: 30,
    idle_timeout_seconds: 300,
    health_check_interval_seconds: 60,
};
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

```rust
// Monitor memory usage
let metrics = system.get_metrics().await?;
if metrics.resource_usage.memory_usage_mb > 1000.0 {
    println!("High memory usage detected");
    
    // Clear caches
    system.clear_caches().await?;
    
    // Reduce batch sizes
    system.update_config(|config| {
        config.performance.connection_pool_size = 10;
        config.components.embedding.batch_size = 50;
    }).await?;
}
```

#### 2. Slow Query Performance

```rust
// Enable performance profiling
let profiler = Profiler::new()
    .with_detailed_timing(true)
    .with_bottleneck_analysis(true);

let profile = profiler.profile_query("test query").await?;
println!("Bottlenecks: {:?}", profile.bottlenecks);
```

#### 3. API Rate Limits

```rust
// Configure retry with backoff
let embedding_config = EmbeddingConfig {
    max_retries: 5,
    retry_delay_ms: 1000,
    backoff_multiplier: 2.0,
    jitter: true,
    ..Default::default()
};
```

### Debug Mode

```rust
use rrag::prelude::*;

// Enable verbose logging and debugging
let system = RragSystemBuilder::new()
    .with_verbose_logging(true)
    .with_debug_mode(true)
    .with_performance_tracking(true)
    .build()
    .await?;
```

### Health Checks

```rust
// Comprehensive health monitoring
let health = system.health_check().await?;

match health.overall_status {
    HealthStatus::Healthy => println!("All systems operational"),
    HealthStatus::Degraded => {
        println!("Some components are degraded:");
        for (component, status) in &health.component_status {
            if *status != HealthStatus::Healthy {
                println!("  {}: {:?}", component, status);
            }
        }
    }
    HealthStatus::Unhealthy => {
        println!("System is unhealthy - immediate attention required");
        // Implement alerting logic
    }
}
```

## Migration Guide

### From Other RAG Frameworks

#### LangChain Migration

```rust
// LangChain equivalent
// from langchain import OpenAI, VectorStore
// llm = OpenAI(api_key="...")
// vectorstore = VectorStore()

// RRAG equivalent
use rrag::prelude::*;
use std::sync::Arc;

let provider = Arc::new(OpenAIEmbeddingProvider::new("your-api-key"));
let embedding_service = EmbeddingService::new(provider);
let storage = Arc::new(InMemoryStorage::new());

let agent = AgentBuilder::new()
    .with_model("openai", "gpt-4")
    .build()?;
```

#### Haystack Migration

```rust
// Haystack DocumentStore equivalent
// document_store = InMemoryDocumentStore()
// retriever = EmbeddingRetriever(document_store=document_store)

// RRAG equivalent
let storage = Arc::new(InMemoryStorage::new());
let retriever = InMemoryRetriever::new()
    .with_storage(storage)
    .with_similarity_threshold(0.8);
```

### Version Upgrades

#### Upgrading from v0.1 to v0.2

1. **Configuration Changes**:
   ```rust
   // Old (v0.1)
   let config = RragConfig::new();
   
   // New (v0.2)
   let config = RragSystemConfig::default();
   ```

2. **API Changes**:
   ```rust
   // Old
   system.query("question").await?;
   
   // New
   system.search("question".to_string(), Some(10)).await?;
   ```

3. **Feature Flag Changes**:
   ```toml
   # Old
   rrag = { version = "0.1", features = ["async"] }
   
   # New
   rrag = { version = "0.2", features = ["observability"] }
   ```

### Best Practices for Migration

1. **Gradual Migration**: Migrate one component at a time
2. **Testing**: Maintain comprehensive tests during migration
3. **Rollback Plan**: Always have a rollback strategy
4. **Performance Monitoring**: Monitor performance before and after
5. **Documentation**: Update documentation as you migrate

## Next Steps

- Explore the [API Documentation](https://docs.rs/rrag)
- Check out [Examples](https://github.com/rrag-team/rrag/tree/main/examples)
- Join our [Community Discord](https://discord.gg/rrag)
- Read the [Contributing Guide](CONTRIBUTING.md)
- Follow us on [GitHub](https://github.com/rrag-team/rrag)

For additional help, please:
- Open an issue on GitHub
- Check our FAQ
- Join community discussions
- Contact the maintainers

Happy building with RRAG! 🚀