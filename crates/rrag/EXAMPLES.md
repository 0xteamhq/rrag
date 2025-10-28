# RRAG Examples

This document contains comprehensive examples demonstrating various features and use cases of the RRAG framework.

## Table of Contents

1. [Basic Examples](#basic-examples)
2. [Document Processing](#document-processing)
3. [Embedding and Retrieval](#embedding-and-retrieval)
4. [Agent Systems](#agent-systems)
5. [Pipeline Processing](#pipeline-processing)
6. [Advanced Features](#advanced-features)
7. [Production Examples](#production-examples)

## Basic Examples

### Hello World RAG

The simplest possible RAG application:

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create a basic RAG system
    let system = RragSystemBuilder::new()
        .with_name("Hello RAG")
        .build()
        .await?;

    // Add some knowledge
    let doc = Document::new("Rust is a systems programming language focused on safety and performance.");
    system.process_document(doc).await?;

    // Query the system
    let response = system.search("What is Rust?".to_string(), Some(1)).await?;

    if !response.results.is_empty() {
        tracing::debug!("Found: {}", response.results[0].content);
    }

    Ok(())
}
```

### Simple Question Answering

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create agent with knowledge base
    let memory = Arc::new(ConversationBufferMemory::new(100));

    let agent = AgentBuilder::new()
        .with_name("Q&A Assistant")
        .with_model("openai", "gpt-3.5-turbo")
        .with_memory(memory)
        .with_system_prompt("You are a helpful assistant. Answer questions based on the context provided.")
        .build()?;

    // Add knowledge documents
    let knowledge = vec![
        "The capital of France is Paris.",
        "Python is a popular programming language for data science.",
        "The human heart has four chambers.",
        "Shakespeare wrote Romeo and Juliet.",
    ];

    // In a real application, you would index these documents
    // and retrieve relevant ones for the agent

    let conversation_id = "user-123";

    // Ask questions
    let questions = vec![
        "What is the capital of France?",
        "Which programming language is popular for data science?",
        "How many chambers does a human heart have?",
    ];

    for question in questions {
        let response = agent.process_message(
            question,
            Some(conversation_id.to_string())
        ).await?;

        tracing::debug!("Q: {}", question);
        tracing::debug!("A: {}\n", response.text);
    }

    Ok(())
}
```

## Document Processing

### Document Chunking Strategies

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    let document = Document::new(
        "This is a long document that needs to be chunked. \
         It contains multiple sentences and paragraphs. \
         Each chunk should be of appropriate size for embedding. \
         The chunking strategy affects retrieval quality."
    );

    // Fixed size chunking
    let fixed_chunker = DocumentChunker::with_strategy(
        ChunkingStrategy::FixedSize { size: 100, overlap: 20 }
    );
    let fixed_chunks = fixed_chunker.chunk_document(&document)?;
    tracing::debug!("Fixed chunks: {}", fixed_chunks.len());

    // Sentence-based chunking
    let sentence_chunker = DocumentChunker::with_strategy(
        ChunkingStrategy::Sentence {
            max_sentences: 3,
            overlap_sentences: 1
        }
    );
    let sentence_chunks = sentence_chunker.chunk_document(&document)?;
    tracing::debug!("Sentence chunks: {}", sentence_chunks.len());

    // Paragraph-based chunking
    let paragraph_chunker = DocumentChunker::with_strategy(
        ChunkingStrategy::Paragraph { max_paragraphs: 2 }
    );
    let paragraph_chunks = paragraph_chunker.chunk_document(&document)?;
    tracing::debug!("Paragraph chunks: {}", paragraph_chunks.len());

    Ok(())
}
```

### Metadata and Content Processing

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create documents with rich metadata
    let documents = vec![
        Document::new("Introduction to Rust programming")
            .with_metadata("category", "programming".into())
            .with_metadata("difficulty", "beginner".into())
            .with_metadata("language", "rust".into())
            .with_metadata("tags", serde_json::json!(["tutorial", "basics"]))
            .with_content_hash(),

        Document::new("Advanced Rust concurrency patterns")
            .with_metadata("category", "programming".into())
            .with_metadata("difficulty", "advanced".into())
            .with_metadata("language", "rust".into())
            .with_metadata("tags", serde_json::json!(["concurrency", "async"]))
            .with_content_hash(),

        Document::new("Python data science fundamentals")
            .with_metadata("category", "data-science".into())
            .with_metadata("difficulty", "intermediate".into())
            .with_metadata("language", "python".into())
            .with_metadata("tags", serde_json::json!(["pandas", "numpy"]))
            .with_content_hash(),
    ];

    // Process documents
    for doc in &documents {
        tracing::debug!("Document: {}", doc.id);
        tracing::debug!("Content: {}", doc.content_str());
        tracing::debug!("Category: {:?}", doc.metadata.get("category"));
        tracing::debug!("Hash: {:?}", doc.content_hash);
        tracing::debug!("Length: {} chars\n", doc.content_length());
    }

    // Check for duplicate content
    let doc1_hash = &documents[0].content_hash;
    let doc2_hash = &documents[1].content_hash;

    if doc1_hash == doc2_hash {
        tracing::debug!("Documents have identical content");
    } else {
        tracing::debug!("Documents have different content");
    }

    Ok(())
}
```

## Embedding and Retrieval

### Multiple Embedding Providers

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    // OpenAI provider
    let openai_provider = Arc::new(
        OpenAIEmbeddingProvider::new("your-openai-key")
            .with_model("text-embedding-3-small")
    );

    // Local provider
    let local_provider = Arc::new(
        LocalEmbeddingProvider::new("sentence-transformers/all-MiniLM-L6-v2", 384)
    );

    // Create services
    let openai_service = EmbeddingService::new(openai_provider);
    let local_service = EmbeddingService::new(local_provider);

    let text = "This is a sample text for embedding comparison";

    // Generate embeddings with different providers
    let openai_embedding = openai_service.embed_document(
        &Document::new(text)
    ).await?;

    let local_embedding = local_service.embed_document(
        &Document::new(text)
    ).await?;

    tracing::debug!("OpenAI embedding: {} dimensions", openai_embedding.dimensions);
    tracing::debug!("Local embedding: {} dimensions", local_embedding.dimensions);

    // Compare provider info
    let openai_info = openai_service.provider_info();
    let local_info = local_service.provider_info();

    tracing::debug!("\nOpenAI Provider:");
    tracing::debug!("  Name: {}", openai_info.name);
    tracing::debug!("  Max batch: {}", openai_info.max_batch_size);
    tracing::debug!("  Models: {:?}", openai_info.supported_models);

    tracing::debug!("\nLocal Provider:");
    tracing::debug!("  Name: {}", local_info.name);
    tracing::debug!("  Max batch: {}", local_info.max_batch_size);
    tracing::debug!("  Models: {:?}", local_info.supported_models);

    Ok(())
}
```

### Batch Embedding Processing

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    let provider = Arc::new(LocalEmbeddingProvider::new("model", 384));
    let service = EmbeddingService::with_config(
        provider,
        EmbeddingConfig {
            batch_size: 10,
            parallel_processing: true,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    );

    // Create a large number of documents
    let documents: Vec<Document> = (0..50)
        .map(|i| {
            Document::new(format!("This is document number {} with some sample content for testing batch processing efficiency", i))
                .with_metadata("index", i.into())
                .with_metadata("batch", "test-batch-1".into())
        })
        .collect();

    tracing::debug!("Processing {} documents in batches...", documents.len());

    let start = std::time::Instant::now();
    let embeddings = service.embed_documents(&documents).await?;
    let duration = start.elapsed();

    tracing::debug!("Generated {} embeddings in {:.2}s", embeddings.len(), duration.as_secs_f64());
    tracing::debug!("Average: {:.2}ms per embedding",
        duration.as_millis() as f64 / embeddings.len() as f64);

    // Verify embeddings
    for (i, embedding) in embeddings.iter().enumerate() {
        tracing::debug!("Embedding {}: {} dimensions, source: {}",
            i, embedding.dimensions, embedding.source_id);
    }

    Ok(())
}
```

### Advanced Search with Filters

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    let storage = Arc::new(InMemoryStorage::new());
    let retriever = InMemoryRetriever::new()
        .with_storage(storage)
        .with_similarity_threshold(0.6);

    // Index documents with metadata
    let documents = vec![
        Document::new("Machine learning algorithms for image classification")
            .with_metadata("category", "AI".into())
            .with_metadata("year", 2023.into())
            .with_metadata("difficulty", "advanced".into()),

        Document::new("Basic introduction to neural networks")
            .with_metadata("category", "AI".into())
            .with_metadata("year", 2022.into())
            .with_metadata("difficulty", "beginner".into()),

        Document::new("Web development with Rust and WebAssembly")
            .with_metadata("category", "Programming".into())
            .with_metadata("year", 2023.into())
            .with_metadata("difficulty", "intermediate".into()),

        Document::new("Database optimization techniques")
            .with_metadata("category", "Database".into())
            .with_metadata("year", 2021.into())
            .with_metadata("difficulty", "advanced".into()),
    ];

    for doc in &documents {
        retriever.index_document(doc).await?;
    }

    // Search with different filter combinations
    let queries = vec![
        // Basic search
        SearchQuery::new("machine learning")
            .with_limit(3),

        // Filter by category
        SearchQuery::new("programming")
            .with_filter("category", "Programming".into())
            .with_limit(3),

        // Filter by year and difficulty
        SearchQuery::new("advanced")
            .with_filter("year", 2023.into())
            .with_filter("difficulty", "advanced".into())
            .with_limit(3),

        // Multiple categories
        SearchQuery::new("algorithms")
            .with_filter("category", serde_json::json!(["AI", "Programming"]))
            .with_min_score(0.5)
            .with_limit(5),
    ];

    for (i, query) in queries.iter().enumerate() {
        tracing::debug!("\n--- Query {} ---", i + 1);
        let results = retriever.search(query.clone()).await?;

        for result in &results {
            tracing::debug!("Score: {:.3} - {}", result.score, result.content);
            if let Some(category) = result.metadata.get("category") {
                tracing::debug!("  Category: {}", category.as_str().unwrap_or("N/A"));
            }
        }

        if results.is_empty() {
            tracing::debug!("No results found");
        }
    }

    Ok(())
}
```

## Agent Systems

### Multi-Tool Agent

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create an agent with multiple tools
    let agent = AgentBuilder::new()
        .with_name("Multi-Tool Assistant")
        .with_model("openai", "gpt-4")
        .with_system_prompt(
            "You are a helpful assistant with access to various tools. \
             Use the calculator for math problems and other tools as needed."
        )
        .with_temperature(0.7)
        .with_tool(Arc::new(Calculator::new()))
        // Uncomment if HTTP feature is enabled
        // .with_tool(Arc::new(HttpTool::new()))
        .with_max_tool_calls(5)
        .build()?;

    let conversation_id = "demo-conversation";

    // Test different types of queries
    let queries = vec![
        "What is 15 multiplied by 23?",
        "Can you calculate the compound interest on $1000 at 5% for 3 years?",
        "What's the square root of 144?",
        "If I have 50 apples and give away 30%, how many do I have left?",
    ];

    for query in queries {
        tracing::debug!("\nUser: {}", query);

        let response = agent.process_message(
            query,
            Some(conversation_id.to_string())
        ).await?;

        tracing::debug!("Assistant: {}", response.text);

        // Show tool usage
        if !response.tool_calls.is_empty() {
            tracing::debug!("Tools used:");
            for tool_call in &response.tool_calls {
                tracing::debug!("  - {}: {} -> {:?}",
                    tool_call.tool_name,
                    tool_call.input,
                    tool_call.result
                );
            }
        }

        tracing::debug!("Processing time: {}ms", response.metadata.duration_ms);
    }

    Ok(())
}
```

### Conversational Agent with Memory

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create different memory types
    let buffer_memory = Arc::new(ConversationBufferMemory::new(20));
    let token_memory = Arc::new(ConversationTokenBufferMemory::new(1000));
    let summary_memory = Arc::new(ConversationSummaryMemory::new(10));

    // Create agents with different memory types
    let buffer_agent = AgentBuilder::new()
        .with_name("Buffer Memory Agent")
        .with_memory(buffer_memory)
        .build()?;

    let token_agent = AgentBuilder::new()
        .with_name("Token Memory Agent")
        .with_memory(token_memory)
        .build()?;

    let summary_agent = AgentBuilder::new()
        .with_name("Summary Memory Agent")
        .with_memory(summary_memory)
        .build()?;

    let conversation_id = "memory-test";

    // Simulate a multi-turn conversation
    let conversation = vec![
        "Hi, my name is Alice and I'm a software engineer.",
        "I work primarily with Rust and Python.",
        "I'm currently building a RAG application.",
        "What programming languages did I mention?",
        "What's my profession?",
        "What am I currently working on?",
    ];

    // Test each agent type
    let agents = vec![
        ("Buffer", &buffer_agent),
        ("Token", &token_agent),
        ("Summary", &summary_agent),
    ];

    for (agent_name, agent) in agents {
        tracing::debug!("\n=== Testing {} Memory Agent ===", agent_name);

        for message in &conversation {
            let response = agent.process_message(
                message,
                Some(format!("{}-{}", conversation_id, agent_name.to_lowercase()))
            ).await?;

            tracing::debug!("User: {}", message);
            tracing::debug!("Agent: {}", response.text);
            tracing::debug!();
        }
    }

    Ok(())
}
```

### Streaming Agent Responses

```rust
use rrag::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> RragResult<()> {
    let agent = AgentBuilder::new()
        .with_name("Streaming Assistant")
        .with_model("openai", "gpt-4")
        .with_temperature(0.8)
        .build()?;

    let queries = vec![
        "Tell me a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming in Rust.",
    ];

    for query in queries {
        tracing::debug!("\nUser: {}", query);
        tracing::debug!("Assistant: ");

        // Stream the response
        let mut stream = agent.stream_message(query, None).await?;

        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(token) => {
                    tracing::debug!("{}", token.content);
                    // Flush to show tokens immediately
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();
                }
                Err(e) => {
                    etracing::debug!("\nStreaming error: {}", e);
                    break;
                }
            }
        }

        tracing::debug!("\n");
    }

    Ok(())
}
```

## Pipeline Processing

### Document Processing Pipeline

```rust
use rrag::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create embedding service
    let provider = Arc::new(LocalEmbeddingProvider::new("model", 384));
    let embedding_service = Arc::new(EmbeddingService::new(provider));

    // Build a comprehensive processing pipeline
    let pipeline = RagPipelineBuilder::new()
        .with_config(PipelineConfig {
            enable_parallelism: true,
            max_parallel_steps: 4,
            continue_on_error: false,
            ..Default::default()
        })
        .add_step(TextPreprocessingStep::new(vec![
            TextOperation::NormalizeWhitespace,
            TextOperation::RemoveExtraWhitespace,
        ]))
        .add_step(DocumentChunkingStep::new(
            ChunkingStrategy::FixedSize { size: 300, overlap: 50 }
        ))
        .add_step(EmbeddingStep::new(embedding_service))
        .add_step(RetrievalStep::new())
        .build();

    // Prepare test documents
    let documents = vec![
        Document::new(
            "Rust is a systems programming language that runs blazingly fast, \
             prevents segfaults, and guarantees thread safety. It accomplishes \
             these goals by being memory safe without using garbage collection."
        ).with_metadata("topic", "rust".into()),

        Document::new(
            "Machine learning is a method of data analysis that automates \
             analytical model building. It is a branch of artificial intelligence \
             based on the idea that systems can learn from data."
        ).with_metadata("topic", "ml".into()),

        Document::new(
            "WebAssembly (abbreviated Wasm) is a binary instruction format for \
             a stack-based virtual machine. Wasm is designed as a portable \
             compilation target for programming languages."
        ).with_metadata("topic", "wasm".into()),
    ];

    // Execute pipeline
    let context = PipelineContext::new(PipelineData::Documents(documents))
        .with_metadata("batch_id", "demo-batch".into())
        .with_metadata("priority", "high".into());

    tracing::debug!("Starting pipeline execution...");
    let start = std::time::Instant::now();

    let result = pipeline.execute(context).await?;

    let duration = start.elapsed();
    tracing::debug!("Pipeline completed in {:.2}s", duration.as_secs_f64());
    tracing::debug!("Total processing time: {}ms", result.total_execution_time());

    // Show execution history
    tracing::debug!("\nExecution History:");
    for step in &result.execution_history {
        tracing::debug!("  {}: {}ms (success: {})",
            step.step_id,
            step.duration_ms,
            step.success
        );

        if !step.success {
            if let Some(error) = &step.error_message {
                tracing::debug!("    Error: {}", error);
            }
        }
    }

    // Show final data type
    match &result.data {
        PipelineData::Embeddings(embeddings) => {
            tracing::debug!("\nFinal output: {} embeddings", embeddings.len());
            for (i, embedding) in embeddings.iter().enumerate() {
                tracing::debug!("  Embedding {}: {} dimensions", i, embedding.dimensions);
            }
        }
        _ => tracing::debug!("Unexpected final data type"),
    }

    Ok(())
}
```

### Custom Pipeline Steps

```rust
use rrag::prelude::*;
use async_trait::async_trait;

// Custom validation step
struct ContentValidationStep {
    min_length: usize,
    max_length: usize,
    required_keywords: Vec<String>,
}

impl ContentValidationStep {
    fn new(min_length: usize, max_length: usize, keywords: Vec<String>) -> Self {
        Self {
            min_length,
            max_length,
            required_keywords: keywords,
        }
    }
}

#[async_trait]
impl PipelineStep for ContentValidationStep {
    fn name(&self) -> &str {
        "content_validation"
    }

    fn description(&self) -> &str {
        "Validates content length and required keywords"
    }

    fn input_types(&self) -> Vec<&'static str> {
        vec!["Document", "Documents"]
    }

    fn output_type(&self) -> &'static str {
        "Document|Documents"
    }

    async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
        let start_time = std::time::Instant::now();
        let step_start = chrono::Utc::now();

        let validation_result = match &context.data {
            PipelineData::Document(doc) => {
                self.validate_document(doc)?;
                Ok(())
            }
            PipelineData::Documents(docs) => {
                for doc in docs {
                    self.validate_document(doc)?;
                }
                Ok(())
            }
            _ => Err(RragError::document_processing(
                "Invalid input type for content validation"
            ))
        };

        // Record execution
        let duration_ms = start_time.elapsed().as_millis() as u64;
        let success = validation_result.is_ok();
        let error_message = validation_result.err().map(|e| e.to_string());

        context.record_step(StepExecution {
            step_id: self.name().to_string(),
            start_time: step_start,
            duration_ms,
            success,
            error_message,
            metadata: [
                ("min_length".to_string(), self.min_length.into()),
                ("max_length".to_string(), self.max_length.into()),
                ("keyword_count".to_string(), self.required_keywords.len().into()),
            ].into_iter().collect(),
        });

        validation_result?;
        Ok(context)
    }
}

impl ContentValidationStep {
    fn validate_document(&self, doc: &Document) -> RragResult<()> {
        let content_length = doc.content_length();

        // Check length constraints
        if content_length < self.min_length {
            return Err(RragError::validation(
                "content_length",
                format!("minimum {}", self.min_length),
                content_length.to_string(),
            ));
        }

        if content_length > self.max_length {
            return Err(RragError::validation(
                "content_length",
                format!("maximum {}", self.max_length),
                content_length.to_string(),
            ));
        }

        // Check required keywords
        let content_lower = doc.content_str().to_lowercase();
        for keyword in &self.required_keywords {
            if !content_lower.contains(&keyword.to_lowercase()) {
                return Err(RragError::validation(
                    "required_keywords",
                    format!("must contain '{}'", keyword),
                    "missing".to_string(),
                ));
            }
        }

        Ok(())
    }
}

// Custom enrichment step
struct MetadataEnrichmentStep;

#[async_trait]
impl PipelineStep for MetadataEnrichmentStep {
    fn name(&self) -> &str {
        "metadata_enrichment"
    }

    fn description(&self) -> &str {
        "Enriches documents with additional metadata"
    }

    fn input_types(&self) -> Vec<&'static str> {
        vec!["Document", "Documents"]
    }

    fn output_type(&self) -> &'static str {
        "Document|Documents"
    }

    async fn execute(&self, mut context: PipelineContext) -> RragResult<PipelineContext> {
        let start_time = std::time::Instant::now();

        let enriched_data = match context.data {
            PipelineData::Document(mut doc) => {
                self.enrich_document(&mut doc).await?;
                PipelineData::Document(doc)
            }
            PipelineData::Documents(mut docs) => {
                for doc in &mut docs {
                    self.enrich_document(doc).await?;
                }
                PipelineData::Documents(docs)
            }
            _ => return Err(RragError::document_processing(
                "Invalid input type for metadata enrichment"
            ))
        };

        context.data = enriched_data;

        // Record successful execution
        context.record_step(StepExecution {
            step_id: self.name().to_string(),
            start_time: chrono::Utc::now(),
            duration_ms: start_time.elapsed().as_millis() as u64,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        });

        Ok(context)
    }
}

impl MetadataEnrichmentStep {
    async fn enrich_document(&self, doc: &mut Document) -> RragResult<()> {
        // Add processing timestamp
        doc.metadata.insert(
            "processed_at".to_string(),
            chrono::Utc::now().to_rfc3339().into(),
        );

        // Add content statistics
        let word_count = doc.content_str().split_whitespace().count();
        doc.metadata.insert("word_count".to_string(), word_count.into());

        let char_count = doc.content_length();
        doc.metadata.insert("char_count".to_string(), char_count.into());

        // Analyze content type (simple heuristics)
        let content_lower = doc.content_str().to_lowercase();
        let content_type = if content_lower.contains("def ") || content_lower.contains("function") {
            "code"
        } else if content_lower.contains("research") || content_lower.contains("study") {
            "academic"
        } else if content_lower.contains("tutorial") || content_lower.contains("how to") {
            "tutorial"
        } else {
            "general"
        };

        doc.metadata.insert("inferred_type".to_string(), content_type.into());

        Ok(())
    }
}

#[tokio::main]
async fn main() -> RragResult<()> {
    // Build pipeline with custom steps
    let pipeline = RagPipelineBuilder::new()
        .add_step(ContentValidationStep::new(
            50,  // min length
            2000, // max length
            vec!["rust".to_string(), "programming".to_string()]
        ))
        .add_step(MetadataEnrichmentStep)
        .add_step(TextPreprocessingStep::new(vec![
            TextOperation::NormalizeWhitespace,
        ]))
        .build();

    // Test documents
    let documents = vec![
        Document::new(
            "This is a comprehensive guide to Rust programming. \
             Rust is a systems programming language that focuses on safety and performance. \
             It prevents common bugs like null pointer dereferences and buffer overflows."
        ),

        Document::new(
            "Short text" // This should fail validation
        ),

        Document::new(
            "A detailed tutorial on web development using various technologies. \
             This covers HTML, CSS, JavaScript, and modern frameworks for building \
             responsive and interactive web applications."
        ), // This should fail validation (no 'rust' keyword)
    ];

    let context = PipelineContext::new(PipelineData::Documents(documents))
        .with_config(PipelineConfig {
            continue_on_error: true, // Continue processing even if some docs fail
            ..Default::default()
        });

    match pipeline.execute(context).await {
        Ok(result) => {
            tracing::debug!("Pipeline completed successfully!");
            tracing::debug!("Total time: {}ms", result.total_execution_time());

            if result.has_failures() {
                tracing::debug!("\nSome steps failed:");
                for step in &result.execution_history {
                    if !step.success {
                        tracing::debug!("  {}: {}", step.step_id,
                            step.error_message.as_ref().unwrap_or(&"Unknown error".to_string()));
                    }
                }
            }

            // Show enriched documents
            if let PipelineData::Documents(docs) = &result.data {
                tracing::debug!("\nProcessed documents:");
                for (i, doc) in docs.iter().enumerate() {
                    tracing::debug!("  Document {}: {} chars", i, doc.content_length());
                    if let Some(word_count) = doc.metadata.get("word_count") {
                        tracing::debug!("    Words: {}", word_count);
                    }
                    if let Some(content_type) = doc.metadata.get("inferred_type") {
                        tracing::debug!("    Type: {}", content_type.as_str().unwrap_or("unknown"));
                    }
                }
            }
        }
        Err(e) => {
            tracing::debug!("Pipeline failed: {}", e);
        }
    }

    Ok(())
}
```

## Advanced Features

### Incremental Indexing

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Set up incremental indexing service
    let service = IncrementalServiceBuilder::new()
        .with_config(IncrementalServiceConfig {
            batch_size: 100,
            change_detection_enabled: true,
            rollback_enabled: true,
            integrity_checking_enabled: true,
            monitoring_enabled: true,
            ..Default::default()
        })
        .build()
        .await?;

    // Initial document set
    let initial_docs = vec![
        Document::new("Initial document 1").with_metadata("version", 1.into()),
        Document::new("Initial document 2").with_metadata("version", 1.into()),
        Document::new("Initial document 3").with_metadata("version", 1.into()),
    ];

    tracing::debug!("Processing initial documents...");
    let initial_context = PipelineContext::new(PipelineData::Documents(initial_docs));
    service.process_initial_batch(initial_context).await?;

    // Simulate document changes over time
    let changes = vec![
        // Add new documents
        DocumentChange::Added(
            Document::new("New document added later")
                .with_metadata("version", 2.into())
        ),

        // Modify existing documents
        DocumentChange::Modified(
            "doc-1".to_string(),
            Document::new("Updated content for document 1")
                .with_metadata("version", 2.into())
        ),

        // Delete documents
        DocumentChange::Deleted("doc-2".to_string()),

        // Bulk changes
        DocumentChange::BulkUpdate(vec![
            Document::new("Bulk update 1").with_metadata("batch", "bulk-1".into()),
            Document::new("Bulk update 2").with_metadata("batch", "bulk-1".into()),
        ]),
    ];

    tracing::debug!("\nProcessing incremental changes...");
    for (i, change) in changes.iter().enumerate() {
        tracing::debug!("  Processing change {}: {:?}", i + 1, change);

        let result = service.process_change(change.clone()).await?;
        tracing::debug!("    Result: {:?}", result);

        // Check system health after each change
        let health = service.check_integrity().await?;
        if health.overall_status != HealthStatus::Healthy {
            tracing::debug!("    Warning: System health degraded: {:?}", health);
        }
    }

    // Show final metrics
    let metrics = service.get_metrics().await?;
    tracing::debug!("\nFinal Metrics:");
    tracing::debug!("  Documents processed: {}", metrics.total_documents_processed);
    tracing::debug!("  Updates processed: {}", metrics.total_updates_processed);
    tracing::debug!("  Errors encountered: {}", metrics.total_errors);
    tracing::debug!("  Average processing time: {:.2}ms", metrics.average_processing_time_ms);

    // Demonstrate rollback capability
    tracing::debug!("\nTesting rollback...");
    let rollback_point = service.create_rollback_point("before_bulk_update").await?;

    // Make some changes we might want to rollback
    let test_change = DocumentChange::Added(
        Document::new("Test document to rollback")
    );
    service.process_change(test_change).await?;

    // Rollback to previous state
    let rollback_result = service.rollback_to_point(rollback_point).await?;
    tracing::debug!("Rollback result: {:?}", rollback_result);

    Ok(())
}
```

### Graph-Based Retrieval

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Create graph retrieval system
    let graph_retriever = GraphRetrievalBuilder::new()
        .with_config(GraphRetrievalConfig {
            enable_entity_extraction: true,
            enable_relationship_mapping: true,
            enable_pagerank_scoring: true,
            max_traversal_depth: 3,
            min_relationship_score: 0.5,
            ..Default::default()
        })
        .with_entity_extractor(EntityExtractor::new())
        .with_query_expander(QueryExpander::new())
        .build()
        .await?;

    // Add documents with rich relationships
    let documents = vec![
        Document::new(
            "Rust is a systems programming language developed by Mozilla. \
             It focuses on memory safety and performance."
        ).with_metadata("type", "programming_language".into()),

        Document::new(
            "Mozilla Firefox is a web browser developed by Mozilla Corporation. \
             It uses the Gecko rendering engine."
        ).with_metadata("type", "software".into()),

        Document::new(
            "WebAssembly (Wasm) is a binary instruction format that can run \
             in web browsers. Rust has excellent WebAssembly support."
        ).with_metadata("type", "technology".into()),

        Document::new(
            "Memory safety prevents buffer overflows and null pointer dereferences. \
             Rust achieves memory safety without garbage collection."
        ).with_metadata("type", "concept".into()),
    ];

    // Build knowledge graph
    tracing::debug!("Building knowledge graph...");
    for doc in &documents {
        graph_retriever.index_document(doc).await?;
    }

    let graph_stats = graph_retriever.get_graph_stats().await?;
    tracing::debug!("Graph built: {} nodes, {} edges",
        graph_stats.node_count, graph_stats.edge_count);

    // Perform graph-based queries
    let queries = vec![
        "What is the relationship between Rust and Mozilla?",
        "How does memory safety relate to programming languages?",
        "What technologies are connected to web browsers?",
        "Show me concepts related to performance and safety",
    ];

    for query in queries {
        tracing::debug!("\n--- Query: {} ---", query);

        let results = graph_retriever.query_with_graph_traversal(
            query,
            GraphTraversalConfig {
                max_depth: 3,
                include_relationship_paths: true,
                score_by_centrality: true,
                expand_query_entities: true,
                ..Default::default()
            }
        ).await?;

        for result in &results {
            tracing::debug!("Score: {:.3} - {}", result.score, result.content);

            // Show relationship path if available
            if let Some(path) = result.metadata.get("relationship_path") {
                tracing::debug!("  Path: {}", path.as_str().unwrap_or(""));
            }

            // Show connected entities
            if let Some(entities) = result.metadata.get("connected_entities") {
                if let Some(entity_list) = entities.as_array() {
                    let entity_names: Vec<String> = entity_list
                        .iter()
                        .filter_map(|e| e.as_str().map(|s| s.to_string()))
                        .collect();
                    tracing::debug!("  Entities: {}", entity_names.join(", "));
                }
            }
        }
    }

    // Analyze graph structure
    tracing::debug!("\n--- Graph Analysis ---");
    let central_nodes = graph_retriever.get_most_central_nodes(5).await?;
    tracing::debug!("Most central nodes:");
    for (i, node) in central_nodes.iter().enumerate() {
        tracing::debug!("  {}: {} (centrality: {:.3})",
            i + 1, node.label, node.centrality_score);
    }

    let communities = graph_retriever.detect_communities().await?;
    tracing::debug!("\nDetected communities: {}", communities.len());
    for (i, community) in communities.iter().enumerate() {
        tracing::debug!("  Community {}: {} nodes", i + 1, community.node_count);
        if let Some(primary_topic) = &community.primary_topic {
            tracing::debug!("    Primary topic: {}", primary_topic);
        }
    }

    Ok(())
}
```

### Multi-Modal Processing

```rust
use rrag::prelude::*;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Set up multi-modal processor
    let processor = MultiModalProcessor::builder()
        .with_text_handler(TextProcessor::new())
        .with_image_handler(ImageProcessor::new())
        .with_pdf_handler(PdfProcessor::new())
        .with_config(MultiModalConfig {
            max_file_size_mb: 50,
            supported_formats: vec![
                "text/plain".to_string(),
                "image/jpeg".to_string(),
                "image/png".to_string(),
                "application/pdf".to_string(),
            ],
            enable_ocr: true,
            enable_image_description: true,
            ..Default::default()
        })
        .build();

    // Process different content types
    let content_items = vec![
        MultiModalContent::Text(
            "This is plain text content about machine learning algorithms."
        ),

        MultiModalContent::Image(
            ImageContent {
                data: std::fs::read("examples/sample_chart.png")
                    .unwrap_or_else(|_| b"mock image data".to_vec()),
                format: "image/png".to_string(),
                metadata: [
                    ("description".to_string(), "Chart showing performance metrics".into()),
                    ("source".to_string(), "internal_report".into()),
                ].into_iter().collect(),
            }
        ),

        MultiModalContent::Pdf(
            PdfContent {
                data: std::fs::read("examples/technical_doc.pdf")
                    .unwrap_or_else(|_| b"mock pdf data".to_vec()),
                metadata: [
                    ("title".to_string(), "Technical Documentation".into()),
                    ("pages".to_string(), 25.into()),
                ].into_iter().collect(),
            }
        ),

        MultiModalContent::Structured(
            StructuredContent {
                data: serde_json::json!({
                    "type": "product_spec",
                    "name": "Widget X",
                    "specifications": {
                        "weight": "1.2kg",
                        "dimensions": "10x20x5cm",
                        "material": "aluminum"
                    }
                }),
                schema: "product_specification".to_string(),
            }
        ),
    ];

    tracing::debug!("Processing {} multi-modal content items...", content_items.len());

    for (i, content) in content_items.iter().enumerate() {
        tracing::debug!("\n--- Processing item {} ---", i + 1);

        let start_time = std::time::Instant::now();
        let result = processor.process_content(content.clone()).await?;
        let processing_time = start_time.elapsed();

        tracing::debug!("Content type: {}", result.content_type);
        tracing::debug!("Processing time: {:.2}s", processing_time.as_secs_f64());

        // Show extracted text
        if !result.extracted_text.is_empty() {
            let preview = if result.extracted_text.len() > 100 {
                format!("{}...", &result.extracted_text[..100])
            } else {
                result.extracted_text.clone()
            };
            tracing::debug!("Extracted text: {}", preview);
        }

        // Show detected entities
        if !result.entities.is_empty() {
            tracing::debug!("Detected entities:");
            for entity in &result.entities {
                tracing::debug!("  {}: {} (confidence: {:.2})",
                    entity.entity_type, entity.text, entity.confidence);
            }
        }

        // Show generated embeddings
        if let Some(embedding) = &result.embedding {
            tracing::debug!("Generated embedding: {} dimensions", embedding.dimensions);
        }

        // Show metadata
        if !result.metadata.is_empty() {
            tracing::debug!("Metadata:");
            for (key, value) in &result.metadata {
                tracing::debug!("  {}: {}", key, value);
            }
        }
    }

    // Demonstrate cross-modal search
    tracing::debug!("\n--- Cross-Modal Search ---");
    let query = "Show me content related to performance and technical specifications";

    let search_results = processor.cross_modal_search(
        query,
        CrossModalSearchConfig {
            include_text: true,
            include_images: true,
            include_structured: true,
            similarity_threshold: 0.7,
            max_results: 10,
        }
    ).await?;

    for result in &search_results {
        tracing::debug!("Score: {:.3} | Type: {} | Content: {}",
            result.relevance_score,
            result.content_type,
            result.summary.chars().take(80).collect::<String>() + "..."
        );
    }

    Ok(())
}
```

## Production Examples

### High-Performance RAG Service

```rust
use rrag::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Production configuration
    let config = RragSystemConfig {
        name: "Production RAG Service".to_string(),
        environment: "production".to_string(),
        version: "1.0.0".to_string(),
        components: ComponentConfigs {
            embedding: EmbeddingConfig {
                provider: "openai".to_string(),
                model: "text-embedding-3-large".to_string(),
                batch_size: 100,
                timeout_seconds: 60,
                max_retries: 3,
                api_key_env: "OPENAI_API_KEY".to_string(),
            },
            storage: StorageConfig {
                backend: "postgresql".to_string(),
                connection_string: Some("postgresql://user:pass@localhost/ragdb".to_string()),
                max_connections: Some(50),
                timeout_seconds: 30,
                enable_compression: true,
            },
            retrieval: RetrievalConfig {
                index_type: "hybrid".to_string(),
                similarity_threshold: 0.75,
                max_results: 20,
                enable_reranking: true,
                cache_results: true,
            },
            memory: MemoryConfig {
                memory_type: "redis".to_string(),
                max_messages: 1000,
                max_tokens: Some(8000),
                enable_summarization: true,
                persistence_enabled: true,
            },
            agent: AgentConfig {
                model_provider: "openai".to_string(),
                model_name: "gpt-4-turbo".to_string(),
                temperature: 0.3,
                max_tokens: 4096,
                max_tool_calls: 5,
                enable_streaming: true,
            },
        },
        performance: PerformanceConfig {
            max_concurrency: 100,
            request_timeout_seconds: 120,
            connection_pool_size: 50,
            cache_size: 50000,
            cache_ttl_seconds: 3600,
            rate_limit_per_second: Some(1000),
        },
        monitoring: MonitoringConfig {
            enable_metrics: true,
            enable_tracing: true,
            log_level: "info".to_string(),
            health_check_interval_seconds: 30,
            metrics_endpoint: Some("http://prometheus:9090".to_string()),
            tracing_endpoint: Some("http://jaeger:14268".to_string()),
        },
        features: FeatureFlags {
            enable_experimental: false,
            enable_async_processing: true,
            enable_auto_retry: true,
            enable_validation: true,
            enable_caching: true,
        },
    };

    // Initialize system
    let system = RragSystem::new(config).await?;
    system.initialize().await?;

    // Set up observability
    let observability = ObservabilityBuilder::new()
        .with_metrics_collection(true)
        .with_real_time_dashboard(true)
        .with_alert_rules(vec![
            AlertRule::new("high_error_rate")
                .condition(AlertCondition::ErrorRateAbove(0.05))
                .severity(AlertSeverity::Critical)
                .notification(AlertNotification::Email("ops@company.com".to_string())),

            AlertRule::new("high_latency")
                .condition(AlertCondition::ResponseTimeAbove(5000))
                .severity(AlertSeverity::Warning)
                .notification(AlertNotification::Slack("#alerts".to_string())),

            AlertRule::new("low_disk_space")
                .condition(AlertCondition::DiskUsageAbove(0.85))
                .severity(AlertSeverity::Warning),
        ])
        .build()
        .await?;

    // Production health monitoring
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(30)
        );

        loop {
            interval.tick().await;

            match system.health_check().await {
                Ok(health) => {
                    match health.overall_status {
                        HealthStatus::Healthy => {
                            tracing::debug!("System healthy - uptime: {}s", health.uptime_seconds);
                        }
                        HealthStatus::Degraded => {
                            tracing::debug!("System degraded:");
                            for (component, status) in &health.component_status {
                                if *status != HealthStatus::Healthy {
                                    tracing::debug!("  {}: {:?}", component, status);
                                }
                            }
                        }
                        HealthStatus::Unhealthy => {
                            etracing::debug!("CRITICAL: System unhealthy!");
                            // Trigger alerts, potentially restart services
                        }
                    }

                    // Log metrics
                    let metrics = system.get_metrics().await;
                    tracing::debug!("Metrics - RPS: {:.2}, P95: {:.2}ms, Error Rate: {:.2}%",
                        metrics.performance.requests_per_second,
                        metrics.performance.p95_response_time_ms,
                        metrics.performance.error_rate * 100.0
                    );
                }
                Err(e) => {
                    etracing::debug!("Health check failed: {}", e);
                }
            }
        }
    });

    // Simulate production workload
    let num_concurrent_requests = 50;
    let mut handles = Vec::new();

    for i in 0..num_concurrent_requests {
        let system_clone = Arc::new(system.clone());
        let handle = tokio::spawn(async move {
            let queries = vec![
                format!("Query {} about machine learning", i),
                format!("Question {} about data processing", i),
                format!("Inquiry {} about system architecture", i),
            ];

            for query in queries {
                let start = std::time::Instant::now();

                match system_clone.search(query.clone(), Some(5)).await {
                    Ok(response) => {
                        let latency = start.elapsed().as_millis();
                        tracing::debug!("Query '{}' completed in {}ms, {} results",
                            query, latency, response.results.len());
                    }
                    Err(e) => {
                        etracing::debug!("Query '{}' failed: {}", query, e);
                    }
                }

                // Small delay to simulate realistic usage
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        });

        handles.push(handle);
    }

    // Wait for all requests to complete
    futures::future::join_all(handles).await;

    // Final system metrics
    let final_metrics = system.get_metrics().await;
    tracing::debug!("\n=== Final System Metrics ===");
    tracing::debug!("Total requests: {}", final_metrics.request_counts.total_requests);
    tracing::debug!("Success rate: {:.2}%",
        (final_metrics.request_counts.successful_requests as f64 /
         final_metrics.request_counts.total_requests as f64) * 100.0);
    tracing::debug!("Average response time: {:.2}ms", final_metrics.performance.average_response_time_ms);
    tracing::debug!("Memory usage: {:.2}MB", final_metrics.resource_usage.memory_usage_mb);

    // Graceful shutdown
    tracing::debug!("\nShutting down system gracefully...");
    system.shutdown().await?;

    Ok(())
}
```

This examples file provides comprehensive demonstrations of RRAG's capabilities, from basic usage to advanced production scenarios. Each example is self-contained and includes detailed explanations of the features being demonstrated.

The examples cover:

- Basic RAG system setup
- Document processing and chunking
- Multiple embedding providers
- Advanced search and filtering
- Agent systems with tools and memory
- Pipeline processing with custom steps
- Advanced features like incremental indexing and graph retrieval
- Production-ready configurations with monitoring

Users can run these examples to understand how to use RRAG effectively in their own applications.
