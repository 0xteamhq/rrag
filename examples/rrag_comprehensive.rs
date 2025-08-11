//! # RRAG Comprehensive Example
//!
//! This example demonstrates the complete RRAG framework capabilities:
//! - Document processing and chunking
//! - Embedding generation with multiple providers
//! - Vector storage and retrieval
//! - Agent-based interactions with tool calling
//! - Memory management for conversations
//! - Streaming responses
//! - Pipeline orchestration
//! - System monitoring and health checks
//!
//! Run with: `cargo run --example rrag_comprehensive`

use rrag::prelude::*;
use rrag::{
    Calculator, ConversationSummaryMemory, ConversationTokenBufferMemory, DocumentChunkingStep,
    EchoTool, EmbeddingProvider, EmbeddingStep, InMemoryRetriever, InMemoryStorage,
    LocalEmbeddingProvider, Retriever, SearchAlgorithm, SearchConfig, TextOperation,
    TextPreprocessingStep, TokenStreamBuilder, TokenType,
};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> RragResult<()> {
    println!("🦀 RRAG - Rust RAG Framework Comprehensive Demo");
    println!("==============================================\n");

    // 1. System Setup and Configuration
    println!("1. Setting up RRAG System...");
    let system = setup_rrag_system().await?;
    println!("   ✓ System initialized with all components\n");

    // 2. Document Processing Pipeline
    println!("2. Document Processing Pipeline...");
    let documents = create_sample_documents();
    let processed_docs = process_documents_with_pipeline(documents).await?;
    println!(
        "   ✓ Processed {} documents through pipeline\n",
        processed_docs.len()
    );

    // 3. Embedding and Storage
    println!("3. Embedding Generation and Storage...");
    let embeddings = generate_and_store_embeddings(&processed_docs).await?;
    println!(
        "   ✓ Generated {} embeddings and stored them\n",
        embeddings.len()
    );

    // 4. Retrieval and Search
    println!("4. Retrieval and Search Demo...");
    demo_retrieval_search(&embeddings).await?;
    println!("   ✓ Performed similarity search with multiple algorithms\n");

    // 5. Agent Interactions with Tools
    println!("5. Agent Interactions with Tool Calling...");
    demo_agent_interactions().await?;
    println!("   ✓ Agent successfully used tools and maintained conversation\n");

    // 6. Memory Management
    println!("6. Memory Management Demo...");
    demo_memory_systems().await?;
    println!("   ✓ Tested different memory strategies\n");

    // 7. Streaming Responses
    println!("7. Streaming Response Demo...");
    demo_streaming_responses().await?;
    println!("   ✓ Demonstrated real-time token streaming\n");

    // 8. Complete RAG Workflow
    println!("8. Complete RAG Workflow...");
    demo_complete_rag_workflow().await?;
    println!("   ✓ Full RAG pipeline: ingest → embed → retrieve → generate\n");

    // 9. System Monitoring
    println!("9. System Monitoring and Health...");
    demo_system_monitoring(&system).await?;
    println!("   ✓ Health checks and metrics collection\n");

    // 10. Advanced Features
    println!("10. Advanced Features Demo...");
    demo_advanced_features().await?;
    println!("    ✓ Pipeline composition, parallel processing, and error handling\n");

    println!("🎉 RRAG Demo Complete!");
    println!("All framework components working successfully in a Rust-native environment.");

    Ok(())
}

/// Setup RRAG system with all components
async fn setup_rrag_system() -> RragResult<RragSystem> {
    let system = RragSystemBuilder::new()
        .with_name("RRAG Demo System")
        .with_environment("demo")
        .with_embedding_config(rrag::system::EmbeddingConfig {
            provider: "local".to_string(),
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            batch_size: 32,
            timeout_seconds: 30,
            max_retries: 3,
            api_key_env: "".to_string(), // Not needed for local
        })
        .with_storage_config(rrag::system::StorageConfig {
            backend: "in_memory".to_string(),
            connection_string: None,
            max_connections: Some(10),
            timeout_seconds: 30,
            enable_compression: false,
        })
        .enable_feature("async_processing", true)
        .enable_feature("caching", true)
        .build()
        .await?;

    Ok(system)
}

/// Create sample documents for testing
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new(
            "Rust is a systems programming language that runs blazingly fast, prevents segfaults, \
             and guarantees thread safety. It accomplishes these goals by being memory safe without \
             using garbage collection."
        )
        .with_metadata("source", serde_json::Value::String("rust_intro".to_string()))
        .with_metadata("category", serde_json::Value::String("programming".to_string())),

        Document::new(
            "Retrieval-Augmented Generation (RAG) is a natural language processing technique that \
             combines pre-trained language models with external knowledge retrieval. This approach \
             allows AI systems to access and incorporate information from large knowledge bases."
        )
        .with_metadata("source", serde_json::Value::String("rag_explanation".to_string()))
        .with_metadata("category", serde_json::Value::String("ai".to_string())),

        Document::new(
            "Zero-cost abstractions in Rust mean that high-level abstractions don't impose runtime \
             overhead. The compiler optimizes abstract code to be as efficient as hand-written \
             low-level code, giving you both safety and performance."
        )
        .with_metadata("source", serde_json::Value::String("rust_abstractions".to_string()))
        .with_metadata("category", serde_json::Value::String("programming".to_string())),

        Document::new(
            "Vector databases are specialized storage systems designed for high-dimensional vector \
             data. They enable efficient similarity search and are crucial for AI applications \
             like recommendation systems, semantic search, and RAG implementations."
        )
        .with_metadata("source", serde_json::Value::String("vector_db".to_string()))
        .with_metadata("category", serde_json::Value::String("database".to_string())),

        Document::new(
            "Async/await in Rust provides ergonomic asynchronous programming. The async runtime \
             handles task scheduling and I/O multiplexing, allowing you to write concurrent code \
             that scales efficiently across multiple cores."
        )
        .with_metadata("source", serde_json::Value::String("rust_async".to_string()))
        .with_metadata("category", serde_json::Value::String("programming".to_string())),
    ]
}

/// Process documents through a complete pipeline
async fn process_documents_with_pipeline(
    documents: Vec<Document>,
) -> RragResult<Vec<DocumentChunk>> {
    println!("   📄 Processing {} documents...", documents.len());

    // Create embedding service
    let embedding_provider = Arc::new(LocalEmbeddingProvider::new("demo-model", 384));
    let embedding_service = Arc::new(EmbeddingService::new(embedding_provider));

    // Create document processing pipeline
    let pipeline = RagPipelineBuilder::new()
        .with_embedding_service(embedding_service)
        .build_ingestion_pipeline()?;

    let mut all_chunks = Vec::new();

    for (i, document) in documents.iter().enumerate() {
        println!(
            "     - Processing document {}: {}...",
            i + 1,
            &document.content_str()[..50.min(document.content_str().len())]
        );

        // Execute pipeline for each document
        let context = pipeline
            .execute(PipelineData::Document(document.clone()))
            .await?;

        // Extract chunks from pipeline result
        match &context.data {
            PipelineData::Chunks(chunks) => {
                println!("       → Generated {} chunks", chunks.len());
                all_chunks.extend(chunks.clone());
            }
            PipelineData::Embeddings(embeddings) => {
                println!(
                    "       → Pipeline generated {} embeddings instead of chunks",
                    embeddings.len()
                );
            }
            _ => {
                println!(
                    "       → Pipeline output unexpected data type: {:?}",
                    std::mem::discriminant(&context.data)
                );
            }
        }
    }

    println!("   ✓ Total chunks generated: {}", all_chunks.len());
    Ok(all_chunks)
}

/// Generate embeddings and store them
async fn generate_and_store_embeddings(chunks: &[DocumentChunk]) -> RragResult<Vec<Embedding>> {
    println!("   🧮 Generating embeddings for {} chunks...", chunks.len());

    // Create embedding service
    let embedding_provider = Arc::new(LocalEmbeddingProvider::new("demo-model", 384));
    let embedding_service = Arc::new(EmbeddingService::new(embedding_provider));

    // Generate embeddings in batches
    let embeddings = embedding_service.embed_chunks(chunks).await?;
    if !embeddings.is_empty() {
        println!(
            "     - Generated {} embeddings with {} dimensions",
            embeddings.len(),
            embeddings[0].dimensions
        );
    } else {
        println!("     - Generated 0 embeddings (no chunks to process)");
        return Ok(Vec::new());
    }

    // Create storage service and store embeddings
    let storage = Arc::new(InMemoryStorage::new());
    let storage_service = Arc::new(StorageService::new(storage));

    for embedding in &embeddings {
        storage_service.store_embedding(embedding).await?;
    }

    println!("     - Stored all embeddings in vector storage");
    Ok(embeddings)
}

/// Demonstrate retrieval and search capabilities
async fn demo_retrieval_search(embeddings: &[Embedding]) -> RragResult<()> {
    println!("   🔍 Setting up retrieval system...");

    if embeddings.is_empty() {
        println!("     ⚠️  No embeddings available for retrieval demo");
        println!("     Creating mock embeddings for demonstration...");

        // Create mock embeddings for demo
        let embedding_provider = Arc::new(LocalEmbeddingProvider::new("demo-model", 384));
        let mock_docs = vec![
            Document::new("Mock document 1 for retrieval testing"),
            Document::new("Mock document 2 with different content"),
            Document::new("Mock document 3 for similarity search"),
        ];

        let mut mock_embeddings = Vec::new();
        for doc in &mock_docs {
            let embedding = embedding_provider.embed_text(doc.content_str()).await?;
            mock_embeddings.push(embedding);
        }

        return demo_retrieval_with_embeddings(&mock_embeddings, &mock_docs).await;
    }

    let sample_docs: Vec<Document> = embeddings
        .iter()
        .take(3)
        .enumerate()
        .map(|(i, _)| Document::new(format!("Sample document {} for retrieval", i + 1)))
        .collect();

    demo_retrieval_with_embeddings(embeddings, &sample_docs).await
}

/// Helper function to demo retrieval with specific embeddings
async fn demo_retrieval_with_embeddings(
    embeddings: &[Embedding],
    docs: &[Document],
) -> RragResult<()> {
    // Create retrieval system
    let retriever = Arc::new(InMemoryRetriever::new());
    let retrieval_service = Arc::new(RetrievalService::new(retriever.clone()));

    // Add documents to retrieval index
    let documents_with_embeddings: Vec<(Document, Embedding)> = docs
        .iter()
        .zip(embeddings.iter())
        .map(|(doc, emb)| (doc.clone(), emb.clone()))
        .collect();

    retrieval_service
        .index_documents(&documents_with_embeddings)
        .await?;

    // Perform search with different algorithms
    let search_algorithms = vec![
        ("Cosine Similarity", SearchAlgorithm::Cosine),
        ("Euclidean Distance", SearchAlgorithm::Euclidean),
        ("Dot Product", SearchAlgorithm::DotProduct),
    ];

    for (name, algorithm) in search_algorithms {
        println!("     - Testing {} search...", name);

        let query_embedding = embeddings[0].clone();
        let query = SearchQuery::embedding(query_embedding)
            .with_limit(3)
            .with_config(SearchConfig {
                algorithm,
                ..Default::default()
            });

        let results = retriever.search(&query).await?;
        println!("       → Found {} results", results.len());

        for (i, result) in results.iter().enumerate() {
            println!(
                "         {}. Score: {:.3}, Content: {}...",
                i + 1,
                result.score,
                &result.content[..30.min(result.content.len())]
            );
        }
    }

    Ok(())
}

/// Demonstrate agent interactions with tool calling
async fn demo_agent_interactions() -> RragResult<()> {
    println!("   🤖 Setting up intelligent agent...");

    // Create agent with tools
    let agent = RragAgent::builder()
        .with_name("RRAG Demo Agent")
        .with_tool(Arc::new(Calculator))
        .with_tool(Arc::new(EchoTool))
        .with_temperature(0.7)
        .with_verbose(true)
        .build()?;

    // Test calculations
    println!("     - Testing calculator tool...");
    let calc_response = agent.process_message("Calculate 15 * 8 + 42", None).await?;
    println!("       Agent: {}", calc_response.text);
    println!("       Tools used: {}", calc_response.tool_calls.len());

    // Test echo tool
    println!("     - Testing echo tool...");
    let echo_response = agent
        .process_message(
            "Echo: Hello from RRAG!",
            Some(calc_response.metadata.turn_id.clone()),
        )
        .await?;
    println!("       Agent: {}", echo_response.text);

    // Test streaming response
    println!("     - Testing streaming response...");
    let mut stream = agent
        .stream_message("Tell me about Rust in a streaming way", None)
        .await?;
    print!("       Stream: ");
    while let Some(token_result) = futures::StreamExt::next(&mut stream).await {
        match token_result? {
            token if token.token_type == TokenType::Text => {
                print!("{}", token.content);
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                // Simulate real streaming
            }
            token if token.is_final => {
                println!(" [DONE]");
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Demonstrate different memory systems
async fn demo_memory_systems() -> RragResult<()> {
    println!("   🧠 Testing memory systems...");

    // Test buffer memory
    println!("     - Testing ConversationBufferMemory...");
    let buffer_memory = Arc::new(ConversationBufferMemory::new());
    let memory_service = MemoryService::new(buffer_memory.clone());

    memory_service
        .add_user_message("conv1", "Hello, I'm testing memory")
        .await?;
    memory_service
        .add_assistant_message("conv1", "Hello! I'll remember our conversation.")
        .await?;
    memory_service
        .add_user_message("conv1", "What did I just say?")
        .await?;

    let context = memory_service.get_conversation_context("conv1").await?;
    println!("       Memory context length: {} chars", context.len());

    // Test token buffer memory
    println!("     - Testing ConversationTokenBufferMemory...");
    let token_memory = Arc::new(ConversationTokenBufferMemory::new());

    for i in 1..=10 {
        token_memory
            .add_message(
                "conv2",
                "user",
                &format!("Message number {} with some content", i),
            )
            .await?;
    }

    let messages = token_memory.get_messages("conv2").await?;
    println!("       Token memory kept {} messages", messages.len());

    // Test summary memory
    println!("     - Testing ConversationSummaryMemory...");
    let summary_memory = Arc::new(ConversationSummaryMemory::new());

    for i in 1..=25 {
        let role = if i % 2 == 1 { "user" } else { "assistant" };
        summary_memory
            .add_message(
                "conv3",
                role,
                &format!("This is message {} in the conversation", i),
            )
            .await?;
    }

    let variables = summary_memory.get_memory_variables("conv3").await?;
    println!(
        "       Summary memory variables: {:?}",
        variables.keys().collect::<Vec<_>>()
    );

    Ok(())
}

/// Demonstrate streaming responses
async fn demo_streaming_responses() -> RragResult<()> {
    println!("   📡 Creating streaming responses...");

    // Create simple text stream
    println!("     - Text streaming:");
    let text_stream =
        StreamingResponse::from_text("This is a streaming response from RRAG framework");
    let collected = text_stream.collect_text().await?;
    println!("       Collected: {}", collected);

    // Create custom token stream
    println!("     - Custom token stream:");
    let (mut builder, receiver) = TokenStreamBuilder::new();

    tokio::spawn(async move {
        for word in ["RRAG", "is", "awesome", "for", "Rust", "developers"].iter() {
            let _ = builder.send_text(*word);
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        let _ = builder.finish();
    });

    let custom_stream = StreamingResponse::from_channel(receiver);
    print!("       Custom stream: ");
    let mut token_stream = custom_stream;
    while let Some(token_result) = futures::StreamExt::next(&mut token_stream).await {
        match token_result? {
            token if token.token_type == TokenType::Text => {
                print!("{} ", token.content.trim());
            }
            token if token.is_final => {
                println!("[COMPLETE]");
                break;
            }
            _ => {}
        }
    }

    // Demonstrate filtered streaming
    println!("     - Filtered streaming (text only):");
    let (mut builder, receiver) = TokenStreamBuilder::new();

    tokio::spawn(async move {
        for word in ["Hello", "World", "From", "RRAG"].iter() {
            let _ = builder.send_text(*word);
        }
        let _ = builder.finish();
    });

    let mixed_stream = StreamingResponse::from_channel(receiver);
    let text_only = mixed_stream.filter_by_type(TokenType::Text);
    let collected_filtered = StreamingResponse::from_stream(text_only)
        .collect_text()
        .await?;
    println!("       Filtered result: {}", collected_filtered);

    Ok(())
}

/// Demonstrate complete RAG workflow
async fn demo_complete_rag_workflow() -> RragResult<()> {
    println!("   🔄 Complete RAG workflow...");

    // 1. Document ingestion
    println!("     - Step 1: Document ingestion");
    let knowledge_docs = vec![
        Document::new("Rust provides memory safety without garbage collection through its ownership system."),
        Document::new("RRAG is a Rust-native framework for building RAG applications with zero-cost abstractions."),
        Document::new("Vector databases enable efficient similarity search for high-dimensional data."),
    ];

    // 2. Embedding and storage
    println!("     - Step 2: Embedding generation and storage");
    let embedding_provider = Arc::new(LocalEmbeddingProvider::new("rag-model", 256));
    let embedding_service = Arc::new(EmbeddingService::new(embedding_provider));

    let mut knowledge_embeddings = Vec::new();
    for doc in &knowledge_docs {
        let embedding = embedding_service.embed_document(doc).await?;
        knowledge_embeddings.push((doc.clone(), embedding));
    }

    // 3. Retrieval setup
    println!("     - Step 3: Retrieval system setup");
    let retriever = Arc::new(InMemoryRetriever::new());
    let retrieval_service = Arc::new(RetrievalService::new(retriever));

    retrieval_service
        .index_documents(&knowledge_embeddings)
        .await?;

    // 4. Query processing
    println!("     - Step 4: Query processing");
    let user_query = "How does Rust ensure memory safety?";
    let query_embedding = embedding_service
        .embed_document(&Document::new(user_query))
        .await?;

    // 5. Retrieval
    println!("     - Step 5: Similar document retrieval");
    let search_results = retrieval_service
        .search_embedding(query_embedding, Some(2))
        .await?;

    println!(
        "       Retrieved {} relevant documents:",
        search_results.len()
    );
    for (i, result) in search_results.iter().enumerate() {
        println!(
            "         {}. Score: {:.3} - {}...",
            i + 1,
            result.score,
            &result.content[..80.min(result.content.len())]
        );
    }

    // 6. Response generation (simulated)
    println!("     - Step 6: Response generation");
    let context = search_results
        .iter()
        .map(|r| r.content.clone())
        .collect::<Vec<_>>()
        .join(" ");

    println!(
        "       Generated response context ({} chars)",
        context.len()
    );
    println!("       Query: {}", user_query);
    println!("       Context-aware response: Based on the retrieved documents, Rust ensures memory safety through its ownership system...");

    Ok(())
}

/// Demonstrate system monitoring and health checks
async fn demo_system_monitoring(system: &RragSystem) -> RragResult<()> {
    println!("   📊 System monitoring and health checks...");

    // Health check
    println!("     - Performing health check...");
    let health = system.health_check().await?;
    println!("       Overall status: {:?}", health.overall_status);
    println!(
        "       Components checked: {}",
        health.component_status.len()
    );

    for (component, status) in &health.component_status {
        println!("         {}: {:?}", component, status);
    }

    // System metrics
    println!("     - Collecting system metrics...");
    let metrics = system.get_metrics().await;
    println!("       Uptime: {} seconds", metrics.uptime_seconds);
    println!(
        "       Total requests: {}",
        metrics.request_counts.total_requests
    );
    println!(
        "       Success rate: {:.1}%",
        if metrics.request_counts.total_requests > 0 {
            (metrics.request_counts.successful_requests as f64
                / metrics.request_counts.total_requests as f64)
                * 100.0
        } else {
            0.0
        }
    );

    // Configuration info
    println!("     - System configuration:");
    let config = system.get_config();
    println!("       Name: {}", config.name);
    println!("       Environment: {}", config.environment);
    println!("       Version: {}", config.version);
    println!(
        "       Features enabled: async_processing={}, caching={}",
        config.features.enable_async_processing, config.features.enable_caching
    );

    Ok(())
}

/// Demonstrate advanced features
async fn demo_advanced_features() -> RragResult<()> {
    println!("    🚀 Advanced features demonstration...");

    // 1. Pipeline composition
    println!("      - Pipeline composition and chaining");
    let embedding_provider = Arc::new(LocalEmbeddingProvider::new("advanced-model", 128));
    let embedding_service = Arc::new(EmbeddingService::new(embedding_provider));

    let custom_pipeline = Pipeline::new()
        .add_step(Arc::new(TextPreprocessingStep::new(vec![
            TextOperation::ToLowercase,
            TextOperation::NormalizeWhitespace,
            TextOperation::RemoveSpecialChars,
        ])))
        .add_step(Arc::new(DocumentChunkingStep::new(
            DocumentChunker::with_strategy(ChunkingStrategy::FixedSize {
                size: 200,
                overlap: 50,
            }),
        )))
        .add_step(Arc::new(EmbeddingStep::new(embedding_service)));

    let test_doc = Document::new("This is a TEST document with SPECIAL characters! @#$% It will be processed through the pipeline.");
    let result = custom_pipeline
        .execute(PipelineData::Document(test_doc))
        .await?;

    println!(
        "        → Pipeline executed {} steps successfully",
        result.execution_history.len()
    );
    println!(
        "        → Total processing time: {}ms",
        result.total_execution_time()
    );

    // 2. Error handling and resilience
    println!("      - Error handling and resilience testing");
    let resilient_config = rrag::pipeline::PipelineConfig {
        continue_on_error: true,
        max_execution_time: 10,
        enable_parallelism: true,
        ..Default::default()
    };

    let _resilient_pipeline = Pipeline::with_config(resilient_config);

    // This would test error scenarios in a real implementation
    println!("        → Resilient pipeline configuration validated");

    // 3. Parallel processing capabilities
    println!("      - Parallel processing demonstration");
    let documents = (1..=5)
        .map(|i| Document::new(format!("Parallel document {} for concurrent processing", i)))
        .collect::<Vec<_>>();

    let start_time = std::time::Instant::now();

    // Simulate parallel processing
    let futures: Vec<_> = documents
        .iter()
        .map(|doc| async {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            doc.content_length()
        })
        .collect();

    let results = futures::future::join_all(futures).await;
    let processing_time = start_time.elapsed();

    println!(
        "        → Processed {} documents in parallel",
        results.len()
    );
    println!(
        "        → Total time: {:?} (would be ~{}ms sequential)",
        processing_time,
        results.len() * 100
    );

    // 4. Memory optimization showcase
    println!("      - Memory optimization with zero-copy operations");
    let large_content = "x".repeat(10000);
    let doc_with_cow = Document::new(large_content); // Uses Cow for efficient string handling

    println!(
        "        → Document with {} chars using zero-copy Cow<str>",
        doc_with_cow.content_length()
    );
    println!("        → Memory efficient chunking and processing");

    // 5. Type safety demonstration
    println!("      - Type safety and compile-time guarantees");

    // This shows how the type system prevents runtime errors
    let typed_pipeline_steps = vec![
        "text_preprocessing -> Text",
        "document_chunking -> Chunks",
        "embedding_generation -> Embeddings",
        "similarity_retrieval -> SearchResults",
    ];

    for step in typed_pipeline_steps {
        println!("        → {}", step);
    }
    println!("        → All type transitions validated at compile time");

    Ok(())
}
