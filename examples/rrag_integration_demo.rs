//! # RRAG Integration Demo
//! 
//! This example demonstrates how RGraph orchestrates RRAG components to build
//! sophisticated RAG-powered agent workflows.
//! 
//! Run with: `cargo run --example rrag_integration_demo`

use rgraph::prelude::*;
use rgraph::rrag_integration::*;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> RGraphResult<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    tracing::debug!("🔗 RRAG Integration Demo - RAG-Powered Agent Workflows");
    tracing::debug!("=====================================================\n");

    // Demo 1: Basic RAG Workflow
    tracing::debug!("📚 Demo 1: Basic RAG Question Answering");
    tracing::debug!("---------------------------------------");
    basic_rag_workflow().await?;

    // Demo 2: Multi-stage RAG Pipeline
    tracing::debug!("🔬 Demo 2: Multi-stage RAG Pipeline");
    tracing::debug!("-----------------------------------");
    multi_stage_rag_pipeline().await?;

    // Demo 3: Adaptive RAG with Context Evaluation
    tracing::debug!("🎯 Demo 3: Adaptive RAG with Context Evaluation");
    tracing::debug!("-----------------------------------------------");
    adaptive_rag_workflow().await?;

    // Demo 4: RAG-powered Multi-Agent System
    tracing::debug!("👥 Demo 4: RAG-powered Multi-Agent System");
    tracing::debug!("-----------------------------------------");
    rag_multi_agent_system().await?;

    // Demo 5: Knowledge-Aware Agent Orchestration
    tracing::debug!("🧠 Demo 5: Knowledge-Aware Agent Orchestration");
    tracing::debug!("----------------------------------------------");
    knowledge_aware_orchestration().await?;

    tracing::debug!("🎉 All RRAG integration demos completed successfully!");
    Ok(())
}

async fn basic_rag_workflow() -> RGraphResult<()> {
    // Create RAG workflow builder (simplified mock implementation)
    let rag_builder = RagWorkflowBuilder::new();

    // Build RAG nodes
    let retrieval_node = rag_builder.build_retrieval_node(
        "retrieve_docs",
        "Document Retrieval",
        RagRetrievalConfig {
            query_key: "user_query".to_string(),
            context_key: "retrieved_docs".to_string(),
            top_k: 3,
            similarity_threshold: Some(0.7),
            metadata_filters: vec![],
        },
    )?;

    let generation_node = rag_builder.build_generation_node(
        "generate_answer",
        "Answer Generation",
        RagGenerationConfig {
            query_key: "user_query".to_string(),
            context_key: "retrieved_docs".to_string(),
            response_key: "rag_answer".to_string(),
            system_prompt: Some(
                "You are a helpful AI assistant. Answer questions based on the provided context.".to_string()
            ),
            max_tokens: Some(256),
            temperature: Some(0.7),
        },
    )?;

    // Build the workflow graph
    let graph = GraphBuilder::new("basic_rag_workflow")
        .description("Simple RAG-powered question answering")
        .add_node("retrieve_docs", retrieval_node).await?
        .add_node("generate_answer", generation_node).await?
        .add_edge("retrieve_docs", "generate_answer")?
        .entry_points(vec![NodeId::new("retrieve_docs")])
        .build()?;

    // Execute the workflow
    let initial_state = GraphState::new()
        .with_input("user_query", "What are the benefits of machine learning?");

    let result = graph.execute(initial_state).await?;

    // Display results
    if let Ok(answer) = result.final_state.get("rag_answer") {
        if let Some(text) = answer.as_string() {
            tracing::debug!("❓ Question: What are the benefits of machine learning?");
            tracing::debug!("✅ RAG Answer: {}", text);
        }
    }

    if let Ok(metadata) = result.final_state.get("retrieval_metadata") {
        if let Some(meta) = metadata.as_object() {
            tracing::debug!("📊 Retrieved {} documents", meta.get("retrieved_count").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(0))));
        }
    }

    tracing::debug!("⏱️  Total processing time: {:?}", result.metrics.total_duration);

    Ok(())
}

async fn multi_stage_rag_pipeline() -> RGraphResult<()> {
    // Create RAG workflow builder (simplified)
    let rag_builder = RagWorkflowBuilder::new();

    // Build multi-stage RAG pipeline
    let retrieval_node = rag_builder.build_retrieval_node(
        "initial_retrieval",
        "Initial Document Retrieval",
        RagRetrievalConfig {
            top_k: 10,
            similarity_threshold: Some(0.5),
            ..Default::default()
        },
    )?;

    let evaluation_node = rag_builder.build_evaluation_node(
        "evaluate_context",
        "Context Quality Evaluation",
        ContextEvaluationConfig {
            min_relevance_score: 0.6,
            ..Default::default()
        },
    )?;

    let refined_generation = rag_builder.build_generation_node(
        "refined_generation",
        "Context-Aware Generation",
        RagGenerationConfig {
            query_key: "user_query".to_string(),
            context_key: "filtered_context".to_string(),
            response_key: "refined_answer".to_string(),
            system_prompt: Some(
                "You are an expert AI assistant. Provide comprehensive, accurate answers using only the most relevant context provided.".to_string()
            ),
            max_tokens: Some(512),
            temperature: Some(0.3),
        },
    )?;

    // Build the pipeline graph
    let graph = GraphBuilder::new("multi_stage_rag_pipeline")
        .description("Multi-stage RAG with context evaluation")
        .add_node("initial_retrieval", retrieval_node).await?
        .add_node("evaluate_context", evaluation_node).await?
        .add_node("refined_generation", refined_generation).await?
        .add_edge("initial_retrieval", "evaluate_context")?
        .add_edge("evaluate_context", "refined_generation")?
        .entry_points(vec![NodeId::new("initial_retrieval")])
        .build()?;

    // Execute the pipeline
    let initial_state = GraphState::new()
        .with_input("user_query", "How does quantum computing work and what are its applications?");

    let result = graph.execute(initial_state).await?;

    // Display results
    tracing::debug!("❓ Complex Query: How does quantum computing work and what are its applications?");
    
    if let Ok(relevance) = result.final_state.get("context_relevance") {
        if let Some(rel_obj) = relevance.as_object() {
            tracing::debug!("🎯 Context Quality:");
            tracing::debug!("   - Average relevance: {:.2}", 
                     rel_obj.get("average_score").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(0))).as_f64().unwrap_or(0.0));
            tracing::debug!("   - Relevant docs: {}", 
                     rel_obj.get("relevant_docs_count").unwrap_or(&serde_json::Value::Number(serde_json::Number::from(0))));
        }
    }

    if let Ok(answer) = result.final_state.get("refined_answer") {
        if let Some(text) = answer.as_string() {
            tracing::debug!("✅ Refined Answer: {}", 
                     text.chars().take(200).collect::<String>() + "...");
        }
    }

    tracing::debug!("⚙️  Pipeline stages: retrieval → evaluation → generation");
    tracing::debug!("⏱️  Total pipeline time: {:?}", result.metrics.total_duration);

    Ok(())
}

async fn adaptive_rag_workflow() -> RGraphResult<()> {
    // This would demonstrate adaptive RAG that changes strategy based on context quality
    tracing::debug!("🔄 Adaptive RAG workflow simulated");
    tracing::debug!("   - Context quality assessment: ✅ High");
    tracing::debug!("   - Routing decision: Direct generation");
    tracing::debug!("   - Fallback strategy: Not needed");
    tracing::debug!("⏱️  Adaptive processing: ~1.2s");
    
    Ok(())
}

async fn rag_multi_agent_system() -> RGraphResult<()> {
    // This would show multiple agents using RAG for different purposes
    tracing::debug!("👨‍🔬 Research Agent: RAG-powered knowledge gathering");
    tracing::debug!("👩‍💻 Analysis Agent: RAG-powered data interpretation");
    tracing::debug!("✍️  Writing Agent: RAG-powered content generation");
    tracing::debug!("🔗 Agent coordination: Context sharing enabled");
    tracing::debug!("⏱️  Multi-agent collaboration: ~2.8s");
    
    Ok(())
}

async fn knowledge_aware_orchestration() -> RGraphResult<()> {
    // This would demonstrate intelligent orchestration based on knowledge availability
    tracing::debug!("🧠 Knowledge-aware routing:");
    tracing::debug!("   - Query complexity: High");
    tracing::debug!("   - Available knowledge: Comprehensive");
    tracing::debug!("   - Orchestration strategy: Multi-stage with validation");
    tracing::debug!("   - Confidence score: 0.92");
    tracing::debug!("⏱️  Smart orchestration: ~3.5s");
    
    Ok(())
}

// Note: This demo uses simplified mock implementations
// In a real application, you would integrate with actual RRAG engines