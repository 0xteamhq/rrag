//! # Graph-Based Retrieval Demo
//!
//! Comprehensive demonstration of the graph-based retrieval system including:
//! - Knowledge graph construction from documents
//! - Entity and relationship extraction
//! - Graph-based retrieval algorithms
//! - Query expansion using graph structure
//! - Integration with traditional retrieval methods

use rrag::graph_retrieval::{
    algorithms::{GraphAlgorithms, PageRankConfig, TraversalConfig},
    builder::PrintProgressCallback,
    entity::{EntityExtractionConfig, RuleBasedEntityExtractor},
    query_expansion::{ExpansionOptions, ExpansionStrategy},
    EdgeType, GraphBuildConfig, GraphConfigBuilder, GraphEdge, GraphNode, GraphRetrievalBuilder,
    KnowledgeGraph, NodeType,
};
use rrag::prelude::*;
use rrag::Retriever;
use tokio;

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("🚀 Starting Graph-Based Retrieval Demo");
    tracing::debug!("=====================================\n");

    // Create sample documents for knowledge graph construction
    let documents = create_sample_documents();
    tracing::debug!(
        "📄 Created {} sample documents about AI and technology",
        documents.len()
    );

    // Demo 1: Basic Graph Construction
    tracing::debug!("\n🔧 Demo 1: Knowledge Graph Construction");
    let graph = demo_graph_construction().await?;

    // Demo 2: Entity and Relationship Extraction
    tracing::debug!("\n🎯 Demo 2: Entity and Relationship Extraction");
    demo_entity_extraction().await?;

    // Demo 3: Graph Algorithms
    tracing::debug!("\n📊 Demo 3: Graph Algorithms");
    demo_graph_algorithms(&graph).await?;

    // Demo 4: Query Expansion
    tracing::debug!("\n🔍 Demo 4: Query Expansion using Graph Structure");
    demo_query_expansion(&graph).await?;

    // Demo 5: Graph-Based Retrieval
    tracing::debug!("\n🔎 Demo 5: Graph-Based Retrieval System");
    demo_graph_retrieval(&documents).await?;

    // Demo 6: Advanced Configuration
    tracing::debug!("\n⚙️  Demo 6: Advanced Configuration Options");
    demo_advanced_configuration().await?;

    tracing::debug!("\n✅ Graph-Based Retrieval Demo completed successfully!");
    Ok(())
}

/// Create sample documents for demonstration
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new("Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.")
            .with_metadata("title", serde_json::Value::String("Introduction to AI".to_string()))
            .with_metadata("category", serde_json::Value::String("technology".to_string())),
        Document::new("Deep learning is a subset of machine learning that uses neural networks with multiple layers. These neural networks are inspired by the structure and function of the human brain.")
            .with_metadata("title", serde_json::Value::String("Deep Learning Basics".to_string()))
            .with_metadata("category", serde_json::Value::String("technology".to_string())),
        Document::new("Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans using natural language. NLP combines computational linguistics with machine learning and deep learning models.")
            .with_metadata("title", serde_json::Value::String("NLP Overview".to_string()))
            .with_metadata("category", serde_json::Value::String("technology".to_string())),
        Document::new("OpenAI developed ChatGPT, a large language model based on the GPT (Generative Pre-trained Transformer) architecture. ChatGPT uses deep learning to generate human-like responses to text inputs.")
            .with_metadata("title", serde_json::Value::String("ChatGPT and Large Language Models".to_string()))
            .with_metadata("category", serde_json::Value::String("technology".to_string())),
        Document::new("Computer vision is a field of AI that trains computers to interpret and understand visual information from the world. It uses machine learning algorithms to identify objects, faces, and scenes in images and videos.")
            .with_metadata("title", serde_json::Value::String("Computer Vision Applications".to_string()))
            .with_metadata("category", serde_json::Value::String("technology".to_string())),
        Document::new("Tesla uses artificial intelligence in its autonomous driving systems. The company's neural networks process data from cameras and sensors to make driving decisions in real-time.")
            .with_metadata("title", serde_json::Value::String("AI in Autonomous Vehicles".to_string()))
            .with_metadata("category", serde_json::Value::String("automotive".to_string())),
    ]
}

/// Demonstrate basic knowledge graph construction
async fn demo_graph_construction() -> RragResult<KnowledgeGraph> {
    tracing::debug!("Building a knowledge graph manually...");

    let mut graph = KnowledgeGraph::new();

    // Create concept nodes
    let ai_node = GraphNode::new("Artificial Intelligence", NodeType::Concept)
        .with_attribute(
            "definition",
            serde_json::Value::String("Branch of computer science".to_string()),
        )
        .with_confidence(1.0);

    let ml_node = GraphNode::new("Machine Learning", NodeType::Concept)
        .with_attribute(
            "definition",
            serde_json::Value::String("Subset of AI".to_string()),
        )
        .with_confidence(0.9);

    let dl_node = GraphNode::new("Deep Learning", NodeType::Concept)
        .with_attribute(
            "definition",
            serde_json::Value::String("Subset of ML using neural networks".to_string()),
        )
        .with_confidence(0.9);

    let nlp_node = GraphNode::new("Natural Language Processing", NodeType::Concept)
        .with_attribute(
            "definition",
            serde_json::Value::String("AI field for language understanding".to_string()),
        )
        .with_confidence(0.8);

    let cv_node = GraphNode::new("Computer Vision", NodeType::Concept)
        .with_attribute(
            "definition",
            serde_json::Value::String("AI field for visual understanding".to_string()),
        )
        .with_confidence(0.8);

    // Create organization nodes
    let openai_node = GraphNode::new("OpenAI", NodeType::Entity("Organization".to_string()))
        .with_attribute(
            "type",
            serde_json::Value::String("AI research company".to_string()),
        )
        .with_confidence(1.0);

    let tesla_node = GraphNode::new("Tesla", NodeType::Entity("Organization".to_string()))
        .with_attribute(
            "type",
            serde_json::Value::String("Electric vehicle company".to_string()),
        )
        .with_confidence(1.0);

    // Store node IDs for edge creation
    let ai_id = ai_node.id.clone();
    let ml_id = ml_node.id.clone();
    let dl_id = dl_node.id.clone();
    let nlp_id = nlp_node.id.clone();
    let cv_id = cv_node.id.clone();
    let openai_id = openai_node.id.clone();
    let tesla_id = tesla_node.id.clone();

    // Add nodes to graph
    graph.add_node(ai_node)?;
    graph.add_node(ml_node)?;
    graph.add_node(dl_node)?;
    graph.add_node(nlp_node)?;
    graph.add_node(cv_node)?;
    graph.add_node(openai_node)?;
    graph.add_node(tesla_node)?;

    // Create hierarchical relationships
    graph.add_edge(
        GraphEdge::new(
            ml_id.clone(),
            ai_id.clone(),
            "is_part_of",
            EdgeType::Hierarchical,
        )
        .with_confidence(0.95)
        .with_weight(0.95),
    )?;

    graph.add_edge(
        GraphEdge::new(
            dl_id.clone(),
            ml_id.clone(),
            "is_part_of",
            EdgeType::Hierarchical,
        )
        .with_confidence(0.9)
        .with_weight(0.9),
    )?;

    graph.add_edge(
        GraphEdge::new(
            nlp_id.clone(),
            ai_id.clone(),
            "is_part_of",
            EdgeType::Hierarchical,
        )
        .with_confidence(0.85)
        .with_weight(0.85),
    )?;

    graph.add_edge(
        GraphEdge::new(
            cv_id.clone(),
            ai_id.clone(),
            "is_part_of",
            EdgeType::Hierarchical,
        )
        .with_confidence(0.85)
        .with_weight(0.85),
    )?;

    // Create application relationships
    graph.add_edge(
        GraphEdge::new(
            openai_id.clone(),
            nlp_id.clone(),
            "develops",
            EdgeType::Semantic("develops".to_string()),
        )
        .with_confidence(0.8)
        .with_weight(0.8),
    )?;

    graph.add_edge(
        GraphEdge::new(
            tesla_id.clone(),
            cv_id.clone(),
            "uses",
            EdgeType::Semantic("uses".to_string()),
        )
        .with_confidence(0.7)
        .with_weight(0.7),
    )?;

    // Calculate and display graph metrics
    let metrics = graph.calculate_metrics();
    tracing::debug!("Graph metrics:");
    tracing::debug!("  - Nodes: {}", metrics.node_count);
    tracing::debug!("  - Edges: {}", metrics.edge_count);
    tracing::debug!("  - Connected components: {}", metrics.connected_components);
    tracing::debug!("  - Density: {:.3}", metrics.density);
    tracing::debug!("  - Average degree: {:.2}", metrics.average_degree);

    Ok(graph)
}

/// Demonstrate entity and relationship extraction
async fn demo_entity_extraction() -> RragResult<()> {
    tracing::debug!("Extracting entities and relationships from text...");

    let config = EntityExtractionConfig::default();
    let extractor = RuleBasedEntityExtractor::new(config)?;

    let text = "John Smith, a software engineer at Google, developed a machine learning model for natural language processing. The model achieved 95% accuracy on Stanford's benchmark dataset.";

    let (entities, relationships) = extractor.extract_all(text, "demo_doc").await?;

    tracing::debug!("Extracted {} entities:", entities.len());
    for entity in &entities {
        tracing::debug!(
            "  - '{}' (type: {:?}, confidence: {:.2})",
            entity.text, entity.entity_type, entity.confidence
        );
    }

    tracing::debug!("Extracted {} relationships:", relationships.len());
    for relationship in &relationships {
        tracing::debug!(
            "  - '{}' --[{}]--> '{}'",
            relationship.source_entity, relationship.relation_type, relationship.target_entity
        );
    }

    Ok(())
}

/// Demonstrate graph algorithms
async fn demo_graph_algorithms(graph: &KnowledgeGraph) -> RragResult<()> {
    tracing::debug!("Running graph algorithms...");

    // PageRank calculation
    tracing::debug!("\n🏆 PageRank scores:");
    let pagerank_config = PageRankConfig::default();
    let pagerank_scores = GraphAlgorithms::pagerank(graph, &pagerank_config)?;

    let mut sorted_scores: Vec<_> = pagerank_scores.iter().collect();
    sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (node_id, score) in sorted_scores.iter().take(5) {
        if let Some(node) = graph.get_node(node_id) {
            tracing::debug!("  - {}: {:.4}", node.label, score);
        }
    }

    // Betweenness centrality
    tracing::debug!("\n🌐 Betweenness centrality:");
    let centrality_scores = GraphAlgorithms::betweenness_centrality(graph);
    let mut sorted_centrality: Vec<_> = centrality_scores.iter().collect();
    sorted_centrality.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (node_id, score) in sorted_centrality.iter().take(5) {
        if let Some(node) = graph.get_node(node_id) {
            tracing::debug!("  - {}: {:.4}", node.label, score);
        }
    }

    // Graph traversal
    tracing::debug!("\n🚶 Graph traversal from 'Artificial Intelligence':");
    let ai_node_id = graph
        .nodes
        .values()
        .find(|node| node.label == "Artificial Intelligence")
        .map(|node| &node.id);

    if let Some(ai_id) = ai_node_id {
        let traversal_config = TraversalConfig::default();
        let visited_nodes = GraphAlgorithms::bfs_search(graph, ai_id, &traversal_config)?;

        tracing::debug!("  Visited nodes in order:");
        for (i, node_id) in visited_nodes.iter().enumerate() {
            if let Some(node) = graph.get_node(node_id) {
                tracing::debug!("    {}. {}", i + 1, node.label);
            }
        }
    }

    Ok(())
}

/// Demonstrate query expansion using graph structure
async fn demo_query_expansion(graph: &KnowledgeGraph) -> RragResult<()> {
    use rrag::graph_retrieval::query_expansion::{ExpansionConfig, GraphQueryExpander};

    tracing::debug!("Demonstrating query expansion...");

    let expansion_config = ExpansionConfig::default();
    let expander = GraphQueryExpander::new(graph.clone(), expansion_config);

    let queries = vec![
        "machine learning",
        "artificial intelligence",
        "deep learning models",
    ];

    for query in queries {
        tracing::debug!("\n🔍 Expanding query: '{}'", query);

        let options = ExpansionOptions {
            strategies: vec![
                ExpansionStrategy::Semantic,
                ExpansionStrategy::Hierarchical,
                ExpansionStrategy::Similarity,
            ],
            max_terms: Some(5),
            min_confidence: 0.3,
            include_original: true,
            ..Default::default()
        };

        let expansion_result = expander.expand_query(query, &options).await?;

        tracing::debug!("  Original: {}", expansion_result.original_query);
        tracing::debug!("  Expanded terms:");
        for term in &expansion_result.expanded_terms {
            tracing::debug!(
                "    - '{}' (strategy: {}, confidence: {:.2})",
                term.term, term.strategy, term.confidence
            );
        }
        tracing::debug!("  Overall confidence: {:.2}", expansion_result.confidence);

        // Get suggestions
        let suggestions = expander.get_suggestions(query, 3).await?;
        if !suggestions.is_empty() {
            tracing::debug!("  Suggestions: {:?}", suggestions);
        }
    }

    Ok(())
}

/// Demonstrate the complete graph-based retrieval system
async fn demo_graph_retrieval(documents: &[Document]) -> RragResult<()> {
    tracing::debug!("Building complete graph-based retrieval system...");

    // Configure the builder
    let build_config = GraphBuildConfig {
        calculate_pagerank: true,
        generate_entity_embeddings: false, // Disabled for demo
        enable_parallel_processing: false, // Simplified for demo
        batch_size: 10,
        ..Default::default()
    };

    let builder = GraphRetrievalBuilder::new()
        .with_config(build_config)
        .with_in_memory_storage(Default::default())
        .with_query_expansion(true)
        .with_pagerank_scoring(true)
        .with_scoring_weights(0.6, 0.4) // Favor graph-based scoring
        .with_expansion_strategies(vec![
            ExpansionStrategy::Semantic,
            ExpansionStrategy::Hierarchical,
            ExpansionStrategy::CoOccurrence,
        ]);

    // Build the retriever with progress tracking
    let progress_callback = Box::new(PrintProgressCallback);
    let retriever = builder
        .build_from_documents(documents.to_vec(), Some(progress_callback))
        .await;

    match retriever {
        Ok(retriever) => {
            tracing::debug!("\n✅ Graph retriever built successfully!");

            // Test retrieval with different queries
            let test_queries = vec![
                "machine learning algorithms",
                "neural networks deep learning",
                "natural language processing",
                "computer vision applications",
            ];

            for query_text in test_queries {
                tracing::debug!("\n🔍 Query: '{}'", query_text);

                let search_query = SearchQuery::text(query_text)
                    .with_limit(3)
                    .with_min_score(0.1);

                match retriever.search(&search_query).await {
                    Ok(results) => {
                        tracing::debug!("  Found {} results:", results.len());
                        for (i, result) in results.iter().enumerate() {
                            tracing::debug!(
                                "    {}. Score: {:.3}, Content: {}",
                                i + 1,
                                result.score,
                                result.content.chars().take(100).collect::<String>()
                            );

                            // Show graph-specific metadata
                            if let Some(graph_score) = result.metadata.get("graph_score") {
                                tracing::debug!("       Graph score: {}", graph_score);
                            }
                            if let Some(entities) = result.metadata.get("related_entities") {
                                tracing::debug!("       Related entities: {}", entities);
                            }
                        }
                    }
                    Err(e) => tracing::debug!("  Search failed: {}", e),
                }
            }

            // Display retriever statistics
            match retriever.stats().await {
                Ok(stats) => {
                    tracing::debug!("\n📊 Retriever statistics:");
                    tracing::debug!("  - Total items: {}", stats.total_items);
                    tracing::debug!("  - Index type: {}", stats.index_type);
                    tracing::debug!("  - Size: {} bytes", stats.size_bytes);
                }
                Err(e) => tracing::debug!("Failed to get stats: {}", e),
            }

            // Health check
            match retriever.health_check().await {
                Ok(is_healthy) => {
                    tracing::debug!(
                        "  - Health status: {}",
                        if is_healthy {
                            "✅ Healthy"
                        } else {
                            "❌ Unhealthy"
                        }
                    );
                }
                Err(e) => tracing::debug!("Health check failed: {}", e),
            }
        }
        Err(e) => {
            error!(" Failed to build graph retriever: {}", e);
            tracing::debug!(
                "This is expected in the demo as we don't have full NLP capabilities implemented"
            );
        }
    }

    Ok(())
}

/// Demonstrate advanced configuration options
async fn demo_advanced_configuration() -> RragResult<()> {
    tracing::debug!("Exploring advanced configuration options...");

    // Create different configuration profiles
    let configs = vec![
        (
            "Lightweight",
            GraphConfigBuilder::new()
                .with_minimal_features()
                .with_entity_confidence_threshold(0.8)
                .with_similarity_threshold(0.6)
                .with_batch_size(50)
                .build(),
        ),
        (
            "High Performance",
            GraphConfigBuilder::new()
                .with_all_features()
                .with_parallel_processing(true)
                .with_num_workers(8)
                .with_batch_size(200)
                .with_memory_limits(2048, 2_000_000, 10_000_000)
                .build(),
        ),
        (
            "Research",
            GraphConfigBuilder::new()
                .with_entity_extraction(true)
                .with_query_expansion(true)
                .with_pagerank_scoring(true)
                .with_max_expansion_terms(20)
                .with_pagerank_damping_factor(0.9)
                .with_traversal_limits(6, 2000)
                .build(),
        ),
    ];

    for (name, config) in configs {
        tracing::debug!("\n⚙️  {} Configuration:", name);

        // Validate configuration
        match config.validate() {
            Ok(warnings) => {
                tracing::debug!("  ✅ Configuration is valid");
                if !warnings.is_empty() {
                    tracing::debug!("  ⚠️  Warnings:");
                    for warning in warnings {
                        tracing::debug!("    - {}", warning);
                    }
                }
            }
            Err(errors) => {
                tracing::debug!("  ❌ Configuration has errors:");
                for error in errors {
                    tracing::debug!("    - {}", error);
                }
            }
        }

        // Display key settings
        tracing::debug!("  Features:");
        tracing::debug!(
            "    - Entity extraction: {}",
            config.features.entity_extraction
        );
        tracing::debug!("    - Query expansion: {}", config.features.query_expansion);
        tracing::debug!(
            "    - PageRank scoring: {}",
            config.features.pagerank_scoring
        );
        tracing::debug!(
            "    - Path-based retrieval: {}",
            config.features.path_based_retrieval
        );

        tracing::debug!("  Performance:");
        tracing::debug!(
            "    - Parallel processing: {}",
            config.performance.enable_parallel_processing
        );
        tracing::debug!("    - Workers: {}", config.performance.num_workers);
        tracing::debug!("    - Batch size: {}", config.performance.batch_size);
        tracing::debug!(
            "    - Memory limit: {} MB",
            config.performance.memory_limits.max_graph_size_mb
        );

        tracing::debug!("  Algorithms:");
        tracing::debug!(
            "    - PageRank damping: {:.2}",
            config.algorithms.pagerank.damping_factor
        );
        tracing::debug!(
            "    - Max traversal depth: {}",
            config.algorithms.traversal.max_depth
        );
        tracing::debug!(
            "    - Similarity threshold: {:.2}",
            config.algorithms.similarity.similarity_threshold
        );
    }

    // Demonstrate serialization
    tracing::debug!("\n💾 Configuration serialization:");
    let config = GraphConfigBuilder::new().with_all_features().build();

    match serde_json::to_string_pretty(&config) {
        Ok(json) => {
            tracing::debug!("Configuration as JSON (first 300 chars):");
            tracing::debug!("{}", json.chars().take(300).collect::<String>());
            if json.len() > 300 {
                tracing::debug!("...(truncated)");
            }
        }
        Err(e) => tracing::debug!("Failed to serialize configuration: {}", e),
    }

    Ok(())
}
