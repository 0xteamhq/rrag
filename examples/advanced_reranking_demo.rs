//! # Advanced Reranking Demo
//!
//! Demonstrates the complete advanced reranking system including:
//! - Cross-encoder models for query-document relevance
//! - Multi-signal reranking with various relevance signals
//! - Learning-to-rank models for sophisticated scoring
//! - Neural reranking with transformer architectures

use rrag::{
    reranking::{
        cross_encoder::CrossEncoderModelType,
        learning_to_rank::LTRModelType,
        neural_reranker::{NeuralArchitecture, NeuralConfig, NeuralReranker},
        AdvancedReranker, AdvancedRerankingConfig, CrossEncoderConfig, CrossEncoderReranker,
        LTRConfig, LearningToRankReranker, MultiSignalConfig, MultiSignalReranker,
        RerankingStrategyType, ScoreCombination, SignalType, SignalWeight,
    },
    SearchResult,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🎯 RRAG Advanced Reranking Demonstration");
    tracing::debug!("========================================\n");

    // Create test search results
    let search_results = create_test_results();
    let query = "machine learning algorithms for data analysis and pattern recognition";

    tracing::debug!("📊 Initial Search Results:");
    for (i, result) in search_results.iter().enumerate() {
        tracing::debug!(
            "  {}. [Score: {:.3}] {} - {}",
            i + 1,
            result.score,
            result.id,
            &result.content[..100.min(result.content.len())]
        );
    }
    tracing::debug!();

    // Demo 1: Cross-Encoder Reranking
    demo_cross_encoder_reranking(query, &search_results).await?;

    // Demo 2: Multi-Signal Reranking
    demo_multi_signal_reranking(query, &search_results).await?;

    // Demo 3: Learning-to-Rank Reranking
    demo_ltr_reranking(query, &search_results).await?;

    // Demo 4: Neural Reranking
    demo_neural_reranking(query, &search_results).await?;

    // Demo 5: Complete Advanced Reranking Pipeline
    demo_complete_advanced_reranking(query, &search_results).await?;

    // Demo 6: Reranking Comparison
    demo_reranking_comparison(query, &search_results).await?;

    Ok(())
}

async fn demo_cross_encoder_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🤖 Cross-Encoder Reranking Demo");
    tracing::debug!("{}", "─".repeat(50));

    // Test different cross-encoder models
    let models = vec![
        ("BERT", CrossEncoderModelType::SimulatedBert),
        ("RoBERTa", CrossEncoderModelType::RoBERTa),
        ("DistilBERT", CrossEncoderModelType::DistilBert),
    ];

    for (model_name, model_type) in models {
        tracing::debug!("\n🔍 Testing {} Cross-Encoder:", model_name);

        let mut config = CrossEncoderConfig::default();
        config.model_type = model_type;

        let reranker = CrossEncoderReranker::new(config);
        let model_info = reranker.get_model_info();

        tracing::debug!("  Model Info:");
        tracing::debug!("    • Name: {}", model_info.name);
        tracing::debug!("    • Parameters: {:?}", model_info.parameters);
        tracing::debug!(
            "    • Max Sequence Length: {}",
            model_info.max_sequence_length
        );

        match reranker.rerank(query, results).await {
            Ok(scores) => {
                let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
                ranked_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                tracing::debug!("  Reranked Results:");
                for (rank, (idx, score)) in ranked_results.iter().take(3).enumerate() {
                    tracing::debug!(
                        "    {}. [Score: {:.3}] {} - {}",
                        rank + 1,
                        score,
                        results[*idx].id,
                        &results[*idx].content[..60.min(results[*idx].content.len())]
                    );
                }
            }
            Err(e) => {
                tracing::debug!("  ❌ Error: {}", e);
            }
        }
    }

    tracing::debug!("\n✅ Cross-Encoder Demo Complete\n");
    Ok(())
}

async fn demo_multi_signal_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("📊 Multi-Signal Reranking Demo");
    tracing::debug!("{}", "─".repeat(50));

    let mut config = MultiSignalConfig::default();

    // Configure signal weights
    config
        .signal_weights
        .insert(SignalType::SemanticRelevance, SignalWeight::Fixed(0.4));
    config
        .signal_weights
        .insert(SignalType::TextualRelevance, SignalWeight::Fixed(0.3));
    config
        .signal_weights
        .insert(SignalType::DocumentQuality, SignalWeight::Fixed(0.2));
    config
        .signal_weights
        .insert(SignalType::DocumentFreshness, SignalWeight::Fixed(0.1));

    tracing::debug!("📋 Signal Configuration:");
    for (signal_type, weight) in &config.signal_weights {
        match weight {
            SignalWeight::Fixed(w) => {
                tracing::debug!("  • {:?}: {:.2}", signal_type, w);
            }
            _ => {
                tracing::debug!("  • {:?}: Dynamic", signal_type);
            }
        }
    }

    let reranker = MultiSignalReranker::new(config);

    match reranker.rerank(query, results).await {
        Ok(scores) => {
            let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
            ranked_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            tracing::debug!("\n📈 Multi-Signal Reranked Results:");
            for (rank, (idx, score)) in ranked_results.iter().enumerate() {
                tracing::debug!(
                    "  {}. [Score: {:.3}] {} - Quality, freshness, and relevance combined",
                    rank + 1,
                    score,
                    results[*idx].id
                );
            }
        }
        Err(e) => {
            error!(" Error: {}", e);
        }
    }

    tracing::debug!("\n✅ Multi-Signal Demo Complete\n");
    Ok(())
}

async fn demo_ltr_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🎯 Learning-to-Rank Reranking Demo");
    tracing::debug!("{}", "─".repeat(50));

    let mut config = LTRConfig::default();
    config.model_type = LTRModelType::SimulatedLambdaMART;

    // Configure model parameters
    config
        .model_parameters
        .insert("num_trees".to_string(), 50.0);
    config
        .model_parameters
        .insert("learning_rate".to_string(), 0.1);
    config.model_parameters.insert("max_depth".to_string(), 8.0);

    tracing::debug!("🌳 LambdaMART Configuration:");
    for (param, value) in &config.model_parameters {
        tracing::debug!("  • {}: {}", param, value);
    }

    let reranker = LearningToRankReranker::new(config);

    match reranker.rerank(query, results).await {
        Ok(scores) => {
            let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
            ranked_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            tracing::debug!("\n🎯 LTR Reranked Results:");
            for (rank, (idx, score)) in ranked_results.iter().enumerate() {
                tracing::debug!(
                    "  {}. [Score: {:.3}] {} - ML-optimized ranking",
                    rank + 1,
                    score,
                    results[*idx].id
                );
            }
        }
        Err(e) => {
            error!(" Error: {}", e);
        }
    }

    tracing::debug!("\n✅ LTR Demo Complete\n");
    Ok(())
}

async fn demo_neural_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🧠 Neural Reranking Demo");
    tracing::debug!("{}", "─".repeat(50));

    let neural_architectures = vec![
        ("Simulated BERT", NeuralArchitecture::SimulatedBERT),
        ("BERT", NeuralArchitecture::BERT),
        ("RoBERTa", NeuralArchitecture::RoBERTa),
    ];

    for (arch_name, architecture) in neural_architectures {
        tracing::debug!("\n🏗️ Testing {} Architecture:", arch_name);

        let mut config = NeuralConfig::default();
        config.architecture = architecture;

        let reranker = NeuralReranker::new(config);

        match reranker.rerank(query, results).await {
            Ok(scores) => {
                let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
                ranked_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                tracing::debug!("  Neural Reranked Results:");
                for (rank, (idx, score)) in ranked_results.iter().take(3).enumerate() {
                    tracing::debug!(
                        "    {}. [Score: {:.3}] {} - Attention-based scoring",
                        rank + 1,
                        score,
                        results[*idx].id
                    );
                }
            }
            Err(e) => {
                tracing::debug!("  ❌ Error: {}", e);
            }
        }
    }

    tracing::debug!("\n✅ Neural Reranking Demo Complete\n");
    Ok(())
}

async fn demo_complete_advanced_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("🚀 Complete Advanced Reranking Pipeline");
    tracing::debug!("{}", "─".repeat(50));

    // Configure the advanced reranker with all components
    let mut config = AdvancedRerankingConfig::default();
    config.enable_cross_encoder = true;
    config.enable_multi_signal = true;
    config.enable_ltr = false; // Disable for this demo to focus on cross-encoder + multi-signal
    config.enable_neural = false;

    config.strategy_order = vec![
        RerankingStrategyType::CrossEncoder,
        RerankingStrategyType::MultiSignal,
    ];

    config.score_combination = ScoreCombination::Weighted(vec![0.7, 0.3]);
    config.max_candidates = 10;
    config.score_threshold = 0.1;

    tracing::debug!("⚙️ Advanced Reranking Configuration:");
    tracing::debug!("  • Cross-Encoder: {}", config.enable_cross_encoder);
    tracing::debug!("  • Multi-Signal: {}", config.enable_multi_signal);
    tracing::debug!("  • LTR: {}", config.enable_ltr);
    tracing::debug!("  • Neural: {}", config.enable_neural);
    tracing::debug!("  • Strategy Order: {:?}", config.strategy_order);
    tracing::debug!("  • Score Combination: {:?}", config.score_combination);
    tracing::debug!("  • Max Candidates: {}", config.max_candidates);
    tracing::debug!("  • Score Threshold: {}", config.score_threshold);

    let reranker = AdvancedReranker::new(config);

    let start_time = std::time::Instant::now();

    match reranker.rerank(query, results.to_vec()).await {
        Ok(reranked_results) => {
            let processing_time = start_time.elapsed();

            tracing::debug!("\n🎯 Advanced Reranked Results:");
            tracing::debug!("⏱️ Processing time: {:?}", processing_time);
            tracing::debug!(
                "📊 Results processed: {} -> {}",
                results.len(),
                reranked_results.len()
            );

            for (rank, result) in reranked_results.iter().enumerate() {
                tracing::debug!(
                    "\n  {}. {} [Final Score: {:.3}]",
                    rank + 1,
                    result.document_id,
                    result.final_score
                );
                tracing::debug!(
                    "     Original Rank: {} → New Rank: {}",
                    result.original_rank + 1,
                    result.new_rank + 1
                );
                tracing::debug!("     Confidence: {:.2}", result.confidence);

                if !result.component_scores.is_empty() {
                    tracing::debug!("     Component Scores:");
                    for (component, score) in &result.component_scores {
                        tracing::debug!("       • {}: {:.3}", component, score);
                    }
                }

                if let Some(explanation) = &result.explanation {
                    tracing::debug!("     Explanation: {}", explanation);
                }

                tracing::debug!(
                    "     Processing: {}ms, {} rerankers used",
                    result.metadata.reranking_time_ms,
                    result.metadata.rerankers_used.len()
                );

                if !result.metadata.warnings.is_empty() {
                    tracing::debug!("     Warnings: {:?}", result.metadata.warnings);
                }
            }
        }
        Err(e) => {
            error!(" Error: {}", e);
        }
    }

    tracing::debug!("\n✅ Advanced Reranking Pipeline Complete\n");
    Ok(())
}

async fn demo_reranking_comparison(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::debug!("📊 Reranking Methods Comparison");
    tracing::debug!("{}", "─".repeat(50));

    let mut comparison_results = Vec::new();

    // Original ranking
    let original_ranking: Vec<(String, f32)> =
        results.iter().map(|r| (r.id.clone(), r.score)).collect();
    comparison_results.push(("Original", original_ranking));

    // Cross-encoder reranking
    let cross_encoder_config = CrossEncoderConfig::default();
    let cross_encoder_reranker = CrossEncoderReranker::new(cross_encoder_config);

    if let Ok(scores) = cross_encoder_reranker.rerank(query, results).await {
        let mut ranked: Vec<(String, f32)> = scores
            .into_iter()
            .map(|(idx, score)| (results[idx].id.clone(), score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        comparison_results.push(("Cross-Encoder", ranked));
    }

    // Multi-signal reranking
    let multi_signal_config = MultiSignalConfig::default();
    let multi_signal_reranker = MultiSignalReranker::new(multi_signal_config);

    if let Ok(scores) = multi_signal_reranker.rerank(query, results).await {
        let mut ranked: Vec<(String, f32)> = scores
            .into_iter()
            .map(|(idx, score)| (results[idx].id.clone(), score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        comparison_results.push(("Multi-Signal", ranked));
    }

    // Neural reranking
    let neural_config = NeuralConfig::default();
    let neural_reranker = NeuralReranker::new(neural_config);

    if let Ok(scores) = neural_reranker.rerank(query, results).await {
        let mut ranked: Vec<(String, f32)> = scores
            .into_iter()
            .map(|(idx, score)| (results[idx].id.clone(), score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        comparison_results.push(("Neural", ranked));
    }

    // Display comparison
    tracing::debug!("📋 Ranking Comparison (Top 5):");
    tracing::debug!(
        "{:<15} {:<15} {:<15} {:<15} {:<15}",
        "Rank", "Original", "Cross-Encoder", "Multi-Signal", "Neural"
    );
    tracing::debug!("{}", "─".repeat(75));

    for rank in 0..5.min(results.len()) {
        print!("{:<15}", format!("#{}", rank + 1));

        for (_method_name, ranking) in &comparison_results {
            let doc_id = if rank < ranking.len() {
                &ranking[rank].0
            } else {
                "N/A"
            };
            print!("{:<15}", doc_id);
        }
        tracing::debug!();
    }

    // Calculate ranking differences
    tracing::debug!("\n📈 Ranking Analysis:");
    if comparison_results.len() > 1 {
        for i in 1..comparison_results.len() {
            let method_name = &comparison_results[i].0;
            let ranking = &comparison_results[i].1;
            let original = &comparison_results[0].1;

            let mut position_changes = 0;
            let mut total_score_diff = 0.0;

            for (doc_id, score) in ranking {
                if let Some(original_pos) = original.iter().position(|(id, _)| id == doc_id) {
                    if let Some(new_pos) = ranking.iter().position(|(id, _)| id == doc_id) {
                        if original_pos != new_pos {
                            position_changes += 1;
                        }
                    }
                }

                if let Some((_, original_score)) = original.iter().find(|(id, _)| id == doc_id) {
                    total_score_diff += (score - original_score).abs();
                }
            }

            tracing::debug!(
                "  • {}: {} position changes, avg score diff: {:.3}",
                method_name,
                position_changes,
                total_score_diff / ranking.len() as f32
            );
        }
    }

    tracing::debug!("\n✅ Reranking Comparison Complete\n");
    Ok(())
}

fn create_test_results() -> Vec<SearchResult> {
    let data = vec![
        ("doc_ml_intro", "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.", 0.85),
        ("doc_algorithms", "Common machine learning algorithms include supervised learning algorithms like linear regression, logistic regression, decision trees, random forests, and support vector machines. Unsupervised learning includes clustering algorithms like k-means and hierarchical clustering.", 0.92),
        ("doc_data_analysis", "Data analysis involves inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, suggesting conclusions, and supporting decision-making. Modern data analysis heavily relies on statistical techniques and machine learning algorithms.", 0.78),
        ("doc_pattern_recognition", "Pattern recognition is the automated recognition of patterns and regularities in data. In machine learning, pattern recognition is accomplished by algorithms that learn from training data to make predictions or decisions without being explicitly programmed for the task.", 0.81),
        ("doc_neural_networks", "Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling, or clustering of raw input.", 0.73),
        ("doc_deep_learning", "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.", 0.69),
        ("doc_statistics", "Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. Statistical analysis is fundamental to data science and machine learning applications.", 0.67),
        ("doc_programming", "Programming languages commonly used in machine learning include Python, R, Java, and Scala. Python is particularly popular due to its extensive libraries like scikit-learn, TensorFlow, and PyTorch.", 0.45),
        ("doc_unrelated", "Cooking is the art, science, and craft of using heat to prepare food for consumption. Cooking techniques and ingredients vary widely across the world, reflecting unique environments, economics, cultural traditions, and trends.", 0.15),
    ];

    data.into_iter()
        .enumerate()
        .map(|(rank, (id, content, score))| SearchResult {
            id: id.to_string(),
            content: content.to_string(),
            score,
            rank,
            metadata: HashMap::new(),
            embedding: None,
        })
        .collect()
}
