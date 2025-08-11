//! # Advanced Reranking Demo
//! 
//! Demonstrates the complete advanced reranking system including:
//! - Cross-encoder models for query-document relevance
//! - Multi-signal reranking with various relevance signals
//! - Learning-to-rank models for sophisticated scoring
//! - Neural reranking with transformer architectures

use rrag::{
    SearchResult,
    reranking::{
        AdvancedReranker, AdvancedRerankingConfig, ScoreCombination, RerankingStrategyType,
        CrossEncoderReranker, CrossEncoderConfig,
        MultiSignalReranker, MultiSignalConfig, SignalType, SignalWeight,
        LearningToRankReranker, LTRConfig,
        neural_reranker::{NeuralReranker, NeuralConfig, NeuralArchitecture},
        cross_encoder::CrossEncoderModelType,
        learning_to_rank::LTRModelType,
    },
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 RRAG Advanced Reranking Demonstration");
    println!("========================================\n");

    // Create test search results
    let search_results = create_test_results();
    let query = "machine learning algorithms for data analysis and pattern recognition";
    
    println!("📊 Initial Search Results:");
    for (i, result) in search_results.iter().enumerate() {
        println!("  {}. [Score: {:.3}] {} - {}", 
                i + 1, 
                result.score, 
                result.id,
                &result.content[..100.min(result.content.len())]
        );
    }
    println!();

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
    println!("🤖 Cross-Encoder Reranking Demo");
    println!("{}", "─".repeat(50));
    
    // Test different cross-encoder models
    let models = vec![
        ("BERT", CrossEncoderModelType::SimulatedBert),
        ("RoBERTa", CrossEncoderModelType::RoBERTa),
        ("DistilBERT", CrossEncoderModelType::DistilBert),
    ];
    
    for (model_name, model_type) in models {
        println!("\n🔍 Testing {} Cross-Encoder:", model_name);
        
        let mut config = CrossEncoderConfig::default();
        config.model_type = model_type;
        
        let reranker = CrossEncoderReranker::new(config);
        let model_info = reranker.get_model_info();
        
        println!("  Model Info:");
        println!("    • Name: {}", model_info.name);
        println!("    • Parameters: {:?}", model_info.parameters);
        println!("    • Max Sequence Length: {}", model_info.max_sequence_length);
        
        match reranker.rerank(query, results).await {
            Ok(scores) => {
                let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
                ranked_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                println!("  Reranked Results:");
                for (rank, (idx, score)) in ranked_results.iter().take(3).enumerate() {
                    println!("    {}. [Score: {:.3}] {} - {}", 
                            rank + 1,
                            score,
                            results[*idx].id,
                            &results[*idx].content[..60.min(results[*idx].content.len())]
                    );
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }
    
    println!("\n✅ Cross-Encoder Demo Complete\n");
    Ok(())
}

async fn demo_multi_signal_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Multi-Signal Reranking Demo");
    println!("{}", "─".repeat(50));
    
    let mut config = MultiSignalConfig::default();
    
    // Configure signal weights
    config.signal_weights.insert(SignalType::SemanticRelevance, SignalWeight::Fixed(0.4));
    config.signal_weights.insert(SignalType::TextualRelevance, SignalWeight::Fixed(0.3));
    config.signal_weights.insert(SignalType::DocumentQuality, SignalWeight::Fixed(0.2));
    config.signal_weights.insert(SignalType::DocumentFreshness, SignalWeight::Fixed(0.1));
    
    println!("📋 Signal Configuration:");
    for (signal_type, weight) in &config.signal_weights {
        match weight {
            SignalWeight::Fixed(w) => {
                println!("  • {:?}: {:.2}", signal_type, w);
            }
            _ => {
                println!("  • {:?}: Dynamic", signal_type);
            }
        }
    }
    
    let reranker = MultiSignalReranker::new(config);
    
    match reranker.rerank(query, results).await {
        Ok(scores) => {
            let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
            ranked_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("\n📈 Multi-Signal Reranked Results:");
            for (rank, (idx, score)) in ranked_results.iter().enumerate() {
                println!("  {}. [Score: {:.3}] {} - Quality, freshness, and relevance combined", 
                        rank + 1,
                        score,
                        results[*idx].id
                );
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }
    
    println!("\n✅ Multi-Signal Demo Complete\n");
    Ok(())
}

async fn demo_ltr_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Learning-to-Rank Reranking Demo");
    println!("{}", "─".repeat(50));
    
    let mut config = LTRConfig::default();
    config.model_type = LTRModelType::SimulatedLambdaMART;
    
    // Configure model parameters
    config.model_parameters.insert("num_trees".to_string(), 50.0);
    config.model_parameters.insert("learning_rate".to_string(), 0.1);
    config.model_parameters.insert("max_depth".to_string(), 8.0);
    
    println!("🌳 LambdaMART Configuration:");
    for (param, value) in &config.model_parameters {
        println!("  • {}: {}", param, value);
    }
    
    let reranker = LearningToRankReranker::new(config);
    
    match reranker.rerank(query, results).await {
        Ok(scores) => {
            let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
            ranked_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("\n🎯 LTR Reranked Results:");
            for (rank, (idx, score)) in ranked_results.iter().enumerate() {
                println!("  {}. [Score: {:.3}] {} - ML-optimized ranking", 
                        rank + 1,
                        score,
                        results[*idx].id
                );
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }
    
    println!("\n✅ LTR Demo Complete\n");
    Ok(())
}

async fn demo_neural_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Reranking Demo");
    println!("{}", "─".repeat(50));
    
    let neural_architectures = vec![
        ("Simulated BERT", NeuralArchitecture::SimulatedBERT),
        ("BERT", NeuralArchitecture::BERT),
        ("RoBERTa", NeuralArchitecture::RoBERTa),
    ];
    
    for (arch_name, architecture) in neural_architectures {
        println!("\n🏗️ Testing {} Architecture:", arch_name);
        
        let mut config = NeuralConfig::default();
        config.architecture = architecture;
        
        let reranker = NeuralReranker::new(config);
        
        match reranker.rerank(query, results).await {
            Ok(scores) => {
                let mut ranked_results: Vec<(usize, f32)> = scores.into_iter().collect();
                ranked_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                println!("  Neural Reranked Results:");
                for (rank, (idx, score)) in ranked_results.iter().take(3).enumerate() {
                    println!("    {}. [Score: {:.3}] {} - Attention-based scoring", 
                            rank + 1,
                            score,
                            results[*idx].id
                    );
                }
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }
    
    println!("\n✅ Neural Reranking Demo Complete\n");
    Ok(())
}

async fn demo_complete_advanced_reranking(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Complete Advanced Reranking Pipeline");
    println!("{}", "─".repeat(50));
    
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
    
    println!("⚙️ Advanced Reranking Configuration:");
    println!("  • Cross-Encoder: {}", config.enable_cross_encoder);
    println!("  • Multi-Signal: {}", config.enable_multi_signal);
    println!("  • LTR: {}", config.enable_ltr);
    println!("  • Neural: {}", config.enable_neural);
    println!("  • Strategy Order: {:?}", config.strategy_order);
    println!("  • Score Combination: {:?}", config.score_combination);
    println!("  • Max Candidates: {}", config.max_candidates);
    println!("  • Score Threshold: {}", config.score_threshold);
    
    let reranker = AdvancedReranker::new(config);
    
    let start_time = std::time::Instant::now();
    
    match reranker.rerank(query, results.to_vec()).await {
        Ok(reranked_results) => {
            let processing_time = start_time.elapsed();
            
            println!("\n🎯 Advanced Reranked Results:");
            println!("⏱️ Processing time: {:?}", processing_time);
            println!("📊 Results processed: {} -> {}", results.len(), reranked_results.len());
            
            for (rank, result) in reranked_results.iter().enumerate() {
                println!("\n  {}. {} [Final Score: {:.3}]", 
                        rank + 1,
                        result.document_id,
                        result.final_score
                );
                println!("     Original Rank: {} → New Rank: {}", 
                        result.original_rank + 1,
                        result.new_rank + 1
                );
                println!("     Confidence: {:.2}", result.confidence);
                
                if !result.component_scores.is_empty() {
                    println!("     Component Scores:");
                    for (component, score) in &result.component_scores {
                        println!("       • {}: {:.3}", component, score);
                    }
                }
                
                if let Some(explanation) = &result.explanation {
                    println!("     Explanation: {}", explanation);
                }
                
                println!("     Processing: {}ms, {} rerankers used", 
                        result.metadata.reranking_time_ms,
                        result.metadata.rerankers_used.len()
                );
                
                if !result.metadata.warnings.is_empty() {
                    println!("     Warnings: {:?}", result.metadata.warnings);
                }
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }
    
    println!("\n✅ Advanced Reranking Pipeline Complete\n");
    Ok(())
}

async fn demo_reranking_comparison(
    query: &str,
    results: &[SearchResult],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Reranking Methods Comparison");
    println!("{}", "─".repeat(50));
    
    let mut comparison_results = Vec::new();
    
    // Original ranking
    let original_ranking: Vec<(String, f32)> = results
        .iter()
        .map(|r| (r.id.clone(), r.score))
        .collect();
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
    println!("📋 Ranking Comparison (Top 5):");
    println!("{:<15} {:<15} {:<15} {:<15} {:<15}", 
             "Rank", "Original", "Cross-Encoder", "Multi-Signal", "Neural");
    println!("{}", "─".repeat(75));
    
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
        println!();
    }
    
    // Calculate ranking differences
    println!("\n📈 Ranking Analysis:");
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
            
            println!("  • {}: {} position changes, avg score diff: {:.3}", 
                     method_name, 
                     position_changes,
                     total_score_diff / ranking.len() as f32
            );
        }
    }
    
    println!("\n✅ Reranking Comparison Complete\n");
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