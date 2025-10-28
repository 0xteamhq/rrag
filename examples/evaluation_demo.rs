//! # Evaluation Framework Demo
//! 
//! Demonstrates the comprehensive evaluation framework including RAGAS metrics,
//! retrieval evaluation, generation evaluation, and end-to-end benchmarks.

use rrag::{
    evaluation::{
        EvaluationService, EvaluationConfig, EvaluationData,
        TestQuery, GroundTruth, SystemResponse, RetrievedDocument, SystemTiming,
        ragas::RagasConfig, 
        retrieval_eval::RetrievalEvalConfig,
        generation_eval::GenerationEvalConfig,
        end_to_end::EndToEndConfig,
        benchmarks::BenchmarkEvaluator,
    },
    RragResult,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> RragResult<()> {
    tracing::debug!("🎯 RRAG Evaluation Framework Demonstration");
    tracing::debug!("==========================================\n");

    // Create comprehensive evaluation configuration
    let mut eval_config = EvaluationConfig::default();
    eval_config.output_config.output_dir = "./evaluation_results".to_string();
    
    // Create evaluation service
    let mut evaluation_service = EvaluationService::new(eval_config);
    
    // Demo 1: RAGAS Evaluation
    tracing::debug!("📊 Demo 1: RAGAS Metrics Evaluation");
    tracing::debug!("─".repeat(40));
    demo_ragas_evaluation().await?;
    tracing::debug!();
    
    // Demo 2: Retrieval Evaluation
    tracing::debug!("🔍 Demo 2: Retrieval Metrics Evaluation");
    tracing::debug!("─".repeat(40));
    demo_retrieval_evaluation().await?;
    tracing::debug!();
    
    // Demo 3: Generation Evaluation
    tracing::debug!("✍️ Demo 3: Generation Quality Evaluation");
    tracing::debug!("─".repeat(40));
    demo_generation_evaluation().await?;
    tracing::debug!();
    
    // Demo 4: End-to-End Evaluation
    tracing::debug!("🚀 Demo 4: End-to-End System Evaluation");
    tracing::debug!("─".repeat(40));
    demo_end_to_end_evaluation().await?;
    tracing::debug!();
    
    // Demo 5: Benchmark Evaluation
    tracing::debug!("🏆 Demo 5: Standard Benchmark Evaluation");
    tracing::debug!("─".repeat(40));
    demo_benchmark_evaluation().await?;
    tracing::debug!();
    
    // Demo 6: Complete Evaluation Pipeline
    tracing::debug!("🎪 Demo 6: Complete Evaluation Pipeline");
    tracing::debug!("─".repeat(40));
    
    // Create comprehensive test data
    let evaluation_data = create_comprehensive_test_data();
    
    // Run complete evaluation
    let evaluation_results = evaluation_service.evaluate(evaluation_data).await?;
    
    // Display results summary
    tracing::debug!("📈 Evaluation Results Summary:");
    for (eval_type, result) in &evaluation_results {
        tracing::debug!("\n{:?} Evaluation:", eval_type);
        for (metric, score) in &result.overall_scores {
            tracing::debug!("  • {}: {:.3}", metric, score);
        }
        
        if !result.summary.insights.is_empty() {
            tracing::debug!("  📝 Key Insights:");
            for insight in &result.summary.insights {
                tracing::debug!("    - {}", insight);
            }
        }
        
        if !result.summary.recommendations.is_empty() {
            tracing::debug!("  💡 Recommendations:");
            for recommendation in &result.summary.recommendations {
                tracing::debug!("    - {}", recommendation);
            }
        }
    }
    
    // Export results
    tracing::debug!("\n📄 Exporting evaluation results...");
    evaluation_service.export_results(&evaluation_results).await?;
    tracing::debug!("✅ Results exported to ./evaluation_results/");
    
    tracing::debug!("\n🎉 Evaluation framework demonstration complete!");
    
    Ok(())
}

async fn demo_ragas_evaluation() -> RragResult<()> {
    use rrag::evaluation::ragas::{RagasEvaluator, RagasConfig};
    
    let config = RagasConfig::default();
    let evaluator = RagasEvaluator::new(config);
    
    tracing::debug!("🔧 RAGAS Configuration:");
    tracing::debug!("  • Metrics: {:?}", evaluator.get_config().metrics);
    
    let test_data = create_ragas_test_data();
    let result = evaluator.evaluate(&test_data)?;
    
    tracing::debug!("\n📊 RAGAS Results:");
    for (metric, score) in &result.overall_scores {
        tracing::debug!("  • {}: {:.3}", metric, score);
    }
    
    tracing::debug!("  📈 Total queries evaluated: {}", result.summary.total_queries);
    tracing::debug!("  ⏱️ Evaluation time: {:.2}ms", result.summary.performance_stats.total_eval_time_ms);
    
    Ok(())
}

async fn demo_retrieval_evaluation() -> RragResult<()> {
    use rrag::evaluation::retrieval_eval::{RetrievalEvaluator, RetrievalEvalConfig};
    
    let config = RetrievalEvalConfig::default();
    let evaluator = RetrievalEvaluator::new(config);
    
    tracing::debug!("🔧 Retrieval Evaluation Configuration:");
    tracing::debug!("  • K values: [1, 3, 5, 10, 20]");
    tracing::debug!("  • Metrics: {:?}", evaluator.supported_metrics());
    
    let test_data = create_retrieval_test_data();
    let result = evaluator.evaluate(&test_data)?;
    
    tracing::debug!("\n🎯 Retrieval Results:");
    for (metric, score) in &result.overall_scores {
        tracing::debug!("  • {}: {:.3}", metric, score);
    }
    
    if !result.summary.insights.is_empty() {
        tracing::debug!("\n💡 Insights:");
        for insight in &result.summary.insights {
            tracing::debug!("  - {}", insight);
        }
    }
    
    Ok(())
}

async fn demo_generation_evaluation() -> RragResult<()> {
    use rrag::evaluation::generation_eval::{GenerationEvaluator, GenerationEvalConfig};
    
    let config = GenerationEvalConfig::default();
    let evaluator = GenerationEvaluator::new(config);
    
    tracing::debug!("🔧 Generation Evaluation Configuration:");
    tracing::debug!("  • Metrics: {:?}", evaluator.supported_metrics());
    
    let test_data = create_generation_test_data();
    let result = evaluator.evaluate(&test_data)?;
    
    tracing::debug!("\n✨ Generation Results:");
    for (metric, score) in &result.overall_scores {
        tracing::debug!("  • {}: {:.3}", metric, score);
    }
    
    if !result.summary.recommendations.is_empty() {
        tracing::debug!("\n🔧 Recommendations:");
        for recommendation in &result.summary.recommendations {
            tracing::debug!("  - {}", recommendation);
        }
    }
    
    Ok(())
}

async fn demo_end_to_end_evaluation() -> RragResult<()> {
    use rrag::evaluation::end_to_end::{EndToEndEvaluator, EndToEndConfig};
    
    let config = EndToEndConfig::default();
    let evaluator = EndToEndEvaluator::new(config);
    
    tracing::debug!("🔧 End-to-End Evaluation Configuration:");
    tracing::debug!("  • User Experience Weight: {:.1}", evaluator.get_config().performance.accuracy);
    tracing::debug!("  • Metrics: {:?}", evaluator.supported_metrics());
    
    let test_data = create_e2e_test_data();
    let result = evaluator.evaluate(&test_data)?;
    
    tracing::debug!("\n🚀 End-to-End Results:");
    for (metric, score) in &result.overall_scores {
        tracing::debug!("  • {}: {:.3}", metric, score);
    }
    
    tracing::debug!("  🔄 System Throughput: {:.1} QPS", result.summary.performance_stats.throughput_qps);
    tracing::debug!("  💾 Peak Memory Usage: {:.1}MB", result.summary.performance_stats.peak_memory_usage_mb);
    
    Ok(())
}

async fn demo_benchmark_evaluation() -> RragResult<()> {
    let benchmark_evaluator = BenchmarkEvaluator::new();
    
    tracing::debug!("🔧 Benchmark Configuration:");
    tracing::debug!("  • Available Benchmarks: {:?}", benchmark_evaluator.supported_metrics());
    
    // Run all benchmarks
    let benchmark_results = benchmark_evaluator.run_all_benchmarks().await?;
    
    tracing::debug!("\n🏆 Benchmark Results:");
    for (benchmark_name, result) in &benchmark_results {
        tracing::debug!("  • {}: {:.3}", benchmark_name, result.overall_score);
        
        if !result.performance_analysis.strengths.is_empty() {
            tracing::debug!("    ✅ Strengths: {}", result.performance_analysis.strengths.join(", "));
        }
        
        if !result.performance_analysis.weaknesses.is_empty() {
            tracing::debug!("    ⚠️ Weaknesses: {}", result.performance_analysis.weaknesses.join(", "));
        }
    }
    
    // Calculate overall benchmark score
    let overall_score: f32 = benchmark_results.values()
        .map(|r| r.overall_score)
        .sum::<f32>() / benchmark_results.len() as f32;
    
    tracing::debug!("\n📊 Overall Benchmark Score: {:.3}", overall_score);
    
    if overall_score > 0.8 {
        tracing::debug!("🎉 Excellent performance across all benchmarks!");
    } else if overall_score > 0.6 {
        tracing::debug!("👍 Good performance with room for improvement");
    } else {
        tracing::debug!("🔧 Performance needs significant improvement");
    }
    
    Ok(())
}

fn create_comprehensive_test_data() -> EvaluationData {
    let queries = vec![
        TestQuery {
            id: "q1".to_string(),
            query: "What is machine learning and how does it work?".to_string(),
            query_type: Some("conceptual".to_string()),
            metadata: HashMap::new(),
        },
        TestQuery {
            id: "q2".to_string(),
            query: "Explain the difference between supervised and unsupervised learning".to_string(),
            query_type: Some("comparative".to_string()),
            metadata: HashMap::new(),
        },
        TestQuery {
            id: "q3".to_string(),
            query: "What are the applications of deep learning in computer vision?".to_string(),
            query_type: Some("application".to_string()),
            metadata: HashMap::new(),
        },
    ];
    
    let ground_truth = vec![
        GroundTruth {
            query_id: "q1".to_string(),
            relevant_docs: vec!["ml_intro".to_string(), "ml_basics".to_string()],
            expected_answer: Some("Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.".to_string()),
            relevance_judgments: [
                ("ml_intro".to_string(), 1.0),
                ("ml_basics".to_string(), 0.8),
                ("ai_overview".to_string(), 0.6),
            ].iter().cloned().collect(),
            metadata: HashMap::new(),
        },
        GroundTruth {
            query_id: "q2".to_string(),
            relevant_docs: vec!["supervised_learning".to_string(), "unsupervised_learning".to_string()],
            expected_answer: Some("Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds patterns in data without labels.".to_string()),
            relevance_judgments: [
                ("supervised_learning".to_string(), 1.0),
                ("unsupervised_learning".to_string(), 1.0),
                ("ml_types".to_string(), 0.7),
            ].iter().cloned().collect(),
            metadata: HashMap::new(),
        },
        GroundTruth {
            query_id: "q3".to_string(),
            relevant_docs: vec!["deep_learning_cv".to_string(), "cv_applications".to_string()],
            expected_answer: Some("Deep learning applications in computer vision include image classification, object detection, facial recognition, and medical image analysis.".to_string()),
            relevance_judgments: [
                ("deep_learning_cv".to_string(), 1.0),
                ("cv_applications".to_string(), 0.9),
                ("neural_networks".to_string(), 0.5),
            ].iter().cloned().collect(),
            metadata: HashMap::new(),
        },
    ];
    
    let system_responses = vec![
        SystemResponse {
            query_id: "q1".to_string(),
            retrieved_docs: vec![
                RetrievedDocument {
                    doc_id: "ml_intro".to_string(),
                    content: "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data.".to_string(),
                    score: 0.92,
                    rank: 0,
                    metadata: HashMap::new(),
                },
                RetrievedDocument {
                    doc_id: "ml_basics".to_string(),
                    content: "The core principle of machine learning involves algorithms that can learn and make decisions from data without being explicitly programmed for specific tasks.".to_string(),
                    score: 0.85,
                    rank: 1,
                    metadata: HashMap::new(),
                },
                RetrievedDocument {
                    doc_id: "ai_overview".to_string(),
                    content: "Artificial intelligence encompasses various technologies including machine learning, natural language processing, and computer vision.".to_string(),
                    score: 0.73,
                    rank: 2,
                    metadata: HashMap::new(),
                },
            ],
            generated_answer: Some("Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed for each specific task. It works by using algorithms that can automatically improve their performance through experience with data.".to_string()),
            timing: SystemTiming {
                total_time_ms: 1200.0,
                retrieval_time_ms: 450.0,
                generation_time_ms: Some(600.0),
                reranking_time_ms: Some(150.0),
            },
            metadata: HashMap::new(),
        },
        SystemResponse {
            query_id: "q2".to_string(),
            retrieved_docs: vec![
                RetrievedDocument {
                    doc_id: "supervised_learning".to_string(),
                    content: "Supervised learning algorithms learn from labeled training examples to make predictions on new, unseen data. Examples include classification and regression tasks.".to_string(),
                    score: 0.89,
                    rank: 0,
                    metadata: HashMap::new(),
                },
                RetrievedDocument {
                    doc_id: "unsupervised_learning".to_string(),
                    content: "Unsupervised learning finds hidden patterns in data without labeled examples. Common techniques include clustering and dimensionality reduction.".to_string(),
                    score: 0.87,
                    rank: 1,
                    metadata: HashMap::new(),
                },
            ],
            generated_answer: Some("Supervised learning uses labeled training data where the correct answers are provided during training, allowing algorithms to learn the relationship between inputs and outputs. Unsupervised learning, on the other hand, works with unlabeled data to discover hidden patterns, structures, or relationships without knowing the correct answers beforehand.".to_string()),
            timing: SystemTiming {
                total_time_ms: 1100.0,
                retrieval_time_ms: 400.0,
                generation_time_ms: Some(550.0),
                reranking_time_ms: Some(150.0),
            },
            metadata: HashMap::new(),
        },
        SystemResponse {
            query_id: "q3".to_string(),
            retrieved_docs: vec![
                RetrievedDocument {
                    doc_id: "deep_learning_cv".to_string(),
                    content: "Deep learning has revolutionized computer vision with applications in image classification, object detection, facial recognition, and autonomous vehicles.".to_string(),
                    score: 0.94,
                    rank: 0,
                    metadata: HashMap::new(),
                },
                RetrievedDocument {
                    doc_id: "cv_applications".to_string(),
                    content: "Computer vision applications of deep learning include medical image analysis, quality control in manufacturing, satellite imagery analysis, and augmented reality.".to_string(),
                    score: 0.88,
                    rank: 1,
                    metadata: HashMap::new(),
                },
            ],
            generated_answer: Some("Deep learning has numerous applications in computer vision, including image classification for categorizing photos, object detection for identifying and locating objects in images, facial recognition for security systems, medical image analysis for disease diagnosis, and autonomous vehicle navigation for self-driving cars.".to_string()),
            timing: SystemTiming {
                total_time_ms: 1350.0,
                retrieval_time_ms: 500.0,
                generation_time_ms: Some(700.0),
                reranking_time_ms: Some(150.0),
            },
            metadata: HashMap::new(),
        },
    ];
    
    let mut context = HashMap::new();
    context.insert("evaluation_type".to_string(), serde_json::Value::String("comprehensive".to_string()));
    context.insert("domain".to_string(), serde_json::Value::String("machine_learning".to_string()));
    
    EvaluationData {
        queries,
        ground_truth,
        system_responses,
        context,
    }
}

fn create_ragas_test_data() -> EvaluationData {
    // Simplified test data for RAGAS demo
    let mut data = create_comprehensive_test_data();
    data.queries = data.queries.into_iter().take(1).collect();
    data.ground_truth = data.ground_truth.into_iter().take(1).collect();
    data.system_responses = data.system_responses.into_iter().take(1).collect();
    data
}

fn create_retrieval_test_data() -> EvaluationData {
    // Test data focused on retrieval metrics
    create_comprehensive_test_data()
}

fn create_generation_test_data() -> EvaluationData {
    // Test data focused on generation quality
    create_comprehensive_test_data()
}

fn create_e2e_test_data() -> EvaluationData {
    // Test data for end-to-end evaluation
    create_comprehensive_test_data()
}