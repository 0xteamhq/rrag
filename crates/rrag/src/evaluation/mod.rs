//! # RRAG Evaluation Framework
//!
//! Enterprise-grade evaluation framework for RAG systems providing comprehensive
//! assessment capabilities based on RAGAS metrics, custom evaluation methods,
//! and industry-standard benchmarks.
//!
//! This module offers a complete evaluation suite for RAG systems, enabling
//! both component-level analysis (retrieval, generation) and end-to-end system
//! evaluation. It supports multiple evaluation methodologies, automated benchmarking,
//! and detailed performance analytics.
//!
//! ## Key Features
//!
//! - **RAGAS Integration**: Industry-standard RAG evaluation metrics
//! - **Multi-Level Evaluation**: Component and system-level assessments
//! - **Automated Benchmarking**: Built-in benchmark datasets and evaluation
//! - **Custom Metrics**: Extensible framework for domain-specific evaluation
//! - **Performance Analytics**: Detailed insights and recommendations
//! - **Export Capabilities**: Multiple output formats (JSON, CSV, HTML, Markdown)
//! - **Real-time Monitoring**: Live evaluation during system operation
//!
//! ## Evaluation Types
//!
//! 1. **RAGAS Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
//! 2. **Retrieval Evaluation**: Precision@K, Recall@K, MRR, NDCG
//! 3. **Generation Evaluation**: BLEU, ROUGE, BERTScore, Semantic similarity
//! 4. **End-to-End Evaluation**: Complete pipeline assessment
//! 5. **Benchmark Evaluation**: Performance on standard datasets
//!
//! ## Examples
//!
//! ### Basic Evaluation Setup
//! ```rust
//! use rrag::evaluation::{
//!     EvaluationService, EvaluationConfig, EvaluationType,
//!     EvaluationData, TestQuery, GroundTruth
//! };
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let config = EvaluationConfig {
//!     enabled_evaluations: vec![
//!         EvaluationType::Ragas,
//!         EvaluationType::Retrieval,
//!         EvaluationType::Generation,
//!     ],
//!     ..Default::default()
//! };
//!
//! let mut evaluator = EvaluationService::new(config);
//! println!("📊 Evaluation service initialized with {} evaluators", 3);
//! # Ok(())
//! # }
//! ```
//!
//! ### Running Comprehensive Evaluation
//! ```rust
//! use std::collections::HashMap;
//!
//! # async fn example() -> rrag::RragResult<()> {
//! # let mut evaluator = rrag::evaluation::EvaluationService::new(rrag::evaluation::EvaluationConfig::default());
//! // Prepare test data
//! let test_queries = vec![
//!     rrag::evaluation::TestQuery {
//!         id: "q1".to_string(),
//!         query: "What is machine learning?".to_string(),
//!         query_type: Some("factual".to_string()),
//!         metadata: HashMap::new(),
//!     },
//!     rrag::evaluation::TestQuery {
//!         id: "q2".to_string(),
//!         query: "Explain neural networks in detail".to_string(),
//!         query_type: Some("conceptual".to_string()),
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let ground_truth = vec![
//!     rrag::evaluation::GroundTruth {
//!         query_id: "q1".to_string(),
//!         relevant_docs: vec!["doc_ml_intro".to_string(), "doc_ml_basics".to_string()],
//!         expected_answer: Some(
//!             "Machine learning is a subset of AI that enables computers to learn...".to_string()
//!         ),
//!         relevance_judgments: [("doc_ml_intro".to_string(), 1.0)].iter().cloned().collect(),
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let evaluation_data = rrag::evaluation::EvaluationData {
//!     queries: test_queries,
//!     ground_truth,
//!     system_responses: vec![], // Would be populated with actual system responses
//!     context: HashMap::new(),
//! };
//!
//! // Run evaluation
//! let results = evaluator.evaluate(evaluation_data).await?;
//!
//! for (eval_type, result) in results {
//!     println!("🏆 {:?} Evaluation Results:", eval_type);
//!     for (metric, score) in result.overall_scores {
//!         println!("  {}: {:.4}", metric, score);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### RAGAS Evaluation
//! ```rust
//! use rrag::evaluation::{
//!     ragas::{RagasEvaluator, RagasConfig, RagasMetric},
//!     SystemResponse, RetrievedDocument, SystemTiming
//! };
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let ragas_config = RagasConfig {
//!     enabled_metrics: vec![
//!         RagasMetric::Faithfulness,
//!         RagasMetric::AnswerRelevancy,
//!         RagasMetric::ContextPrecision,
//!         RagasMetric::ContextRecall,
//!     ],
//!     ..Default::default()
//! };
//!
//! let ragas_evaluator = RagasEvaluator::new(ragas_config);
//!
//! // Prepare system response for evaluation
//! let system_response = SystemResponse {
//!     query_id: "q1".to_string(),
//!     retrieved_docs: vec![
//!         RetrievedDocument {
//!             doc_id: "doc_1".to_string(),
//!             content: "Machine learning is a method of data analysis...".to_string(),
//!             score: 0.95,
//!             rank: 1,
//!             metadata: HashMap::new(),
//!         }
//!     ],
//!     generated_answer: Some(
//!         "Machine learning is a subset of artificial intelligence...".to_string()
//!     ),
//!     timing: SystemTiming {
//!         total_time_ms: 250.0,
//!         retrieval_time_ms: 120.0,
//!         generation_time_ms: Some(130.0),
//!         reranking_time_ms: None,
//!     },
//!     metadata: HashMap::new(),
//! };
//!
//! println!("📈 RAGAS evaluation completed with {} metrics", 4);
//! # Ok(())
//! # }
//! ```
//!
//! ### Retrieval-Specific Evaluation
//! ```rust
//! use rrag::evaluation::retrieval_eval::{
//!     RetrievalEvaluator, RetrievalEvalConfig, RetrievalMetric
//! };
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let retrieval_config = RetrievalEvalConfig {
//!     metrics: vec![
//!         RetrievalMetric::PrecisionAtK(10),
//!         RetrievalMetric::RecallAtK(10),
//!         RetrievalMetric::MeanReciprocalRank,
//!         RetrievalMetric::NDCG(10),
//!     ],
//!     k_values: vec![1, 5, 10, 20],
//!     ..Default::default()
//! };
//!
//! let retrieval_evaluator = RetrievalEvaluator::new(retrieval_config);
//!
//! // Results will include:
//! // - Precision@1, @5, @10, @20
//! // - Recall@1, @5, @10, @20  
//! // - Mean Reciprocal Rank
//! // - Normalized Discounted Cumulative Gain
//!
//! println!("🏁 Retrieval evaluation configured for multiple K values");
//! # Ok(())
//! # }
//! ```
//!
//! ### Generation Quality Evaluation
//! ```rust
//! use rrag::evaluation::generation_eval::{
//!     GenerationEvaluator, GenerationEvalConfig, GenerationMetric
//! };
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let generation_config = GenerationEvalConfig {
//!     metrics: vec![
//!         GenerationMetric::BLEU,
//!         GenerationMetric::ROUGE("rouge-l".to_string()),
//!         GenerationMetric::BERTScore,
//!         GenerationMetric::SemanticSimilarity,
//!     ],
//!     reference_free: false,
//!     ..Default::default()
//! };
//!
//! let generation_evaluator = GenerationEvaluator::new(generation_config);
//!
//! // Evaluates generated answers against reference answers
//! // Provides detailed analysis of:
//! // - Lexical similarity (BLEU, ROUGE)
//! // - Semantic similarity (BERTScore, embeddings)
//! // - Factual accuracy
//! // - Fluency and coherence
//!
//! println!("✍️ Generation evaluation ready for quality assessment");
//! # Ok(())
//! # }
//! ```
//!
//! ### End-to-End System Evaluation
//! ```rust
//! use rrag::evaluation::end_to_end::{
//!     EndToEndEvaluator, EndToEndConfig, E2EMetric
//! };
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let e2e_config = EndToEndConfig {
//!     metrics: vec![
//!         E2EMetric::OverallAccuracy,
//!         E2EMetric::ResponseTime,
//!         E2EMetric::UserSatisfaction,
//!         E2EMetric::CostEfficiency,
//!     ],
//!     include_ablation_study: true,
//!     ..Default::default()
//! };
//!
//! let e2e_evaluator = EndToEndEvaluator::new(e2e_config);
//!
//! // Comprehensive system evaluation including:
//! // - End-to-end accuracy
//! // - Performance benchmarks
//! // - Resource utilization
//! // - Error analysis
//! // - Component contribution analysis
//!
//! println!("🎆 End-to-end evaluation configured for complete system assessment");
//! # Ok(())
//! # }
//! ```
//!
//! ### Automated Benchmarking
//! ```rust
//! use rrag::evaluation::benchmarks::{
//!     BenchmarkEvaluator, BenchmarkSuite, BenchmarkDataset
//! };
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let benchmark_evaluator = BenchmarkEvaluator::new();
//!
//! let benchmark_suite = BenchmarkSuite {
//!     datasets: vec![
//!         BenchmarkDataset::MS_MARCO,
//!         BenchmarkDataset::Natural_Questions,
//!         BenchmarkDataset::SQuAD_2_0,
//!         BenchmarkDataset::BEIR,
//!     ],
//!     custom_datasets: vec![], // Add domain-specific datasets
//!     evaluation_mode: "comprehensive".to_string(),
//! };
//!
//! // Run against standard benchmarks
//! // let results = benchmark_evaluator.run_benchmark_suite(benchmark_suite).await?;
//!
//! println!("📅 Benchmark evaluation ready with {} standard datasets", 4);
//! # Ok(())
//! # }
//! ```
//!
//! ### Exporting Evaluation Results
//! ```rust
//! use rrag::evaluation::{ExportFormat, OutputConfig};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! # let evaluator = rrag::evaluation::EvaluationService::new(rrag::evaluation::EvaluationConfig::default());
//! # let results = std::collections::HashMap::new(); // Mock results
//! // Configure export options
//! let output_config = OutputConfig {
//!     export_formats: vec![
//!         ExportFormat::Json,    // Machine-readable results
//!         ExportFormat::Html,    // Interactive reports
//!         ExportFormat::Csv,     // Spreadsheet analysis
//!         ExportFormat::Markdown // Documentation
//!     ],
//!     output_dir: "./evaluation_results".to_string(),
//!     include_detailed_logs: true,
//!     generate_visualizations: true,
//! };
//!
//! // Export comprehensive results
//! evaluator.export_results(&results).await?;
//!
//! println!("📊 Results exported in multiple formats:");
//! println!("  • evaluation_results.json - Complete data");
//! println!("  • evaluation_report.html - Interactive dashboard");
//! println!("  • evaluation_summary.csv - Quick analysis");
//! println!("  • evaluation_report.md - Documentation");
//! # Ok(())
//! # }
//! ```
//!
//! ### Real-time Evaluation Monitoring
//! ```rust
//! # async fn example() -> rrag::RragResult<()> {
//! # let evaluator = rrag::evaluation::EvaluationService::new(rrag::evaluation::EvaluationConfig::default());
//! // Monitor evaluation metrics in real-time
//! let metrics = evaluator.get_metrics()?;
//!
//! for (metric_name, records) in metrics {
//!     let latest = records.last().unwrap();
//!     match metric_name.as_str() {
//!         "evaluation_time_ms" => {
//!             if latest.value > 5000.0 {
//!                 println!("⚠️  Evaluation taking longer than expected: {:.1}ms", latest.value);
//!             }
//!         }
//!         "evaluation_errors" => {
//!             if latest.value > 0.0 {
//!                 println!("❌ Evaluation errors detected: {}", latest.value);
//!             }
//!         }
//!         _ => {
//!             println!("📈 {}: {:.3}", metric_name, latest.value);
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Evaluation Best Practices
//!
//! ### Dataset Preparation
//! - Use diverse, representative test queries
//! - Include edge cases and challenging examples
//! - Ensure high-quality ground truth annotations
//! - Balance different query types and complexities
//!
//! ### Metric Selection
//! - Choose metrics aligned with your use case
//! - Combine automatic and human evaluation
//! - Consider both accuracy and efficiency metrics
//! - Include domain-specific evaluation criteria
//!
//! ### Performance Optimization
//! - Run evaluations in batch for efficiency
//! - Use parallel evaluation when possible
//! - Cache expensive computations
//! - Monitor resource usage during evaluation
//!
//! ### Result Interpretation
//! - Consider statistical significance
//! - Analyze results by query type and complexity
//! - Look for systematic errors and patterns
//! - Compare against established baselines
//!
//! ## Integration with RAG Systems
//!
//! ```rust
//! use rrag::{RragSystemBuilder, evaluation::EvaluationConfig};
//!
//! # async fn example() -> rrag::RragResult<()> {
//! let rag_system = RragSystemBuilder::new()
//!     .with_evaluation(
//!         EvaluationConfig::production()
//!             .with_ragas_metrics(true)
//!             .with_real_time_monitoring(true)
//!             .with_automated_benchmarking(true)
//!     )
//!     .build()
//!     .await?;
//!
//! // System automatically evaluates performance and provides insights
//! let results = rag_system.search_with_evaluation("query", Some(10)).await?;
//! # Ok(())
//! # }
//! ```

pub mod benchmarks;
pub mod end_to_end;
pub mod generation_eval;
pub mod metrics;
pub mod ragas;
pub mod retrieval_eval;

use crate::{RragError, RragResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main evaluation service
pub struct EvaluationService {
    /// Configuration
    config: EvaluationConfig,

    /// Evaluators for different components
    evaluators: HashMap<EvaluationType, Box<dyn Evaluator>>,

    /// Metrics collection
    metrics_collector: Box<dyn MetricsCollector>,
}

/// Configuration for evaluation service
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Enabled evaluation types
    pub enabled_evaluations: Vec<EvaluationType>,

    /// RAGAS configuration
    pub ragas_config: ragas::RagasConfig,

    /// Retrieval evaluation config
    pub retrieval_config: retrieval_eval::RetrievalEvalConfig,

    /// Generation evaluation config
    pub generation_config: generation_eval::GenerationEvalConfig,

    /// End-to-end evaluation config
    pub e2e_config: end_to_end::EndToEndConfig,

    /// Output configuration
    pub output_config: OutputConfig,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            enabled_evaluations: vec![
                EvaluationType::Ragas,
                EvaluationType::Retrieval,
                EvaluationType::Generation,
            ],
            ragas_config: ragas::RagasConfig::default(),
            retrieval_config: retrieval_eval::RetrievalEvalConfig::default(),
            generation_config: generation_eval::GenerationEvalConfig::default(),
            e2e_config: end_to_end::EndToEndConfig::default(),
            output_config: OutputConfig::default(),
        }
    }
}

/// Types of evaluation
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationType {
    /// RAGAS metrics evaluation
    Ragas,
    /// Retrieval-specific evaluation
    Retrieval,
    /// Generation-specific evaluation
    Generation,
    /// End-to-end system evaluation
    EndToEnd,
    /// Benchmark evaluation
    Benchmark,
}

/// Output configuration for evaluation results
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Export formats
    pub export_formats: Vec<ExportFormat>,

    /// Output directory
    pub output_dir: String,

    /// Include detailed logs
    pub include_detailed_logs: bool,

    /// Generate visualizations
    pub generate_visualizations: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            export_formats: vec![ExportFormat::Json, ExportFormat::Csv],
            output_dir: "./evaluation_results".to_string(),
            include_detailed_logs: true,
            generate_visualizations: false,
        }
    }
}

/// Export formats for evaluation results
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
    Markdown,
}

/// Main trait for evaluators
pub trait Evaluator: Send + Sync {
    /// Evaluator name
    fn name(&self) -> &str;

    /// Run evaluation
    fn evaluate(&self, evaluation_data: &EvaluationData) -> RragResult<EvaluationResult>;

    /// Get supported metrics
    fn supported_metrics(&self) -> Vec<String>;

    /// Get evaluator configuration
    fn get_config(&self) -> EvaluatorConfig;
}

/// Configuration for individual evaluators
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Evaluator name
    pub name: String,

    /// Version
    pub version: String,

    /// Supported metrics
    pub metrics: Vec<String>,

    /// Performance characteristics
    pub performance: EvaluatorPerformance,
}

/// Performance characteristics of evaluators
#[derive(Debug, Clone)]
pub struct EvaluatorPerformance {
    /// Average evaluation time per sample (ms)
    pub avg_time_per_sample_ms: f32,

    /// Memory usage (MB)
    pub memory_usage_mb: f32,

    /// Accuracy of evaluation
    pub accuracy: f32,
}

/// Input data for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationData {
    /// Test queries
    pub queries: Vec<TestQuery>,

    /// Ground truth data
    pub ground_truth: Vec<GroundTruth>,

    /// System responses
    pub system_responses: Vec<SystemResponse>,

    /// Additional context
    pub context: HashMap<String, serde_json::Value>,
}

/// Test query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestQuery {
    /// Query ID
    pub id: String,

    /// Query text
    pub query: String,

    /// Expected query type/intent
    pub query_type: Option<String>,

    /// Query metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Ground truth data for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    /// Query ID
    pub query_id: String,

    /// Relevant document IDs
    pub relevant_docs: Vec<String>,

    /// Expected answer/response
    pub expected_answer: Option<String>,

    /// Relevance judgments (document_id -> relevance_score)
    pub relevance_judgments: HashMap<String, f32>,

    /// Additional ground truth data
    pub metadata: HashMap<String, serde_json::Value>,
}

/// System response for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResponse {
    /// Query ID
    pub query_id: String,

    /// Retrieved documents
    pub retrieved_docs: Vec<RetrievedDocument>,

    /// Generated answer (if applicable)
    pub generated_answer: Option<String>,

    /// System timing information
    pub timing: SystemTiming,

    /// Response metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Retrieved document information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedDocument {
    /// Document ID
    pub doc_id: String,

    /// Document content
    pub content: String,

    /// Retrieval score
    pub score: f32,

    /// Rank in retrieval results
    pub rank: usize,

    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// System timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTiming {
    /// Total response time (ms)
    pub total_time_ms: f32,

    /// Retrieval time (ms)
    pub retrieval_time_ms: f32,

    /// Generation time (ms)
    pub generation_time_ms: Option<f32>,

    /// Reranking time (ms)
    pub reranking_time_ms: Option<f32>,
}

/// Evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Evaluation ID
    pub id: String,

    /// Evaluation type
    pub evaluation_type: String,

    /// Overall scores
    pub overall_scores: HashMap<String, f32>,

    /// Per-query results
    pub per_query_results: Vec<QueryEvaluationResult>,

    /// Summary statistics
    pub summary: EvaluationSummary,

    /// Evaluation metadata
    pub metadata: EvaluationMetadata,
}

/// Per-query evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvaluationResult {
    /// Query ID
    pub query_id: String,

    /// Metric scores
    pub scores: HashMap<String, f32>,

    /// Error analysis
    pub errors: Vec<EvaluationError>,

    /// Additional details
    pub details: HashMap<String, serde_json::Value>,
}

/// Evaluation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationError {
    /// Error type
    pub error_type: String,

    /// Error message
    pub message: String,

    /// Error severity
    pub severity: ErrorSeverity,

    /// Suggested fixes
    pub suggestions: Vec<String>,
}

/// Error severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Evaluation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    /// Number of queries evaluated
    pub total_queries: usize,

    /// Average scores across all metrics
    pub avg_scores: HashMap<String, f32>,

    /// Standard deviations
    pub std_deviations: HashMap<String, f32>,

    /// Performance statistics
    pub performance_stats: PerformanceStats,

    /// Key insights
    pub insights: Vec<String>,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Average evaluation time per query
    pub avg_eval_time_ms: f32,

    /// Total evaluation time
    pub total_eval_time_ms: f32,

    /// Memory usage during evaluation
    pub peak_memory_usage_mb: f32,

    /// Throughput (queries per second)
    pub throughput_qps: f32,
}

/// Evaluation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetadata {
    /// Evaluation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Evaluation version
    pub evaluation_version: String,

    /// System configuration
    pub system_config: HashMap<String, serde_json::Value>,

    /// Environment information
    pub environment: HashMap<String, String>,

    /// Git commit hash (if available)
    pub git_commit: Option<String>,
}

/// Trait for collecting metrics during evaluation
pub trait MetricsCollector: Send + Sync {
    /// Start collecting metrics
    fn start_collection(&mut self) -> RragResult<()>;

    /// Stop collecting metrics
    fn stop_collection(&mut self) -> RragResult<()>;

    /// Record a metric
    fn record_metric(
        &mut self,
        name: &str,
        value: f32,
        labels: Option<&HashMap<String, String>>,
    ) -> RragResult<()>;

    /// Get collected metrics
    fn get_metrics(&self) -> RragResult<HashMap<String, Vec<MetricRecord>>>;

    /// Export metrics to file
    fn export_metrics(&self, format: &ExportFormat, output_path: &str) -> RragResult<()>;
}

/// Individual metric record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Metric name
    pub name: String,

    /// Metric value
    pub value: f32,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Labels/tags
    pub labels: HashMap<String, String>,
}

impl EvaluationService {
    /// Create new evaluation service
    pub fn new(config: EvaluationConfig) -> Self {
        let mut service = Self {
            config: config.clone(),
            evaluators: HashMap::new(),
            metrics_collector: Box::new(DefaultMetricsCollector::new()),
        };

        // Initialize evaluators
        service.initialize_evaluators();

        service
    }

    /// Initialize evaluators based on configuration
    fn initialize_evaluators(&mut self) {
        for eval_type in &self.config.enabled_evaluations {
            let evaluator: Box<dyn Evaluator> = match eval_type {
                EvaluationType::Ragas => {
                    Box::new(ragas::RagasEvaluator::new(self.config.ragas_config.clone()))
                }
                EvaluationType::Retrieval => Box::new(retrieval_eval::RetrievalEvaluator::new(
                    self.config.retrieval_config.clone(),
                )),
                EvaluationType::Generation => Box::new(generation_eval::GenerationEvaluator::new(
                    self.config.generation_config.clone(),
                )),
                EvaluationType::EndToEnd => Box::new(end_to_end::EndToEndEvaluator::new(
                    self.config.e2e_config.clone(),
                )),
                EvaluationType::Benchmark => Box::new(benchmarks::BenchmarkEvaluator::new()),
            };

            self.evaluators.insert(eval_type.clone(), evaluator);
        }
    }

    /// Run evaluation on provided data
    pub async fn evaluate(
        &mut self,
        data: EvaluationData,
    ) -> RragResult<HashMap<EvaluationType, EvaluationResult>> {
        let mut results = HashMap::new();

        // Start metrics collection
        self.metrics_collector.start_collection()?;

        let start_time = std::time::Instant::now();

        // Run each enabled evaluation
        for (eval_type, evaluator) in &self.evaluators {
            println!("Running {} evaluation...", evaluator.name());

            let eval_start = std::time::Instant::now();

            match evaluator.evaluate(&data) {
                Ok(result) => {
                    let eval_time = eval_start.elapsed().as_millis() as f32;
                    self.metrics_collector.record_metric(
                        "evaluation_time_ms",
                        eval_time,
                        Some(
                            &[("evaluator".to_string(), evaluator.name().to_string())]
                                .iter()
                                .cloned()
                                .collect(),
                        ),
                    )?;

                    results.insert(eval_type.clone(), result);
                    println!(
                        "✅ {} evaluation completed in {:.2}ms",
                        evaluator.name(),
                        eval_time
                    );
                }
                Err(e) => {
                    eprintln!("❌ {} evaluation failed: {}", evaluator.name(), e);
                    self.metrics_collector.record_metric(
                        "evaluation_errors",
                        1.0,
                        Some(
                            &[("evaluator".to_string(), evaluator.name().to_string())]
                                .iter()
                                .cloned()
                                .collect(),
                        ),
                    )?;
                }
            }
        }

        let total_time = start_time.elapsed().as_millis() as f32;
        self.metrics_collector
            .record_metric("total_evaluation_time_ms", total_time, None)?;

        // Stop metrics collection
        self.metrics_collector.stop_collection()?;

        Ok(results)
    }

    /// Export evaluation results
    pub async fn export_results(
        &self,
        results: &HashMap<EvaluationType, EvaluationResult>,
    ) -> RragResult<()> {
        // Create output directory
        std::fs::create_dir_all(&self.config.output_config.output_dir).map_err(|e| {
            RragError::evaluation(format!("Failed to create output directory: {}", e))
        })?;

        for format in &self.config.output_config.export_formats {
            match format {
                ExportFormat::Json => self.export_json(results).await?,
                ExportFormat::Csv => self.export_csv(results).await?,
                ExportFormat::Html => self.export_html(results).await?,
                ExportFormat::Markdown => self.export_markdown(results).await?,
            }
        }

        Ok(())
    }

    /// Export results as JSON
    async fn export_json(
        &self,
        results: &HashMap<EvaluationType, EvaluationResult>,
    ) -> RragResult<()> {
        let json_path = format!(
            "{}/evaluation_results.json",
            self.config.output_config.output_dir
        );
        let json_content = serde_json::to_string_pretty(results)
            .map_err(|e| RragError::evaluation(format!("Failed to serialize results: {}", e)))?;

        std::fs::write(&json_path, json_content)
            .map_err(|e| RragError::evaluation(format!("Failed to write JSON file: {}", e)))?;

        println!("✅ Results exported to {}", json_path);
        Ok(())
    }

    /// Export results as CSV
    async fn export_csv(
        &self,
        results: &HashMap<EvaluationType, EvaluationResult>,
    ) -> RragResult<()> {
        let csv_path = format!(
            "{}/evaluation_summary.csv",
            self.config.output_config.output_dir
        );
        let mut csv_content = String::new();

        // Header
        csv_content.push_str("evaluator,metric,value\n");

        // Data
        for (eval_type, result) in results {
            for (metric, value) in &result.overall_scores {
                csv_content.push_str(&format!("{:?},{},{}\n", eval_type, metric, value));
            }
        }

        std::fs::write(&csv_path, csv_content)
            .map_err(|e| RragError::evaluation(format!("Failed to write CSV file: {}", e)))?;

        println!("✅ Summary exported to {}", csv_path);
        Ok(())
    }

    /// Export results as HTML
    async fn export_html(
        &self,
        results: &HashMap<EvaluationType, EvaluationResult>,
    ) -> RragResult<()> {
        let html_path = format!(
            "{}/evaluation_report.html",
            self.config.output_config.output_dir
        );
        let mut html_content = String::from(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>RRAG Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #333; margin-bottom: 30px; }
        .evaluator { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; }
        .metric { margin: 10px 0; }
        .score { font-weight: bold; color: #2196F3; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 RRAG Evaluation Report</h1>
        <p>Generated on: "#,
        );

        html_content.push_str(
            &chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
        );
        html_content.push_str("</p>\n    </div>\n");

        for (eval_type, result) in results {
            html_content.push_str(&format!(
                r#"
    <div class="evaluator">
        <h2>📊 {:?} Evaluation</h2>
        <h3>Overall Scores</h3>
        <table>
            <tr><th>Metric</th><th>Score</th></tr>"#,
                eval_type
            ));

            for (metric, score) in &result.overall_scores {
                html_content.push_str(&format!(
                    "<tr><td>{}</td><td class=\"score\">{:.4}</td></tr>",
                    metric, score
                ));
            }

            html_content.push_str("</table>\n");

            if !result.summary.insights.is_empty() {
                html_content.push_str("<h3>Key Insights</h3><ul>");
                for insight in &result.summary.insights {
                    html_content.push_str(&format!("<li>{}</li>", insight));
                }
                html_content.push_str("</ul>");
            }

            html_content.push_str("    </div>\n");
        }

        html_content.push_str("</body>\n</html>");

        std::fs::write(&html_path, html_content)
            .map_err(|e| RragError::evaluation(format!("Failed to write HTML file: {}", e)))?;

        println!("✅ Report exported to {}", html_path);
        Ok(())
    }

    /// Export results as Markdown
    async fn export_markdown(
        &self,
        results: &HashMap<EvaluationType, EvaluationResult>,
    ) -> RragResult<()> {
        let md_path = format!(
            "{}/evaluation_report.md",
            self.config.output_config.output_dir
        );
        let mut md_content = String::from("# 🎯 RRAG Evaluation Report\n\n");

        md_content.push_str(&format!(
            "**Generated on:** {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        for (eval_type, result) in results {
            md_content.push_str(&format!("## 📊 {:?} Evaluation\n\n", eval_type));

            md_content.push_str("### Overall Scores\n\n");
            md_content.push_str("| Metric | Score |\n|--------|-------|\n");

            for (metric, score) in &result.overall_scores {
                md_content.push_str(&format!("| {} | {:.4} |\n", metric, score));
            }

            if !result.summary.insights.is_empty() {
                md_content.push_str("\n### Key Insights\n\n");
                for insight in &result.summary.insights {
                    md_content.push_str(&format!("- {}\n", insight));
                }
            }

            if !result.summary.recommendations.is_empty() {
                md_content.push_str("\n### Recommendations\n\n");
                for recommendation in &result.summary.recommendations {
                    md_content.push_str(&format!("- {}\n", recommendation));
                }
            }

            md_content.push_str("\n---\n\n");
        }

        std::fs::write(&md_path, md_content)
            .map_err(|e| RragError::evaluation(format!("Failed to write Markdown file: {}", e)))?;

        println!("✅ Markdown report exported to {}", md_path);
        Ok(())
    }

    /// Get evaluation metrics
    pub fn get_metrics(&self) -> RragResult<HashMap<String, Vec<MetricRecord>>> {
        self.metrics_collector.get_metrics()
    }
}

/// Default metrics collector implementation
pub struct DefaultMetricsCollector {
    metrics: HashMap<String, Vec<MetricRecord>>,
    collecting: bool,
}

impl DefaultMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            collecting: false,
        }
    }
}

impl MetricsCollector for DefaultMetricsCollector {
    fn start_collection(&mut self) -> RragResult<()> {
        self.collecting = true;
        self.metrics.clear();
        Ok(())
    }

    fn stop_collection(&mut self) -> RragResult<()> {
        self.collecting = false;
        Ok(())
    }

    fn record_metric(
        &mut self,
        name: &str,
        value: f32,
        labels: Option<&HashMap<String, String>>,
    ) -> RragResult<()> {
        if !self.collecting {
            return Ok(());
        }

        let record = MetricRecord {
            name: name.to_string(),
            value,
            timestamp: chrono::Utc::now(),
            labels: labels.cloned().unwrap_or_default(),
        };

        self.metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(record);
        Ok(())
    }

    fn get_metrics(&self) -> RragResult<HashMap<String, Vec<MetricRecord>>> {
        Ok(self.metrics.clone())
    }

    fn export_metrics(&self, format: &ExportFormat, output_path: &str) -> RragResult<()> {
        match format {
            ExportFormat::Json => {
                let json_content = serde_json::to_string_pretty(&self.metrics).map_err(|e| {
                    RragError::evaluation(format!("Failed to serialize metrics: {}", e))
                })?;
                std::fs::write(output_path, json_content).map_err(|e| {
                    RragError::evaluation(format!("Failed to write metrics file: {}", e))
                })?;
            }
            _ => {
                return Err(RragError::evaluation(
                    "Unsupported export format for metrics".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_config_creation() {
        let config = EvaluationConfig::default();
        assert!(config.enabled_evaluations.contains(&EvaluationType::Ragas));
        assert!(config
            .enabled_evaluations
            .contains(&EvaluationType::Retrieval));
        assert!(config
            .enabled_evaluations
            .contains(&EvaluationType::Generation));
    }

    #[test]
    fn test_evaluation_data_creation() {
        let query = TestQuery {
            id: "test_1".to_string(),
            query: "What is machine learning?".to_string(),
            query_type: Some("factual".to_string()),
            metadata: HashMap::new(),
        };

        let ground_truth = GroundTruth {
            query_id: "test_1".to_string(),
            relevant_docs: vec!["doc_1".to_string(), "doc_2".to_string()],
            expected_answer: Some("Machine learning is...".to_string()),
            relevance_judgments: HashMap::new(),
            metadata: HashMap::new(),
        };

        let data = EvaluationData {
            queries: vec![query],
            ground_truth: vec![ground_truth],
            system_responses: vec![],
            context: HashMap::new(),
        };

        assert_eq!(data.queries.len(), 1);
        assert_eq!(data.ground_truth.len(), 1);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = DefaultMetricsCollector::new();

        collector.start_collection().unwrap();
        collector.record_metric("test_metric", 0.85, None).unwrap();
        collector.stop_collection().unwrap();

        let metrics = collector.get_metrics().unwrap();
        assert!(metrics.contains_key("test_metric"));
        assert_eq!(metrics["test_metric"].len(), 1);
        assert_eq!(metrics["test_metric"][0].value, 0.85);
    }
}
