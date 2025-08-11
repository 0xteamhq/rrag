//! # Batch Processing System
//! 
//! Optimized batch processing for large-scale incremental updates.
//! Handles queue management, error handling, and performance optimization.

use crate::{RragError, RragResult};
use crate::incremental::index_manager::{IndexUpdate, UpdateResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, Semaphore};
use uuid::Uuid;

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Minimum batch size before processing
    pub min_batch_size: usize,
    
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    
    /// Maximum concurrent batches
    pub max_concurrent_batches: usize,
    
    /// Enable priority-based batching
    pub enable_priority_batching: bool,
    
    /// Enable adaptive batch sizing
    pub enable_adaptive_sizing: bool,
    
    /// Error handling strategy
    pub error_handling: ErrorHandlingStrategy,
    
    /// Retry configuration
    pub retry_config: RetryConfig,
    
    /// Performance optimization settings
    pub optimization: BatchOptimizationConfig,
}

/// Error handling strategies for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingStrategy {
    /// Fail entire batch on first error
    FailFast,
    /// Continue processing despite individual failures
    ContinueOnError,
    /// Isolate failed items and retry separately
    IsolateAndRetry,
    /// Use circuit breaker pattern
    CircuitBreaker,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: u32,
    
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Maximum delay between retries
    pub max_delay_ms: u64,
    
    /// Jitter factor (0.0 to 1.0)
    pub jitter_factor: f64,
}

/// Batch optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptimizationConfig {
    /// Enable batch deduplication
    pub enable_deduplication: bool,
    
    /// Enable operation reordering
    pub enable_reordering: bool,
    
    /// Enable batch compression
    pub enable_compression: bool,
    
    /// Memory pool size for batching
    pub memory_pool_size: usize,
    
    /// Enable parallel processing within batches
    pub enable_parallel_processing: bool,
    
    /// Target processing time per batch in milliseconds
    pub target_processing_time_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            min_batch_size: 10,
            batch_timeout_ms: 5000,
            max_concurrent_batches: 5,
            enable_priority_batching: true,
            enable_adaptive_sizing: true,
            error_handling: ErrorHandlingStrategy::ContinueOnError,
            retry_config: RetryConfig::default(),
            optimization: BatchOptimizationConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
            jitter_factor: 0.1,
        }
    }
}

impl Default for BatchOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_deduplication: true,
            enable_reordering: true,
            enable_compression: false,
            memory_pool_size: 1024 * 1024 * 50, // 50MB
            enable_parallel_processing: true,
            target_processing_time_ms: 10000, // 10 seconds
        }
    }
}

/// Batch operation container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOperation {
    /// Batch ID
    pub batch_id: String,
    
    /// Operations in this batch
    pub operations: Vec<IndexUpdate>,
    
    /// Batch priority (derived from operations)
    pub priority: u8,
    
    /// Batch creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Expected processing time estimate
    pub estimated_processing_time_ms: u64,
    
    /// Batch metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result of batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Batch ID
    pub batch_id: String,
    
    /// Overall success status
    pub success: bool,
    
    /// Individual operation results
    pub operation_results: Vec<UpdateResult>,
    
    /// Total processing time
    pub processing_time_ms: u64,
    
    /// Number of successful operations
    pub successful_operations: usize,
    
    /// Number of failed operations
    pub failed_operations: usize,
    
    /// Batch-level errors
    pub batch_errors: Vec<String>,
    
    /// Performance statistics
    pub stats: BatchProcessingStats,
    
    /// Retry information
    pub retry_info: Option<RetryInfo>,
}

/// Processing statistics for a batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingStats {
    /// Queue wait time
    pub queue_wait_time_ms: u64,
    
    /// Actual processing time
    pub processing_time_ms: u64,
    
    /// Memory usage during processing
    pub peak_memory_usage_mb: f64,
    
    /// CPU utilization during processing
    pub cpu_utilization_percent: f64,
    
    /// Throughput (operations per second)
    pub throughput_ops_per_second: f64,
    
    /// Optimization metrics
    pub optimizations_applied: Vec<String>,
}

/// Retry information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryInfo {
    /// Current retry attempt
    pub attempt: u32,
    
    /// Maximum retries allowed
    pub max_retries: u32,
    
    /// Next retry time
    pub next_retry_at: chrono::DateTime<chrono::Utc>,
    
    /// Retry reason
    pub retry_reason: String,
    
    /// Failed operations to retry
    pub failed_operations: Vec<String>,
}

/// Queue management system
pub struct QueueManager {
    /// High priority queue
    high_priority_queue: Arc<Mutex<VecDeque<BatchOperation>>>,
    
    /// Normal priority queue
    normal_priority_queue: Arc<Mutex<VecDeque<BatchOperation>>>,
    
    /// Low priority queue
    low_priority_queue: Arc<Mutex<VecDeque<BatchOperation>>>,
    
    /// Retry queue
    retry_queue: Arc<Mutex<VecDeque<(BatchOperation, RetryInfo)>>>,
    
    /// Queue statistics
    stats: Arc<RwLock<QueueStats>>,
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    /// Queue sizes by priority
    pub queue_sizes: HashMap<String, usize>,
    
    /// Average wait times
    pub average_wait_times_ms: HashMap<String, f64>,
    
    /// Total items processed
    pub total_processed: u64,
    
    /// Current throughput
    pub current_throughput: f64,
    
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Batch executor for processing operations
pub struct BatchExecutor {
    /// Configuration
    config: BatchOptimizationConfig,
    
    /// Concurrency control
    semaphore: Arc<Semaphore>,
    
    /// Processing statistics
    stats: Arc<RwLock<ExecutorStats>>,
}

/// Executor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorStats {
    /// Total batches processed
    pub batches_processed: u64,
    
    /// Average processing time
    pub avg_processing_time_ms: f64,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Current active batches
    pub active_batches: usize,
    
    /// Peak concurrent batches
    pub peak_concurrent_batches: usize,
    
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Main batch processor
pub struct BatchProcessor {
    /// Configuration
    config: BatchConfig,
    
    /// Queue manager
    queue_manager: Arc<QueueManager>,
    
    /// Batch executor
    executor: Arc<BatchExecutor>,
    
    /// Current batches being assembled
    current_batches: Arc<RwLock<HashMap<String, Vec<IndexUpdate>>>>,
    
    /// Batch timers
    batch_timers: Arc<RwLock<HashMap<String, tokio::time::Instant>>>,
    
    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<ProcessingMetrics>>,
}

/// Overall processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total operations processed
    pub total_operations: u64,
    
    /// Total batches processed
    pub total_batches: u64,
    
    /// Average batch size
    pub avg_batch_size: f64,
    
    /// Overall throughput
    pub throughput_ops_per_second: f64,
    
    /// Error rates
    pub error_rate: f64,
    
    /// Retry statistics
    pub retry_stats: RetryStats,
    
    /// Performance trends
    pub performance_trends: Vec<PerformanceDataPoint>,
    
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Retry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryStats {
    /// Total retries attempted
    pub total_retries: u64,
    
    /// Successful retries
    pub successful_retries: u64,
    
    /// Failed retries (exhausted)
    pub failed_retries: u64,
    
    /// Average retry attempts per operation
    pub avg_retry_attempts: f64,
}

/// Performance data point for trending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Throughput at this point
    pub throughput: f64,
    
    /// Queue depth at this point
    pub queue_depth: usize,
    
    /// Error rate at this point
    pub error_rate: f64,
    
    /// Average processing time
    pub avg_processing_time_ms: f64,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub async fn new(config: BatchConfig) -> RragResult<Self> {
        let queue_manager = Arc::new(QueueManager {
            high_priority_queue: Arc::new(Mutex::new(VecDeque::new())),
            normal_priority_queue: Arc::new(Mutex::new(VecDeque::new())),
            low_priority_queue: Arc::new(Mutex::new(VecDeque::new())),
            retry_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(QueueStats {
                queue_sizes: HashMap::new(),
                average_wait_times_ms: HashMap::new(),
                total_processed: 0,
                current_throughput: 0.0,
                last_updated: chrono::Utc::now(),
            })),
        });

        let executor = Arc::new(BatchExecutor {
            config: config.optimization.clone(),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_batches)),
            stats: Arc::new(RwLock::new(ExecutorStats {
                batches_processed: 0,
                avg_processing_time_ms: 0.0,
                success_rate: 0.0,
                active_batches: 0,
                peak_concurrent_batches: 0,
                last_updated: chrono::Utc::now(),
            })),
        });

        let processor = Self {
            config,
            queue_manager,
            executor,
            current_batches: Arc::new(RwLock::new(HashMap::new())),
            batch_timers: Arc::new(RwLock::new(HashMap::new())),
            task_handles: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(RwLock::new(ProcessingMetrics {
                total_operations: 0,
                total_batches: 0,
                avg_batch_size: 0.0,
                throughput_ops_per_second: 0.0,
                error_rate: 0.0,
                retry_stats: RetryStats {
                    total_retries: 0,
                    successful_retries: 0,
                    failed_retries: 0,
                    avg_retry_attempts: 0.0,
                },
                performance_trends: Vec::new(),
                last_updated: chrono::Utc::now(),
            })),
        };

        processor.start_background_tasks().await?;
        Ok(processor)
    }

    /// Add operation to batch processing queue
    pub async fn add_operation(&self, operation: IndexUpdate) -> RragResult<String> {
        let batch_key = self.determine_batch_key(&operation).await?;
        
        // Add to current batch
        {
            let mut current_batches = self.current_batches.write().await;
            let batch = current_batches.entry(batch_key.clone()).or_insert_with(Vec::new);
            batch.push(operation);

            // Start timer if this is the first operation in the batch
            if batch.len() == 1 {
                let mut timers = self.batch_timers.write().await;
                timers.insert(batch_key.clone(), tokio::time::Instant::now());
            }

            // Check if batch is ready for processing
            if batch.len() >= self.config.max_batch_size {
                let operations = std::mem::take(batch);
                drop(current_batches);
                
                // Remove timer
                let mut timers = self.batch_timers.write().await;
                timers.remove(&batch_key);
                drop(timers);
                
                // Create and queue batch
                self.create_and_queue_batch(operations).await?;
            }
        }

        Ok(batch_key)
    }

    /// Process a batch of operations
    pub async fn process_batch(&self, batch: BatchOperation) -> RragResult<BatchResult> {
        let _permit = self.executor.semaphore.acquire().await
            .map_err(|e| RragError::timeout("acquire_semaphore", 30000))?;

        let start_time = std::time::Instant::now();
        let queue_wait_time = start_time.elapsed();

        // Update executor stats
        {
            let mut stats = self.executor.stats.write().await;
            stats.active_batches += 1;
            stats.peak_concurrent_batches = 
                std::cmp::max(stats.peak_concurrent_batches, stats.active_batches);
        }

        // Apply optimizations
        let optimized_operations = self.optimize_batch(&batch.operations).await?;
        
        // Process operations
        let mut operation_results = Vec::new();
        let mut successful_operations = 0;
        let mut failed_operations = 0;
        let mut batch_errors = Vec::new();

        for operation in optimized_operations {
            match self.process_single_operation(&operation).await {
                Ok(result) => {
                    if result.success {
                        successful_operations += 1;
                    } else {
                        failed_operations += 1;
                    }
                    operation_results.push(result);
                }
                Err(e) => {
                    failed_operations += 1;
                    batch_errors.push(e.to_string());
                    
                    // Create error result
                    operation_results.push(UpdateResult {
                        operation_id: operation.operation_id.clone(),
                        success: false,
                        operations_completed: Vec::new(),
                        conflicts: Vec::new(),
                        processing_time_ms: 0,
                        items_affected: 0,
                        error: Some(e.to_string()),
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        let processing_time = start_time.elapsed();
        let success = match self.config.error_handling {
            ErrorHandlingStrategy::FailFast => failed_operations == 0,
            ErrorHandlingStrategy::ContinueOnError => successful_operations > 0,
            ErrorHandlingStrategy::IsolateAndRetry => true, // Always succeed, handle retries separately
            ErrorHandlingStrategy::CircuitBreaker => failed_operations < successful_operations,
        };

        // Update executor stats
        {
            let mut stats = self.executor.stats.write().await;
            stats.active_batches -= 1;
            stats.batches_processed += 1;
            stats.avg_processing_time_ms = 
                (stats.avg_processing_time_ms + processing_time.as_millis() as f64) / 2.0;
            stats.success_rate = if stats.batches_processed > 0 {
                // Simplified success rate calculation
                successful_operations as f64 / (successful_operations + failed_operations) as f64
            } else {
                0.0
            };
            stats.last_updated = chrono::Utc::now();
        }

        // Create batch result
        let result = BatchResult {
            batch_id: batch.batch_id,
            success,
            operation_results,
            processing_time_ms: processing_time.as_millis() as u64,
            successful_operations,
            failed_operations,
            batch_errors,
            stats: BatchProcessingStats {
                queue_wait_time_ms: queue_wait_time.as_millis() as u64,
                processing_time_ms: processing_time.as_millis() as u64,
                peak_memory_usage_mb: 0.0, // Would be measured in production
                cpu_utilization_percent: 0.0, // Would be measured in production
                throughput_ops_per_second: successful_operations as f64 / processing_time.as_secs_f64(),
                optimizations_applied: vec!["deduplication".to_string()], // Track applied optimizations
            },
            retry_info: None,
        };

        // Update overall metrics
        self.update_metrics(&result).await?;

        Ok(result)
    }

    /// Get current processing metrics
    pub async fn get_metrics(&self) -> ProcessingMetrics {
        self.metrics.read().await.clone()
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        self.queue_manager.stats.read().await.clone()
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        // Check if background tasks are running
        let handles = self.task_handles.lock().await;
        let all_running = handles.iter().all(|handle| !handle.is_finished());
        
        // Check queue health
        let queue_stats = self.get_queue_stats().await;
        let total_queue_size: usize = queue_stats.queue_sizes.values().sum();
        let queue_healthy = total_queue_size < self.config.max_batch_size * 10; // Arbitrary health threshold
        
        Ok(all_running && queue_healthy)
    }

    /// Start background processing tasks
    async fn start_background_tasks(&self) -> RragResult<()> {
        let mut handles = self.task_handles.lock().await;
        
        // Batch formation task
        handles.push(self.start_batch_formation_task().await);
        
        // Batch processing task
        handles.push(self.start_batch_processing_task().await);
        
        // Timeout monitoring task
        handles.push(self.start_timeout_monitoring_task().await);
        
        // Metrics collection task
        handles.push(self.start_metrics_collection_task().await);
        
        Ok(())
    }

    /// Start batch formation monitoring task
    async fn start_batch_formation_task(&self) -> tokio::task::JoinHandle<()> {
        let current_batches = Arc::clone(&self.current_batches);
        let batch_timers = Arc::clone(&self.batch_timers);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_millis(config.batch_timeout_ms / 4)
            );

            loop {
                interval.tick().await;
                
                // Check for batches that should be processed due to timeout
                let mut batches_to_process = Vec::new();
                
                {
                    let current_batches_read = current_batches.read().await;
                    let timers = batch_timers.read().await;
                    
                    for (batch_key, timer) in timers.iter() {
                        if timer.elapsed().as_millis() as u64 >= config.batch_timeout_ms {
                            if let Some(operations) = current_batches_read.get(batch_key) {
                                if operations.len() >= config.min_batch_size {
                                    batches_to_process.push(batch_key.clone());
                                }
                            }
                        }
                    }
                }
                
                // Process timeout batches
                for batch_key in batches_to_process {
                    let operations = {
                        let mut current_batches_write = current_batches.write().await;
                        current_batches_write.remove(&batch_key).unwrap_or_default()
                    };
                    
                    {
                        let mut timers = batch_timers.write().await;
                        timers.remove(&batch_key);
                    }
                    
                    if !operations.is_empty() {
                        // Create and queue batch (would need access to self here)
                        // In a real implementation, this would be handled differently
                    }
                }
            }
        })
    }

    /// Start batch processing task
    async fn start_batch_processing_task(&self) -> tokio::task::JoinHandle<()> {
        let queue_manager = Arc::clone(&self.queue_manager);
        let executor = Arc::clone(&self.executor);

        tokio::spawn(async move {
            loop {
                // Try to get next batch from queues (priority order)
                let batch = {
                    // High priority first
                    let mut high_queue = queue_manager.high_priority_queue.lock().await;
                    if let Some(batch) = high_queue.pop_front() {
                        Some(batch)
                    } else {
                        drop(high_queue);
                        
                        // Normal priority next
                        let mut normal_queue = queue_manager.normal_priority_queue.lock().await;
                        if let Some(batch) = normal_queue.pop_front() {
                            Some(batch)
                        } else {
                            drop(normal_queue);
                            
                            // Low priority last
                            let mut low_queue = queue_manager.low_priority_queue.lock().await;
                            low_queue.pop_front()
                        }
                    }
                };

                if let Some(batch) = batch {
                    // Process batch (simplified - would need full context)
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                } else {
                    // No batches to process, sleep briefly
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        })
    }

    /// Start timeout monitoring task
    async fn start_timeout_monitoring_task(&self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                // Monitor for stuck operations, cleanup expired data, etc.
                // Implementation would depend on specific requirements
            }
        })
    }

    /// Start metrics collection task
    async fn start_metrics_collection_task(&self) -> tokio::task::JoinHandle<()> {
        let metrics = Arc::clone(&self.metrics);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Collect performance data point
                let mut metrics_guard = metrics.write().await;
                let data_point = PerformanceDataPoint {
                    timestamp: chrono::Utc::now(),
                    throughput: metrics_guard.throughput_ops_per_second,
                    queue_depth: 0, // Would get actual queue depth
                    error_rate: metrics_guard.error_rate,
                    avg_processing_time_ms: 0.0, // Would calculate from recent batches
                };
                
                metrics_guard.performance_trends.push(data_point);
                
                // Limit trend history
                if metrics_guard.performance_trends.len() > 1000 {
                    metrics_guard.performance_trends.remove(0);
                }
                
                metrics_guard.last_updated = chrono::Utc::now();
            }
        })
    }

    /// Determine batch key for operation grouping
    async fn determine_batch_key(&self, operation: &IndexUpdate) -> RragResult<String> {
        if self.config.enable_priority_batching {
            Ok(format!("priority_{}", operation.priority))
        } else {
            Ok("default".to_string())
        }
    }

    /// Create and queue a batch
    async fn create_and_queue_batch(&self, operations: Vec<IndexUpdate>) -> RragResult<()> {
        let batch_id = Uuid::new_v4().to_string();
        let priority = operations.iter().map(|op| op.priority).max().unwrap_or(5);
        
        let batch = BatchOperation {
            batch_id,
            operations,
            priority,
            created_at: chrono::Utc::now(),
            estimated_processing_time_ms: 1000, // Would estimate based on operations
            metadata: HashMap::new(),
        };

        // Add to appropriate priority queue
        match priority {
            8..=10 => {
                let mut queue = self.queue_manager.high_priority_queue.lock().await;
                queue.push_back(batch);
            }
            4..=7 => {
                let mut queue = self.queue_manager.normal_priority_queue.lock().await;
                queue.push_back(batch);
            }
            _ => {
                let mut queue = self.queue_manager.low_priority_queue.lock().await;
                queue.push_back(batch);
            }
        }

        Ok(())
    }

    /// Apply optimizations to a batch
    async fn optimize_batch(&self, operations: &[IndexUpdate]) -> RragResult<Vec<IndexUpdate>> {
        let mut optimized = operations.to_vec();

        // Deduplication
        if self.config.optimization.enable_deduplication {
            optimized = self.deduplicate_operations(optimized).await?;
        }

        // Reordering
        if self.config.optimization.enable_reordering {
            optimized = self.reorder_operations(optimized).await?;
        }

        Ok(optimized)
    }

    /// Remove duplicate operations
    async fn deduplicate_operations(&self, operations: Vec<IndexUpdate>) -> RragResult<Vec<IndexUpdate>> {
        let mut seen_documents = std::collections::HashSet::new();
        let mut deduplicated = Vec::new();

        for operation in operations {
            // Simple deduplication based on document ID
            let document_id = match &operation.operation {
                crate::incremental::index_manager::IndexOperation::Add { document, .. } => Some(&document.id),
                crate::incremental::index_manager::IndexOperation::Update { document_id, .. } => Some(document_id),
                crate::incremental::index_manager::IndexOperation::Delete { document_id } => Some(document_id),
                _ => None,
            };

            if let Some(doc_id) = document_id {
                if !seen_documents.contains(doc_id) {
                    seen_documents.insert(doc_id.clone());
                    deduplicated.push(operation);
                }
                // Skip duplicate operations for the same document
            } else {
                // Keep operations that don't have document IDs
                deduplicated.push(operation);
            }
        }

        Ok(deduplicated)
    }

    /// Reorder operations for optimal processing
    async fn reorder_operations(&self, mut operations: Vec<IndexUpdate>) -> RragResult<Vec<IndexUpdate>> {
        // Sort by priority (descending) and then by timestamp (ascending)
        operations.sort_by(|a, b| {
            b.priority.cmp(&a.priority)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
        });

        Ok(operations)
    }

    /// Process a single operation (placeholder)
    async fn process_single_operation(&self, operation: &IndexUpdate) -> RragResult<UpdateResult> {
        // This is a placeholder - in production, this would call the actual index manager
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        Ok(UpdateResult {
            operation_id: operation.operation_id.clone(),
            success: true,
            operations_completed: vec!["processed".to_string()],
            conflicts: Vec::new(),
            processing_time_ms: 10,
            items_affected: 1,
            error: None,
            metadata: HashMap::new(),
        })
    }

    /// Update overall metrics
    async fn update_metrics(&self, result: &BatchResult) -> RragResult<()> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_batches += 1;
        metrics.total_operations += result.operation_results.len() as u64;
        
        if metrics.total_batches > 0 {
            metrics.avg_batch_size = metrics.total_operations as f64 / metrics.total_batches as f64;
        }
        
        // Update throughput
        metrics.throughput_ops_per_second = result.stats.throughput_ops_per_second;
        
        // Update error rate
        if metrics.total_operations > 0 {
            metrics.error_rate = result.failed_operations as f64 / metrics.total_operations as f64;
        }
        
        metrics.last_updated = chrono::Utc::now();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incremental::index_manager::IndexOperation;
    use crate::Document;

    #[tokio::test]
    async fn test_batch_processor_creation() {
        let config = BatchConfig::default();
        let processor = BatchProcessor::new(config).await.unwrap();
        assert!(processor.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_add_operation_to_batch() {
        let processor = BatchProcessor::new(BatchConfig::default()).await.unwrap();
        
        let doc = Document::new("Test content");
        let operation = IndexOperation::Add {
            document: doc,
            chunks: Vec::new(),
            embeddings: Vec::new(),
        };
        
        let update = IndexUpdate {
            operation_id: Uuid::new_v4().to_string(),
            operation,
            priority: 5,
            timestamp: chrono::Utc::now(),
            source: "test".to_string(),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            max_retries: 3,
            retry_count: 0,
        };
        
        let batch_key = processor.add_operation(update).await.unwrap();
        assert!(!batch_key.is_empty());
    }

    #[tokio::test]
    async fn test_batch_optimization() {
        let processor = BatchProcessor::new(BatchConfig::default()).await.unwrap();
        
        // Create operations with same document ID (should be deduplicated)
        let mut operations = Vec::new();
        for i in 0..3 {
            let doc = Document::with_id("same_doc", format!("Content {}", i));
            let operation = IndexOperation::Update {
                document_id: "same_doc".to_string(),
                document: doc,
                chunks: Vec::new(),
                embeddings: Vec::new(),
                change_result: crate::incremental::change_detection::ChangeResult {
                    change_type: crate::incremental::change_detection::ChangeType::ContentChanged,
                    document_id: "same_doc".to_string(),
                    previous_hash: None,
                    current_hash: format!("hash_{}", i),
                    delta: crate::incremental::change_detection::ContentDelta {
                        added_chars: 10,
                        removed_chars: 0,
                        modified_chars: 5,
                        previous_size: 10,
                        current_size: 20,
                        change_percentage: 0.5,
                    },
                    metadata_changes: crate::incremental::change_detection::MetadataChanges {
                        added_keys: Vec::new(),
                        removed_keys: Vec::new(),
                        modified_keys: Vec::new(),
                        previous_metadata: HashMap::new(),
                        current_metadata: HashMap::new(),
                    },
                    timestamps: crate::incremental::change_detection::ChangeTimestamps {
                        detected_at: chrono::Utc::now(),
                        last_modified: None,
                        previous_check: None,
                        time_since_change: None,
                    },
                    chunk_changes: Vec::new(),
                    confidence: 1.0,
                },
            };
            
            let update = IndexUpdate {
                operation_id: Uuid::new_v4().to_string(),
                operation,
                priority: 5,
                timestamp: chrono::Utc::now(),
                source: "test".to_string(),
                metadata: HashMap::new(),
                dependencies: Vec::new(),
                max_retries: 3,
                retry_count: 0,
            };
            
            operations.push(update);
        }
        
        let optimized = processor.optimize_batch(&operations).await.unwrap();
        
        // Should have only one operation after deduplication
        assert_eq!(optimized.len(), 1);
    }

    #[test]
    fn test_error_handling_strategies() {
        let strategies = vec![
            ErrorHandlingStrategy::FailFast,
            ErrorHandlingStrategy::ContinueOnError,
            ErrorHandlingStrategy::IsolateAndRetry,
            ErrorHandlingStrategy::CircuitBreaker,
        ];
        
        // Ensure all strategies are different
        for (i, strategy1) in strategies.iter().enumerate() {
            for (j, strategy2) in strategies.iter().enumerate() {
                if i != j {
                    assert_ne!(format!("{:?}", strategy1), format!("{:?}", strategy2));
                }
            }
        }
    }

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 1000);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.jitter_factor >= 0.0 && config.jitter_factor <= 1.0);
    }
}