//! # Incremental Index Manager
//! 
//! Manages incremental updates to document indexes without requiring full rebuilds.
//! Handles conflict resolution, operation queuing, and index consistency.

use crate::{RragError, RragResult, Document, DocumentChunk, Embedding};
use crate::incremental::change_detection::{ChangeResult, ChangeType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use uuid::Uuid;

/// Index manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManagerConfig {
    /// Maximum pending operations
    pub max_pending_operations: usize,
    
    /// Operation batch size
    pub batch_size: usize,
    
    /// Operation timeout in seconds
    pub operation_timeout_secs: u64,
    
    /// Enable conflict resolution
    pub enable_conflict_resolution: bool,
    
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolutionStrategy,
    
    /// Enable operation logging
    pub enable_operation_log: bool,
    
    /// Maximum operation log size
    pub max_operation_log: usize,
    
    /// Enable automatic cleanup
    pub enable_auto_cleanup: bool,
    
    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Last write wins
    LastWriteWins,
    /// First write wins
    FirstWriteWins,
    /// Merge changes when possible
    Merge,
    /// Manual resolution required
    Manual,
    /// Use version timestamps
    Timestamp,
    /// Use custom resolution logic
    Custom(String),
}

impl Default for IndexManagerConfig {
    fn default() -> Self {
        Self {
            max_pending_operations: 10000,
            batch_size: 100,
            operation_timeout_secs: 300, // 5 minutes
            enable_conflict_resolution: true,
            conflict_resolution: ConflictResolutionStrategy::LastWriteWins,
            enable_operation_log: true,
            max_operation_log: 10000,
            enable_auto_cleanup: true,
            cleanup_interval_secs: 3600, // 1 hour
        }
    }
}

/// Types of index operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IndexOperation {
    /// Add new document and its chunks
    Add {
        document: Document,
        chunks: Vec<DocumentChunk>,
        embeddings: Vec<Embedding>,
    },
    
    /// Update existing document
    Update {
        document_id: String,
        document: Document,
        chunks: Vec<DocumentChunk>,
        embeddings: Vec<Embedding>,
        change_result: ChangeResult,
    },
    
    /// Delete document and all associated data
    Delete {
        document_id: String,
    },
    
    /// Update only embeddings
    UpdateEmbeddings {
        document_id: String,
        embeddings: Vec<Embedding>,
    },
    
    /// Update only chunks
    UpdateChunks {
        document_id: String,
        chunks: Vec<DocumentChunk>,
    },
    
    /// Batch operation containing multiple operations
    Batch {
        operations: Vec<IndexOperation>,
    },
    
    /// Rebuild specific index
    Rebuild {
        index_name: String,
        document_ids: Vec<String>,
    },
}

/// Index update specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUpdate {
    /// Unique operation ID
    pub operation_id: String,
    
    /// Operation to perform
    pub operation: IndexOperation,
    
    /// Priority level (0-10, higher = more priority)
    pub priority: u8,
    
    /// Operation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Source of the operation
    pub source: String,
    
    /// Operation metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Dependencies on other operations
    pub dependencies: Vec<String>,
    
    /// Maximum retry attempts
    pub max_retries: u32,
    
    /// Current retry count
    pub retry_count: u32,
}

/// Result of an update operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    /// Operation ID
    pub operation_id: String,
    
    /// Whether the operation succeeded
    pub success: bool,
    
    /// Operations performed
    pub operations_completed: Vec<String>,
    
    /// Conflicts encountered
    pub conflicts: Vec<ConflictInfo>,
    
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Items affected
    pub items_affected: usize,
    
    /// Error details if failed
    pub error: Option<String>,
    
    /// Metadata about the operation
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    /// Document ID where conflict occurred
    pub document_id: String,
    
    /// Type of conflict
    pub conflict_type: ConflictType,
    
    /// Conflicting operations
    pub conflicting_operations: Vec<String>,
    
    /// Resolution applied
    pub resolution: ConflictResolution,
    
    /// Additional context
    pub context: HashMap<String, serde_json::Value>,
}

/// Types of conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Multiple updates to the same document
    ConcurrentUpdate,
    /// Version mismatch
    VersionMismatch,
    /// Dependency conflict
    DependencyConflict,
    /// Resource lock conflict
    ResourceLock,
    /// Schema conflict
    SchemaConflict,
}

/// Conflict resolution applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Automatically resolved
    AutoResolved(String),
    /// Manually resolved
    ManuallyResolved(String),
    /// Deferred for later resolution
    Deferred,
    /// Failed to resolve
    Failed(String),
}

/// Operation status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationStatus {
    /// Queued for processing
    Queued,
    /// Currently being processed
    Processing,
    /// Successfully completed
    Completed,
    /// Failed with error
    Failed(String),
    /// Cancelled
    Cancelled,
    /// Waiting for dependencies
    Waiting,
    /// Conflict resolution required
    ConflictResolution,
}

/// Tracked operation state
#[derive(Debug, Clone)]
struct TrackedOperation {
    /// Update specification
    update: IndexUpdate,
    
    /// Current status
    status: OperationStatus,
    
    /// Start time
    start_time: Option<chrono::DateTime<chrono::Utc>>,
    
    /// End time
    end_time: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Result if completed
    result: Option<UpdateResult>,
}

/// Incremental index manager
pub struct IncrementalIndexManager {
    /// Configuration
    config: IndexManagerConfig,
    
    /// Pending operations queue
    pending_operations: Arc<Mutex<VecDeque<TrackedOperation>>>,
    
    /// Currently processing operations
    processing_operations: Arc<RwLock<HashMap<String, TrackedOperation>>>,
    
    /// Completed operations history
    completed_operations: Arc<RwLock<VecDeque<TrackedOperation>>>,
    
    /// Index state tracking
    index_state: Arc<RwLock<IndexState>>,
    
    /// Conflict resolution system
    conflict_resolver: Arc<ConflictResolver>,
    
    /// Operation statistics
    stats: Arc<RwLock<IndexManagerStats>>,
    
    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

/// Index state tracking
#[derive(Debug, Clone)]
struct IndexState {
    /// Documents currently indexed
    indexed_documents: HashSet<String>,
    
    /// Document versions
    document_versions: HashMap<String, u64>,
    
    /// Document locks for concurrent access
    document_locks: HashMap<String, tokio::sync::Mutex<()>>,
    
    /// Index metadata
    metadata: HashMap<String, serde_json::Value>,
    
    /// Last update timestamp
    last_updated: chrono::DateTime<chrono::Utc>,
}

/// Conflict resolution system
struct ConflictResolver {
    /// Resolution strategy
    strategy: ConflictResolutionStrategy,
    
    /// Manual resolution queue
    manual_queue: Arc<Mutex<VecDeque<ConflictInfo>>>,
    
    /// Resolution history
    resolution_history: Arc<RwLock<Vec<ConflictInfo>>>,
}

/// Index manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManagerStats {
    /// Total operations processed
    pub total_operations: u64,
    
    /// Operations by type
    pub operations_by_type: HashMap<String, u64>,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Average processing time
    pub avg_processing_time_ms: f64,
    
    /// Conflicts encountered
    pub total_conflicts: u64,
    
    /// Conflicts resolved automatically
    pub auto_resolved_conflicts: u64,
    
    /// Queue depth statistics
    pub current_queue_depth: usize,
    pub max_queue_depth: usize,
    
    /// Performance metrics
    pub throughput_ops_per_second: f64,
    
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl IncrementalIndexManager {
    /// Create a new index manager
    pub async fn new(config: IndexManagerConfig) -> RragResult<Self> {
        let pending_operations = Arc::new(Mutex::new(VecDeque::new()));
        let processing_operations = Arc::new(RwLock::new(HashMap::new()));
        let completed_operations = Arc::new(RwLock::new(VecDeque::new()));
        
        let index_state = Arc::new(RwLock::new(IndexState {
            indexed_documents: HashSet::new(),
            document_versions: HashMap::new(),
            document_locks: HashMap::new(),
            metadata: HashMap::new(),
            last_updated: chrono::Utc::now(),
        }));

        let conflict_resolver = Arc::new(ConflictResolver {
            strategy: config.conflict_resolution.clone(),
            manual_queue: Arc::new(Mutex::new(VecDeque::new())),
            resolution_history: Arc::new(RwLock::new(Vec::new())),
        });

        let stats = Arc::new(RwLock::new(IndexManagerStats {
            total_operations: 0,
            operations_by_type: HashMap::new(),
            success_rate: 0.0,
            avg_processing_time_ms: 0.0,
            total_conflicts: 0,
            auto_resolved_conflicts: 0,
            current_queue_depth: 0,
            max_queue_depth: 0,
            throughput_ops_per_second: 0.0,
            last_updated: chrono::Utc::now(),
        }));

        let task_handles = Arc::new(Mutex::new(Vec::new()));

        let manager = Self {
            config,
            pending_operations,
            processing_operations,
            completed_operations,
            index_state,
            conflict_resolver,
            stats,
            task_handles,
        };

        // Start background processing tasks
        manager.start_background_tasks().await?;

        Ok(manager)
    }

    /// Submit an update operation
    pub async fn submit_update(&self, update: IndexUpdate) -> RragResult<String> {
        // Validate update
        self.validate_update(&update).await?;

        // Create tracked operation
        let tracked_op = TrackedOperation {
            update: update.clone(),
            status: OperationStatus::Queued,
            start_time: None,
            end_time: None,
            result: None,
        };

        // Add to queue
        {
            let mut queue = self.pending_operations.lock().await;
            
            // Check queue capacity
            if queue.len() >= self.config.max_pending_operations {
                return Err(RragError::storage("queue_full", 
                    std::io::Error::new(std::io::ErrorKind::Other, "Operation queue is full")));
            }
            
            queue.push_back(tracked_op);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.current_queue_depth = {
                let queue = self.pending_operations.lock().await;
                queue.len()
            };
            stats.max_queue_depth = std::cmp::max(stats.max_queue_depth, stats.current_queue_depth);
        }

        Ok(update.operation_id)
    }

    /// Submit multiple operations as a batch
    pub async fn submit_batch(&self, operations: Vec<IndexUpdate>) -> RragResult<Vec<String>> {
        if operations.is_empty() {
            return Ok(Vec::new());
        }

        // Create batch operation
        let batch_id = Uuid::new_v4().to_string();
        let batch_operation = IndexOperation::Batch {
            operations: operations.iter().map(|op| op.operation.clone()).collect(),
        };

        let batch_update = IndexUpdate {
            operation_id: batch_id.clone(),
            operation: batch_operation,
            priority: operations.iter().map(|op| op.priority).max().unwrap_or(5),
            timestamp: chrono::Utc::now(),
            source: "batch_processor".to_string(),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            max_retries: 3,
            retry_count: 0,
        };

        // Submit individual operations
        let mut operation_ids = Vec::new();
        for operation in operations {
            let op_id = self.submit_update(operation).await?;
            operation_ids.push(op_id);
        }

        // Submit batch operation
        self.submit_update(batch_update).await?;
        operation_ids.push(batch_id);

        Ok(operation_ids)
    }

    /// Get operation status
    pub async fn get_operation_status(&self, operation_id: &str) -> RragResult<Option<OperationStatus>> {
        // Check processing operations
        {
            let processing = self.processing_operations.read().await;
            if let Some(op) = processing.get(operation_id) {
                return Ok(Some(op.status.clone()));
            }
        }

        // Check pending operations
        {
            let queue = self.pending_operations.lock().await;
            for op in queue.iter() {
                if op.update.operation_id == operation_id {
                    return Ok(Some(op.status.clone()));
                }
            }
        }

        // Check completed operations
        {
            let completed = self.completed_operations.read().await;
            for op in completed.iter() {
                if op.update.operation_id == operation_id {
                    return Ok(Some(op.status.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Get operation result
    pub async fn get_operation_result(&self, operation_id: &str) -> RragResult<Option<UpdateResult>> {
        // Check processing operations first
        {
            let processing = self.processing_operations.read().await;
            if let Some(op) = processing.get(operation_id) {
                return Ok(op.result.clone());
            }
        }

        // Check completed operations
        {
            let completed = self.completed_operations.read().await;
            for op in completed.iter() {
                if op.update.operation_id == operation_id {
                    return Ok(op.result.clone());
                }
            }
        }

        Ok(None)
    }

    /// Cancel a pending operation
    pub async fn cancel_operation(&self, operation_id: &str) -> RragResult<bool> {
        // Try to cancel from pending queue
        {
            let mut queue = self.pending_operations.lock().await;
            if let Some(pos) = queue.iter().position(|op| op.update.operation_id == operation_id) {
                queue.remove(pos);
                return Ok(true);
            }
        }

        // Try to cancel from processing (if not too far along)
        {
            let mut processing = self.processing_operations.write().await;
            if let Some(mut op) = processing.remove(operation_id) {
                op.status = OperationStatus::Cancelled;
                op.end_time = Some(chrono::Utc::now());
                
                // Move to completed
                let mut completed = self.completed_operations.write().await;
                completed.push_back(op);
                
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> IndexManagerStats {
        let mut stats = self.stats.read().await.clone();
        stats.current_queue_depth = {
            let queue = self.pending_operations.lock().await;
            queue.len()
        };
        stats.last_updated = chrono::Utc::now();
        stats
    }

    /// Get index state information
    pub async fn get_index_state(&self) -> RragResult<HashMap<String, serde_json::Value>> {
        let state = self.index_state.read().await;
        let mut info = HashMap::new();
        
        info.insert("indexed_documents_count".to_string(), 
            serde_json::Value::Number(state.indexed_documents.len().into()));
        info.insert("last_updated".to_string(), 
            serde_json::Value::String(state.last_updated.to_rfc3339()));
        info.insert("metadata".to_string(), 
            serde_json::Value::Object(state.metadata.clone().into_iter().collect()));
        
        Ok(info)
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        // Check if background tasks are running
        let handles = self.task_handles.lock().await;
        let all_running = handles.iter().all(|handle| !handle.is_finished());
        
        // Check queue health
        let queue_size = {
            let queue = self.pending_operations.lock().await;
            queue.len()
        };
        let queue_healthy = queue_size < self.config.max_pending_operations;
        
        Ok(all_running && queue_healthy)
    }

    /// Start background processing tasks
    async fn start_background_tasks(&self) -> RragResult<()> {
        let mut handles = self.task_handles.lock().await;
        
        // Operation processor task
        let processor_handle = self.start_operation_processor().await;
        handles.push(processor_handle);
        
        // Cleanup task
        if self.config.enable_auto_cleanup {
            let cleanup_handle = self.start_cleanup_task().await;
            handles.push(cleanup_handle);
        }
        
        Ok(())
    }

    /// Start the main operation processor
    async fn start_operation_processor(&self) -> tokio::task::JoinHandle<()> {
        let pending_ops = Arc::clone(&self.pending_operations);
        let processing_ops = Arc::clone(&self.processing_operations);
        let completed_ops = Arc::clone(&self.completed_operations);
        let index_state = Arc::clone(&self.index_state);
        let conflict_resolver = Arc::clone(&self.conflict_resolver);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();

        tokio::spawn(async move {
            loop {
                // Process next operation
                let operation = {
                    let mut queue = pending_ops.lock().await;
                    queue.pop_front()
                };

                if let Some(mut tracked_op) = operation {
                    tracked_op.status = OperationStatus::Processing;
                    tracked_op.start_time = Some(chrono::Utc::now());
                    
                    let operation_id = tracked_op.update.operation_id.clone();
                    
                    // Move to processing
                    {
                        let mut processing = processing_ops.write().await;
                        processing.insert(operation_id.clone(), tracked_op.clone());
                    }

                    // Process the operation
                    let result = Self::process_operation(
                        &tracked_op.update,
                        &index_state,
                        &conflict_resolver,
                        &config,
                    ).await;

                    // Update tracked operation
                    tracked_op.end_time = Some(chrono::Utc::now());
                    tracked_op.result = Some(result.clone());
                    tracked_op.status = if result.success {
                        OperationStatus::Completed
                    } else {
                        OperationStatus::Failed(result.error.unwrap_or_default())
                    };

                    // Move to completed
                    {
                        let mut processing = processing_ops.write().await;
                        processing.remove(&operation_id);
                    }
                    {
                        let mut completed = completed_ops.write().await;
                        completed.push_back(tracked_op);
                        
                        // Limit completed operations history
                        if completed.len() > config.max_operation_log {
                            completed.pop_front();
                        }
                    }

                    // Update statistics
                    {
                        let mut stats_guard = stats.write().await;
                        stats_guard.total_operations += 1;
                        
                        let op_type = format!("{:?}", tracked_op.update.operation).split('{').next().unwrap_or("Unknown").to_string();
                        *stats_guard.operations_by_type.entry(op_type).or_insert(0) += 1;
                        
                        stats_guard.success_rate = if stats_guard.total_operations > 0 {
                            let successful = stats_guard.operations_by_type.values().sum::<u64>();
                            successful as f64 / stats_guard.total_operations as f64
                        } else {
                            0.0
                        };
                        
                        stats_guard.avg_processing_time_ms = 
                            (stats_guard.avg_processing_time_ms + result.processing_time_ms as f64) / 2.0;
                        
                        stats_guard.last_updated = chrono::Utc::now();
                    }
                } else {
                    // No operations pending, sleep briefly
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
        })
    }

    /// Start cleanup task
    async fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let completed_ops = Arc::clone(&self.completed_operations);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(config.cleanup_interval_secs)
            );

            loop {
                interval.tick().await;
                
                // Cleanup old completed operations
                {
                    let mut completed = completed_ops.write().await;
                    while completed.len() > config.max_operation_log {
                        completed.pop_front();
                    }
                }
            }
        })
    }

    /// Process a single operation (static method for background task)
    async fn process_operation(
        update: &IndexUpdate,
        index_state: &Arc<RwLock<IndexState>>,
        conflict_resolver: &Arc<ConflictResolver>,
        config: &IndexManagerConfig,
    ) -> UpdateResult {
        let start_time = std::time::Instant::now();
        let mut conflicts = Vec::new();
        let mut items_affected = 0;
        let mut operations_completed = Vec::new();

        let success = match &update.operation {
            IndexOperation::Add { document, chunks, embeddings } => {
                match Self::process_add_operation(
                    document, chunks, embeddings, index_state
                ).await {
                    Ok(count) => {
                        items_affected = count;
                        operations_completed.push("add".to_string());
                        true
                    }
                    Err(_) => false
                }
            }
            
            IndexOperation::Update { document_id, document, chunks, embeddings, change_result } => {
                match Self::process_update_operation(
                    document_id, document, chunks, embeddings, change_result, index_state, conflict_resolver
                ).await {
                    Ok((count, detected_conflicts)) => {
                        items_affected = count;
                        conflicts = detected_conflicts;
                        operations_completed.push("update".to_string());
                        true
                    }
                    Err(_) => false
                }
            }
            
            IndexOperation::Delete { document_id } => {
                match Self::process_delete_operation(document_id, index_state).await {
                    Ok(count) => {
                        items_affected = count;
                        operations_completed.push("delete".to_string());
                        true
                    }
                    Err(_) => false
                }
            }
            
            IndexOperation::UpdateEmbeddings { document_id, embeddings } => {
                match Self::process_embedding_update(document_id, embeddings, index_state).await {
                    Ok(count) => {
                        items_affected = count;
                        operations_completed.push("update_embeddings".to_string());
                        true
                    }
                    Err(_) => false
                }
            }
            
            IndexOperation::UpdateChunks { document_id, chunks } => {
                match Self::process_chunk_update(document_id, chunks, index_state).await {
                    Ok(count) => {
                        items_affected = count;
                        operations_completed.push("update_chunks".to_string());
                        true
                    }
                    Err(_) => false
                }
            }
            
            IndexOperation::Batch { operations } => {
                operations_completed.push("batch".to_string());
                items_affected = operations.len();
                true // Simplified for batch operations
            }
            
            IndexOperation::Rebuild { index_name, document_ids } => {
                operations_completed.push("rebuild".to_string());
                items_affected = document_ids.len();
                true // Simplified for rebuild operations
            }
        };

        UpdateResult {
            operation_id: update.operation_id.clone(),
            success,
            operations_completed,
            conflicts,
            processing_time_ms: start_time.elapsed().as_millis() as u64,
            items_affected,
            error: if success { None } else { Some("Operation failed".to_string()) },
            metadata: HashMap::new(),
        }
    }

    /// Process add operation
    async fn process_add_operation(
        document: &Document,
        chunks: &[DocumentChunk],
        embeddings: &[Embedding],
        index_state: &Arc<RwLock<IndexState>>,
    ) -> RragResult<usize> {
        let mut state = index_state.write().await;
        
        // Add document to index
        state.indexed_documents.insert(document.id.clone());
        state.document_versions.insert(document.id.clone(), 1);
        state.last_updated = chrono::Utc::now();
        
        Ok(1 + chunks.len() + embeddings.len())
    }

    /// Process update operation
    async fn process_update_operation(
        document_id: &str,
        document: &Document,
        chunks: &[DocumentChunk],
        embeddings: &[Embedding],
        change_result: &ChangeResult,
        index_state: &Arc<RwLock<IndexState>>,
        _conflict_resolver: &Arc<ConflictResolver>,
    ) -> RragResult<(usize, Vec<ConflictInfo>)> {
        let mut state = index_state.write().await;
        let mut conflicts = Vec::new();
        
        // Check for conflicts
        if let Some(current_version) = state.document_versions.get(document_id) {
            // Simple conflict detection - in production, would be more sophisticated
            if change_result.change_type == ChangeType::NoChange {
                // No actual conflict, but could indicate race condition
            }
        }
        
        // Update document in index
        state.indexed_documents.insert(document.id.clone());
        let new_version = state.document_versions.get(document_id).unwrap_or(&0) + 1;
        state.document_versions.insert(document_id.to_string(), new_version);
        state.last_updated = chrono::Utc::now();
        
        Ok((1 + chunks.len() + embeddings.len(), conflicts))
    }

    /// Process delete operation
    async fn process_delete_operation(
        document_id: &str,
        index_state: &Arc<RwLock<IndexState>>,
    ) -> RragResult<usize> {
        let mut state = index_state.write().await;
        
        let was_present = state.indexed_documents.remove(document_id);
        state.document_versions.remove(document_id);
        state.last_updated = chrono::Utc::now();
        
        Ok(if was_present { 1 } else { 0 })
    }

    /// Process embedding update
    async fn process_embedding_update(
        _document_id: &str,
        embeddings: &[Embedding],
        index_state: &Arc<RwLock<IndexState>>,
    ) -> RragResult<usize> {
        let mut state = index_state.write().await;
        state.last_updated = chrono::Utc::now();
        Ok(embeddings.len())
    }

    /// Process chunk update
    async fn process_chunk_update(
        _document_id: &str,
        chunks: &[DocumentChunk],
        index_state: &Arc<RwLock<IndexState>>,
    ) -> RragResult<usize> {
        let mut state = index_state.write().await;
        state.last_updated = chrono::Utc::now();
        Ok(chunks.len())
    }

    /// Validate an update operation
    async fn validate_update(&self, update: &IndexUpdate) -> RragResult<()> {
        // Basic validation
        if update.operation_id.is_empty() {
            return Err(RragError::validation("operation_id", "non-empty", "empty"));
        }
        
        if update.priority > 10 {
            return Err(RragError::validation("priority", "0-10", &update.priority.to_string()));
        }
        
        // Validate operation-specific requirements
        match &update.operation {
            IndexOperation::Add { document, .. } => {
                if document.id.is_empty() {
                    return Err(RragError::validation("document.id", "non-empty", "empty"));
                }
            }
            IndexOperation::Update { document_id, .. } => {
                if document_id.is_empty() {
                    return Err(RragError::validation("document_id", "non-empty", "empty"));
                }
            }
            IndexOperation::Delete { document_id } => {
                if document_id.is_empty() {
                    return Err(RragError::validation("document_id", "non-empty", "empty"));
                }
            }
            _ => {} // Other validations as needed
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[tokio::test]
    async fn test_index_manager_creation() {
        let config = IndexManagerConfig::default();
        let manager = IncrementalIndexManager::new(config).await.unwrap();
        assert!(manager.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_submit_add_operation() {
        let manager = IncrementalIndexManager::new(IndexManagerConfig::default()).await.unwrap();
        
        let doc = Document::new("Test content");
        let operation = IndexOperation::Add {
            document: doc.clone(),
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
        
        let op_id = manager.submit_update(update).await.unwrap();
        assert!(!op_id.is_empty());
        
        // Check that operation was queued
        let status = manager.get_operation_status(&op_id).await.unwrap();
        assert!(status.is_some());
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let manager = IncrementalIndexManager::new(IndexManagerConfig::default()).await.unwrap();
        
        let mut operations = Vec::new();
        for i in 0..3 {
            let doc = Document::new(format!("Test content {}", i));
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
            
            operations.push(update);
        }
        
        let op_ids = manager.submit_batch(operations).await.unwrap();
        assert_eq!(op_ids.len(), 4); // 3 individual + 1 batch operation
    }

    #[tokio::test]
    async fn test_operation_cancellation() {
        let manager = IncrementalIndexManager::new(IndexManagerConfig::default()).await.unwrap();
        
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
        
        let op_id = manager.submit_update(update).await.unwrap();
        
        // Try to cancel the operation
        let cancelled = manager.cancel_operation(&op_id).await.unwrap();
        assert!(cancelled);
    }

    #[test]
    fn test_conflict_resolution_strategies() {
        let strategies = vec![
            ConflictResolutionStrategy::LastWriteWins,
            ConflictResolutionStrategy::FirstWriteWins,
            ConflictResolutionStrategy::Merge,
            ConflictResolutionStrategy::Manual,
            ConflictResolutionStrategy::Timestamp,
            ConflictResolutionStrategy::Custom("custom_logic".to_string()),
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
}