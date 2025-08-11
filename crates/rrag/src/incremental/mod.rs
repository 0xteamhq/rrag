//! # Incremental Indexing System for RRAG
//! 
//! This module provides a comprehensive incremental indexing system that allows
//! efficient document updates in production RAG systems without requiring full re-indexing.
//! 
//! ## Key Features
//! 
//! - **Incremental Operations**: Add, update, delete operations without full rebuilds
//! - **Change Detection**: Efficient delta processing and conflict resolution  
//! - **Vector Index Updates**: Smart vector index management without full rebuilds
//! - **Document Versioning**: Complete document versioning and conflict resolution
//! - **Batch Processing**: Optimized batch processing for large-scale updates
//! - **Consistency Checks**: Index consistency and integrity verification
//! - **Rollback Support**: Complete rollback capabilities for failed operations
//! - **Performance Monitoring**: Comprehensive metrics and performance tracking
//! 
//! ## Architecture Overview
//! 
//! ```text
//! ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
//! │   Change Detection  │────│   Index Manager     │────│   Vector Store      │
//! │   - Content Hash    │    │   - Update Tracker  │    │   - Incremental     │
//! │   - Version Check   │    │   - Conflict Res.   │    │   - Batch Updates   │
//! └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
//!                                       │
//! ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
//! │   Rollback System   │────│   Batch Processor   │────│   Integrity Check   │
//! │   - Operation Log   │    │   - Queue Mgmt      │    │   - Consistency     │
//! │   - State Snapshots │    │   - Error Handling  │    │   - Validation      │
//! └─────────────────────┘    └─────────────────────┘    └─────────────────────┘
//! ```

pub mod change_detection;
pub mod index_manager;
pub mod batch_processor;
pub mod versioning;
pub mod rollback;
pub mod integrity;
pub mod vector_updates;
pub mod monitoring;

// Re-exports for convenience
pub use change_detection::{
    ChangeDetector, ChangeResult, ChangeType, DocumentChange, 
    ContentDelta, ChangeDetectionConfig
};

pub use index_manager::{
    IncrementalIndexManager, IndexOperation, IndexUpdate, 
    UpdateResult, ConflictResolution, IndexManagerConfig
};

pub use batch_processor::{
    BatchProcessor, BatchOperation, BatchConfig, BatchResult,
    BatchProcessingStats, BatchExecutor, QueueManager
};

pub use versioning::{
    DocumentVersion, VersionManager, VersionConflict, 
    VersionResolution, VersioningConfig, VersionHistory
};

pub use rollback::{
    RollbackManager, RollbackOperation, RollbackPoint, 
    RecoveryResult, RollbackConfig, OperationLog
};

pub use integrity::{
    IntegrityChecker, ConsistencyReport, IntegrityError,
    ValidationResult, IntegrityConfig, HealthMetrics
};

pub use vector_updates::{
    VectorUpdateManager, VectorOperation, EmbeddingUpdate,
    VectorBatch, VectorUpdateConfig, IndexUpdateStrategy
};

pub use monitoring::{
    IncrementalMetrics, PerformanceTracker, IndexingStats,
    MonitoringConfig, AlertConfig, MetricsCollector
};

use crate::RragResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Main incremental indexing service that orchestrates all components
pub struct IncrementalIndexingService {
    /// Change detection system
    change_detector: Arc<ChangeDetector>,
    
    /// Index management system
    index_manager: Arc<IncrementalIndexManager>,
    
    /// Batch processing system
    batch_processor: Arc<BatchProcessor>,
    
    /// Version management system
    version_manager: Arc<VersionManager>,
    
    /// Rollback management system
    rollback_manager: Arc<RollbackManager>,
    
    /// Integrity checking system
    integrity_checker: Arc<IntegrityChecker>,
    
    /// Vector update system
    vector_manager: Arc<VectorUpdateManager>,
    
    /// Performance monitoring
    metrics: Arc<RwLock<IncrementalMetrics>>,
    
    /// Service configuration
    config: IncrementalServiceConfig,
}

/// Configuration for the incremental indexing service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalServiceConfig {
    /// Enable automatic change detection
    pub auto_change_detection: bool,
    
    /// Enable batch processing optimization
    pub enable_batch_processing: bool,
    
    /// Enable version conflict resolution
    pub enable_version_resolution: bool,
    
    /// Enable automatic integrity checks
    pub auto_integrity_checks: bool,
    
    /// Enable rollback capabilities
    pub enable_rollback: bool,
    
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    
    /// Maximum batch size for operations
    pub max_batch_size: usize,
    
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    
    /// Integrity check interval in seconds
    pub integrity_check_interval_secs: u64,
    
    /// Enable metrics collection
    pub collect_metrics: bool,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable vector index optimization
    pub optimize_vector_index: bool,
    
    /// Enable smart conflict resolution
    pub smart_conflict_resolution: bool,
    
    /// Enable predictive prefetching
    pub enable_prefetching: bool,
    
    /// Enable compression for storage
    pub enable_compression: bool,
    
    /// Memory pool size for operations
    pub memory_pool_size: usize,
    
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
}

impl Default for IncrementalServiceConfig {
    fn default() -> Self {
        Self {
            auto_change_detection: true,
            enable_batch_processing: true,
            enable_version_resolution: true,
            auto_integrity_checks: true,
            enable_rollback: true,
            enable_monitoring: true,
            max_batch_size: 1000,
            batch_timeout_ms: 5000,
            max_concurrent_ops: 10,
            integrity_check_interval_secs: 3600, // 1 hour
            collect_metrics: true,
            optimization: OptimizationConfig::default(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            optimize_vector_index: true,
            smart_conflict_resolution: true,
            enable_prefetching: false,
            enable_compression: true,
            memory_pool_size: 1024 * 1024 * 100, // 100MB
            enable_parallel_processing: true,
        }
    }
}

/// Service builder for easy configuration
pub struct IncrementalServiceBuilder {
    config: IncrementalServiceConfig,
}

impl IncrementalServiceBuilder {
    pub fn new() -> Self {
        Self {
            config: IncrementalServiceConfig::default(),
        }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.batch_timeout_ms = timeout_ms;
        self
    }

    pub fn with_concurrency(mut self, max_ops: usize) -> Self {
        self.config.max_concurrent_ops = max_ops;
        self
    }

    pub fn enable_feature(mut self, feature: &str, enabled: bool) -> Self {
        match feature {
            "auto_change_detection" => self.config.auto_change_detection = enabled,
            "batch_processing" => self.config.enable_batch_processing = enabled,
            "version_resolution" => self.config.enable_version_resolution = enabled,
            "integrity_checks" => self.config.auto_integrity_checks = enabled,
            "rollback" => self.config.enable_rollback = enabled,
            "monitoring" => self.config.enable_monitoring = enabled,
            _ => {} // Ignore unknown features
        }
        self
    }

    pub fn with_optimization(mut self, optimization: OptimizationConfig) -> Self {
        self.config.optimization = optimization;
        self
    }

    pub async fn build(self) -> RragResult<IncrementalIndexingService> {
        IncrementalIndexingService::new(self.config).await
    }
}

impl Default for IncrementalServiceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Service implementation
impl IncrementalIndexingService {
    /// Create a new incremental indexing service
    pub async fn new(config: IncrementalServiceConfig) -> RragResult<Self> {
        // Initialize all components with their respective configurations
        let change_detector = Arc::new(ChangeDetector::new(
            ChangeDetectionConfig::default()
        ).await?);

        let index_manager = Arc::new(IncrementalIndexManager::new(
            IndexManagerConfig::default()
        ).await?);

        let batch_processor = Arc::new(BatchProcessor::new(
            BatchConfig::default()
        ).await?);

        let version_manager = Arc::new(VersionManager::new(
            VersioningConfig::default()
        ).await?);

        let rollback_manager = Arc::new(RollbackManager::new(
            RollbackConfig::default()
        ).await?);

        let integrity_checker = Arc::new(IntegrityChecker::new(
            IntegrityConfig::default()
        ).await?);

        let vector_manager = Arc::new(VectorUpdateManager::new(
            VectorUpdateConfig::default()
        ).await?);

        let metrics = Arc::new(RwLock::new(IncrementalMetrics::new()));

        Ok(Self {
            change_detector,
            index_manager,
            batch_processor,
            version_manager,
            rollback_manager,
            integrity_checker,
            vector_manager,
            metrics,
            config,
        })
    }

    /// Get service metrics
    pub async fn get_metrics(&self) -> IncrementalMetrics {
        self.metrics.read().await.clone()
    }

    /// Perform health check on all components
    pub async fn health_check(&self) -> RragResult<HashMap<String, bool>> {
        let mut health_status = HashMap::new();
        
        health_status.insert("change_detector".to_string(), 
            self.change_detector.health_check().await?);
        health_status.insert("index_manager".to_string(), 
            self.index_manager.health_check().await?);
        health_status.insert("batch_processor".to_string(), 
            self.batch_processor.health_check().await?);
        health_status.insert("version_manager".to_string(), 
            self.version_manager.health_check().await?);
        health_status.insert("rollback_manager".to_string(), 
            self.rollback_manager.health_check().await?);
        health_status.insert("integrity_checker".to_string(), 
            self.integrity_checker.health_check().await?);
        health_status.insert("vector_manager".to_string(), 
            self.vector_manager.health_check().await?);

        Ok(health_status)
    }

    /// Get service configuration
    pub fn get_config(&self) -> &IncrementalServiceConfig {
        &self.config
    }

    /// Update service configuration
    pub async fn update_config(&mut self, new_config: IncrementalServiceConfig) -> RragResult<()> {
        self.config = new_config;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_builder() {
        let service = IncrementalServiceBuilder::new()
            .with_batch_size(500)
            .with_timeout(1000)
            .with_concurrency(5)
            .enable_feature("monitoring", true)
            .build()
            .await
            .unwrap();

        assert_eq!(service.config.max_batch_size, 500);
        assert_eq!(service.config.batch_timeout_ms, 1000);
        assert_eq!(service.config.max_concurrent_ops, 5);
        assert!(service.config.enable_monitoring);
    }

    #[tokio::test]
    async fn test_service_creation() {
        let config = IncrementalServiceConfig::default();
        let service = IncrementalIndexingService::new(config).await.unwrap();
        
        assert!(service.config.auto_change_detection);
        assert!(service.config.enable_batch_processing);
        assert!(service.config.enable_version_resolution);
    }

    #[tokio::test]
    async fn test_health_check() {
        let service = IncrementalServiceBuilder::new().build().await.unwrap();
        let health = service.health_check().await.unwrap();
        
        assert!(health.len() >= 7); // All components should report health
        assert!(health.values().all(|&healthy| healthy)); // All should be healthy initially
    }
}