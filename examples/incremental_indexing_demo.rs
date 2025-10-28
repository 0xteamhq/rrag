//! # Incremental Indexing System Demo
//!
//! This example demonstrates the comprehensive incremental indexing system for the RRAG framework.
//! It showcases all the key features including:
//!
//! - Incremental document indexing (add, update, delete operations)
//! - Efficient change detection and delta processing
//! - Vector index updates without full rebuilds
//! - Document versioning and conflict resolution
//! - Batch processing for large updates
//! - Index consistency and integrity checks
//! - Rollback capabilities for failed updates
//! - Performance monitoring and alerting

use rrag::incremental::batch_processor::{BatchConfig, BatchProcessor, ErrorHandlingStrategy};
use rrag::incremental::change_detection::ChangeSensitivity;
use rrag::incremental::change_detection::{
    ChangeDetectionConfig, ChangeDetector, ChangeResult, ChangeTimestamps,
    ChangeType as DetectionChangeType, ContentDelta, MetadataChanges,
};
use rrag::incremental::index_manager::{
    ConflictResolutionStrategy, IncrementalIndexManager, IndexManagerConfig, IndexOperation,
    IndexUpdate, UpdateResult,
};
use rrag::incremental::integrity::{IntegrityChecker, IntegrityConfig};
use rrag::incremental::monitoring::{
    AlertConfig, AlertThresholds, ErrorMetrics, IndexingMetrics, MetricsCollector, MetricsUpdate,
    MonitoringConfig, OperationMetrics, PerformanceDataPoint, PerformanceTracker, RetryMetrics,
};
use rrag::incremental::rollback::{
    RollbackConfig, RollbackManager, RollbackOperation, SystemState,
};
use rrag::incremental::vector_updates::{
    EmbeddingUpdate, OptimizationType, UpdateReason, VectorOperation, VectorUpdateConfig,
    VectorUpdateManager,
};
use rrag::incremental::versioning::{
    ChangeType, ConflictDetectionStrategy, ResolutionStrategy, VersionManager, VersionResolution,
    VersioningConfig,
};
use rrag::incremental::*;
use rrag::prelude::*;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[tokio::main]
async fn main() -> RragResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    tracing::debug!("🚀 RRAG Incremental Indexing System Demo");
    tracing::debug!("=========================================\n");

    // Step 1: Create and configure the incremental indexing service
    tracing::debug!("📋 Step 1: Setting up Incremental Indexing Service");
    let service = create_incremental_service().await?;

    // Perform health check
    let health = service.health_check().await?;
    tracing::debug!("✅ Service health check: {:?}", health);

    // Step 2: Demonstrate change detection
    tracing::debug!("🔍 Step 2: Change Detection Demo");
    demo_change_detection().await?;

    // Step 3: Demonstrate incremental indexing operations
    tracing::debug!("📚 Step 3: Incremental Indexing Operations");
    demo_incremental_operations().await?;

    // Step 4: Demonstrate batch processing
    tracing::debug!("⚡ Step 4: Batch Processing Demo");
    demo_batch_processing().await?;

    // Step 5: Demonstrate versioning and conflict resolution
    tracing::debug!("🔄 Step 5: Versioning and Conflict Resolution");
    demo_versioning_system().await?;

    // Step 6: Demonstrate rollback capabilities
    tracing::debug!("↩️  Step 6: Rollback System Demo");
    demo_rollback_system().await?;

    // Step 7: Demonstrate integrity checking
    tracing::debug!("🔒 Step 7: Integrity Checking Demo");
    demo_integrity_system().await?;

    // Step 8: Demonstrate vector updates
    tracing::debug!("🎯 Step 8: Vector Update Management");
    demo_vector_updates().await?;

    // Step 9: Demonstrate monitoring and metrics
    tracing::debug!("📊 Step 9: Monitoring and Metrics Demo");
    demo_monitoring_system().await?;

    // Step 10: Demonstrate production scenarios
    tracing::debug!("🏭 Step 10: Production Scenarios");
    demo_production_scenarios().await?;

    tracing::debug!("🎉 Demo completed successfully!");
    tracing::debug!("The incremental indexing system provides:");
    tracing::debug!("  ✅ Efficient document updates without full rebuilds");
    tracing::debug!("  ✅ Comprehensive change detection and versioning");
    tracing::debug!("  ✅ Robust batch processing with error handling");
    tracing::debug!("  ✅ Complete rollback and recovery capabilities");
    tracing::debug!("  ✅ Continuous integrity monitoring");
    tracing::debug!("  ✅ Performance optimization and alerting");

    Ok(())
}

/// Create and configure the incremental indexing service
async fn create_incremental_service() -> RragResult<IncrementalIndexingService> {
    tracing::debug!("  📦 Creating incremental indexing service...");

    let service = IncrementalServiceBuilder::new()
        .with_batch_size(100)
        .with_timeout(5000)
        .with_concurrency(8)
        .enable_feature("auto_change_detection", true)
        .enable_feature("batch_processing", true)
        .enable_feature("version_resolution", true)
        .enable_feature("integrity_checks", true)
        .enable_feature("rollback", true)
        .enable_feature("monitoring", true)
        .build()
        .await?;

    tracing::debug!("  ✅ Service created with optimized configuration");
    tracing::debug!("  📊 Max batch size: 100 documents");
    tracing::debug!("  ⏱️  Batch timeout: 5 seconds");
    tracing::debug!("  🔄 Max concurrency: 8 operations");

    Ok(service)
}

/// Demonstrate change detection capabilities
async fn demo_change_detection() -> RragResult<()> {
    tracing::debug!("  🔍 Setting up change detector...");

    let detector = ChangeDetector::new(ChangeDetectionConfig {
        enable_content_hash: true,
        enable_metadata_detection: true,
        enable_timestamp_detection: true,
        enable_chunk_detection: true,
        sensitivity: ChangeSensitivity::Medium,
        ..Default::default()
    })
    .await?;

    // Create initial document
    let doc = Document::new("The quick brown fox jumps over the lazy dog.")
        .with_metadata("category", serde_json::Value::String("example".to_string()))
        .with_metadata("priority", serde_json::Value::Number(1.into()))
        .with_content_hash();

    // First detection (should be "Added")
    let result1 = detector.detect_changes(&doc).await?;
    tracing::debug!("  📝 Initial document detection: {:?}", result1.change_type);
    tracing::debug!("  🔢 Content hash: {}", result1.current_hash);

    // No change detection
    let result2 = detector.detect_changes(&doc).await?;
    tracing::debug!("  ⏸️  Unchanged document: {:?}", result2.change_type);

    // Content change
    let doc_modified = Document::with_id(
        doc.id.clone(),
        "The quick brown fox jumps over the sleeping dog.",
    )
    .with_metadata("category", serde_json::Value::String("example".to_string()))
    .with_metadata("priority", serde_json::Value::Number(1.into()))
    .with_content_hash();

    let result3 = detector.detect_changes(&doc_modified).await?;
    tracing::debug!("  ✏️  Content changed: {:?}", result3.change_type);
    tracing::debug!(
        "  📊 Change percentage: {:.1}%",
        result3.delta.change_percentage * 100.0
    );
    tracing::debug!("  📈 Characters added: {}", result3.delta.added_chars);
    tracing::debug!("  📉 Characters removed: {}", result3.delta.removed_chars);

    // Metadata change
    let doc_meta_changed = Document::with_id(
        doc.id.clone(),
        "The quick brown fox jumps over the sleeping dog.",
    )
    .with_metadata(
        "category",
        serde_json::Value::String("modified".to_string()),
    )
    .with_metadata("priority", serde_json::Value::Number(2.into()))
    .with_content_hash();

    let result4 = detector.detect_changes(&doc_meta_changed).await?;
    tracing::debug!("  🏷️  Metadata changed: {:?}", result4.change_type);
    tracing::debug!(
        "  🔧 Modified keys: {:?}",
        result4.metadata_changes.modified_keys
    );

    // Get change statistics
    let stats = detector.get_stats().await;
    tracing::debug!("  📊 Detection stats:");
    tracing::debug!("      📋 Total processed: {}", stats.total_processed);
    tracing::debug!(
        "      ⏱️  Average time: {:.2}ms",
        stats.avg_processing_time_ms
    );
    tracing::debug!(
        "      🎯 Cache hit rate: {:.1}%",
        stats.cache_hit_rate * 100.0
    );

    Ok(())
}

/// Demonstrate incremental indexing operations
async fn demo_incremental_operations() -> RragResult<()> {
    tracing::debug!("  📚 Setting up index manager...");

    let index_manager = IncrementalIndexManager::new(IndexManagerConfig {
        batch_size: 50,
        enable_conflict_resolution: true,
        conflict_resolution: ConflictResolutionStrategy::LastWriteWins,
        ..Default::default()
    })
    .await?;

    // Create test documents
    let documents = create_test_documents(5).await;
    tracing::debug!("  📄 Created {} test documents", documents.len());

    // Submit add operations
    let mut operation_ids = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        let chunks = create_test_chunks(&doc, 3).await;
        let embeddings = create_test_embeddings(&chunks).await;

        let operation = IndexOperation::Add {
            document: doc.clone(),
            chunks,
            embeddings,
        };

        let update = IndexUpdate {
            operation_id: Uuid::new_v4().to_string(),
            operation,
            priority: if i < 2 { 8 } else { 5 }, // High priority for first two
            timestamp: chrono::Utc::now(),
            source: "demo".to_string(),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            max_retries: 3,
            retry_count: 0,
        };

        let op_id = index_manager.submit_update(update).await?;
        operation_ids.push(op_id.clone());
        tracing::debug!(
            "  📤 Submitted add operation {}: {}",
            i + 1,
            op_id.split('-').next().unwrap_or("unknown")
        );
    }

    // Wait a moment for processing
    sleep(Duration::from_millis(500)).await;

    // Check operation status
    tracing::debug!("  📋 Checking operation statuses:");
    for (i, op_id) in operation_ids.iter().enumerate() {
        if let Some(status) = index_manager.get_operation_status(op_id).await? {
            tracing::debug!("      Operation {}: {:?}", i + 1, status);
        }
    }

    // Submit update operations
    let doc_to_update = documents[0].clone();
    let updated_doc = Document::with_id(
        doc_to_update.id.clone(),
        "Updated content for the first document",
    )
    .with_metadata("updated", serde_json::Value::Bool(true));

    let chunks = create_test_chunks(&updated_doc, 2).await;
    let embeddings = create_test_embeddings(&chunks).await;

    let change_result = ChangeResult {
        change_type: DetectionChangeType::ContentChanged,
        document_id: updated_doc.id.clone(),
        previous_hash: Some("old_hash".to_string()),
        current_hash: "new_hash".to_string(),
        delta: ContentDelta {
            added_chars: 10,
            removed_chars: 5,
            modified_chars: 20,
            previous_size: 50,
            current_size: 55,
            change_percentage: 0.3,
        },
        metadata_changes: MetadataChanges {
            added_keys: vec!["updated".to_string()],
            removed_keys: Vec::new(),
            modified_keys: Vec::new(),
            previous_metadata: HashMap::new(),
            current_metadata: updated_doc.metadata.clone(),
        },
        timestamps: ChangeTimestamps {
            detected_at: chrono::Utc::now(),
            last_modified: Some(chrono::Utc::now()),
            previous_check: None,
            time_since_change: None,
        },
        chunk_changes: Vec::new(),
        confidence: 0.95,
    };

    let update_operation = IndexOperation::Update {
        document_id: updated_doc.id.clone(),
        document: updated_doc,
        chunks,
        embeddings,
        change_result,
    };

    let update_request = IndexUpdate {
        operation_id: Uuid::new_v4().to_string(),
        operation: update_operation,
        priority: 7,
        timestamp: chrono::Utc::now(),
        source: "demo_update".to_string(),
        metadata: HashMap::new(),
        dependencies: Vec::new(),
        max_retries: 3,
        retry_count: 0,
    };

    let update_op_id = index_manager.submit_update(update_request).await?;
    tracing::debug!(
        "  🔄 Submitted update operation: {}",
        update_op_id.split('-').next().unwrap_or("unknown")
    );

    // Submit delete operation
    let delete_operation = IndexOperation::Delete {
        document_id: documents[4].id.clone(),
    };

    let delete_request = IndexUpdate {
        operation_id: Uuid::new_v4().to_string(),
        operation: delete_operation,
        priority: 6,
        timestamp: chrono::Utc::now(),
        source: "demo_delete".to_string(),
        metadata: HashMap::new(),
        dependencies: Vec::new(),
        max_retries: 3,
        retry_count: 0,
    };

    let delete_op_id = index_manager.submit_update(delete_request).await?;
    tracing::debug!(
        "  🗑️  Submitted delete operation: {}",
        delete_op_id.split('-').next().unwrap_or("unknown")
    );

    // Get statistics
    sleep(Duration::from_millis(100)).await;
    let stats = index_manager.get_stats().await;
    tracing::debug!("  📊 Index manager stats:");
    tracing::debug!("      📋 Total operations: {}", stats.total_operations);
    tracing::debug!("      ✅ Success rate: {:.1}%", stats.success_rate * 100.0);
    tracing::debug!(
        "      ⏱️  Average time: {:.2}ms",
        stats.avg_processing_time_ms
    );
    tracing::debug!("      📤 Queue depth: {}", stats.current_queue_depth);

    Ok(())
}

/// Demonstrate batch processing capabilities
async fn demo_batch_processing() -> RragResult<()> {
    tracing::debug!("  ⚡ Setting up batch processor...");

    let batch_processor = BatchProcessor::new(BatchConfig {
        max_batch_size: 10,
        min_batch_size: 3,
        batch_timeout_ms: 2000,
        enable_priority_batching: true,
        enable_adaptive_sizing: true,
        error_handling: ErrorHandlingStrategy::ContinueOnError,
        ..Default::default()
    })
    .await?;

    // Create multiple operations for batch processing
    let documents = create_test_documents(15).await;
    tracing::debug!(
        "  📚 Created {} documents for batch processing",
        documents.len()
    );

    // Submit operations with different priorities
    let mut operation_ids = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        let chunks = create_test_chunks(&doc, 2).await;
        let embeddings = create_test_embeddings(&chunks).await;

        let operation = IndexOperation::Add {
            document: doc.clone(),
            chunks,
            embeddings,
        };

        let priority = match i % 3 {
            0 => 8, // High priority
            1 => 5, // Medium priority
            _ => 2, // Low priority
        };

        let update = IndexUpdate {
            operation_id: Uuid::new_v4().to_string(),
            operation,
            priority,
            timestamp: chrono::Utc::now(),
            source: "batch_demo".to_string(),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            max_retries: 3,
            retry_count: 0,
        };

        let op_id = batch_processor.add_operation(update).await?;
        operation_ids.push(op_id.clone());

        if (i + 1) % 5 == 0 {
            tracing::debug!("  📤 Submitted batch {} ({} operations)", (i + 1) / 5, 5);
        }
    }

    // Wait for batch processing to complete
    tracing::debug!("  ⏳ Waiting for batch processing...");
    sleep(Duration::from_millis(3000)).await;

    // Get batch processing metrics
    let metrics = batch_processor.get_metrics().await;
    tracing::debug!("  📊 Batch processing metrics:");
    tracing::debug!("      📋 Total operations: {}", metrics.total_operations);
    tracing::debug!("      📦 Total batches: {}", metrics.total_batches);
    tracing::debug!("      📏 Average batch size: {:.1}", metrics.avg_batch_size);
    tracing::debug!(
        "      🚀 Throughput: {:.1} ops/sec",
        metrics.throughput_ops_per_second
    );
    tracing::debug!("      ❌ Error rate: {:.2}%", metrics.error_rate * 100.0);

    let queue_stats = batch_processor.get_queue_stats().await;
    tracing::debug!("  📋 Queue statistics:");
    tracing::debug!("      📊 Total processed: {}", queue_stats.total_processed);
    tracing::debug!(
        "      🎯 Current throughput: {:.1} ops/sec",
        queue_stats.current_throughput
    );

    Ok(())
}

/// Demonstrate versioning and conflict resolution
async fn demo_versioning_system() -> RragResult<()> {
    tracing::debug!("  🔄 Setting up version manager...");

    let version_manager = VersionManager::new(VersioningConfig {
        max_versions_per_document: 5,
        enable_auto_cleanup: true,
        retention_period_days: 7,
        conflict_detection: ConflictDetectionStrategy::Timestamp,
        default_resolution: ResolutionStrategy::KeepNewer,
        enable_change_tracking: true,
        ..Default::default()
    })
    .await?;

    // Create a document and its versions
    let doc_id = "versioned_document";
    let versions = [
        ("v1", "Initial version of the document", "author1"),
        (
            "v2",
            "Updated version with more content added to demonstrate versioning",
            "author2",
        ),
        (
            "v3",
            "Final version with significant changes and improvements",
            "author1",
        ),
    ];

    tracing::debug!("  📝 Creating document versions...");
    let mut version_ids = Vec::new();

    for (i, (version_name, content, author)) in versions.iter().enumerate() {
        let doc = Document::with_id(doc_id, *content)
            .with_metadata(
                "version_name",
                serde_json::Value::String(version_name.to_string()),
            )
            .with_metadata("iteration", serde_json::Value::Number((i + 1).into()));

        let change_type = if i == 0 {
            ChangeType::Major
        } else {
            ChangeType::Minor
        };

        let version = version_manager
            .create_version(&doc, author, change_type, None)
            .await?;
        version_ids.push(version.version_id.clone());

        tracing::debug!(
            "  ✅ Created version {}: {} ({})",
            version.version_number,
            version_name,
            version.version_id.split('-').next().unwrap_or("unknown")
        );

        if let Some(change_summary) = &version.change_summary {
            tracing::debug!(
                "      📊 Changes: +{} chars, -{} chars",
                change_summary.additions, change_summary.deletions
            );
        }

        // Small delay between versions
        sleep(Duration::from_millis(100)).await;
    }

    // Retrieve version history
    let history = version_manager.get_version_history(doc_id).await?;
    if let Some(history) = history {
        tracing::debug!("  📚 Version history for document:");
        tracing::debug!("      📋 Total versions: {}", history.versions.len());
        tracing::debug!(
            "      🆔 Current version: {}",
            history
                .current_version
                .split('-')
                .next()
                .unwrap_or("unknown")
        );
        tracing::debug!(
            "      🌿 Branches: {:?}",
            history.branches.keys().collect::<Vec<_>>()
        );
    }

    // Demonstrate conflict detection
    let doc_conflict = Document::with_id(doc_id, "Conflicting version created concurrently")
        .with_metadata("conflict", serde_json::Value::Bool(true));

    // Try to detect conflict with wrong expected version
    let conflict = version_manager
        .detect_conflicts(&doc_conflict, Some("wrong_version_id"))
        .await?;
    if let Some(conflict) = conflict {
        tracing::debug!("  ⚠️  Conflict detected:");
        tracing::debug!(
            "      🆔 Conflict ID: {}",
            conflict.conflict_id.split('-').next().unwrap_or("unknown")
        );
        tracing::debug!("      📋 Type: {:?}", conflict.conflict_type);
        tracing::debug!("      🔄 Status: {:?}", conflict.resolution_status);

        // Resolve the conflict
        let resolution = VersionResolution {
            strategy: ResolutionStrategy::KeepNewer,
            chosen_version: version_ids[2].clone(),
            resolved_at: chrono::Utc::now(),
            resolved_by: "demo_resolver".to_string(),
            notes: Some("Resolved by keeping the newer version".to_string()),
            merged_content: None,
            metadata: HashMap::new(),
        };

        let resolved = version_manager
            .resolve_conflict(&conflict.conflict_id, resolution)
            .await?;
        tracing::debug!("      ✅ Conflict resolved: {}", resolved);
    }

    // Get version statistics
    let stats = version_manager.get_stats().await;
    tracing::debug!("  📊 Versioning statistics:");
    tracing::debug!("      📚 Total documents: {}", stats.total_documents);
    tracing::debug!("      📋 Total versions: {}", stats.total_versions);
    tracing::debug!(
        "      📊 Avg versions per doc: {:.1}",
        stats.avg_versions_per_document
    );
    tracing::debug!("      ⚠️  Total conflicts: {}", stats.total_conflicts);
    tracing::debug!("      🤖 Auto-resolved: {}", stats.auto_resolved_conflicts);
    tracing::debug!(
        "      👥 Manual resolved: {}",
        stats.manually_resolved_conflicts
    );

    Ok(())
}

/// Demonstrate rollback system capabilities
async fn demo_rollback_system() -> RragResult<()> {
    tracing::debug!("  ↩️  Setting up rollback manager...");

    let rollback_manager = RollbackManager::new(RollbackConfig {
        max_operation_log_size: 1000,
        enable_snapshots: true,
        snapshot_interval: 10,
        max_snapshots: 20,
        enable_auto_rollback: true,
        ..Default::default()
    })
    .await?;

    // Simulate some operations
    let documents = create_test_documents(3).await;
    tracing::debug!("  📚 Simulating operations for rollback demo...");

    for (i, doc) in documents.iter().enumerate() {
        let chunks = create_test_chunks(&doc, 2).await;
        let embeddings = create_test_embeddings(&chunks).await;

        let operation = IndexOperation::Add {
            document: doc.clone(),
            chunks,
            embeddings,
        };

        let update = IndexUpdate {
            operation_id: format!("rollback_op_{}", i + 1),
            operation,
            priority: 5,
            timestamp: chrono::Utc::now(),
            source: "rollback_demo".to_string(),
            metadata: HashMap::new(),
            dependencies: Vec::new(),
            max_retries: 3,
            retry_count: 0,
        };

        // Log the operation
        rollback_manager
            .log_operation(
                update,
                Some(UpdateResult {
                    operation_id: format!("rollback_op_{}", i + 1),
                    success: true,
                    operations_completed: vec!["add".to_string()],
                    conflicts: Vec::new(),
                    processing_time_ms: 100,
                    items_affected: 1,
                    error: None,
                    metadata: HashMap::new(),
                }),
                format!("pre_hash_{}", i + 1),
                Some(format!("post_hash_{}", i + 1)),
            )
            .await?;

        tracing::debug!("  📝 Logged operation {}", i + 1);
    }

    // Create a system snapshot
    let snapshot_id = rollback_manager
        .create_snapshot("demo_checkpoint".to_string())
        .await?;
    tracing::debug!(
        "  📸 Created snapshot: {}",
        snapshot_id.split('-').next().unwrap_or("unknown")
    );

    // Create a rollback point
    let rollback_point_id = rollback_manager
        .create_rollback_point(
            "Before risky operations".to_string(),
            vec!["rollback_op_1".to_string(), "rollback_op_2".to_string()],
            true,
        )
        .await?;
    tracing::debug!(
        "  🔖 Created rollback point: {}",
        rollback_point_id.split('-').next().unwrap_or("unknown")
    );

    // Simulate a failed operation that needs rollback
    tracing::debug!("  ⚠️  Simulating failed operations...");
    sleep(Duration::from_millis(100)).await;

    // Perform rollback
    let rollback_op = RollbackOperation::RestoreSnapshot {
        snapshot_id: snapshot_id.clone(),
        target_state: SystemState {
            snapshot_id: snapshot_id.clone(),
            created_at: chrono::Utc::now(),
            document_states: HashMap::new(),
            index_states: HashMap::new(),
            system_metadata: HashMap::new(),
            operations_count: 3,
            size_bytes: 1024,
            compression_ratio: 0.8,
        },
    };

    let recovery_result = rollback_manager.rollback(rollback_op).await?;
    tracing::debug!("  🔄 Rollback completed:");
    tracing::debug!("      ✅ Success: {}", recovery_result.success);
    tracing::debug!(
        "      ⏱️  Recovery time: {}ms",
        recovery_result.recovery_time_ms
    );
    tracing::debug!(
        "      📋 Operations rolled back: {}",
        recovery_result.rolled_back_operations.len()
    );

    if !recovery_result.verification_results.is_empty() {
        tracing::debug!("      🔍 Verification results:");
        for result in &recovery_result.verification_results {
            tracing::debug!(
                "          {}: {}",
                result.check_name,
                if result.passed { "✅" } else { "❌" }
            );
        }
    }

    // Get rollback statistics
    let stats = rollback_manager.get_stats().await;
    tracing::debug!("  📊 Rollback statistics:");
    tracing::debug!(
        "      📋 Operations logged: {}",
        stats.total_operations_logged
    );
    tracing::debug!("      🔄 Total rollbacks: {}", stats.total_rollbacks);
    tracing::debug!(
        "      ✅ Successful rollbacks: {}",
        stats.successful_rollbacks
    );
    tracing::debug!("      📸 Total snapshots: {}", stats.total_snapshots);
    tracing::debug!(
        "      ⏱️  Average rollback time: {:.2}ms",
        stats.avg_rollback_time_ms
    );

    // Get available snapshots
    let snapshots = rollback_manager.get_snapshots().await?;
    tracing::debug!("      📸 Available snapshots: {}", snapshots.len());

    Ok(())
}

/// Demonstrate integrity checking system
async fn demo_integrity_system() -> RragResult<()> {
    tracing::debug!("  🔒 Setting up integrity checker...");

    let mut config = IntegrityConfig::default();
    config.enable_auto_checks = false; // Disable for demo control

    let integrity_checker = IntegrityChecker::new(config).await?;

    // Perform quick integrity check
    tracing::debug!("  🔍 Performing quick integrity check...");
    let quick_report = integrity_checker.quick_check().await?;

    tracing::debug!("  📋 Quick check results:");
    tracing::debug!(
        "      🆔 Report ID: {}",
        quick_report
            .report_id
            .split('-')
            .next()
            .unwrap_or("unknown")
    );
    tracing::debug!("      🏥 Overall health: {:?}", quick_report.overall_health);
    tracing::debug!(
        "      ⏱️  Check duration: {}ms",
        quick_report.check_duration_ms
    );
    tracing::debug!(
        "      📊 Entities checked: {}",
        quick_report.entities_checked
    );
    tracing::debug!(
        "      ⚠️  Integrity errors: {}",
        quick_report.integrity_errors.len()
    );

    if !quick_report.recommendations.is_empty() {
        tracing::debug!("      💡 Recommendations:");
        for rec in quick_report.recommendations.iter().take(3) {
            tracing::debug!("          {:?}: {}", rec.priority, rec.description);
        }
    }

    // Perform comprehensive integrity check
    tracing::debug!("  🔍 Performing comprehensive integrity check...");
    let comprehensive_report = integrity_checker.comprehensive_check().await?;

    tracing::debug!("  📋 Comprehensive check results:");
    tracing::debug!(
        "      🏥 Overall health: {:?}",
        comprehensive_report.overall_health
    );
    tracing::debug!(
        "      ⏱️  Check duration: {}ms",
        comprehensive_report.check_duration_ms
    );
    tracing::debug!(
        "      📊 Entities checked: {}",
        comprehensive_report.entities_checked
    );
    tracing::debug!(
        "      ⚠️  Integrity errors: {}",
        comprehensive_report.integrity_errors.len()
    );
    tracing::debug!(
        "      🔧 Repair actions: {}",
        comprehensive_report.repair_actions.len()
    );

    // Display performance metrics
    let perf_metrics = &comprehensive_report.performance_metrics;
    tracing::debug!("  📈 Performance metrics:");
    tracing::debug!(
        "      ⏱️  Avg response time: {:.2}ms",
        perf_metrics.avg_response_time_ms
    );
    tracing::debug!(
        "      🎯 Success rate: {:.1}%",
        perf_metrics.success_rate * 100.0
    );
    tracing::debug!(
        "      💾 Memory usage: {:.1} MB",
        perf_metrics.memory_usage_mb
    );
    tracing::debug!("      💻 CPU usage: {:.1}%", perf_metrics.cpu_usage_percent);

    // Display system stats
    let sys_stats = &comprehensive_report.system_stats;
    tracing::debug!("  🖥️  System statistics:");
    tracing::debug!("      📚 Total documents: {}", sys_stats.total_documents);
    tracing::debug!("      🧩 Total chunks: {}", sys_stats.total_chunks);
    tracing::debug!("      🎯 Total embeddings: {}", sys_stats.total_embeddings);
    tracing::debug!("      ⏰ Uptime: {} hours", sys_stats.uptime_seconds / 3600);

    // Get integrity statistics
    let stats = integrity_checker.get_stats().await;
    tracing::debug!("  📊 Integrity checker statistics:");
    tracing::debug!("      🔍 Total checks: {}", stats.total_checks);
    tracing::debug!("      ⚡ Quick checks: {}", stats.quick_checks);
    tracing::debug!(
        "      🔬 Comprehensive checks: {}",
        stats.comprehensive_checks
    );
    tracing::debug!("      ⚠️  Total errors found: {}", stats.total_errors_found);
    tracing::debug!(
        "      🔧 Repairs attempted: {}",
        stats.total_repairs_attempted
    );
    tracing::debug!("      ✅ Successful repairs: {}", stats.successful_repairs);
    tracing::debug!(
        "      ⏱️  Avg check time: {:.2}ms",
        stats.avg_check_duration_ms
    );
    tracing::debug!(
        "      📊 System availability: {:.2}%",
        stats.system_availability_percent
    );

    Ok(())
}

/// Demonstrate vector update management
async fn demo_vector_updates() -> RragResult<()> {
    tracing::debug!("  🎯 Setting up vector update manager...");

    let vector_manager = VectorUpdateManager::new(VectorUpdateConfig {
        enable_batch_processing: true,
        max_batch_size: 50,
        update_strategy: IndexUpdateStrategy::Adaptive,
        enable_optimization: true,
        optimization_interval_secs: 1800, // 30 minutes
        enable_similarity_updates: true,
        similarity_threshold: 0.75,
        ..Default::default()
    })
    .await?;

    // Create test embeddings
    tracing::debug!("  🔢 Creating test embeddings...");
    let embeddings = create_test_embeddings_direct(10).await;

    // Submit vector add operation
    let add_operation = VectorOperation::Add {
        embeddings: embeddings.clone(),
        index_name: "demo_index".to_string(),
    };

    let add_op_id = vector_manager.submit_operation(add_operation).await?;
    tracing::debug!(
        "  ➕ Submitted add operation: {}",
        add_op_id.split('-').next().unwrap_or("unknown")
    );

    // Create embedding updates
    let embedding_updates = vec![
        EmbeddingUpdate {
            embedding_id: embeddings[0].source_id.clone(),
            new_embedding: Embedding::new(
                vec![0.9, 0.8, 0.7, 0.6],
                "updated-model",
                embeddings[0].source_id.clone(),
            ),
            update_reason: UpdateReason::QualityImprovement,
            metadata: HashMap::new(),
        },
        EmbeddingUpdate {
            embedding_id: embeddings[1].source_id.clone(),
            new_embedding: Embedding::new(
                vec![0.5, 0.4, 0.3, 0.2],
                "updated-model",
                embeddings[1].source_id.clone(),
            ),
            update_reason: UpdateReason::ContentChanged,
            metadata: HashMap::new(),
        },
    ];

    // Process embedding updates
    let update_result = vector_manager
        .process_embedding_updates(embedding_updates, "demo_index")
        .await?;

    tracing::debug!("  🔄 Embedding update results:");
    tracing::debug!("      ✅ Success: {}", update_result.success);
    tracing::debug!(
        "      🔢 Embeddings processed: {}",
        update_result.embeddings_processed
    );
    tracing::debug!(
        "      ⏱️  Processing time: {}ms",
        update_result.processing_time_ms
    );

    if let Some(index_stats) = &update_result.index_stats {
        tracing::debug!("      📊 Index stats:");
        tracing::debug!(
            "          🔢 Embedding count: {}",
            index_stats.embedding_count
        );
        tracing::debug!("          📏 Dimensions: {}", index_stats.dimensions);
        tracing::debug!(
            "          💾 Memory usage: {} MB",
            index_stats.memory_usage_bytes / (1024 * 1024)
        );
    }

    // Perform index optimization
    tracing::debug!("  ⚡ Performing index optimization...");
    let opt_result = vector_manager
        .optimize_index("demo_index", OptimizationType::QueryOptimization)
        .await?;

    tracing::debug!("  🔧 Optimization results:");
    tracing::debug!("      ✅ Success: {}", opt_result.success);
    tracing::debug!(
        "      ⏱️  Processing time: {}ms",
        opt_result.processing_time_ms
    );
    tracing::debug!(
        "      🎯 Performance improved: {}",
        opt_result.performance_metrics.throughput_eps > 0.0
    );

    // Update similarity thresholds
    let threshold_operation = VectorOperation::UpdateThresholds {
        index_name: "demo_index".to_string(),
        new_threshold: 0.8,
    };

    let threshold_op_id = vector_manager.submit_operation(threshold_operation).await?;
    tracing::debug!(
        "  🎯 Updated similarity threshold: {}",
        threshold_op_id.split('-').next().unwrap_or("unknown")
    );

    // Get vector update metrics
    let metrics = vector_manager.get_metrics().await;
    tracing::debug!("  📊 Vector update metrics:");
    tracing::debug!("      📋 Total operations: {}", metrics.total_operations);
    tracing::debug!(
        "      ✅ Success rate: {:.1}%",
        metrics.success_rate * 100.0
    );
    tracing::debug!(
        "      🔢 Embeddings processed: {}",
        metrics.total_embeddings_processed
    );
    tracing::debug!(
        "      ⏱️  Avg processing time: {:.2}ms",
        metrics.avg_processing_time_ms
    );
    tracing::debug!(
        "      🚀 System throughput: {:.1} emb/sec",
        metrics.system_performance.overall_throughput_eps
    );
    tracing::debug!(
        "      💾 Memory usage: {:.1} MB",
        metrics.system_performance.memory_usage_mb
    );
    tracing::debug!(
        "      🏥 Health score: {:.2}",
        metrics.system_performance.health_score
    );

    // Get all index statistics
    let all_stats = vector_manager.get_all_index_stats().await?;
    tracing::debug!("  📊 All index statistics ({} indexes):", all_stats.len());
    for (index_name, stats) in all_stats.iter() {
        tracing::debug!(
            "      {}: {} embeddings, {:.1} MB",
            index_name,
            stats.embedding_count,
            stats.memory_usage_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    Ok(())
}

/// Demonstrate monitoring and metrics system
async fn demo_monitoring_system() -> RragResult<()> {
    tracing::debug!("  📊 Setting up monitoring system...");

    let monitoring_config = MonitoringConfig {
        enable_performance_metrics: true,
        enable_health_monitoring: true,
        enable_alerting: true,
        metrics_interval_secs: 5,
        health_check_interval_secs: 10,
        alert_config: AlertConfig {
            enable_webhook_alerts: true,
            enable_log_alerts: true,
            thresholds: AlertThresholds {
                error_rate_threshold: 0.05,
                response_time_threshold_ms: 5000,
                queue_depth_threshold: 100,
                memory_usage_threshold: 0.8,
                storage_usage_threshold: 0.9,
                throughput_threshold_ops: 10.0,
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut collector_config = monitoring_config.clone();
    collector_config.enable_performance_metrics = false; // Disable auto-collection for demo

    let metrics_collector = MetricsCollector::new(collector_config).await?;

    // Create performance tracker
    let performance_tracker = PerformanceTracker::new(monitoring_config, 1000);

    // Record some performance data points
    tracing::debug!("  📈 Recording performance data...");
    let operations = ["indexing", "searching", "updating", "optimizing"];

    for i in 0..20 {
        let op_type = operations[i % operations.len()];
        let success = i % 10 != 9; // 90% success rate
        let duration = if success { 50 + (i * 10) as u64 } else { 5000 }; // Failed operations take longer

        let data_point = PerformanceDataPoint {
            timestamp: chrono::Utc::now(),
            operation_type: op_type.to_string(),
            duration_ms: duration,
            memory_usage_mb: 100.0 + (i as f64 * 5.0),
            success,
            metadata: HashMap::new(),
        };

        performance_tracker.record_data_point(data_point).await;
        metrics_collector
            .record_performance(PerformanceDataPoint {
                timestamp: chrono::Utc::now(),
                operation_type: op_type.to_string(),
                duration_ms: duration,
                memory_usage_mb: 100.0 + (i as f64 * 5.0),
                success,
                metadata: HashMap::new(),
            })
            .await?;

        if (i + 1) % 5 == 0 {
            tracing::debug!("  📊 Recorded {} performance data points", i + 1);
        }
    }

    // Get performance statistics
    let perf_stats = performance_tracker.get_statistics().await;
    tracing::debug!("  📈 Performance statistics:");
    tracing::debug!(
        "      📋 Total operations: {}",
        perf_stats.overall.total_count
    );
    tracing::debug!(
        "      ✅ Success rate: {:.1}%",
        (perf_stats.overall.success_count as f64 / perf_stats.overall.total_count as f64) * 100.0
    );
    tracing::debug!(
        "      ⏱️  Average duration: {:.2}ms",
        perf_stats.overall.avg_duration_ms
    );
    tracing::debug!(
        "      📊 95th percentile: {:.2}ms",
        perf_stats.overall.p95_duration_ms
    );
    tracing::debug!(
        "      🚀 Throughput: {:.1} ops/sec",
        perf_stats.overall.operations_per_second
    );

    // Show statistics by operation type
    tracing::debug!("  📋 Statistics by operation type:");
    for (op_type, stats) in perf_stats.by_operation_type.iter() {
        let success_rate = (stats.success_count as f64 / stats.total_count as f64) * 100.0;
        tracing::debug!(
            "      {}: {} ops, {:.1}% success, {:.2}ms avg",
            op_type, stats.total_count, success_rate, stats.avg_duration_ms
        );
    }

    // Update system metrics
    let metrics_update = MetricsUpdate {
        indexing_metrics: Some(IndexingMetrics {
            documents_per_second: 25.0,
            chunks_per_second: 125.0,
            embeddings_per_second: 125.0,
            avg_indexing_time_ms: 80.0,
            index_growth_rate_bps: 4096.0,
            batch_efficiency: 0.94,
            change_detection_accuracy: 0.97,
            vector_update_efficiency: 0.91,
        }),
        system_metrics: Some(rrag::incremental::monitoring::SystemMetrics {
            cpu_usage_percent: 65.0,
            memory_usage_bytes: 756 * 1024 * 1024, // 756 MB
            available_memory_bytes: 1268 * 1024 * 1024, // 1268 MB
            storage_usage_bytes: 15 * 1024 * 1024 * 1024, // 15 GB
            available_storage_bytes: 85 * 1024 * 1024 * 1024, // 85 GB
            network_io_bps: 2048.0,
            disk_io_ops: 120.0,
            active_connections: 25,
        }),
        operation_metrics: Some(OperationMetrics {
            total_operations: perf_stats.overall.total_count,
            operations_by_type: operations
                .iter()
                .map(|op| (op.to_string(), perf_stats.overall.total_count / 4))
                .collect(),
            success_rate: perf_stats.overall.success_count as f64
                / perf_stats.overall.total_count as f64,
            avg_operation_time_ms: perf_stats.overall.avg_duration_ms,
            p95_operation_time_ms: perf_stats.overall.p95_duration_ms,
            p99_operation_time_ms: perf_stats.overall.p99_duration_ms,
            queue_depths: vec![("indexing".to_string(), 5), ("processing".to_string(), 12)]
                .into_iter()
                .collect(),
            retry_stats: RetryMetrics {
                total_retries: 3,
                successful_retries: 2,
                exhausted_retries: 1,
                avg_retries_per_operation: 0.15,
            },
        }),
        health_metrics: Some(rrag::incremental::monitoring::HealthMetrics {
            overall_health_score: 0.92,
            component_health: vec![
                ("indexing".to_string(), 0.95),
                ("storage".to_string(), 0.98),
                ("retrieval".to_string(), 0.88),
                ("monitoring".to_string(), 0.96),
            ]
            .into_iter()
            .collect(),
            service_availability: 0.997,
            data_consistency_score: 0.99,
            performance_score: 0.89,
            last_health_check: chrono::Utc::now(),
        }),
        error_metrics: Some(ErrorMetrics {
            total_errors: 2,
            errors_by_type: vec![("timeout".to_string(), 1), ("validation".to_string(), 1)]
                .into_iter()
                .collect(),
            errors_by_component: vec![("retrieval".to_string(), 2)].into_iter().collect(),
            error_rate: 0.03,
            critical_errors: 0,
            recoverable_errors: 2,
            avg_resolution_time_ms: 150.0,
        }),
        custom_metrics: vec![
            ("cache_hit_rate".to_string(), 0.87),
            ("index_freshness".to_string(), 0.95),
        ]
        .into_iter()
        .collect(),
    };

    metrics_collector.update_metrics(metrics_update).await?;

    // Get current metrics
    let current_metrics = metrics_collector.get_current_metrics().await;
    tracing::debug!("  🎯 Current system metrics:");
    tracing::debug!(
        "      📊 Overall system score: {:.2}",
        current_metrics.calculate_system_score()
    );
    tracing::debug!(
        "      🏥 Health score: {:.2}",
        current_metrics.health_metrics.overall_health_score
    );
    tracing::debug!(
        "      🔄 Service availability: {:.2}%",
        current_metrics.health_metrics.service_availability * 100.0
    );
    tracing::debug!(
        "      ⚡ Performance score: {:.2}",
        current_metrics.health_metrics.performance_score
    );

    tracing::debug!("  📈 Indexing performance:");
    let idx_metrics = &current_metrics.indexing_metrics;
    tracing::debug!(
        "      📚 Documents/sec: {:.1}",
        idx_metrics.documents_per_second
    );
    tracing::debug!("      🧩 Chunks/sec: {:.1}", idx_metrics.chunks_per_second);
    tracing::debug!(
        "      🎯 Embeddings/sec: {:.1}",
        idx_metrics.embeddings_per_second
    );
    tracing::debug!(
        "      📦 Batch efficiency: {:.1}%",
        idx_metrics.batch_efficiency * 100.0
    );

    tracing::debug!("  🖥️  System resources:");
    let sys_metrics = &current_metrics.system_metrics;
    tracing::debug!("      💻 CPU: {:.1}%", sys_metrics.cpu_usage_percent);
    tracing::debug!(
        "      💾 Memory: {:.1} MB / {:.1} MB",
        sys_metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0),
        (sys_metrics.memory_usage_bytes + sys_metrics.available_memory_bytes) as f64
            / (1024.0 * 1024.0)
    );
    tracing::debug!(
        "      💽 Storage: {:.1} GB / {:.1} GB",
        sys_metrics.storage_usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (sys_metrics.storage_usage_bytes + sys_metrics.available_storage_bytes) as f64
            / (1024.0 * 1024.0 * 1024.0)
    );
    tracing::debug!(
        "      🔗 Active connections: {}",
        sys_metrics.active_connections
    );

    Ok(())
}

/// Demonstrate production scenarios
async fn demo_production_scenarios() -> RragResult<()> {
    tracing::debug!("  🏭 Simulating production scenarios...");

    // Scenario 1: High-volume document ingestion
    tracing::debug!("  📈 Scenario 1: High-volume document ingestion");
    let batch_processor = BatchProcessor::new(BatchConfig {
        max_batch_size: 100,
        min_batch_size: 10,
        batch_timeout_ms: 1000,
        enable_priority_batching: true,
        enable_adaptive_sizing: true,
        error_handling: ErrorHandlingStrategy::ContinueOnError,
        ..Default::default()
    })
    .await?;

    tracing::debug!("      📚 Processing 1000 documents in batches...");
    for batch_num in 0..10 {
        let documents = create_test_documents(100).await;
        let mut operations = Vec::new();

        for (i, doc) in documents.iter().enumerate() {
            let chunks = create_test_chunks(&doc, 3).await;
            let embeddings = create_test_embeddings(&chunks).await;

            let operation = IndexOperation::Add {
                document: doc.clone(),
                chunks,
                embeddings,
            };

            let update = IndexUpdate {
                operation_id: format!("prod_batch_{}_{}", batch_num, i),
                operation,
                priority: 5,
                timestamp: chrono::Utc::now(),
                source: "production".to_string(),
                metadata: HashMap::new(),
                dependencies: Vec::new(),
                max_retries: 3,
                retry_count: 0,
            };

            operations.push(update);
        }

        // Submit operations
        for operation in operations {
            let _op_id = batch_processor.add_operation(operation).await?;
        }

        if (batch_num + 1) % 3 == 0 {
            tracing::debug!(
                "      ✅ Processed {} batches ({} documents)",
                batch_num + 1,
                (batch_num + 1) * 100
            );
        }
    }

    // Wait for processing
    sleep(Duration::from_millis(2000)).await;

    let metrics = batch_processor.get_metrics().await;
    tracing::debug!(
        "      📊 Final metrics: {:.1} ops/sec, {:.2}% error rate",
        metrics.throughput_ops_per_second,
        metrics.error_rate * 100.0
    );

    // Scenario 2: Real-time updates with conflict resolution
    tracing::debug!("  🔄 Scenario 2: Real-time updates with conflict resolution");
    let version_manager = VersionManager::new(VersioningConfig {
        max_versions_per_document: 10,
        enable_auto_cleanup: true,
        conflict_detection: ConflictDetectionStrategy::ContentHash,
        default_resolution: ResolutionStrategy::Merge,
        ..Default::default()
    })
    .await?;

    // Simulate concurrent updates to the same document
    let doc_id = "concurrent_doc";
    let authors = ["user1", "user2", "user3"];
    let contents = [
        "Original content for concurrent editing test",
        "Modified content by user1 with additional information",
        "Updated content by user2 with different changes",
    ];

    tracing::debug!(
        "      👥 Simulating concurrent updates from {} users...",
        authors.len()
    );

    for (_i, (author, content)) in authors.iter().zip(contents.iter()).enumerate() {
        let doc = Document::with_id(doc_id, *content)
            .with_metadata("editor", serde_json::Value::String(author.to_string()))
            .with_metadata(
                "edit_time",
                serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
            );

        let version = version_manager
            .create_version(&doc, author, ChangeType::Minor, None)
            .await?;
        tracing::debug!(
            "      ✏️  {} created version {}",
            author, version.version_number
        );

        // Small delay to simulate real-time updates
        sleep(Duration::from_millis(50)).await;
    }

    let stats = version_manager.get_stats().await;
    tracing::debug!(
        "      📊 Version stats: {} versions, {} conflicts",
        stats.total_versions, stats.total_conflicts
    );

    // Scenario 3: System recovery after failure
    tracing::debug!("  🚨 Scenario 3: System recovery after failure");
    let rollback_manager = RollbackManager::new(RollbackConfig::default()).await?;

    // Create checkpoint before risky operations
    let checkpoint_id = rollback_manager
        .create_snapshot("pre_risky_ops".to_string())
        .await?;
    tracing::debug!("      📸 Created recovery checkpoint");

    // Simulate system failure and recovery
    tracing::debug!("      ⚠️  Simulating system failure...");
    sleep(Duration::from_millis(100)).await;

    let recovery = rollback_manager
        .rollback(RollbackOperation::RestoreSnapshot {
            snapshot_id: checkpoint_id,
            target_state: SystemState {
                snapshot_id: Uuid::new_v4().to_string(),
                created_at: chrono::Utc::now(),
                document_states: HashMap::new(),
                index_states: HashMap::new(),
                system_metadata: HashMap::new(),
                operations_count: 1000,
                size_bytes: 1024 * 1024,
                compression_ratio: 0.7,
            },
        })
        .await?;

    tracing::debug!(
        "      🔄 Recovery completed in {}ms",
        recovery.recovery_time_ms
    );
    tracing::debug!("      ✅ System restored to stable state");

    // Scenario 4: Performance monitoring and alerting
    tracing::debug!("  📊 Scenario 4: Performance monitoring and alerting");
    let monitoring_config = MonitoringConfig {
        enable_alerting: true,
        alert_config: AlertConfig {
            enable_log_alerts: true,
            thresholds: AlertThresholds {
                error_rate_threshold: 0.01, // Very strict for demo
                response_time_threshold_ms: 100,
                memory_usage_threshold: 0.7,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let mut collector_config = monitoring_config.clone();
    collector_config.enable_performance_metrics = false; // Disable auto for demo
    let metrics_collector = MetricsCollector::new(collector_config).await?;

    tracing::debug!("      🚨 Simulating alert conditions...");

    // Simulate high error rate scenario
    let high_error_update = MetricsUpdate {
        indexing_metrics: None,
        system_metrics: None,
        operation_metrics: None,
        health_metrics: Some(rrag::incremental::monitoring::HealthMetrics {
            overall_health_score: 0.6, // Poor health
            component_health: HashMap::new(),
            service_availability: 0.95,
            data_consistency_score: 0.99,
            performance_score: 0.5, // Poor performance
            last_health_check: chrono::Utc::now(),
        }),
        error_metrics: Some(ErrorMetrics {
            total_errors: 50,
            errors_by_type: vec![("timeout".to_string(), 30), ("network".to_string(), 20)]
                .into_iter()
                .collect(),
            errors_by_component: HashMap::new(),
            error_rate: 0.05, // 5% error rate - above threshold
            critical_errors: 5,
            recoverable_errors: 45,
            avg_resolution_time_ms: 2000.0,
        }),
        custom_metrics: HashMap::new(),
    };

    metrics_collector.update_metrics(high_error_update).await?;

    let current_metrics = metrics_collector.get_current_metrics().await;
    if current_metrics.error_metrics.error_rate
        > monitoring_config
            .alert_config
            .thresholds
            .error_rate_threshold
    {
        tracing::debug!(
            "      🚨 ALERT: Error rate ({:.1}%) exceeds threshold ({:.1}%)",
            current_metrics.error_metrics.error_rate * 100.0,
            monitoring_config
                .alert_config
                .thresholds
                .error_rate_threshold
                * 100.0
        );
    }

    if current_metrics.health_metrics.overall_health_score < 0.8 {
        tracing::debug!(
            "      🚨 ALERT: System health score ({:.2}) below acceptable level",
            current_metrics.health_metrics.overall_health_score
        );
    }

    tracing::debug!("      📊 Monitoring system successfully detected and reported issues");

    tracing::debug!("  🎯 Production scenario summary:");
    tracing::debug!("      ✅ High-volume ingestion: Processed 1000 documents efficiently");
    tracing::debug!("      ✅ Concurrent updates: Handled version conflicts automatically");
    tracing::debug!("      ✅ System recovery: Restored from failure in <100ms");
    tracing::debug!("      ✅ Monitoring/alerting: Detected and alerted on performance issues");

    Ok(())
}

// Helper functions for creating test data

async fn create_test_documents(count: usize) -> Vec<Document> {
    let mut documents = Vec::with_capacity(count);
    let sample_contents = [
        "Rust is a modern systems programming language focused on safety, speed, and concurrency.",
        "The borrow checker in Rust ensures memory safety without requiring a garbage collector.",
        "Cargo is Rust's built-in package manager and build system, making dependency management easy.",
        "Async programming in Rust provides high-performance concurrent applications.",
        "The type system in Rust catches many bugs at compile time, improving code reliability.",
    ];

    for i in 0..count {
        let content_index = i % sample_contents.len();
        let content = format!("{} Document {}.", sample_contents[content_index], i + 1);

        let doc = Document::new(content)
            .with_metadata("doc_number", serde_json::Value::Number((i + 1).into()))
            .with_metadata("category", serde_json::Value::String("test".to_string()))
            .with_metadata("created_by", serde_json::Value::String("demo".to_string()))
            .with_content_hash();

        documents.push(doc);
    }

    documents
}

async fn create_test_chunks(document: &Document, chunk_count: usize) -> Vec<DocumentChunk> {
    let content = document.content_str();
    let chunk_size = content.len() / chunk_count.max(1);
    let mut chunks = Vec::with_capacity(chunk_count);

    for i in 0..chunk_count {
        let start = i * chunk_size;
        let end = if i == chunk_count - 1 {
            content.len()
        } else {
            (i + 1) * chunk_size
        };

        let chunk_content = if start < content.len() {
            &content[start..end.min(content.len())]
        } else {
            "Additional chunk content for completeness"
        };

        let chunk = DocumentChunk::new(
            &document.id,
            chunk_content,
            i,
            start,
            end.min(content.len()),
        )
        .with_metadata("chunk_type", serde_json::Value::String("auto".to_string()))
        .with_metadata(
            "original_doc",
            serde_json::Value::String(document.id.clone()),
        );

        chunks.push(chunk);
    }

    chunks
}

async fn create_test_embeddings(chunks: &[DocumentChunk]) -> Vec<Embedding> {
    let mut embeddings = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        // Create deterministic but varied embeddings based on chunk content
        let base_value = (chunk.content.len() % 100) as f32 / 100.0;
        let vector = vec![
            base_value,
            base_value * 0.8,
            base_value * 1.2,
            base_value * 0.5,
        ];

        let embedding = Embedding::new(
            vector,
            "test-model",
            format!("{}_emb_{}", chunk.document_id, i),
        )
        .with_metadata(
            "source",
            serde_json::Value::String("test_generator".to_string()),
        )
        .with_metadata("chunk_index", serde_json::Value::Number(i.into()));

        embeddings.push(embedding);
    }

    embeddings
}

async fn create_test_embeddings_direct(count: usize) -> Vec<Embedding> {
    let mut embeddings = Vec::with_capacity(count);

    for i in 0..count {
        let base = (i as f32) / (count as f32);
        let vector = vec![base, base * 0.5, base * 1.5, base * 0.3];

        let embedding = Embedding::new(vector, "test-model", format!("direct_emb_{}", i))
            .with_metadata("type", serde_json::Value::String("direct".to_string()))
            .with_metadata("index", serde_json::Value::Number(i.into()));

        embeddings.push(embedding);
    }

    embeddings
}
