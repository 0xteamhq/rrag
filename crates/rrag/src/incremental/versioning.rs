//! # Document Versioning System
//!
//! Provides comprehensive document versioning and conflict resolution for incremental indexing.
//! Handles version tracking, conflict detection, and resolution strategies.

use crate::{Document, Metadata, RragResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Versioning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Maximum versions to keep per document
    pub max_versions_per_document: usize,

    /// Enable automatic version cleanup
    pub enable_auto_cleanup: bool,

    /// Version retention period in days
    pub retention_period_days: u32,

    /// Conflict detection strategy
    pub conflict_detection: ConflictDetectionStrategy,

    /// Default resolution strategy
    pub default_resolution: ResolutionStrategy,

    /// Enable version compression
    pub enable_version_compression: bool,

    /// Enable detailed change tracking
    pub enable_change_tracking: bool,
}

/// Conflict detection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictDetectionStrategy {
    /// Compare version numbers only
    VersionNumber,
    /// Compare timestamps
    Timestamp,
    /// Compare content hashes
    ContentHash,
    /// Compare version vectors (for distributed systems)
    VersionVector,
    /// Custom detection logic
    Custom(String),
}

/// Resolution strategies for conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Always keep the newer version
    KeepNewer,
    /// Always keep the older version
    KeepOlder,
    /// Merge changes when possible
    Merge,
    /// Manual resolution required
    Manual,
    /// Use metadata priority
    MetadataPriority(String),
    /// Custom resolution logic
    Custom(String),
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            max_versions_per_document: 10,
            enable_auto_cleanup: true,
            retention_period_days: 30,
            conflict_detection: ConflictDetectionStrategy::Timestamp,
            default_resolution: ResolutionStrategy::KeepNewer,
            enable_version_compression: true,
            enable_change_tracking: true,
        }
    }
}

/// Document version representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentVersion {
    /// Unique version identifier
    pub version_id: String,

    /// Document ID this version belongs to
    pub document_id: String,

    /// Version number (incremental)
    pub version_number: u64,

    /// Version timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Author/source of this version
    pub author: String,

    /// Content hash for integrity checking
    pub content_hash: String,

    /// Metadata hash for change detection
    pub metadata_hash: String,

    /// Version tags for categorization
    pub tags: Vec<String>,

    /// Change summary
    pub change_summary: Option<ChangeSummary>,

    /// Parent version (for branching)
    pub parent_version: Option<String>,

    /// Branch information
    pub branch: String,

    /// Version metadata
    pub metadata: Metadata,

    /// Compressed document data (if compression enabled)
    pub compressed_data: Option<Vec<u8>>,

    /// Size of the version data
    pub data_size_bytes: u64,
}

/// Summary of changes in a version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeSummary {
    /// Type of change
    pub change_type: ChangeType,

    /// Number of lines/characters added
    pub additions: usize,

    /// Number of lines/characters removed
    pub deletions: usize,

    /// Number of lines/characters modified
    pub modifications: usize,

    /// Affected sections/chunks
    pub affected_sections: Vec<String>,

    /// Change description
    pub description: Option<String>,
}

/// Types of changes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    /// Major content changes
    Major,
    /// Minor content changes
    Minor,
    /// Metadata-only changes
    Metadata,
    /// Formatting changes
    Formatting,
    /// Content restructuring
    Restructure,
    /// Initial creation
    Initial,
}

/// Version conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionConflict {
    /// Conflict identifier
    pub conflict_id: String,

    /// Document ID where conflict occurred
    pub document_id: String,

    /// Conflicting versions
    pub conflicting_versions: Vec<String>,

    /// Conflict detection timestamp
    pub detected_at: chrono::DateTime<chrono::Utc>,

    /// Type of conflict
    pub conflict_type: ConflictType,

    /// Conflict resolution status
    pub resolution_status: ResolutionStatus,

    /// Automatic resolution applied
    pub auto_resolution: Option<VersionResolution>,

    /// Manual resolution if needed
    pub manual_resolution: Option<VersionResolution>,

    /// Conflict context
    pub context: HashMap<String, serde_json::Value>,
}

/// Types of version conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Concurrent modifications
    ConcurrentModification,
    /// Version number mismatch
    VersionMismatch,
    /// Timestamp inconsistency
    TimestampInconsistency,
    /// Content hash mismatch
    ContentMismatch,
    /// Branch merge conflict
    BranchMergeConflict,
    /// Dependency conflict
    DependencyConflict,
}

/// Resolution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionStatus {
    /// Conflict is unresolved
    Unresolved,
    /// Automatically resolved
    AutoResolved,
    /// Manually resolved
    ManuallyResolved,
    /// Resolution in progress
    InProgress,
    /// Resolution failed
    Failed,
}

/// Version resolution details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionResolution {
    /// Resolution strategy used
    pub strategy: ResolutionStrategy,

    /// Chosen version
    pub chosen_version: String,

    /// Resolution timestamp
    pub resolved_at: chrono::DateTime<chrono::Utc>,

    /// Resolution author
    pub resolved_by: String,

    /// Resolution notes
    pub notes: Option<String>,

    /// Merged content if applicable
    pub merged_content: Option<Document>,

    /// Resolution metadata
    pub metadata: Metadata,
}

/// Version history for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// Document ID
    pub document_id: String,

    /// All versions (ordered by version number)
    pub versions: VecDeque<DocumentVersion>,

    /// Current active version
    pub current_version: String,

    /// Branch information
    pub branches: HashMap<String, String>, // branch_name -> latest_version_id

    /// Version tree (parent-child relationships)
    pub version_tree: HashMap<String, Vec<String>>, // parent_id -> child_ids

    /// History metadata
    pub metadata: Metadata,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Version manager for handling document versions
pub struct VersionManager {
    /// Configuration
    config: VersioningConfig,

    /// Document version histories
    version_histories: Arc<RwLock<HashMap<String, VersionHistory>>>,

    /// Active conflicts
    conflicts: Arc<RwLock<HashMap<String, VersionConflict>>>,

    /// Version statistics
    stats: Arc<RwLock<VersionStats>>,

    /// Background task handles
    task_handles: Arc<tokio::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

/// Version management statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStats {
    /// Total documents with versions
    pub total_documents: usize,

    /// Total versions stored
    pub total_versions: u64,

    /// Average versions per document
    pub avg_versions_per_document: f64,

    /// Total conflicts detected
    pub total_conflicts: u64,

    /// Auto-resolved conflicts
    pub auto_resolved_conflicts: u64,

    /// Manually resolved conflicts
    pub manually_resolved_conflicts: u64,

    /// Unresolved conflicts
    pub unresolved_conflicts: usize,

    /// Storage usage in bytes
    pub storage_usage_bytes: u64,

    /// Compression ratio (if enabled)
    pub compression_ratio: f64,

    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl VersionManager {
    /// Create a new version manager
    pub async fn new(config: VersioningConfig) -> RragResult<Self> {
        let manager = Self {
            config,
            version_histories: Arc::new(RwLock::new(HashMap::new())),
            conflicts: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VersionStats {
                total_documents: 0,
                total_versions: 0,
                avg_versions_per_document: 0.0,
                total_conflicts: 0,
                auto_resolved_conflicts: 0,
                manually_resolved_conflicts: 0,
                unresolved_conflicts: 0,
                storage_usage_bytes: 0,
                compression_ratio: 1.0,
                last_updated: chrono::Utc::now(),
            })),
            task_handles: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        };

        manager.start_background_tasks().await?;
        Ok(manager)
    }

    /// Create a new version of a document
    pub async fn create_version(
        &self,
        document: &Document,
        author: &str,
        change_type: ChangeType,
        branch: Option<&str>,
    ) -> RragResult<DocumentVersion> {
        let version_id = Uuid::new_v4().to_string();
        let branch = branch.unwrap_or("main").to_string();

        // Get or create version history
        let mut histories = self.version_histories.write().await;
        let history = histories
            .entry(document.id.clone())
            .or_insert_with(|| VersionHistory {
                document_id: document.id.clone(),
                versions: VecDeque::new(),
                current_version: version_id.clone(),
                branches: HashMap::new(),
                version_tree: HashMap::new(),
                metadata: HashMap::new(),
                last_updated: chrono::Utc::now(),
            });

        // Determine version number
        let version_number = history.versions.len() as u64 + 1;

        // Get parent version
        let parent_version = history.branches.get(&branch).cloned();

        // Create content and metadata hashes
        let content_hash = self.compute_hash(document.content_str()).await?;
        let metadata_json = serde_json::to_string(&document.metadata)?;
        let metadata_hash = self.compute_hash(&metadata_json).await?;

        // Create change summary
        let change_summary = if self.config.enable_change_tracking {
            self.compute_change_summary(document, &parent_version, change_type.clone(), history)
                .await?
        } else {
            None
        };

        // Compress data if enabled
        let (compressed_data, data_size) = if self.config.enable_version_compression {
            let data = serde_json::to_vec(document)?;
            let compressed = self.compress_data(&data).await?;
            let size = compressed.len() as u64;
            (Some(compressed), size)
        } else {
            let data = serde_json::to_vec(document)?;
            (None, data.len() as u64)
        };

        // Create version
        let version = DocumentVersion {
            version_id: version_id.clone(),
            document_id: document.id.clone(),
            version_number,
            created_at: chrono::Utc::now(),
            author: author.to_string(),
            content_hash,
            metadata_hash,
            tags: Vec::new(),
            change_summary,
            parent_version: parent_version.clone(),
            branch: branch.clone(),
            metadata: document.metadata.clone(),
            compressed_data,
            data_size_bytes: data_size,
        };

        // Add to history
        history.versions.push_back(version.clone());
        history.current_version = version_id.clone();
        history.branches.insert(branch, version_id.clone());
        history.last_updated = chrono::Utc::now();

        // Update version tree
        if let Some(parent) = &parent_version {
            history
                .version_tree
                .entry(parent.clone())
                .or_insert_with(Vec::new)
                .push(version_id.clone());
        }

        // Cleanup old versions if necessary
        if history.versions.len() > self.config.max_versions_per_document {
            history.versions.pop_front();
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_versions += 1;
            stats.total_documents = histories.len();
            stats.avg_versions_per_document =
                stats.total_versions as f64 / stats.total_documents as f64;
            stats.storage_usage_bytes += data_size;
            stats.last_updated = chrono::Utc::now();
        }

        Ok(version)
    }

    /// Get a specific version of a document
    pub async fn get_version(
        &self,
        document_id: &str,
        version_id: &str,
    ) -> RragResult<Option<DocumentVersion>> {
        let histories = self.version_histories.read().await;

        if let Some(history) = histories.get(document_id) {
            for version in &history.versions {
                if version.version_id == version_id {
                    return Ok(Some(version.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Get the current version of a document
    pub async fn get_current_version(
        &self,
        document_id: &str,
    ) -> RragResult<Option<DocumentVersion>> {
        let histories = self.version_histories.read().await;

        if let Some(history) = histories.get(document_id) {
            return self
                .get_version(document_id, &history.current_version)
                .await;
        }

        Ok(None)
    }

    /// Get version history for a document
    pub async fn get_version_history(
        &self,
        document_id: &str,
    ) -> RragResult<Option<VersionHistory>> {
        let histories = self.version_histories.read().await;
        Ok(histories.get(document_id).cloned())
    }

    /// Detect conflicts when updating a document
    pub async fn detect_conflicts(
        &self,
        document: &Document,
        expected_version: Option<&str>,
    ) -> RragResult<Option<VersionConflict>> {
        let histories = self.version_histories.read().await;

        if let Some(history) = histories.get(&document.id) {
            if let Some(expected) = expected_version {
                if history.current_version != expected {
                    // Version conflict detected
                    let conflict_id = Uuid::new_v4().to_string();
                    let conflict = VersionConflict {
                        conflict_id,
                        document_id: document.id.clone(),
                        conflicting_versions: vec![
                            history.current_version.clone(),
                            expected.to_string(),
                        ],
                        detected_at: chrono::Utc::now(),
                        conflict_type: ConflictType::VersionMismatch,
                        resolution_status: ResolutionStatus::Unresolved,
                        auto_resolution: None,
                        manual_resolution: None,
                        context: HashMap::new(),
                    };

                    return Ok(Some(conflict));
                }
            }
        }

        Ok(None)
    }

    /// Resolve a version conflict
    pub async fn resolve_conflict(
        &self,
        conflict_id: &str,
        resolution: VersionResolution,
    ) -> RragResult<bool> {
        let mut conflicts = self.conflicts.write().await;

        if let Some(conflict) = conflicts.get_mut(conflict_id) {
            conflict.manual_resolution = Some(resolution);
            conflict.resolution_status = ResolutionStatus::ManuallyResolved;

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.manually_resolved_conflicts += 1;
                stats.unresolved_conflicts = conflicts
                    .values()
                    .filter(|c| c.resolution_status == ResolutionStatus::Unresolved)
                    .count();
            }

            return Ok(true);
        }

        Ok(false)
    }

    /// Get all unresolved conflicts
    pub async fn get_unresolved_conflicts(&self) -> RragResult<Vec<VersionConflict>> {
        let conflicts = self.conflicts.read().await;
        Ok(conflicts
            .values()
            .filter(|c| c.resolution_status == ResolutionStatus::Unresolved)
            .cloned()
            .collect())
    }

    /// Get version statistics
    pub async fn get_stats(&self) -> VersionStats {
        self.stats.read().await.clone()
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        let handles = self.task_handles.lock().await;
        let all_running = handles.iter().all(|handle| !handle.is_finished());

        let stats = self.get_stats().await;
        let healthy_stats = stats.unresolved_conflicts < 1000; // Arbitrary threshold

        Ok(all_running && healthy_stats)
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(&self) -> RragResult<()> {
        let mut handles = self.task_handles.lock().await;

        if self.config.enable_auto_cleanup {
            handles.push(self.start_cleanup_task().await);
        }

        handles.push(self.start_conflict_auto_resolution_task().await);

        Ok(())
    }

    /// Start version cleanup task
    async fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let version_histories = Arc::clone(&self.version_histories);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // 1 hour

            loop {
                interval.tick().await;

                let cutoff_date = chrono::Utc::now()
                    - chrono::Duration::days(config.retention_period_days as i64);
                let mut histories = version_histories.write().await;

                for history in histories.values_mut() {
                    history
                        .versions
                        .retain(|version| version.created_at > cutoff_date);
                }
            }
        })
    }

    /// Start automatic conflict resolution task
    async fn start_conflict_auto_resolution_task(&self) -> tokio::task::JoinHandle<()> {
        let conflicts = Arc::clone(&self.conflicts);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // 5 minutes

            loop {
                interval.tick().await;

                let mut conflicts_guard = conflicts.write().await;
                let mut resolved_count = 0;

                for conflict in conflicts_guard.values_mut() {
                    if conflict.resolution_status == ResolutionStatus::Unresolved {
                        // Apply automatic resolution based on strategy
                        if let Some(auto_resolution) =
                            Self::apply_auto_resolution(conflict, &config.default_resolution).await
                        {
                            conflict.auto_resolution = Some(auto_resolution);
                            conflict.resolution_status = ResolutionStatus::AutoResolved;
                            resolved_count += 1;
                        }
                    }
                }

                // Update statistics
                if resolved_count > 0 {
                    let mut stats_guard = stats.write().await;
                    stats_guard.auto_resolved_conflicts += resolved_count;
                    stats_guard.unresolved_conflicts = conflicts_guard
                        .values()
                        .filter(|c| c.resolution_status == ResolutionStatus::Unresolved)
                        .count();
                }
            }
        })
    }

    /// Apply automatic conflict resolution
    async fn apply_auto_resolution(
        conflict: &VersionConflict,
        strategy: &ResolutionStrategy,
    ) -> Option<VersionResolution> {
        match strategy {
            ResolutionStrategy::KeepNewer => {
                // Choose the version with the most recent timestamp
                // This is simplified - would need access to version data
                Some(VersionResolution {
                    strategy: strategy.clone(),
                    chosen_version: conflict.conflicting_versions[0].clone(), // Simplified
                    resolved_at: chrono::Utc::now(),
                    resolved_by: "auto_resolver".to_string(),
                    notes: Some("Automatically resolved by keeping newer version".to_string()),
                    merged_content: None,
                    metadata: HashMap::new(),
                })
            }
            ResolutionStrategy::KeepOlder => Some(VersionResolution {
                strategy: strategy.clone(),
                chosen_version: conflict.conflicting_versions.last().unwrap().clone(),
                resolved_at: chrono::Utc::now(),
                resolved_by: "auto_resolver".to_string(),
                notes: Some("Automatically resolved by keeping older version".to_string()),
                merged_content: None,
                metadata: HashMap::new(),
            }),
            ResolutionStrategy::Manual => None, // Cannot auto-resolve manual conflicts
            _ => None,                          // Other strategies would need more implementation
        }
    }

    /// Compute content hash
    async fn compute_hash(&self, content: &str) -> RragResult<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Compute change summary between versions
    async fn compute_change_summary(
        &self,
        document: &Document,
        parent_version_id: &Option<String>,
        change_type: ChangeType,
        history: &VersionHistory,
    ) -> RragResult<Option<ChangeSummary>> {
        if let Some(parent_id) = parent_version_id {
            // Find parent version and compare
            if let Some(parent_version) =
                history.versions.iter().find(|v| v.version_id == *parent_id)
            {
                // Simple change detection - in production, would use proper diff algorithms
                let current_size = document.content_str().len();
                let estimated_previous_size = parent_version.data_size_bytes as usize;

                let (additions, deletions) = if current_size > estimated_previous_size {
                    (current_size - estimated_previous_size, 0)
                } else {
                    (0, estimated_previous_size - current_size)
                };

                return Ok(Some(ChangeSummary {
                    change_type,
                    additions,
                    deletions,
                    modifications: std::cmp::min(current_size, estimated_previous_size),
                    affected_sections: Vec::new(), // Would be computed with proper diff
                    description: None,
                }));
            }
        }

        // No parent version - initial creation
        Ok(Some(ChangeSummary {
            change_type: ChangeType::Initial,
            additions: document.content_str().len(),
            deletions: 0,
            modifications: 0,
            affected_sections: Vec::new(),
            description: Some("Initial version".to_string()),
        }))
    }

    /// Compress version data (placeholder implementation)
    async fn compress_data(&self, data: &[u8]) -> RragResult<Vec<u8>> {
        // In production, would use actual compression (gzip, lz4, etc.)
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    #[tokio::test]
    async fn test_version_manager_creation() {
        let config = VersioningConfig::default();
        let manager = VersionManager::new(config).await.unwrap();
        assert!(manager.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_create_version() {
        let manager = VersionManager::new(VersioningConfig::default())
            .await
            .unwrap();
        let doc = Document::new("Test content");

        let version = manager
            .create_version(&doc, "test_author", ChangeType::Initial, None)
            .await
            .unwrap();

        assert_eq!(version.document_id, doc.id);
        assert_eq!(version.version_number, 1);
        assert_eq!(version.author, "test_author");
        assert_eq!(version.branch, "main");
    }

    #[tokio::test]
    async fn test_version_retrieval() {
        let manager = VersionManager::new(VersioningConfig::default())
            .await
            .unwrap();
        let doc = Document::new("Test content");

        let version = manager
            .create_version(&doc, "test_author", ChangeType::Initial, None)
            .await
            .unwrap();

        // Test get_version
        let retrieved = manager
            .get_version(&doc.id, &version.version_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retrieved.version_id, version.version_id);

        // Test get_current_version
        let current = manager.get_current_version(&doc.id).await.unwrap().unwrap();
        assert_eq!(current.version_id, version.version_id);
    }

    #[tokio::test]
    async fn test_multiple_versions() {
        let manager = VersionManager::new(VersioningConfig::default())
            .await
            .unwrap();
        let doc1 = Document::with_id("test_doc", "Initial content");
        let doc2 = Document::with_id("test_doc", "Updated content");

        // Create first version
        let version1 = manager
            .create_version(&doc1, "author1", ChangeType::Initial, None)
            .await
            .unwrap();
        assert_eq!(version1.version_number, 1);

        // Create second version
        let version2 = manager
            .create_version(&doc2, "author2", ChangeType::Major, None)
            .await
            .unwrap();
        assert_eq!(version2.version_number, 2);

        // Check current version is the latest
        let current = manager
            .get_current_version("test_doc")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.version_id, version2.version_id);

        // Check version history
        let history = manager
            .get_version_history("test_doc")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(history.versions.len(), 2);
    }

    #[tokio::test]
    async fn test_conflict_detection() {
        let manager = VersionManager::new(VersioningConfig::default())
            .await
            .unwrap();
        let doc = Document::new("Test content");

        // Create initial version
        let version1 = manager
            .create_version(&doc, "author1", ChangeType::Initial, None)
            .await
            .unwrap();

        // Try to update with wrong expected version
        let conflict = manager
            .detect_conflicts(&doc, Some("wrong_version_id"))
            .await
            .unwrap();
        assert!(conflict.is_some());

        let conflict = conflict.unwrap();
        assert_eq!(conflict.document_id, doc.id);
        assert_eq!(conflict.conflict_type, ConflictType::VersionMismatch);

        // Try with correct version - should not conflict
        let no_conflict = manager
            .detect_conflicts(&doc, Some(&version1.version_id))
            .await
            .unwrap();
        assert!(no_conflict.is_none());
    }

    #[test]
    fn test_change_types() {
        let change_types = vec![
            ChangeType::Major,
            ChangeType::Minor,
            ChangeType::Metadata,
            ChangeType::Formatting,
            ChangeType::Restructure,
            ChangeType::Initial,
        ];

        // Ensure all change types are different
        for (i, type1) in change_types.iter().enumerate() {
            for (j, type2) in change_types.iter().enumerate() {
                if i != j {
                    assert_ne!(type1, type2);
                }
            }
        }
    }

    #[test]
    fn test_resolution_strategies() {
        let strategies = vec![
            ResolutionStrategy::KeepNewer,
            ResolutionStrategy::KeepOlder,
            ResolutionStrategy::Merge,
            ResolutionStrategy::Manual,
            ResolutionStrategy::MetadataPriority("priority".to_string()),
            ResolutionStrategy::Custom("custom_logic".to_string()),
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
