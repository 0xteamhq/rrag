//! # Change Detection System
//! 
//! Efficient change detection and delta processing for incremental indexing.
//! Uses content hashing, timestamps, and metadata comparison to detect changes.

use crate::{RragError, RragResult, Document, DocumentChunk, Metadata};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use tokio::sync::RwLock;

/// Change detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeDetectionConfig {
    /// Enable content hash comparison
    pub enable_content_hash: bool,
    
    /// Enable metadata change detection
    pub enable_metadata_detection: bool,
    
    /// Enable timestamp-based change detection
    pub enable_timestamp_detection: bool,
    
    /// Chunk-level change detection
    pub enable_chunk_detection: bool,
    
    /// Hash algorithm to use
    pub hash_algorithm: HashAlgorithm,
    
    /// Change detection sensitivity
    pub sensitivity: ChangeSensitivity,
    
    /// Maximum change history to keep
    pub max_change_history: usize,
}

/// Available hash algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HashAlgorithm {
    Default,
    Sha256,
    Blake3,
    Xxhash,
}

/// Change detection sensitivity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeSensitivity {
    /// Only detect major changes (content, structure)
    Low,
    /// Detect content and metadata changes
    Medium,
    /// Detect all changes including minor formatting
    High,
    /// Detect even whitespace and case changes
    Strict,
}

impl Default for ChangeDetectionConfig {
    fn default() -> Self {
        Self {
            enable_content_hash: true,
            enable_metadata_detection: true,
            enable_timestamp_detection: true,
            enable_chunk_detection: true,
            hash_algorithm: HashAlgorithm::Default,
            sensitivity: ChangeSensitivity::Medium,
            max_change_history: 1000,
        }
    }
}

/// Type of change detected
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Document was added (new)
    Added,
    /// Document content was modified
    ContentChanged,
    /// Document metadata was modified
    MetadataChanged,
    /// Document was moved or renamed
    Moved,
    /// Document was deleted
    Deleted,
    /// No change detected
    NoChange,
    /// Multiple types of changes
    Multiple(Vec<ChangeType>),
}

/// Detailed change result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeResult {
    /// Type of change detected
    pub change_type: ChangeType,
    
    /// Document ID
    pub document_id: String,
    
    /// Previous content hash
    pub previous_hash: Option<String>,
    
    /// Current content hash
    pub current_hash: String,
    
    /// Delta information
    pub delta: ContentDelta,
    
    /// Metadata changes
    pub metadata_changes: MetadataChanges,
    
    /// Timestamp information
    pub timestamps: ChangeTimestamps,
    
    /// Chunk-level changes
    pub chunk_changes: Vec<ChunkChange>,
    
    /// Change confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Content delta information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentDelta {
    /// Added content (approximate)
    pub added_chars: usize,
    
    /// Removed content (approximate)
    pub removed_chars: usize,
    
    /// Modified content (approximate)
    pub modified_chars: usize,
    
    /// Total content size before change
    pub previous_size: usize,
    
    /// Total content size after change
    pub current_size: usize,
    
    /// Change percentage (0.0 to 1.0)
    pub change_percentage: f64,
}

/// Metadata changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataChanges {
    /// Added metadata keys
    pub added_keys: Vec<String>,
    
    /// Removed metadata keys
    pub removed_keys: Vec<String>,
    
    /// Modified metadata keys
    pub modified_keys: Vec<String>,
    
    /// Previous metadata (subset for comparison)
    pub previous_metadata: HashMap<String, serde_json::Value>,
    
    /// Current metadata (subset for comparison)
    pub current_metadata: HashMap<String, serde_json::Value>,
}

/// Change timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeTimestamps {
    /// When the change was detected
    pub detected_at: chrono::DateTime<chrono::Utc>,
    
    /// Last known modification time
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Previous check timestamp
    pub previous_check: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Time since last change
    pub time_since_change: Option<chrono::Duration>,
}

/// Chunk-level change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChange {
    /// Chunk index
    pub chunk_index: usize,
    
    /// Type of change for this chunk
    pub change_type: ChangeType,
    
    /// Chunk hash before change
    pub previous_hash: Option<String>,
    
    /// Chunk hash after change
    pub current_hash: String,
    
    /// Content delta for this chunk
    pub delta: ContentDelta,
}

/// Document change tracking entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChange {
    /// Document ID
    pub document_id: String,
    
    /// Change result
    pub change_result: ChangeResult,
    
    /// Version information
    pub version: u64,
    
    /// Change source/trigger
    pub source: String,
    
    /// Additional context
    pub context: HashMap<String, serde_json::Value>,
}

/// Change detection system
pub struct ChangeDetector {
    /// Configuration
    config: ChangeDetectionConfig,
    
    /// Document state cache
    document_cache: RwLock<HashMap<String, DocumentState>>,
    
    /// Change history
    change_history: RwLock<Vec<DocumentChange>>,
    
    /// Statistics
    stats: RwLock<ChangeDetectionStats>,
}

/// Cached document state for comparison
#[derive(Debug, Clone)]
struct DocumentState {
    /// Content hash
    content_hash: String,
    
    /// Metadata hash
    metadata_hash: String,
    
    /// Chunk hashes
    chunk_hashes: Vec<String>,
    
    /// Last check timestamp
    last_checked: chrono::DateTime<chrono::Utc>,
    
    /// Document metadata subset
    metadata_snapshot: Metadata,
    
    /// Content size
    content_size: usize,
    
    /// Version
    version: u64,
}

/// Change detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeDetectionStats {
    /// Total documents processed
    pub total_processed: u64,
    
    /// Changes detected by type
    pub changes_by_type: HashMap<String, u64>,
    
    /// Average processing time
    pub avg_processing_time_ms: f64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// False positive rate (estimated)
    pub false_positive_rate: f64,
    
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl ChangeDetector {
    /// Create a new change detector
    pub async fn new(config: ChangeDetectionConfig) -> RragResult<Self> {
        Ok(Self {
            config,
            document_cache: RwLock::new(HashMap::new()),
            change_history: RwLock::new(Vec::new()),
            stats: RwLock::new(ChangeDetectionStats {
                total_processed: 0,
                changes_by_type: HashMap::new(),
                avg_processing_time_ms: 0.0,
                cache_hit_rate: 0.0,
                false_positive_rate: 0.0,
                last_updated: chrono::Utc::now(),
            }),
        })
    }

    /// Detect changes in a document
    pub async fn detect_changes(&self, document: &Document) -> RragResult<ChangeResult> {
        let start_time = std::time::Instant::now();
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_processed += 1;
        }

        // Get current document state
        let current_state = self.compute_document_state(document, None).await?;
        
        // Get previous state from cache
        let cache = self.document_cache.read().await;
        let previous_state = cache.get(&document.id);
        
        let change_result = match previous_state {
            Some(prev_state) => {
                self.compare_states(&document.id, prev_state, &current_state).await?
            }
            None => {
                // New document
                ChangeResult {
                    change_type: ChangeType::Added,
                    document_id: document.id.clone(),
                    previous_hash: None,
                    current_hash: current_state.content_hash.clone(),
                    delta: ContentDelta {
                        added_chars: current_state.content_size,
                        removed_chars: 0,
                        modified_chars: 0,
                        previous_size: 0,
                        current_size: current_state.content_size,
                        change_percentage: 1.0,
                    },
                    metadata_changes: MetadataChanges {
                        added_keys: current_state.metadata_snapshot.keys().cloned().collect(),
                        removed_keys: Vec::new(),
                        modified_keys: Vec::new(),
                        previous_metadata: HashMap::new(),
                        current_metadata: current_state.metadata_snapshot.clone(),
                    },
                    timestamps: ChangeTimestamps {
                        detected_at: chrono::Utc::now(),
                        last_modified: Some(document.created_at),
                        previous_check: None,
                        time_since_change: None,
                    },
                    chunk_changes: Vec::new(),
                    confidence: 1.0,
                }
            }
        };

        // Update cache with new state
        drop(cache);
        {
            let mut cache = self.document_cache.write().await;
            cache.insert(document.id.clone(), current_state.clone());
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            let change_type_str = format!("{:?}", change_result.change_type);
            *stats.changes_by_type.entry(change_type_str).or_insert(0) += 1;
            
            let processing_time = start_time.elapsed().as_millis() as f64;
            stats.avg_processing_time_ms = 
                (stats.avg_processing_time_ms + processing_time) / 2.0;
            stats.last_updated = chrono::Utc::now();
        }

        // Add to change history if change detected
        if change_result.change_type != ChangeType::NoChange {
            let document_change = DocumentChange {
                document_id: document.id.clone(),
                change_result: change_result.clone(),
                version: current_state.version,
                source: "change_detector".to_string(),
                context: HashMap::new(),
            };

            let mut history = self.change_history.write().await;
            history.push(document_change);
            
            // Limit history size
            if history.len() > self.config.max_change_history {
                history.remove(0);
            }
        }

        Ok(change_result)
    }

    /// Detect changes with chunked document
    pub async fn detect_changes_with_chunks(
        &self, 
        document: &Document, 
        chunks: &[DocumentChunk]
    ) -> RragResult<ChangeResult> {
        let start_time = std::time::Instant::now();
        
        // Compute current state with chunks
        let current_state = self.compute_document_state(document, Some(chunks)).await?;
        
        // Get previous state
        let cache = self.document_cache.read().await;
        let previous_state = cache.get(&document.id);
        
        let mut change_result = match previous_state {
            Some(prev_state) => {
                self.compare_states(&document.id, prev_state, &current_state).await?
            }
            None => {
                // New document with chunks
                let chunk_changes: Vec<ChunkChange> = chunks.iter().enumerate().map(|(i, chunk)| {
                    ChunkChange {
                        chunk_index: i,
                        change_type: ChangeType::Added,
                        previous_hash: None,
                        current_hash: current_state.chunk_hashes[i].clone(),
                        delta: ContentDelta {
                            added_chars: chunk.content.len(),
                            removed_chars: 0,
                            modified_chars: 0,
                            previous_size: 0,
                            current_size: chunk.content.len(),
                            change_percentage: 1.0,
                        },
                    }
                }).collect();

                ChangeResult {
                    change_type: ChangeType::Added,
                    document_id: document.id.clone(),
                    previous_hash: None,
                    current_hash: current_state.content_hash.clone(),
                    delta: ContentDelta {
                        added_chars: current_state.content_size,
                        removed_chars: 0,
                        modified_chars: 0,
                        previous_size: 0,
                        current_size: current_state.content_size,
                        change_percentage: 1.0,
                    },
                    metadata_changes: MetadataChanges {
                        added_keys: current_state.metadata_snapshot.keys().cloned().collect(),
                        removed_keys: Vec::new(),
                        modified_keys: Vec::new(),
                        previous_metadata: HashMap::new(),
                        current_metadata: current_state.metadata_snapshot.clone(),
                    },
                    timestamps: ChangeTimestamps {
                        detected_at: chrono::Utc::now(),
                        last_modified: Some(document.created_at),
                        previous_check: None,
                        time_since_change: None,
                    },
                    chunk_changes,
                    confidence: 1.0,
                }
            }
        };

        // Add chunk-level analysis if enabled
        if self.config.enable_chunk_detection && change_result.chunk_changes.is_empty() {
            if let Some(prev_state) = previous_state {
                change_result.chunk_changes = self.analyze_chunk_changes(
                    &prev_state.chunk_hashes,
                    &current_state.chunk_hashes,
                    chunks
                ).await?;
            }
        }

        // Update cache
        drop(cache);
        {
            let mut cache = self.document_cache.write().await;
            cache.insert(document.id.clone(), current_state);
        }

        Ok(change_result)
    }

    /// Get change history for a document
    pub async fn get_change_history(&self, document_id: &str) -> RragResult<Vec<DocumentChange>> {
        let history = self.change_history.read().await;
        Ok(history.iter()
            .filter(|change| change.document_id == document_id)
            .cloned()
            .collect())
    }

    /// Get change detection statistics
    pub async fn get_stats(&self) -> ChangeDetectionStats {
        self.stats.read().await.clone()
    }

    /// Clear change history
    pub async fn clear_history(&self) -> RragResult<()> {
        let mut history = self.change_history.write().await;
        history.clear();
        Ok(())
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        // Simple health check - verify we can access our data structures
        let _cache = self.document_cache.read().await;
        let _history = self.change_history.read().await;
        let _stats = self.stats.read().await;
        Ok(true)
    }

    /// Compute document state for comparison
    async fn compute_document_state(
        &self, 
        document: &Document, 
        chunks: Option<&[DocumentChunk]>
    ) -> RragResult<DocumentState> {
        // Compute content hash
        let content_hash = self.compute_hash(document.content_str()).await?;
        
        // Compute metadata hash
        let metadata_json = serde_json::to_string(&document.metadata)
            .map_err(|e| RragError::serialization_with_message("document_metadata", e.to_string()))?;
        let metadata_hash = self.compute_hash(&metadata_json).await?;
        
        // Compute chunk hashes if provided
        let chunk_hashes = if let Some(chunks) = chunks {
            let mut hashes = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                let chunk_hash = self.compute_hash(&chunk.content).await?;
                hashes.push(chunk_hash);
            }
            hashes
        } else {
            Vec::new()
        };

        Ok(DocumentState {
            content_hash,
            metadata_hash,
            chunk_hashes,
            last_checked: chrono::Utc::now(),
            metadata_snapshot: document.metadata.clone(),
            content_size: document.content_str().len(),
            version: 1, // Would be managed by version system
        })
    }

    /// Compare two document states
    async fn compare_states(
        &self,
        document_id: &str,
        previous: &DocumentState,
        current: &DocumentState
    ) -> RragResult<ChangeResult> {
        let mut change_types = Vec::new();
        
        // Check content changes
        if previous.content_hash != current.content_hash {
            change_types.push(ChangeType::ContentChanged);
        }
        
        // Check metadata changes
        if previous.metadata_hash != current.metadata_hash {
            change_types.push(ChangeType::MetadataChanged);
        }

        let change_type = match change_types.len() {
            0 => ChangeType::NoChange,
            1 => change_types.into_iter().next().unwrap(),
            _ => ChangeType::Multiple(change_types),
        };

        // Compute content delta
        let delta = self.compute_content_delta(previous, current).await?;
        
        // Compute metadata changes
        let metadata_changes = self.compute_metadata_changes(
            &previous.metadata_snapshot,
            &current.metadata_snapshot
        ).await?;

        // Compute confidence score
        let confidence = self.compute_confidence(&change_type, &delta).await?;

        Ok(ChangeResult {
            change_type,
            document_id: document_id.to_string(),
            previous_hash: Some(previous.content_hash.clone()),
            current_hash: current.content_hash.clone(),
            delta,
            metadata_changes,
            timestamps: ChangeTimestamps {
                detected_at: chrono::Utc::now(),
                last_modified: None,
                previous_check: Some(previous.last_checked),
                time_since_change: Some(chrono::Utc::now() - previous.last_checked),
            },
            chunk_changes: Vec::new(), // Filled separately if needed
            confidence,
        })
    }

    /// Analyze chunk-level changes
    async fn analyze_chunk_changes(
        &self,
        previous_hashes: &[String],
        current_hashes: &[String],
        current_chunks: &[DocumentChunk]
    ) -> RragResult<Vec<ChunkChange>> {
        let mut chunk_changes = Vec::new();
        
        let max_len = std::cmp::max(previous_hashes.len(), current_hashes.len());
        
        for i in 0..max_len {
            let prev_hash = previous_hashes.get(i);
            let curr_hash = current_hashes.get(i);
            let chunk = current_chunks.get(i);
            
            let (change_type, current_hash, delta) = match (prev_hash, curr_hash, chunk) {
                (Some(prev), Some(curr), Some(chunk)) => {
                    if prev != curr {
                        let delta = ContentDelta {
                            added_chars: 0, // Would need more sophisticated diff
                            removed_chars: 0,
                            modified_chars: chunk.content.len(),
                            previous_size: chunk.content.len(), // Approximation
                            current_size: chunk.content.len(),
                            change_percentage: 0.5, // Approximation
                        };
                        (ChangeType::ContentChanged, curr.clone(), delta)
                    } else {
                        continue; // No change
                    }
                }
                (None, Some(curr), Some(chunk)) => {
                    let delta = ContentDelta {
                        added_chars: chunk.content.len(),
                        removed_chars: 0,
                        modified_chars: 0,
                        previous_size: 0,
                        current_size: chunk.content.len(),
                        change_percentage: 1.0,
                    };
                    (ChangeType::Added, curr.clone(), delta)
                }
                (Some(_), None, _) => {
                    let delta = ContentDelta {
                        added_chars: 0,
                        removed_chars: 0, // Would need previous chunk size
                        modified_chars: 0,
                        previous_size: 0,
                        current_size: 0,
                        change_percentage: 1.0,
                    };
                    (ChangeType::Deleted, String::new(), delta)
                }
                _ => continue,
            };
            
            chunk_changes.push(ChunkChange {
                chunk_index: i,
                change_type,
                previous_hash: prev_hash.cloned(),
                current_hash,
                delta,
            });
        }
        
        Ok(chunk_changes)
    }

    /// Compute content hash based on configuration
    async fn compute_hash(&self, content: &str) -> RragResult<String> {
        let normalized_content = match self.config.sensitivity {
            ChangeSensitivity::Low => {
                // Only hash significant content, ignore formatting
                content.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
                    .to_lowercase()
            }
            ChangeSensitivity::Medium => {
                // Normalize whitespace but preserve structure
                content.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
            }
            ChangeSensitivity::High => {
                // Preserve most formatting, normalize case
                content.to_lowercase()
            }
            ChangeSensitivity::Strict => {
                // Use content as-is
                content.to_string()
            }
        };

        match self.config.hash_algorithm {
            HashAlgorithm::Default => {
                let mut hasher = DefaultHasher::new();
                hasher.write(normalized_content.as_bytes());
                Ok(format!("{:x}", hasher.finish()))
            }
            HashAlgorithm::Sha256 => {
                // In production, use actual SHA256
                let mut hasher = DefaultHasher::new();
                hasher.write(normalized_content.as_bytes());
                Ok(format!("sha256:{:x}", hasher.finish()))
            }
            HashAlgorithm::Blake3 => {
                // In production, use actual BLAKE3
                let mut hasher = DefaultHasher::new();
                hasher.write(normalized_content.as_bytes());
                Ok(format!("blake3:{:x}", hasher.finish()))
            }
            HashAlgorithm::Xxhash => {
                // In production, use actual xxHash
                let mut hasher = DefaultHasher::new();
                hasher.write(normalized_content.as_bytes());
                Ok(format!("xxhash:{:x}", hasher.finish()))
            }
        }
    }

    /// Compute content delta between states
    async fn compute_content_delta(
        &self,
        previous: &DocumentState,
        current: &DocumentState
    ) -> RragResult<ContentDelta> {
        let size_diff = current.content_size as i64 - previous.content_size as i64;
        
        let (added_chars, removed_chars) = if size_diff > 0 {
            (size_diff as usize, 0)
        } else {
            (0, (-size_diff) as usize)
        };

        let change_percentage = if previous.content_size == 0 {
            1.0
        } else {
            (size_diff.abs() as f64) / (previous.content_size as f64)
        };

        Ok(ContentDelta {
            added_chars,
            removed_chars,
            modified_chars: std::cmp::min(previous.content_size, current.content_size),
            previous_size: previous.content_size,
            current_size: current.content_size,
            change_percentage: change_percentage.min(1.0),
        })
    }

    /// Compute metadata changes
    async fn compute_metadata_changes(
        &self,
        previous: &Metadata,
        current: &Metadata
    ) -> RragResult<MetadataChanges> {
        let prev_keys: HashSet<String> = previous.keys().cloned().collect();
        let curr_keys: HashSet<String> = current.keys().cloned().collect();

        let added_keys: Vec<String> = curr_keys.difference(&prev_keys).cloned().collect();
        let removed_keys: Vec<String> = prev_keys.difference(&curr_keys).cloned().collect();
        
        let mut modified_keys = Vec::new();
        for key in prev_keys.intersection(&curr_keys) {
            if previous.get(key) != current.get(key) {
                modified_keys.push(key.clone());
            }
        }

        Ok(MetadataChanges {
            added_keys,
            removed_keys,
            modified_keys,
            previous_metadata: previous.clone(),
            current_metadata: current.clone(),
        })
    }

    /// Compute change confidence score
    async fn compute_confidence(&self, change_type: &ChangeType, delta: &ContentDelta) -> RragResult<f64> {
        let base_confidence = match change_type {
            ChangeType::Added | ChangeType::Deleted => 1.0,
            ChangeType::NoChange => 1.0,
            ChangeType::ContentChanged => {
                // Higher confidence for larger changes
                0.7 + (delta.change_percentage * 0.3)
            }
            ChangeType::MetadataChanged => 0.8,
            ChangeType::Moved => 0.9,
            ChangeType::Multiple(_) => 0.9,
        };

        Ok(base_confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_change_detector_creation() {
        let config = ChangeDetectionConfig::default();
        let detector = ChangeDetector::new(config).await.unwrap();
        assert!(detector.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_new_document_detection() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default()).await.unwrap();
        let doc = Document::new("Test content");
        
        let result = detector.detect_changes(&doc).await.unwrap();
        assert_eq!(result.change_type, ChangeType::Added);
        assert_eq!(result.document_id, doc.id);
        assert!(result.delta.added_chars > 0);
    }

    #[tokio::test]
    async fn test_no_change_detection() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default()).await.unwrap();
        let doc = Document::new("Test content");
        
        // First detection should show as added
        let result1 = detector.detect_changes(&doc).await.unwrap();
        assert_eq!(result1.change_type, ChangeType::Added);
        
        // Second detection should show no change
        let result2 = detector.detect_changes(&doc).await.unwrap();
        assert_eq!(result2.change_type, ChangeType::NoChange);
    }

    #[tokio::test]
    async fn test_content_change_detection() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default()).await.unwrap();
        let doc1 = Document::with_id("test", "Original content");
        let doc2 = Document::with_id("test", "Modified content");
        
        // First detection
        detector.detect_changes(&doc1).await.unwrap();
        
        // Second detection with modified content
        let result = detector.detect_changes(&doc2).await.unwrap();
        assert_eq!(result.change_type, ChangeType::ContentChanged);
        assert!(result.delta.change_percentage > 0.0);
    }

    #[tokio::test]
    async fn test_metadata_change_detection() {
        let detector = ChangeDetector::new(ChangeDetectionConfig::default()).await.unwrap();
        let doc1 = Document::with_id("test", "Same content")
            .with_metadata("key1", serde_json::Value::String("value1".to_string()));
        let doc2 = Document::with_id("test", "Same content")
            .with_metadata("key1", serde_json::Value::String("value2".to_string()));
        
        // First detection
        detector.detect_changes(&doc1).await.unwrap();
        
        // Second detection with modified metadata
        let result = detector.detect_changes(&doc2).await.unwrap();
        assert_eq!(result.change_type, ChangeType::MetadataChanged);
        assert!(!result.metadata_changes.modified_keys.is_empty());
    }

    #[test]
    fn test_hash_algorithms() {
        // Test that different algorithms produce different formats
        let config_default = ChangeDetectionConfig {
            hash_algorithm: HashAlgorithm::Default,
            ..Default::default()
        };
        let config_sha256 = ChangeDetectionConfig {
            hash_algorithm: HashAlgorithm::Sha256,
            ..Default::default()
        };
        
        assert_ne!(
            format!("{:?}", config_default.hash_algorithm),
            format!("{:?}", config_sha256.hash_algorithm)
        );
    }

    #[test]
    fn test_change_sensitivity() {
        let sensitivities = [
            ChangeSensitivity::Low,
            ChangeSensitivity::Medium,
            ChangeSensitivity::High,
            ChangeSensitivity::Strict,
        ];
        
        // All sensitivity levels should be different
        for (i, sens1) in sensitivities.iter().enumerate() {
            for (j, sens2) in sensitivities.iter().enumerate() {
                if i != j {
                    assert_ne!(format!("{:?}", sens1), format!("{:?}", sens2));
                }
            }
        }
    }
}