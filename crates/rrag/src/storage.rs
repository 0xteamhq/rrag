//! # RRAG Storage System
//! 
//! Pluggable storage backends with async I/O and efficient serialization.
//! Designed for Rust's ownership model and zero-copy operations where possible.

use crate::{RragError, RragResult, Document, DocumentChunk, Embedding};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Storage entry that can contain documents, chunks, or embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageEntry {
    Document(Document),
    Chunk(DocumentChunk),
    Embedding(Embedding),
    Metadata(HashMap<String, serde_json::Value>),
}

/// Storage key for efficient lookups
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StorageKey {
    /// Entry type
    pub entry_type: EntryType,
    
    /// Unique identifier
    pub id: String,
    
    /// Optional namespace/collection
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntryType {
    Document,
    Chunk,
    Embedding,
    Metadata,
}

impl StorageKey {
    pub fn document(id: impl Into<String>) -> Self {
        Self {
            entry_type: EntryType::Document,
            id: id.into(),
            namespace: None,
        }
    }

    pub fn chunk(document_id: impl Into<String>, chunk_index: usize) -> Self {
        Self {
            entry_type: EntryType::Chunk,
            id: format!("{}_{}", document_id.into(), chunk_index),
            namespace: None,
        }
    }

    pub fn embedding(id: impl Into<String>) -> Self {
        Self {
            entry_type: EntryType::Embedding,
            id: id.into(),
            namespace: None,
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Generate a storage path for file-based systems
    pub fn to_path(&self) -> PathBuf {
        let type_str = match self.entry_type {
            EntryType::Document => "documents",
            EntryType::Chunk => "chunks",
            EntryType::Embedding => "embeddings",
            EntryType::Metadata => "metadata",
        };

        let mut path = PathBuf::from(type_str);
        
        if let Some(namespace) = &self.namespace {
            path.push(namespace);
        }
        
        path.push(format!("{}.json", self.id));
        path
    }
}

/// Storage query for bulk operations
#[derive(Debug, Clone)]
pub struct StorageQuery {
    /// Entry type filter
    pub entry_type: Option<EntryType>,
    
    /// Namespace filter
    pub namespace: Option<String>,
    
    /// Key prefix filter
    pub key_prefix: Option<String>,
    
    /// Metadata filters
    pub metadata_filters: HashMap<String, serde_json::Value>,
    
    /// Maximum results
    pub limit: Option<usize>,
    
    /// Offset for pagination
    pub offset: Option<usize>,
}

impl StorageQuery {
    pub fn new() -> Self {
        Self {
            entry_type: None,
            namespace: None,
            key_prefix: None,
            metadata_filters: HashMap::new(),
            limit: None,
            offset: None,
        }
    }

    pub fn documents() -> Self {
        Self::new().with_entry_type(EntryType::Document)
    }

    pub fn chunks() -> Self {
        Self::new().with_entry_type(EntryType::Chunk)
    }

    pub fn embeddings() -> Self {
        Self::new().with_entry_type(EntryType::Embedding)
    }

    pub fn with_entry_type(mut self, entry_type: EntryType) -> Self {
        self.entry_type = Some(entry_type);
        self
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = Some(prefix.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

impl Default for StorageQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Core storage trait for different backends
#[async_trait]
pub trait Storage: Send + Sync {
    /// Storage backend name
    fn name(&self) -> &str;
    
    /// Store an entry
    async fn put(&self, key: &StorageKey, entry: &StorageEntry) -> RragResult<()>;
    
    /// Retrieve an entry
    async fn get(&self, key: &StorageKey) -> RragResult<Option<StorageEntry>>;
    
    /// Delete an entry
    async fn delete(&self, key: &StorageKey) -> RragResult<bool>;
    
    /// Check if an entry exists
    async fn exists(&self, key: &StorageKey) -> RragResult<bool>;
    
    /// List keys matching a query
    async fn list_keys(&self, query: &StorageQuery) -> RragResult<Vec<StorageKey>>;
    
    /// Bulk get operation
    async fn get_many(&self, keys: &[StorageKey]) -> RragResult<Vec<(StorageKey, Option<StorageEntry>)>>;
    
    /// Bulk put operation
    async fn put_many(&self, entries: &[(StorageKey, StorageEntry)]) -> RragResult<()>;
    
    /// Bulk delete operation
    async fn delete_many(&self, keys: &[StorageKey]) -> RragResult<usize>;
    
    /// Clear all entries (optional)
    async fn clear(&self) -> RragResult<()> {
        Err(RragError::storage("clear", std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Clear operation not supported"
        )))
    }
    
    /// Get storage statistics
    async fn stats(&self) -> RragResult<StorageStats>;
    
    /// Health check
    async fn health_check(&self) -> RragResult<bool>;
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total number of entries
    pub total_entries: usize,
    
    /// Entries by type
    pub entries_by_type: HashMap<String, usize>,
    
    /// Storage size in bytes
    pub size_bytes: u64,
    
    /// Available space in bytes (if applicable)
    pub available_bytes: Option<u64>,
    
    /// Backend type
    pub backend_type: String,
    
    /// Last updated
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// In-memory storage implementation
pub struct InMemoryStorage {
    /// Internal storage map
    data: Arc<tokio::sync::RwLock<HashMap<StorageKey, StorageEntry>>>,
    
    /// Configuration
    config: MemoryStorageConfig,
}

#[derive(Debug, Clone)]
pub struct MemoryStorageConfig {
    /// Maximum number of entries
    pub max_entries: Option<usize>,
    
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<u64>,
}

impl Default for MemoryStorageConfig {
    fn default() -> Self {
        Self {
            max_entries: Some(100_000),
            max_memory_bytes: Some(1_000_000_000), // 1GB
        }
    }
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config: MemoryStorageConfig::default(),
        }
    }

    pub fn with_config(config: MemoryStorageConfig) -> Self {
        Self {
            data: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if we're within memory limits
    async fn check_limits(&self) -> RragResult<()> {
        let data = self.data.read().await;
        
        if let Some(max_entries) = self.config.max_entries {
            if data.len() >= max_entries {
                return Err(RragError::storage(
                    "memory_limit",
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        format!("Exceeded maximum entries: {}", max_entries)
                    )
                ));
            }
        }
        
        Ok(())
    }

    /// Filter entries based on query
    fn matches_query(&self, key: &StorageKey, query: &StorageQuery) -> bool {
        // Check entry type
        if let Some(entry_type) = &query.entry_type {
            if key.entry_type != *entry_type {
                return false;
            }
        }
        
        // Check namespace
        if let Some(namespace) = &query.namespace {
            match &key.namespace {
                Some(key_ns) if key_ns == namespace => {},
                None if namespace.is_empty() => {},
                _ => return false,
            }
        }
        
        // Check prefix
        if let Some(prefix) = &query.key_prefix {
            if !key.id.starts_with(prefix) {
                return false;
            }
        }
        
        true
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    fn name(&self) -> &str {
        "in_memory"
    }

    async fn put(&self, key: &StorageKey, entry: &StorageEntry) -> RragResult<()> {
        self.check_limits().await?;
        
        let mut data = self.data.write().await;
        data.insert(key.clone(), entry.clone());
        Ok(())
    }

    async fn get(&self, key: &StorageKey) -> RragResult<Option<StorageEntry>> {
        let data = self.data.read().await;
        Ok(data.get(key).cloned())
    }

    async fn delete(&self, key: &StorageKey) -> RragResult<bool> {
        let mut data = self.data.write().await;
        Ok(data.remove(key).is_some())
    }

    async fn exists(&self, key: &StorageKey) -> RragResult<bool> {
        let data = self.data.read().await;
        Ok(data.contains_key(key))
    }

    async fn list_keys(&self, query: &StorageQuery) -> RragResult<Vec<StorageKey>> {
        let data = self.data.read().await;
        let mut keys: Vec<StorageKey> = data
            .keys()
            .filter(|key| self.matches_query(key, query))
            .cloned()
            .collect();
        
        // Apply offset and limit
        if let Some(offset) = query.offset {
            if offset < keys.len() {
                keys = keys.into_iter().skip(offset).collect();
            } else {
                keys.clear();
            }
        }
        
        if let Some(limit) = query.limit {
            keys.truncate(limit);
        }
        
        Ok(keys)
    }

    async fn get_many(&self, keys: &[StorageKey]) -> RragResult<Vec<(StorageKey, Option<StorageEntry>)>> {
        let data = self.data.read().await;
        let results = keys
            .iter()
            .map(|key| (key.clone(), data.get(key).cloned()))
            .collect();
        Ok(results)
    }

    async fn put_many(&self, entries: &[(StorageKey, StorageEntry)]) -> RragResult<()> {
        self.check_limits().await?;
        
        let mut data = self.data.write().await;
        for (key, entry) in entries {
            data.insert(key.clone(), entry.clone());
        }
        Ok(())
    }

    async fn delete_many(&self, keys: &[StorageKey]) -> RragResult<usize> {
        let mut data = self.data.write().await;
        let mut deleted = 0;
        for key in keys {
            if data.remove(key).is_some() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    async fn clear(&self) -> RragResult<()> {
        let mut data = self.data.write().await;
        data.clear();
        Ok(())
    }

    async fn stats(&self) -> RragResult<StorageStats> {
        let data = self.data.read().await;
        
        let mut entries_by_type = HashMap::new();
        for key in data.keys() {
            let type_str = match key.entry_type {
                EntryType::Document => "documents",
                EntryType::Chunk => "chunks", 
                EntryType::Embedding => "embeddings",
                EntryType::Metadata => "metadata",
            };
            *entries_by_type.entry(type_str.to_string()).or_insert(0) += 1;
        }
        
        // Estimate memory usage (rough calculation)
        let estimated_size = data.len() * 1024; // Rough estimate per entry
        
        Ok(StorageStats {
            total_entries: data.len(),
            entries_by_type,
            size_bytes: estimated_size as u64,
            available_bytes: self.config.max_memory_bytes.map(|max| max - estimated_size as u64),
            backend_type: "in_memory".to_string(),
            last_updated: chrono::Utc::now(),
        })
    }

    async fn health_check(&self) -> RragResult<bool> {
        // Simple health check - try to read the data
        let _data = self.data.read().await;
        Ok(true)
    }
}

/// File-based storage implementation
pub struct FileStorage {
    /// Base directory for storage
    base_dir: PathBuf,
    
    /// Configuration
    config: FileStorageConfig,
}

#[derive(Debug, Clone)]
pub struct FileStorageConfig {
    /// Whether to create directories automatically
    pub create_dirs: bool,
    
    /// File permissions (Unix only)
    pub file_permissions: Option<u32>,
    
    /// Whether to use compression
    pub compress: bool,
    
    /// Sync writes to disk immediately
    pub sync_writes: bool,
}

impl Default for FileStorageConfig {
    fn default() -> Self {
        Self {
            create_dirs: true,
            file_permissions: None,
            compress: false,
            sync_writes: false,
        }
    }
}

impl FileStorage {
    pub async fn new(base_dir: impl AsRef<Path>) -> RragResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir)
                .await
                .map_err(|e| RragError::storage("create_directory", e))?;
        }
        
        Ok(Self {
            base_dir,
            config: FileStorageConfig::default(),
        })
    }

    pub async fn with_config(base_dir: impl AsRef<Path>, config: FileStorageConfig) -> RragResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        
        if config.create_dirs && !base_dir.exists() {
            fs::create_dir_all(&base_dir)
                .await
                .map_err(|e| RragError::storage("create_directory", e))?;
        }
        
        Ok(Self { base_dir, config })
    }

    /// Get the full file path for a storage key
    fn get_file_path(&self, key: &StorageKey) -> PathBuf {
        self.base_dir.join(key.to_path())
    }

    /// Ensure parent directory exists
    async fn ensure_parent_dir(&self, file_path: &Path) -> RragResult<()> {
        if let Some(parent) = file_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .await
                    .map_err(|e| RragError::storage("create_parent_directory", e))?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Storage for FileStorage {
    fn name(&self) -> &str {
        "file_system"
    }

    async fn put(&self, key: &StorageKey, entry: &StorageEntry) -> RragResult<()> {
        let file_path = self.get_file_path(key);
        self.ensure_parent_dir(&file_path).await?;
        
        let json_data = serde_json::to_vec_pretty(entry)
            .map_err(|e| RragError::storage("serialize", e))?;
        
        let mut file = fs::File::create(&file_path)
            .await
            .map_err(|e| RragError::storage("create_file", e))?;
        
        file.write_all(&json_data)
            .await
            .map_err(|e| RragError::storage("write_file", e))?;
        
        if self.config.sync_writes {
            file.sync_all()
                .await
                .map_err(|e| RragError::storage("sync_file", e))?;
        }
        
        Ok(())
    }

    async fn get(&self, key: &StorageKey) -> RragResult<Option<StorageEntry>> {
        let file_path = self.get_file_path(key);
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        let mut file = fs::File::open(&file_path)
            .await
            .map_err(|e| RragError::storage("open_file", e))?;
        
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .await
            .map_err(|e| RragError::storage("read_file", e))?;
        
        let entry = serde_json::from_slice(&contents)
            .map_err(|e| RragError::storage("deserialize", e))?;
        
        Ok(Some(entry))
    }

    async fn delete(&self, key: &StorageKey) -> RragResult<bool> {
        let file_path = self.get_file_path(key);
        
        if !file_path.exists() {
            return Ok(false);
        }
        
        fs::remove_file(&file_path)
            .await
            .map_err(|e| RragError::storage("delete_file", e))?;
        
        Ok(true)
    }

    async fn exists(&self, key: &StorageKey) -> RragResult<bool> {
        let file_path = self.get_file_path(key);
        Ok(file_path.exists())
    }

    async fn list_keys(&self, _query: &StorageQuery) -> RragResult<Vec<StorageKey>> {
        // This is a simplified implementation
        // In production, you'd want more efficient directory traversal
        let keys = Vec::new();
        
        // For now, return empty - would need recursive directory walking
        // This is a limitation of the simple file storage implementation
        Ok(keys)
    }

    async fn get_many(&self, keys: &[StorageKey]) -> RragResult<Vec<(StorageKey, Option<StorageEntry>)>> {
        let mut results = Vec::with_capacity(keys.len());
        
        for key in keys {
            let entry = self.get(key).await?;
            results.push((key.clone(), entry));
        }
        
        Ok(results)
    }

    async fn put_many(&self, entries: &[(StorageKey, StorageEntry)]) -> RragResult<()> {
        for (key, entry) in entries {
            self.put(key, entry).await?;
        }
        Ok(())
    }

    async fn delete_many(&self, keys: &[StorageKey]) -> RragResult<usize> {
        let mut deleted = 0;
        
        for key in keys {
            if self.delete(key).await? {
                deleted += 1;
            }
        }
        
        Ok(deleted)
    }

    async fn stats(&self) -> RragResult<StorageStats> {
        // Calculate directory size and file counts
        // This is a simplified implementation
        Ok(StorageStats {
            total_entries: 0, // Would need directory traversal
            entries_by_type: HashMap::new(),
            size_bytes: 0,
            available_bytes: None,
            backend_type: "file_system".to_string(),
            last_updated: chrono::Utc::now(),
        })
    }

    async fn health_check(&self) -> RragResult<bool> {
        // Check if base directory is accessible
        Ok(self.base_dir.exists() && self.base_dir.is_dir())
    }
}

/// High-level storage service with caching and batching
pub struct StorageService {
    /// Active storage backend
    storage: Arc<dyn Storage>,
    
    /// Service configuration
    config: StorageServiceConfig,
}

#[derive(Debug, Clone)]
pub struct StorageServiceConfig {
    /// Enable write batching
    pub enable_batching: bool,
    
    /// Batch size for bulk operations
    pub batch_size: usize,
    
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    
    /// Enable read caching
    pub enable_caching: bool,
    
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for StorageServiceConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            batch_size: 100,
            batch_timeout_ms: 1000,
            enable_caching: false,
            cache_ttl_seconds: 300,
        }
    }
}

impl StorageService {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            storage,
            config: StorageServiceConfig::default(),
        }
    }

    pub fn with_config(storage: Arc<dyn Storage>, config: StorageServiceConfig) -> Self {
        Self { storage, config }
    }

    /// Store a document
    pub async fn store_document(&self, document: &Document) -> RragResult<()> {
        let key = StorageKey::document(&document.id);
        let entry = StorageEntry::Document(document.clone());
        self.storage.put(&key, &entry).await
    }

    /// Store a chunk
    pub async fn store_chunk(&self, chunk: &DocumentChunk) -> RragResult<()> {
        let key = StorageKey::chunk(&chunk.document_id, chunk.chunk_index);
        let entry = StorageEntry::Chunk(chunk.clone());
        self.storage.put(&key, &entry).await
    }

    /// Store an embedding
    pub async fn store_embedding(&self, embedding: &Embedding) -> RragResult<()> {
        let key = StorageKey::embedding(&embedding.source_id);
        let entry = StorageEntry::Embedding(embedding.clone());
        self.storage.put(&key, &entry).await
    }

    /// Retrieve a document
    pub async fn get_document(&self, document_id: &str) -> RragResult<Option<Document>> {
        let key = StorageKey::document(document_id);
        match self.storage.get(&key).await? {
            Some(StorageEntry::Document(doc)) => Ok(Some(doc)),
            _ => Ok(None),
        }
    }

    /// Get storage statistics
    pub async fn get_stats(&self) -> RragResult<StorageStats> {
        self.storage.stats().await
    }

    /// Health check
    pub async fn health_check(&self) -> RragResult<bool> {
        self.storage.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_in_memory_storage() {
        let storage = InMemoryStorage::new();
        
        let doc = Document::new("Test document");
        let key = StorageKey::document(&doc.id);
        let entry = StorageEntry::Document(doc.clone());
        
        // Test put and get
        storage.put(&key, &entry).await.unwrap();
        
        let retrieved = storage.get(&key).await.unwrap();
        assert!(retrieved.is_some());
        
        if let Some(StorageEntry::Document(retrieved_doc)) = retrieved {
            assert_eq!(retrieved_doc.id, doc.id);
            assert_eq!(retrieved_doc.content_str(), doc.content_str());
        }
        
        // Test exists
        assert!(storage.exists(&key).await.unwrap());
        
        // Test delete
        assert!(storage.delete(&key).await.unwrap());
        assert!(!storage.exists(&key).await.unwrap());
    }

    #[tokio::test]
    async fn test_file_storage() {
        let temp_dir = TempDir::new().unwrap();
        let storage = FileStorage::new(temp_dir.path()).await.unwrap();
        
        let doc = Document::new("Test document for file storage");
        let key = StorageKey::document(&doc.id);
        let entry = StorageEntry::Document(doc.clone());
        
        // Test put and get
        storage.put(&key, &entry).await.unwrap();
        
        let retrieved = storage.get(&key).await.unwrap();
        assert!(retrieved.is_some());
        
        if let Some(StorageEntry::Document(retrieved_doc)) = retrieved {
            assert_eq!(retrieved_doc.id, doc.id);
        }
        
        // Test file exists on disk
        let file_path = temp_dir.path().join(key.to_path());
        assert!(file_path.exists());
    }

    #[test]
    fn test_storage_key() {
        let doc_key = StorageKey::document("doc1");
        assert_eq!(doc_key.entry_type, EntryType::Document);
        assert_eq!(doc_key.id, "doc1");
        
        let chunk_key = StorageKey::chunk("doc1", 5);
        assert_eq!(chunk_key.entry_type, EntryType::Chunk);
        assert_eq!(chunk_key.id, "doc1_5");
        
        let ns_key = doc_key.with_namespace("test_namespace");
        assert_eq!(ns_key.namespace, Some("test_namespace".to_string()));
    }

    #[tokio::test]
    async fn test_storage_service() {
        let storage = Arc::new(InMemoryStorage::new());
        let service = StorageService::new(storage);
        
        let doc = Document::new("Test document for service");
        
        // Store document
        service.store_document(&doc).await.unwrap();
        
        // Retrieve document
        let retrieved = service.get_document(&doc.id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, doc.id);
        
        // Check stats
        let stats = service.get_stats().await.unwrap();
        assert_eq!(stats.total_entries, 1);
    }
}