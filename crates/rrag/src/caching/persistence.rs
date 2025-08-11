//! # Cache Persistence
//! 
//! Persistence layer for cache data across restarts.

use super::{
    PersistenceConfig, PersistenceFormat, QueryCacheEntry, EmbeddingCacheEntry,
    SemanticCacheEntry, ResultCacheEntry, CacheStats
};
use crate::{RragResult, RragError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;
use std::io::{Read, Write};
use std::collections::HashMap;

/// Cache persistence manager
pub struct PersistenceManager {
    /// Configuration
    config: PersistenceConfig,
    
    /// Storage path
    storage_path: PathBuf,
    
    /// Serializer based on format
    serializer: Box<dyn CacheSerializer>,
    
    /// Persistence statistics
    stats: PersistenceStats,
}

/// Persistence statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceStats {
    /// Total saves
    pub save_count: u64,
    
    /// Total loads
    pub load_count: u64,
    
    /// Failed saves
    pub save_failures: u64,
    
    /// Failed loads
    pub load_failures: u64,
    
    /// Total bytes written
    pub bytes_written: u64,
    
    /// Total bytes read
    pub bytes_read: u64,
    
    /// Last save time
    pub last_save: Option<std::time::SystemTime>,
    
    /// Last load time
    pub last_load: Option<std::time::SystemTime>,
}

/// Cache serializer trait
pub trait CacheSerializer: Send + Sync {
    /// Serialize cache data
    fn serialize_cache_data(&self, data: &PersistedCacheData) -> RragResult<Vec<u8>>;
    
    /// Deserialize cache data
    fn deserialize_cache_data(&self, data: &[u8]) -> RragResult<PersistedCacheData>;
    
    /// Get format name
    fn format_name(&self) -> &str;
}

/// Binary serializer using bincode
pub struct BinarySerializer;

impl CacheSerializer for BinarySerializer {
    fn serialize_cache_data(&self, data: &PersistedCacheData) -> RragResult<Vec<u8>> {
        bincode::serialize(data)
            .map_err(|e| RragError::serialization_with_message("binary", e.to_string()))
    }
    
    fn deserialize_cache_data(&self, data: &[u8]) -> RragResult<PersistedCacheData> {
        bincode::deserialize(data)
            .map_err(|e| RragError::serialization_with_message("binary", e.to_string()))
    }
    
    fn format_name(&self) -> &str {
        "binary"
    }
}

/// JSON serializer
pub struct JsonSerializer;

impl CacheSerializer for JsonSerializer {
    fn serialize_cache_data(&self, data: &PersistedCacheData) -> RragResult<Vec<u8>> {
        serde_json::to_vec(data)
            .map_err(|e| RragError::serialization_with_message("json", e.to_string()))
    }
    
    fn deserialize_cache_data(&self, data: &[u8]) -> RragResult<PersistedCacheData> {
        serde_json::from_slice(data)
            .map_err(|e| RragError::serialization_with_message("json", e.to_string()))
    }
    
    fn format_name(&self) -> &str {
        "json"
    }
}

/// MessagePack serializer
pub struct MessagePackSerializer;

impl CacheSerializer for MessagePackSerializer {
    fn serialize_cache_data(&self, data: &PersistedCacheData) -> RragResult<Vec<u8>> {
        rmp_serde::to_vec(data)
            .map_err(|e| RragError::serialization_with_message("msgpack", e.to_string()))
    }
    
    fn deserialize_cache_data(&self, data: &[u8]) -> RragResult<PersistedCacheData> {
        rmp_serde::from_slice(data)
            .map_err(|e| RragError::serialization_with_message("msgpack", e.to_string()))
    }
    
    fn format_name(&self) -> &str {
        "msgpack"
    }
}

/// Persisted cache data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedCacheData {
    /// Version for compatibility
    pub version: u32,
    
    /// Timestamp of persistence
    pub timestamp: std::time::SystemTime,
    
    /// Query cache entries
    pub query_cache: HashMap<String, QueryCacheEntry>,
    
    /// Embedding cache entries
    pub embedding_cache: HashMap<String, EmbeddingCacheEntry>,
    
    /// Semantic cache entries
    pub semantic_cache: HashMap<String, SemanticCacheEntry>,
    
    /// Result cache entries
    pub result_cache: HashMap<String, ResultCacheEntry>,
    
    /// Cache statistics
    pub stats: HashMap<String, CacheStats>,
    
    /// Metadata
    pub metadata: PersistenceMetadata,
}

/// Persistence metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceMetadata {
    /// Application version
    pub app_version: String,
    
    /// Cache configuration hash
    pub config_hash: String,
    
    /// Total entries
    pub total_entries: usize,
    
    /// Compression enabled
    pub compression_enabled: bool,
    
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl PersistenceManager {
    /// Create new persistence manager
    pub fn new(config: PersistenceConfig) -> RragResult<Self> {
        let storage_path = PathBuf::from(&config.storage_path);
        
        // Create storage directory if it doesn't exist
        if !storage_path.exists() {
            fs::create_dir_all(&storage_path)
                .map_err(|e| RragError::storage("create_cache_directory", e))?;
        }
        
        let serializer: Box<dyn CacheSerializer> = match config.format {
            PersistenceFormat::Binary => Box::new(BinarySerializer),
            PersistenceFormat::Json => Box::new(JsonSerializer),
            PersistenceFormat::MessagePack => Box::new(MessagePackSerializer),
        };
        
        Ok(Self {
            config,
            storage_path,
            serializer,
            stats: PersistenceStats::default(),
        })
    }
    
    /// Save cache data to disk
    pub fn save(&mut self, data: &PersistedCacheData) -> RragResult<()> {
        let start = std::time::Instant::now();
        
        // Serialize data
        let serialized = self.serializer.serialize_cache_data(data)?;
        
        // Write to temporary file first
        let temp_path = self.get_temp_path();
        let mut file = fs::File::create(&temp_path)
            .map_err(|e| RragError::storage("create_temp_file", e))?;
        
        file.write_all(&serialized)
            .map_err(|e| RragError::storage("write_cache_data", e))?;
        
        file.sync_all()
            .map_err(|e| RragError::storage("sync_cache_file", e))?;
        
        // Rename to final path (atomic on most systems)
        let final_path = self.get_cache_path();
        fs::rename(&temp_path, &final_path)
            .map_err(|e| RragError::storage("rename_cache_file", e))?;
        
        // Update stats
        self.stats.save_count += 1;
        self.stats.bytes_written += serialized.len() as u64;
        self.stats.last_save = Some(std::time::SystemTime::now());
        
        let duration = start.elapsed();
        tracing::info!(
            "Cache saved: {} entries, {} bytes, {:?}",
            data.metadata.total_entries,
            serialized.len(),
            duration
        );
        
        Ok(())
    }
    
    /// Load cache data from disk
    pub fn load(&mut self) -> RragResult<PersistedCacheData> {
        let start = std::time::Instant::now();
        let cache_path = self.get_cache_path();
        
        if !cache_path.exists() {
            return Err(RragError::memory("load_cache", "Cache file not found"));
        }
        
        // Read file
        let mut file = fs::File::open(&cache_path)
            .map_err(|e| RragError::storage("open_cache_file", e))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| RragError::storage("read_cache_file", e))?;
        
        // Deserialize data
        let data = self.serializer.deserialize_cache_data(&buffer)?;
        
        // Validate version
        if data.version != CACHE_VERSION {
            return Err(RragError::validation(
                "cache_version",
                format!("version {}", CACHE_VERSION),
                format!("version {}", data.version)
            ));
        }
        
        // Update stats
        self.stats.load_count += 1;
        self.stats.bytes_read += buffer.len() as u64;
        self.stats.last_load = Some(std::time::SystemTime::now());
        
        let duration = start.elapsed();
        tracing::info!(
            "Cache loaded: {} entries, {} bytes, {:?}",
            data.metadata.total_entries,
            buffer.len(),
            duration
        );
        
        Ok(data)
    }
    
    /// Save cache asynchronously
    pub async fn save_async(&mut self, data: PersistedCacheData) -> RragResult<()> {
        let serializer = self.create_serializer();
        let path = self.get_cache_path();
        let temp_path = self.get_temp_path();
        
        // Spawn blocking task for IO
        tokio::task::spawn_blocking(move || {
            let serialized = serializer.serialize_cache_data(&data)?;
            
            let mut file = fs::File::create(&temp_path)
                .map_err(|e| RragError::storage("create_temp_file", e))?;
            
            file.write_all(&serialized)
                .map_err(|e| RragError::storage("write_cache_data", e))?;
            
            file.sync_all()
                .map_err(|e| RragError::storage("sync_cache_file", e))?;
            
            fs::rename(&temp_path, &path)
                .map_err(|e| RragError::storage("rename_cache_file", e))?;
            
            Ok(())
        })
        .await
        .map_err(|e| RragError::memory("async_save", e.to_string()))?
    }
    
    /// Create backup of current cache
    pub fn backup(&self) -> RragResult<()> {
        let cache_path = self.get_cache_path();
        if !cache_path.exists() {
            return Ok(());
        }
        
        let backup_path = self.get_backup_path();
        fs::copy(&cache_path, &backup_path)
            .map_err(|e| RragError::storage("create_backup", e))?;
        
        tracing::info!("Cache backed up to {:?}", backup_path);
        Ok(())
    }
    
    /// Restore from backup
    pub fn restore(&self) -> RragResult<()> {
        let backup_path = self.get_backup_path();
        if !backup_path.exists() {
            return Err(RragError::memory("restore_backup", "Backup file not found"));
        }
        
        let cache_path = self.get_cache_path();
        fs::copy(&backup_path, &cache_path)
            .map_err(|e| RragError::storage("restore_from_backup", e))?;
        
        tracing::info!("Cache restored from backup");
        Ok(())
    }
    
    /// Clean old cache files
    pub fn cleanup(&self, keep_days: u32) -> RragResult<()> {
        let cutoff = std::time::SystemTime::now()
            - std::time::Duration::from_secs(keep_days as u64 * 86400);
        
        let entries = fs::read_dir(&self.storage_path)
            .map_err(|e| RragError::storage("read_cache_directory", e))?;
        
        let mut removed = 0;
        for entry in entries {
            let entry = entry.map_err(|e| RragError::storage("read_directory_entry", e))?;
            let metadata = entry.metadata()
                .map_err(|e| RragError::storage("read_file_metadata", e))?;
            
            if let Ok(modified) = metadata.modified() {
                if modified < cutoff {
                    fs::remove_file(entry.path())
                        .map_err(|e| RragError::storage("remove_old_cache", e))?;
                    removed += 1;
                }
            }
        }
        
        tracing::info!("Cleaned up {} old cache files", removed);
        Ok(())
    }
    
    /// Get cache file path
    fn get_cache_path(&self) -> PathBuf {
        self.storage_path.join("cache.dat")
    }
    
    /// Get temporary file path
    fn get_temp_path(&self) -> PathBuf {
        self.storage_path.join("cache.tmp")
    }
    
    /// Get backup file path
    fn get_backup_path(&self) -> PathBuf {
        self.storage_path.join("cache.bak")
    }
    
    /// Create serializer instance
    fn create_serializer(&self) -> Box<dyn CacheSerializer> {
        match self.config.format {
            PersistenceFormat::Binary => Box::new(BinarySerializer),
            PersistenceFormat::Json => Box::new(JsonSerializer),
            PersistenceFormat::MessagePack => Box::new(MessagePackSerializer),
        }
    }
}

/// Cache version for compatibility checking
const CACHE_VERSION: u32 = 1;

impl Default for PersistenceStats {
    fn default() -> Self {
        Self {
            save_count: 0,
            load_count: 0,
            save_failures: 0,
            load_failures: 0,
            bytes_written: 0,
            bytes_read: 0,
            last_save: None,
            last_load: None,
        }
    }
}

impl Default for PersistedCacheData {
    fn default() -> Self {
        Self {
            version: CACHE_VERSION,
            timestamp: std::time::SystemTime::now(),
            query_cache: HashMap::new(),
            embedding_cache: HashMap::new(),
            semantic_cache: HashMap::new(),
            result_cache: HashMap::new(),
            stats: HashMap::new(),
            metadata: PersistenceMetadata::default(),
        }
    }
}

impl Default for PersistenceMetadata {
    fn default() -> Self {
        Self {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            config_hash: String::new(),
            total_entries: 0,
            compression_enabled: false,
            custom: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    fn create_test_config(dir: &Path) -> PersistenceConfig {
        PersistenceConfig {
            enabled: true,
            storage_path: dir.to_str().unwrap().to_string(),
            auto_save_interval: std::time::Duration::from_secs(60),
            format: PersistenceFormat::Binary,
        }
    }
    
    #[test]
    fn test_binary_serializer() {
        let serializer = BinarySerializer;
        let data = PersistedCacheData::default();
        
        let serialized = serializer.serialize_cache_data(&data).unwrap();
        let deserialized = serializer.deserialize_cache_data(&serialized).unwrap();
        
        assert_eq!(data.version, deserialized.version);
    }
    
    #[test]
    fn test_json_serializer() {
        let serializer = JsonSerializer;
        let data = PersistedCacheData::default();
        
        let serialized = serializer.serialize_cache_data(&data).unwrap();
        let deserialized = serializer.deserialize_cache_data(&serialized).unwrap();
        
        assert_eq!(data.version, deserialized.version);
    }
    
    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path());
        let mut manager = PersistenceManager::new(config).unwrap();
        
        let data = PersistedCacheData::default();
        manager.save(&data).unwrap();
        
        let loaded = manager.load().unwrap();
        assert_eq!(loaded.version, data.version);
    }
    
    #[test]
    fn test_backup_and_restore() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path());
        let mut manager = PersistenceManager::new(config).unwrap();
        
        let data = PersistedCacheData::default();
        manager.save(&data).unwrap();
        
        manager.backup().unwrap();
        
        // Delete original
        fs::remove_file(manager.get_cache_path()).unwrap();
        
        // Restore from backup
        manager.restore().unwrap();
        
        let loaded = manager.load().unwrap();
        assert_eq!(loaded.version, data.version);
    }
}