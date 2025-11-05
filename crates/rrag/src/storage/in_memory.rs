//! # In-Memory Storage Implementation
//!
//! Fast, thread-safe in-memory storage using HashMap with RwLock.

use super::memory::{Memory, MemoryQuery, MemoryStats, MemoryValue};
use crate::{RragError, RragResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for in-memory storage
#[derive(Debug, Clone)]
pub struct InMemoryConfig {
    /// Maximum number of keys allowed
    pub max_keys: Option<usize>,

    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<u64>,

    /// Enable automatic eviction when limits are reached
    pub enable_eviction: bool,
}

impl Default for InMemoryConfig {
    fn default() -> Self {
        Self {
            max_keys: Some(100_000),
            max_memory_bytes: Some(1_000_000_000), // 1GB
            enable_eviction: false,
        }
    }
}

/// Entry stored in memory with metadata
#[derive(Debug, Clone)]
struct MemoryEntry {
    value: MemoryValue,
    created_at: chrono::DateTime<chrono::Utc>,
    accessed_at: chrono::DateTime<chrono::Utc>,
}

/// In-memory storage implementation
pub struct InMemoryStorage {
    /// Internal storage
    data: Arc<RwLock<HashMap<String, MemoryEntry>>>,

    /// Configuration
    config: InMemoryConfig,
}

impl InMemoryStorage {
    /// Create a new in-memory storage with default configuration
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            config: InMemoryConfig::default(),
        }
    }

    /// Create a new in-memory storage with custom configuration
    pub fn with_config(config: InMemoryConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if we're within limits
    async fn check_limits(&self) -> RragResult<()> {
        let data = self.data.read().await;

        if let Some(max_keys) = self.config.max_keys {
            if data.len() >= max_keys {
                return Err(RragError::storage(
                    "memory_limit",
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        format!("Exceeded maximum keys: {}", max_keys),
                    ),
                ));
            }
        }

        Ok(())
    }

    /// Check if a key matches the query pattern
    fn matches_query(&self, key: &str, query: &MemoryQuery) -> bool {
        // Check key pattern
        if let Some(pattern) = &query.key_pattern {
            if !key.starts_with(pattern) {
                return false;
            }
        }

        // Check namespace (keys can be prefixed with namespace::)
        if let Some(namespace) = &query.namespace {
            let expected_prefix = format!("{}::", namespace);
            if !key.starts_with(&expected_prefix) {
                return false;
            }
        }

        true
    }

    /// Estimate memory usage (rough calculation)
    fn estimate_memory_usage(&self, data: &HashMap<String, MemoryEntry>) -> u64 {
        let mut total = 0u64;

        for (key, entry) in data.iter() {
            // Key size
            total += key.len() as u64;

            // Value size (rough estimate)
            total += match &entry.value {
                MemoryValue::String(s) => s.len() as u64,
                MemoryValue::Integer(_) => 8,
                MemoryValue::Float(_) => 8,
                MemoryValue::Boolean(_) => 1,
                MemoryValue::Json(j) => j.to_string().len() as u64,
                MemoryValue::Bytes(b) => b.len() as u64,
                MemoryValue::List(l) => l.len() as u64 * 64, // Rough estimate
                MemoryValue::Map(m) => m.len() as u64 * 128, // Rough estimate
            };

            // Metadata overhead
            total += 64;
        }

        total
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Memory for InMemoryStorage {
    fn backend_name(&self) -> &str {
        "in_memory"
    }

    async fn set(&self, key: &str, value: MemoryValue) -> RragResult<()> {
        self.check_limits().await?;

        let mut data = self.data.write().await;
        let now = chrono::Utc::now();

        data.insert(
            key.to_string(),
            MemoryEntry {
                value,
                created_at: now,
                accessed_at: now,
            },
        );

        Ok(())
    }

    async fn get(&self, key: &str) -> RragResult<Option<MemoryValue>> {
        let mut data = self.data.write().await;

        if let Some(entry) = data.get_mut(key) {
            entry.accessed_at = chrono::Utc::now();
            Ok(Some(entry.value.clone()))
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, key: &str) -> RragResult<bool> {
        let mut data = self.data.write().await;
        Ok(data.remove(key).is_some())
    }

    async fn exists(&self, key: &str) -> RragResult<bool> {
        let data = self.data.read().await;
        Ok(data.contains_key(key))
    }

    async fn keys(&self, query: &MemoryQuery) -> RragResult<Vec<String>> {
        let data = self.data.read().await;

        let mut keys: Vec<String> = data
            .keys()
            .filter(|key| self.matches_query(key, query))
            .cloned()
            .collect();

        // Apply offset
        if let Some(offset) = query.offset {
            if offset < keys.len() {
                keys = keys.into_iter().skip(offset).collect();
            } else {
                keys.clear();
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            keys.truncate(limit);
        }

        Ok(keys)
    }

    async fn mget(&self, keys: &[String]) -> RragResult<Vec<Option<MemoryValue>>> {
        let mut data = self.data.write().await;
        let now = chrono::Utc::now();

        let mut results = Vec::with_capacity(keys.len());

        for key in keys {
            if let Some(entry) = data.get_mut(key) {
                entry.accessed_at = now;
                results.push(Some(entry.value.clone()));
            } else {
                results.push(None);
            }
        }

        Ok(results)
    }

    async fn mset(&self, pairs: &[(String, MemoryValue)]) -> RragResult<()> {
        self.check_limits().await?;

        let mut data = self.data.write().await;
        let now = chrono::Utc::now();

        for (key, value) in pairs {
            data.insert(
                key.clone(),
                MemoryEntry {
                    value: value.clone(),
                    created_at: now,
                    accessed_at: now,
                },
            );
        }

        Ok(())
    }

    async fn mdelete(&self, keys: &[String]) -> RragResult<usize> {
        let mut data = self.data.write().await;
        let mut deleted = 0;

        for key in keys {
            if data.remove(key).is_some() {
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    async fn clear(&self, namespace: Option<&str>) -> RragResult<()> {
        let mut data = self.data.write().await;

        if let Some(ns) = namespace {
            let prefix = format!("{}::", ns);
            data.retain(|key, _| !key.starts_with(&prefix));
        } else {
            data.clear();
        }

        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> RragResult<usize> {
        let data = self.data.read().await;

        if let Some(ns) = namespace {
            let prefix = format!("{}::", ns);
            Ok(data.keys().filter(|key| key.starts_with(&prefix)).count())
        } else {
            Ok(data.len())
        }
    }

    async fn health_check(&self) -> RragResult<bool> {
        // Try to read the data
        let _data = self.data.read().await;
        Ok(true)
    }

    async fn stats(&self) -> RragResult<MemoryStats> {
        let data = self.data.read().await;

        let memory_bytes = self.estimate_memory_usage(&data);

        // Count namespaces (keys with :: separator)
        let namespace_count = data
            .keys()
            .filter_map(|key| key.split_once("::").map(|(ns, _)| ns))
            .collect::<std::collections::HashSet<_>>()
            .len();

        let mut extra = std::collections::HashMap::new();
        extra.insert(
            "max_keys".to_string(),
            serde_json::json!(self.config.max_keys),
        );
        extra.insert(
            "max_memory_bytes".to_string(),
            serde_json::json!(self.config.max_memory_bytes),
        );

        Ok(MemoryStats {
            total_keys: data.len(),
            memory_bytes,
            backend_type: "in_memory".to_string(),
            namespace_count,
            last_updated: chrono::Utc::now(),
            extra,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_basic() {
        let storage = InMemoryStorage::new();

        // Test set and get
        storage
            .set("test_key", MemoryValue::String("test_value".to_string()))
            .await
            .unwrap();

        let value = storage.get("test_key").await.unwrap();
        assert!(value.is_some());
        assert_eq!(value.unwrap().as_string().unwrap(), "test_value");

        // Test exists
        assert!(storage.exists("test_key").await.unwrap());
        assert!(!storage.exists("nonexistent").await.unwrap());

        // Test delete
        assert!(storage.delete("test_key").await.unwrap());
        assert!(!storage.exists("test_key").await.unwrap());
    }

    #[tokio::test]
    async fn test_in_memory_bulk_operations() {
        let storage = InMemoryStorage::new();

        // Test mset
        let pairs = vec![
            ("key1".to_string(), MemoryValue::Integer(1)),
            ("key2".to_string(), MemoryValue::Integer(2)),
            ("key3".to_string(), MemoryValue::Integer(3)),
        ];
        storage.mset(&pairs).await.unwrap();

        // Test mget
        let keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string()];
        let values = storage.mget(&keys).await.unwrap();
        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|v| v.is_some()));

        // Test mdelete
        let deleted = storage.mdelete(&keys).await.unwrap();
        assert_eq!(deleted, 3);
    }

    #[tokio::test]
    async fn test_in_memory_namespace() {
        let storage = InMemoryStorage::new();

        // Add keys with namespace
        storage
            .set("ns1::key1", MemoryValue::String("value1".to_string()))
            .await
            .unwrap();
        storage
            .set("ns1::key2", MemoryValue::String("value2".to_string()))
            .await
            .unwrap();
        storage
            .set("ns2::key1", MemoryValue::String("value3".to_string()))
            .await
            .unwrap();

        // Count by namespace
        assert_eq!(storage.count(Some("ns1")).await.unwrap(), 2);
        assert_eq!(storage.count(Some("ns2")).await.unwrap(), 1);

        // Clear namespace
        storage.clear(Some("ns1")).await.unwrap();
        assert_eq!(storage.count(Some("ns1")).await.unwrap(), 0);
        assert_eq!(storage.count(Some("ns2")).await.unwrap(), 1);
    }
}
